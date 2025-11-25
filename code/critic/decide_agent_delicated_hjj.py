from typing import Dict, List, Union
from datetime import datetime
import uuid
import os
import io
import base64
import json
import aiohttp
import numpy as np
import pandas as pd
import requests
import copy
from autogen_core import MessageContext, RoutedAgent, TopicId, default_subscription, message_handler
from autogen_core.models import (ChatCompletionClient, LLMMessage, SystemMessage, UserMessage, AssistantMessage)

from save_sketches import save_sketches
from visualize.curve_cubic_spline import Visualizer
from animator import Animator, Animator_dedicated
from message_class_hjj import DecisionTask, ReviewTask, ReviewResult, DecisionResult, ActionChoice, ResamplingTask, ResamplingResult, FinalReviewResult
from evaluate.gag.json_find import json_find
# from instruct_following_hjj import generate_counter_factual_instruct, check_consistency, check_consistency_batch
from instruct_following import generate_counter_factual_instruct, check_consistency, check_consistency_batch
# from instruct_following_hjj import generate_counter_factual_instruct, generate_counter_factual_multi_step
import matplotlib.pyplot as plt
from evaluate.xt import XT
from evaluate.xg import XG
from evaluate.pitch_control import PitchControl
from evaluate.consistency import consistency, ConsistencyResult

# 全局变量存储最终结果
final_decision_result = None

class ModelClient:
    def __init__(self, base_url: str='http://210.75.240.143:8099'):
        self.base_url = base_url
        self.headers = {
            'Content-Type': 'application/json',
        }

    async def create(self, X=None, TE_x=None, TE_y=None, lang=None):
        data = {"X": X, "TE_x": TE_x, "TE_y": TE_y, "lang": lang}
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/predict",
                headers=self.headers,
                json=data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result
                else:
                    print(f"API请求失败，状态码: {response.status}, 响应: {await response.text()}")
                    return None

    def create_sync(self, X=None, TE=None, text=None):
        data = {"X": X, "TE": TE, "LE": text}

        response = requests.post(f"{self.base_url}/predict", headers=self.headers, json=data)
        # 检查响应状态码
        if response.status_code == 200:
            print("API请求成功")  # 解析 JSON 响应
        else:
            print(f"API请求失败，状态码: {response.status_code}, 响应: {response.text}")
        return response

@default_subscription
class DecisionAgentDedicated(RoutedAgent):
    def __init__(self, model_client, base_path='sketches', vlz_mode='static'):
        super().__init__("A Decision Agent")
        self._model_client = model_client
        self._session_memory: Dict[str, List[DecisionTask | ReviewTask | ReviewResult | ActionChoice]] = {}
        self._system_messages: str = """You are a football analyst. Here are descriptions for current match:\n\n"""
        self._visualizer = Visualizer()
        self._animator = Animator_dedicated(self._visualizer)
        self._animator_pdf = Animator_dedicated(self._visualizer, save2PIL=False)
        self.cf_sketches_pdf_path = {}
        self.cf_sketches_mp4_path = {}
        self.xg = XG()
        self.xt = XT()
        self.pc = PitchControl()
        self.base_path = base_path

        if vlz_mode not in ['static', 'dynamic']:
            raise ValueError("vlz_mode must be either 'static' or 'dynamic'")
        self.vlz_mode=vlz_mode


    async def make_decision(self, decision_task):
        response = await self._model_client.create(
            X=decision_task.history,
            TE_x=decision_task.time_hist,
            TE_y=decision_task.time_pred,
            lang=decision_task.condition,
        )
        return response

    @message_handler # 对一个指令反事实成各个集合供模型进行指令选择
    async def handle_decision_task(self, message: DecisionTask, ctx: MessageContext) -> None:
        print("\n=== Starting Decision Task ===")
        print(f"Received task: {message.condition}...")  # Print first 100 chars of task
        print(f"Stage: {message.step}, Scenario: {message.description['scenario_requirements']}")
        # 生成进程id
        session_id = message.description['session_id']
        # TODO 改成main里面传入self.path
        self.path = f'IS_{datetime.now().strftime("%Y-%m-%d")}_index{message.index}_{session_id}_step{message.step}'
        global save_path
        save_path = self.path
        os.makedirs(os.path.join(self.base_path, self.path), exist_ok=True)
        self.save_txt = open(os.path.join(self.base_path, self.path, 'reasoning.txt'), 'a')
        # os.makedirs(os.path.join('sketches', self.path, 'questionnaire_video'), exist_ok=True)

        print(f"Generated session ID: {session_id}")
        self._session_memory.setdefault(session_id, []).append(message)

        response = await self.make_decision(message)
        prediction = response['prediction']
        cf_instruct_list = generate_counter_factual_instruct(
            current_instruct=message.condition,
            player_name_list=message.description['name_list'],
            player_role_list=message.description['role_list']
        )
        cf_prediction_list = []
        prediction_candidate = {message.condition: prediction}

        metrics_candidate = {}
        for cf_instruct in cf_instruct_list:
            new_message = copy.deepcopy(message)
            new_message.condition = cf_instruct
            response = await self.make_decision(new_message)
            cf_prediction_list.append(response['prediction'])
            prediction_candidate[cf_instruct] = response['prediction']
        for key, value in prediction_candidate.items():
            metrics_candidate[key] = {
                'xg': self.xg.get_metrics(value[0]),
                'xt': self.xt.get_metrics(value[0]),
                'pc': self.pc.get_metrics(value[0]),
                'consistency': consistency(instruct=key, model_output=value[0], player_name_list=message.description['name_list'], player_role_list=message.description['role_list']).pass_both,
                'consistency_end': consistency(instruct=key, model_output=value[0], player_name_list=message.description['name_list'], player_role_list=message.description['role_list']).pass_end,
                'consistency_end_error': consistency(instruct=key, model_output=value[0], player_name_list=message.description['name_list'], player_role_list=message.description['role_list']).end_error,
            }

        with open(os.path.join(self.base_path, self.path, 'cf_metrics.txt'), 'w') as p:
            json.dump(metrics_candidate, p, indent=4)
        global current_cf_metrics
        current_cf_metrics = metrics_candidate

        plt.ioff()
        # print(f"prompt: {message.task}")
        carrier_role, recipient_role = json.loads(message.condition)['carrier_role'], json.loads(message.condition)['recipient_role']
        sketch = {'history': self._convert_pil_to_base64(self._animator.animate_predict_forQ(
                    prediction=message.history[0],
                    name_list=message.description['name_list'],
                    role_list=message.description['role_list'],
                )),
                'prediction': self._convert_pil_to_base64(self._animator.animate_predict_forQ(
                    x = message.history[0],
                    prediction = prediction[0],
                    carrier_role = carrier_role,
                    recipient_role = recipient_role,
                    name_list = message.description['name_list'],
                    role_list = message.description['role_list'],
                ))}

        label_sketch = {'label': self._convert_pil_to_base64(self._animator.animate_predict_forQ(
                x = message.history[0],
                prediction = message.label[0],
                carrier_role = carrier_role,
                recipient_role = recipient_role,
                name_list = message.description['name_list'],
                role_list = message.description['role_list'],
                ))}
        os.makedirs(os.path.join(self.base_path, self.path, 'questionnaire_pdf'), exist_ok=True)
        os.makedirs(os.path.join(self.base_path, self.path, 'cf_sketches_pdf'), exist_ok=True)

        self._animator_pdf.animate_predict_forQ(prediction=message.history[0], name_list=message.description['name_list'], role_list=message.description['role_list'], path=os.path.join(self.base_path, self.path, 'questionnaire_pdf', 'history.pdf'))
        self._animator_pdf.animate_predict_forQ(x = message.history[0], prediction = prediction[0], carrier_role = carrier_role, recipient_role = recipient_role, name_list = message.description['name_list'], role_list = message.description['role_list'], path=os.path.join(self.base_path, self.path, 'questionnaire_pdf', 'prediction.pdf'))
        self._animator_pdf.animate_predict_forQ(x = message.history[0], prediction = message.label[0], carrier_role = carrier_role, recipient_role = recipient_role, name_list = message.description['name_list'], role_list = message.description['role_list'], path=os.path.join(self.base_path, self.path, 'questionnaire_pdf', 'label.pdf'))
        self._animator_pdf.video_predict_forQ(prediction=message.history[0], name_list=message.description['name_list'], role_list=message.description['role_list'], path=os.path.join(self.base_path, self.path, 'questionnaire_pdf', 'history.mp4'))
        self._animator_pdf.video_predict_forQ(x = message.history[0], prediction = prediction[0], carrier_role = carrier_role, recipient_role = recipient_role, name_list = message.description['name_list'], role_list = message.description['role_list'], path=os.path.join(self.base_path, self.path, 'questionnaire_pdf', 'prediction.mp4'))
        self._animator_pdf.video_predict_forQ(x = message.history[0], prediction = message.label[0], carrier_role = carrier_role, recipient_role = recipient_role, name_list = message.description['name_list'], role_list = message.description['role_list'], path=os.path.join(self.base_path, self.path, 'questionnaire_pdf', 'label.mp4'))
        plt.close('all')

        # Save sketches with session_id
        save_sketches(sketch, path=os.path.join(self.base_path, self.path, 'questionnaire'), dpi=600)
        save_sketches(label_sketch, path=os.path.join(self.base_path, self.path, 'questionnaire'), dpi=600)
        # cf_sketches = {message.condition: sketch['prediction']}
        # for cf_instruct, cf_prediction in zip(cf_instruct_list, cf_prediction_list):
        #     sketch[cf_instruct] = self._convert_pil_to_base64(self._animator.animate_predict_forQ(
        #         x = message.history[0],
        #         prediction = cf_prediction[0],
        #         name_list = message.description['name_list'],
        #         role_list = message.description['role_list'],
        #     ))
        # for cf_instruct, cf_prediction in prediction_candidate.items(

        cf_sketches = {}
        for cf_instruct, cf_prediction in prediction_candidate.items():
            carrier_role, recipient_role = json.loads(cf_instruct)['carrier_role'], json.loads(cf_instruct)['recipient_role']
            sketch[cf_instruct] = self._convert_pil_to_base64(self._animator.animate_predict_forQ(
                x = message.history[0],
                prediction = cf_prediction[0],
                carrier_role = carrier_role,
                recipient_role = recipient_role,
                name_list = message.description['name_list'],
                role_list = message.description['role_list'],
            ))
            cf_sketches[cf_instruct] = sketch[cf_instruct]
            self._animator_pdf.animate_predict_forQ(x = message.history[0], prediction = cf_prediction[0], carrier_role = carrier_role, recipient_role = recipient_role, name_list = message.description['name_list'], role_list = message.description['role_list'], path=os.path.join(self.base_path, self.path, 'cf_sketches_pdf', f'{cf_instruct}.pdf'))
            self._animator_pdf.video_predict_forQ(x = message.history[0], prediction = cf_prediction[0], carrier_role = carrier_role, recipient_role = recipient_role, name_list = message.description['name_list'], role_list = message.description['role_list'], path=os.path.join(self.base_path, self.path, 'cf_sketches_pdf', f'{cf_instruct}.mp4'))
            self.cf_sketches_pdf_path[cf_instruct] = os.path.join(self.base_path, self.path, 'cf_sketches_pdf', f'{cf_instruct}.pdf')
            self.cf_sketches_mp4_path[cf_instruct] = os.path.join(self.base_path, self.path, 'cf_sketches_pdf', f'{cf_instruct}.mp4')

        save_sketches(cf_sketches, path=os.path.join(self.base_path, self.path, 'cf_sketches'), dpi=600)
        # save_sketches(
        #     {'whatever': sketch['prediction']},
        #     path=os.path.join('sketches', self.path, 'cf_sketches'),
        #     specific_name=message.condition, dpi=600
        # )
        plt.close('all')
        review_task = ReviewTask(
            session_id=session_id,
            path=self.path,
            decision_task_dict=message.__dict__,
            decision_sketch=sketch,
            decision_candidate=prediction_candidate,
            decision=prediction,
        )
        self._session_memory[session_id].append(review_task)

        # print("Decision Result:", file=self.save_txt, end='\n')
        # print("-" * 80, file=self.save_txt, end='\n')
        with open(os.path.join(self.base_path, self.path, 'reasoning.txt'), 'a') as save_txt:
            print(f"Initial condition:\n{message.condition}", file=save_txt, end='\n')
            print("-" * 80, file=save_txt, end='\n\n')
        # print(f"Prediction:\n{prediction}", file=self.save_txt, end='\n')
        # print("-" * 80, file=self.save_txt, end='\n\n')
        await self.publish_message(
            review_task,
            topic_id=TopicId("default", self.id.key)
        )

    # 处理多次采样任务的handler
    @message_handler
    async def handle_resampling_task(self, message: ActionChoice, ctx: MessageContext) -> None:
        self._session_memory[message.session_id].append(message)
        print("==== Handle decision review ====")
        print(f"Received ReviewResult with approved={message.approved}")
        print(f"Event advised: {message.event_advised}")

        # Obtain the request from previous messages.
        action_choice_request = next(m for m in reversed(self._session_memory[message.session_id]) if isinstance(m, ReviewTask))  # TODO 感觉这里可以优化，很乱
        assert action_choice_request is not None

        # TODO resample 改成list
        resampled_predictions = {} # flm输出
        resampled_sketches = {} # 图形保存
        for i in range(message.num_samples):
            print(f"Resampling iteration {i+1}/{message.num_samples}")
            action_choice_request.decision_task_dict['condition'] = message.event_advised

            # -- added TODO 增加优雅性
            decision_task_dict = action_choice_request.decision_task_dict
            resampled_decision_task = DecisionTask(
                history=decision_task_dict['history'],
                condition=json.dumps(message.event_advised),  # 使用新的事件
                time_hist=decision_task_dict['time_hist'],
                time_pred=decision_task_dict['time_pred'],
                description=decision_task_dict['description'],
                label=decision_task_dict['label'],
                index=decision_task_dict['index'],
                step=decision_task_dict['step']
            )
            response = await self.make_decision(resampled_decision_task)

            # response = await self.make_decision(action_choice_request.decision_task_dict)
            prediction = response['prediction']
            resampled_predictions[f'sample_{i+1}'] = prediction

            # 绘图
            resampled_sketches[f'sample_{i+1}'] = self._convert_pil_to_base64(self._animator.animate_predict_forQ(
                x = resampled_decision_task.history[0],
                prediction = prediction[0],
                carrier_role = message.event_advised['carrier_role'],
                recipient_role = message.event_advised['recipient_role'],
                name_list = resampled_decision_task.description['name_list'],
                role_list = resampled_decision_task.description['role_list'],
            ))
            plt.close('all')

        save_sketches(resampled_sketches, path=os.path.join(self.base_path, self.path, 'resampling_sketches'), dpi=600)
        resampling_result = ResamplingResult(
            session_id=message.session_id,
            path=self.path,
            event_advised=message.event_advised,
            base_decision_task=action_choice_request.decision_task_dict,
            resampled_predictions=resampled_predictions,
            resampled_sketches=resampled_sketches
        )

        # 发布重采样结果给 Review Agent
        await self.publish_message(
            resampling_result,
            topic_id=TopicId("default", self.id.key)
        )

    @message_handler # 把模型给出的指令，包装成下一个指令，再输出
    async def handle_decision_review_result(self, message: FinalReviewResult, ctx: MessageContext) -> None:
        # Store the review result in the session memory.
        self._session_memory[message.session_id].append(message)
        print("==== Handle final decision review ====")
        print(f"Received ReviewResult with approved={message.approved}")
        print(f"Event advised: {message.event_advised}")

        # Obtain the request from previous messages.
        review_request = next(m for m in reversed(self._session_memory[message.session_id]) if isinstance(m, ReviewTask))
        assert review_request is not None

        print(f"Approval is {message.approved}")

        print(review_request.decision_candidate.keys())
        new_decision = np.array(review_request.decision_candidate[json.dumps(message.event_advised)]) # 选择ReviewAgent建议的事件对应的预测
        # new_decision = message.selected_prediction
        old_history = np.array(review_request.decision_task_dict['history'])
        new_decision = np.concatenate([new_decision, old_history[:,:4,:,2:]], axis=-1)

        # 创建新的决策任务
        new_decision_task = next(m for m in reversed(self._session_memory[message.session_id]) if isinstance(m, DecisionTask)) # 获取最近的一个DecisionTask对象
        new_decision_task.history = np.concatenate([old_history[:, -1:, :, :], new_decision], axis=1).tolist()  # 将新的决策结果合并到历史数据中
        # 更新的condition
        new_decision_task.condition = json.dumps({
            'event_type': 'Carry',
            'carrier_name': message.event_advised['recipient_name'],
            'carrier_role': message.event_advised['recipient_role'],
            'recipient_name': message.event_advised['recipient_name'],
            'recipient_role': message.event_advised['recipient_role']
            })

        # 存储到全局变量
        global next_decision_task
        next_decision_task = new_decision_task
        global chosen_instruction
        chosen_instruction = json.dumps(message.event_advised)

        decision_result = DecisionResult(
            task_dict=new_decision_task.__dict__,
            decision=new_decision.tolist(),
            event_advised=message.event_advised,
        )

        # TODO 这里decision candidate（筛选领域）和decision_sketch不一致
        save_sketches(
            # {'whatever': message.resampled_sketches[f'sample_{message.selected_option_index}']},
            {'whatever': review_request.decision_sketch[json.dumps(message.event_advised)]},
            path=os.path.join(self.base_path, self.path, 'questionnaire'),
            specific_name='cf_prediction', dpi=600)
        import shutil
        shutil.copy(
            self.cf_sketches_pdf_path[json.dumps(message.event_advised)],
            os.path.join(self.base_path, self.path, 'questionnaire_pdf', 'cf_prediction.pdf')
        )
        shutil.copy(
            self.cf_sketches_mp4_path[json.dumps(message.event_advised)],
            os.path.join(self.base_path, self.path, 'questionnaire_pdf', 'cf_prediction.mp4')
        )
        await self.publish_message(
            decision_result,
            topic_id=TopicId("default", self.id.key),
            )


    @staticmethod
    def _convert_pil_to_base64(pil_image):
        """Convert PIL Image to base64 string."""
        buffered = io.BytesIO()
        pil_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/jpeg;base64,{img_str}"
