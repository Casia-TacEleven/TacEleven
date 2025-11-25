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
from message_class_delicated import DecisionTask, ReviewTask, ReviewResult, DecisionResult
from evaluate.gag.json_find import json_find
from instruct_following import generate_counter_factual_instruct, check_consistency, check_consistency_batch
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
    def __init__(self, model_client):
        super().__init__("A Decision Agent")
        self._model_client = model_client
        self._session_memory: Dict[str, List[DecisionTask | ReviewTask | ReviewResult]] = {}
        self._system_messages: str = """You are a football analyst. Here are descriptions for current match:\n\n"""
        self._visualizer = Visualizer()
        self._animator = Animator_dedicated(self._visualizer)
        self._animator_pdf = Animator_dedicated(self._visualizer, save2PIL=False)
        self.cf_sketches_pdf_path = {}
        self.xg = XG()
        self.xt = XT()
        self.pc = PitchControl()

    async def make_decision(self, decision_task):
        # 生成回复 with chat completion API.
        response = await self._model_client.create(
            X=decision_task.history,
            TE_x=decision_task.time_hist,
            TE_y=decision_task.time_pred,
            lang=decision_task.condition,
        )
        return response

    @message_handler
    async def handle_decision_task(self, message: DecisionTask, ctx: MessageContext) -> None:
        print("\n=== Starting Decision Task ===")
        print(f"Received task: {message.condition}...")  # Print first 100 chars of task
        # 生成进程id
        session_id = message.description['session_id']
        global save_path
        save_path = f'IS_{datetime.now().strftime("%Y-%m-%d")}_index{message.index}_{session_id}_step{message.step}'
        self.path = save_path
        os.makedirs(os.path.join('sketches', self.path), exist_ok=True)
        self.save_txt = open(os.path.join('sketches', self.path, 'reasoning.txt'), 'a')
        # os.makedirs(os.path.join('sketches', self.path, 'questionnaire_video'), exist_ok=True)

        print(f"Generated session ID: {session_id}")
        self._session_memory.setdefault(session_id, []).append(message)

        response = await self.make_decision(message)
        prediction = response['prediction']
        # 反事实全空间
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
        with open(os.path.join('sketches', self.path, 'cf_metrics.txt'), 'w') as p:
            json.dump(metrics_candidate, p, indent=4)
        global current_cf_metrics
        current_cf_metrics = metrics_candidate
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


        os.makedirs(os.path.join('sketches', self.path, 'questionnaire_pdf'), exist_ok=True)
        os.makedirs(os.path.join('sketches', self.path, 'cf_sketches_pdf'), exist_ok=True)

        self._animator_pdf.animate_predict_forQ(prediction=message.history[0], name_list=message.description['name_list'], role_list=message.description['role_list'], path=os.path.join('sketches', self.path, 'questionnaire_pdf', 'history.pdf'))
        self._animator_pdf.animate_predict_forQ(x = message.history[0], prediction = prediction[0], carrier_role = carrier_role, recipient_role = recipient_role, name_list = message.description['name_list'], role_list = message.description['role_list'], path=os.path.join('sketches', self.path, 'questionnaire_pdf', 'prediction.pdf'))
        self._animator_pdf.animate_predict_forQ(x = message.history[0], prediction = message.label[0], carrier_role = carrier_role, recipient_role = recipient_role, name_list = message.description['name_list'], role_list = message.description['role_list'], path=os.path.join('sketches', self.path, 'questionnaire_pdf', 'label.pdf'))



        # # Save sketches with session_id
        # save_sketches(sketch, path=os.path.join('sketches', self.path, datetime.now().strftime("%H-%M-%S")), dpi=600)
        # save_sketches(label_sketch, path=os.path.join('sketches', self.path), dpi=600)
        save_sketches(sketch, path=os.path.join('sketches', self.path, 'questionnaire'), dpi=600)
        save_sketches(label_sketch, path=os.path.join('sketches', self.path, 'questionnaire'), dpi=600)

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
            self._animator_pdf.animate_predict_forQ(x = message.history[0], prediction = cf_prediction[0], carrier_role = carrier_role, recipient_role = recipient_role, name_list = message.description['name_list'], role_list = message.description['role_list'], path=os.path.join('sketches', self.path, 'cf_sketches_pdf', f'{cf_instruct}.pdf'))
            self.cf_sketches_pdf_path[cf_instruct] = os.path.join('sketches', self.path, 'cf_sketches_pdf', f'{cf_instruct}.pdf')
        save_sketches(cf_sketches, path=os.path.join('sketches', self.path, 'cf_sketches'), dpi=600)

        # 发布评论任务
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
        print(f"Initial condition:\n{message.condition}", file=self.save_txt, end='\n')
        print("-" * 80, file=self.save_txt, end='\n\n')
        # print(f"Prediction:\n{prediction}", file=self.save_txt, end='\n')
        # print("-" * 80, file=self.save_txt, end='\n\n')
        await self.publish_message(
            review_task,
            topic_id=TopicId("default", self.id.key)
        )

    @message_handler
    async def handle_decision_review_result(self, message: ReviewResult, ctx: MessageContext) -> None:
        # Store the review result in the session memory.
        self._session_memory[message.session_id].append(message)
        print("==== Handle decision review ====")
        print(f"Received ReviewResult with approved={message.approved}")
        print(f"Event advised: {message.event_advised}")

        # Obtain the request from previous messages.
        review_request = next(m for m in reversed(self._session_memory[message.session_id]) if isinstance(m, ReviewTask))
        assert review_request is not None
        print(f"Approval is {message.approved}")
        new_decision = np.array(review_request.decision_candidate[json.dumps(message.event_advised)])
        old_history = np.array(review_request.decision_task_dict['history'])
        new_decision = np.concatenate([new_decision, old_history[:,:4,:,2:]], axis=-1)

        decision_task = next(m for m in reversed(self._session_memory[message.session_id]) if isinstance(m, DecisionTask))
        new_decision_task = copy.deepcopy(decision_task)
        new_decision_task.history = np.concatenate([old_history[:, -1:, :, :], new_decision], axis=1).tolist()
        new_decision_task.condition = json.dumps(
            {'event_type': 'Carry',
             'carrier_name': message.event_advised['recipient_name'],
             'carrier_role': message.event_advised['recipient_role'],
             'recipient_name': message.event_advised['recipient_name'],
             'recipient_role': message.event_advised['recipient_role']}
        )

        # 存储到全局变量
        global next_decision_task
        next_decision_task = new_decision_task
        global chosen_instruction
        chosen_instruction = json.dumps(message.event_advised)

        # Publish the code writing result.
        decision_result = DecisionResult(
            task_dict=decision_task.__dict__,
            decision=new_decision.tolist(),
            event_advised=message.event_advised,
        )

        save_sketches(
            {'whatever': review_request.decision_sketch[json.dumps(message.event_advised)]},
            path=os.path.join('sketches', self.path, 'questionnaire'),
            specific_name='cf_prediction', dpi=600)
        import shutil
        shutil.copy(
            self.cf_sketches_pdf_path[json.dumps(message.event_advised)],
            os.path.join('sketches', self.path, 'questionnaire_pdf', 'cf_prediction.pdf')
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
