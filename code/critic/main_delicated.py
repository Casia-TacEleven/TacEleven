import asyncio
import json
import os
import numpy as np
import random
from autogen_core import SingleThreadedAgentRuntime, DefaultTopicId
from autogen_ext.models.openai import OpenAIChatCompletionClient
import gc
import uuid
from openai import OpenAI

from config import Config
from message_class_delicated import DecisionTask, ReviewTask, ReviewResult, DecisionResult
from decide_agent_delicated_2 import ModelClient, DecisionAgentDedicated
from review_agent_delicated_2 import ReviewAgent
from visualize.curve_cubic_spline import Visualizer
from animator import Animator_dedicated
from evaluate.xt import XT
from evaluate.xg import XG
from evaluate.pitch_control import PitchControl

# 初始化配置
config = Config()
os.environ['OPENAI_API_KEY'] = config.OPENAI_API_KEY
os.environ['OPENAI_BASE_URL'] = config.OPENAI_BASE_URL


def process_data(data, n_hist=5, n_pred=5, remove_sth=True):
    X = np.array(data['x']).transpose(0, 2, 1)
    Y = np.array(data['y']).transpose(0, 2, 1)
    TE_X = np.array(data['t_x']).transpose(1, 0)
    TE_Y = np.array(data['t_y']).transpose(1, 0)
    lang = data['p']
    name_list = data['name_list']
    role_list = data['role_list']
    if remove_sth:
        lang_json = json.loads(lang)
        match_name = lang_json['file_name']
        del lang_json['file_name']
        del lang_json['pass_attribution']
        lang = json.dumps(lang_json)  # 转回 JSON 字符串

    def sample_uniform(array, n):
        T = array.shape[0]
        if n >= T:
            return array
        indices = np.linspace(0, T-1, n).astype(int)
        return array[indices]
    X = sample_uniform(X, n_hist)
    Y = sample_uniform(Y, n_pred)
    TE_X = sample_uniform(TE_X, n_hist)
    TE_Y = sample_uniform(TE_Y, n_pred)
    Y = Y[1:, :, :]
    TE_Y = TE_Y[1:, :]

    X = X[None, :].tolist()
    Y = Y[None, :].tolist()
    TE_X = TE_X[None, :].tolist()
    TE_Y = TE_Y[None, :].tolist()

    history = X
    condition = lang
    label = Y
    time_hist = TE_X
    time_pred = TE_Y
    description = {
        'match_name': match_name,
        'name_list': name_list,
        'role_list': role_list,
    }

    return history, condition, time_hist, time_pred, description, label


async def ma_reflection(history, condition, time_hist, time_pred, description, label, index, step):
    # 重置全局结果变量
    import decide_agent_delicated_2
    decide_agent_delicated_2.next_decision_task = None

    runtime = SingleThreadedAgentRuntime()

    # Create ModelClient instances that we can clean up later
    model_client = ModelClient(
        # base_url='http://210.75.240.13:8099',
        base_url='http://210.75.240.143:9326',
    )

    openai_client = OpenAI(
        api_key = config.OPENAI_API_KEY,
        base_url=config.OPENAI_BASE_URL
    )

    try:
        await DecisionAgentDedicated.register(
            runtime,
            "DecisionAgentDedicated",
            lambda: DecisionAgentDedicated(model_client=model_client)
        )
        await ReviewAgent.register(
            runtime,
            "ReviewAgent",
            lambda: ReviewAgent(model_client=openai_client)
        )
        runtime.start()
        await runtime.publish_message(
            message=DecisionTask(
                history=history,
                condition=condition,
                time_hist=time_hist,
                time_pred=time_pred,
                description=description,
                label=label,
                index=index,
                step=step
                ),
            topic_id=DefaultTopicId(),
        )

        await runtime.stop_when_idle()

        import decide_agent_delicated_2
        return decide_agent_delicated_2.next_decision_task, decide_agent_delicated_2.current_cf_metrics,  decide_agent_delicated_2.chosen_instruction, decide_agent_delicated_2.save_path

    finally:
        # Clean up OpenAI client sessions to prevent event loop closure errors
        # Try to close OpenAI client if it has a close method
        # if hasattr(openai_client, 'close'):
        #     await openai_client.close()
        # elif hasattr(openai_client, '_client') and hasattr(openai_client._client, 'close'):
        #     await openai_client._client.close()
        # elif hasattr(openai_client, '_http_client') and hasattr(openai_client._http_client, 'aclose'):
        #     await openai_client._http_client.aclose()

        # Force garbage collection to clean up any remaining references
        gc.collect()


async def run_ma_reflection_with_result(history, condition, time_hist, time_pred, description, label, index, step):
    new_task, current_cf_metrics, chosen_instruction, save_path = await ma_reflection(history, condition, time_hist, time_pred, description, label, index, step)
    # print(f"Final decision result: {result}")
    if new_task:
        import numpy as np
        print(f"new history: {np.array(new_task.history).shape}")
    return new_task, current_cf_metrics, chosen_instruction, save_path

with open('/home/trl/fllm/ma_reflection/test_indices.txt', 'r') as f:
    test_indices = [int(line.strip()) for line in f if line.strip().isdigit()]

# with open('/home/trl/fllm/ma_reflection/givego_curated_0730_dist_thresh_2_with_player.jsonl', 'r') as file:
#     # Read the file line by line
#     for index, line in enumerate(file):
#         try:
#         # if index in [5,10,15,20,25,30,35,40,45,50]:
#             if index in test_indices[:60]:
            # if index in [25]:
with open('/home/trl/fllm/ma_reflection/givego_curated_1014_classic_withUNSUC.jsonl', 'r') as file:
    for index, line in enumerate(file):
        attempts = 0
        try:
            if index in [154, 176, 217, 244, 330, 395, 527, 554, 656, 696]:
                data = json.loads(line.strip())
                history, condition, time_hist, time_pred, description, label = process_data(data)
                session_id = str(uuid.uuid4())
                description['session_id'] = session_id
                step = 1
                new_task, current_cf_metrics, chosen_instruction, save_path = asyncio.run(run_ma_reflection_with_result(history, condition, time_hist, time_pred, description, label, index, step=step))
                # if current_cf_metrics[chosen_instruction]['xg']['ball']['end']<0.25:
                #     while True:
                #         step += 1
                #         new_task, current_cf_metrics, chosen_instruction = asyncio.run(run_ma_reflection_with_result(new_task.history, new_task.condition, new_task.time_hist, new_task.time_pred, new_task.description, new_task.label, index, step=step))
                #         if current_cf_metrics[chosen_instruction]['xg']['ball']['end']>0.25:
                #             break

                metric_cf = {}
                for key, value in current_cf_metrics.items():
                    metric_cf[key] = {}
                    metric_cf[key]['consistency'] = int(value['consistency'])  #
                    metric_cf[key]['consistency_end'] = int(value['consistency_end'])  #
                    metric_cf[key]['consistency_end_error'] = value['consistency_end_error']  #
                    metric_cf[key]['xg'] = value['xg']['ball']['end']
                    metric_cf[key]['xt_attack'] = value['xt']['attack']['end']
                    metric_cf[key]['xt_defend'] = value['xt']['defend']['end']
                    metric_cf[key]['pc_attack'] = value['pc']['attack']['end']
                    metric_cf[key]['pc_defend'] = value['pc']['defend']['end']
                    metric_cf[key]['d_xg'] = (value['xg']['ball']['end'] - value['xg']['ball']['start'])
                    metric_cf[key]['d_xt_attack'] = (value['xt']['attack']['end'] - value['xt']['attack']['start'])
                    metric_cf[key]['d_xt_defend'] = (value['xt']['defend']['end'] - value['xt']['defend']['start'])
                    metric_cf[key]['d_pc_attack'] = (value['pc']['attack']['end'] - value['pc']['attack']['start'])
                    metric_cf[key]['d_pc_defend'] = (value['pc']['defend']['end'] - value['pc']['defend']['start'])
                # print(current_cf_metrics_change[chosen_instruction])
                # print(current_cf_metrics_change)

                metric_label = metric_cf[condition]

                metric_mar = metric_cf[chosen_instruction]

                metric_random = random.choice(list(metric_cf.values()))

                metric_mean = {}
                for metric in metric_cf[next(iter(metric_cf))].keys():
                    metric_mean[metric] = np.mean([value[metric] for value in metric_cf.values()])

                whole_metric = {
                    'random': metric_random,
                    'mean': metric_mean,
                    'label': metric_label,
                    'mar': metric_mar,
                    'condition': condition,
                }
                output_file = f'./sketches/{save_path}/whole_metric_output.json'
                with open(output_file, 'w') as outfile:
                    json.dump(whole_metric, outfile, indent=4)
                print(f"Metrics saved to {output_file}")
                # new_task = result
                # result_2 = asyncio.run(run_ma_reflection_with_result(new_task.history, new_task.condition, new_task.time_hist, new_task.time_pred, new_task.description, new_task.label, index, step=2))
                # new_task = result_2
                # result_3 = asyncio.run(run_ma_reflection_with_result(new_task.history, new_task.condition, new_task.time_hist, new_task.time_pred, new_task.description, new_task.label, index, step=3))
        except json.JSONDecodeError:
            continue  # Skip invalid JSON lines
        except ValueError:
            continue
        except Exception as e:
            print(f"Error processing index {index}: {e}")
            continue