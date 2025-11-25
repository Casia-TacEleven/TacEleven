import asyncio
import json
import os
import numpy as np
import random
from autogen_core import SingleThreadedAgentRuntime, DefaultTopicId
from autogen_ext.models.openai import OpenAIChatCompletionClient
from config import Config
from message_class_hjj import DecisionTask, ReviewTask, ReviewResult, DecisionResult
from decide_agent_delicated_hjj import ModelClient, DecisionAgentDedicated
from review_agent_delicated_hjj import ReviewAgent
from visualize.curve_cubic_spline import Visualizer
from animator import Animator_dedicated
import gc
import uuid
from evaluate.xt import XT
from evaluate.xg import XG
from evaluate.pitch_control import PitchControl

import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
plt.rcParams['figure.max_open_warning'] = 0  # 禁用警告
plt.ioff()  # 关闭交互模式

# 初始化配置
config = Config()
os.environ['OPENAI_API_KEY'] = config.OPENAI_API_KEY
os.environ['OPENAI_BASE_URL'] = config.OPENAI_BASE_URL

from openai import OpenAI
openai_client = OpenAI(
    api_key=config.OPENAI_API_KEY,
    base_url=config.OPENAI_BASE_URL,
)

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


def metric_evaluation(current_condition, current_cf_metrics, chosen_instruction, save_path, base_path = 'sketches'):
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


    metric_label = metric_cf[current_condition]
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
        'condition': current_condition,
    }
    output_file = f'{base_path}/{save_path}/whole_metric_output.json'
    with open(output_file, 'w') as outfile:
        json.dump(whole_metric, outfile, indent=4)



async def ma_reflection(history, condition, time_hist, time_pred, description, label, index, step, base_path='sketches'):
    # 重置全局结果变量
    import decide_agent_delicated_hjj
    decide_agent_delicated_hjj.next_decision_task = None

    runtime = SingleThreadedAgentRuntime()

    # Create ModelClient instances that we can clean up later
    model_client = ModelClient(
        base_url='http://210.75.240.143:8099',
    )

    try:
        # 注册DecisionAgent 和 ReviewerAgent
        await DecisionAgentDedicated.register(
            runtime,
            "DecisionAgentDedicated",
            lambda: DecisionAgentDedicated(model_client=model_client, base_path=base_path)
        )

        await ReviewAgent.register(
            runtime,
            "ReviewAgent",
            lambda: ReviewAgent(model_client=openai_client, base_path=base_path)
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

        # 返回最终决策结果
        import decide_agent_delicated_hjj
        return decide_agent_delicated_hjj.next_decision_task, decide_agent_delicated_hjj.current_cf_metrics,  decide_agent_delicated_hjj.chosen_instruction, decide_agent_delicated_hjj.save_path

    finally:
        gc.collect()


async def run_ma_reflection_with_result(history, condition, time_hist, time_pred, description, label, index, step, base_path='sketches'):
    new_task, current_cf_metrics, chosen_instruction, save_path = await ma_reflection(history, condition, time_hist, time_pred, description, label, index, step, base_path)
    # print(f"Final decision result: {result}")
    if new_task:
        import numpy as np
        print(f"new history: {np.array(new_task.history).shape}")
    return new_task, current_cf_metrics, chosen_instruction, save_path

with open('/home/trl/fllm/ma_reflection/test_indices.txt', 'r') as f:
    test_indices = [int(line.strip()) for line in f if line.strip().isdigit()]

MAX_STEP=1
BASE_PATH = '/home/trl/fllm/sketches_errors'

async def main():
    # with open('/home/trl/fllm/ma_reflection/givego_curated_0730_dist_thresh_2_with_player.jsonl', 'r') as file:
    for repeat_idx in range(1, 2):
        # base_path = f'/home/trl/fllm/sketches_1014'
        base_path =f'{BASE_PATH}/sketches_{repeat_idx}'
        with open('/home/trl/fllm/ma_reflection/givego_curated_1014_classic_withUNSUC.jsonl', 'r') as file:
            for index, line in enumerate(file):
                attempts = 0
                try:
                    path_list = []
                    # if index in test_indices[:50]:
                    # if index in [152, 174, 215, 242, 328, 393, 525, 552, 654, 694]:
                    if index in [152+2, 174+2, 215+2, 242+2, 328+2, 393+2, 525+2, 552+2, 654+2, 694+2]:

                    # if index in [217]:
                        # print(f"\n\n=== Processing Entry {index}(Repeat {repeat_idx}/5) ===")

                        data = json.loads(line.strip())
                        history, condition, time_hist, time_pred, description, label = process_data(data)
                        # 定义场景
                        # scenario_requirements = "It is hoped to achieve partial cooperation among three players, with a third-man-run tactic with first forward, second backward and finally forward long to the most threatening area. Thus, the goal is to break through the defense of the opposing players, increase the offensive advantage and the hope of breakthrough as well as attacks "
                        # scenario_requirements = "It is hoped to achieve partial cooperation among these three people: Abdallah Dipo Sima, Ibrahima Niane, and Yan Valery. Thus, the goal is to break through the defense of the opposing players, increase the offensive advantage and the hope of breakthrough as well as attacks"  # 2过1
                        # scenario_requirements = "The goal is to break through the defense of the opposing players, increase the offensive advantage and the hope of breakthroughs and attacks. Aiming to create breakthroughs through partial cooperation, teamwork and creating goal-scoring opportunities."
                        # 中性版本（团队合作）
                        # scenario_requirements = "The goal is to break through the opposing defense, gain offensive advantages, and create potential breakthroughs and attacks. The strategy should emphasize local cooperation and teamwork to generate goal-scoring opportunities while keeping overall tactical balance."
                        # 激进版本
                        # scenario_requirements = "The goal is to aggressively break through the opposing defense, maximize offensive pressure, and prioritize creating scoring opportunities. The strategy should emphasize rapid forward movements, high-risk high-reward plays, and relentless attacks through local cooperation and teamwork, aiming to overwhelm the defense and increase the chances of scoring."
                        # 稳健版本（防守优先，控球耐心推进）
                        # scenario_requirements = "The goal is to maintain defensive stability while seeking controlled opportunities to attack. The strategy should emphasize patience, ball possession, and minimizing risks. Instead of forcing breakthroughs, the focus is on careful buildup through local cooperation and teamwork, waiting for safe chances to create goal-scoring opportunities while ensuring strong defensive coverage."

                        # 后场转移
                        # scenario_requirements = "Design a football tactic where the defensive players in the backline work cohesively to execute lateral ball movements. The primary goal is to shift the ball horizontally across the defensive line to manage the opposition's pressure and create opportunities for a strategic buildup. Ensure the tactic focuses on maintaining defensive stability while enabling smooth transitions into attacking phases."                    # 后场转移
                        # 后场转移2
                        # scenario_requirements = "Design a football tactic where the defensive players in the backline work cohesively to execute lateral ball movements. The main objective is to shift the ball horizontally across the defensive line to relieve opposition pressure, maintain possession stability, and prepare for progressive buildup. The tactic must emphasize defensive compactness, secure passing lanes, and smooth transitions into midfield. The ultimate attacking goal is to use backline circulation as a platform to advance the ball into higher zones and create goal-scoring opportunities through progressive passes, switches of play, or launching attacks that penetrate the opposition’s defensive structure."
                        # 后场转移3
                        # scenario_requirements = "Design a football tactic where the defensive players in the backline work cohesively to execute lateral ball movements, relieve opposition pressure, maintain possession stability, and create opportunities for advancing play. The objective is to shift the ball horizontally across the defensive line to open new passing lanes, connect with midfield players, and progress the ball forward while ensuring defensive compactness and smooth transitions into attacking phases."
                        # 中路

                        # 边路配合
                        # scenario_requirements = "Create a football tactic that focuses on the coordination between wide players to execute either overlapping runs or cutting inside to break through the opposition's defensive line. For overlapping runs, emphasize the timing and positioning of the wide players and full-backs to stretch the defense and create crossing opportunities. For cutting inside, focus on the ability of the wide players to move into central areas, combine with midfielders or forwards, and exploit gaps in the defensive structure. Ensure the tactic balances width and central penetration while maintaining fluidity and attacking intent."
                        # scenario_requirements = "Design a football tactic focusing on wide-area coordination. The main objective is to utilize overlapping and underlapping runs, as well as cut-ins from wide players, to stretch the opposition and break defensive lines. The tactic must emphasize timing, positioning, and wide-midfield combinations to create space for crossing, diagonal passes into the box, or switches to the far side, while balancing width with central penetration."
                        # scenario_requirements = "Design a football tactic that focuses on the coordination between wide players to execute either overlapping runs or cutting inside to break through the opposition's defensive line. For overlapping runs, emphasize the timing and positioning of the wide players and full-backs to stretch the defense and create crossing opportunities. For cutting inside, focus on the ability of the wide players to move into central areas, combine with midfielders or forwards, and exploit gaps in the defensive structure. Ensure the tactic balances width and central penetration while maintaining fluidity and attacking intent. The ultimate attacking goal is to transform wide progressions into goal-scoring chances through accurate crosses, cut-backs, diagonal passes, or combination play leading to shots inside the penalty area."
                        # 边路配合3
                        # scenario_requirements = "Design a football tactic that focuses on the coordination between wide players to execute overlapping runs or cutting inside, with the aim of breaking through the opposition’s defensive line, progressing the ball forward, and creating goal-scoring opportunities. For overlapping runs, emphasize the timing and positioning of wide players and full-backs to stretch the defense, generate width, and deliver accurate crosses or cut-backs. For cutting inside, highlight the ability of wide players to combine with midfielders or forwards, exploit central gaps, and create shooting chances from dangerous positions. The tactic must balance wide progression with central penetration while ensuring fluidity, attacking intent, and effective conversion of wide play into offensive advantages."

                        # 追求真实性 希望战略追求真实性，结合各个球员特点
                        # scenario_requirements = "Design a football tactic that aims to break through the opposing defense, gain offensive advantages, and create realistic attacking opportunities.  The tactic should emphasize local cooperation, short-range coordination, and small-group interplay rather than full-pitch vision or idealized patterns.Each player’s decisions should reflect their own technical ability, pace, awareness, and positional habits, not perfect synchronization or omniscient understanding of the game.The overall style should be authentic and conservative, maintaining tactical balance and realistic constraints on passing and movement.Passers should only act based on what they can realistically see or anticipate nearby, focusing on natural rhythm, local link-up plays, and situational awareness."
                        # scenario_requirements = "Design a football tactic that aims to break through the opposing defense, gain offensive advantages, and create realistic attacking opportunities. The decisions made by each player in each step need to be consistent with the player's own characteristics based on your knowledge of the specific player which is never perfect synchronization or omniscient understanding of the game. Incorporate the real perspective of the ball handler as much as possible and avoid using a global perspective.""
                        scenario_requirements = "Design a football tactic that aligns with realistic on-field player cognition and conform to true cognition— all decisions, movements, and passes must reflect what real players can perceive and execute in actual match conditions. Takes into account the characteristics of players and incorporate the real perspective of the ball handler as much as possible and avoid using a global perspective"

                        max_step = MAX_STEP
                        description['scenario_requirements'] = scenario_requirements
                        description['max_step'] = max_step

                        current_history = history
                        current_condition = condition
                        current_time_hist = time_hist
                        current_time_pred = time_pred
                        current_description = description
                        current_label = label

                        session_id = str(uuid.uuid4())
                        description['session_id'] = session_id

                        for step in range(1, max_step + 1):
                            print(f"\n=== Executing Step {step}/{max_step} ===")
                            new_task, current_cf_metrics, chosen_instruction, save_path  = await run_ma_reflection_with_result(
                                current_history,
                                current_condition,
                                current_time_hist,
                                current_time_pred,
                                current_description,
                                current_label,
                                index,
                                step,
                                base_path

                            )
                            if new_task is None:
                                print(f"Step {step} failed, stopping execution")
                                break
                            try:
                                metric_evaluation(current_condition, current_cf_metrics, chosen_instruction, save_path, base_path)
                            except KeyError:
                                break

                            if step < max_step:
                                current_history = new_task.history
                                current_condition = new_task.condition
                                current_time_hist = new_task.time_hist
                                current_time_pred = new_task.time_pred
                                current_description = new_task.description
                                current_label = new_task.label
                        path_list.append(save_path)

                        print(f"Completed all {max_step} steps")

                except json.JSONDecodeError:
                    attempts += 1  # Skip invalid JSON lines
                    continue
                except ValueError:
                    attempts += 1
                    continue

if __name__ == '__main__':
    asyncio.run(main())