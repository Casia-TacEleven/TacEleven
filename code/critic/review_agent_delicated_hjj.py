from datetime import datetime
import os
import re
from config import Config
from autogen_core import MessageContext, RoutedAgent, TopicId, default_subscription, message_handler, Image
from autogen_core.models import ChatCompletionClient, SystemMessage, UserMessage
from message_class_hjj import ReviewTask, ReviewResult, ActionChoice, ResamplingResult, FinalReviewResult
import json
from tool.dist_to_receive import calculate_receive_distance

config = Config()   

def extract_dict_from_str(mixed_str):# TODO 考虑在这里增加完成与否的解析
    try:
        # Find the first '{' and the last '}' in the string
        mixed_str = mixed_str.split('###event')[-1].strip()
        

        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        match = re.search(json_pattern, mixed_str, re.DOTALL)
        # match = re.search(r'\{.*?\}', mixed_str, re.DOTALL)。# 用非贪婪匹配，取第一个 { ... }
        

        if not match:
            print("Unsuccessfully parsed review response")
            return None
        if match:
            json_str = match.group(0)
            data_dict = json.loads(json_str)
            print("Successfully parsed review response")
            return data_dict

    except json.JSONDecodeError as e:
        # Handle the case where the string is not valid JSON
        print(f"Error decoding JSON: {e}")
        return None



def extract_selected_option(response_text, options_list):
    """
    从模型响应中提取选择的选项索引
    """
    response_text = response_text.strip()
    # 查找 ###selected_option 部分
    # selected_section_match = re.search(r'###Option\s*\n(.*?)(?:\n###|$)', response_text, re.DOTALL | re.IGNORECASE)
    selected_section_match = re.search(r'##\s*selected_option\s*\n(.*?)(?:\n##|$)', response_text, re.DOTALL | re.IGNORECASE)
    if selected_section_match:
        selected_text = selected_section_match.group(1).strip()
        
        # 方法1: 直接匹配 "Option X" 格式
        option_match = re.search(r'Option\s+(\d+)', selected_text, re.IGNORECASE)
        if option_match:
            option_num = int(option_match.group(1))
            option_index = option_num - 1  # 转换为0-based索引
            
            # 验证索引是否有效
            if 0 <= option_index < len(options_list):
                return option_index
    
    return None



@default_subscription
class ReviewAgent(RoutedAgent):
    def __init__(self, model_client: ChatCompletionClient, base_path='sketches'):
        super().__init__("A Review Agent")
        self._model_client = model_client
        self._memory = []
        self.enable_resampling = True  # 控制是否启用重采样
        self.num_samples = 5  # 重采样次数
        self.base_path = base_path


    def _process_review_task(self, message: ReviewTask) -> dict:
        """Process the ReviewTask to extract and format images for OpenAI.

        Args:
            message: The ReviewTask containing decision sketches and predictions

        Returns:
            dict: Processed data containing:
                - prediction_image: Base64 encoded prediction visualization
                - history_images: List of Base64 encoded history visualizations
        """
        print("\n=== Processing Review Task Images ===")
        processed_data = {
            'prediction_image': None,
            'history_images': [],
            'cf_images': {},
        }

        try:
            # Extract prediction image
            print("Processing prediction image...")
            if 'prediction' in message.decision_sketch:
                prediction_img_data = message.decision_sketch['prediction']
                if isinstance(prediction_img_data, str) and prediction_img_data.startswith('data:image'):
                    print("Found base64 prediction image")
                    processed_data['prediction_image'] = prediction_img_data

            # Extract history images
            print("Processing history images...")
            if 'history' in message.decision_sketch:
                history_img_data = message.decision_sketch['history']
                if isinstance(history_img_data, str) and history_img_data.startswith('data:image'):
                    print(f"Found base64 history image")
                    processed_data['history_images'].append(history_img_data)
            # print(f"Processed {len(processed_data['history_images'])} history images")

            print("Processing counterfactual images...")
            for key in message.decision_sketch:
                if 'event_type' in key:  # counterfactual key's attribute
                    cf_img_data = message.decision_sketch[key]
                    if isinstance(cf_img_data, str) and cf_img_data.startswith('data:image'):
                        # print(f"Found base64 counterfactual image: {key}")
                        processed_data['cf_images'][key] = (cf_img_data)
            print(f"Processed {len(processed_data['cf_images'])} counterfactual images")

        except Exception as e:
            print(f"Error processing images: {type(e).__name__}")
            print(f"Error details: {str(e)}")

        return processed_data

    # TODO:[hjj] -> look
    @message_handler
    async def handle_review_task(self, message: ReviewTask, ctx: MessageContext) -> None:
        self.save_txt = open(os.path.join(self.base_path, message.path, 'reasoning.txt'), 'a')

        print("\n=== Starting Review Task ===")
        print(f"Session ID: {message.session_id}")

        # Process the review task images
        print("Processing review task images...")
        processed_images = self._process_review_task(message)

        # openai changed format qvq-max
        system_content = """# Role: You are a football analyst reviewer. Your task is to:
            1. Review the given match situation and proposed events
            2. Provide counterfactual suggestions if needed
            3. Return your analysis in a structured format
            
## Profile
- **language**: English
- **description**: A highly specialized AI designed to evaluate and optimize tactical scenarios in sports, particularly focusing on visualized data analysis for strategic decision-making.
- **background**: Developed by a team of sports analysts and AI experts, this AI has been trained on vast datasets from professional sports, including soccer, basketball, and other team-based games. It excels in interpreting complex visualizations and translating them into actionable insights.
- **personality**: Analytical, detail-oriented, and objective. The AI provides clear, concise, and unbiased evaluations, ensuring that decisions are based on data-driven logic rather than intuition.
- **expertise**: Sports tactics, data visualization interpretation, strategic planning, and performance analysis.
- **target_audience**: Coaches, sports analysts, and team strategists who rely on visual data to make informed decisions during gameplay.
            """

        user_content = []
        user_content.append({
            "type": "text",
            "text": f"""Given the match situation:
{message.decision_task_dict["description"]}
The following is historical match situation visualization
"""
             })

        for history_img in processed_images['history_images']:
            if history_img.startswith('data:image/jpeg;base64,'):
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": history_img} # TODO 这里需要split(',')[1]吗
                })

        user_content.append({
            "type": "text",
            "text": "The following is predicted attacking tactic visualization"
        })
        if processed_images['prediction_image'] and processed_images['prediction_image'].startswith('data:image/jpeg;base64,'):
            user_content.append({
                "type": "image_url",
                "image_url": {"url": processed_images['prediction_image']}
            })
        
        # Add counterfactual images
        user_content.append({
            "type": "text",
            "text": "The followings are counterfactual predicted attacking tactic visualizations"
        })

        for cf_img_key, cf_img_data in processed_images['cf_images'].items():
            if cf_img_data.startswith('data:image/jpeg;base64,'):
                user_content.append({
                    "type": "text",
                    "text": f"Counterfactual Instruction: {cf_img_key} with following predicted attacking tactic visualization"
                })
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": cf_img_data}
                })

        candidate = [json.loads(key) for key in processed_images['cf_images'].keys()]
        scenario_requirements = message.decision_task_dict["description"]["scenario_requirements"]

        instructions_text = f"""This is the previous historical situation, including the decisions made by the attacking players before:{message.decision_task_dict["history"]}
## Initialization
You are evaluating a multi-stage tactical scenario with the MANDATORY STRATEGIC OBJECTIVE:
{scenario_requirements}

In all visualizations: red = teammates, blue = opponents, yellow = ball. Trajectories progress from shallow to deep, ending at the scatter point.
Player nodes corresponding to each event are highlighted with yellow edges.
Attacking team(red) is on the left and defending team(blue) is on the right.
Beside the scatters, the attacking player names and role-initials are displayed.
In the predicted attacking tactic visualization, the teammates with the ball is predicted, and the opponents are also predicted.
In the predicted attacking tactic visualization, the transparent trajectories represent the historical visualizations for easy comparison.
                              
You are REQUIRED to:
1. **Enumerate and analyze every possible passing option** that could realistically occur from the current carrier, with attention to lane openness, distance, angle, interception risk, pressure, receiver orientation, continuation options, and alignment with the scenario requirements. Additionally, consider tactical continuity and coherence - analyze how each option connects logically to previous moves and maintains strategic flow toward the objective.
2. **Individually evaluate each candidate event** from the list below. For each candidate, provide a compact but detailed micro-analysis (advantages, risks, execution likelihood, tactical fit).  
3. **Construct a horizontal comparison** of all candidate options against each other. Use structured reasoning to rank them based on consistency with scenario requirements, scoring advantage, risk, formation organization, player suitability, and tactical execution success rate.  
It is necessary to include an analysis of each sketch and  ALL {len(candidate)} candidate options, covering aspects including consistency between the current tactic and the language instructions(must), scoring advantage, risk of lossing ball, formation organization, player suitability, and tactical execution success rate. Finally, a horizontal comparison of the analyses for all sketches should be conducted in a list format, and based on this comparison, an optimal choice should be made.
4. **Select exactly one candidate** as the next optimal tactical move. 
5. **Check the consistency between the current tactic sketch and the language instructions, must refuse if the ball is not close to the expected recipient in the sketch,as well as the player suitability considering player history attributes, based on your knowledge of the specific player,

The candidate events are:
""" + json.dumps(candidate) + """
## Output Obligation
- In the reasoning section, first present the **full per-pass analysis**, then the **candidate-focused comparison table**, and finally a concise justification for the best choice.  
- In the event section, copy the selected candidate **EXACTLY as it appears** (JSON only, no extra text). Output the event object like: {"event_type": "...", "carrier_name": "...", "carrier_role": "...", "recipient_name": "...", "recipient_role": "..."}
- Only one JSON object is allowed in the `###event` block. """ 

        user_content.append({
            "type": "text",
            "text": instructions_text
        })

        messages = [   
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]

        completion_usage = None
        while True:
            completion = self._model_client.chat.completions.create(
                model=config.DEFAULT_MODEL, 
                messages=messages,
                stream = True,
                stream_options={"include_usage": True}
            )
            reasoning_content = ""  # 定义完整思考过程
            answer_content = ""     # 定义完整回复
            is_answering = False   # 判断是否结束思考过程并开始回复

            try:
                for chunk in completion:
                    # 如果chunk.choices为空，则打印usage
                    if not chunk.choices:
                        print("\nUsage:")
                        completion_usage = chunk.usage
                    else:
                        delta = chunk.choices[0].delta
                        # 打印思考过程
                        if hasattr(delta, 'reasoning_content') and delta.reasoning_content != None:
                            print(delta.reasoning_content, end='', flush=True)
                            reasoning_content += delta.reasoning_content
                        else:
                            # 开始回复
                            if delta.content != "" and is_answering is False:
                                print("\n" + "=" * 20 + "Complete Answer" + "=" * 20 + "\n")
                                is_answering = True
                            # 打印回复过程
                            print(delta.content, end='', flush=True)
                            answer_content += delta.content
            except:
                continue

            print(f"Received response of length: {len(answer_content)}")
            # Parse response and create review result
            print("\nParsing review response...")
            review_dict = extract_dict_from_str(answer_content)
            if review_dict is not None:
                break


        event_advised = review_dict

        with open(os.path.join(self.base_path, message.path, 'reasoning.txt'), 'a') as save_txt:
            print(f"\n=== STEP {message.decision_task_dict['step']} ===", file=save_txt, end='\n')
            print("\nReview Result Summary:", file=save_txt, end='\n')
            print("-" * 80, file=save_txt, end='\n')
            print(f">> Completion Usage: {completion_usage}", file=self.save_txt, end='\n')
            print(f">> Prompt Cost: {completion_usage.prompt_tokens/1000*0.008}", file=self.save_txt, end='\n')
            print(f">> Answer Cost: {completion_usage.completion_tokens/1000*0.032}", file=self.save_txt, end='\n')

            print(f">> Reasoning: {reasoning_content}", file=self.save_txt, end='\n')
            print(f">> Final Answer: {answer_content}", file=self.save_txt, end='\n')

            print(f">> Event Advised: {event_advised}", file=save_txt, end='\n')
            print("-" * 80, file=save_txt, end='\n')
            print("=== Review Task Completed ===", file=save_txt, end='\n')

        # 跳过重采样方法
        final_review_result = FinalReviewResult(
            session_id=message.session_id,
            event_advised=event_advised,
            approved=True,
            needs_resampling=False,  # 重采样完成，设置为False
            selected_prediction=None, 
            resampled_sketches = None,
            selected_option_index=None
            
        )

        await self.publish_message(
            final_review_result,
            topic_id=TopicId("default", self.id.key)
        )


        # 重采样任务
        # action_choice = ActionChoice(
        #     session_id=message.session_id,
        #     event_advised=event_advised,  # Pass the complete event_advised dictionary
        #     approved=True,
        #     needs_resampling=self.enable_resampling,  # 重采样
        #     num_samples=self.num_samples  # 重采样次数
        # )
        
        # print("\nPublishing review result...")
        # await self.publish_message(
        #     action_choice,
        #     topic_id=TopicId("default", self.id.key)
        # )


# 3. **Example Illustrations**:
#    1. **Example 1**:
#       - **Title**: Tactical Move Analysis
#       - **format type**: text/markdown
#       - **illustrate**: This example demonstrates how to analyze a tactical move and select the optimal option based on the given scenario.
#       - **examples**:
#         ```
#         ###reasoning
#         After analyzing the current tactical situation, it is evident that the attacking team (red) is positioned favorably on the left side of the field. The ball is currently in the possession of Player A, who is in a central position. The defending team (blue) is spread out, leaving gaps in their formation.

#         Evaluating the candidate options:
#         - Option 1: Pass to Player B on the right wing. This move offers a high scoring advantage but increases the risk of losing the ball due to the defender's proximity.
#         - Option 2: Move the ball to Player C in the center. This option maintains possession and allows for better control of the game, but it may not create an immediate scoring opportunity.
#         - Option 3: Execute a long pass to Player D on the far left. This move could exploit the space behind the defense but carries a higher risk of interception.

#         Based on the analysis, Option 2 is the most balanced choice, offering a moderate scoring advantage while minimizing the risk of losing the ball.

#         ###event
#         {{"event_type": "...", "carrier_name": "...", "carrier_role": "...", "recipient_name": "...", "recipient_role": "..."}}
#         ```


# ## Workflows
# - **Target**: Evaluate and select the optimal tactical move based on the given scenario requirements and visualized data.
# - **Step 1**: Analyze the current tactical situation, including player positions, ball movement, and team formations.
# - **Step 2**: Evaluate each candidate option, considering factors such as consistency with the scenario requirements, scoring advantage, risk of losing the ball, formation organization, player suitability, and tactical execution success rate.
# - **Step 3**: Conduct a horizontal comparison of all candidate options and select the one that best aligns with the overall strategic objectives.
# - **Xxpected result**: A detailed analysis of the chosen tactical move, including reasoning and a JSON object representing the selected event.

# ## OutputFormat
# 1. **Output Format Type**:
#    - **format**: text/markdown
#    - **structure**: The output will be divided into two main sections: "reasoning" and "event."
#    - **style**: Professional and analytical, with a focus on clarity and precision.
#    - **special_requirements**: The JSON object in the "event" section must match the exact format provided in the candidates.
# 2. **Format Specifications**:
#    - **indentation**: Use consistent indentation (e.g., 4 spaces) for nested elements.
#    - **sections**: Clearly separate the "reasoning" and "event" sections using markdown headers (`###reasoning` and `###event`).
#    - **highlighting**: Use bold text (`**bold**`) to emphasize key points in the reasoning section.
# 3. **Content Quality Standards:
#    - Clarity: Each sentence must contribute meaningful analysis
#    - Precision: Avoid redundant phrases, repetitive words, or meaningless text
#    - Focus: Stay strictly on tactical evaluation and reasoning
# 3. **Example Illustrations**:
#    1. **Example 1**:
#       - **Title**: Tactical Move Analysis
#       - **format type**: text/markdown
#       - **illustrate**: This example demonstrates how to analyze a tactical move and select the optimal option based on the given scenario.
#       - **examples**:
#         ```
#         ###reasoning
#         After analyzing the current tactical situation, it is evident that the attacking team (red) is positioned favorably on the left side of the field. The ball is currently in the possession of Player A, who is in a central position. The defending team (blue) is spread out, leaving gaps in their formation.

#         Evaluating the candidate options:
#         - Option 1: Pass to Player B on the right wing. This move offers a high scoring advantage but increases the risk of losing the ball due to the defender's proximity.
#         - Option 2: Move the ball to Player C in the center. This option maintains possession and allows for better control of the game, but it may not create an immediate scoring opportunity.
#         - Option 3: Execute a long pass to Player D on the far left. This move could exploit the space behind the defense but carries a higher risk of interception.

#         Based on the analysis, Option 2 is the most balanced choice, offering a moderate scoring advantage while minimizing the risk of losing the ball.

#         ###event
#         {{"event_type": "...", "carrier_name": "...", "carrier_role": "...", "recipient_name": "...", "recipient_role": "..."}}
#         ```

# ## Initialization
# In all visualizations: red = teammates, blue = opponents, yellow = ball. Trajectories progress from shallow to deep, ending at the scatter point.
# Player nodes corresponding to each event are highlighted with yellow edges.
# Attacking team(red) is on the left and defending team(blue) is on the right.
# Beside the scatters, the attacking player names and role-initials are displayed.
# In the predicted attacking tactic visualization, the teammates with the ball is predicted, and the opponents are also predicted.
# In the predicted attacking tactic visualization, the transparent trajectories represent the historical visualizations for easy comparison.
                              
# You are evaluating a multi-stage tactical scenario with the MANDATORY STRATEGIC OBJECTIVE:
# {scenario_requirements}

# You are REQUIRED to:
# 1. **Enumerate and analyze every possible passing option** that could realistically occur from the current carrier, with attention to lane openness, distance, angle, interception risk, pressure, receiver orientation, continuation options, and alignment with the scenario requirements. Additionally, consider tactical continuity and coherence - analyze how each option connects logically to previous moves and maintains strategic flow toward the objective.
# 2. **Individually evaluate each candidate event** from the list below. For each candidate, provide a compact but detailed micro-analysis (advantages, risks, execution likelihood, tactical fit).  
# 3. **Construct a horizontal comparison** of all candidate options against each other. Use structured reasoning to rank them based on consistency with scenario requirements, scoring advantage, risk, formation organization, player suitability, and tactical execution success rate.  
# It is necessary to include an analysis of each sketch and  ALL {len(candidate)} candidate options, covering aspects including consistency between the current tactic and the language instructions(must), scoring advantage, risk of lossing ball, formation organization, player suitability, and tactical execution success rate. Finally, a horizontal comparison of the analyses for all sketches should be conducted in a list format, and based on this comparison, an optimal choice should be made.
# 4. **Select exactly one candidate** as the optimal tactical move. The choice must be unique and justified.
# 5. **CRITICAL REQUIREMENTS: Do not repeat words. End with [END] and stop.

# The candidate events are:
# """ + json.dumps(candidate) + """

# ## Output Obligation
# - In the reasoning section, first present the **full per-pass analysis**, then the **candidate-focused comparison table**, and finally a concise justification for the best choice.  
# - In the event section, copy the selected candidate **EXACTLY as it appears** (JSON only, no extra text). Output the event object like: {"event_type": "...", "carrier_name": "...", "carrier_role": "...", "recipient_name": "...", "recipient_role": "..."}
# - Only one JSON object is allowed in the `###event` block. """

                                    
# Maintain output quality - no repetitive words, filler text, or meaningless content. End with [END] and stop! 

# ## Initialization
# As the AI Tactical Scenario Evaluator, strictly prioritize the scenario requirements in all evaluations. You are evaluating a multi-stage tactical scenario with the MANDATORY STRATEGIC OBJECTIVE, which you must follow and treat as the highest priority: {scenario_requirements}
# The event must be chosen uniquely from """ + json.dumps(candidate) + """ Copy the selected event object EXACTLY as it appears in the candidates, and the event should only contains one output of json content.

# , and provide the output in the specified format

# """
# In all visualizations: red = teammates, blue = opponents, yellow = ball. Trajectories progress from shallow to deep, ending at the scatter point.
# Player nodes corresponding to each event are highlighted with yellow edges.
# Attacking team(red) is on the left and defending team(blue) is on the right.
# Beside the scatters, the attacking player names and role-initials are displayed.
# In the predicted attacking tactic visualization, the teammates with the ball is predicted, and the opponents are also predicted.
# In the predicted attacking tactic visualization, the transparent trajectories represent the historical visualizations for easy comparison.

# You are evaluating a multi-stage tactical scenario with the MANDATORY STRATEGIC OBJECTIVE:
# {scenario_requirements}

# CRITICAL: The selected option MUST be consistent with and directly advance the above scenario requirements. You need to analyze the scenario and STRICTLY follow the overall requirements of the tactical scenario.
# Do not repeat words. End with [END] and stop.  

# MUST respond strictly in this format:
# ###reasoning
# In the [reasoning] part, it is necessary to include an analysis of each sketch and  ALL {len(candidate)} candidate options, covering aspects including consistency between the current tactic and the language instructions(must), scoring advantage, risk of lossing ball, formation organization, player suitability, and tactical execution success rate. Finally, a horizontal comparison of the analyses for all sketches should be conducted in a list format, and based on this comparison, an optimal choice should be made.

# ###event
# The [event] must be chosen uniquely from """ + json.dumps(candidate) + """
# output the event object like: {"event_type": "...", "carrier_name": "...", "carrier_role": "...", "recipient_name": "...", "recipient_role": "..."}, Copy the selected event object EXACTLY as it appears in the candidates
#  """
# )


# CRITICAL REQUIREMENTS:
# 1. prioritize scenario requirement
# 2. You MUST select ONE complete event object from the candidate list above
# 3. Copy the selected event object EXACTLY as it appears in the candidates



        # Add final text instructions with schema options
#         user_message_content.append("""
# In all visualizations: red = teammates, blue = opponents, green = ball. Trajectories progress from shallow to deep, ending at the scatter point.
# Attacking team(red) is on the left and defending team(blue) is on the right.
# Beside the scatters, the attacking player names and role-initials are displayed.
# In the predicted attacking tactic visualization, the teammates with the ball is predicted, and the opponents are also predicted.
# In the predicted attacking tactic visualization, the transparent trajectories represent the historical visualizations for easy comparison.

# MUST respond strictly in this format:
# ###reasoning
# In the [reasoning] part, it is necessary to include an analysis of each sketch, covering aspects including consistency between the current tactic and the language instructions(must), scoring advantage, risk of lossing ball, formation organization, player suitability, and tactical execution success rate. Finally, a horizontal comparison of the analyses for all sketches should be conducted in a list format, and based on this comparison, an optimal choice should be made.
# ###event
# The [event] must be chosen uniquely from """ + json.dumps(candidate)
# )



        #  ["Goalkeeper", "Left Back", "Left Center Back", "Center Back", "Right Center Back", "Right Back",
        #     "Left Wing Back", "Right Wing Back", "Left Defensive Midfield", "Right Defensive Midfield",
        #     "Left Center Midfield", "Right Center Midfield",
        #     "Center Attacking Midfield", "Left Attacking Midfield", "Right Attacking Midfield",
        #     "Left Wing", "Right Wing", "Left Midfielder", "Right Midfielder", "Midfielder",
        #     "Center Forward", "Forward", "Attacking Midfielder", "Defensive Midfielder"]



    # 处理重采样结果，让模型选择最佳预测
    @message_handler
    async def handle_resampling_result(self, message: ResamplingResult, ctx: MessageContext) -> None:
        print("\n=== Starting Resampling Review ===")
        print(f"Session ID: {message.session_id}")
        print(f"Received {len(message.resampled_predictions)} resampled predictions")
        self.save_txt = open(os.path.join(self.base_path, message.path, 'reasoning.txt'), 'a')

        # 设置模型prompt
        user_message_content = []
        user_message_content.append({
            "type": "text",
            "text": """You are a football analyst reviewing multiple predictions for the same tactical scenario.
            Your task is to select the best prediction from the given options based on tactical effectiveness,
            realism, and strategic advantage. Analyze each prediction carefully."""
             })
        user_message_content.append({
            "type": "text",
            "text": f"""
            You are reviewing {len(message.resampled_predictions)} different predictions for the same tactical event:
            Event: {message.event_advised}

            Below are the visualizations for each prediction option:
            """
        })

        for i, (sample_key, sketch_data) in enumerate(message.resampled_sketches.items()):
            user_message_content.append({
                "type": "text",
                "text": f"\nPrediction Option {i + 1} ({sample_key}):"
            })
            if sketch_data.startswith('data:image/jpeg;base64,'):
                user_message_content.append({
                    "type": "image_url",
                    "image_url": {"url": sketch_data}
                })

        options_list = [f"Option {i + 1}" for i in range(len(message.resampled_predictions))]
  
        instructions_text = f"""
In all visualizations: red = teammates, blue = opponents, yellow = ball. Trajectories progress from shallow to deep, ending at the scatter point. 
Player nodes corresponding to each event are highlighted with yellow edges.
Attacking team(red) is on the left and defending team(blue) is on the right.
Beside the scatters, the attacking player names and role-initials are displayed.
In the predicted attacking tactic visualization, the teammates with the ball is predicted, and the opponents are also predicted.
In the predicted attacking tactic visualization, the transparent trajectories represent the historical visualizations for easy comparison.

Please analyze each prediction option and select the best one based on:
1. Tactical soundness and realism
2. Strategic advantage and scoring potential  
3. Risk assessment and ball retention
4. Player positioning and movement efficiency

MUST respond strictly in this format:
###reasoning
In the [reasoning] part, analysis and select the prediction image with predicted player positions that best fits the next tactical instruction:{message.event_advised}
Consider both the positions of the receiver, the passer as well as the ball and the smoothness of the movement trajectory

###selected_option
[Choose exactly one option from the following list: {', '.join(options_list)}]
IMPORTANT: 
- Your selected_option MUST be exactly one of: {', '.join(options_list)}
- Do not include any additional text in the selected_option section
- Example valid response: "Option 1" or "Option 3"
"""
        
        user_message_content.append({
            "type": "text",
            "text": instructions_text
        })

        messages = [
            {"role": "user", "content": user_message_content}
        ]

        while True:
            completion = self._model_client.chat.completions.create(
                model=config.DEFAULT_MODEL , 
                messages=messages,
                stream = True,
                stream_options={"include_usage": True}
            )
            reasoning_content = ""  # 定义完整思考过程
            answer_content = ""     # 定义完整回复
            is_answering = False   # 判断是否结束思考过程并开始回复

            for chunk in completion:
                # 如果chunk.choices为空，则打印usage
                if not chunk.choices:
                    print("\nUsage:")
                    completion_usage = chunk.usage
                else:
                    delta = chunk.choices[0].delta
                    # 打印思考过程
                    if hasattr(delta, 'reasoning_content') and delta.reasoning_content != None:
                        print(delta.reasoning_content, end='', flush=True)
                        reasoning_content += delta.reasoning_content
                    else:
                        # 开始回复
                        if delta.content != "" and is_answering is False:
                            print("\n" + "=" * 20 + "Complete Answer" + "=" * 20 + "\n")
                            is_answering = True
                        # 打印回复过程
                        print(delta.content, end='', flush=True)
                        answer_content += delta.content

            print(f"Received resampling selection response: {answer_content}")
            resample_choice = extract_selected_option(answer_content, options_list)
            print("Successfully parsed resample response")
            if resample_choice is not None:
                break
            else:
                print("Failed to parse selected option, retrying...")
                continue
        selected_prediction = message.resampled_predictions[f'sample_{resample_choice}']

        with open(os.path.join(self.base_path, message.path, 'reasoning.txt'), 'a') as save_txt:
            print(f"\n=== RESAMPLING SELECTION ===", file=save_txt)
            print(f"Selected Option: {resample_choice}", file=save_txt)
            print(f">> Completion Usage: {completion_usage}", file=self.save_txt, end='\n')
            print(f">> Prompt Cost: {completion_usage.prompt_tokens/1000*0.008}", file=self.save_txt, end='\n')
            print(f">> Answer Cost: {completion_usage.completion_tokens/1000*0.032}", file=self.save_txt, end='\n')

            print(f">> Reasoning: {reasoning_content}", file=self.save_txt, end='\n')
            print(f">> Final Answer: {answer_content}", file=self.save_txt, end='\n')

            print("-" * 80, file=save_txt)
        
        final_review_result = FinalReviewResult(
            session_id=message.session_id,
            event_advised=message.event_advised,
            approved=True,
            needs_resampling=False,  # 重采样完成，设置为False
            selected_prediction=selected_prediction, 
            resampled_sketches = message.resampled_sketches,
            selected_option_index=resample_choice
            
        )

        await self.publish_message(
            final_review_result,
            topic_id=TopicId("default", self.id.key)
        )
        
        print(f"=== Resampling Review Completed, Selected Option {resample_choice + 1} ===")
        


    