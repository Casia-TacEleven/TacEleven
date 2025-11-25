from datetime import datetime
import os
from autogen_core import MessageContext, RoutedAgent, TopicId, default_subscription, message_handler, Image
from autogen_core.models import ChatCompletionClient, SystemMessage, UserMessage
from message_class_delicated import ReviewTask, ReviewResult
import json
from tool.dist_to_receive import calculate_receive_distance


def extract_dict_from_str(mixed_str):
    try:
        # Find the first '{' and the last '}' in the string
        mixed_str = mixed_str.split('###event')[-1].strip()
        start = mixed_str.find('{')
        end = mixed_str.rfind('}')

        if start != -1 and end != -1:
            # Extract the substring that contains the JSON object
            json_str = mixed_str[start:end+1]
            # Parse the JSON string into a Python dictionary
            data_dict = json.loads(json_str)
            return data_dict
        else:
            print("No JSON object found in the string.")
            return None
    except json.JSONDecodeError as e:
        # Handle the case where the string is not valid JSON
        print(f"Error decoding JSON: {e}")
        return None


@default_subscription
class ReviewAgent(RoutedAgent):
    def __init__(self, model_client: ChatCompletionClient):
        super().__init__("A Review Agent")
        self._model_client = model_client
        self._memory = []

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

    @message_handler
    async def handle_review_task(self, message: ReviewTask, ctx: MessageContext) -> None:
        self.save_txt = open(os.path.join('sketches', message.path, 'reasoning.txt'), 'a')

        print("\n=== Starting Review Task ===")
        print(f"Session ID: {message.session_id}")

        # Process the review task images
        print("Processing review task images...")
        processed_images = self._process_review_task(message)

        user_message_content = []

        user_message_content.append({
            "type": "text",
            "text": f"""You are a football analyst reviewer. Your task is to:
1. Review the given match situation and proposed events
2. Provide counterfactual suggestions if needed
3. Return your analysis in a structured format

Given the match situation:
{message.decision_task_dict["description"]}

The following is historical match situation visualization
"""})

        for history_img in processed_images['history_images']:
            if history_img.startswith('data:image'):
                user_message_content.append({
                    "type": "image_url",
                    "image_url": {"url": history_img}
                })

        user_message_content.append({
            "type": "text",
            "text": f"""The following is predicted attacking tactic visualization under the factual instruction
"""})
        if processed_images['prediction_image'] and processed_images['prediction_image'].startswith('data:image/jpeg;base64,'):
            user_message_content.append({
                "type": "image_url",
                "image_url": {"url": processed_images['prediction_image']}
            })

        user_message_content.append({
            "type": "text",
            "text": f"""The followings are counterfactual optional predicted attacking tactic visualizations
"""})
        for cf_img_key, cf_img_data in processed_images['cf_images'].items():
            if cf_img_data.startswith('data:image'):
                user_message_content.append({
                    "type": "text",
                    "text": f"""Counterfactual Instruction: {cf_img_key} with following predicted attacking tactic visualization
"""})
                user_message_content.append({
                    "type": "image_url",
                    "image_url": {"url": cf_img_data}
                })
        candidate = [json.loads(key) for key in processed_images['cf_images'].keys()]
        # Add final text instructions with schema options

        user_message_content.append({
            "type": "text",
            "text": f"""In all visualizations: red = teammates, blue = opponents, yellow = ball. Trajectories progress from shallow to deep, ending at the scatter point.
Player nodes corresponding to each event are highlighted with yellow edges.
Attacking team(red) is on the left and defending team(blue) is on the right.
Beside the scatters, the attacking player names and role-initials are displayed.
In the predicted attacking tactic visualization, the transparent trajectories represent the historical visualizations for easy comparison. First and foremost, the sketch must reflect the instructions, with event relevant players close to the ball.

[reasoning]
In the [reasoning] part, it is necessary to include an analysis of each sketch.
The reasoning must address the following aspects in the given sequence:
1. check the consistency between the current tactic sketch and the language instructions, must refuse if the ball is not close to the expected recipient in the sketch,
2. assess the authenticity of the tactical sketch and evaluate its feasibility given the current match situation.
3. analyse the scoring advantage,
4. analyse the risk of losing ball,
5. player suitability considering player history attributes, based on your knowledge of the specific player,
6. tactical execution success rate.

You final answer must be strictly in following format:
[summary]
A horizontal comparison of the analyses for all sketches should be conducted in a list format, and based on this comparison, an optimal choice should be made.
[event]
The [event] must be chosen uniquely from """ + json.dumps(candidate)+ """
"""
})

        # input 0.008 yuan / 1k-tokens, output 0.032 yuan / 1k-tokens

        messages = [{
            "role": "user",
            "content": user_message_content,
        }]
        completion_usage = None

        # Send to model
        while True:
            completion = self._model_client.chat.completions.create(
                model="qvq-max",  # 此处以 qvq-max 为例，可按需更换模型名称
                messages=messages,
                stream=True,
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
                            print("\n" + "=" * 20 + "完整回复" + "=" * 20 + "\n")
                            is_answering = True
                        # 打印回复过程
                        print(delta.content, end='', flush=True)
                        answer_content += delta.content

            print(f"Received response of length: {len(answer_content)}")
            # Parse response and create review result
            print("\nParsing review response...")
            review_dict = extract_dict_from_str(answer_content)
            print("Successfully parsed review response")
            if review_dict is not None:
                break
            # # TODO: Handle invalid JSON response
            # else:
            #     return False

        # Extract the complete event_advised dictionary which includes offensive_event, ball_trajectory, etc.
        event_advised = review_dict

        review_result = ReviewResult(
            session_id=message.session_id,
            event_advised=event_advised,  # Pass the complete event_advised dictionary
            approved=True
        )

        # Publish review result
        print("\nPublishing review result...")
        await self.publish_message(
            review_result,
            topic_id=TopicId("default", self.id.key)
        )

        print("\nReview Result Summary:", file=self.save_txt, end='\n')
        print("-" * 80, file=self.save_txt, end='\n')
        print(f">> Completion Usage: {completion_usage}", file=self.save_txt, end='\n')
        print(f">> Prompt Cost: {completion_usage.prompt_tokens/1000*0.008}", file=self.save_txt, end='\n')
        print(f">> Answer Cost: {completion_usage.completion_tokens/1000*0.032}", file=self.save_txt, end='\n')

        print(f">> Reasoning: {reasoning_content}", file=self.save_txt, end='\n')
        print(f">> Final Answer: {answer_content}", file=self.save_txt, end='\n')

        print(f">> Event Advised: {event_advised}", file=self.save_txt, end='\n')
        print("-" * 80, file=self.save_txt, end='\n')
        print("=== Review Task Completed ===", file=self.save_txt, end='\n')


        #  ["Goalkeeper", "Left Back", "Left Center Back", "Center Back", "Right Center Back", "Right Back",
        #     "Left Wing Back", "Right Wing Back", "Left Defensive Midfield", "Right Defensive Midfield",
        #     "Left Center Midfield", "Right Center Midfield",
        #     "Center Attacking Midfield", "Left Attacking Midfield", "Right Attacking Midfield",
        #     "Left Wing", "Right Wing", "Left Midfielder", "Right Midfielder", "Midfielder",
        #     "Center Forward", "Forward", "Attacking Midfielder", "Defensive Midfielder"]