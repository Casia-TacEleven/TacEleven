import json
import numpy as np
import torch
from dataclasses import dataclass


@dataclass
class ConsistencyResult:
    start_error: float
    end_error: float
    pass_start: bool
    pass_end: bool
    pass_both: bool
    top_n: int


def generate_counter_factual_instruct(current_instruct, player_name_list, player_role_list):
    current_instruct = json.loads(current_instruct)
    current_player_name, current_player_role = current_instruct['carrier_name'], current_instruct['carrier_role']
    assert current_player_role == player_role_list[player_name_list.index(current_player_name)], f"Current carrier name {current_player_name} and role {current_player_role} do not match in the player lists."

    # assert player_name_list.index(current_player_name) == player_role_list.index(current_player_role), \
    #     f"Current carrier name {current_player_name} and role {current_player_role} do not match in the player lists."
    # Candidate
    event_type_cdd = ['Pass', 'Carry']
    carrier_name_cdd = [current_player_name]
    carrier_role_cdd = [current_player_role]
    recipient_name_cdd = player_name_list[1:12]  # Only attacking players
    recipient_role_cdd = player_role_list[1:12]  # Only attacking players
    counter_factual_instructs = []
    for event_type in event_type_cdd:
        for carrier_name, carrier_role in zip(carrier_name_cdd, carrier_role_cdd):
            if event_type == 'Pass':
                for recipient_name, recipient_role in zip(recipient_name_cdd, recipient_role_cdd):
                    if recipient_name != carrier_name:
                        instruct = {
                            "event_type": event_type,
                            "carrier_name": carrier_name,
                            "carrier_role": carrier_role,
                            "recipient_name": recipient_name,
                            "recipient_role": recipient_role
                        }
                        counter_factual_instructs.append(json.dumps(instruct))
            if event_type == 'Carry':
                recipient_name, recipient_role = current_player_name, current_player_role
                instruct = {
                    "event_type": event_type,
                    "carrier_name": carrier_name,
                    "carrier_role": carrier_role,
                    "recipient_name": recipient_name,
                    "recipient_role": recipient_role
                }
                counter_factual_instructs.append(json.dumps(instruct))
    counter_factual_instructs = [instruct for instruct in counter_factual_instructs if instruct != json.dumps(current_instruct)]  # 去掉当前指令
    return counter_factual_instructs


def consistency(instruct, model_output, player_name_list, player_role_list, top_n=2):
    """
    Calculate the consistency between model outputs and instructions provided.
        :param instruct: JSON string containing the instructions '...' .
        :param model_output: The output from the model with shape (T, N, 2).
        :param player_name_list: List of player names.
        :param player_role_list: List of player roles, w/ the same index of player_name_list.
    """
    instruct_dict = json.loads(instruct)
    event_type = instruct_dict['event_type']
    carrier_name = instruct_dict['carrier_name']
    carrier_role = instruct_dict['carrier_role']
    recipient_name = instruct_dict['recipient_name']
    recipient_role = instruct_dict['recipient_role']

    assert carrier_role == player_role_list[player_name_list.index(carrier_name)], \
        f"Carrier name {carrier_name} and role {carrier_role} do not match in the player lists."
    assert recipient_role == player_role_list[player_name_list.index(recipient_name)], \
        f"Recipient name {recipient_name} and role {recipient_role} do not match in the player lists."

    carrier_index = player_name_list.index(carrier_name)
    recipient_index = player_name_list.index(recipient_name)
    ball_index = 0  # Setting in the dataset.

    if type(model_output) is list:
        model_output = np.array(model_output)
    else:
        model_output = model_output.cpu().numpy()  # Ensure model_output is a numpy array.
    # Calculate errors for all players at the start and end
    start_errors = np.linalg.norm(model_output[0, 1:12, :] - model_output[0, ball_index, :], axis=1)
    end_errors = np.linalg.norm(model_output[-1, 1:12, :] - model_output[-1, ball_index, :], axis=1)

    # Check if the carrier is among the top two smallest start errors
    carrier_start_rank = np.argsort(start_errors)[:top_n] + 1
    pass_start = carrier_index in carrier_start_rank

    # Check if the recipient is among the top two smallest end errors
    recipient_end_rank = np.argsort(end_errors)[:top_n] + 1
    pass_end = recipient_index in recipient_end_rank

    start_error = np.linalg.norm(model_output[0, carrier_index, :] - model_output[0, ball_index, :])
    end_error = np.linalg.norm(model_output[-1, recipient_index, :] - model_output[-1, ball_index, :])

    return ConsistencyResult(
        start_error=start_error,
        end_error=end_error,
        pass_start=pass_start,
        pass_end=pass_end,
        pass_both=pass_start and pass_end,
        top_n=top_n
    )


if __name__ == "__main__":
    # Example usage
    instruct = json.dumps({
        "event_type": "pass",
        "carrier_name": "Player1",
        "carrier_role": "forward",
        "recipient_name": "Player2",
        "recipient_role": "midfielder"
    })
    model_output = torch.randn(8, 10, 4, 2)  # Simulated output for 10 time steps, 4 players, 2D coordinates
    player_name_list = ["Ball", "Player1", "Player2", "Player3"]
    player_role_list = ["Ball", "forward", "midfielder", "back"]
    cf_I = generate_counter_factual_instruct(instruct, player_name_list, player_role_list)
    print(f"Counter-factual instructions: {cf_I}")

    cons = consistency(instruct, model_output, player_name_list, player_role_list)
    print(cons)