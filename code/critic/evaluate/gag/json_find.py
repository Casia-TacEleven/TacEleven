import json
import pandas as pd


def json_find(text):
    struct_json = json.loads(text)
    if type(struct_json) == list:
        struct_json = struct_json[0]
    off_traj = struct_json.get('offensive_trajectory', None)
    off_event = struct_json.get('offensive_event', None)
    ball_traj = struct_json.get('ball_trajectory', None)
    def_traj = struct_json.get('defensive_trajectory', None)

    df_off = None
    df_ball = None
    df_def = None
    event_type = None
    event_player1 = None
    event_player2 = None

    off_traj = [{'player': traj_player['player_name'],
                 'posX.a': traj_player['trajectory'][0]['x'],
                 'posY.a': traj_player['trajectory'][0]['y'],
                 'posX.b': traj_player['trajectory'][1]['x'],
                 'posY.b': traj_player['trajectory'][1]['y'],
                 'posX.c': traj_player['trajectory'][2]['x'],
                 'posY.c': traj_player['trajectory'][2]['y'],
                 } for traj_player in off_traj]
    df_off = pd.DataFrame(off_traj)

    ball_traj = [{
        'posX.a': ball_traj[0]['x'],
        'posY.a': ball_traj[0]['y'],
        'posZ.a': ball_traj[0]['z'],
        'posX.b': ball_traj[1]['x'],
        'posY.b': ball_traj[1]['y'],
        'posZ.b': ball_traj[1]['z'],
        'posX.c': ball_traj[2]['x'],
        'posY.c': ball_traj[2]['y'],
        'posZ.c': ball_traj[2]['z'],
    }]
    df_ball = pd.DataFrame(ball_traj)

    if def_traj is not None:
        def_traj = [{'player': traj_player['player_name'],
                     'posX.a': traj_player['trajectory'][0]['x'],
                     'posY.a': traj_player['trajectory'][0]['y'],
                     'posX.b': traj_player['trajectory'][1]['x'],
                     'posY.b': traj_player['trajectory'][1]['y'],
                     'posX.c': traj_player['trajectory'][2]['x'],
                     'posY.c': traj_player['trajectory'][2]['y'],
                     } for traj_player in def_traj]
        df_def = pd.DataFrame(def_traj)
    event_type = off_event.get('event_type', None)
    event_player1 = off_event.get('player', None)
    event_player2 = off_event.get('pass_recipient', None)

    find_out = {
        'df_offense': df_off,
        'df_ball': df_ball,
        'df_defense': df_def,
        'event_type': event_type,
        'event_player1': event_player1,
        'event_player2': event_player2,
    }

    return find_out
