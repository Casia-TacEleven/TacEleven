import re
import pandas as pd


def re_find(text):
    offense_pattern = re.compile(
        r'Player ([\w\s.-]+?) runs from \(([\d.]+), ([\d.]+)\), passing through \(([\d.]+), ([\d.]+)\), and finally reaches \(([\d.]+), ([\d.]+)\)')
    defense_pattern = offense_pattern
    ball_pattern = re.compile(
        r'The ball moves from \(([\d.]+), ([\d.]+), ([\d.]+)\), passing through \(([\d.]+), ([\d.]+), ([\d.]+)\), and finally reaches \(([\d.]+), ([\d.]+), ([\d.]+)\)')
    event_pattern = re.compile(r'This is a (\w+) executed by ([\w\s.-]+)')

    event_part = text.split("The attacking event information is:")[1].split("The trajectory of ball is: ")[0].strip()
    ball_part = text.split("The trajectory of ball is:")[1].strip()

    offense_part = None
    match_offense = None
    df_offense = None

    defense_part = None
    match_defense = None
    df_defense = None
    if "attacking players are:" in text:
        offense_part = text.split("attacking players are:")[1].split("The attacking event information is:")[0].strip()
        match_offense = offense_pattern.findall(offense_part)
        match_offense = [(p, float(x1), float(y1), float(x2), float(y2), float(x3), float(y3)) for p, x1, y1, x2, y2, x3, y3 in match_offense]
        df_offense = pd.DataFrame(match_offense, columns=['player', 'posX.a', 'posY.a', 'posX.b', 'posY.b', 'posX.c', 'posY.c'])

    if "The trajectories of defensive players are: " in text:
        defense_part = text.split("The trajectories of defensive players are: ")[1].strip()
        match_defense = defense_pattern.findall(defense_part)
        match_defense = [(p, float(x1), float(y1), float(x2), float(y2), float(x3), float(y3)) for p, x1, y1, x2, y2, x3, y3 in match_defense]
        df_defense = pd.DataFrame(match_defense, columns=['player', 'posX.a', 'posY.a', 'posX.b', 'posY.b', 'posX.c', 'posY.c'])

    match_event = event_pattern.search(event_part)
    match_ball = ball_pattern.findall(ball_part)

    event_type, other = match_event.groups()
    event_player1 = None
    event_player2 = None
    if event_type == 'Carry':
        event_player1 = other.split('.')[0]
    if event_type == 'Pass':
        other_pattern = re.compile(r'([\w\s.-]+) to ([\w\s.-]+)\.')
        event_player1, event_player2 = other_pattern.search(other).groups()
        event_player2 = event_player2.split('.')[0]
    match_ball = [(float(x1), float(y1), float(z1), float(x2), float(y2), float(z2), float(x3), float(y3), float(z3)) for x1, y1, z1, x2, y2, z2, x3, y3, z3 in match_ball]
    df_ball = pd.DataFrame(match_ball, columns=['posX.a', 'posY.a', 'posZ.a', 'posX.b', 'posY.b', 'posZ.b', 'posX.c', 'posY.c', 'posZ.c'])

    find_out = {
        'df_offense': df_offense,
        'df_ball': df_ball,
        'df_defense': df_defense,
        'event_type': event_type,
        'event_player1': event_player1,
        'event_player2': event_player2,
    }
    return find_out

