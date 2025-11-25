import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from scipy.stats import binned_statistic_2d
from mplsoccer import Pitch
from matplotlib import pyplot as plt

class XT:
    def __init__(self, path='/home/trl/fllm/ma_reflection/evaluate/data/expected_threat/event_ChampionsLeague_2425_421.csv', 
                    load_xt_map='/home/trl/fllm/ma_reflection/evaluate/data/expected_threat/xt_map.pkl'):
        self.path = path
        self.pitchLengthX = 105
        self.pitchWidthY = 68
        self.X_BINS = 32  # 统计栅格数量
        self.Y_BINS = 20
        self.interval = 100
        if not load_xt_map:
            self.event = self._load_event()
            self.xt_map = self._get_xt_map()
            with open('/home/trl/fllm/ma_reflection/evaluate/data/expected_threat/xt_map.pkl', 'wb') as f:
                pickle.dump(self.xt_map, f)
        else:
            with open(load_xt_map, 'rb') as f:  # 注意要用二进制模式 'rb'
                self.xt_map = pickle.load(f)
                
        pitch = Pitch(line_color='black', pitch_type='custom', pitch_length=self.pitchLengthX,
                    pitch_width=self.pitchWidthY,
                    line_zorder=2)
        self.draw_binStatistic(statistic=self.xt_map, pitch=pitch)
        
    def _get_xt_map(self, moves_backtrace=5):
        move_df = self.event[self.event['type'] == 'Pass']
        move_df = move_df.loc[(((move_df["endX"] != 0) & (move_df["endY"] != self.pitchWidthY)) &
                               ((move_df["endX"] != self.pitchLengthX) & (move_df["endY"] != 0)))]
        pitch = Pitch(line_color='black', pitch_type='custom', pitch_length=self.pitchLengthX,
                pitch_width=self.pitchWidthY,
                line_zorder=2)
        move = pitch.bin_statistic(move_df.x, move_df.y, statistic='count', bins=(self.X_BINS, self.Y_BINS), normalize=False)
        move_count = move["statistic"]

        shot_df = self.event.loc[self.event['type'] == "Shot"]
        shot = pitch.bin_statistic(shot_df.x, shot_df.y, statistic='count', bins=(self.X_BINS, self.Y_BINS), normalize=False)
        shot_count = shot["statistic"]
        
        goal_df = shot_df.loc[shot_df["outcome"] == 'Goal']
        goal = pitch.bin_statistic(goal_df.x, goal_df.y, statistic='count', bins=(self.X_BINS, self.Y_BINS), normalize=False)
        goal_count = goal["statistic"]
        
        move_probability = move_count / (move_count + shot_count + 1e-16)
        shot_probability = shot_count / (move_count + shot_count + 1e-16)
        goal_probability = goal_count / (shot_count + 1e-16)
        goal_probability[np.isnan(goal_probability)] = 0
        
        move_df["start_sector"] = move_df.apply(
            lambda row: tuple([i[0] for i in binned_statistic_2d(np.ravel(row.x), np.ravel(row.y),
                                                                 values="None", statistic="count",
                                                                 bins=(self.X_BINS, self.Y_BINS),
                                                                 range=[[0, self.pitchLengthX + 1],
                                                                        [0, self.pitchWidthY + 1]],
                                                                 expand_binnumbers=True)[3]]), axis=1)
        # move end index
        move_df["end_sector"] = move_df.apply(
            lambda row: tuple([i[0] for i in binned_statistic_2d(np.ravel(row.endX), np.ravel(row.endY),
                                                                 values="None", statistic="count",
                                                                 bins=(self.X_BINS, self.Y_BINS),
                                                                 range=[[0, self.pitchLengthX], [0, self.pitchWidthY]],
                                                                 expand_binnumbers=True)[3]]), axis=1)
        # create a new column for counting
        move_df['count'] = 0

        # df with summed events from each index
        df_count_starts = move_df.groupby(["start_sector"])["count"].count().reset_index()
        df_count_starts.rename(columns={'count': 'count_starts'}, inplace=True)

        transition_matrices = []
        # go through every grid
        x_values = range(1, self.X_BINS + 1)
        y_values = range(1, self.Y_BINS + 1)
        bin_x, bin_y = np.meshgrid(x_values, y_values)

        for x, y in zip(bin_x.flatten(), bin_y.flatten()):
            start_sector_df = df_count_starts[df_count_starts['start_sector'] == (x, y)]
            if start_sector_df.empty:
                T_matrix = np.zeros((self.Y_BINS, self.X_BINS))
            else:
                start_sector = start_sector_df['start_sector'].iloc[0]
                count_starts = start_sector_df['count_starts'].iloc[0]
                this_sector = move_df.loc[move_df["start_sector"] == start_sector]
                df_count_ends = this_sector.groupby(["end_sector"])["count"].count().reset_index()
                df_count_ends.rename(columns={'count': 'count_ends'}, inplace=True)
                T_matrix = np.zeros((self.Y_BINS, self.X_BINS))
                for j, row2 in df_count_ends.iterrows():
                    end_sector = row2["end_sector"]
                    value = row2["count_ends"]
                    T_matrix[end_sector[1] - 1][end_sector[0] - 1] = value
                T_matrix = T_matrix / count_starts
            transition_matrices.append(T_matrix)
        #######################################
        #  Calculating Expected Threat matrix #
        #######################################
        transition_matrices_array = np.array(transition_matrices)
        xt_map = np.zeros((self.Y_BINS, self.X_BINS))
        for i in range(moves_backtrace):
            shot_expected_payoff = goal_probability * shot_probability * 0.683 + (
                        1 - goal_probability) * shot_probability * 0.594
            move_expected_payoff = move_probability * (
                np.sum(np.sum(transition_matrices_array * xt_map, axis=2), axis=1).reshape(self.Y_BINS, self.X_BINS))
            xt_map = shot_expected_payoff + move_expected_payoff
        # let's plot it!
        title = f'Expected Threat matrix after {str(i + 1)} moves'

        return xt_map

    def get_metrics(self, trajectory: list):
        # input:  trajectory [T, N, 2]
        # output: metrics {}
        assert len(trajectory) > 1, len(trajectory)
        trajectory = np.array(trajectory)
        trajectory[..., 0] += self.pitchLengthX/2
        trajectory[..., 1] += self.pitchWidthY/2
        trajectory[..., 0] = np.clip(trajectory[..., 0], 0+0.01, self.pitchLengthX-0.01)
        trajectory[..., 1] = np.clip(trajectory[..., 1], 0+0.01, self.pitchWidthY-0.01)
        
        time = trajectory.shape[0]
        trajectory_ball, trajectory_attack, trajectory_defend = trajectory[:, 0:1, :], trajectory[:, 1:12, :], trajectory[:, 12:23, :]
        def bin_positions(x, y): 
            return binned_statistic_2d(x, y, values="None", 
                                       statistic="count", bins=(self.X_BINS, self.Y_BINS),
                                       range=[[0, self.pitchLengthX], [0, self.pitchWidthY]],
                                       expand_binnumbers=True).statistic.T
        metrics = {
            'ball': {'average': None, 'start': None, 'end': None, 'raw': []},
            'attack': {'average': None, 'start': None, 'end': None, 'raw': []},
            'defend': {'average': None, 'start': None, 'end': None, 'raw': []}
        }
        for t in range(time):
            ball_x, ball_y = trajectory_ball[t, :, 0].tolist(), trajectory_ball[t, :, 1].tolist()
            ball_bin = bin_positions(ball_x, ball_y)
            sum_score = np.sum(self.xt_map * ball_bin)
            metrics['ball']['raw'].append(sum_score) 
            
            attack_x, attack_y = trajectory_attack[t, :, 0].tolist(), trajectory_attack[t, :, 1].tolist()
            attack_bin = bin_positions(attack_x, attack_y)
            sum_score = np.sum(self.xt_map * attack_bin)
            metrics['attack']['raw'].append(sum_score)
            
            defend_x, defend_y = trajectory_defend[t, :, 0].tolist(), trajectory_defend[t, :, 1].tolist()
            defend_bin = bin_positions(defend_x, defend_y)
            defend_bin = np.flip(defend_bin, axis=(0, 1))  # defend 180-rotate
            sum_score = np.sum(self.xt_map * defend_bin)
            metrics['defend']['raw'].append(sum_score)
        metrics['ball']['average'], metrics['ball']['start'], metrics['ball']['end'] = np.mean(metrics['ball']['raw']), metrics['ball']['raw'][0], metrics['ball']['raw'][-1]
        metrics['attack']['average'], metrics['attack']['start'], metrics['attack']['end'] = np.mean(metrics['attack']['raw']), metrics['attack']['raw'][0], metrics['attack']['raw'][-1]
        metrics['defend']['average'], metrics['defend']['start'], metrics['defend']['end'] = np.mean(metrics['defend']['raw']), metrics['defend']['raw'][0], metrics['defend']['raw'][-1]
        return metrics
        
    def draw_binStatistic(self, statistic, pitch=None, is_flipud=True, title='bin statistic', is_save=True, save_path=None):
        if pitch is None:
            pitch = Pitch(line_color='black',
                          pitch_type='custom',
                          pitch_length=self.pitchLengthX,
                          pitch_width=self.pitchWidthY,
                          line_zorder=2)
        temp = pitch.bin_statistic(0, 0, statistic='count', bins=(self.X_BINS, self.Y_BINS), normalize=False)

        # plot it
        fig, ax = pitch.grid(grid_height=0.9, title_height=0.06, axis=False,
                             endnote_height=0.01, title_space=0, endnote_space=0)
        # 画图原点和数据原点是翻转的
        temp['statistic'] = statistic if not is_flipud else np.flipud(statistic)
        pcm = pitch.heatmap(temp, cmap='Oranges', edgecolor='grey', ax=ax['pitch'], vmin=0, vmax=1)
        labels = pitch.label_heatmap(temp, color='blue', fontsize=9,
                                     ax=ax['pitch'], ha='center', va='center',
                                     str_format="{0:,.2f}", zorder=3)
        # legend to our plot
        ax_cbar = fig.add_axes((1, 0.093, 0.03, 0.786))
        cbar = plt.colorbar(pcm, cax=ax_cbar)
        # fig.suptitle(title, fontsize=30)
        if not is_save:
            plt.show()
        else:
            plt.savefig(f'xt_map.pdf' if save_path is None else save_path)

        
    def _load_event(self):
        print('-' * 20 + 'start load event' + '-' * 20)
        # 读取event
        data = pd.read_csv(self.path)
        event = pd.DataFrame(columns=['minute',
                                      'second',
                                      'period',
                                      'frame',
                                      'endFrame',
                                      'duration',
                                      'possessionId',
                                      'possessionTeamId_event',
                                      'teamId_event',
                                      'playerId_event',
                                      'type',
                                      'outcome',
                                      'x',
                                      'y',
                                      'endX',
                                      'endY',
                                      'passAtb',
                                      'alignedFrame',
                                      'pattern',
                                      'recipient_id',
                                      'obv_score',
                                      'obv_concede',
                                      'obv_total',
                                      'under_pressure',
                                      'counter_pressure',
                                      'clearance_body_part_name',
                                      'clearance_aerial_won',
                                      ])
        event = event.astype({
            'minute': int,
            'second': int,
            'period': int,
            'frame': int,
            'endFrame': int,
            'possessionId': int,
            'possessionTeamId_event': int,
            'teamId_event': int,
            'playerId_event': int,
        })
        k = 0
        for i in range(data.shape[0]):
            # 筛除部分event
            if pd.isnull(data['player_id'][i]):
                continue
            if data['event_type_name'][i] not in ['Pass', 'Carry', 'Shot']:
                continue
            ## frame
            period = data['period'][i]
            timestamp = data['timestamp'][i]
            try:
                datetime_obj = self._timestamp_parse(timestamp)
            except:
                continue
            frame = (int(datetime_obj.hour) * 36000 + int(datetime_obj.minute) * 600 + int(datetime_obj.second) * 10
                     + int(datetime_obj.microsecond // 1e5))
            #  - (period - 1) * 45 * 60 * 10
            if (not pd.isnull(data['duration'][i])) and data['duration'][i] > 0:
                duration = int(data['duration'][i] / (self.interval / 1000))
            else:
                duration = 0
            endFrame = int(frame + duration)
            print(f'Process index {i}')

            ## 队伍
            teamId = data['team_id'][i]

            ## 动作结果
            outcome = None
            if data['event_type_name'][i] == 'Pass':
                outcome = 'Successful' if pd.isnull(data['outcome_name'][i]) else 'Unsuccessful'
            if data['event_type_name'][i] == 'Shot':
                outcome = data['outcome_name'][i]
            endX = None
            endY = None
            if data['event_type_name'][i] in ['Pass', 'Carry', 'Shot']:
                endX = data['end_location_x'][i] / 120 * self.pitchLengthX
                endY = data['end_location_y'][i] / 80 * self.pitchWidthY

            passAtb = None
            column_data = data.get('pass_pass_luster_label')
            if column_data is not None:
                if pd.isnull(column_data[i]):
                    print(f"{i}:Element is null")

            clearance_body_part_name = None
            clearance_aerial_won = None
            if data['event_type_name'][i] == 'Clearance':
                clearance_body_part_name = data['clearance_body_part_name'][i]
                clearance_aerial_won = data['clearance_aerial_won'][i]

            location_x = data['location_x'][i] / 120 * self.pitchLengthX
            location_y = data['location_y'][i] / 80 * self.pitchWidthY

            under_pressure = data['under_pressure'][i]
            counter_pressure = data['counterpress'][i]
            obv_score = data['obv_for_net'][i]
            obv_concede = data['obv_against_net'][i]
            obv_total = data['obv_total_net'][i]
            if pd.isnull(under_pressure):
                under_pressure = False
            if pd.isnull(counter_pressure):
                counter_pressure = False
            if pd.isnull(obv_score):
                obv_score = 0
            if pd.isnull(obv_concede):
                obv_concede = 0
            if pd.isnull(obv_total):
                obv_total = 0

            temp = [data['minute'][i],
                    data['second'][i],
                    period,
                    frame,
                    endFrame,
                    duration,
                    data['possession'][i],
                    data['possession_team_id'][i],
                    teamId,
                    data['player_id'][i],
                    data['event_type_name'][i],
                    outcome,
                    location_x,
                    location_y,
                    endX,
                    endY,
                    passAtb,
                    frame,
                    data['play_pattern_name'][i],
                    data['pass_recipient_id'][i],
                    obv_score,
                    obv_concede,
                    obv_total,
                    under_pressure,
                    counter_pressure,
                    clearance_body_part_name,
                    clearance_aerial_won,
                    ]
            event.loc[k] = temp
            k += 1
        return event
    
    @staticmethod
    def _timestamp_parse(timestamp):
        datetime_obj = None
        if timestamp.count(':') == 2:
            # '00:00:00.000'
            datetime_obj = datetime.strptime(timestamp, '%H:%M:%S.%f')
        elif timestamp.count(':') == 1:
            # '00:00.0'
            datetime_obj = datetime.strptime(timestamp, '%M:%S.%f')
        else:
            raise ValueError

        return datetime_obj
        
        
if __name__ == '__main__':
    xt = XT(path='/home/trl/fllm/ma_reflection/evaluate/data/expected_threat/event_ChampionsLeague_2425_421.csv', 
            load_xt_map='/home/trl/fllm/ma_reflection/evaluate/data/expected_threat/xt_map.pkl')