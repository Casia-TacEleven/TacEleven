import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal as mvn

class PitchControl:
    def __init__(self):
        self.pitchLengthX = 105
        self.pitchWidthY = 68
        self.interval = 100
        self.sizeX = 50
        self.sizeY = int(self.sizeX * self.pitchWidthY / self.pitchLengthX)
        self.xx, self.yy = np.meshgrid(np.linspace(0, self.pitchLengthX, self.sizeX),
                                       np.linspace(0, self.pitchWidthY, self.sizeY))
        self.xx_2, self.yy_2 = np.meshgrid(np.linspace(-self.pitchLengthX / 2, self.pitchLengthX / 2, self.sizeX),
                                           np.linspace(-self.pitchWidthY / 2, self.pitchWidthY / 2, self.sizeY))
        self.allSize = self.sizeX * self.sizeY
        self.xx_2_flatten = self.xx_2.flatten()
        self.yy_2_flatten = self.yy_2.flatten()
        
        
    def get_metrics(self, trajectory: list):
        assert len(trajectory) > 1, len(trajectory)
        trajectory = np.array(trajectory)
        trajectory = trajectory[..., :2]
        time = trajectory.shape[0]
        
        velocity = np.zeros_like(trajectory)
        velocity_central = (trajectory[2:] - trajectory[:-2]) / 2  # 除以2是因间隔2帧
        velocity[1:-1] = velocity_central  # 中间帧用中心差分
        velocity[0] = trajectory[1] - trajectory[0]  # 首帧
        velocity[-1] = trajectory[-1] - trajectory[-2]  # 尾帧
        
        
        metrics = {
            'attack': {'average': None, 'start': None, 'end': None, 'raw': []},
            'defend': {'average': None, 'start': None, 'end': None, 'raw': []}
        }
        for t in range(time):
            traj_frame = trajectory[t]
            velo_frame = velocity[t]
            heatmap, _, _ = self._pitch_control_main(traj_frame, velo_frame)
            attack_rate = (heatmap > 0.5).sum() / heatmap.size
            defend_rate = (heatmap < 0.5).sum() / heatmap.size
            metrics['attack']['raw'].append(attack_rate)
            metrics['defend']['raw'].append(defend_rate)
        metrics['attack']['average'], metrics['attack']['start'], metrics['attack']['end'] = np.mean(metrics['attack']['raw']), metrics['attack']['raw'][0], metrics['attack']['raw'][-1]
        metrics['defend']['average'], metrics['defend']['start'], metrics['defend']['end'] = np.mean(metrics['defend']['raw']), metrics['defend']['raw'][0], metrics['defend']['raw'][-1]
        return metrics
    
    
    def _pitch_control(self, pos, vel, location, factor=None):
        if factor is None:
            factor = {'vel': 0.25, 'phy': 6.5}
        jitter = 1e-5  # to prevent identically zero covariance matrices when velocity is zero
        vel = vel + np.array([jitter, jitter])
        vel_norm = np.linalg.norm(vel)  # [3.44, 2,54]
        # angle between velocity vector & x-axis
        theta = np.arccos(vel[0] / (vel_norm+1e-10))
        # rotation matrix
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        mu = pos + vel * factor['vel']
        Srat = (vel_norm / 12) ** 6  # 13
        Ri = factor['phy']
        S = np.array([
            [np.exp(Srat) * Ri, 0], 
            [0, np.exp(-Srat) * Ri]
        ])

        Sigma = np.matmul(R, S)
        Sigma = np.matmul(Sigma, S)
        Sigma = np.matmul(Sigma, np.linalg.inv(R))
        # Sigma = R @ S @ R.T
        
        out = mvn.pdf(location, mu, Sigma, allow_singular=True) / mvn.pdf(mu, mu, Sigma, allow_singular=True)
        return out

    def _pitch_control_main(self, traj_frame, velo_frame, factor_list=None):
        Z_attack = np.zeros(self.allSize)
        Z_defend = np.zeros(self.allSize)
        Z_attack_list = []
        Z_defend_list = []
        for j in range(1, 12):
            pos = traj_frame[j, :]
            vel = velo_frame[j, :]
            factor = None  # {'vel': 0.25, 'phy': 6.5}
            if factor_list is not None:
                if self.player['playerId_event'][j] in [item["playerId"] for item in factor_list]:
                    idx = [item["playerId"] for item in factor_list].index(self.player['playerId_event'][j])
                    factor = factor_list[idx]['factor']
            Z_attack += self._pitch_control(pos, vel, np.c_[self.xx_2_flatten, self.yy_2_flatten], factor)
            Z_attack_list.append(self._pitch_control(pos, vel, np.c_[self.xx_2_flatten, self.yy_2_flatten], factor))
            
        for j in range(12, 23):
            pos = traj_frame[j, :]
            vel = velo_frame[j, :]
            factor = None  # {'vel': 0.25, 'phy': 6.5}
            if factor_list is not None:
                if self.player['playerId_event'][j] in [item["playerId"] for item in factor_list]:
                    idx = [item["playerId"] for item in factor_list].index(self.player['playerId_event'][j])
                    factor = factor_list[idx]['factor']
            Z_defend += self._pitch_control(pos, vel, np.c_[self.xx_2_flatten, self.yy_2_flatten], factor)
            Z_defend_list.append(self._pitch_control(pos, vel, np.c_[self.xx_2_flatten, self.yy_2_flatten], factor))
        
        Z_attack = Z_attack.reshape((self.sizeY, self.sizeX))
        Z_defend = Z_defend.reshape((self.sizeY, self.sizeX))
        return np.flipud(1 / (1 + np.exp((Z_defend - Z_attack)))), Z_attack_list, Z_defend_list  # >0.5是attack, <0.5是defend
    