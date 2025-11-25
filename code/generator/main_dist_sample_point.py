import argparse
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import warnings
import random
import string
import torch.distributed as dist

warnings.filterwarnings("ignore")

from utils.utils_ import log_string, plot_train_val_loss
from utils.utils_ import count_parameters
from utils.utils_ import make_dir
from models.TacGen_once_predict import TacGen, LossCriterion
from models.train_dist_sample_point import train_dist

parser = argparse.ArgumentParser()

parser.add_argument('--L', type=int, default=2,
                    help='number of STAtt Blocks')
parser.add_argument('--K', type=int, default=4,
                    help='number of attention heads')
parser.add_argument('--d', type=int, default=64,
                    help='dims of each head attention outputs')
parser.add_argument('--train_ratio', type=float, default=0.7,
                    help='training set [default : 0.7]')
parser.add_argument('--val_ratio', type=float, default=0.3,
                    help='validation set [default : 0.1]')
parser.add_argument('--batch_size', type=int, default=32,
                    help='batch size')
parser.add_argument('--num_workers', type=int, default=8,
                    help='worker number')
parser.add_argument('--max_epoch', type=int, default=100,
                    help='epoch to run')
parser.add_argument('--patience', type=int, default=5,
                    help='patience for early stop')
parser.add_argument('--learning_rate', type=float, default=1e-4,
                    help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=10,
                    help='decay epoch')
exp_time = time.strftime("%Y%m%d_%H%M%S")
random_str = ''.join(random.choices(string.ascii_letters + string.digits, k=16))
parser.add_argument('--log_file', default=f'./saves/full/{exp_time}_{random_str}',
                    help='log file')
parser.add_argument('--device', default=f'cuda:3',
                    help='cuda_device')
parser.add_argument('--local_rank', type=int, default=int(os.getenv('LOCAL_RANK', 0)),
                    help='GPU id to use for distributed training')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')
parser.add_argument('--use_wandb', type=bool, default=True,
                    help='use wandb for logging')
parser.add_argument('--data_scale', type=float, default=1.0)
parser.add_argument('--beta', type=float, default=10)
parser.add_argument('--n_hist', type=int, default=3)
parser.add_argument('--n_pred', type=int, default=3)
parser.add_argument('--mask', type=str, default='true', choices=['true', 'false'])


parser.add_argument('--data_path', type=str, default='/data/zhaosiyao/gmanlc/dataset/givego_curated_0730_dist_thresh_2_with_player.jsonl', help='path to the dataset')
args = parser.parse_args()

def main():
    model = TacGen(
        L=args.L,
        K=args.K,
        d=args.d,
        mask=args.mask
    )
    loss_criterion = LossCriterion(beta=args.beta)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_epoch, gamma=0.9)

    train_dist(model, args, loss_criterion, optimizer, scheduler)


if __name__ == '__main__':
    main()
