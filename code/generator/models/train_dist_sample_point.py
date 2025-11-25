import time
import math
import os
import datetime
import wandb
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import random_split
from torch.utils.data.distributed import DistributedSampler
from transformers import BertModel, BertTokenizer, GPT2Tokenizer, GPT2Model

from utils.utils_ import *

def reduce_mean(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt


def train_dist(model, args, loss_criterion, optimizer, scheduler):
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(args.local_rank)  # 将当前进程绑定到指定 GPU
    device = torch.device(f'cuda:{torch.cuda.current_device()}')
    world_size = dist.get_world_size()
    # print(f"RANK: {os.getenv('RANK')}, LOCAL_RANK: {os.getenv('LOCAL_RANK')}, WORLD_SIZE: {os.getenv('WORLD_SIZE')}")
    # print(f"Process {dist.get_rank()} is using GPU {torch.cuda.current_device()} (local_rank={args.local_rank})")
    if dist.get_rank() == 0:
        start = time.time()
        make_dir(args.log_file)
        log = open(f'{args.log_file}/train_log.txt', 'w')
        log_string(log, str(args)[10: -1])
        log_string(log, 'trainable parameters: {:,}'.format(model.count_parameters()))
        if args.use_wandb:
            wandb.init(
                project="train_givego",
                name=args.log_file,
                config=args,
                mode='offline'
            )
    full_dataset = WholeDataset(args.data_path, data_scale=args.data_scale)
    dataset_size = len(full_dataset)
    val_size_abs = int(dataset_size * args.val_ratio)
    train_size_abs = dataset_size - val_size_abs

    generator = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = random_split(full_dataset, [train_size_abs, val_size_abs], generator=generator)

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  sampler=train_sampler,
                                  drop_last=True,
                                  num_workers=args.num_workers,
                                  collate_fn=lambda batch: collate_fn_sample_point(batch, n_hist=args.n_hist, n_pred=args.n_pred))
    val_dataloader = DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                sampler=val_sampler,
                                drop_last=True,
                                num_workers=args.num_workers,
                                collate_fn=lambda batch: collate_fn_sample_point(batch, n_hist=args.n_hist, n_pred=args.n_pred))
    if dist.get_rank() == 0:
        log_string(log, f'train: {len(train_dataset)}\t')
        log_string(log, f'val: {len(val_dataset)}\t')
        log_string(log, 'data loaded!')
        log_string(log, '**** training model ****')
    model.use_cuda(device)
    model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    wait = 0
    val_loss_min = float('inf')
    best_model_wts = None
    train_total_loss = []
    val_total_loss = []

    # Train & validation
    for epoch in range(args.max_epoch):
        train_sampler.set_epoch(epoch)  # 设置 epoch，确保数据分布一致
        val_sampler.set_epoch(epoch)
        if wait >= args.patience and dist.get_rank() == 0:
            log_string(log, f'early stop at epoch: {epoch:04d}')
            break
        # train
        start_train = time.time()
        model.train()
        train_loss = torch.tensor(0., device=device, requires_grad=False)
        train_mse_loss = torch.tensor(0., device=device, requires_grad=False)
        train_batch_num = torch.tensor(0., device=device, requires_grad=False)
        for batch_idx, (X, Y, TE_X, TE_Y, lang_list, _, _) in enumerate(train_dataloader):
            X = X.to(device)
            Y = Y.to(device)
            # X = (X-args.mean) / args.std
            TE_X = TE_X.to(device)
            TE_Y = TE_Y.to(device)
            optimizer.zero_grad()
            out_pos = model(X, TE_X, TE_Y, lang_list)
            # out_pos = out_pos * args.std + args.mean  # de-normalize
            loss_batch = loss_criterion(out_pos, Y)
            train_loss = train_loss + loss_batch
            train_mse_loss = train_mse_loss + loss_batch
            train_batch_num = train_batch_num + 1
            loss_batch.backward()
            optimizer.step()

        train_loss /= train_batch_num
        train_mse_loss /= train_batch_num
        train_loss = reduce_mean(train_loss, world_size)
        train_mse_loss = reduce_mean(train_mse_loss, world_size)
        train_total_loss.append(train_loss.item())
        end_train = time.time()


        start_val = time.time()
        model.eval()
        with torch.no_grad():
            val_loss = torch.tensor(0., device=device, requires_grad=False)
            val_mse_loss = torch.tensor(0., device=device, requires_grad=False)
            val_batch_num = torch.tensor(0., device=device, requires_grad=False)
            metrics = {
                'mse': torch.tensor(0., device=device, requires_grad=False),
                'mae': torch.tensor(0., device=device, requires_grad=False),
                'rmse': torch.tensor(0., device=device, requires_grad=False),
                'mape': torch.tensor(0., device=device, requires_grad=False),
                'ade': torch.tensor(0., device=device, requires_grad=False),
                'sde': torch.tensor(0., device=device, requires_grad=False),
                'fde': torch.tensor(0., device=device, requires_grad=False)
            }

            for batch_idx, (X, Y, TE_X, TE_Y, lang_list, _, _) in enumerate(val_dataloader):
                X = X.to(device)
                Y = Y.to(device)
                # X = (X-args.mean) / args.std
                TE_X = TE_X.to(device)
                TE_Y = TE_Y.to(device)
                out_pos = model(X, TE_X, TE_Y, lang_list)
                # out_pos = out_pos * args.std + args.mean  # de-normalize
                loss_batch = loss_criterion(out_pos, Y)
                val_loss = val_loss + loss_batch
                val_mse_loss = val_mse_loss + loss_batch
                val_batch_num = val_batch_num + 1
                metrics_outcome = calculate_metrics2(out_pos, Y)
                for key in metrics:
                    metrics[key] = metrics[key] + metrics_outcome[key]
            val_loss = val_loss / val_batch_num
            val_mse_loss = val_mse_loss / val_batch_num
            for key in metrics:
                metrics[key] = metrics[key] / val_batch_num
            val_loss = reduce_mean(val_loss, world_size)
            val_mse_loss = reduce_mean(val_mse_loss, world_size)
            for key in metrics:
                metrics[key] = reduce_mean(metrics[key], world_size)
            val_total_loss.append(val_loss.item())

            # GPU0 记录并保存
            if dist.get_rank() == 0:
                if args.use_wandb:
                    wandb.log({
                        "epoch": epoch,

                        "train_loss": train_loss,
                        "train_mse_loss": train_mse_loss,
                        "val_loss": val_loss,
                        "val_mse_loss": val_mse_loss,

                        "val_mae": metrics["mae"],
                        "val_rmse": metrics["rmse"],
                        "val_mape": metrics["mape"],
                        "val_ade": metrics["ade"],
                        "val_sde": metrics["sde"],
                        "val_fde": metrics["fde"],
                    })
                end_val = time.time()
                log_string(
                    log,
                    '%s | epoch: %04d/%d, training time: %.1fs, inference time: %.1fs' %
                    (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), epoch + 1,
                    args.max_epoch, end_train - start_train, end_val - start_val))
                log_string(log, f'train loss: {train_loss:.4f}, val_loss: {val_loss:.4f}')
                log_string(log, f'train mse loss: {train_mse_loss:.4f}, val mse loss: {val_mse_loss:.4f}')

                # 记录详细的轨迹预测评价指标
                log_string(log, f'Trajectory Prediction Metrics:')
                log_string(log, f'  ADE: {metrics["ade"]:.4f}')
                log_string(log, f'  SDE: {metrics["sde"]:.4f}')
                log_string(log, f'  FDE: {metrics["fde"]:.4f}')
                log_string(log, f'  MAE: {metrics["mae"]:.4f}, RMSE: {metrics["rmse"]:.4f}, MAPE: {metrics["mape"]:.2f}%')
                if val_loss <= val_loss_min:
                    log_string(
                        log,
                        f'val loss decrease from {val_loss_min:.4f} to {val_loss:.4f}, saving model to {args.log_file}/ckpt.pkl')
                    wait = 0
                    val_loss_min = val_loss
                    best_model_wts = model.state_dict()
                    torch.save(best_model_wts, f'{args.log_file}/ckpt.pkl')
                    if args.use_wandb:
                        wandb.save(f'{args.log_file}/ckpt.pkl')
                else:
                    wait += 1
                plot_train_val_loss(train_total_loss, val_total_loss, f'{args.log_file}/train_val_loss.png')

        scheduler.step()
    if dist.get_rank() == 0:
        end = time.time()

        model.load_state_dict(best_model_wts)
        torch.save(best_model_wts, f'{args.log_file}/ckpt.pkl')
        log_string(log, f'Training and validation are completed, and model has been stored as {args.log_file}/ckpt.pkl')
        log_string(log, '=' * 80)
        log_string(log, 'FINAL TRAINING SUMMARY:')
        log_string(log, f'Best Validation Loss: {val_loss_min:.4f}')
        log_string(log, f'Total Training Epochs: {epoch + 1}')
        log_string(log, '=' * 80)
        wandb.save(f'{args.log_file}/ckpt.pkl')
        log_string(log, 'total time: %.1fmin' % ((end - start) / 60))
        log.close()
    return train_total_loss, val_total_loss
