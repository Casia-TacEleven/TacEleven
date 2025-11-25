import time
import math
import os
import datetime
import wandb
from utils.utils_ import *
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from transformers import BertModel, BertTokenizer, GPT2Tokenizer, GPT2Model


class LanguageEmbedding:
    def __init__(
        self, model_name: str = '/data/zhaosiyao/gmanlc/pretrained_model/google-bert-bert-base-uncased'
    ):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)

    def embed(self, texts: list, max_length: int = 256) -> torch.FloatTensor:
        with torch.no_grad():
            encoded = self.tokenizer(texts, padding='longest', truncation=True, return_tensors='pt')
            # encoded = self.tokenizer(texts, max_length=max_length, padding='max_length', truncation=True, return_tensors='pt')
            input_ids = encoded['input_ids']
            attention_mask = encoded['attention_mask']

            # Json String Masking
            json_special_tokens = ['{', '}', '[', ']', ':', ',']
            json_token_ids = self.tokenizer.convert_tokens_to_ids(json_special_tokens)
            json_mask = torch.zeros_like(input_ids, dtype=torch.int)
            for token_id in json_token_ids:
                json_mask = json_mask + (input_ids == token_id).int()
            combined_mask = attention_mask * (1 - json_mask)
            outputs = self.model(input_ids=input_ids, attention_mask=combined_mask, output_hidden_states=True)
            last_hidden_state = outputs.last_hidden_state  # shape: (batch_size, sequence_length, hidden_size)
        return last_hidden_state  # pooler_output=tensor[batch_size, max_length, 768]


def train_dist(model, args, log, loss_criterion, optimizer, scheduler):
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(args.local_rank)  # 将当前进程绑定到指定 GPU
    device = torch.device(f'cuda:{torch.cuda.current_device()}')
    # print(f"RANK: {os.getenv('RANK')}, LOCAL_RANK: {os.getenv('LOCAL_RANK')}, WORLD_SIZE: {os.getenv('WORLD_SIZE')}")
    # print(f"Process {dist.get_rank()} is using GPU {torch.cuda.current_device()} (local_rank={args.local_rank})")
    # 初始化 wandb
    if dist.get_rank() == 0:
        if args.use_wandb:
            wandb.init(
                project="train_gmanlc_dist",
                name=args.log_file,
                config=args
            )
    train_path = '/data/zhaosiyao/gmanlc/dataset/lcts2_s4_train.jsonl'
    val_path = '/data/zhaosiyao/gmanlc/dataset/lcts2_s4_val.jsonl'
    test_path = '/data/zhaosiyao/gmanlc/dataset/lcts2_s4_test.jsonl'
    train_dataset = WholeDataset(train_path, data_scale=args.data_scale)
    val_dataset = WholeDataset(val_path, data_scale=args.data_scale)
    mean, std = 2.7, 19.6
    language_embedding = LanguageEmbedding()
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  sampler=train_sampler,
                                  drop_last=True,
                                  num_workers=args.num_workers,
                                  collate_fn=lambda batch: collate_fn(batch, language_embedding, mean, std, args.num_his, args.num_pred))
    val_dataloader = DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                sampler=val_sampler,
                                drop_last=True,
                                num_workers=args.num_workers,
                                collate_fn=lambda batch: collate_fn(batch, language_embedding, mean, std, args.num_his, args.num_pred))
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
        if wait >= args.patience and dist.get_rank() == 0:
            log_string(log, f'early stop at epoch: {epoch:04d}')
            break
        # train
        start_train = time.time()
        model.train()
        train_loss = 0
        n_tl = 0
        for batch_idx, (X, TE, LE, label) in enumerate(train_dataloader):
            X = X.to(device)

            TE = TE.to(device)
            LE = LE.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            pred, mu, logvar = model(X, TE, LE)
            pred = pred * std + mean  # pred反归一化
            loss_batch = loss_criterion(pred, label, mu, logvar)
            train_loss += float(loss_batch) * X.shape[0]
            n_tl += X.shape[0]
            loss_batch.backward()
            optimizer.step()
            # if torch.cuda.is_available():
            #     torch.cuda.empty_cache()
            if (batch_idx+1) % 100 == 0 and dist.get_rank() == 0:
                print(f'Training batch: {batch_idx+1} in epoch:{epoch}, training batch loss:{loss_batch.item():.4f}')
            del X, TE, LE, label, pred, loss_batch
        train_loss /= n_tl
        train_total_loss.append(train_loss)
        end_train = time.time()
        start_val = time.time()
        val_loss = 0
        n_vl = 0
        metrics_sum = {
            "mse": 0, "mae": 0, "rmse": 0, "mape": 0,
            "ade": 0, "sde": 0, "fde": 0, "miss_rate": 0,
            "angular_error": 0, "velocity_error": 0
        }
        model.eval()
        with torch.no_grad():
            for batch_idx, (X, TE, LE, label) in enumerate(val_dataloader):
                X = X.to(device)
                TE = TE.to(device)
                LE = LE.to(device)
                label = label.to(device)
                pred, mu, logvar = model(X, TE, LE)
                pred = pred * std + mean
                loss_batch = loss_criterion(pred, label, mu, logvar)
                val_loss += float(loss_batch) * X.shape[0]
                n_vl += X.shape[0]
                metrics = calculate_metrics(pred, label)
                for key in metrics:
                    metrics_sum[key] += metrics[key] * X.shape[0]
                del X, TE, LE, label, pred, loss_batch
        if dist.get_rank() == 0:
            val_loss /= n_vl
            for key in metrics_sum:
                metrics_sum[key] /= n_vl
            val_total_loss.append(val_loss)
            if args.use_wandb:
                wandb.log({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_mae": metrics_sum["mae"],
                    "val_rmse": metrics_sum["rmse"],
                    "val_mape": metrics_sum["mape"],
                    "val_ade": metrics_sum["ade"],
                    "val_sde": metrics_sum["sde"],
                    "val_fde": metrics_sum["fde"],
                    "val_miss_rate": metrics_sum["miss_rate"],
                    "val_angular_error": metrics_sum["angular_error"],
                    "val_velocity_error": metrics_sum["velocity_error"],
                    })
            end_val = time.time()
            log_string(
                log,
                '%s | epoch: %04d/%d, training time: %.1fs, inference time: %.1fs' %
                (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), epoch + 1,
                args.max_epoch, end_train - start_train, end_val - start_val))
            log_string(
                log, f'train loss: {train_loss:.4f}, val_loss: {val_loss:.4f}')

            # 记录详细的轨迹预测评价指标
            log_string(log, f'Trajectory Prediction Metrics:')
            log_string(log, f'  ADE: {metrics_sum["ade"]:.4f}')
            log_string(log, f'  SDE: {metrics_sum["sde"]:.4f}')
            log_string(log, f'  FDE: {metrics_sum["fde"]:.4f}')
            log_string(log, f'  Miss Rate: {metrics_sum["miss_rate"]:.2f}%')
            log_string(log, f'  Angular Error: {metrics_sum["angular_error"]:.2f}°')
            log_string(log, f'  Velocity Error: {metrics_sum["velocity_error"]:.4f}')
            log_string(log, f'  MAE: {metrics_sum["mae"]:.4f}, RMSE: {metrics_sum["rmse"]:.4f}, MAPE: {metrics_sum["mape"]:.2f}%')
            if val_loss <= val_loss_min:
                log_string(
                    log,
                    f'val loss decrease from {val_loss_min:.4f} to {val_loss:.4f}, saving model to {args.log_file}/best_model.pkl')
                wait = 0
                val_loss_min = val_loss
                best_model_wts = model.state_dict()
                torch.save(best_model_wts, f'{args.log_file}/ckpt.pkl')
                if args.use_wandb:
                    wandb.save(f'{args.log_file}/ckpt.pkl')
            else:
                wait += 1
        scheduler.step()
    if dist.get_rank() == 0:
        model.load_state_dict(best_model_wts)
        torch.save(best_model_wts, f'{args.log_file}/ckpt.pkl')
        log_string(log, f'Training and validation are completed, and model has been stored as {args.log_file}/best_model.pkl')
        log_string(log, '=' * 80)
        log_string(log, 'FINAL TRAINING SUMMARY:')
        log_string(log, f'Best Validation Loss: {val_loss_min:.4f}')
        log_string(log, f'Total Training Epochs: {epoch + 1}')
        log_string(log, '=' * 80)
        wandb.save(f'{args.log_file}/ckpt.pkl')
    return train_total_loss, val_total_loss
