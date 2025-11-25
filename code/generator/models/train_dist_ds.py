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
import deepspeed
import functools

def my_collate_fn(batch, mean, std, num_his, num_pred):
    if not hasattr(my_collate_fn, 'language_embedding'):
        my_collate_fn.language_embedding = LanguageEmbedding()
    language_embedding = my_collate_fn.language_embedding
    return collate_fn(batch, language_embedding, mean, std, num_his, num_pred)

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
    deepspeed.init_distributed(dist_backend="nccl")  # 初始化分布式环境
    torch.cuda.set_device(args.local_rank)  # 将当前进程绑定到指定 GPU    
    device = torch.device(f'cuda:{torch.cuda.current_device()}')
    print(f"RANK: {os.getenv('RANK')}, LOCAL_RANK: {os.getenv('LOCAL_RANK')}, WORLD_SIZE: {os.getenv('WORLD_SIZE')}")
    print(f"Process {dist.get_rank()} is using GPU {torch.cuda.current_device()} (local_rank={args.local_rank})")
    # 初始化 wandb
    if args.use_wandb:
        wandb.init(
            project="train_gmanlc_dist",
            name=args.log_file,
            config=args
        )
    train_path = '/data/zhaosiyao/gmanlc/dataset/lcts2_s4_train.jsonl'
    val_path = '/data/zhaosiyao/gmanlc/dataset/lcts2_s4_val.jsonl'
    test_path = '/data/zhaosiyao/gmanlc/dataset/lcts2_s4_test.jsonl'
    train_dataset = WholeDataset(train_path)
    val_dataset = WholeDataset(val_path)
    mean, std = 2.7, 19.6
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    my_collate = functools.partial(
        my_collate_fn,
        mean=mean,
        std=std,
        num_his=args.num_his,
        num_pred=args.num_pred
    )
    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=args.batch_size, 
                                  sampler=train_sampler,
                                  drop_last=True,
                                  num_workers=args.num_workers, 
                                  collate_fn=my_collate)
    val_dataloader = DataLoader(val_dataset, 
                                batch_size=args.batch_size, 
                                sampler=val_sampler,
                                drop_last=True,
                                num_workers=args.num_workers, 
                                collate_fn=my_collate)
    log_string(log, f'train: {len(train_dataset)}\t')
    log_string(log, f'val: {len(val_dataset)}\t')
    log_string(log, 'data loaded!')
    log_string(log, '**** training model ****')
    model.use_cuda(device)  # 将模型移动到指定 GPU
    model_engine, optimizer, _, _=deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
        # collate_fn=collate_fn,
        # training_data=train_dataset,
        config=args.ds_config
    )
    log_string(log, 'deepspeed初始化完成')
    wait = 0
    val_loss_min = float('inf')
    train_total_loss = []
    val_total_loss = []

    # Train & validation
    for epoch in range(args.max_epoch):
        print(f'epoch: {epoch+1}/{args.max_epoch}')
        if wait >= args.patience:
            log_string(log, f'early stop at epoch: {epoch:04d}')
            break
        # train
        start_train = time.time()
        model_engine.train()
        train_loss = 0
        n_tl = 0
        for batch_idx, (X, TE, LE, label) in enumerate(train_dataloader):
            X = X.to(device).half()
            TE = TE.to(device).half()
            LE = LE.to(device).half()
            label = label.to(device).half()

            optimizer.zero_grad()
            pred = model_engine(X, TE, LE)
            pred = pred * std + mean  # pred反归一化
            loss_batch = loss_criterion(pred, label)
            train_loss += float(loss_batch) * X.shape[0]
            n_tl += X.shape[0]
            model_engine.backward(loss_batch)
            model_engine.step()
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
        metrics_sum = {"mse": 0, "mae": 0, "rmse": 0, "mape": 0}
        model_engine.eval()
        with torch.no_grad():
            for batch_idx, (X, TE, LE, label) in enumerate(val_dataloader):
                X = X.to(device).half()
                TE = TE.to(device).half()
                LE = LE.to(device).half()
                label = label.to(device).half()
                pred = model_engine(X, TE, LE)
                pred = pred * std + mean
                loss_batch = loss_criterion(pred, label)
                val_loss += float(loss_batch) * X.shape[0]
                n_vl += X.shape[0]
                metrics = calculate_metrics(pred, label)
                for key in metrics:
                    metrics_sum[key] += metrics[key] * X.shape[0]
                del X, TE, LE, label, pred, loss_batch
        # val_loss /= n_vl
        # for key in metrics_sum:
        #     metrics_sum[key] /= n_vl
        # val_total_loss.append(val_loss)

        # 汇总所有进程的 val_loss 和 n_vl
        val_loss_tensor = torch.tensor(val_loss, device=device)
        n_vl_tensor = torch.tensor(n_vl, device=device)
        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(n_vl_tensor, op=dist.ReduceOp.SUM)
        val_loss = val_loss_tensor.item() / n_vl_tensor.item()
        for key in metrics_sum:
            metric_tensor = torch.tensor(metrics_sum[key], device=device)
            dist.all_reduce(metric_tensor, op=dist.ReduceOp.SUM)
            metrics_sum[key] = metric_tensor.item() / n_vl_tensor.item()
        val_total_loss.append(val_loss)
        end_val = time.time()

        if dist.get_rank() == 0:
            if args.use_wandb:
                wandb.log({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_mae": metrics_sum["mae"],
                    "val_rmse": metrics_sum["rmse"],
                    "val_mape": metrics_sum["mape"],
                    })
            log_string(log,
                   '%s | epoch: %04d/%d, training time: %.1fs, inference time: %.1fs' %
                   (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), epoch + 1,
                    args.max_epoch, end_train - start_train, end_val - start_val))
            log_string(log, 
                       f'train loss: {train_loss:.4f}, val_loss: {val_loss:.4f}')
            if val_loss <= val_loss_min:
                log_string(
                    log,
                    f'val loss decrease from {val_loss_min:.4f} to {val_loss:.4f}, saving model to {args.log_file}/best_model.pkl')
                wait = 0
                val_loss_min = val_loss
                model_engine.save_checkpoint(f'{args.log_file}/ckpt.pkl')
                if args.use_wandb:
                    wandb.save(f'{args.log_file}/ckpt.pkl')
            else:
                wait += 1
        scheduler.step()

    if dist.get_rank() == 0:
        model_engine.save_checkpoint(f'{args.log_file}/ckpt.pkl')
        log_string(log, f'Training and validation are completed, and model has been stored as {args.log_file}/best_model.pkl')
        wandb.save(f'{args.log_file}/ckpt.pkl')
    return train_total_loss, val_total_loss
