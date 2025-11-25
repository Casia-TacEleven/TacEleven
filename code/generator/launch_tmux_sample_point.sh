#!/bin/bash

# 创建日志目录
LOG_DIR="./logs"
mkdir -p $LOG_DIR

# 通用 NCCL 设置
COMMON_NCCL_ENV="NCCL_SOCKET_IFNAME=lo NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 TORCH_NCCL_BLOCKING_WAIT=1 NCCL_DEBUG=ERROR NCCL_NTHREADS=128 NCCL_BUFFSIZE=1048576 TORCH_NCCL_ASYNC_ERROR_HANDLING=1"

# 启动函数
start_experiment() {
    TMUX_NAME=$1
    GPUS=$2
    NPROC_PER_NODE=$3
    PORT=$4
    CPUS=$5
    D_VAL=$6
    SCALE=$7
    N_HIST=$8
    N_PRED=$9
    MASK=${10}
    LOG_FILE=${11}

    tmux new-session -d -s $TMUX_NAME
    tmux send-keys -t $TMUX_NAME "
source ~/.bashrc
conda activate fllm
taskset -c $CPUS bash -c '
$COMMON_NCCL_ENV MASTER_ADDR=127.0.0.1 MASTER_PORT=$PORT CUDA_VISIBLE_DEVICES=$GPUS \
torchrun --nproc-per-node $NPROC_PER_NODE --master-port $PORT main_dist_sample_point.py \
    --L 4 --K 4 --d $D_VAL --batch_size 128 --data_scale $SCALE \
    --n_hist $N_HIST --n_pred $N_PRED --mask $MASK \
    > $LOG_FILE 2>&1'" C-m
}

# 启动四个实验
# start_experiment  exp1  0,1  2  29907  0-15   256  1  3  3  true  $LOG_DIR/exp1.log
# start_experiment  exp2  2,3  2  29908  16-31  256  1  3  5  true  $LOG_DIR/exp2.log
# start_experiment  exp3  4,5  2  29909  32-47  256  1  5  3  true  $LOG_DIR/exp3.log
# start_experiment  exp4  6,7  2  29910  48-63  256  1  5  5  true  $LOG_DIR/exp4.log
# start_experiment  exp1  0,1,2,3   4   29915  0-31   256  0.05  5  5   true $LOG_DIR/exp1.log
# start_experiment  exp2  4,5,6,7   4   29920  32-63  256  0.05  5  5  false $LOG_DIR/exp2.log


start_experiment  exp-a  0,1,2,3   4   29918   0-31    128  1  5  5   true   $LOG_DIR/exp-a.log
start_experiment  exp-b  4,5,6,7   4   29919   32-63   256  1  5  5   true   $LOG_DIR/exp-b.log

# start_experiment  exp  0,1,2,3,4,5,6,7  8   29930  0-63  256  0.01  5  5  true  $LOG_DIR/exp.log


# start_experiment exp1 0,1 29901 0-15 128 0.0025 /data/zhaosiyao/gmanlc/dataset/givego_curated_0711_dist_thresh_5_frames_thresh_15.jsonl $LOG_DIR/exp1.log
# start_experiment exp2 2,3 29902 16-31 256 0.0025 /data/zhaosiyao/gmanlc/dataset/givego_curated_0711_dist_thresh_5_frames_thresh_15.jsonl $LOG_DIR/exp2.log
# start_experiment exp3 4,5 29903 32-47 128 0.001  /data/zhaosiyao/gmanlc/dataset/givego_curated_0711_dist_thresh_5.jsonl             $LOG_DIR/exp3.log
# start_experiment exp4 6,7 29904 48-63 256 0.001  /data/zhaosiyao/gmanlc/dataset/givego_curated_0711_dist_thresh_5.jsonl             $LOG_DIR/exp4.log

# start_experiment exp 0,1,2,3,4,5,6,7 29904 0-63 128 0.001  /data/zhaosiyao/gmanlc/dataset/givego_curated_0711_dist_thresh_5.jsonl  $LOG_DIR/exp.log


echo "✅ All distributed training jobs launched in tmux."
echo "💡 Use the following to monitor:"
echo "  tmux attach -t exp1"
