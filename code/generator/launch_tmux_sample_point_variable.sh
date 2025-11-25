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
    USE_VARIATIONAL=${11}
    DATA_PATH=${12}
    LOG_FILE=${13}

    tmux new-session -d -s $TMUX_NAME
    tmux send-keys -t $TMUX_NAME "
source ~/.bashrc
conda activate fllm
taskset -c $CPUS bash -c '
$COMMON_NCCL_ENV WANDB_API_KEY=43bbc087c6aea1a23c7b081944db182051fb3e03 \
MASTER_ADDR=127.0.0.1 MASTER_PORT=$PORT CUDA_VISIBLE_DEVICES=$GPUS \
torchrun --nproc-per-node $NPROC_PER_NODE --master-port $PORT main_dist_sample_point_variable.py \
    --L 6 --K 4 --d $D_VAL --batch_size 32 --data_scale $SCALE \
    --n_hist $N_HIST --n_pred $N_PRED --mask $MASK --use_variational $USE_VARIATIONAL \
    --data_path $DATA_PATH > $LOG_FILE 2>&1'" C-m
}

# 启动四个实验
# start_experiment  exp1  0,1  2  29207  0-15   256    1  3  3  true  $LOG_DIR/exp1.log
# start_experiment  exp2  4,5  2  29208  16-31  256    1  3  5  true  $LOG_DIR/exp2.log
# start_experiment  exp3  4,5  2  29209  32-47  256    1  5  3  true  $LOG_DIR/exp3.log
# start_experiment  exp4  6,7  2  29210  48-63  256    1  5  5  true  $LOG_DIR/exp4.log
# start_experiment  exp1  0,1   2   29911  0-15   32   0.01  5  5   true  $LOG_DIR/exp1.log
# start_experiment  exp2  2,3   2   29912  16-31  64   0.01  5  5   true  $LOG_DIR/exp2.log
# start_experiment  exp3  4,5   2   29913  32-47  128  0.01  5  5   true  $LOG_DIR/exp3.log
# start_experiment  exp4  6,7   2   29914  48-63  256  0.01  5  5   true  $LOG_DIR/exp4.log

# start_experiment  exp1  0,1   2   29911  0-15   32   1  5  5   true  $LOG_DIR/exp1.log
# start_experiment  exp2  2,3   2   29912  16-31  64   1  5  5   true  $LOG_DIR/exp2.log
# start_experiment  exp3  4,5   2   29913  32-47  128  1  5  5   true  $LOG_DIR/exp3.log
# start_experiment  exp4  6,7   2   29914  48-63  256  1  5  5   true  $LOG_DIR/exp4.log

# start_experiment  exp8  0,1   2   29920  0-15  256  0.56  5  5   true  true  /home/trl/fllm/ma_reflection/givego_curated_0730_dist_thresh_2_with_player.jsonl $LOG_DIR/exp8.log
# start_experiment  exp7  2,3   2   29930  16-31  128  0.56  5  5   true  true  /home/trl/fllm/ma_reflection/givego_curated_0730_dist_thresh_2_with_player.jsonl $LOG_DIR/exp7.log
# start_experiment  exp6  4,5   2   29940  32-47  64   0.56  5  5   true  true  /home/trl/fllm/ma_reflection/givego_curated_0730_dist_thresh_2_with_player.jsonl $LOG_DIR/exp6.log
# start_experiment  exp5  4,5   2   29950  48-63  32   0.56  5  5   true  true  /home/trl/fllm/ma_reflection/givego_curated_0730_dist_thresh_2_with_player.jsonl $LOG_DIR/exp5.log


# start_experiment  exp8  0,1   2   29920  0-15  256  0.32  5  5   true  true  /home/trl/fllm/ma_reflection/givego_curated_0730_dist_thresh_2_with_player.jsonl $LOG_DIR/exp8.log
# start_experiment  exp7  2,3   2   29930  16-31  128  0.32  5  5   true  true  /home/trl/fllm/ma_reflection/givego_curated_0730_dist_thresh_2_with_player.jsonl $LOG_DIR/exp7.log
# start_experiment  exp6  4,5   2   29940  32-47  64   0.32  5  5   true  true  /home/trl/fllm/ma_reflection/givego_curated_0730_dist_thresh_2_with_player.jsonl $LOG_DIR/exp6.log
# start_experiment  exp5  4,5   2   29950  32-47  32   0.32  5  5   true  true  /home/trl/fllm/ma_reflection/givego_curated_0730_dist_thresh_2_with_player.jsonl $LOG_DIR/exp5.log


# start_experiment  exp8  0,1   2   29920  0-15  256  0.18  5  5   true  true  /home/trl/fllm/ma_reflection/givego_curated_0730_dist_thresh_2_with_player.jsonl $LOG_DIR/exp8.log
# start_experiment  exp7  2,3   2   29930  16-31  128  0.18  5  5   true  true  /home/trl/fllm/ma_reflection/givego_curated_0730_dist_thresh_2_with_player.jsonl $LOG_DIR/exp7.log
# start_experiment  exp6  4,5   2   29940  32-47  64   0.18  5  5   true  true  /home/trl/fllm/ma_reflection/givego_curated_0730_dist_thresh_2_with_player.jsonl $LOG_DIR/exp6.log
# start_experiment  exp5  4,5   2   29950  48-63  32   0.18  5  5   true  true  /home/trl/fllm/ma_reflection/givego_curated_0730_dist_thresh_2_with_player.jsonl $LOG_DIR/exp5.log

# start_experiment  exp8  0,1   2   29920  0-15  256  0.1  5  5   true  true  /home/trl/fllm/ma_reflection/givego_curated_0730_dist_thresh_2_with_player.jsonl $LOG_DIR/exp8.log
# start_experiment  exp7  2,3   2   29930  16-31  128  0.1  5  5   true  true  /home/trl/fllm/ma_reflection/givego_curated_0730_dist_thresh_2_with_player.jsonl $LOG_DIR/exp7.log
# start_experiment  exp6  4,5   2   29940  32-47  64   0.1  5  5   true  true  /home/trl/fllm/ma_reflection/givego_curated_0730_dist_thresh_2_with_player.jsonl $LOG_DIR/exp6.log
# start_experiment  exp5  4,5   2   29950  48-63  32   0.1  5  5   true  true  /home/trl/fllm/ma_reflection/givego_curated_0730_dist_thresh_2_with_player.jsonl $LOG_DIR/exp5.log
# start_experiment  exp4  4,5   2   29951  48-63  16   0.1  5  5   true  true  /home/trl/fllm/ma_reflection/givego_curated_0730_dist_thresh_2_with_player.jsonl $LOG_DIR/exp4.log


# start_experiment  exp  0,1,2,3,4,5,6,7   8   29918  0-63   256  1  5  5   true  true  /home/trl/fllm/ma_reflection/givego_curated_0728_dist_thresh_5_with_player.jsonl  $LOG_DIR/exp.log
# start_experiment  exp  0,1,2,3,4,5,6,7   8   29918  0-63   256  1  5  5   true  false  /home/trl/fllm/ma_reflection/givego_curated_0730_dist_thresh_2_with_player.jsonl  $LOG_DIR/exp.log



# start_experiment  exp_1B  0,1,2,3,4,5,6,7   8   29918  0-63   256  1  5  5   true  true  /home/trl/fllm/ma_reflection/givego_curated_0730_dist_thresh_2_with_player.jsonl  $LOG_DIR/exp.log

# start_experiment  exp_1B-10  2,3   2   29922  0-31   256  0.1  5  5   true  true  /home/trl/fllm/ma_reflection/givego_curated_0730_dist_thresh_2_with_player.jsonl  $LOG_DIR/exp1.log

# start_experiment  exp_1B  0,1,2,3,4,5,6,7  8  29926  0-63   512  0.32  5  5   true  true  /home/trl/fllm/ma_reflection/givego_curated_0730_dist_thresh_2_with_player.jsonl  $LOG_DIR/exp2.log

start_experiment  exp_1B  4,5,6,7  4  29928  0-63   512  0.32  5  5   true  true  /home/trl/fllm/ma_reflection/givego_curated_0730_dist_thresh_2_with_player.jsonl  $LOG_DIR/exp2.log

# start_experiment exp1 0,1 29901 0-15 128 0.0025 /data/zhaosiyao/gmanlc/dataset/givego_curated_0711_dist_thresh_5_frames_thresh_15.jsonl $LOG_DIR/exp1.log
# start_experiment exp2 2,3 29902 16-31 256 0.0025 /data/zhaosiyao/gmanlc/dataset/givego_curated_0711_dist_thresh_5_frames_thresh_15.jsonl $LOG_DIR/exp2.log
# start_experiment exp3 4,5 29903 32-47 128 0.001  /data/zhaosiyao/gmanlc/dataset/givego_curated_0711_dist_thresh_5.jsonl             $LOG_DIR/exp3.log
# start_experiment exp4 6,7 29904 48-63 256 0.001  /data/zhaosiyao/gmanlc/dataset/givego_curated_0711_dist_thresh_5.jsonl             $LOG_DIR/exp4.log

# start_experiment exp 0,1,2,3,4,5,6,7 29904 0-63 128 0.001  /data/zhaosiyao/gmanlc/dataset/givego_curated_0711_dist_thresh_5.jsonl  $LOG_DIR/exp.log


echo "✅ All distributed training jobs launched in tmux."
echo "💡 Use the following to monitor:"
echo "  tmux attach -t xxx"
