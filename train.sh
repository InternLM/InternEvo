#!/bin/bash
set -ex

T=`date +%Y%m%d_%H%M%S`   # 获取时间戳

# 激活真实运行环境
source /mnt/afs/share_data/huangting3/envs/internvl_qwen3.sh
which python

ROOT=/mnt/afs/huangting3/workspace/InternEvo  # 代码目录，用自己的
cd $ROOT

export JOB_NAME="ht_debug_fsdp2"
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
# export CUDA_LAUNCH_BLOCKING=1


OUT_DIR=/mnt/afs/huangting3/workspace/InternEvo/RUN/debug_dcp # 输出路径(代码相关)
mkdir -p ${OUT_DIR}/logs         # 日志路径(代码相关)
# LOG_FILE=${OUT_DIR}/logs/log_node${RANK}_${T}.log
LOG_FILE=${OUT_DIR}/logs/log_node${RANK}.log

CONFIG=/mnt/afs/huangting3/workspace/InternEvo/configs/7B_isp_sft.py

cd $ROOT
export PYTHONPATH=$ROOT:$PYTHONPATH
export ENV_PATH=${ENV_PATH:-"/mnt/afs/share_data/huangting3/envs/internvl_torch2.5.1_qwen3"}

# export NCCL_DEBUG=INFO
export NCCL_IB_TIMEOUT=22   
export NCCL_IB_RETRY_CNT=13 
export NCCL_IB_AR_THRESHOLD=0

MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-10086}

GPUS_PER_NODE=${GPUS_PER_NODE:-8}
RANK=${RANK:-0} # srun env node rank
WORLD_SIZE=${WORLD_SIZE:-1} # srun env node num

echo "nnodes=${WORLD_SIZE}, node_rank=${RANK}"

$ENV_PATH/bin/torchrun \
    --nproc_per_node=$GPUS_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --nnodes=$WORLD_SIZE \
    --node_rank=$RANK \
    train.py --config $CONFIG --launcher torch --seed 42 --backend nccl \
    2>&1 | tee ${LOG_FILE}
