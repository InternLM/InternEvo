#!/bin/bash

ENV_PATH="/conda_env/lightrlhf.sh"

source ${ENV_PATH} 

which python 

# cd /cpfs01/user/xiongyingtong/flash-attention
# sudo python setup.py install

# pip install transformers==4.45.2


############################### volcengine env #####################

export MASTER_ADDR=$MASTER_ADDR

export NNODES=2
export NODE_RANK=$RANK
export GPUS_PER_NODE=8
export MASTER_PORT=$MASTER_PORT

# Compute total world size (number of processes)
export WORLD_SIZE=$((NNODES * GPUS_PER_NODE))

# export NCCL_DEBUG=INFO

export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

############################### volcengine env ####################
cd  /cpfs01/user/xiongyingtong/InternEvo


#python examples/demo_grpo/train_ppo.py \
torchrun --nnodes $NNODES --nproc-per-node $GPUS_PER_NODE --node_rank $NODE_RANK --master-port $MASTER_PORT --master-addr $MASTER_ADDR train.py \
   --config configs/test_per.py \
   --launcher torch \
   2>&1 | tee xyt_16gpu.log


# bash examples/simple_rl/run_simple_rl_ali.sh