#! /bin/bash


NNODES=1
GPUS_PER_NODE=8
GPU_NUM=$((${GPUS_PER_NODE}*${NNODES}))
WORLD_SIZE=$((${GPUS_PER_NODE}*${NNODES}))
HOST_NAME=`hostname`
NODE_RANK=0
MASTER_PORT=12345
MASTER_ADDR="127.0.0.1"
echo NNODES=$NNODES NODE_RANK=$NODE_RANK HOST_FILE=${HOST_FILE} HOST_NAME=${HOST_NAME} MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}

LAUNCHER=" \
       torchrun \
       --nproc_per_node ${GPUS_PER_NODE} \
       --nnodes ${NNODES} \
       --node_rank ${NODE_RANK} \
       --master_addr ${MASTER_ADDR} \
       --master_port ${MASTER_PORT} \
       /share/work/zhenghuihuang/lumina/InternEvo/train.py \
         --config /share/work/zhenghuihuang/lumina/InternEvo/configs/ZHH_test.py --launcher "torch"
       "

${LAUNCHER}


