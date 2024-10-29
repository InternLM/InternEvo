#! /bin/bash



BASH_PATH=$(dirname "$0")
HOST_FILE=$1
NNODES=$2
GPUS_PER_NODE=16
GPU_NUM=$((${GPUS_PER_NODE}*${NNODES}))
WORLD_SIZE=$((${GPUS_PER_NODE}*${NNODES}))
HOST_NAME=`hostname`
NODE_RANK=`python3 ${BASH_PATH}/return_myrank.py ${HOST_FILE} ${HOST_NAME} `
MASTER_ADDR=`python3 ${BASH_PATH}/return_master_addr_from_hostfile.py ${HOST_FILE} `
MASTER_PORT=$3
echo NNODES=$NNODES NODE_RANK=$NODE_RANK HOST_FILE=${HOST_FILE} HOST_NAME=${HOST_NAME} MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}


LAUNCHER=" \
       torchrun \
       --nproc_per_node ${GPUS_PER_NODE} \
       --nnodes ${NNODES} \
       --node_rank ${NODE_RANK} \
       --master_addr ${MASTER_ADDR} \
       --master_port ${MASTER_PORT} \
       ${BASH_PATH}/../train.py \
       --config ${BASH_PATH}/../configs/7B_chameleon.py --launcher "torch"
        "
#         --config ${BASH_PATH}/../configs/34B_chameleon_4nodes_tianshu.py --launcher "torch"
#       "

${LAUNCHER}


