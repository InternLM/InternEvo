#!/bin/bash

set -e 
# set -u

NNODES=2
NPROC_PER_NODE=8
CONFIG_FILE=configs/7B_isp_sft.py
LOG_DIR=./log_record
DATA_NAME=pg19
DATA_SIZE=all
BUCKET_SIZE=B4096
SEQ_LEN=64k_16n
DP_TP_PP=1_2_4
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")  
# LOG_FILE=${LOG_DIR}/${DATA_NAME}_${DATA_SIZE}_${BUCKET_SIZE}_${SEQ_LEN}_${DP_TP_PP}_${TIMESTAMP}.log
LOG_FILE=test.log


torchrun \
    --nnodes=${NNODES} \
    --nproc_per_node=${NPROC_PER_NODE} \
    --node_rank=${NODE_RANK} \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    train.py \
    --config ${CONFIG_FILE} \
    --launcher "torch" \
    2>&1 | tee ${LOG_FILE}