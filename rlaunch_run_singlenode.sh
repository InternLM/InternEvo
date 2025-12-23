#!/bin/bash

set -e 
set -u


export CUDA_DEVICE_MAX_CONNECTIONS=0
export CUDA_LAUNCH_BLOCKING=1
# export OMP_NUM_THREADS=1

NNODES=1
NPROC_PER_NODE=8
CONFIG_FILE=configs/7B_llama2.py
LOG_DIR=./log_record2
DATA_NAME=github
DATA_SIZE=all
BUCKET_SIZE=B1024
SEQ_LEN=64k_2n
DP_TP_PP=1_8_1
BUCKET_MODE="U"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")  
# LOG_FILE=${LOG_DIR}/${DATA_NAME}_${DATA_SIZE}_${BUCKET_MODE}_${BUCKET_SIZE}_${SEQ_LEN}_${DP_TP_PP}_${TIMESTAMP}.log
LOG_FILE=test_test.log


torchrun \
    --nnodes=${NNODES} \
    --nproc_per_node=${NPROC_PER_NODE} \
    train.py \
    --config ${CONFIG_FILE} \
    --launcher "torch" \
    2>&1 | tee ${LOG_FILE}