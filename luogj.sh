#!/bin/bash

source /root/miniconda3/bin/activate
conda activate internevo
cd /mnt/shared-storage-user/ailab-sys/lusitian/workspace/InternEvo

# # 遍历所有环境变量
# for var in $(env | awk -F= '{print $1}'); do
#     # 检查变量名是否合法
#     if ! [[ $var =~ ^[a-zA-Z_][a-zA-Z0-9_]*$ ]]; then
#         echo "Skipping invalid variable: $var"
#         unset $var
#     fi
# done

# # set -u
# export CUDA_HOME=/usr/local/cuda-11.8
# export PATH=$CUDA_HOME/bin:$PATH
# export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# # 添加动态库路径
# sudo tee /etc/ld.so.conf.d/nvidia.conf <<EOF
# /usr/local/nvidia/lib64
# /usr/local/nvidia/lib
# /usr/local/cuda-11.8/lib64
# EOF

DEFAULT_NODE_RANK=0
DEFAULT_NNODES=1
DEFAULT_GPUS_PER_NODE=2
DEFAULT_MASTER_ADDR="127.0.0.1"
DEFAULT_MASTER_PORT=36001

set -ex

# 刷新动态库缓存
sudo ldconfig
# 验证是否识别
ldconfig -p | grep libcuda.so

echo "START-=-="
export NODE_RANK=${NODE_RANK:-$DEFAULT_NODE_RANK}
export NNODES=${NODE_COUNT:-$DEFAULT_NNODES}
export NPROC_PER_NODE=${PROC_PER_NODE:-$DEFAULT_GPUS_PER_NODE}
export MASTER_ADDR=${MASTER_ADDR:-$DEFAULT_MASTER_ADDR}
export MASTER_PORT=${MASTER_PORT:-$DEFAULT_MASTER_PORT}
export WORLD_SIZE=$((NPROC_PER_NODE * NNODES))
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
export CUDA_LAUNCH_BLOCKING=0

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

nvidia-smi

torchrun \
    --nnodes=${NNODES} \
    --nproc_per_node=${NPROC_PER_NODE} \
    --node_rank=${NODE_RANK} \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    train.py \
    --config ${CONFIG_FILE} \
    --launcher "torch" \
    >&1 | tee ${LOG_FILE}
