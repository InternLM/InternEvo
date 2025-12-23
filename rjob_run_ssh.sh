#!/bin/bash
set -e
set -x
# sudo -i
# source /root/.bashrc
# source /root/miniconda3/bin/activate
# conda activate internevo
# cd /mnt/shared-storage-user/ailab-sys/lusitian/workspace/InternEvo

# # 遍历所有环境变量
# for var in $(env | awk -F= '{print $1}'); do
#     # 检查变量名是否合法
#     if ! [[ $var =~ ^[a-zA-Z_][a-zA-Z0-9_]*$ ]]; then
#         echo "Skipping invalid variable: $var"
#         unset $var
#     fi
# done

# set -u
# export CUDA_HOME=/usr/local/cuda-11.8
# export PATH=$CUDA_HOME/bin:$PATH
# export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 添加动态库路径
# sudo tee /etc/ld.so.conf.d/nvidia.conf <<EOF
# /usr/local/nvidia/lib64
# /usr/local/nvidia/lib
# /usr/local/cuda-11.8/lib64
# EOF
# # 刷新动态库缓存
# sudo ldconfig
# # 验证是否识别
# ldconfig -p | grep libcuda.so
# echo "START-=-="
export MASTER_ADDR=10.102.207.103
export GPUS_PER_NODE=8
export MASTER_PORT=6001
export NNODES=2
export NODE_RANK=0
export WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
export CUDA_LAUNCH_BLOCKING=0
export NCCL_DEBUG=INFO
# export OMP_NUM_THREADS=1

CONFIG_FILE=configs/7B_isp_sft.py
LOG_DIR=./log_record
DATA_NAME=pg19
DATA_SIZE=all
BUCKET_SIZE=B4096
SEQ_LEN=64k_16n
DP_TP_PP=1_4_4
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")  
# LOG_FILE=${LOG_DIR}/${DATA_NAME}_${DATA_SIZE}_${BUCKET_SIZE}_${SEQ_LEN}_${DP_TP_PP}_${TIMESTAMP}.log
LOG_FILE=test.log

nvidia-smi
python -c "import torch; print(torch.__version__);print(torch.version.cuda);print(torch.cuda.is_available()); print(torch.cuda.device_count())"

torchrun \
    --nnodes=${NNODES} \
    --nproc_per_node=${GPUS_PER_NODE} \
    --node_rank=${NODE_RANK} \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    train.py \
    --config ${CONFIG_FILE} \
    --launcher "torch" \
    2>&1 | tee ${LOG_FILE}
