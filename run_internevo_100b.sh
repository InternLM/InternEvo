source /usr/local/Ascend/ascend-toolkit/set_env.sh

export INTERNLM_ACCELERATOR=npu
export HCCL_IF_BASE_PORT=30000
export HCCL_CONNECT_TIMEOUT=1200
export HCCL_INTRA_ROCE_ENABLE=1
export HCCL_INTRA_PCIE_ENABLE=0
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest

# # use ditorch
# export PYTHONPATH=/pjlab_data/code/ditorch:/pjlab_data/code/DeepLinkExt/:$PYTHONPATH
# export INTERNLM_ACCELERATOR=ditorch
# export DEEPLINK_EXT_PLATFORM_TYPE=torch_npu
# export DITORCH_SHOW_DEVICE_AS_CUDA=0

cd /pjlab_data/code/InternEvo

echo "MASTER_ADDR: ${MASTER_ADDR}"
echo "MASTER_PORT: ${MASTER_PORT}"
echo "NNODES: ${WORLD_SIZE}"
echo "RANK: ${RANK}"
torchrun --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --nproc_per_node=8 --nnodes=$WORLD_SIZE --node_rank=$RANK train.py --config configs/100b.py --launcher torch 2>&1 | tee /pjlab_data/logs/internevo_100b_train_log_$RANK

