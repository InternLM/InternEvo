# A+K
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=8666
export WORLD_SIZE=1
export RANK=0

export GPU_NUMS=4
export USER=root
export TZ=UTC-8
export HCCL_IF_BASE_PORT=30000
export HCCL_CONNECT_TIMEOUT=1200
export HCCL_INTRA_ROCE_ENABLE=1
export HCCL_INTRA_PCIE_ENABLE=0


export PYTHONPATH=/pj_data30t/tangzhiyi/ditorch:/pj_data30t/tangzhiyi/DeepLinkExt/:$PYTHONPATH
export INTERNLM_ACCELERATOR=ditorch
export DEEPLINK_EXT_PLATFORM_TYPE=torch_npu

log_file="log_$(date +%Y%m%d_%H%M%S)"

cd /pj_data30t/tangzhiyi/InternEvo

# torchrun --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --nproc_per_node=4 --nnodes=$WORLD_SIZE --node_rank=$RANK train.py --config configs/7b.py --launcher torch --seed 1024

torchrun --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --nproc_per_node=4 --nnodes=$WORLD_SIZE --node_rank=$RANK train.py --config configs/7b.py --launcher torch --seed 1024 2>&1 | tee -a /pj_data30t/logs/7b_internlm2_ckpt10_epoch1000.log