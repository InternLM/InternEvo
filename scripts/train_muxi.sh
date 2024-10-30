#! /bin/bash

# Setting the environment variables
export MACA_PATH=/opt/maca
export MACA_CLANG_PATH=${MACA_PATH}/mxgpu_llvm/bin
export MACA_CLANG=${MACA_PATH}/mxgpu_llvm
export DEVINFO_ROOT=${MACA_PATH}
export CUCC_PATH=${MACA_PATH}/tools/cu-bridge
export CUDA_PATH=${CUCC_PATH}
export PATH=${CUCC_PATH}:${MACA_PATH}/bin:${MACA_CLANG}/bin:/opt/conda/bin:${PATH}
export LD_LIBRARY_PATH=${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib:${LD_LIBRARY_PATH}
export PATH=/opt/maca/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:${PATH}

#source activate intern

# ROOT=/software/home/shenhao/work
# export PYTHONPATH=/share/work/sh/work/software/datasets/src/:$PYTHONPATH

export ISU_FASTMODEL=1 # must be set, otherwise may induce precision error
export USE_TDUMP=OFF # optional, use to control whether generating debug file
export TMEM_LOG=OFF # optional, use to control whether generating debug file
export DEBUG_ITRACE=0 # optional, use to control whether generating debug file

# export MXLOG_LEVEL=err,MCBLAS=info
# export MCBLAS_CUSTOMIZED_CONFIG_PATH="/software/home/shenhao/bench-tool/to/config.yaml"

export MACA_SMALL_PAGESIZE_ENABLE=1
export MALLOC_THRESHOLD=99
export FORCE_ACTIVATE_WAIT=1

export MCCL_MAX_NCHANNELS=16
export MCCL_P2P_LEVEL=SYS
export MCCL_LIMIT_RING_LL_THREADTHRESHOLDS=1

export vmmDefragment=1

# export CUDA_LAUNCH_BLOCKING=True
# export MACA_LAUNCH_BLOCKING=True

export CUDA_DEVICE_MAX_CONNECTIONS=1
export MACA_SMALL_PAGESIZE_ENABLE=1
export SET_DEVICE_NUMA_PREFERRED=1
export MAX_JOBS=20
#export MCCL_RING_ACCURACY=1
#export FORCE_ACTIVE_WAIT=2
#export MCCL_FAST_WRITE_BACK=1
#export MCCL_EARLY_WRITE_BACK=15
#export GLOO_SOCKET_IFNAME=ens1np0
#export MCCL_ENABLE_FC=0
#export MCCL_MAX_NCHANNELS=18
#export MCCL_ALGO=Ring
#export MCCL_P2P_LEVEL=SYS

#export MCCL_IB_HCA=mlx5_0,mlx5_1

BASH_PATH=$(dirname "$0") 
HOST_FILE=$1
NNODES=$2
GPUS_PER_NODE=8
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
#         --config ${BASH_PATH}/../configs/34B_chameleon_2nodes_muxi.py --launcher "torch"
#       "

${LAUNCHER}


