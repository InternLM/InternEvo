#!/bin/bash


IMAGE="pjlab-shanghai-acr-registry-vpc.cn-shanghai.cr.aliyuncs.com/pjlab-eflops/xiongyingtong:xiongyingtong-lightrlhf-0320"

DLC_CONFIG="demo_dlc.config"

# gpu numbers
GPU_NUMS=16

# job name
JOB_NAME="demo"

# your cmd
# 可以构建任务shell 脚本，然后
# DLC_CMD="bash your_shell.sh"
        # export https_proxy=https://xiongyingtong:gVWjgcX7uXSoVDUBWJhFBsslRlER668B2boU7lHJxVMwYzL6Uxo5ljfaNJ4J@aliyun-proxy.pjlab.org.cn:13128 && \
        # export http_proxy=https://xiongyingtong:gVWjgcX7uXSoVDUBWJhFBsslRlER668B2boU7lHJxVMwYzL6Uxo5ljfaNJ4J@aliyun-proxy.pjlab.org.cn:13128 && \
DLC_CMD="bash /cpfs01/user/xiongyingtong/InternEvo/ali_script/run.sh" #"pwd && bash LightRLHF/examples/ali_scripts/dlc_bash.sh"

# 优先级， 1-4，4最高
PRIORITY=1

PARTITION="llm_s"
WORKSPACE_ID="wsbuzbigeh1hjmst"
DLC_PATH="/cpfs01/shared/public/dlc"



function do_dsw() {
    echo "do_dsw (only support job whose worldsize % 8 == 0 or worldsize < 8)"
    worker_cpu_total=180
    worker_mem_total=1800

    if [[ $GPU_NUMS -lt 8 ]]; then
        num_nodes=1
        num_tasks_per_node=${GPU_NUMS}
        let node_mems=${worker_mem_total}*GPU_NUMS/8
        let cpu_nums=${worker_cpu_total}*GPU_NUMS/8
        shared_memory="10Gi"
        worker_gpu=${GPU_NUMS}
    else
        let num_nodes=GPU_NUMS/8
        num_tasks_per_node=8
        node_mems=${worker_mem_total}
        cpu_nums=${worker_cpu_total}
        shared_memory="200Gi"
        worker_gpu=8
    fi


    ${DLC_PATH} create job --config ${DLC_CONFIG} \
--kind PyTorchJob \
--name ${JOB_NAME} \
--priority $PRIORITY \
--worker_count $num_nodes \
--worker_cpu $cpu_nums \
--worker_gpu $worker_gpu \
--worker_memory "${node_mems}Gi" \
--worker_image ${IMAGE} \
--workspace_id ${WORKSPACE_ID} \
--worker_shared_memory ${shared_memory} \
--command "${DLC_CMD}"

}

do_dsw