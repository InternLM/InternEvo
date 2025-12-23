rjob delete rjob-test-heyu
rjob submit --name=rjob-U-test --gpu=8 --memory=1500000 --cpu=96 \
--charged-group=sys_gpu --private-machine=group \
--mount=gpfs://gpfs1/ailab-sys/lusitian:/mnt/shared-storage-user/ailab-sys/lusitian \
--mount=gpfs://gpfs1/lusitian:/mnt/shared-storage-user/lusitian \
--image=registry.h.pjlab.org.cn/ailab/pytorch:2.7.1-cuda12.8-cudnn9-devel \
-P 2 \
--custom-resources rdma/mlnx_shared=8 \
-e DISTRIBUTED_JOB=true \
--host-network=true \
--enable-sshd \
-- bash -exc "/mnt/shared-storage-user/ailab-sys/lusitian/workspace/InternEvo/rjob_run.sh"

# -P 2 \
# --custom-resources rdma/mlnx_shared=8 \
# -e DISTRIBUTED_JOB=true \
# --image=registry.h.pjlab.org.cn/ailab-sys-sys_gpu/internevo:fa2.6.3-torch2.4-cu118 \
# --image=registry.h.pjlab.org.cn/ailab/xpuyu:0.3.0rc0-pt2.6-cu126-20250415--fix-max-hca-20250422 \
# --mount=gpfs://gpfs1/ailab-sys/matenghui:/mnt/shared-storage-user/ailab-sys/matenghui \
# --positive-tags node/gpu-lg-cmc-h-h200-0836.host.h.pjlab.org.cn \
