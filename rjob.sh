rjob delete rjob-demo
rjob submit --name=rjob-demo --gpu=8 --memory=1600000 --cpu=128 \
--charged-group=sys_gpu --private-machine=group \
--mount=gpfs://gpfs1/ailab-sys/lusitian:/mnt/shared-storage-user/ailab-sys/lusitian \
--image=registry.h.pjlab.org.cn/ailab-puyu/pytorch:23.03-py3-ssh1 \
-P 1 \
--host-network=true \
-e DISTRIBUTED_JOB=true \
-- bash -exc /mnt/shared-storage-user/ailab-sys/lusitian/test/run.sh
