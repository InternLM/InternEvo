srun -p llm_s --async -o /mnt/petrelfs/lusitian/workspace/InternEvo-fork/log_output/20B_16k_32g/03051-20B-ckpt-fa-Dweb-16k-8144-z8-G32-S50.out \
-N 4 -n 32 --ntasks-per-node=8 --gpus-per-task=1 python train.py --config ./configs/20B_internlm2.py --profiling

# 19 51 25 20B fa 16k
# 19 55 35 7B CKPT SC FA 16K
# 20 15 25 7B ckpt scoffload 16k
# 20 26 35 7B cpuoffload 16k

# 11 35 ckpt0.5
