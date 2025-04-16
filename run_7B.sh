srun -p llm_s --async -o /mnt/petrelfs/lusitian/workspace/InternEvo-fork/log_output/7B_16k_16g/0304-cpuoff10-fa-Dweb-16k-8122-z8-G16-S50.out \
-N 2 -n 16 --ntasks-per-node=8 --gpus-per-task=1 python train.py --config ./configs/7B_internlm2.py --profiling


#--async -o /mnt/petrelfs/lusitian/workspace/InternEvo-fork/log_output/7B_16k_16g/0304-ckpt-fa-Dweb-16k-8122-z8-G16-S50.out \


