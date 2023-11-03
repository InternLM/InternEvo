"""
基于internlm实现的llama2 70B模型的config，下面的配置可以在32张80GB-A800上运行
为了方便加载原始llama2权重，并行配置做了稍许修改，pp：2->4, tp:4->8
srun -p llm_s -n32 -N4 --ntasks-per-node=8 --gpus-per-task=1 python train.py --config configs/70B_sft_llama2.py 
"""
cudnn_deterministic = False
cudnn_benchmark = False

JOB_NAME = "70B_llama"
model_type = "LLAMA"
DO_ALERT = False

SEQ_LEN = 8192
HIDDEN_SIZE = 8192
NUM_ATTENTION_HEAD = 64
NUM_KV_ATTENTION_HEAD = 8
MLP_RATIO = 3.5
NUM_LAYER = 80
VOCAB_SIZE = 32000

LEARNING_RATE = 2e-5
MIN_LR = 6e-6
OPTIMIZER_WARMUP_STEP = 0

# llama2的初始化模型权重路径
LOAD_CKPT_INFO = dict(
    path="local:/mnt/petrelfs/share_data/llm_llama/llama2_raw/llama-2-70b/", 
    content=("model",), 
    ckpt_type="LLAMA",
)

# Ckpt folder format:
# fs: 'local:/mnt/nfs/XXX'
# SAVE_CKPT_FOLDER = "local:llm_ckpts"
SAVE_CKPT_FOLDER = None # ckp保存路径

TRAIN_FOLDER = "dolly_tokenizer_llama/train" # 数据集路径，如果为None则会使用dummy data训练
VALID_FOLDER = "dolly_tokenizer_llama/valid"

# boto3 Ckpt folder format:
# import os
# BOTO3_IP = os.environ["BOTO3_IP"] # boto3 bucket endpoint
# SAVE_CKPT_FOLDER = f"boto3:s3://model_weights.{BOTO3_IP}/internlm"
# LOAD_CKPT_FOLDER = f"boto3:s3://model_weights.{BOTO3_IP}/internlm/snapshot/1/"
CHECKPOINT_EVERY = 5000000

ckpt = dict(
    enable_save_ckpt=False,  # enable ckpt save.
    save_ckpt_folder=SAVE_CKPT_FOLDER,  # Path to save training ckpt.
    # 'load_ckpt_info' setting guide:
    # 1. the 'path' indicate ckpt path,
    # 2. the 'content‘ means what states will be loaded, support: "model", "sampler", "optimizer", "scheduler", "all"
    # 3. the ’ckpt_type‘ means the type of checkpoint to be loaded, now only 'normal' type is supported.
    load_ckpt_info=LOAD_CKPT_INFO,
    # 'auto_resume' is designed to automatically load the latest checkpoint from 'save_ckpt_folder' when encountering
    # training interruptions/hangs caused by hardware failures, using a scheduling system (such as k8s/slurm)
    # with an automatic restart mechanism upon training reboot.
    # Please be aware that if `auto_resume` is not set (its default value is True), it will not load the checkpoint
    # path specified in `load_ckpt_info` by default.
    # If you want to initialize your model weights from another model, you must set `auto_resume` to False.
    # If you want to train from scratch, please set `auto_resume` to False and 'load_ckpt_info' to None.
    auto_resume=False,
    checkpoint_every=CHECKPOINT_EVERY,
    async_upload=True,  # async ckpt upload. (only work for boto3 ckpt)
    async_upload_tmp_folder="/dev/shm/internlm_tmp_ckpt/",  # path for temporarily files during asynchronous upload.
    oss_snapshot_freq=int(CHECKPOINT_EVERY / 2),  # snapshot ckpt save frequency.
)

data = dict(
    seq_len=SEQ_LEN,
    # micro_num means the number of micro_batch contained in one gradient update
    micro_num=8,
    # packed_length = micro_bsz * SEQ_LEN
    micro_bsz=1,
    # defaults to the value of micro_num
    valid_micro_num=8,
    # defaults to 0, means disable evaluate
    valid_every=50,
    pack_sample_into_one=False,
    total_steps=50000,
    skip_batches="",
    rampup_batch_size="",
    # Datasets with less than 50 rows will be discarded
    min_length=50,
    train_folder=TRAIN_FOLDER,
    valid_folder=VALID_FOLDER,
    empty_cache_and_diag_interval=10,
    diag_outlier_ratio=1.1,
)

grad_scaler = dict(
    fp16=dict(
        # the initial loss scale, defaults to 2**16
        initial_scale=2**14,
        # the minimum loss scale, defaults to None
        min_scale=1,
        # the number of steps to increase loss scale when no overflow occurs
        growth_interval=1000,
    ),
    # the multiplication factor for increasing loss scale, defaults to 2
    growth_factor=2,
    # the multiplication factor for decreasing loss scale, defaults to 0.5
    backoff_factor=0.5,
    # the maximum loss scale, defaults to None
    max_scale=2**24,
    # the number of overflows before decreasing loss scale, defaults to 2
    hysteresis=2,
)

hybrid_zero_optimizer = dict(
    # Enable low_level_optimzer overlap_communication
    overlap_sync_grad=True, # overlap 梯度的同步
    overlap_sync_param=False, # overlap 参数的同步，默认不开
    # bucket size for nccl communication params
    reduce_bucket_size=512 * 1024 * 1024,
    # grad clipping
    clip_grad_norm=1.0,
)

loss = dict(
    label_smoothing=0,
)

adam = dict(
    lr=LEARNING_RATE,
    adam_beta1=0.9,
    adam_beta2=0.95,
    adam_beta2_c=0,
    adam_eps=1e-8,
    weight_decay=0.01,
)

lr_scheduler = dict(
    total_steps=data["total_steps"],
    init_steps=OPTIMIZER_WARMUP_STEP,  # optimizer_warmup_step
    warmup_ratio=0.01,
    eta_min=MIN_LR,
    last_epoch=-1,
)

beta2_scheduler = dict(
    init_beta2=adam["adam_beta2"],
    c=adam["adam_beta2_c"],
    cur_iter=-1,
)

model = dict(
    checkpoint=True,   # 是否开启重计算，可以为一个0-1之间的小数表示开启重计算layer的比例
    num_chunks=1,
    num_attention_heads=NUM_ATTENTION_HEAD,
    embed_split_hidden=True,
    vocab_size=VOCAB_SIZE,
    embed_grad_scale=1,  # 清华的embedding放缩技巧，如果为1的话，不放缩
    parallel_output=True,  # 最后的输出是否需要gather起来，如果不gather的话，每个tensor parallel获取的就是自己对应的结果
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYER,
    no_bias=True,
    mlp_ratio=MLP_RATIO,
    apply_post_layer_norm=False,
    dtype="torch.bfloat16",
    norm_type="rmsnorm",
    layer_norm_epsilon=1e-5,
    num_kv_attention_heads=NUM_KV_ATTENTION_HEAD,
)


# 并行设置可以调整，这里仅仅是提供参考：
parallel = dict(
    # internlm目前只支持zero1及其变种
    # zero=-1为标准zero1,
    # zero=1表示关闭zero,
    # zero可以等于[1, dp_worldsize]之间的整数，表示在多少个dp rank之间切分OS
    zero1=dict(size=-1, fsdp=False),
    tensor=4,
    pipeline=dict(size=8, interleaved_overlap=True),
    sequence_parallel=False,
)

monitor = dict(
    # feishu alert configs
    alert=dict(
        enable_feishu_alert=DO_ALERT,
        feishu_alert_address=None,  # feishu webhook to send alert message
        light_monitor_address=None,  # light_monitor address to send heartbeat
    ),
)
