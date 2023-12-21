SEQ_LEN = 8096
MLP_RATIO = 8 / 3
MICRO_NUM=2
MICRO_BATCH_SIZE=1
LEARNING_RATE = 1e-5
MIN_LR = 6e-6
TOTAL_STEP=50
INITIAL_SCALE=2**14
INIT_MODEL="/mnt/petrelfs/wangguoteng.p/InternLM/openxlab/internlm-7b"
DTYPE="torch.float16"
USE_FA=True

adam={
    "lr": LEARNING_RATE,
    "adam_beta1": 0.9,
    "adam_beta2": 0.95,
    "adam_beta2_c": 0,
    "adam_eps": 1e-08,
    "weight_decay": 0.01
}
lr_scheduler={
    "total_steps": TOTAL_STEP,
    "init_steps": 0,
    "warmup_ratio": 0.025,
    "eta_min": MIN_LR,
    "last_epoch": -1
}
beta2_scheduler={
    "init_beta2":adam["adam_beta2"],
    "c": adam["adam_beta2_c"],
    "cur_iter": -1
}
grad_scaler={
    "fp16": {
        "initial_scale": INITIAL_SCALE,
        "min_scale": 1,
        "growth_interval": 1000
    },
    "growth_factor": 2,
    "backoff_factor": 0.5,
    "max_scale": None,
    "hysteresis": 2
}


data={
    "seq_len": SEQ_LEN,
    "micro_num": MICRO_NUM,
    "micro_bsz": MICRO_BATCH_SIZE,
    "valid_micro_num": 4,
    "valid_every": 500,
    "pack_sample_into_one": False,
    "total_steps": TOTAL_STEP,
    "skip_batches": "",
    "rampup_batch_size": "",
    "min_length": 50,
    "train_folder": "/mnt/petrelfs/share_data/wangguoteng.p/dolly_tokenizer_v7/train/",
    "valid_folder": None,
    "empty_cache_and_diag_interval": 500,
    "diag_outlier_ratio": 1.1
}
hybrid_zero_optimizer={
    "overlap_sync_grad": True,
    "overlap_sync_param": False,
    "reduce_bucket_size": 536870912,
    "clip_grad_norm": 1.0
}
loss={
    "label_smoothing": 0
}
model={
    "checkpoint": True,
    "num_attention_heads": 32,
    "embed_split_hidden": True,
    "vocab_size": 103168,
    "embed_grad_scale": 1,
    "parallel_output": True,
    "hidden_size": 4096,
    "num_layers": 32,
    "mlp_ratio": MLP_RATIO,
    "apply_post_layer_norm": False,
    "dtype": f"{DTYPE}",
    "norm_type": "rmsnorm",
    "layer_norm_epsilon": 1e-06,
    "use_flash_attn": USE_FA,
    "num_chunks": 1
}
parallel={
    "zero1": {
        "size": -1,
        "fsdp": False
    },
    "tensor": 1,
    "pipeline": {
        "size": 1,
        "interleaved_overlap": True
    },
    "sequence_parallel": False
}
ckpt={
    "enable_save_ckpt": False,
    "save_ckpt_folder": None,
    "load_ckpt_folder": None,
    "load_ckpt_info": {
        "path": INIT_MODEL,
        "content": [
            "model"
        ],
        "ckpt_type": "internlm"
    },
    "auto_resume": False,
    "checkpoint_every": 50,
    "async_upload": True,
    "async_upload_tmp_folder": "/dev/shm/internlm_tmp_ckpt/",
    "oss_snapshot_freq": 25
}
monitor={
    "alert": {
        "enable_feishu_alert": False,
        "feishu_alert_address": None,
        "light_monitor_address": None
    }
}
JOB_NAME="hw_7B_dolly_sft"

