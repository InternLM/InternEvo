data={
    "seq_len": 2048,
    "micro_num": 1,
    "micro_bsz": 4,
    "valid_micro_num": 4,
    "valid_every": 500,
    "pack_sample_into_one": False,
    "total_steps": 1000,
    "skip_batches": "",
    "rampup_batch_size": "",
    "min_length": 50,
    "train_folder": "/mnt/petrelfs/share_data/lijiaxing/wikitext-2-tokenize/train",
    "valid_folder": "/mnt/petrelfs/share_data/lijiaxing/wikitext-2-tokenize/valid",
    "empty_cache_and_diag_interval": 500,
    "diag_outlier_ratio": 1.1
}
grad_scaler={
    "fp16": {
        "initial_scale": 65536,
        "min_scale": 1,
        "growth_interval": 1000
    },
    "growth_factor": 2,
    "backoff_factor": 0.5,
    "max_scale": 16777216,
    "hysteresis": 2
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
adam={
    "lr": 2e-06,
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "adam_beta2_c": 0,
    "adam_eps": 1e-08,
    "weight_decay": 0.01
}
lr_scheduler={
    "total_steps": 1000,
    "init_steps": 0,
    "warmup_ratio": 0.03,
    "eta_min": 1e-05,
    "last_epoch": -1
}
beta2_scheduler={
    "init_beta2": 0.95,
    "c": 0,
    "cur_iter": -1
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
    "mlp_ratio": 2.6666666666666665,
    "apply_post_layer_norm": False,
    "dtype": "torch.float16",
    "norm_type": "rmsnorm",
    "layer_norm_epsilon": 1e-06,
    "use_flash_attn": False,
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
    "enable_save_ckpt": True,
    "save_ckpt_folder": "local:/mnt/petrelfs/wangguoteng.p/InternLM/huawei_ckpt_internlm_7b_20231213",
    "load_ckpt_info": {
        "path": "/mnt/petrelfs/share/lijiaxing/Huawei_init_ckpt",
        "content": [
            "all"
        ],
        "ckpt_type": "internlm"
    },
    "auto_resume": False,
    "checkpoint_every": 100,
    "async_upload": True,
    "async_upload_tmp_folder": "/dev/shm/internlm_tmp_ckpt/",
    "oss_snapshot_freq": 100
}
monitor={
    "alert": {
        "enable_feishu_alert": False,
        "feishu_alert_address": None,
        "light_monitor_address": None
    }
}
JOB_NAME="hw_7B_wiki"
SEQ_LEN=2048
MLP_RATIO=2.6666666666666665

