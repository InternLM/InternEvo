import os
import random
import socket

import numpy as np
import torch

import internlm
from internlm.accelerator import get_accelerator
from internlm.core.context import global_context as gpc
from internlm.core.context.parallel_context import Config
from internlm.data.utils import unpack_type_ids
from internlm.initialize.launch import args_sanity_check

internlm_accelerator = get_accelerator()

config_7B = Config(
    dict(
        parallel=dict(
            zero1=dict(size=-1),
            tensor=dict(size=1, mode="mtp"),
            pipeline=dict(size=1, interleaved_overlap=True),
            weight=dict(size=1, overlap=True),
        ),
        data=dict(
            type="tokenized",
            seq_len=2048,
            micro_num=4,
            micro_bsz=2,
            pack_sample_into_one=False,
            min_length=50,
            total_steps=10,
            valid_micro_num=4,
            valid_every=300,
            rampup_batch_size=None,
            diag_outlier_ratio=1.1,
            train_folder=None,
            valid_folder=None,
            num_worker=0,
        ),
        model=dict(
            checkpoint=False,
            num_attention_heads=32,
            embed_split_hidden=True,
            vocab_size=103168,
            embed_grad_scale=1,
            parallel_output=True,
            hidden_size=4096,
            num_layers=32,
            mlp_ratio=8 / 3,
            apply_post_layer_norm=False,
            dtype="torch.bfloat16",
            norm_type="rmsnorm",
            layer_norm_epsilon=1e-5,
            use_flash_attn=True,
            num_chunks=1,
        ),
        model_type="INTERNLM",
        alert_address=None,
        monitor=dict(alert=dict(enable_feishu_alert=False, feishu_alert_address=None, light_monitor_address=None)),
        grad_scaler=dict(
            fp16=dict(
                initial_scale=2**16,
                min_scale=1,
                growth_interval=1000,
            ),
            growth_factor=2,
            backoff_factor=0.5,
            max_scale=2**24,
            hysteresis=2,
        ),
        adam=dict(
            lr=1e-4,
            adam_beta1=0.9,
            adam_beta2=0.95,
            adam_beta2_c=0,
            adam_eps=1e-8,
            weight_decay=0.01,
        ),
        hybrid_zero_optimizer=dict(
            overlap_sync_grad=True,
            overlap_sync_param=False,
            reduce_bucket_size=512 * 1024 * 1024,
            clip_grad_norm=1.0,
        ),
        beta2_scheduler=dict(
            init_beta2=0.95,
            c=0,
            cur_iter=-1,
        ),
        lr_scheduler=dict(
            total_steps=10,
            init_steps=0,
            warmup_ratio=0.01,
            eta_min=1e-5,
            last_epoch=-1,
        ),
        ckpt=dict(
            enable_save_ckpt=False,
            auto_resume=False,
        ),
        loss=dict(
            label_smoothing=0,
        ),
    )
)


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def build_environment(rank, world_size, free_port, config):
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(free_port)
    internlm_accelerator.empty_cache()
    # launcher="torch"
    internlm.launch_from_torch(config=config, seed=1024)
    args_sanity_check()


def seed_all(seed, cuda_deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if internlm_accelerator.is_available():
        internlm_accelerator.manual_seed(seed)
        internlm_accelerator.manual_seed_all(seed)
    if cuda_deterministic:  # slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def load_new_batch(train_dl, train_iter):
    try:
        batch = next(train_iter)
    except StopIteration:
        train_iter = iter(train_dl)
        batch = next(train_iter)

    if batch[0].get("type_ids", None) is not None:
        # if use_flash_attn is False, we need to unpack type_ids
        if not gpc.config.model.use_flash_attn:
            batch[0]["type_ids"] = unpack_type_ids(batch[0]["type_ids"], batch[0]["cu_seqlens"])

    return batch, train_iter
