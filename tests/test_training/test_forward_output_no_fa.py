import multiprocessing as mp
import os
import random
import socket

import numpy as np
import pytest
import torch

import internlm
from internlm.accelerator import get_accelerator
from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from internlm.core.context.parallel_context import Config
from internlm.core.trainer import Trainer
from internlm.data import build_train_loader_with_data_type
from internlm.initialize.launch import args_sanity_check
from internlm.model.losses import FlashGPTLMLoss
from internlm.model.metrics import AccPerplex, SchedulerMetricHook
from internlm.train import (
    initialize_model,
    initialize_optimizer,
    initialize_parallel_communicator,
)
from internlm.utils.common import get_current_device
from internlm.utils.logger import get_logger

logger = get_logger(__file__)
internlm_accelerator = get_accelerator()

TOTAL_STEPS = 1
config = Config(
    dict(
        VOCAB_SIZE=103168,
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
            total_steps=TOTAL_STEPS,
            valid_micro_num=4,
            valid_every=300,
            rampup_batch_size=None,
            diag_outlier_ratio=1.1,
            use_packed_dataset=False,
        ),
        model=dict(
            checkpoint=True,
            num_attention_heads=32,
            embed_split_hidden=True,
            vocab_size=92544,
            embed_grad_scale=1,
            parallel_output=False,
            hidden_size=4096,
            num_layers=32,
            mlp_ratio=8 / 3,
            apply_post_layer_norm=False,
            dtype="torch.bfloat16",
            norm_type="rmsnorm",
            layer_norm_epsilon=1e-5,
            use_flash_attn=False,
            num_chunks=1,
            no_bias=True,
        ),
        model_type="INTERNLM2_PUBLIC",
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
            total_steps=TOTAL_STEPS,
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


def train_check_output(args):
    # init
    rank, world_size, free_port = args
    build_environment(rank, world_size, free_port, config)

    total_steps = gpc.config.data.total_steps

    try:
        share_path = os.environ["share_path"]
    except KeyError:
        assert False, "plese set environment variable 'share_path'"

    batch_path = os.path.join(share_path, "quailty_assurance/7B_no_flash_attention/batch_no_pack.pt")

    # set seed
    seed_all(1024)

    # initialize model
    model = initialize_model()
    _ = initialize_parallel_communicator(model)

    # initialize loss function
    criterion = FlashGPTLMLoss(parallel_output=False, label_smoothing=gpc.config.loss.label_smoothing)

    optimizer, beta2_scheduler, lr_scheduler = initialize_optimizer(model=model)

    _, dataset_types = build_train_loader_with_data_type()

    metric = AccPerplex(
        device=get_current_device(),
        tp_pg=gpc.get_group(ParallelMode.TENSOR),
        dp_pg=gpc.get_group(ParallelMode.DATA),
        dataset_types=dataset_types,
    )

    scheduler_hooks = [
        SchedulerMetricHook(
            metric=metric,
            skip=(
                gpc.is_using_parallel_mode(ParallelMode.PIPELINE)
                and hasattr(gpc.config.model, "num_chunks")
                and gpc.config.model.num_chunks > 1
                and gpc.config.parallel["pipeline"].get("interleaved_overlap", False)
            ),
        ),
    ]

    engine, scheduler = internlm.initialize_trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        lr_scheduler=lr_scheduler,
        beta2_scheduler=beta2_scheduler,
        scheduler_hooks=scheduler_hooks,
    )
    trainer = Trainer(engine, scheduler)

    # transfer the train data loader into train data iterator
    trainer.train()

    for _ in range(total_steps):
        batch = torch.load(batch_path)
        if batch[0].get("type_ids", None) is not None:
            metric.set_current_type_ids(type_ids=batch[0].pop("type_ids", None))
        # zero the grads of parameters
        output, _, _ = trainer.execute_schedule(
            batch,
            forward_only=True,
            return_loss=True,
            return_output_label=True,
        )

    if gpc.is_rank_for_log():
        standard_output_with_fa = torch.load(
            os.path.join(share_path, "quailty_assurance/7B_no_flash_attention/output_with_fa_internlm2.pt")
        )
        tensor1 = standard_output_with_fa[0][0]
        tensor2 = output[0][0][0]

        if torch.equal(tensor1, tensor2):
            logger.info("Outputs are totally equal")
        else:
            logger.warning("Outputs are not totally equal")
            print(f"tensor1: {tensor1}, tensor2: {tensor2}", flush=True)
            max_diff, index_max_diff = (tensor1 - tensor2).abs().max(dim=0)
            max_diff = max_diff.item()
            index_max_diff = index_max_diff.item()
            rtol = max_diff / abs(tensor2[index_max_diff])
            logger.info(
                f"The relative error is {rtol}. Between {tensor1[index_max_diff]} and {tensor2[index_max_diff]}"
            )
            assert False, f"The relative error is {rtol}"


def test_output():
    free_port = find_free_port()
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=8) as pool:
        pool.map(
            train_check_output,
            [[rank, 8, free_port] for rank in range(8)],
        )
        pool.close()
        pool.join()


if __name__ == "__main__":
    pytest.main(["-s", "-q", "test_forward_output_no_fa.py"])
