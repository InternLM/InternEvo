import multiprocessing as mp

from internlm.accelerator import get_accelerator

backup_ForkingPickler = mp.reduction.ForkingPickler
backup_dump = mp.reduction.dump
import os  # noqa: E402  #pylint: disable=wrong-import-position
import random  # noqa: E402  #pylint: disable=wrong-import-position
import shutil  # noqa: E402  #pylint: disable=wrong-import-position
import socket  # noqa: E402  #pylint: disable=wrong-import-position

import numpy as np  # noqa: E402  #pylint: disable=wrong-import-position
import pytest  # noqa: E402  #pylint: disable=wrong-import-position
import torch  # noqa: E402  #pylint: disable=wrong-import-position
import torch.distributed as dist  # noqa: E402  #pylint: disable=wrong-import-position

import internlm  # noqa: E402  #pylint: disable=wrong-import-position
from internlm.checkpoint import (  # noqa: E402  #pylint: disable=wrong-import-position
    CheckpointManager,
)
from internlm.core.context import (  # noqa: E402  #pylint: disable=wrong-import-position
    ParallelMode,
)
from internlm.core.context import (  # noqa: E402  #pylint: disable=wrong-import-position
    global_context as gpc,
)
from internlm.core.context.parallel_context import (  # noqa: E402  #pylint: disable=wrong-import-position
    Config,
)
from internlm.core.trainer import (  # noqa: E402  #pylint: disable=wrong-import-position
    TrainState,
)
from internlm.data import (  # noqa: E402  #pylint: disable=wrong-import-position
    build_train_loader_with_data_type,
)
from internlm.initialize.launch import (  # noqa: E402  #pylint: disable=wrong-import-position
    args_sanity_check,
)
from internlm.model.losses import (  # noqa: E402  #pylint: disable=wrong-import-position
    FlashGPTLMLoss,
)
from internlm.model.metrics import (  # noqa: E402  #pylint: disable=wrong-import-position
    AccPerplex,
    SchedulerMetricHook,
)
from internlm.train import (  # noqa: E402  #pylint: disable=wrong-import-position
    initialize_model,
    initialize_optimizer,
    load_new_batch,
)
from internlm.utils.common import (  # noqa: E402  #pylint: disable=wrong-import-position
    get_current_device,
    launch_time,
)
from internlm.utils.logger import (  # noqa: E402  #pylint: disable=wrong-import-position
    get_logger,
)

logger = get_logger(__file__)
internlm_accelerator = get_accelerator()

TOTAL_STEPS = 10
temp_folder = "temp_ckpt_for_check_loss"
config = Config(
    dict(
        parallel=dict(
            zero1=dict(size=-1, fsdp=False),
            pipeline=dict(size=1, interleaved_overlap=False),
            sequence_parallel=False,
            tensor=1,
        ),
        data=dict(
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
            train_folder=os.path.join(
                os.environ["share_path"], "quailty_assurance/0623_scratch_tokenized_filtered/train"
            ),
            valid_folder=os.path.join(
                os.environ["share_path"], "quailty_assurance/0623_scratch_tokenized_filtered/val"
            ),
            num_worker=0,
        ),
        model=dict(
            checkpoint=False,
            num_attention_heads=16,
            embed_split_hidden=True,
            vocab_size=103168,
            embed_grad_scale=1,
            parallel_output=True,
            hidden_size=4096,
            num_layers=16,
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
            overlap_sync_param=True,
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
            enable_save_ckpt=True,
            save_ckpt_folder=f"local:{temp_folder}/",
            auto_resume=False,
            checkpoint_every=5,
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


def train_model(args):
    # init
    rank, world_size, train_round, free_port = args
    build_environment(rank, world_size, free_port, config)
    total_steps = 6

    if train_round == 1:
        gpc.config.ckpt.enable_save_ckpt = False
        gpc.config.ckpt._add_item(
            "load_ckpt_info", dict(path=f"local:{temp_folder}/5/", content=("all",), ckpt_type="internevo")
        )
    else:
        assert (
            os.path.exists(temp_folder) is False
        ), f"Error: ckpt temp folder '{temp_folder}' already exists, please check it."

    # set seed
    seed_all(1024)

    # get and broadcast current time
    current_time = launch_time()
    objs = [current_time]
    dist.broadcast_object_list(objs, src=0)
    current_time = objs[0]

    # initialize model
    model = initialize_model()

    # initialize loss function
    criterion = FlashGPTLMLoss(parallel_output=True, label_smoothing=gpc.config.loss.label_smoothing)

    # initialize the train and validation data loader
    train_dl, dataset_types = build_train_loader_with_data_type()

    train_state = TrainState(gpc.config, train_dl.batch_sampler)

    optimizer, beta2_scheduler, lr_scheduler = initialize_optimizer(model=model)

    ckpt_manager = CheckpointManager(
        ckpt_config=gpc.config.ckpt,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_dl=train_dl,
        model_config=gpc.config.model,
        model_config_file=None,
        feishu_address=gpc.config.monitor.alert.feishu_alert_address,
    )

    ckpt_manager.try_resume_training(train_state, current_time)

    # initialize metric for calculating accuracy and perplexity
    metric = AccPerplex(
        device=get_current_device(),
        tp_pg=gpc.get_group(ParallelMode.TENSOR),
        dp_pg=gpc.get_group(ParallelMode.DATA),
        dataset_types=dataset_types,
    )

    # initialize trainer
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

    trainer, train_dl, _, _ = internlm.initialize_trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_dataloader=train_dl,
        lr_scheduler=lr_scheduler,
        beta2_scheduler=beta2_scheduler,
        scheduler_hooks=scheduler_hooks,
    )

    trainer.train()
    train_iter = iter(train_dl)
    # transfer the train data loader into train data iterator
    for batch_count in range(train_state.batch_count, total_steps):
        # load batch data
        batch, train_iter = load_new_batch(train_dl=train_dl, train_iter=train_iter, train_state=train_state)

        train_state.batch_count = batch_count
        train_state.num_consumed_samples_in_epoch += len(batch[1])

        # zero the grads of parameters
        trainer.zero_grad()

        # process data
        if batch[0].get("type_ids", None) is not None:
            metric.set_current_type_ids(type_ids=batch[0].pop("type_ids", None))

        _, _, loss = trainer.execute_schedule(
            batch,
            forward_only=False,
            return_loss=True,
            return_output_label=False,
        )

        # update parameters
        trainer_result = trainer.step()
        assert trainer_result is not None

        success_update, grad_norm_groups = trainer_result
        if success_update:  # update parameters successfully
            train_state.step_count += 1
        else:
            train_state.inf_nan_skip_batches += 1  # record the amount of updating parameters unsuccessfully.
            if -1 in grad_norm_groups.values() and gpc.is_rank_for_log():  # -1 encodes a specific failure case
                logger.warning(f"Warning: skip parameter update at step {batch_count}.")

        ckpt_manager.try_save_checkpoint(train_state)

    ckpt_manager.wait_async_upload_finish()
    internlm_accelerator.empty_cache()
    dist.barrier()

    if gpc.is_rank_for_log():
        if train_round == 1:
            shutil.rmtree(temp_folder)
        return loss.item(), batch


def test_loss():
    mp.reduction.ForkingPickler = backup_ForkingPickler
    mp.reduction.dump = backup_dump
    results = []
    free_port = find_free_port()
    ctx = mp.get_context("spawn")
    for train_round in range(2):
        with ctx.Pool(processes=8) as pool:
            result = pool.map(
                train_model,
                [[rank, 8, train_round, free_port] for rank in range(8)],
            )
            results.append(result)
            pool.close()
            pool.join()
    loss_round_1, loss_round_2 = results[0][0][0], results[1][0][0]
    input_ids_round_1, input_ids_round_2 = results[0][0][1][0]["input_ids"], results[1][0][1][0]["input_ids"]

    assert torch.equal(input_ids_round_1, input_ids_round_2), "Error: data batch is not aligned when loading ckpt"
    assert torch.allclose(
        torch.tensor(loss_round_1), torch.tensor(loss_round_2), rtol=1e-3, atol=1e-3
    ), "Error: ckpt has something wrong, loss is not close."


if __name__ == "__main__":
    pytest.main(["-s", "-q", "test_load_ckpt_loss.py"])
