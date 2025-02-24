import multiprocessing as mp

import pytest

import internlm
from internlm.accelerator import get_accelerator
from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from internlm.core.trainer import Trainer
from internlm.data import build_train_loader_with_data_type
from internlm.model.losses import InternLoss
from internlm.model.metrics import AccPerplex
from internlm.train import (
    get_scheduler_hooks,
    initialize_model,
    initialize_optimizer,
    initialize_parallel_communicator,
)
from internlm.utils.logger import get_logger
from tests.common_fixture import (
    build_environment,
    config_7B,
    find_free_port,
    get_current_device,
    load_new_batch,
    seed_all,
)

logger = get_logger(__file__)
internlm_accelerator = get_accelerator()

# init config
config = config_7B
total_steps = 5
config.data.total_steps = total_steps
config.lr_scheduler.total_steps = total_steps
config.model.use_flash_attn = False
config.parallel.pipeline = dict(size=2, interleaved_overlap=True)


def train_check(args):
    # init
    rank, world_size, free_port, mode, num_chunks = args
    config.model.num_chunks = num_chunks
    config.parallel.tensor = dict(size=2, mode=f"{mode}")
    if mode == "isp":
        config.parallel.weight = dict(size=4, overlap=True)

    build_environment(rank, world_size, free_port, config)

    # set seed
    seed_all(1024)

    # initialize model
    model = initialize_model()

    # initialize isp communicator
    isp_communicator = initialize_parallel_communicator(model)

    # initialize loss function
    criterion = InternLoss(parallel_output=True, label_smoothing=gpc.config.loss.label_smoothing)

    optimizer, beta2_scheduler, lr_scheduler = initialize_optimizer(model, isp_communicator)

    train_dl, dataset_types = build_train_loader_with_data_type()

    metric = AccPerplex(
        device=get_current_device(),
        tp_pg=gpc.get_group(ParallelMode.TENSOR),
        dp_pg=gpc.get_group(ParallelMode.DATA),
        dataset_types=dataset_types,
    )

    engine, scheduler = internlm.initialize_trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        lr_scheduler=lr_scheduler,
        beta2_scheduler=beta2_scheduler,
        scheduler_hooks=get_scheduler_hooks(metric, optimizer, isp_communicator),
    )
    trainer = Trainer(engine, scheduler)

    # transfer the train data loader into train data iterator
    trainer.train()

    train_iter = iter(train_dl)

    for batch_count in range(total_steps):
        if gpc.is_rank_for_log():
            print(f"{mode}: {batch_count}", flush=True)

        # load batch data
        batch, train_iter = load_new_batch(train_dl=train_dl, train_iter=train_iter)

        # zero the grads of parameters
        trainer.zero_grad()

        # process data
        if batch[0].get("type_ids", None) is not None:
            metric.set_current_type_ids(type_ids=batch[0].pop("type_ids", None))

        # zero the grads of parameters
        _, _, _ = trainer.execute_schedule(
            batch,
            forward_only=False,
            return_loss=True,
            return_output_label=False,
        )

        trainer.step()
        internlm_accelerator.reset_peak_memory_stats()


mode_list = ["mtp"]
num_chunks = [1, 2]


@pytest.mark.parametrize("mode", mode_list)
@pytest.mark.parametrize("num_chunks", num_chunks)
def test_train(mode, num_chunks):
    free_port = find_free_port()
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=8) as pool:
        pool.map(
            train_check,
            [[rank, 8, free_port, mode, num_chunks] for rank in range(8)],
        )
        pool.close()
        pool.join()
