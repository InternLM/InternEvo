import multiprocessing as mp

import pytest
import torch

import internlm
from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from internlm.model.loss import FlashGPTLMLoss
from internlm.model.metrics import AccPerplex
from internlm.train import (
    get_scheduler_hooks,
    get_train_data_loader,
    initialize_isp_communicator,
    initialize_model,
    initialize_optimizer,
)
from internlm.utils.logger import get_logger
from tests.common_fixture import (
    build_environment,
    config_7B,
    find_free_port,
    load_new_batch,
    seed_all,
)

logger = get_logger(__file__)

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
        config.parallel.weight = dict(size=4, overlap=True, memory_pool=True)

    build_environment(rank, world_size, free_port, config)

    # set seed
    seed_all(1024)

    # initialize model
    model = initialize_model()

    # initialize isp communicator
    isp_communicator = initialize_isp_communicator(model)

    # initialize loss function
    criterion = FlashGPTLMLoss(parallel_output=True, label_smoothing=gpc.config.loss.label_smoothing)

    optimizer, beta2_scheduler, lr_scheduler = initialize_optimizer(model, isp_communicator)

    train_dl, dataset_types = get_train_data_loader(num_worker=0)

    metric = AccPerplex(
        device=torch.cuda.current_device(),
        tp_pg=gpc.get_group(ParallelMode.TENSOR),
        dp_pg=gpc.get_group(ParallelMode.DATA),
        dataset_types=dataset_types,
    )

    trainer, train_dl, _, _ = internlm.initialize_trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_dataloader=train_dl,
        lr_scheduler=lr_scheduler,
        beta2_scheduler=beta2_scheduler,
        scheduler_hooks=get_scheduler_hooks(metric, optimizer, isp_communicator),
    )

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

        if isp_communicator and isp_communicator.enable_memory_pool:
            isp_communicator.memory_pool.reset_lazy_pools()

        trainer.step()
        torch.cuda.reset_peak_memory_stats()


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
