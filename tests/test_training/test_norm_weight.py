import gc
import multiprocessing as mp
import os

import pytest
import torch

import internlm
from internlm.accelerator import get_accelerator
from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from internlm.core.trainer import Trainer
from internlm.data import build_train_loader_with_data_type
from internlm.model.losses import FlashGPTLMLoss
from internlm.model.metrics import AccPerplex
from internlm.train import (
    get_scheduler_hooks,
    initialize_model,
    initialize_optimizer,
    initialize_parallel_communicator,
)
from internlm.utils.common import get_current_device
from internlm.utils.logger import get_logger
from tests.common_fixture import (
    build_environment,
    config_7B,
    find_free_port,
    load_new_batch,
    seed_all,
)

logger = get_logger(__file__)
config = config_7B
internlm_accelerator = get_accelerator()


def compute_rotol(tensor1, tensor2):
    torch.set_printoptions(precision=10)
    max_diff, index_max_diff = (tensor1 - tensor2).abs().max(dim=0)
    max_diff = max_diff.item()
    index_max_diff = index_max_diff.item()
    rtol = max_diff / abs(tensor2[index_max_diff])
    logger.info(
        f"The max diff between two tensors is {max_diff}, which is the diff between element "
        f"{tensor1[index_max_diff]} and {tensor2[index_max_diff]}. The relative diff is {rtol}."
    )


def check_norm_pos(name, norm_list):
    for i in range(7):
        for j in range(len(norm_list[i])):
            if not torch.equal(norm_list[i][j], norm_list[i + 1][j]):
                compute_rotol(norm_list[i][j], norm_list[i + 1][j])
                assert False, f"The {name} weights of block between different ranks are not equal."


def train_check_norm_weight(args):
    # init
    rank, world_size, free_port, sp = args
    total_steps = 2000
    share_data_path = os.environ["share_data_path"]
    config.data.total_steps = total_steps
    config.lr_scheduler.total_steps = total_steps
    config.parallel.tensor = dict(size=2, mode=f"{sp}")
    if sp == "isp":
        config.parallel.weight = dict(size=4, overlap=True, memory_pool=True)
    config.data.train_folder = os.path.join(share_data_path, "quality_assurance/0715_data/train")

    build_environment(rank, world_size, free_port, config)

    # set seed
    seed_all(1024)

    # initialize model
    model = initialize_model()

    # initialize isp communicator
    isp_communicator = initialize_parallel_communicator(model)

    # initialize loss function
    criterion = FlashGPTLMLoss(parallel_output=True, label_smoothing=gpc.config.loss.label_smoothing)

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
        if gpc.is_rank_for_log() and batch_count % 100 == 0:
            print(f"batch_count: {batch_count}", flush=True)
        if batch_count % 100 == 0:
            internlm_accelerator.empty_cache()
            gc.collect()

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

        internlm_accelerator.reset_peak_memory_stats()

    blocks_norm1_list = []
    blocks_norm2_list = []

    for block in model.model.blocks:
        blocks_norm1_list.append(block.norm1.weight.detach().to("cpu"))
        blocks_norm2_list.append(block.norm2.weight.detach().to("cpu"))
    if hasattr(model.model, "norm"):
        model_norm = model.model.norm.weight.detach().to("cpu")
    else:
        model_norm = None

    return blocks_norm1_list, blocks_norm2_list, model_norm


def check_result(result):
    norm1_ranks = []
    norm2_ranks = []
    model_norm_ranks = []
    for rank in range(8):
        norm1_ranks.append(result[rank][0])
        norm2_ranks.append(result[rank][1])
        if result[rank][2] is not None:
            model_norm_ranks.append(result[rank][2])

    check_norm_pos("norm1", norm1_ranks)
    check_norm_pos("norm2", norm2_ranks)
    for i in range(len(model_norm_ranks) - 1):
        if not torch.equal(model_norm_ranks[i], model_norm_ranks[i + 1]):
            compute_rotol(model_norm_ranks[i], model_norm_ranks[i + 1])
            assert False, "The norm weights of model between different ranks are not equal."


@pytest.mark.check_norm_msp
def test_check_norm_msp():
    free_port = find_free_port()
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=8) as pool:
        result = pool.map(
            train_check_norm_weight,
            [[rank, 8, free_port, "msp"] for rank in range(8)],
        )
        pool.close()
        pool.join()

    check_result(result)
    print("msp check pass", flush=True)


@pytest.mark.check_norm_fsp
def test_check_norm_fsp():
    free_port = find_free_port()
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=8) as pool:
        result = pool.map(
            train_check_norm_weight,
            [[rank, 8, free_port, "fsp"] for rank in range(8)],
        )
        pool.close()
        pool.join()

    check_result(result)
    print("fsp check pass", flush=True)


@pytest.mark.check_norm_isp
def test_check_norm_isp():
    free_port = find_free_port()
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=8) as pool:
        result = pool.map(
            train_check_norm_weight,
            [[rank, 8, free_port, "isp"] for rank in range(8)],
        )
        pool.close()
        pool.join()

    check_result(result)
    print("isp check pass", flush=True)
