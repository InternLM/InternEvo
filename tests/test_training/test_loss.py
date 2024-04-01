import math
import os

import pytest
import torch.distributed as dist

import internlm
from internlm.checkpoint import CheckpointManager
from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from internlm.core.trainer import TrainState
from internlm.data import build_train_loader_with_data_type
from internlm.initialize import initialize_distributed_env
from internlm.model.losses import FlashGPTLMLoss
from internlm.model.metrics import AccPerplex
from internlm.train import (
    get_scheduler_hooks,
    initialize_isp_communicator,
    initialize_model,
    initialize_optimizer,
    load_new_batch,
)
from internlm.utils.common import BatchSkipper, get_current_device, launch_time
from internlm.utils.gputest import empty_cache_and_diag
from internlm.utils.megatron_timers import megatron_timer as timer

CONFIG_FILE_PATH = os.getenv("CONFIG_FILE_PATH", "./configs/7B_sft.py")
TOTAL_STEPS = 10
LOSS_SPIKE_LIMIT = 1.5
LOSS_DEVIATION_LIMIT = 0.2
# dp_size = 4
BASELINE_LOSS_LIST = [
    11.63298511505127,
    7.826552391052246,
    6.727715969085693,
    6.182102680206299,
    5.395875930786133,
    5.394404411315918,
    5.053983688354492,
    4.741974353790283,
    4.629222869873047,
    4.6164231300354,
]

cur_loss_list = []


def train(
    dp_size: int = 1,
    tp_size: int = 1,
    wp_size: int = 1,
    pp_size: int = 1,
    num_chunks: int = 2,
    interleaved: bool = False,
    tp_mode: str = "mtp",
    enable_sp: bool = False,
    enable_ckpt: bool = False,
    model_type: str = "INTERNLM",
):
    # initialize distributed environment
    initialize_distributed_env(config=CONFIG_FILE_PATH)
    assert hasattr(gpc, "config") and gpc.config is not None

    # check parallel config
    assert (
        gpc.get_world_size(ParallelMode.DATA) == dp_size
    ), f"data parallel size: {gpc.get_world_size(ParallelMode.DATA)} is not as expected {dp_size}"
    assert (
        gpc.get_world_size(ParallelMode.TENSOR) == tp_size
    ), f"tensor parallel size: {gpc.get_world_size(ParallelMode.TENSOR)} is not as expected {tp_size}"
    assert (
        gpc.get_world_size(ParallelMode.WEIGHT) == wp_size
    ), f"weight parallel size: {gpc.get_world_size(ParallelMode.WEIGHT)} is not as expected {wp_size}"
    assert (
        gpc.get_world_size(ParallelMode.PIPELINE) == pp_size
    ), f"pipeline parallel size: {gpc.get_world_size(ParallelMode.PIPELINE)} is not as expected {pp_size}"
    if interleaved:
        assert (
            gpc.is_using_parallel_mode(ParallelMode.PIPELINE)
            and hasattr(gpc.config.model, "num_chunks")
            and gpc.config.model.num_chunks == num_chunks
        )
        assert gpc.config.parallel["pipeline"].get(
            "interleaved_overlap", False
        ), "interleaved overlap must be enabled when using interleave pipeline scheduler"
    if enable_sp:
        assert gpc.config.parallel.get(
            "sequence_parallel", False
        ), "sequence_parallel must be True when enable_sp is True"
    assert gpc.config.parallel["tensor"]["mode"] == tp_mode

    # init setting
    gpc.config.data.total_steps = TOTAL_STEPS
    gpc.config.data.fixed_random_dataset_seqlen = False
    gpc.config.lr_scheduler.total_steps = TOTAL_STEPS
    gpc.config.model_type = model_type
    total_steps = gpc.config.data.total_steps
    skip_batches = gpc.config.data.skip_batches
    label_smoothing = gpc.config.loss.label_smoothing

    # update ckpt config
    if enable_ckpt:
        gpc.config.ckpt.enable_save_ckpt = True
        gpc.config.ckpt.checkpoint_every = 10
        gpc.config.ckpt.save_ckpt_folder = "local:llm_ckpts/"
        gpc.config.ckpt.load_ckpt_folder = "local:llm_ckpts/"
        gpc.config.ckpt.load_ckpt_info["content"] = ("all",)
        gpc.config.ckpt.oss_snapshot_freq = 100

    # get and broadcast current time
    current_time = launch_time()
    objs = [current_time]
    dist.broadcast_object_list(objs, src=0)
    current_time = objs[0]

    # initialize model
    model = initialize_model()

    # initialize isp communicator
    isp_communicator = initialize_isp_communicator(model)

    # initialize loss function
    criterion = FlashGPTLMLoss(parallel_output=True, label_smoothing=label_smoothing)

    # initialize the train data loader
    train_dl, dataset_types = build_train_loader_with_data_type()

    # initialize and resume train state
    train_state = TrainState(gpc.config, train_dl.batch_sampler)

    optimizer, beta2_scheduler, lr_scheduler = initialize_optimizer(model, isp_communicator)

    with open(CONFIG_FILE_PATH, "r") as f:
        config_lines = f.readlines()
    ckpt_manager = CheckpointManager(
        ckpt_config=gpc.config.ckpt,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_dl=train_dl,
        model_config=gpc.config.model,
        model_config_file="".join(config_lines),
        feishu_address=gpc.config.monitor.alert.feishu_alert_address,
    )

    # Loading other persistent training states.
    ckpt_manager.try_resume_training(train_state, current_time)

    # initialize metric for calculating accuracy and perplexity
    metric = AccPerplex(
        device=get_current_device(),
        tp_pg=gpc.get_group(ParallelMode.TENSOR),
        dp_pg=gpc.get_group(ParallelMode.DATA),
        dataset_types=dataset_types,
    )

    # initialize trainer
    trainer, train_dl, _, _ = internlm.initialize_trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_dataloader=train_dl,
        lr_scheduler=lr_scheduler,
        beta2_scheduler=beta2_scheduler,
        scheduler_hooks=get_scheduler_hooks(metric, optimizer, isp_communicator),
    )

    # initialize the batch skipper
    batch_skipper = BatchSkipper(skip_batches)

    trainer.train()

    # transfer the train data loader into train data iterator
    train_iter = iter(train_dl)

    # start iterating the train data and begin training
    for batch_count in range(train_state.batch_count, total_steps):
        empty_cache_and_diag(batch_count, interval=gpc.config.data.empty_cache_and_diag_interval)
        timer("one-batch").start()

        # load batch data
        batch, train_iter = load_new_batch(train_dl=train_dl, train_iter=train_iter, train_state=train_state)

        # record the consumed samples in training
        train_state.batch_count = batch_count
        train_state.num_consumed_samples_in_epoch += len(batch[1])
        if batch_skipper(batch_count):  # skip this batch
            if gpc.is_rank_for_log():
                print(f"Skip batch count:`{batch_count}`...")
            timer("one-batch").stop()
            continue

        # zero the grads of parameters
        trainer.zero_grad()
        # process data
        if batch[0].get("type_ids", None) is not None:
            metric.set_current_type_ids(type_ids=batch[0].pop("type_ids", None))

        # do forward and backward
        timer("fwd-bwd").start()

        # Compatible for non-moe
        moe_loss = None
        if hasattr(gpc.config.model, "num_experts"):
            _, _, loss, moe_loss = trainer.execute_schedule(
                batch, forward_only=False, return_loss=True, return_output_label=False
            )
        else:
            _, _, loss = trainer.execute_schedule(
                batch, forward_only=False, return_loss=True, return_output_label=False
            )
        if gpc.is_rank_for_log():
            assert loss is not None and not math.isnan(loss.item())
            global cur_loss_list
            cur_loss_list.append((loss.item() - moe_loss.item() if moe_loss is not None else loss.item()))
        timer("fwd-bwd").stop()

        if isp_communicator and isp_communicator.enable_memory_pool:
            isp_communicator.memory_pool.reset_lazy_pools()

        # update parameters, and returns (success_update, grad_norm)
        trainer_result = trainer.step()
        assert trainer_result is not None

        success_update, _ = trainer_result
        assert success_update, "Error: grad norm inf or nan occurs!"
        if success_update:  # update parameters successfully
            train_state.step_count += 1
        else:
            train_state.inf_nan_skip_batches += 1  # record the amount of updating parameters unsuccessfully.

        timer("one-batch").stop()

        # checkpoint the training states in specific steps, which is determined by the args "checkpoint_every"
        # # save batch sampler that tracks the true consumed samples
        now_break = ckpt_manager.try_save_checkpoint(train_state)
        if now_break:
            break

    ckpt_manager.wait_async_upload_finish()


def check_loss_spike():
    if gpc.is_rank_for_log():
        for step in range(1, TOTAL_STEPS):
            assert (
                cur_loss_list[step] < cur_loss_list[step - 1] * LOSS_SPIKE_LIMIT
            ), f"The loss spike occurs, {cur_loss_list[step - 1]}->{cur_loss_list[step]}, please check it!"


def check_loss_accuracy():
    if gpc.is_rank_for_log():
        for cur, target in zip(cur_loss_list, BASELINE_LOSS_LIST):
            assert (
                abs(cur - target) < LOSS_DEVIATION_LIMIT
            ), f"The loss accuracy is abnormal, {target}->{cur}, please check it!"


@pytest.mark.training_4GPU
def test_training_loss_with_dp4():
    # model training
    train(dp_size=4)

    # print loss value
    print(f"cur_loss_list: {cur_loss_list}", flush=True)

    check_loss_spike()
    check_loss_accuracy()


@pytest.mark.training_8GPU_4DP2TP
def test_training_loss_with_dp4_tp2():
    # model training
    train(dp_size=4, tp_size=2)

    # print loss value
    print(f"cur_loss_list: {cur_loss_list}", flush=True)

    check_loss_spike()
    check_loss_accuracy()


@pytest.mark.training_8GPU_4DP2TPSP
def test_training_loss_with_dp4_tp2_sp():
    # model training
    train(dp_size=4, tp_size=2, tp_mode="fsp", enable_sp=True)

    # print loss value
    print(f"cur_loss_list: {cur_loss_list}", flush=True)

    check_loss_spike()
    check_loss_accuracy()


@pytest.mark.training_8GPU_4DP2PP
def test_training_loss_with_dp4_pp2():
    # model training
    train(dp_size=4, pp_size=2)

    # print loss value
    print(f"cur_loss_list: {cur_loss_list}", flush=True)

    check_loss_spike()
    check_loss_accuracy()


@pytest.mark.training_8GPU_4DP2PP_InterleavedOverlap
def test_training_loss_with_dp4_pp2_interleaved_overlap():
    # model training
    train(dp_size=4, pp_size=2, interleaved=True)

    # print loss value
    print(f"cur_loss_list: {cur_loss_list}", flush=True)

    check_loss_spike()
    check_loss_accuracy()


@pytest.mark.training_16GPU_4DP2TP2PP_MTP
def test_training_loss_with_dp4_tp2_pp2_mtp():
    # model training
    train(dp_size=4, tp_size=2, pp_size=2)

    # print loss value
    print(f"cur_loss_list: {cur_loss_list}", flush=True)

    check_loss_spike()
    check_loss_accuracy()


@pytest.mark.training_16GPU_4DP2TP2PP_MSP
def test_training_loss_with_dp4_tp2_pp2_msp():
    # model training
    train(dp_size=4, tp_size=2, pp_size=2, tp_mode="msp")

    # print loss value
    print(f"cur_loss_list: {cur_loss_list}", flush=True)

    check_loss_spike()
    check_loss_accuracy()


@pytest.mark.training_16GPU_4DP2TP2PP_FSP
def test_training_loss_with_dp4_tp2_pp2_fsp():
    # model training
    train(dp_size=4, tp_size=2, pp_size=2, tp_mode="fsp")

    # print loss value
    print(f"cur_loss_list: {cur_loss_list}", flush=True)

    check_loss_spike()
    check_loss_accuracy()


@pytest.mark.training_8GPU_ISP
def test_training_with_isp():
    # update config file
    global CONFIG_FILE_PATH
    CONFIG_FILE_PATH = "./configs/7B_isp_sft.py"

    # model training
    train(dp_size=4, tp_size=2, wp_size=4, tp_mode="isp", enable_sp=True)


@pytest.mark.training_8GPU_ISP_SAVE_CKPT
def test_training_with_isp_save_ckpt():
    # update config file
    global CONFIG_FILE_PATH
    CONFIG_FILE_PATH = "./configs/7B_isp_sft.py"

    # model training save ckpt
    train(dp_size=4, tp_size=2, wp_size=4, tp_mode="isp", enable_sp=True, enable_ckpt=True)


@pytest.mark.training_8GPU_ISP_LOAD_CKPT
def test_training_with_isp_load_ckpt():
    # update config file
    global CONFIG_FILE_PATH
    CONFIG_FILE_PATH = "./configs/7B_isp_sft.py"

    global TOTAL_STEPS
    TOTAL_STEPS = 20

    # model training load ckpt
    train(dp_size=4, tp_size=2, wp_size=4, tp_mode="isp", enable_sp=True, enable_ckpt=True)


@pytest.mark.training_llama2
def test_training_llama2():
    # update config file
    global CONFIG_FILE_PATH
    CONFIG_FILE_PATH = "./configs/7B_llama2.py"

    train(dp_size=8, model_type="LLAMA2")


@pytest.mark.training_internlm2
def test_training_internlm2():
    # update config file
    global CONFIG_FILE_PATH
    CONFIG_FILE_PATH = "./configs/7B_internlm2.py"

    train(dp_size=8, model_type="INTERNLM2_PUBLIC")
