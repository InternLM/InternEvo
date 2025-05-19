import math
import os
from functools import reduce

import pytest
import torch
import torch.distributed as dist

from internlm.accelerator import AcceleratorType, get_accelerator
from internlm.checkpoint import CheckpointManager
from internlm.checkpoint.load_funcs import LOAD_FUNC_DICT
from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from internlm.core.parallel.shard import partition_uniform
from internlm.core.trainer import (
    Trainer,
    TrainState,
    get_scheduler_hooks,
    load_new_batch,
)
from internlm.data import build_train_loader_with_data_type
from internlm.initialize import initialize_launcher, initialize_trainer
from internlm.initialize.initialize_model import (
    initialize_model_and_parallel_communicator,
)
from internlm.initialize.initialize_optimizer import initialize_optimizer
from internlm.model.model_ops.losses import InternLoss
from internlm.model.model_ops.utils import get_parallel_size_from_file
from internlm.utils.common import BatchSkipper, launch_time
from internlm.utils.config import Config
from internlm.utils.gputest import empty_cache_and_diag
from internlm.utils.megatron_timers import megatron_timer as timer
from internlm.utils.storage_manager import get_fns, llm_load

CONFIG_FILE_PATH = os.getenv("CONFIG_FILE_PATH", "./configs/7B_internlm2.py")
INTERNLM2_CKPT_PATH = os.path.join(os.environ["share_path"], "quailty_assurance/test_loss_pri/model_ckpt")
TOTAL_STEPS = 10
LOSS_SPIKE_LIMIT = 1.5
LOSS_DEVIATION_LIMIT = 0.02
# dp_size = 4
BASELINE_LOSS_LIST = [
    12.362918853759766,
    12.404379844665527,
    12.348219871520996,
    12.194982528686523,
    11.80469036102295,
    11.573806762695312,
    10.045475006103516,
    9.660882949829102,
    9.172087669372559,
    4.799427032470703,
]

cur_loss_list = []
internlm_accelerator = get_accelerator()


def load_internlm2_with_dynamic_parallel_size(folder, model):
    """Load InternLM2 with dynamic parallel size."""
    assert folder is not None, "Please specify the folder of the pretrained model"
    assert gpc.config.model_type in ["INTERNLM2"], "dynamic_parallel is only for INTERNLM2"

    fns = get_fns(folder)
    model_fns, old_tp, old_pp = get_parallel_size_from_file(fns)  # pylint: disable=W0612

    tp = gpc.get_world_size(ParallelMode.TENSOR)
    tp_rank = gpc.get_local_rank(ParallelMode.TENSOR)
    assert old_tp % tp == 0 or tp % old_tp == 0, (
        f"Expected TP size in loaded checkpoint to be fit with TP size in current config, but got {old_tp} in "
        f"checkpoint and {tp} in current config"
    )

    correspond_tps = []

    if old_tp <= tp:
        correspond_tps.append(tp_rank // (tp // old_tp))
        ratio = tp // old_tp
        rank = tp_rank % ratio
    else:
        for i in range(old_tp // tp):
            correspond_tps.append(tp_rank * (old_tp // tp) + i)
        rank = 0
        ratio = 1

    current_states = {}

    pp = gpc.get_world_size(ParallelMode.PIPELINE)  # noqa: F841 # pylint: disable=W0612

    assert gpc.config.model.num_chunks == 1, "May cause future collisions, ignore this if necessary"

    old_pp_partition = partition_uniform(gpc.config.model.num_layers, old_pp, 1)

    for idx, parts in enumerate(old_pp_partition):
        start, end = parts[0]
        if model.last_layer <= start or model.first_layer >= end:
            continue
        tmp_states = {}

        for correspond_tp in correspond_tps:
            model_name = f"model_tp{correspond_tp}_pp{idx}.pt"
            states = llm_load(os.path.join(folder, model_name), map_location="cpu")
            states = {k.replace("model.", ""): v for k, v in states.items()}
            for i in range(start, end):
                if i >= model.last_layer:
                    break
                if i < model.first_layer:
                    continue

                for name in list(states.keys()):
                    if f".{i-start}." in name:
                        to_name = name.replace(f".{i-start}.", f".{i-model.first_layer}.")

                        if gpc.config.model_type == "INTERNLM2":
                            if "norm" in name:
                                tmp_states[to_name] = [states.pop(name)]
                            elif any(x in name for x in ("wo", "w2")):
                                tmp_states[to_name] = tmp_states.get(to_name, [])
                                tmp_states[to_name].append(states.pop(name).chunk(ratio, dim=1)[rank])
                            elif any(x in name for x in ("w1", "w3")):
                                tmp_states[to_name] = tmp_states.get(to_name, [])
                                tmp_states[to_name].append(states.pop(name).chunk(ratio, dim=0)[rank])
                            elif any(x in name for x in ("wqkv",)):
                                tmp_states[to_name] = tmp_states.get(to_name, [])
                                if tp > gpc.config.model.num_kv_attention_heads:
                                    assert old_tp <= gpc.config.model.num_kv_attention_heads, (
                                        f"`old_tp ({old_tp}) => tp ({tp})` is not supported. "
                                        "At least one of `tp` and `old_tp` should be less than or "
                                        "equal to `num_kv_attention_heads`"
                                    )
                                    # Suitable for cases where the num_kv_attention_head is small,
                                    # but you want to have a large TP Size
                                    q_per_kv = (
                                        gpc.config.model.num_attention_heads // gpc.config.model.num_kv_attention_heads
                                    )
                                    head_dim = gpc.config.model.hidden_size // gpc.config.model.num_attention_heads
                                    index = torch.concat(
                                        (
                                            torch.arange(q_per_kv).chunk(ratio, dim=0)[tp_rank % ratio],
                                            torch.tensor([q_per_kv, q_per_kv + 1]),
                                        )
                                    )
                                    index = index + (q_per_kv + 2) * (tp_rank // ratio)
                                    index = index % (
                                        (q_per_kv + 2) * (gpc.config.model.num_kv_attention_heads / old_tp)
                                    )
                                    index = index * head_dim
                                    index = index.repeat_interleave(head_dim) + torch.arange(head_dim).repeat(
                                        index.shape[0]
                                    )
                                    tmp_states[to_name].append(
                                        torch.index_select(states.pop(name), 0, index.to(torch.int32))
                                    )
                                else:
                                    tmp_states[to_name].append(states.pop(name).chunk(ratio, dim=0)[rank])
                            else:
                                raise KeyError(f"Unknown key {name}.")

                        else:
                            assert False, "unsupported model type"

            if "tok_embeddings.weight" in states and model.first_layer == 0:
                tmp_states["tok_embeddings.weight"] = tmp_states.get("tok_embeddings.weight", [])
                tmp_states["tok_embeddings.weight"].append(states["tok_embeddings.weight"].chunk(ratio, dim=1)[rank])
            if "output.weight" in states and model.last_layer == gpc.config.model.num_layers:
                tmp_states["norm.weight"] = [states["norm.weight"]]
                tmp_states["output.weight"] = tmp_states.get("output.weight", [])
                tmp_states["output.weight"].append(states["output.weight"].chunk(ratio, dim=0)[rank])

            states = {}

        for name in list(tmp_states.keys()):
            data = tmp_states.pop(name)
            if len(data) == 1:
                current_states[name] = data[0]
            else:
                current_states[name] = torch.concat(
                    data, dim=1 if name == "tok_embeddings.weight" or any(x in name for x in ("wo", "w2")) else 0
                )
                # Merge copied kv heads
                if "wqkv" in name and old_tp > gpc.config.model.num_kv_attention_heads:
                    assert (
                        tp <= gpc.config.model.num_kv_attention_heads
                    ), "new_tp should be less than or equal to num_kv_attention_heads"
                    head_dim = gpc.config.model.hidden_size // gpc.config.model.num_attention_heads
                    q_per_kv = gpc.config.model.num_attention_heads // gpc.config.model.num_kv_attention_heads
                    copied_times = old_tp // gpc.config.model.num_kv_attention_heads
                    cur_q_per_kv = q_per_kv // copied_times

                    # pylint: disable=all
                    def duplicate_kv_index(i):
                        if i % (cur_q_per_kv + 2) >= cur_q_per_kv:
                            return i
                        else:
                            return -100

                    def unique_kv_index(i):
                        if i // (cur_q_per_kv + 2) == copied_times - 1 or i % (cur_q_per_kv + 2) < cur_q_per_kv:
                            return i
                        else:
                            return -100

                    # pylint: enable=all

                    # Verify
                    duplicate_index = [duplicate_kv_index(i) for i in range((cur_q_per_kv + 2) * copied_times)]
                    duplicate_index = [i for i in duplicate_index if i != -100]
                    duplicate_index = _duplicate_index = torch.tensor(duplicate_index)
                    for i in range(gpc.config.model.num_kv_attention_heads // tp - 1):
                        duplicate_index = torch.concat(
                            (duplicate_index, _duplicate_index + duplicate_index.max() + 1), dim=0
                        )
                    duplicate_kv = []
                    for index in duplicate_index.reshape(-1, copied_times * 2).chunk(copied_times, dim=-1):
                        index = index.reshape(-1) * head_dim
                        index = index.repeat_interleave(head_dim) + torch.arange(head_dim).repeat(index.shape[0])
                        duplicate_kv.append(torch.index_select(current_states[name], 0, index))
                    assert reduce(
                        lambda x, y: x and y,
                        [torch.allclose(duplicate_kv[0], x, atol=1e-5) for x in duplicate_kv[1:]],
                    ), "Copied kv heads are not equal after training!"

                    # Merge
                    unique_index = [unique_kv_index(i) for i in range((cur_q_per_kv + 2) * copied_times)]
                    unique_index = [i for i in unique_index if i != -100]
                    unique_index = _unique_index = torch.tensor(unique_index)
                    for i in range(gpc.config.model.num_kv_attention_heads // tp - 1):
                        unique_index = torch.concat((unique_index, _unique_index + unique_index.max() + 1), dim=0)
                    unique_index = unique_index * head_dim
                    unique_index = unique_index.repeat_interleave(head_dim) + torch.arange(head_dim).repeat(
                        unique_index.shape[0]
                    )
                    current_states[name] = torch.index_select(current_states[name], 0, unique_index)
    missing_keys, unexpected_keys = model.load_state_dict(current_states, strict=False)

    if gpc.get_local_rank(ParallelMode.DATA) == 0:
        pp_rank = 0 if not gpc.is_initialized(ParallelMode.PIPELINE) else gpc.get_local_rank(ParallelMode.PIPELINE)
        print(
            f"Missing keys:{missing_keys}, unexpected keys:{unexpected_keys} in "
            f"tp:{gpc.get_local_rank(ParallelMode.TENSOR)}, pp:{pp_rank}",
            flush=True,
        )


LOAD_FUNC_DICT["internlm2_test"] = load_internlm2_with_dynamic_parallel_size


def train(
    dp_size: int = 1,
    tp_size: int = 1,
    wp_size: int = 1,
    pp_size: int = 1,
    num_chunks: int = 2,
    interleaved: bool = False,
    tp_mode: str = "mtp",
    enable_sp: bool = False,
    save_ckpt: bool = False,
    load_ckpt: bool = False,
    model_type: str = "INTERNLM2",
    optimizer_ver: str = "v1",
    pp_mode: str = "1F1B",
):
    # initialize distributed environment
    config = Config.from_file(CONFIG_FILE_PATH)

    # init setting
    config.data.total_steps = 50000
    config.data.fixed_random_dataset_seqlen = False
    config.data.micro_num = 4
    config.data.micro_bsz = 2
    config.lr_scheduler.total_steps = config.data.total_steps
    config.model_type = model_type
    config.ckpt.load_ckpt_folder = None
    config.ckpt.load_ckpt_info = None
    config.ckpt.auto_resume = False
    total_steps = TOTAL_STEPS
    skip_batches = config.data.skip_batches
    label_smoothing = config.loss.label_smoothing
    config.parallel.zero1 = dict(size=-1)
    config.parallel.tensor = dict(size=1, mode="mtp")
    config.parallel.pipeline = dict(size=1, interleaved_overlap=True, mode="1f1b")
    config.parallel.weight = dict(size=1, overlap=True)

    if optimizer_ver == "v2":
        config.hybrid_zero_optimizer.use_split_tensor_optim = True
        config.all_gather_size = 512 * 1024 * 1024
        config.model.checkpoint = True

    # update ckpt config
    if model_type == "INTERNLM2" and tp_mode != "isp" and interleaved is False:
        config.ckpt.load_ckpt_info = dict(path=INTERNLM2_CKPT_PATH, content=("model",), ckpt_type="internlm2_test")

    if save_ckpt:
        config.ckpt.enable_save_ckpt = True
        config.ckpt.checkpoint_every = 10
        config.ckpt.save_ckpt_folder = "local:llm_ckpts/"
        config.ckpt.oss_snapshot_freq = 100

    if load_ckpt:
        config.ckpt.load_ckpt_info = dict(path="local:llm_ckpts/10", content=("all",), ckpt_type="internevo")

    # update parallel config
    config.parallel.tensor = dict(size=tp_size, mode=tp_mode)
    if pp_mode == "ZBH1":
        config.hybrid_zero_optimizer.overlap_sync_grad = False

    config.parallel.pipeline = dict(size=pp_size, mode=pp_mode)
    config.parallel.weight = dict(size=wp_size, overlap=True, launch_allgather_before="wo", forward_overlap_per="layer")
    if interleaved is True:
        config.parallel.pipeline = dict(size=pp_size, interleaved_overlap=True, mode=pp_mode)
        config.model.num_chunks = num_chunks

    if "use_packed_dataset" not in config.data:
        config.data.use_packed_dataset = True
    if tp_mode == "isp" and internlm_accelerator.get_accelerator_backend() in [
        AcceleratorType.NPU,
        AcceleratorType.DIPU,
        AcceleratorType.DITORCH,
    ]:
        config.data.use_packed_dataset = False

    if internlm_accelerator.get_accelerator_backend() == AcceleratorType.GPU:
        launcher = "slurm"
    else:
        launcher = "torch"
        config.model.parallel_output = False
        config.model.checkpoint = True

    initialize_launcher(config=config, launcher=launcher)
    assert hasattr(gpc, "config") and gpc.config is not None

    gpc.config.ckpt.need_metadata = False
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

    # get and broadcast current time
    current_time = launch_time()
    objs = [current_time]
    dist.broadcast_object_list(objs, src=0)
    current_time = objs[0]

    # initialize model and isp_communicator
    model, isp_communicator = initialize_model_and_parallel_communicator()

    # initialize loss function
    criterion = InternLoss(parallel_output=gpc.config.model.parallel_output, label_smoothing=label_smoothing)

    # initialize the train data loader
    train_dl, _ = build_train_loader_with_data_type()

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
    metric = None

    # initialize trainer
    engine, scheduler = initialize_trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        lr_scheduler=lr_scheduler,
        beta2_scheduler=beta2_scheduler,
        scheduler_hooks=get_scheduler_hooks(metric, optimizer, isp_communicator),
    )
    trainer = Trainer(engine, scheduler)

    # initialize the batch skipper
    batch_skipper = BatchSkipper(skip_batches)

    trainer.train()

    train_iter = iter(train_dl)

    if model_type == "INTERNLM2":
        data_path = os.path.join(os.environ["share_path"], "quailty_assurance/test_loss/data_batch_4DP")
        data_batch = torch.load(f"{data_path}/{gpc.get_local_rank(ParallelMode.DATA)}_data_batch.pt")

    # start iterating the train data and begin training
    for batch_count in range(train_state.batch_count, total_steps):
        empty_cache_and_diag(batch_count, interval=gpc.config.data.empty_cache_and_diag_interval)
        timer("one-batch").start()

        if model_type == "INTERNLM2":
            if batch_count >= 10:
                batch = data_batch[batch_count - 10]
            else:
                batch = data_batch[batch_count]
        else:
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
            batch[0].pop("type_ids", None)

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
            global cur_loss_list  # pylint: disable=W0602
            cur_loss_list.append((loss.item() - moe_loss.item() if moe_loss is not None else loss.item()))
        timer("fwd-bwd").stop()

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


@pytest.mark.training_4GPU_optimizer_v2
def test_training_loss_with_dp4_optimizer_v2():
    # model training
    train(dp_size=4, optimizer_ver="v2")

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


@pytest.mark.training_8GPU_4DP2TPSP_optimizer_v2
def test_training_loss_with_dp4_tp2_sp_optimizer_v2():
    # model training
    train(dp_size=4, tp_size=2, tp_mode="fsp", enable_sp=True, optimizer_ver="v2")

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


@pytest.mark.training_8GPU_4DP2PP_ZB
def test_training_loss_with_dp4_pp2_zero_bubble():
    # model training
    train(dp_size=4, pp_size=2, pp_mode="ZBH1")

    # print loss value
    print(f"cur_loss_list: {cur_loss_list}", flush=True)

    check_loss_spike()
    check_loss_accuracy()


@pytest.mark.training_8GPU_4DP2PP_optimizer_v2
def test_training_loss_with_dp4_pp2_optimizer_v2():
    # model training
    train(dp_size=4, pp_size=2, optimizer_ver="v2")

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


@pytest.mark.training_16GPU_4DP2TP2PP_MSP_optimizer_v2
def test_training_loss_with_dp4_tp2_pp2_msp_optimizer_v2():
    # model training
    train(dp_size=4, tp_size=2, pp_size=2, tp_mode="msp", optimizer_ver="v2")

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
    global CONFIG_FILE_PATH, BASELINE_LOSS_LIST
    CONFIG_FILE_PATH = "./configs/7B_isp_sft.py"
    BASELINE_LOSS_LIST = [
        12.159960746765137,
        12.22106647491455,
        12.106496810913086,
        11.951896667480469,
        11.644429206848145,
        11.459924697875977,
        10.127229690551758,
        9.795705795288086,
        9.255647659301758,
        5.301709175109863,
    ]

    # model training
    train(dp_size=4, tp_size=2, wp_size=4, tp_mode="isp", enable_sp=True)

    # print loss value
    print(f"cur_loss_list: {cur_loss_list}", flush=True)

    check_loss_spike()
    check_loss_accuracy()


@pytest.mark.training_8GPU_ISP_SAVE_CKPT
def test_training_with_isp_save_ckpt():
    # update config file
    global CONFIG_FILE_PATH
    CONFIG_FILE_PATH = "./configs/7B_isp_sft.py"

    # model training save ckpt
    train(dp_size=4, tp_size=2, wp_size=4, tp_mode="isp", enable_sp=True, save_ckpt=True)


@pytest.mark.training_8GPU_ISP_LOAD_CKPT
def test_training_with_isp_load_ckpt():
    # update config file
    global CONFIG_FILE_PATH
    CONFIG_FILE_PATH = "./configs/7B_isp_sft.py"

    global TOTAL_STEPS
    TOTAL_STEPS = 20

    # model training load ckpt
    train(dp_size=4, tp_size=2, wp_size=4, tp_mode="isp", enable_sp=True, load_ckpt=True)


@pytest.mark.training_llama2
def test_training_llama2():
    # update config file
    global CONFIG_FILE_PATH
    CONFIG_FILE_PATH = "./configs/7B_llama2.py"

    train(dp_size=8, model_type="LLAMA2")
