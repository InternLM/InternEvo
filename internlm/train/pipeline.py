#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import math
import time
from typing import Callable, Iterable, List, Optional, Tuple, TypeVar, Union

import torch
from torch import nn
from torch.utils.data import DataLoader

from internlm.accelerator import AcceleratorType, get_accelerator
from internlm.core.context import (
    IS_REPLICA_ZERO_PARALLEL,
    IS_TENSOR_DATA_PARALLEL,
    IS_TENSOR_EXPERT_DATA_PARALLEL,
    IS_TENSOR_ZERO_PARALLEL,
    IS_WEIGHT_ZERO_PARALLEL,
    ParallelMode,
)
from internlm.core.context import global_context as gpc
from internlm.core.context.random import set_mode
from internlm.core.naive_amp import (
    NaiveAMPModel,
    set_fp32_attr_to_module,
    unwrap_naive_amp,
)
from internlm.core.parallel.comm.isp import (
    ISPCommModelConfig,
    ISPCommunicator,
    ISPCommunicatorSchedulerHook,
)
from internlm.core.parallel.comm.tensor import (
    EmbbedingSequenceParallelCommunicator,
    EmbbedingTensorParallelCommunicator,
    HeadSequenceParallelCommunicator,
    HeadTensorParallelCommunicator,
    LinearRole,
    MoESequenceParallelCommunicator,
    SequenceParallelCommunicator,
    TensorParallelCommunicator,
)
from internlm.core.parallel.comm.zero import ParamAsyncBcastHandler
from internlm.core.trainer import TrainState
from internlm.data.utils import unpack_type_ids
from internlm.model.builder import create_model
from internlm.model.metrics import SchedulerMetricHook
from internlm.model.modules.embedding import Embedding1D
from internlm.model.modules.linear import (
    ColumnParallelLinear,
    ParallelLinearWithCommExt,
    RewardModelLinear,
    RowParallelLinear,
    ScaleColumnParallelLinear,
)
from internlm.model.modules.utils import is_moe_param
from internlm.model.moe.megablock.mlp import (
    MegaBlockFeedForward,
    MegaBlockGroupedFeedForward,
)
from internlm.model.moe.moe import MoE
from internlm.model.ops.norm import RMSNorm
from internlm.model.registry import register_model_initializer
from internlm.monitor import set_env_var
from internlm.monitor.monitor import monitor_manager as mm
from internlm.solver.optimizer import (
    FSDPadaptOptimizer,
    HybridZeroOptimizer,
    HybridZeroOptimizer_v2,
)
from internlm.solver.optimizer.compatible_adamw import new_compatible_adamw
from internlm.solver.schedulers.beta2_scheduler import Beta2Scheduler
from internlm.solver.schedulers.lr_scheduler import FineTuneCosineAnnealingWarmupLR
from internlm.train.utils import create_param_groups, map_param_block
from internlm.utils.common import DummyProfile, SchedulerHook, get_current_device
from internlm.utils.logger import get_logger
from internlm.utils.megatron_timers import megatron_timer as timer
from internlm.utils.parallel import (
    is_replica_zero_parallel_parameter,
    is_tensor_data_parallel_parameter,
    is_tensor_expert_data_parallel_parameter,
    is_tensor_zero_parallel_parameter,
    is_using_isp,
    is_weight_zero_parallel_parameter,
    sync_model_param,
    sync_model_replica_param_group,
)
from internlm.utils.timeout import llm_timeout

try:
    import torch_npu
except (ImportError, ModuleNotFoundError):
    pass

logger = get_logger(__file__)
internlm_accelerator = get_accelerator()


def set_fp32_attr_for_model(model: Union[nn.Module, nn.ModuleList]):
    if not isinstance(model, nn.ModuleList):
        model = [model]

    for _chunk in model:
        for _, module in _chunk.named_modules():
            if isinstance(module, (RMSNorm, nn.LayerNorm)) and gpc.config.get("use_fp32_norm", False):
                set_fp32_attr_to_module(module)


def set_parallel_attr_for_param_groups(model: Union[nn.Module, nn.ModuleList]):
    def _check_module(name, module):
        # layer_norm
        if isinstance(module, (RMSNorm, nn.LayerNorm)):
            for param in module.parameters():
                setattr(param, IS_REPLICA_ZERO_PARALLEL, True)

        if isinstance(module, MoE):
            for param in module.moe_layer.gate.parameters():
                setattr(param, IS_REPLICA_ZERO_PARALLEL, True)

        # embedding and head

        if isinstance(module, (Embedding1D, ScaleColumnParallelLinear)):
            for param in module.parameters():
                if gpc.is_initialized(ParallelMode.TENSOR) and is_using_isp():
                    setattr(param, IS_TENSOR_DATA_PARALLEL, True)
                elif gpc.is_initialized(ParallelMode.TENSOR) and not is_using_isp():
                    setattr(param, IS_TENSOR_ZERO_PARALLEL, True)

        # for linear module
        if isinstance(
            module,
            (ParallelLinearWithCommExt, MegaBlockFeedForward, MegaBlockGroupedFeedForward),
        ):
            for param in module.parameters():
                if gpc.is_initialized(ParallelMode.EXPERT_DATA) and is_moe_param(param):
                    # module should be MoE experts's linear
                    setattr(param, IS_TENSOR_EXPERT_DATA_PARALLEL, True)
                elif not is_moe_param(param) and gpc.is_initialized(ParallelMode.TENSOR) and not is_using_isp():
                    setattr(param, IS_TENSOR_ZERO_PARALLEL, True)
                elif not is_moe_param(param) and gpc.is_initialized(ParallelMode.WEIGHT) and is_using_isp():
                    setattr(param, IS_WEIGHT_ZERO_PARALLEL, True)

        # for vit and vit project
        if "vision_tower" in name.lower() or "vision_proj" in name.lower():
            for param in module.parameters():
                setattr(param, IS_REPLICA_ZERO_PARALLEL, True)

    def _check_module_hf(_, module):
        # TODO: check parallel attribute for hf model
        for param in module.parameters():
            if gpc.is_initialized(ParallelMode.TENSOR) and is_using_isp():
                setattr(param, IS_TENSOR_DATA_PARALLEL, True)
            elif gpc.is_initialized(ParallelMode.TENSOR) and not is_using_isp():
                setattr(param, IS_TENSOR_ZERO_PARALLEL, True)

    for _chunk in unwrap_naive_amp(model):
        # set param parallel attribute
        for name, module in _chunk.named_modules():
            if gpc.config.model_type == "hf":
                _check_module_hf(name, module)
            else:
                _check_module(name, module)

        for name, param in _chunk.named_parameters():
            assert (
                is_replica_zero_parallel_parameter(param)
                or is_tensor_data_parallel_parameter(param)
                or is_tensor_zero_parallel_parameter(param)
                or is_weight_zero_parallel_parameter(param)
                or is_tensor_expert_data_parallel_parameter(param)
            ), f"parameter with name:{name} has no parallel attribution."


@llm_timeout(func_name="initialize_model")
def initialize_model(pre_process_func: Optional[Callable] = None, post_process_func: Optional[Callable] = None):
    """
    Initialize model with Automatic Mixed Precision.

    Returns:
        torch.nn.Module:
            The neural network model to be trained or evaluated.
    """
    if pre_process_func:
        pre_process_output = pre_process_func()

    register_model_initializer()

    model = create_model(model_type=gpc.config.model_type, **(gpc.config.model))

    if post_process_func:
        post_process_func(pre_process_output)

    # should be set before NaiveAMPModel
    set_fp32_attr_for_model(model)

    if isinstance(model, nn.ModuleList):
        model = nn.ModuleList(
            [
                NaiveAMPModel(
                    model=_m,
                    output_to_fp32=False,  # manually controlled by interleaved pipleline scheduler
                    dtype=gpc.config.model.get("dtype", torch.half),
                    sync_buffer=False,
                )
                for _m in model
            ]
        )
    else:
        model = NaiveAMPModel(
            model=model,
            output_to_fp32=gpc.is_no_pp_or_last_stage(),
            dtype=gpc.config.model.get("dtype", torch.half),
            sync_buffer=False,
        )

    set_parallel_attr_for_param_groups(model)

    # This sync is very important, cause the model weights kept in optimizer are copied
    # from the origin parameters in the memory, so we should make sure the dp sync
    # does not influence the model weights in optimizer be different with the origin parameters.
    sync_model_param(model)

    # This function is needed to make sure parameters that are not splitted by tensor parallelism are
    # the same across tensor parallelism.
    sync_model_replica_param_group(model)

    # Change random state mode to ParallelMode.DATA after model is built, guaranteeing the random
    # state in the same dp group are all the same.
    random_mode = ParallelMode.WEIGHT_DATA if is_using_isp() else ParallelMode.DATA
    set_mode(random_mode)

    return model


_T = TypeVar("_T")


def _submodule_filter(model: Union[nn.Module, nn.ModuleList], target_cls: Union[_T, Tuple[_T]]) -> Iterable[_T]:
    for _chunk in unwrap_naive_amp(model):
        for _module in _chunk.modules():
            if not isinstance(_module, target_cls):
                continue

            yield _module


def initialize_parallel_communicator(model: Union[nn.Module, nn.ModuleList]):
    """
    Initialize communicator for isp tensor parallel mode.

    Args:
        model (:class:`torch.nn.Module`): Your model instance to be trained or evaluated.

    Returns:
        An isp communicator for managing comp/comm overlap and memory pool.
    """
    isp_communicator = None
    _retain_out_sharded = gpc.config.model.get("parallel_output", True)

    if is_using_isp():
        isp_communicator = ISPCommunicator(
            model,
            ISPCommModelConfig(
                gpc.config.model.dtype,
                get_current_device(),
                gpc.config.model.checkpoint,
            ),
            gpc.config.parallel.weight.overlap,
            gpc.config.parallel.weight.memory_pool,
            gpc.get_group(ParallelMode.WEIGHT),
        )
        # register communicator for isp column parallel linear.
        ColumnParallelLinear.register_cls_communicator(isp_communicator)
        # row parallel linear will not be used.
        RowParallelLinear.register_cls_communicator(None)
        _head_communicator = HeadSequenceParallelCommunicator(ParallelMode.TENSOR, _retain_out_sharded)
        _embedding_communicator = EmbbedingSequenceParallelCommunicator(ParallelMode.TENSOR)

    # register communictor for mtp/msp/fsp linear.

    # tensor parallel
    if gpc.config.parallel.tensor.mode == "mtp":
        ColumnParallelLinear.register_cls_communicator(
            TensorParallelCommunicator(process_group=gpc.get_group(ParallelMode.TENSOR), role=LinearRole.COLUMN)
        )
        RowParallelLinear.register_cls_communicator(
            TensorParallelCommunicator(process_group=gpc.get_group(ParallelMode.TENSOR), role=LinearRole.ROW)
        )
        _head_communicator = HeadTensorParallelCommunicator(ParallelMode.TENSOR, _retain_out_sharded)
        _embedding_communicator = EmbbedingTensorParallelCommunicator(ParallelMode.TENSOR)
    # sequence parallel
    if gpc.config.parallel.tensor.mode in ("msp", "fsp"):
        save_total_input_as_activation = gpc.config.parallel.tensor.mode == "msp"

        ColumnParallelLinear.register_cls_communicator(
            SequenceParallelCommunicator(
                process_group=gpc.get_group(ParallelMode.TENSOR),
                role=LinearRole.COLUMN,
                save_total_input_as_activation=save_total_input_as_activation,
            )
        )
        RowParallelLinear.register_cls_communicator(
            SequenceParallelCommunicator(
                gpc.get_group(ParallelMode.TENSOR),
                role=LinearRole.ROW,
                save_total_input_as_activation=save_total_input_as_activation,
            )
        )

        _head_communicator = HeadSequenceParallelCommunicator(
            ParallelMode.TENSOR, _retain_out_sharded, save_total_input_as_activation
        )
        _embedding_communicator = EmbbedingSequenceParallelCommunicator(ParallelMode.TENSOR)

        # MoE sequence parallel
        if gpc.config.model.get("num_experts", 1) > 1:
            _column_communicator = TensorParallelCommunicator(
                process_group=gpc.get_group(ParallelMode.TENSOR), role=LinearRole.COLUMN
            )
            _row_communicator = TensorParallelCommunicator(
                process_group=gpc.get_group(ParallelMode.TENSOR), role=LinearRole.ROW
            )
            for moe in _submodule_filter(model, MoE):
                # 1. the linear in MoE degrades the parallel communication pattern from sp to tp
                for column_linear in _submodule_filter(moe, ColumnParallelLinear):
                    column_linear.register_communicator(_column_communicator)
                for row_linear in _submodule_filter(moe, RowParallelLinear):
                    row_linear.register_communicator(_row_communicator)
                # 2. register MoESequenceParallelCommunicator for MoE layer
                MoESequenceParallelCommunicator(ParallelMode.TENSOR).register_module_hook(moe)

    # register communitorc for embedding layer.
    for embedding in _submodule_filter(model, Embedding1D):
        _embedding_communicator.register_module_hook(embedding)

    # register communictor for head layer.
    ScaleColumnParallelLinear.register_cls_communicator(_head_communicator)
    RewardModelLinear.register_cls_communicator(_head_communicator)

    return isp_communicator


@llm_timeout(func_name="initialize_optimizer")
def initialize_optimizer(model: Union[nn.Module, nn.ModuleList], isp_communicator: ISPCommunicator = None):
    """
    Initialize optimizer.

    Args:
        model (:class:`torch.nn.Module`): Your model instance to be trained or evaluated.

    Returns:
        A tuple of (optimizer, beta2_scheduler, lr_scheduler).
    """

    adam_cfg = gpc.config.adam
    zero_cfg = gpc.config.hybrid_zero_optimizer
    grad_scal_cfg = gpc.config.grad_scaler

    if "use_split_tensor_optim" in zero_cfg and zero_cfg.use_split_tensor_optim:
        map_param_block(model)

    params = create_param_groups(model, adam_cfg.weight_decay)

    naive_optimizer = new_compatible_adamw(
        params=params,
        lr=adam_cfg.lr,
        betas=(adam_cfg.adam_beta1, adam_cfg.adam_beta2),
        eps=adam_cfg.adam_eps,
    )

    if (
        zero_cfg.overlap_sync_grad
        and gpc.is_using_parallel_mode(ParallelMode.PIPELINE)
        and gpc.is_pipeline_first_stage() is False
    ):
        # When pipeline parallelism is enabled, we prefer to only enable optimizer
        # gradient communication overlap in the first stage, to avoid amplifying
        # the communication overhead stage by stage in cases where the optimizer
        # communication overhead is greater than the compute overhead.
        # For pipeline stages except the first, even if overlap is not enabled,
        # their gradient synchronization overhead can be well hidden by
        # the inherent bubbles of pipeline parallelism.
        zero_cfg.overlap_sync_grad = False

    if zero_cfg.overlap_sync_param:
        param_bcast_sync_handler = ParamAsyncBcastHandler(ParallelMode.ZERO1, model, isp_communicator)
    else:
        param_bcast_sync_handler = None

    if not gpc.config.parallel.zero1.fsdp:
        if (
            "use_split_tensor_optim" not in gpc.config.hybrid_zero_optimizer
            or not gpc.config.hybrid_zero_optimizer.use_split_tensor_optim
        ):
            optimizer = HybridZeroOptimizer(
                naive_optimizer,
                grad_scal_cfg=grad_scal_cfg,
                zero_cfg=zero_cfg,
                param_bcast_sync_handler=param_bcast_sync_handler,
                isp_communicator=isp_communicator,
            )
        else:
            optimizer = HybridZeroOptimizer_v2(
                naive_optimizer,
                grad_scal_cfg=grad_scal_cfg,
                zero_cfg=zero_cfg,
                param_bcast_sync_handler=param_bcast_sync_handler,
                isp_communicator=isp_communicator,
            )
    else:
        optimizer = FSDPadaptOptimizer(
            naive_optimizer,
            grad_scal_cfg=grad_scal_cfg,
            zero_cfg=zero_cfg,
        )

    beta2_scheduler = Beta2Scheduler(optimizer=naive_optimizer, **gpc.config.beta2_scheduler)

    lr_scheduler = FineTuneCosineAnnealingWarmupLR(optimizer, **gpc.config.lr_scheduler)

    return optimizer, beta2_scheduler, lr_scheduler


def get_scheduler_hooks(metric, zero_optim, isp_communicator) -> List[SchedulerHook]:
    scheduler_hooks: List[SchedulerHook] = []

    if metric is not None:
        scheduler_hooks.append(
            SchedulerMetricHook(
                metric=metric,
                skip=(
                    gpc.is_using_parallel_mode(ParallelMode.PIPELINE)
                    and hasattr(gpc.config.model, "num_chunks")
                    and gpc.config.model.num_chunks > 1
                    and gpc.config.parallel["pipeline"].get("interleaved_overlap", False)
                ),
            ),
        )

    if isp_communicator is not None and gpc.config.parallel["weight"].get("overlap", False):
        scheduler_hooks.append(ISPCommunicatorSchedulerHook(isp_communicator, zero_optim))

    return scheduler_hooks


@llm_timeout(func_name="load_new_batch")
def load_new_batch(train_dl: DataLoader, train_iter: Iterable, train_state: TrainState):
    """
    Load and return the new batch data based on training data loader.

    Args:
        train_dl (torch.utils.data.DataLoader): Dataloader for training.
        train_iter (Iterable): Data iterator from which get a batch of data, obtained by calling iter(dataloader).
        train_state (TrainState): Current training state.

    Returns: A batch data and the updated train_iter.
    """

    timer("batch-gen").start()
    try:
        batch = next(train_iter)  # structure is ({'input_ids': Tensor, 'cu_seqlens': Tensor}, Tensor)
        if hasattr(train_state, "batch_sampler_iter"):
            next(train_state.batch_sampler_iter)
    except StopIteration:
        train_iter = iter(train_dl)
        batch = next(train_iter)
        train_state.num_consumed_samples_in_epoch = 0
        if hasattr(train_state, "batch_sampler"):
            train_state.batch_sampler.batch_count = 0
            train_state.batch_sampler.num_consumed_samples_in_epoch = 0
            train_state.batch_sampler_iter = iter(train_state.batch_sampler)
            next(train_state.batch_sampler_iter)
    timer("batch-gen").stop()

    if batch[0].get("type_ids", None) is not None:
        # if use_packed_dataset is False, we need to unpack type_ids
        if not gpc.config.data.use_packed_dataset:
            if gpc.config.data.type != "hf" or gpc.config.model_type != "hf":
                batch[0]["type_ids"] = unpack_type_ids(batch[0]["type_ids"], batch[0]["cu_seqlens"])

    return batch, train_iter


def initialize_llm_profile(profiling: bool = False, start_time: str = None):
    """Initialize and return the profiler context manager instance."""

    if profiling and gpc.get_local_rank(ParallelMode.DATA) == 0 and gpc.get_local_rank(ParallelMode.TENSOR) == 0:
        schedule_config = {"wait": 1, "warmup": 1, "active": 1, "repeat": 1, "skip_first": 3}
        trace_path = (
            f"RUN/{gpc.config.JOB_NAME}/{start_time}/traces/rank{gpc.get_global_rank()}_"
            f"dp{gpc.get_local_rank(ParallelMode.DATA)}_"
            f"wp{gpc.get_local_rank(ParallelMode.WEIGHT)}_"
            f"tp{gpc.get_local_rank(ParallelMode.TENSOR)}"
        )
        if internlm_accelerator.get_accelerator_backend() == AcceleratorType.NPU:
            experimental_config = torch_npu.profiler._ExperimentalConfig(
                aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
                profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                l2_cache=False,
            )
            llm_profile = torch_npu.profiler.profile(
                activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
                schedule=torch_npu.profiler.schedule(**schedule_config),
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(trace_path),
                record_shapes=True,
                profile_memory=True,
                with_stack=False,
                with_flops=False,
                with_modules=False,
                experimental_config=experimental_config,
            )
            logger.info(f"Do profiling for NPU on rank {gpc.get_global_rank()}!")
        else:
            llm_profile = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(**schedule_config),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(trace_path),
                with_stack=True,
                with_modules=True,
                profile_memory=True,
            )
            logger.info(f"Do profiling for GPU on rank {gpc.get_global_rank()}!")
    else:
        llm_profile = DummyProfile()

    return llm_profile


@llm_timeout(func_name="record_current_batch_training_metrics")
def record_current_batch_training_metrics(
    get_tflops_func,
    logger,
    writer,
    success_update,
    batch_count,
    batch,
    train_state,
    optimizer,
    beta2_scheduler,
    trainer,
    start_time,
    very_begining_time,
    loss,
    moe_loss,
    grad_norm,
    metric,
):
    """
    Print some training metrics of current batch.
    """

    set_env_var(key="LAST_ACTIVE_TIMESTAMP", value=int(time.time()))

    timer.store_last_timers()
    if success_update in (0, True):
        train_state.num_consumed_tokens += batch[1].nelement() * gpc.get_world_size(ParallelMode.DATA)
    if gpc.is_no_pp_or_last_stage():
        acc_perplex = metric.get_metric()

    if success_update and gpc.is_rank_for_log():
        lr = optimizer.param_groups[0]["lr"]
        if hasattr(trainer.engine.optimizer, "grad_scaler"):
            scaler = trainer.engine.optimizer.grad_scaler._scale.item()
        elif hasattr(trainer.engine.optimizer.optim, "grad_scaler"):
            scaler = trainer.engine.optimizer.optim.grad_scaler._scale.item()

        num_tokens_in_batch = batch[1].nelement()
        real_num_tokens = math.ceil(acc_perplex.pop("real_token_num") / gpc.get_world_size(ParallelMode.GLOBAL))
        # TODO: check logic
        if gpc.config.data.type == "hf" and gpc.config.model_type == "hf" and not gpc.config.data.use_packed_dataset:
            num_samples_in_batch = gpc.config.data.micro_bsz * gpc.config.data.micro_num
            max_length_in_batch = batch[0]["attention_mask"].sum(dim=1).max().item()
            max_samples_in_batch = gpc.config.data.micro_bsz
            min_samples_in_batch = gpc.config.data.micro_bsz
        else:
            num_samples_in_batch = sum([len(b) - 1 for b in batch[0]["cu_seqlens"]])
            max_length_in_batch = max([(b[1:] - b[:-1]).max().item() for b in batch[0]["cu_seqlens"]])
            max_samples_in_batch = max([len(b) - 1 for b in batch[0]["cu_seqlens"]])
            min_samples_in_batch = min([len(b) - 1 for b in batch[0]["cu_seqlens"]])
        time_cost = time.time() - start_time
        tk_per_gpu = round(
            num_tokens_in_batch * gpc.get_world_size(ParallelMode.DATA) / gpc.get_world_size(ParallelMode.GLOBAL),
            4,
        )
        tgs_statistic = train_state.tgs_statistic
        tgs_statistic["sum_step"] += 1
        tgs_statistic["sum_tg"] += tk_per_gpu
        tgs_statistic["total_time"] = time.time() - very_begining_time
        tgs_statistic["sum_last_tg_10"] += tk_per_gpu
        tgs_statistic["sum_last_time_10"] += time_cost
        tgs_statistic["sum_last_tg_50"] += tk_per_gpu
        tgs_statistic["sum_last_time_50"] += time_cost
        tgs_statistic["SMA_tg_50"] += tk_per_gpu
        tgs_statistic["SMA_time_50"] += time_cost
        tgs_statistic["SMA_tg_50_list"].append(tk_per_gpu)
        tgs_statistic["SMA_time_50_list"].append(time_cost)
        if tgs_statistic["sum_step"] > 50:
            tgs_statistic["SMA_tg_50"] -= tgs_statistic["SMA_tg_50_list"][0]
            tgs_statistic["SMA_time_50"] -= tgs_statistic["SMA_time_50_list"][0]
            tgs_statistic["SMA_tg_50_list"].popleft()
            tgs_statistic["SMA_time_50_list"].popleft()

        last_tgs_1 = round(tk_per_gpu / time_cost, 2)
        tgs_statistic["sum_tgs"] += last_tgs_1

        if tgs_statistic["sum_step"] % 10 == 0:
            tgs_statistic["last_tgs_10"] = round(tgs_statistic["sum_last_tg_10"] / tgs_statistic["sum_last_time_10"], 2)
            tgs_statistic["sum_last_tg_10"] = 0
            tgs_statistic["sum_last_time_10"] = 0

        if tgs_statistic["sum_step"] % 50 == 0:
            tgs_statistic["last_tgs_50"] = round(tgs_statistic["sum_last_tg_50"] / tgs_statistic["sum_last_time_50"], 2)
            tgs_statistic["sum_last_tg_50"] = 0
            tgs_statistic["sum_last_time_50"] = 0

        last_tgs_10 = tgs_statistic["last_tgs_10"]
        last_tgs_50 = tgs_statistic["last_tgs_50"]

        tgs_all = round(tgs_statistic["sum_tg"] / tgs_statistic["total_time"], 2)
        tgs_avg = round(tgs_statistic["sum_tgs"] / tgs_statistic["sum_step"], 2)
        tgs_SMA = round(tgs_statistic["SMA_tg_50"] / tgs_statistic["SMA_time_50"], 2)

        tflops = get_tflops_func(time_cost)

        tgs_origin = round(
            num_tokens_in_batch
            * gpc.get_world_size(ParallelMode.DATA)
            / gpc.get_world_size(ParallelMode.GLOBAL)
            / time_cost,
            2,
        )

        real_tgs = round(
            real_num_tokens / time_cost,
            2,
        )

        infos = {
            "tflops": tflops,
            "step": batch_count,
            "loss": loss.item() - moe_loss.item() if moe_loss is not None else loss.item(),
            "real_tgs": real_tgs,
            "tgs (tokens/gpu/second)": tgs_origin,
            "tgs/last_tgs_1": last_tgs_1,
            "tgs/tgs_all": tgs_all,
            "tgs/tgs_avg": tgs_avg,
            "tgs/tgs_SMA": tgs_SMA,
            "tgs/last_tgs_10": last_tgs_10,
            "tgs/last_tgs_50": last_tgs_50,
            "lr": lr,
            "loss_scale": scaler,
            "grad_norm": grad_norm,
        }
        if moe_loss is not None:
            infos["moe_loss"] = moe_loss.item()

        infos["micro_num"] = len(batch[1])
        infos["num_consumed_tokens"] = train_state.num_consumed_tokens
        infos["inf_nan_skip_batches"] = train_state.inf_nan_skip_batches
        infos["num_samples_in_batch"] = num_samples_in_batch  # the number of batches which have the most samples
        infos["largest_length"] = max_length_in_batch  # the longest input
        infos["largest_batch"] = max_samples_in_batch  # the batch with the most samples
        infos["smallest_batch"] = min_samples_in_batch
        infos["adam_beta2"] = beta2_scheduler.get_beta2()

        fwd_bwd_time = round(timer("fwd-bwd").elapsed(), 2)
        infos["fwd_bwd_time"] = fwd_bwd_time
        bwd_time = round(timer("bwd").elapsed(), 2)
        infos["bwd_time"] = bwd_time

        for key, value in acc_perplex.items():
            infos[key] = value

        line = ""
        for key, value in infos.items():
            line += f"{key}={value} "
            if isinstance(value, dict):
                writer.add_scalars(key=key, value=value, step=train_state.step_count)
            else:
                writer.add_scalar(key=key, value=value, step=train_state.step_count)

        logger.info(line)

        # if loss spike occurs, send alert info to feishu
        mm.monitor_loss_spike(
            alert_address=gpc.config.monitor.alert.feishu_alert_address,
            step_count=batch_count,
            cur_step_loss=loss.item(),
        )
