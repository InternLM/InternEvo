from typing import Iterable, Tuple, TypeVar, Union

from torch import nn

from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from internlm.core.naive_amp import unwrap_naive_amp
from internlm.core.parallel.comm import (
    EmbeddingSequenceParallelCommunicator,
    EmbeddingTensorParallelCommunicator,
    EmbeddingWeightParallelCommunicator,
    HeadSequenceParallelCommunicator,
    HeadTensorParallelCommunicator,
    HeadWeightParallelCommunicator,
    ISPCommModelConfig,
    ISPCommunicator,
    ISPCommunicatorWrapper,
    LinearRole,
    MoESequenceParallelCommunicator,
    SequenceParallelCommunicator,
    TensorParallelCommunicator,
)
from internlm.model.model_ops.modules.embedding import Embedding1D
from internlm.model.model_ops.modules.linear import (
    ColumnParallelLinear,
    GroupedColumnLinear,
    GroupedRowLinear,
    GroupedWPLinear,
    RewardModelLinear,
    RowParallelLinear,
    ScaleColumnParallelLinear,
)
from internlm.model.model_ops.moe import Experts, MoE
from internlm.utils.common import get_current_device
from internlm.utils.parallel import is_using_fsdp, is_using_isp
from internlm.utils.utils import TensorParallelMode

_T = TypeVar("_T")


def submodule_filter(model: Union[nn.Module, nn.ModuleList], target_cls: Union[_T, Tuple[_T]]) -> Iterable[_T]:
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
        An isp communicator for managing comp/comm overlap.
    """
    isp_communicator_wrapper = None
    _retain_out_sharded = gpc.config.model.get("parallel_output", True)

    if is_using_isp():
        isp_communicator = ISPCommunicator(
            model,
            ISPCommModelConfig(
                gpc.config.model.dtype,
                get_current_device(),
                gpc.config.model.checkpoint,
            ),
            gpc.config.parallel.weight.overlap and not is_using_fsdp(),
            gpc.get_group(ParallelMode.WEIGHT),
            is_moe=False,
            selective_ckpt_offload=gpc.config.get("selective_checkpoint_offload", False),
            early_reduce_scatter_release=gpc.config.parallel.weight.early_reduce_scatter_release,
        )
        # register communicator for isp column parallel linear.
        ColumnParallelLinear.register_cls_communicator(isp_communicator)
        # row parallel linear will not be used.
        RowParallelLinear.register_cls_communicator(None)
        _head_communicator = HeadWeightParallelCommunicator(
            weight_process_group=gpc.get_group(ParallelMode.WEIGHT),
            seq_process_group=gpc.get_group(ParallelMode.TENSOR),
            retain_out_sharded=_retain_out_sharded,
        )
        _embedding_communicator = EmbeddingWeightParallelCommunicator(ParallelMode.WEIGHT)

        if gpc.config.model.get("num_experts", 1) > 1:
            # register communicator for moe isp column parallel linear.
            # NOTE: this wil overwrite registed communicator
            moe_isp_communicator = ISPCommunicator(
                model,
                ISPCommModelConfig(
                    gpc.config.model.dtype,
                    get_current_device(),
                    gpc.config.model.checkpoint,
                ),
                gpc.config.parallel.expert_weight.overlap,
                gpc.get_group(ParallelMode.EXPERT_WEIGHT),
                is_moe=True,
                early_reduce_scatter_release=gpc.config.parallel.expert_weight.early_reduce_scatter_release,
            )
            for moe in submodule_filter(model, Experts):
                for column_linear in submodule_filter(moe, (ColumnParallelLinear, GroupedWPLinear)):
                    column_linear.register_communicator(moe_isp_communicator)
                for row_linear in submodule_filter(moe, RowParallelLinear):
                    row_linear.register_communicator(None)

            isp_communicator_wrapper = ISPCommunicatorWrapper([isp_communicator, moe_isp_communicator])
        else:
            isp_communicator_wrapper = ISPCommunicatorWrapper([isp_communicator])

    # register communictor for mtp/msp/fsp linear.

    # tensor parallel
    if gpc.config.parallel.tensor.mode == TensorParallelMode.mtp.name:
        ColumnParallelLinear.register_cls_communicator(
            TensorParallelCommunicator(process_group=gpc.get_group(ParallelMode.TENSOR), role=LinearRole.COLUMN)
        )
        RowParallelLinear.register_cls_communicator(
            TensorParallelCommunicator(process_group=gpc.get_group(ParallelMode.TENSOR), role=LinearRole.ROW)
        )

        if gpc.config.model.get("num_experts", 1) > 1:
            GroupedColumnLinear.register_cls_communicator(
                TensorParallelCommunicator(process_group=gpc.get_group(ParallelMode.TENSOR), role=LinearRole.COLUMN)
            )
            GroupedRowLinear.register_cls_communicator(
                TensorParallelCommunicator(process_group=gpc.get_group(ParallelMode.TENSOR), role=LinearRole.ROW)
            )
            GroupedWPLinear.register_cls_communicator(None)
            # treat as sequence paralle if no_tp
            if gpc.config.parallel.expert.no_tp:
                _column_communicator = TensorParallelCommunicator(
                    process_group=gpc.get_group(ParallelMode.EXPERT_TENSOR), role=LinearRole.COLUMN
                )
                _row_communicator = TensorParallelCommunicator(
                    process_group=gpc.get_group(ParallelMode.EXPERT_TENSOR), role=LinearRole.ROW
                )
                for moe in submodule_filter(model, MoE):
                    # 1. the linear in MoE degrades as no tp communication pattern
                    for column_linear in submodule_filter(moe, ColumnParallelLinear):
                        column_linear.register_communicator(_column_communicator)
                    for row_linear in submodule_filter(moe, RowParallelLinear):
                        row_linear.register_communicator(_row_communicator)
                    # 2. register MoESequenceParallelCommunicator for MoE layer
                    MoESequenceParallelCommunicator(ParallelMode.TENSOR, reverse=True).register_module_hook(moe)

        _head_communicator = HeadTensorParallelCommunicator(ParallelMode.TENSOR, _retain_out_sharded)
        _embedding_communicator = EmbeddingTensorParallelCommunicator(ParallelMode.TENSOR)
    # sequence parallel
    if gpc.config.parallel.tensor.mode in (TensorParallelMode.msp.name, TensorParallelMode.fsp.name):
        save_total_input_as_activation = gpc.config.parallel.tensor.mode == TensorParallelMode.msp.name

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
        if gpc.config.model.get("num_experts", 1) > 1:
            GroupedColumnLinear.register_cls_communicator(
                SequenceParallelCommunicator(
                    process_group=gpc.get_group(ParallelMode.TENSOR),
                    role=LinearRole.COLUMN,
                    save_total_input_as_activation=save_total_input_as_activation,
                )
            )
            GroupedRowLinear.register_cls_communicator(
                SequenceParallelCommunicator(
                    gpc.get_group(ParallelMode.TENSOR),
                    role=LinearRole.ROW,
                    save_total_input_as_activation=save_total_input_as_activation,
                )
            )
            GroupedWPLinear.register_cls_communicator(None)
            if gpc.config.parallel.expert.no_tp:
                _column_communicator = TensorParallelCommunicator(
                    process_group=gpc.get_group(ParallelMode.EXPERT_TENSOR), role=LinearRole.COLUMN
                )
                _row_communicator = TensorParallelCommunicator(
                    process_group=gpc.get_group(ParallelMode.EXPERT_TENSOR), role=LinearRole.ROW
                )
                for moe in submodule_filter(model, MoE):
                    # 1. the linear in MoE degrades as no tp communication pattern
                    for column_linear in submodule_filter(moe, ColumnParallelLinear):
                        column_linear.register_communicator(_column_communicator)
                    for row_linear in submodule_filter(moe, RowParallelLinear):
                        row_linear.register_communicator(_row_communicator)

        _head_communicator = HeadSequenceParallelCommunicator(
            ParallelMode.TENSOR, _retain_out_sharded, save_total_input_as_activation
        )

        _embedding_communicator = EmbeddingSequenceParallelCommunicator(ParallelMode.TENSOR)

    # register communitorc for embedding layer.
    if not is_using_fsdp():
        for embedding in submodule_filter(model, Embedding1D):
            _embedding_communicator.register_module_hook(embedding)

    # register communictor for head layer.
    ScaleColumnParallelLinear.register_cls_communicator(_head_communicator)
    RewardModelLinear.register_cls_communicator(_head_communicator)

    return isp_communicator_wrapper
