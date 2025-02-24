"""
A simple operator selector, used for compatibility with different platforms such as CUDA and Ascend,
as well as whether to enable flash-attn operator optimization, may be replaced by a more comprehensive
operator compatibility layer in the future.

This file implements support for the cross entropy operators.
"""

from enum import Enum

import torch
from torch import nn

from internlm.accelerator import AcceleratorType, get_accelerator
from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from internlm.model.ops.cross_entropy_ops import (
    CrossEntropyApexVocabParallel,
    CrossEntropyLossApex,
    CrossEntropyPython,
)
from internlm.utils.logger import get_logger

logger = get_logger(__file__)
internlm_accelerator = get_accelerator()


def average_losses_across_data_parallel_group(losses):
    """Reduce a tensor of losses across all GPUs."""
    averaged_losses = torch.cat([loss.clone().detach().view(1) for loss in losses])
    torch.distributed.all_reduce(averaged_losses, group=gpc.get_group(ParallelMode.DATA))
    averaged_losses = averaged_losses / gpc.get_world_size(ParallelMode.DATA)

    return averaged_losses


class CrossEntropyOpType(Enum):
    torch_naive = 1  # CrossEntropy from torch
    flash_vocab_parallel = 2  # VocabParallel CorssEntropy from flash_attn
    apex_naive = 3  # CrossEntropy from apex
    py_vocab_parallel = 4  # self-implemented VocabParallel CrossEntropy
    py_naive = 5  # self-implemented CrossEntropy
    # sequence_parallel = 6 # self-implemented SequenceParallel CrossEntropy


cross_entropy_op_name_map = {
    "torch_naive": CrossEntropyOpType.torch_naive,
    "flash_vocab_parallel": CrossEntropyOpType.flash_vocab_parallel,
    "apex_naive": CrossEntropyOpType.apex_naive,
    "py_vocab_parallel": CrossEntropyOpType.py_vocab_parallel,
    "py_naive": CrossEntropyOpType.py_naive,
    # "sequence_parallel": CrossEntropyOpType.sequence_parallel,
}


# TODO: ops是否需要实现更加统一的形式
def new_cross_entropy(
    op_type: str = "py_vocab_parallel",
    ignore_index: int = -100,
    label_smoothing: float = 0,
    parallel_output: bool = False,
    inplace_backward: bool = True,
    reduction: str = "none",
):
    try:
        op_type = cross_entropy_op_name_map[op_type]
    except KeyError:
        raise KeyError(f"op_type only support: {cross_entropy_op_name_map.keys()}")

    if internlm_accelerator.get_accelerator_backend() is not AcceleratorType.GPU:
        assert op_type in [
            CrossEntropyOpType.torch_naive,
            CrossEntropyOpType.py_vocab_parallel,
        ], "no-GPU env only support 'torch_naive' or 'py_vocab_parallel loss function"

    if op_type == CrossEntropyOpType.torch_naive:

        assert parallel_output is False, (
            "'torch_naive' (nn.CrossEntropyLoss) don't support parallel_output, "
            "try use 'flash_vocab_parallel' or 'py_vocab_parallel'"
        )

        return nn.CrossEntropyLoss(reduction=reduction, label_smoothing=label_smoothing, ignore_index=ignore_index)

    elif op_type == CrossEntropyOpType.flash_vocab_parallel:

        assert gpc.get_group(ParallelMode.TENSOR) is not None, "The process group should not be None."

        try:
            from flash_attn.losses.cross_entropy import (
                CrossEntropyLoss as FlashCrossEntropyLoss,
            )

            flash_cross_entropy_impl = True
        except (ModuleNotFoundError, ImportError):
            flash_cross_entropy_impl = False

        assert (
            gpc.config.model.get("use_flash_attn", False) and flash_cross_entropy_impl
        ), "Only flash cross entropy support parallel_output"

        assert (
            internlm_accelerator.get_accelerator_backend() is AcceleratorType.GPU
        ), "flash cross entropy only support gpu backend"

        logger.warning(
            "You are using flash_attn cross_entropy operators, \
            which may result loss divergency in long sequence."
        )

        return FlashCrossEntropyLoss(
            ignore_index=ignore_index,
            reduction=reduction,
            label_smoothing=label_smoothing,
            process_group=gpc.get_group(ParallelMode.TENSOR),
            inplace_backward=inplace_backward,
        )

    elif op_type == CrossEntropyOpType.apex_naive:
        assert parallel_output is False, (
            "'apex_naive' (nn.CrossEntropyLoss) can'ts support parallel_output,"
            "try use 'flash_vocab_parallel' or 'py_vocab_parallel'"
        )

        return CrossEntropyLossApex(
            ignore_index=ignore_index,
            reduction=reduction,
            inplace_backward=inplace_backward,
            label_smoothing=label_smoothing,
        )

    elif op_type == CrossEntropyOpType.py_vocab_parallel:
        assert gpc.get_group(ParallelMode.TENSOR) is not None, "The process group should not be None."

        return CrossEntropyApexVocabParallel(
            ignore_index=ignore_index,
            reduction=reduction,
            label_smoothing=label_smoothing,
            process_group=gpc.get_group(ParallelMode.TENSOR),
        )

    elif op_type == CrossEntropyOpType.py_naive:
        assert parallel_output is False, (
            "'py_naive' (nn.CrossEntropyLoss) don't support parallel_output,"
            "try use 'flash_vocab_parallel' or 'py_vocab_parallel'"
        )
        return CrossEntropyPython(ignore_index=ignore_index, reduction=reduction)

    else:
        raise RuntimeError(f"unkown loss function type: {op_type}")
