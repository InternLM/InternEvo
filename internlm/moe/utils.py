from typing import Any, Tuple

import torch
from torch import Tensor


# Based on https://github.com/pytorch/pytorch/pull/40762
class AllToAll(torch.autograd.Function):
    """
    All to all communication
    """

    @staticmethod
    def forward(
        ctx: Any,
        inputs: Tensor,
        output_split_sizes=None,
        input_split_sizes=None,
        group: torch.distributed.ProcessGroup = None,
        async_op=False,
    ) -> Tensor:  # type: ignore

        ctx.input_shape = inputs.shape
        ctx.output_split_sizes = output_split_sizes
        ctx.input_split_sizes = input_split_sizes
        ctx.group = group

        inputs = inputs.contiguous()
        out = (
            torch.empty_like(inputs)
            if output_split_sizes is None
            else inputs.new_empty(size=[sum(output_split_sizes)] + list(inputs.size()[1:]))
        )
        handle = torch.distributed.all_to_all_single(
            out,
            inputs,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
            async_op=async_op,
        )

        # if async_op=False, handle will be None
        return out, handle

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor, _) -> Tuple[None, Tensor]:
        if ctx.needs_input_grad[0]:
            grad_output = grad_output.contiguous()
            out = torch.empty(ctx.input_shape, device=grad_output.device, dtype=grad_output.dtype)
            torch.distributed.all_to_all_single(
                out,
                grad_output,
                output_split_sizes=ctx.input_split_sizes,
                input_split_sizes=ctx.output_split_sizes,
                group=ctx.group,
            )
            return out, None, None, None, None
        return None, None, None, None, None


def all_to_all(x, output_split_sizes=None, input_split_sizes=None, group=None, async_op=False):
    return AllToAll.apply(x, output_split_sizes, input_split_sizes, group, async_op)
