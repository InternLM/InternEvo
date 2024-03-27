import sys

import torch

from internlm.accelerator import get_accelerator
from internlm.model.utils import Silu

try:
    import stk
except ImportError:
    pass

internlm_accelerator = get_accelerator()


class TensorParallelBmm(torch.autograd.Function):
    """
    Tensor parallel sdd
    """

    @staticmethod
    @internlm_accelerator.amp.custom_fwd
    def forward(ctx, x, w, group=None):
        # [m, k] x [n, k] = [m, n]
        if not x.is_contiguous() or not w.is_contiguous():
            raise ValueError("Expected contiguous 'x' and 'w'.")

        ctx.group = group
        ctx.save_for_backward(
            x,
            w,
        )

        return torch.bmm(x, w)

    @staticmethod
    @internlm_accelerator.amp.custom_bwd
    def backward(ctx, grad):
        x, w = ctx.saved_tensors[:2]

        dx = None
        if ctx.needs_input_grad[0]:
            dx = torch.bmm(grad, w.transpose(-2, -1))
        if ctx.group is not None:
            handle_x = torch.distributed.all_reduce(dx, group=ctx.group, async_op=True)

        dw = None
        if ctx.needs_input_grad[1]:
            dw = torch.bmm(x.transpose(-2, -1), grad)

        # NOTE: Be careful to wait and only cast dw to the output dtype once
        # we've blocked on the asynchronous NCCL operation.
        if ctx.group is not None:
            handle_x.wait()

        dw = dw.to(w.dtype)
        return dx, dw, None


def tensor_parallel_bmm(x, w, group=None):
    return TensorParallelBmm.apply(x, w, group)


def promote_scalar(x: torch.Tensor) -> torch.Tensor:
    return x.view(1) if len(x.size()) == 0 else x


def _gather_weights(w, group, parallel_w=None, async_op=False):
    """Gather the weights across the process group.

    Args:
      w: torch.Tensor, local shard of the weights.
      group: ProcessGroup, the group to gather across.
      parallel_w: torch.Tensor, option output tensor to use
       for the gather.
      async_op: Whether to gather asynchronously.

    Returns:
      The gathered weights tensor and a handle for asynchronous
      communication.
    """
    n, k = w.shape
    world_size = torch.distributed.get_world_size(group)

    if parallel_w is None:
        parallel_w = torch.empty(n * world_size, k, device=w.device, dtype=w.dtype)
    handle = torch.distributed.all_gather_into_tensor(parallel_w, w, group=group, async_op=async_op)
    return parallel_w, handle


def _scaled_reduce_scatter(parallel_dw, group, dw=None, async_op=False):
    """Scatter reduce the weights across the process group.

    Args:
      parallel_dw: torch.Tensor, local shard of the weights.
      group: ProcessGroup, the group to scatter-reduce across.
      dw: torch.Tensor, option output tensor to use for the op.
      async_op: Whether to scatter reduce asynchronously.

    Returns:
      The reduced weights tensor, scaled by 1 / world_size, and
      a handle for asynchronous communication.
    """
    n, k = parallel_dw.shape
    world_size = torch.distributed.get_world_size(group)
    assert (n % world_size) == 0

    # Pre-scale the gradients by the world size.
    #
    # NOTE: Reduce in float32, always.
    parallel_dw = parallel_dw.float() / world_size

    if dw is None:
        dw = torch.empty(n // world_size, k, device=parallel_dw.device, dtype=torch.float32)
    handle = torch.distributed.reduce_scatter_tensor(dw, parallel_dw, group=group, async_op=async_op)
    return dw, handle


class WeightParallelSddNt(torch.autograd.Function):
    """
    Weight parallel sdd
    """

    @staticmethod
    @internlm_accelerator.amp.custom_fwd
    def forward(ctx, x, w, topo, group):
        # [m, k] x [n, k] = [m, n]
        if not x.is_contiguous() or not w.is_contiguous():
            raise ValueError("Expected contiguous 'x' and 'w'.")

        ctx.group = group
        ctx.shape = topo.shape
        ctx.save_for_backward(
            x,
            w,
            topo.row_indices,
            topo.column_indices,
            topo.offsets,
            topo.column_indices_t,
            topo.offsets_t,
            topo.block_offsets_t,
        )

        # TODO(tgale): Support prefetching forward weights.
        parallel_w, _ = _gather_weights(w, group)
        return stk.ops.sdd(x, parallel_w.t(), topo).data

    @staticmethod
    @internlm_accelerator.amp.custom_bwd
    def backward(ctx, grad):
        x, w = ctx.saved_tensors[:2]
        grad = stk.Matrix(ctx.shape, grad, *ctx.saved_tensors[2:])

        # Start the weight gather asynchronously to overlap with the
        # weight gradient computation.
        parallel_w, handle = _gather_weights(w, ctx.group, async_op=True)
        parallel_dw = None
        if ctx.needs_input_grad[1]:
            parallel_dw = stk.ops.dsd(grad.t(), x)

        # Start the weight gradient reduce scatter to overlap with the
        # data gradient computation.
        handle.wait()
        dw, handle = _scaled_reduce_scatter(parallel_dw, ctx.group, async_op=True)
        dx = None
        if ctx.needs_input_grad[0]:
            dx = stk.ops.dsd(grad, parallel_w)

        # NOTE: Be careful to wait and only cast dw to the output dtype once
        # we've blocked on the asynchronous NCCL operation.
        handle.wait()
        dw = dw.to(w.dtype)
        return dx, dw, None, None


class WeightParallelDsdNn(torch.autograd.Function):
    """
    Weight parallel dsd
    """

    @staticmethod
    @internlm_accelerator.amp.custom_fwd
    def forward(
        ctx, shape, data, row_indices, column_indices, offsets, column_indices_t, offsets_t, block_offsets_t, w, group
    ):
        # [m, k] x [k, n] = [m, n]
        if not data.is_contiguous() or not w.is_contiguous():
            raise ValueError("Expected contiguous 'data' and 'w'.")

        ctx.group = group
        ctx.shape = shape
        ctx.save_for_backward(
            data, row_indices, column_indices, offsets, column_indices_t, offsets_t, block_offsets_t, w
        )
        x = stk.Matrix(shape, data, row_indices, column_indices, offsets, column_indices_t, offsets_t, block_offsets_t)

        # TODO(tgale): Support prefetching forward weights.
        parallel_w, _ = _gather_weights(w, group)
        return stk.ops.dsd(x, parallel_w)

    @staticmethod
    @internlm_accelerator.amp.custom_bwd
    def backward(ctx, grad):
        x = stk.Matrix(ctx.shape, *ctx.saved_tensors[:-1])
        w = ctx.saved_tensors[-1]

        # Start the weight gather asynchronously to overlap with the
        # weight gradient computation.
        parallel_w, handle = _gather_weights(w, ctx.group, async_op=True)
        parallel_dw = None
        if ctx.needs_input_grad[-2]:
            parallel_dw = stk.ops.dsd(x.t(), grad)

        # Start the weight gradient reduce scatter to overlap with the
        # data gradient computation.
        handle.wait()
        dw, handle = _scaled_reduce_scatter(parallel_dw, ctx.group, async_op=True)
        dx = None
        if ctx.needs_input_grad[1]:
            dx = stk.ops.sdd(grad, parallel_w.t(), x)

        # NOTE: Be careful to wait and only cast dw to the output dtype once
        # we've blocked on the asynchronous NCCL operation.
        handle.wait()
        dw = dw.to(w.dtype)
        return None, dx.data, None, None, None, None, None, None, dw, None


class TensorParallelSddNt(torch.autograd.Function):
    """
    Tensor parallel sdd
    """

    @staticmethod
    @internlm_accelerator.amp.custom_fwd
    def forward(ctx, x, w, topo, group):
        # [m, k] x [n, k] = [m, n]
        if not x.is_contiguous() or not w.is_contiguous():
            raise ValueError("Expected contiguous 'x' and 'w'.")

        ctx.group = group
        ctx.shape = topo.shape
        ctx.save_for_backward(
            x,
            w,
            topo.row_indices,
            topo.column_indices,
            topo.offsets,
            topo.column_indices_t,
            topo.offsets_t,
            topo.block_offsets_t,
        )

        return stk.ops.sdd(x, w.t(), topo).data

    @staticmethod
    @internlm_accelerator.amp.custom_bwd
    def backward(ctx, grad):
        x, w = ctx.saved_tensors[:2]
        grad = stk.Matrix(ctx.shape, grad, *ctx.saved_tensors[2:])

        dw = None
        if ctx.needs_input_grad[1]:
            dw = stk.ops.dsd(grad.t(), x)

        dx = None
        if ctx.needs_input_grad[0]:
            dx = stk.ops.dsd(grad, w)
        handle_x = torch.distributed.all_reduce(dx, group=ctx.group, async_op=True)

        # NOTE: Be careful to wait and only cast dw to the output dtype once
        # we've blocked on the asynchronous NCCL operation.
        handle_x.wait()

        dw = dw.to(w.dtype)
        return dx, dw, None, None


class TensorParallelDsdNn(torch.autograd.Function):
    """
    Tensor parallel dsd
    """

    @staticmethod
    @internlm_accelerator.amp.custom_fwd
    def forward(
        ctx, shape, data, row_indices, column_indices, offsets, column_indices_t, offsets_t, block_offsets_t, w, group
    ):
        # [m, k] x [k, n] = [m, n]
        if not data.is_contiguous() or not w.is_contiguous():
            raise ValueError("Expected contiguous 'data' and 'w'.")

        ctx.group = group
        ctx.shape = shape
        ctx.save_for_backward(
            data, row_indices, column_indices, offsets, column_indices_t, offsets_t, block_offsets_t, w
        )
        x = stk.Matrix(shape, data, row_indices, column_indices, offsets, column_indices_t, offsets_t, block_offsets_t)

        out = stk.ops.dsd(x, w)
        torch.distributed.all_reduce(out, group=group)

        return out

    @staticmethod
    @internlm_accelerator.amp.custom_bwd
    def backward(ctx, grad):
        x = stk.Matrix(ctx.shape, *ctx.saved_tensors[:-1])
        w = ctx.saved_tensors[-1]

        dw = None
        if ctx.needs_input_grad[-2]:
            dw = stk.ops.dsd(x.t(), grad)

        dx = None
        if ctx.needs_input_grad[1]:
            dx = stk.ops.sdd(grad, w.t(), x)

        dw = dw.to(w.dtype)
        return None, dx.data, None, None, None, None, None, None, dw, None


# TODO merge two sdd into one kernel
def sdd_nt(a, b, topo, group, parallel_mode):
    parallel_impl = WeightParallelSddNt if parallel_mode == "weight" else TensorParallelSddNt
    return stk.Matrix(
        topo.size(),
        parallel_impl.apply(a, b, topo, group),
        topo.row_indices,
        topo.column_indices,
        topo.offsets,
        topo.column_indices_t,
        topo.offsets_t,
        topo.block_offsets_t,
    )


def dsd_nn(a, b, group, parallel_mode):
    parallel_impl = WeightParallelDsdNn if parallel_mode == "weight" else TensorParallelDsdNn
    return parallel_impl.apply(
        a.size(),
        a.data,
        a.row_indices,
        a.column_indices,
        a.offsets,
        a.column_indices_t,
        a.offsets_t,
        a.block_offsets_t,
        b,
        group,
    )


def act_fn(x1, x2, topo):
    with torch.set_grad_enabled(torch.is_grad_enabled()):
        out = Silu(x1.data, x2.data)
        y = stk.Matrix(
            topo.size(),
            out,
            topo.row_indices,
            topo.column_indices,
            topo.offsets,
            topo.column_indices_t,
            topo.offsets_t,
            topo.block_offsets_t,
        )

        return y


# check dependency
def check_megablock_installed():
    try:
        from megablocks import ops  # noqa # pylint: disable=W0611
    except ModuleNotFoundError:
        print(
            "MegaBlocks not found, please see "
            "https://github.com/stanford-futuredata/megablocks/. "
            "Note that MegaBlocks depends on mosaicml-turbo, which only "
            "supports python 3.10.",
            flush=True,
        )
        sys.exit()


def check_stk_installed():
    try:
        import stk  # noqa # pylint: disable=W0611
    except ModuleNotFoundError:
        print("STK not found: please see https://github.com/stanford-futuredata/stk", flush=True)
        sys.exit()
