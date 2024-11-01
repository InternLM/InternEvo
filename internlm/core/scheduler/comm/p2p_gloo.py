#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from typing import List, Tuple, Union

import torch
import torch.distributed as dist

from internlm.accelerator import get_accelerator
from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc

from .utils import gather_split_1d_tensor, split_tensor_into_1d_equal_chunks

TensorShape = Union[torch.Size, List[int], Tuple[int]]
internlm_accelerator = get_accelerator()


def create_recv_buffer_with_shapes_by_gloo(recv_shapes, dtype, scatter_gather_tensors):
    from .p2p import _get_tensor_shape
    if isinstance(recv_shapes, torch.Size):
        recv_chunk_shape, recv_split = _get_tensor_shape(recv_shapes, scatter_gather_tensors)
        buffer_recv = torch.empty(recv_chunk_shape, requires_grad=True, device="cpu", dtype=dtype)
        return buffer_recv, recv_split
    buffer_recv = []
    for recv_shape in recv_shapes:
        recv_chunk_shape, recv_split = _get_tensor_shape(recv_shape, scatter_gather_tensors)
        tensor_recv = torch.empty(recv_chunk_shape, requires_grad=True, device="cpu", dtype=dtype)
        buffer_recv.append(tensor_recv)
    return buffer_recv, recv_split


def process_object_to_send_by_gloo(object_send, scatter_gather_tensors):
    from .p2p import _get_tensor_shape
    if isinstance(object_send, torch.Tensor):
        send_split = _get_tensor_shape(object_send.shape, scatter_gather_tensors)[1]
        if send_split:
            object_send = split_tensor_into_1d_equal_chunks(object_send)
        object_send = object_send.cpu()
        return object_send

    object_send_list = []
    for tensor_send in object_send:
        send_split = _get_tensor_shape(tensor_send.shape, scatter_gather_tensors)[1]
        if send_split:
            object_send_list.append(split_tensor_into_1d_equal_chunks(tensor_send).cpu())
        else:
            object_send_list.append(tensor_send.cpu())
    object_send = tuple(object_send_list)

    return object_send

def _communicate_by_gloo(
        object_send_next: Union[torch.Tensor, List[torch.Tensor]] = None,
        object_send_prev: Union[torch.Tensor, List[torch.Tensor]] = None,
        recv_prev: bool = False,
        recv_next: bool = False,
        recv_prev_shape: Union[torch.Size, List[torch.Size]] = None,
        recv_next_shape: Union[torch.Size, List[torch.Size]] = None,
        prev_rank: int = None,
        next_rank: int = None,
        dtype: torch.dtype = None,
        scatter_gather_tensors: bool = False,
) -> Tuple[Union[torch.Tensor, List[torch.Tensor]]]:
    """
    Adapted from megatron.p2p_communication.
    Communicate tensors between stages. Used as helper method in other
    communication methods that are used in pipeline schedule.
    Takes the following arguments:
        object_send_next (Union[:class:`torch.Tensor`, List[:class:`torch.Tensor`]]): tensor to send to next rank
        (no tensor sent if set to None).
        object_send_prev (Union[:class:`torch.Tensor`, List[:class:`torch.Tensor`]]): tensor to send to prev rank
        (no tensor sent if set to None).
        recv_prev (bool): boolean for whether tensor should be received from
                   previous rank.
        recv_next (bool): boolean for whether tensor should be received from
                   next rank.
        recv_prev_shape (Union[:class:`torch.Size`, List[:class:`torch.Size`]]): shape of the tensor to be received
        from the previous stage, defualts to None.
        recv_next_shape (Union[:class:`torch.Size`, List[:class:`torch.Size`]]): shape of the tensor to be received
        from the next stage, defualts to None.
        prev_rank (int): the rank of the previous pipeline stage, defualts to None,
        next_rank (int): the rank of the next pipeline stage, defualts to None,
        dtype (torch.dtype): data type of intermediate buffers, defaults to None
        scatter_gather_tensors (bool): whether to scatter and gather tensor between pipeline stages, defaults to False

    Returns:
        Tuple[Union[:class:`torch.Tensor`, List[:class:`torch.Tensor`]]]: returns tensor_recv_prev, tensor_recv_next
    """

    from .p2p import filling_ops_queue

    # Create placeholder tensors for receive in forward and backward directions
    # if needed.
    tensor_recv_prev = None
    tensor_recv_next = None

    if recv_prev:
        assert recv_prev_shape is not None
        tensor_recv_prev, recv_prev_split = create_recv_buffer_with_shapes_by_gloo(
            recv_prev_shape, dtype, scatter_gather_tensors
        )

    if recv_next:
        assert recv_next_shape is not None
        tensor_recv_next, recv_next_split = create_recv_buffer_with_shapes_by_gloo(
            recv_next_shape, dtype, scatter_gather_tensors
        )

    if object_send_prev is not None or recv_prev:
        if prev_rank is None:
            prev_rank = gpc.get_prev_global_rank(ParallelMode.PIPELINE)

    if object_send_next is not None or recv_next:
        if next_rank is None:
            next_rank = gpc.get_next_global_rank(ParallelMode.PIPELINE)

    if object_send_prev is not None:
        object_send_prev = process_object_to_send_by_gloo(object_send_prev, scatter_gather_tensors)

    if object_send_next is not None:
        object_send_next = process_object_to_send_by_gloo(object_send_next, scatter_gather_tensors)

    ops = []
    if object_send_prev is not None:
        filling_ops_queue(object_send_prev, dist.isend, prev_rank, ops)

    if tensor_recv_prev is not None:
        filling_ops_queue(tensor_recv_prev, dist.irecv, prev_rank, ops)

    if tensor_recv_next is not None:
        filling_ops_queue(tensor_recv_next, dist.irecv, next_rank, ops)

    if object_send_next is not None:
        filling_ops_queue(object_send_next, dist.isend, next_rank, ops)

    if len(ops) > 0:
        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()
    # To protect against race condition when using batch_isend_irecv().
    internlm_accelerator.synchronize()

    if tensor_recv_prev is not None:
        tensor_recv_prev = tensor_recv_prev.cuda()
    if tensor_recv_next is not None:
        tensor_recv_next = tensor_recv_next.cuda()
    if recv_prev and recv_prev_split:
        if isinstance(tensor_recv_prev, torch.Tensor):
            tensor_recv_prev = gather_split_1d_tensor(tensor_recv_prev).view(recv_prev_shape).requires_grad_()
        else:
            for index in range(len(tensor_recv_prev)):
                tensor_recv_prev[index] = (
                    gather_split_1d_tensor(tensor_recv_prev[index]).view(recv_prev_shape[index]).requires_grad_()
                )
                tensor_recv_prev[index] = tensor_recv_prev[index].cuda()

    if recv_next and recv_next_split:
        if isinstance(tensor_recv_next, torch.Tensor):
            tensor_recv_next = gather_split_1d_tensor(tensor_recv_next).view(recv_next_shape).requires_grad_()
        else:
            for index in range(len(tensor_recv_next)):
                tensor_recv_next[index] = (
                    gather_split_1d_tensor(tensor_recv_next[index]).view(recv_next_shape[index]).requires_grad_()
                )
                tensor_recv_next[index] = tensor_recv_next[index].cuda()


    return tensor_recv_prev, tensor_recv_next
