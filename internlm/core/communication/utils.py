# adopted from https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/communication

from collections import OrderedDict
from typing import Dict, List, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn

from internlm.core.communication.isp import ISPCommunicator
from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from internlm.core.naive_amp import NaiveAMPModel
from internlm.model.modules.embedding import Embedding1D
from internlm.model.ops.linear import BaseScaleColumnParallelLinear
from internlm.utils.common import get_current_device

TensorShape = Union[torch.Size, List[int], Tuple[int]]


def send_meta_helper(obj, next_rank, tensor_kwargs):
    send_shape = torch.tensor(obj.size(), **tensor_kwargs)
    send_ndims = torch.tensor(len(obj.size()), **tensor_kwargs)
    dist.send(send_ndims, next_rank)
    dist.send(send_shape, next_rank)


def send_obj_meta(obj, next_rank=None):
    """Sends obj meta information before sending a specific obj.
    Since the recipient must know the shape of the obj in p2p communications,
    meta information of the obj should be sent before communications. This function
    synchronizes with :func:`recv_obj_meta`.

    Args:
        obj (Union[:class:`torch.Tensor`, List[:class:`torch.Tensor`]]): obj to be sent.
        need_meta (bool, optional): If False, meta information won't be sent.
        next_rank (int): The rank of the next member in pipeline parallel group.

    Returns:
        bool: False
    """
    if next_rank is None:
        next_rank = gpc.get_next_global_rank(ParallelMode.PIPELINE)

    tensor_kwargs = {"dtype": torch.long, "device": get_current_device()}
    if isinstance(obj, torch.Tensor):
        send_obj_nums = torch.tensor(1, **tensor_kwargs)
        dist.send(send_obj_nums, next_rank)
        send_meta_helper(obj, next_rank, tensor_kwargs)
    else:
        send_obj_nums = torch.tensor(len(obj), **tensor_kwargs)
        dist.send(send_obj_nums, next_rank)
        for tensor_to_send in obj:
            send_meta_helper(tensor_to_send, next_rank, tensor_kwargs)


def recv_meta_helper(prev_rank, tensor_kwargs):
    recv_ndims = torch.empty((), **tensor_kwargs)
    dist.recv(recv_ndims, prev_rank)
    recv_shape = torch.empty(recv_ndims, **tensor_kwargs)
    dist.recv(recv_shape, prev_rank)
    return recv_shape


def recv_obj_meta(prev_rank=None) -> torch.Size:
    """Receives obj meta information before receiving a specific obj.
    Since the recipient must know the shape of the obj in p2p communications,
    meta information of the obj should be received before communications. This function
    synchronizes with :func:`send_obj_meta`.

    Args:
        obj_shape (Union[:class:`torch.Size`, List[:class:`torch.Size`]]): The shape of the obj to be received.
        prev_rank (int): The rank of the source of the obj.

    Returns:
        Union[:class:`torch.Size`, List[:class:`torch.Size`]]: The shape of the obj to be received.
    """
    if prev_rank is None:
        prev_rank = gpc.get_prev_global_rank(ParallelMode.PIPELINE)

    tensor_kwargs = {"dtype": torch.long, "device": get_current_device()}
    recv_obj_nums = torch.empty((), **tensor_kwargs)
    dist.recv(recv_obj_nums, prev_rank)
    if recv_obj_nums.item() == 1:
        recv_shape = recv_meta_helper(prev_rank, tensor_kwargs)
        obj_shape = torch.Size(recv_shape)
    else:
        obj_shape = []
        for _ in range(recv_obj_nums.item()):
            recv_shape = recv_meta_helper(prev_rank, tensor_kwargs)
            obj_shape.append(torch.Size(recv_shape))

    return obj_shape


def split_tensor_into_1d_equal_chunks(tensor: torch.Tensor, new_buffer=False) -> torch.Tensor:
    """Break a tensor into equal 1D chunks.

    Args:
        tensor (:class:`torch.Tensor`): Tensor to be split before communication.
        new_buffer (bool, optional): Whether to use a new buffer to store sliced tensor.

    Returns:
        :class:`torch.Tensor`: The split tensor
    """
    partition_size = torch.numel(tensor) // gpc.get_world_size(ParallelMode.TENSOR)
    start_index = partition_size * gpc.get_local_rank(ParallelMode.TENSOR)
    end_index = start_index + partition_size
    if new_buffer:
        data = torch.empty(partition_size, dtype=tensor.dtype, device=get_current_device(), requires_grad=False)
        data.copy_(tensor.view(-1)[start_index:end_index])
    else:
        data = tensor.view(-1)[start_index:end_index]
    return data


def gather_split_1d_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Opposite of above function, gather values from model parallel ranks.

    Args:
        tensor (:class:`torch.Tensor`): Tensor to be gathered after communication.
    Returns:
        :class:`torch.Tensor`: The gathered tensor.
    """
    world_size = gpc.get_world_size(ParallelMode.TENSOR)
    numel = torch.numel(tensor)
    numel_gathered = world_size * numel
    gathered = torch.empty(numel_gathered, dtype=tensor.dtype, device=get_current_device(), requires_grad=False)
    chunks = [gathered[i * numel : (i + 1) * numel] for i in range(world_size)]
    dist.all_gather(chunks, tensor, group=gpc.get_group(ParallelMode.TENSOR))
    return gathered


class ParamAsyncBcastHandler:
    """
    Model Partition Handler for overlap broadcast with forward
    """

    def __init__(
        self, zero1_mode: ParallelMode, model: Union[nn.Module, nn.ModuleList], isp_communicator: ISPCommunicator = None
    ) -> None:
        self._block_to_param: Dict[nn.Module, List[nn.Parameter]] = OrderedDict()
        self._param_to_rank: Dict[nn.Parameter, int] = {}
        self._block_to_rank: Dict[nn.Module, int] = {}
        self._bcast_handles: Dict[int, List[dist.Work]] = {}

        zero1_size = gpc.get_world_size(zero1_mode)
        total_param_num = sum(p.numel() for p in model.parameters())
        avg_param_num = total_param_num * 1.0 // zero1_size

        # initialize an empty list for _bcast_handles of each rank
        self._bcast_handles = {rank: [] for rank in range(zero1_size)}

        # just want to share same for loop for ModuleList and Module
        if not isinstance(model, nn.ModuleList):
            model = [model]

        # record the parameters to transformer/embeding/head/norm block
        for _chunk in model:
            if isinstance(_chunk, NaiveAMPModel):
                _chunk = _chunk.model

            for _, children in _chunk.named_children():
                # should be the transformer block definaton in modeling_xxx.py
                if isinstance(children, nn.ModuleList):
                    # record the block that a parameter belongs to
                    for _, block in enumerate(children):
                        # self._block_to_param[f"{name}.{idx}"] = list(block.parameters())
                        self._block_to_param[block] = list(block.parameters())
                else:
                    # record the block that a parameter belongs to
                    # self._block_to_param[name] = list(children.parameters())
                    self._block_to_param[children] = list(children.parameters())

        alloc_num = 0
        rank_to_go = 0

        # process the parameters in block_to_param sequencially,
        # allocate each parameter to a local rank of ParallelMode.ZERO1,
        # NOTE that we do NOT consider following scenarios:
        # 1) whether a parameter is trainable;
        # 2) paramters maybe in different optimizer group
        for block, params in self._block_to_param.items():
            # allocate a model block to a local rank of ParallelMode.ZERO1
            self._block_to_rank[block] = [rank_to_go]
            for p in params:
                alloc_num = alloc_num + p.numel()
                # in this case, allocate the param to next rank if possible
                if alloc_num > avg_param_num * 1.01 and rank_to_go < zero1_size - 1:
                    rank_to_go = rank_to_go + 1
                    alloc_num = 0
                    self._block_to_rank[block].append(rank_to_go)
                # allocate a parameter to a local rank of ParallelMode.ZERO1
                self._param_to_rank[p] = rank_to_go

        # register_forward_pre_hook for transformer/embeding/norm/xxx block
        self._register_sync_parameters_hook(isp_communicator)

    def _register_sync_parameters_hook(self, isp_communicator: ISPCommunicator = None) -> None:
        def _pre_forward_hook(model: nn.Module, *args, **kwargs):  # pylint: disable=W0613
            bcast_handles = []
            # gather all required broadcast hanles into a list
            for rank in self._block_to_rank[model]:
                bcast_handles.extend(self._bcast_handles[rank])
                # need to clear _bcast_handles since they would be processed later
                self._bcast_handles[rank] = []
            # wait all required broadcast handles to be completed
            for handle in bcast_handles:
                handle.wait()

        # register_forward_pre_hook for transformer/embeding/norm/xxx block
        for block, _ in self._block_to_rank.items():
            # TODO: remove special handling for embedding and head layers,
            # instead implement support for weight parallelism of embedding and head layers within the ISP.

            # NOTE: Although the layernorm layer does not have explicit processing,
            # both ISPCommunicator and ParamAsyncBcastHandler handle transformer blocks as granularity,
            # so everything is fine.

            embedding_head_cls = (Embedding1D, BaseScaleColumnParallelLinear)
            if gpc.config.model.use_flash_attn:
                from flash_attn.modules.embedding import ParallelGPT2Embeddings

                embedding_head_cls = (Embedding1D, ParallelGPT2Embeddings, BaseScaleColumnParallelLinear)

            if isp_communicator is None or isinstance(block, embedding_head_cls):
                block.register_forward_pre_hook(_pre_forward_hook)
        if isp_communicator:
            isp_communicator.register_prerequisite_for_forward_prefetch_hooks(_pre_forward_hook)

    def get_rank_by_param(self, param) -> int:
        return self._param_to_rank[param]

    def add_bcast_handle(self, rank, handle) -> None:
        self._bcast_handles[rank].append(handle)
