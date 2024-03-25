"""
communication for zero parallel
"""

from collections import OrderedDict
from typing import Dict, List, Union

from flash_attn.modules.embedding import ParallelGPT2Embeddings
from torch import nn

from internlm.core.context import ParallelMode
from internlm.core.naive_amp import NaiveAMPModel
from internlm.core.parallel.comm.isp import ISPCommunicator
from internlm.model.modules.embedding import Embedding1D
from internlm.model.modules.mlp import ScaleColumnParallelLinear


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
            if isp_communicator is None or isinstance(
                block, (Embedding1D, ParallelGPT2Embeddings, ScaleColumnParallelLinear)
            ):
                block.register_forward_pre_hook(_pre_forward_hook)
        if isp_communicator:
            isp_communicator.register_prerequisite_for_forward_prefetch_hooks(_pre_forward_hook)

    def get_rank_by_param(self, param) -> int:
        return self._param_to_rank[param]

    def add_bcast_handle(self, rank, handle) -> None:
        self._bcast_handles[rank].append(handle)
