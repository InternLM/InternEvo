"""
communication for zero parallel
"""

from collections import OrderedDict
from typing import Dict, List, Union

from torch import distributed as dist
from torch import nn

from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from internlm.core.naive_amp import unwrap_naive_amp
from internlm.core.parallel.comm.isp import ISPCommunicator
from internlm.model.modules.embedding import Embedding1D
from internlm.model.modules.linear import ScaleColumnParallelLinear
from internlm.solver.optimizer.utils import flatten


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
        self._block_to_name: Dict[nn.Module, str] = {}

        zero1_size = gpc.get_world_size(zero1_mode)
        total_param_num = sum(p.numel() for p in model.parameters())
        avg_param_num = total_param_num * 1.0 // zero1_size

        # initialize an empty list for _bcast_handles of each rank
        self._bcast_handles = {rank: [] for rank in range(zero1_size)}
        # initialize an empty list for _allgather_handles
        self._block_allgather_handles = {}
        self._block_master_params = {}
        self._block_working_params = {}
        self._block_gathered_params = {}
        self._block_allgather_order = {}

        # record the parameters to transformer/embeding/head/norm block
        for _chunk in unwrap_naive_amp(model):
            for name, children in _chunk.named_children():
                # should be the transformer block definaton in modeling_xxx.py
                if isinstance(children, nn.ModuleList):
                    # record the block that a parameter belongs to
                    for idx, block in enumerate(children):
                        block_name = name + f"_{idx}"
                        # self._block_to_param[f"{name}.{idx}"] = list(block.parameters())
                        self._block_to_param[block] = list(block.parameters())
                        self._block_to_name[block] = block_name
                else:
                    # record the block that a parameter belongs to
                    # self._block_to_param[name] = list(children.parameters())
                    self._block_to_param[children] = list(children.parameters())
                    self._block_to_name[children] = name

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

        for block_name in self._block_to_name.values():
            self._block_allgather_handles[block_name] = None
            self._block_master_params[block_name] = []
            self._block_working_params[block_name] = []
            self._block_gathered_params[block_name] = []
            self._block_allgather_order[block_name] = -1

        # register_forward_pre_hook for transformer/embeding/norm/xxx block
        if (
            "use_split_tensor_optim" not in gpc.config.hybrid_zero_optimizer
            or not gpc.config.hybrid_zero_optimizer.use_split_tensor_optim
        ):
            self._register_sync_parameters_hook(isp_communicator)
        else:
            self._register_sync_parameters_hook_v2(isp_communicator)

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
            if isp_communicator is None or isinstance(block, (Embedding1D, ScaleColumnParallelLinear)):
                block.register_forward_pre_hook(_pre_forward_hook)
        if isp_communicator:
            isp_communicator.register_prerequisite_for_forward_prefetch_hooks(_pre_forward_hook)

    def _register_sync_parameters_hook_v2(self, isp_communicator: ISPCommunicator = None) -> None:
        def _pre_forward_hook(model: nn.Module, *args, **kwargs):  # pylint: disable=W0613
            # For each block, wait corresponding all_gather handle to be completed
            # For each all_gather handle, several consecutive blocks may be involved
            # In this case only the first block of the handle needs to deal with it
            block_name = self._block_to_name[model]
            if self._block_allgather_order[block_name] == 1:
                if self._block_allgather_handles[block_name] is None:
                    return
                self._block_allgather_handles[block_name].wait()

                # reorganize gatherd params to update working param
                # [[A1, B1], [A2, B2]] -> [[A1.reshape, A2.reshape], [B1.reshape, B2.reshape]]
                block_master_params = self._block_master_params[block_name]
                gathered_params = self._block_gathered_params[block_name]
                all_splited_param_list = []
                offset = 0
                for p in block_master_params:
                    param_size = p.numel()
                    all_splited_param = []
                    for all_params in gathered_params:
                        split_params = all_params[offset : offset + param_size].reshape(p.shape)
                        all_splited_param.append(split_params)
                    offset += param_size
                    all_splited_param_list.append(all_splited_param)
                assert len(all_splited_param_list) == len(self._block_working_params[block_name])
                # Update working parameters
                for working_param, all_splited_param in zip(
                    self._block_working_params[block_name], all_splited_param_list
                ):
                    working_param.data.copy_(flatten(all_splited_param)[: working_param.numel()].view_as(working_param))

                self._block_allgather_handles[block_name] = None
                self._block_gathered_params[block_name] = []
                self._block_working_params[block_name] = []

        # register_forward_pre_hook for transformer/embeding/norm/xxx block
        for block, _ in self._block_to_rank.items():
            # TODO: remove special handling for embedding and head layers,
            # instead implement support for weight parallelism of embedding and head layers within the ISP.

            # NOTE: Although the layernorm layer does not have explicit processing,
            # both ISPCommunicator and ParamAsyncBcastHandler handle transformer blocks as granularity,
            # so everything is fine.
            if isp_communicator is None or isinstance(block, (Embedding1D, ScaleColumnParallelLinear)):
                block.register_forward_pre_hook(_pre_forward_hook)
        if isp_communicator:
            isp_communicator.register_prerequisite_for_forward_prefetch_hooks(_pre_forward_hook)

    def get_rank_by_param(self, param) -> int:
        return self._param_to_rank[param]

    def add_bcast_handle(self, rank, handle) -> None:
        self._bcast_handles[rank].append(handle)

    def add_allgather_handle(self, handle, master_param, working_param, gatherd_param, block_name) -> None:
        assert self._block_allgather_handles[block_name] is None
        self._block_allgather_handles[block_name] = handle
        self._block_master_params[block_name] = master_param
        self._block_working_params[block_name] = working_param
        self._block_gathered_params[block_name] = gatherd_param
        self._block_allgather_order[block_name] = 1
