#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, List, Union

import torch
from torch import distributed as dist
from torch import nn

from internlm.core.context import global_context as gpc
from internlm.core.naive_amp import NaiveAMPModel
from internlm.model.ops.linear import ISPLinear
from internlm.model.utils import all_gather_raw, reduce_scatter_raw
from internlm.utils.common import SchedulerHook, get_current_device


@dataclass
class ISPCommModelConfig:
    """
    model config for isp communicator.
    """

    dtype: torch.dtype = torch.half
    device: torch.device = get_current_device()
    activation_checkpointing: float = 0.0
    module_shapes: Dict[str, torch.Size] = None


class MemoryPool:
    """
    memory pool for isp communication.
    """

    def __init__(
        self,
        model_conf: ISPCommModelConfig,
        with_bias: bool = False,
    ) -> None:
        self._dtype = model_conf.dtype
        self._device = model_conf.device
        self._module_shapes = model_conf.module_shapes

        # due to intern sequence parallel communication overlap, we need
        # **two** memory pools for current block weights and the next block weights.
        self.__all_gather_pool_len = 2
        # memory pool for weight all gather communications.
        self._all_gather_weight_memory_pool = [
            {
                name: torch.zeros(shape, dtype=self._dtype, device=self._device).contiguous()
                for name, shape in self._module_shapes.items()
            }
            for _ in range(self.__all_gather_pool_len)
        ]
        # memory pool for bias all gather communications.
        if not with_bias:
            self._all_gather_bias_memory_pool = None
        else:
            self._all_gather_bias_memory_pool = [
                {
                    name: torch.zeros(shape[0], dtype=self._dtype, device=self._device).contiguous()
                    for name, shape in self._module_shapes.items()
                }
                for _ in range(self.__all_gather_pool_len)
            ]

        # memory pool for reduce scatter communications, allocated lazily.
        self._reduce_scatter_memory_pool = {}
        # memory pool for constant zero tensors, allocated lazily.
        self._zero_const_pool = {}

    def allocate_constant_zero(self, size: tuple) -> torch.Tensor:
        if size not in self._zero_const_pool:
            self._zero_const_pool[size] = torch.zeros(*size, dtype=self._dtype, device=self._device).contiguous()

        return self._zero_const_pool[size]

    def allocate_all_gather_memory(self, block_index: int, module_name: str, is_bias: bool = False) -> torch.Tensor:
        # TODO: should we trace the usage of each memory block to avoid reusing
        # same memory block, which will hides some potential bugs.
        if not is_bias:
            mem = self._all_gather_weight_memory_pool[block_index % 2][module_name]
        else:
            enable_bias = self._all_gather_bias_memory_pool is not None
            assert enable_bias, "memory bool for bias is disabled."

            mem = self._all_gather_bias_memory_pool[block_index % 2][module_name]

        return mem

    def allocate_reduce_scatter_memory(self, key: tuple) -> torch.Tensor:
        # if key not in dict
        if key not in self._reduce_scatter_memory_pool:
            self._reduce_scatter_memory_pool[key] = []

        for index, mem_item in enumerate(self._reduce_scatter_memory_pool[key]):
            if mem_item.idle is True:
                self._reduce_scatter_memory_pool[key][index].idle = False
                return self._reduce_scatter_memory_pool[key][index]

        # if the memory pool is all used
        new_item = torch.zeros(
            key,
            dtype=self._dtype,
            device=self._device,
        ).contiguous()
        setattr(new_item, "idle", False)
        setattr(new_item, "index", len(self._reduce_scatter_memory_pool[key]))
        self._reduce_scatter_memory_pool[key].append(new_item)

        return new_item

    def free_reduce_scatter_memory(self, key, index):
        self._reduce_scatter_memory_pool[key][index].idle = True

    def reset_lazy_pools(self) -> None:
        # Should memory pool re-allocate all gather memory for every interation?
        # Currently, it just clear the memory pool for reduce scatter communication.
        self._zero_const_pool = {}
        self._reduce_scatter_memory_pool = {}


class ISPOverlapState:
    """
    Overlap state for isp.
    """

    def __init__(self) -> None:
        self.num_blocks: int = 0
        self.ckpt_block_num: int = 0
        self.isp_outs: List[nn.Module] = []
        self.isp_modules: List[nn.Module] = []
        self.index_to_isp_modules: Dict[int, nn.Module] = {}
        self.index_to_block: Dict[int, nn.Module] = {}
        self.module_to_index: Dict[nn.Module, int] = {}
        self.weight_global_handle: Dict[str, Any] = {}
        self.weight_global_output: Dict[str, torch.Tensor] = {}
        self.bias_global_handle: Dict[str, Any] = {}
        self.bias_global_output: Dict[str, torch.Tensor] = {}


class ISPCommunicator:
    """
    ISP Communicator for managing the all-gather and reduce_scatter of Intern Sequence Parallel.
    """

    def __init__(
        self,
        model: Union[nn.Module, nn.ModuleList],
        model_conf: ISPCommModelConfig,
        overlap: bool = False,
        enable_memory_pool: bool = False,
        process_group: dist.ProcessGroup = None,
    ) -> None:
        self.process_group = process_group
        self.overlap = overlap
        self.enable_memory_pool = overlap and enable_memory_pool
        self.model_conf = model_conf
        self.is_forward = True
        self.reduce_scatter_handlers = {}
        self._module_shapes = {}
        self._forward_prefetch_prerequisites = []

        # real overlap state for each chunk.
        self._overlap_states: Dict[int, ISPOverlapState] = {}

        # inner interface variables of overlap state.
        self._num_blocks = None
        self._ckpt_block_num = None
        self._isp_outs = None
        self._isp_modules = None
        # key: isp module; value: module global all-gather op handle
        self._weight_global_handle = None
        # key: isp module; value: module bias global all-gather op handle
        self._bias_global_handle = None
        # key: isp module; value: module global weight after all-gather op
        self._weight_global_output = None
        # key: isp module; value: module bias global weight after all-gather op
        self._bias_global_output = None
        # key: isp module; value: transformer block index
        self._module_to_index = None
        # key: transformer block index; value: isp modules
        self._index_to_isp_modules = None
        # key: transformer block index; value: transformer block
        self._index_to_block = None

        # init overlap states if necessary.
        if self.overlap:
            # just want to share same for loop for modulelist and module.
            model = model if isinstance(model, nn.ModuleList) else [model]
            # build overlap states for every chunk.
            for chunk_id, chunk in enumerate(model):
                if isinstance(chunk, NaiveAMPModel):
                    chunk = chunk.model
                self._parse_model_structure(chunk_id, chunk)
            # register overlap hooks for every chunk.
            for chunk_id in range(len(model)):
                self.switch_current_model_chunk(chunk_id)
                self._register_sync_parameters_hook()
            # switch to chunk 0 at first.
            self.switch_current_model_chunk(0)
            self.model_conf.module_shapes = self._module_shapes

            # init memory pool if necessary.
            if self.enable_memory_pool:
                self.memory_pool = MemoryPool(self.model_conf, with_bias=True)
            else:
                self.memory_pool = None

    def _parse_model_structure(self, cid: int, model: nn.Module) -> None:
        self._overlap_states[cid] = ISPOverlapState()

        # Important: only works for llama-class models
        for _, children in model.named_children():
            if isinstance(children, nn.ModuleList):
                self._overlap_states[cid].ckpt_block_num = int(self.model_conf.activation_checkpointing * len(children))

                for idx, block in enumerate(children):
                    self._overlap_states[cid].index_to_isp_modules[idx] = []
                    self._overlap_states[cid].index_to_block[idx] = block
                    for sub_name, sub in block.named_children():
                        for name, child in sub.named_children():
                            if name in ["out_proj", "wo"]:
                                self._overlap_states[cid].isp_outs.append(child)
                                self._overlap_states[cid].module_to_index[child] = idx
                            if isinstance(child, ISPLinear):
                                if name not in self._module_shapes:
                                    origin_shape = tuple(
                                        [child.weight.shape[0] * gpc.weight_parallel_size]
                                        + list(child.weight.shape[1:])
                                    )
                                    self._module_shapes[name] = torch.Size(origin_shape)
                                self._overlap_states[cid].module_to_index[child] = idx
                                self._overlap_states[cid].isp_modules.append(child)
                                self._overlap_states[cid].index_to_isp_modules[idx].append(child)

                                setattr(child, "isp_name", name)

                                full_name = f"{cid}.{idx}.{sub_name}.{name}"
                                setattr(
                                    child.weight,
                                    "isp_reduce_scatter_name",
                                    f"{full_name}.weight",
                                )
                                if child.bias is not None:
                                    setattr(
                                        child.bias,
                                        "isp_reduce_scatter_name",
                                        f"{full_name}.bias",
                                    )

        self._overlap_states[cid].num_blocks = len(self._overlap_states[cid].index_to_isp_modules)

    def _all_gather_module_weight(self, module):
        with_bias = module.bias is not None
        block_index = self._module_to_index[module]

        # prepare memory pool allocator for weight and bias.
        if self.enable_memory_pool:
            weight_memory_pool_allocator = partial(
                self.memory_pool.allocate_all_gather_memory,
                block_index,
                module.isp_name,
            )
        else:
            weight_memory_pool_allocator = None

        if self.enable_memory_pool and with_bias:
            bias_memory_pool_allocator = partial(
                self.memory_pool.allocate_all_gather_memory,
                block_index,
                module.isp_name,
                is_bias=True,
            )
        else:
            bias_memory_pool_allocator = None

        # submit the all-gather communication for weight and bias.
        if with_bias:
            bias_output, bias_handle = all_gather_raw(
                module.bias,
                self.process_group,
                async_op=True,
                memory_pool_allocator=bias_memory_pool_allocator,
            )
            self._bias_global_handle[module] = bias_handle
            self._bias_global_output[module] = bias_output

        weight_output, weight_handle = all_gather_raw(
            module.weight,
            self.process_group,
            async_op=True,
            memory_pool_allocator=weight_memory_pool_allocator,
        )
        self._weight_global_handle[module] = weight_handle
        self._weight_global_output[module] = weight_output

    def _all_gather_block_weight(self, block_index: int):
        block = self._index_to_block[block_index]

        # wait for prerequisite conditions
        if self.is_forward:
            for callback in self._forward_prefetch_prerequisites:
                callback(block)

        # prefetch parameters for all isp modules of the block
        for module in self._index_to_isp_modules[block_index]:
            self._all_gather_module_weight(module)

    def _wait_handle(self, module):
        handle = self._weight_global_handle[module]
        if handle is not None:
            handle.wait()

        if module.bias is None:
            return

        bias_handle = self._bias_global_handle[module]
        if bias_handle is not None:
            bias_handle.wait()

    def _clear_handle(self, module):
        if module in self._weight_global_handle:
            del self._weight_global_handle[module]
        if module in self._bias_global_handle:
            del self._bias_global_handle[module]

    def _clear_weight(self, module):
        if module in self._weight_global_output:
            del self._weight_global_output[module]
        if module in self._bias_global_output:
            del self._bias_global_output[module]

    def _pre_forward_hook_for_first_block(self, *args):  # pylint: disable=W0613
        """
        prefetch weight for block 0 before forward.
        """
        if self.is_forward is True:
            self._all_gather_block_weight(0)

    def _pre_forward_hook_for_last_ckpt_block(self, *args):  # pylint: disable=W0613
        if self.is_forward is False:
            self._all_gather_block_weight(self._ckpt_block_num - 1)

    def _pre_forward_hook_for_out_proj(self, module: nn.Module, *args):  # pylint: disable=W0613
        block_index = self._module_to_index[module]

        if (block_index - 1 < self._ckpt_block_num) and self.is_forward is False:
            if block_index - 1 >= 0:
                self._all_gather_block_weight(block_index - 1)
        else:
            # start the all-gather for next block
            if block_index + 1 < self._num_blocks:
                self._all_gather_block_weight(block_index + 1)

    def _pre_forward_hook_for_module(self, module: nn.Module, *args):  # pylint: disable=W0613
        if module not in self._weight_global_handle:
            self._all_gather_module_weight(module)

        self._wait_handle(module)

    def _post_forward_hook_for_module(self, module: nn.Module, *args):  # pylint: disable=W0613
        self._clear_handle(module)
        if not ((self._module_to_index[module] < self._ckpt_block_num) and self.is_forward is False):
            self._clear_weight(module)

    def _pre_backward_hook_for_module(self, module: nn.Module, *args):  # pylint: disable=W0613
        # wait handle for current module
        if module not in self._weight_global_handle:
            self._all_gather_module_weight(module)

        self._wait_handle(module)

        # start the all-gather for next module
        module_index = self._isp_modules.index(module)
        if module_index - 1 >= 0:
            next_module = self._isp_modules[module_index - 1]
            if self._module_to_index[next_module] >= self._ckpt_block_num:
                self._all_gather_module_weight(next_module)

    def _post_backward_hook_for_module(self, module, *args):  # pylint: disable=W0613
        self._clear_handle(module)
        self._clear_weight(module)

    def _register_sync_parameters_hook(self) -> None:
        """
        register forward hooks and backward hooks for isp modules.
        """
        # register forward hooks
        # 1. register pre_forward_hook @block_0 to prefetch for block 0
        # 2. register pre_forward_hook @block_(ckpt_block_num-1) to prefetch for the last ckpt block
        # 3. register pre_forward_hook @out_proj module to prefetch for next block,
        #    notice that next block's all_gather op should be after current block's all_to_all op
        # 4. register pre_forward_hook @isp_module to wait handle for current module
        # 5. register post_forward_hook @isp_module to release resource
        self._index_to_block[0].register_forward_pre_hook(self._pre_forward_hook_for_first_block)

        if self._ckpt_block_num >= 1:
            self._index_to_block[self._ckpt_block_num - 1].register_forward_pre_hook(
                self._pre_forward_hook_for_last_ckpt_block
            )

        for out_proj in self._isp_outs:
            out_proj.register_forward_pre_hook(self._pre_forward_hook_for_out_proj)

        for module in self._isp_modules:
            module.register_forward_pre_hook(self._pre_forward_hook_for_module)
            module.register_forward_hook(self._post_forward_hook_for_module)

        # register backward hooks
        # 1. register pre_backward_hook @isp_module to wait handle for current module and to prefetch for next module
        # 2. register post_backward_hook @isp_module to release resource
        if self._ckpt_block_num < self._num_blocks:
            for module in self._isp_modules:
                module.register_full_backward_pre_hook(self._pre_backward_hook_for_module)

        for module in self._isp_modules:
            module.register_full_backward_hook(self._post_backward_hook_for_module)

    def _get_constant_zero(self, size: tuple) -> torch.Tensor:
        if self.enable_memory_pool:
            return self.memory_pool.allocate_constant_zero(size)
        else:
            return torch.zeros(
                *size,
                dtype=self.model_conf.dtype,
                device=self.model_conf.device,
            ).contiguous()

    def switch_current_model_chunk(self, chunk_id: int) -> None:
        self._isp_outs = self._overlap_states[chunk_id].isp_outs
        self._isp_modules = self._overlap_states[chunk_id].isp_modules
        self._weight_global_handle = self._overlap_states[chunk_id].weight_global_handle
        self._bias_global_handle = self._overlap_states[chunk_id].bias_global_handle
        self._weight_global_output = self._overlap_states[chunk_id].weight_global_output
        self._bias_global_output = self._overlap_states[chunk_id].bias_global_output
        self._module_to_index = self._overlap_states[chunk_id].module_to_index
        self._index_to_isp_modules = self._overlap_states[chunk_id].index_to_isp_modules
        self._index_to_block = self._overlap_states[chunk_id].index_to_block
        self._ckpt_block_num = self._overlap_states[chunk_id].ckpt_block_num
        self._num_blocks = self._overlap_states[chunk_id].num_blocks

    def register_prerequisite_for_forward_prefetch_hooks(self, prerequisite_func: Callable) -> None:
        """
        Registers a callback function that specifies a prerequisite condition for
        prefetching parameters before forward computation.

        This method allows users to define custom logic that must be satisfied before
        parameters are fetched for the next forward pass. This can be useful for
        implementing complex parameter update strategies or for coordinating
        parameter access with other system components.

        Args:
            prerequisite_func (Callable): A callable that represents the prerequisite
                                    condition. This function will be invoked before
                                    the parameters are prefetched, and its return value
                                    will determine whether the prefetching should proceed.

        Returns:
            None: This method does not return any value.

        Raises:
            TypeError: If the provided 'prerequisite_func' is not callable.
        """
        if not callable(prerequisite_func):
            raise TypeError("The provided prerequisite function must be callable.")

        self._forward_prefetch_prerequisites.append(prerequisite_func)

    # communication operation interfaces

    def all_gather(self, tensor: torch.Tensor, module: nn.Module, is_bias: bool = False):
        if dist.get_world_size(self.process_group) <= 1:
            return tensor

        if not self.overlap:
            result, _ = all_gather_raw(tensor, self.process_group, async_op=False)
        elif is_bias:
            result = self._bias_global_output[module]
        else:
            result = self._weight_global_output[module]

        return result

    def reduce_scatter(
        self,
        tensor: torch.Tensor,
        model: nn.Module,
        op: dist.ReduceOp,
        is_bias: bool = False,
    ):
        if dist.get_world_size(self.process_group) <= 1:
            return tensor, None

        if not self.overlap:
            result, handle = reduce_scatter_raw(tensor, self.process_group, op=op, async_op=True)
        else:
            if is_bias:
                assert hasattr(model.bias, "isp_reduce_scatter_name")
                key = getattr(model.bias, "isp_reduce_scatter_name")
            else:
                assert hasattr(model.weight, "isp_reduce_scatter_name")
                key = getattr(model.weight, "isp_reduce_scatter_name")

            self.reduce_scatter_handlers[key] = reduce_scatter_raw(
                tensor,
                self.process_group,
                op=op,
                async_op=True,
                memory_pool_allocator=(
                    self.memory_pool.allocate_reduce_scatter_memory if self.enable_memory_pool else None
                ),
            )

            result, handle = (
                self._get_constant_zero(
                    (
                        tensor.shape[0] // dist.get_world_size(self.process_group),
                        *tensor.shape[1:],
                    )
                ),
                None,
            )

        return result, handle


class ISPCommunicatorSchedulerHook(SchedulerHook):
    """
    SchedulerHook for isp overlap handler
    """

    def __init__(self, overlap_handler: ISPCommunicator, zero_optim) -> None:
        self._isp_communicator = overlap_handler
        self._zero_optim = zero_optim

    def before_forward(self, scheduler, inputs) -> None:
        self._isp_communicator.is_forward = True
        # switch model chunk before forward
        chunk_id = 0 if gpc.virtual_pipeline_parallel_rank is None else gpc.virtual_pipeline_parallel_rank
        self._isp_communicator.switch_current_model_chunk(chunk_id)

    def after_forward(self, scheduler, outputs) -> None:
        pass

    def before_criterion(self, scheduler, outputs, label) -> None:
        pass

    def after_criterion(self, scheduler, loss) -> None:
        pass

    def before_backward(self, scheduler, outputs, outputs_grad) -> None:
        self._isp_communicator.is_forward = False
        # switch model chunk before backward
        chunk_id = 0 if gpc.virtual_pipeline_parallel_rank is None else gpc.virtual_pipeline_parallel_rank
        self._isp_communicator.switch_current_model_chunk(chunk_id)

    def after_backward(self, scheduler, inputs_grad) -> None:
        # accumulate left gradients in last bucket after backward.
        self._zero_optim.accumulate_left_grads_after_backward()
        # reset lazy memory pools for reduce scatter after every micro step.
        if self._isp_communicator and self._isp_communicator.enable_memory_pool:
            self._isp_communicator.memory_pool.reset_lazy_pools()

    def post_helper_func(self, scheduler, outputs, label) -> None:
        pass
