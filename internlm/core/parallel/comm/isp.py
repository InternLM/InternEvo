#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
communication for isp parallel.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import distributed as dist
from torch import nn
from torch._prims_common import make_contiguous_strides_for

from internlm.accelerator import get_accelerator
from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from internlm.core.naive_amp import unwrap_naive_amp
from internlm.core.parallel.comm.attn_offload import get_offload_manager
from internlm.core.parallel.comm.utils import (
    DUMMY_HANDLE_CONST,
    AsyncCommHandle,
    _gather,
    _split,
    all_gather_raw,
    apply_to_tensors_only,
    expandKVPacked,
    reduce_scatter_raw,
)
from internlm.model.modules.embedding import Embedding1D
from internlm.model.modules.linear import ParallelLinearWithCommExt
from internlm.model.modules.utils import is_moe_param
from internlm.utils.common import SchedulerHook, get_current_device
from internlm.utils.utils import (
    CuSeqlenType,
    QKVPackType,
    TensorParallelMode,
    check_attention_argument,
    params_dispatch_with_condition,
)

internlm_accelerator = get_accelerator()


# not really useful, only for code hint.
class WPCommunicator(ABC):
    """
    Common communicator interface for weight parallel
    """

    @abstractmethod
    def communication_mode(self) -> str:
        """
        communication mode of communictor
        """
        pass

    @abstractmethod
    def weight_hook(self, tensor: torch.Tensor, async_op: bool = False, **kwargs) -> torch.Tensor:
        """
        communication for weight when forward/backward.
        """
        pass

    @abstractmethod
    def grad_hook(self, tensor: torch.Tensor, async_op: bool = False, **kwargs) -> Tuple[torch.Tensor, AsyncCommHandle]:
        """
        communication for grad when backward.
        """
        pass


class HeadWeightParallelCommunicator(WPCommunicator):
    """
    Weight parallel communicator for Head module.
    """

    def __init__(
        self,
        weight_process_group: dist.ProcessGroup = None,
        seq_process_group: dist.ProcessGroup = None,
        retain_out_sharded: bool = True,
    ) -> None:
        self.weight_process_group = weight_process_group
        self.seq_process_group = seq_process_group
        self._seq_parallel_mode = ParallelMode.TENSOR
        self._seq_world_size = gpc.get_world_size(ParallelMode.TENSOR)
        self._retain_out_sharded = retain_out_sharded
        self._seq_dim = 1
        self._hid_dim = 2

    def communication_mode(self) -> str:
        return "wp"

    def weight_hook(
        self,
        tensor: torch.Tensor,
        async_op: bool = False,
        module: nn.Module = None,  # pylint: disable=W0613
        is_bias: bool = False,  # pylint: disable=W0613
    ) -> torch.Tensor:
        if dist.get_world_size(self.weight_process_group) <= 1:
            return tensor

        result, _ = all_gather_raw(tensor, self.weight_process_group, async_op=async_op)
        return result

    def grad_hook(
        self,
        tensor: torch.Tensor,
        async_op: bool = False,
        module: nn.Module = None,  # pylint: disable=W0613
        reduce_op: dist.ReduceOp = dist.ReduceOp.AVG,
        is_bias: bool = False,  # pylint: disable=W0613
    ) -> Tuple[torch.Tensor, AsyncCommHandle]:
        if dist.get_world_size(self.weight_process_group) <= 1:
            return tensor, DUMMY_HANDLE_CONST

        result, handle = reduce_scatter_raw(tensor, self.weight_process_group, op=reduce_op, async_op=async_op)
        return result, handle

        # rewrite grad_output communication hook

    def grad_output_hook(
        self, grad_output: torch.Tensor, async_op: bool = False  # pylint: disable=W0613
    ) -> Tuple[torch.Tensor, AsyncCommHandle]:
        """
        split grad_output if retain_out_sharded is False.
        """

        # gather hidden_states dim and split seq dim when parallel_output is True
        if self._retain_out_sharded:
            if self._seq_world_size <= 1:
                return grad_output, DUMMY_HANDLE_CONST
            else:
                _seq_splited_list = [
                    t.contiguous() for t in torch.tensor_split(grad_output, self._seq_world_size, dim=self._seq_dim)
                ]
                output_list = [torch.empty_like(_seq_splited_list[0]) for _ in range(self._seq_world_size)]
                dist.all_to_all(output_list, _seq_splited_list, group=self.seq_process_group)
                grad_output = torch.cat(output_list, dim=self._hid_dim).contiguous()
                return grad_output, DUMMY_HANDLE_CONST
        # split seq dim when parallel_output is False
        else:
            if self._seq_world_size <= 1:
                return grad_output, DUMMY_HANDLE_CONST
            else:
                return _split(grad_output, parallel_mode=self._seq_parallel_mode, dim=self._seq_dim), DUMMY_HANDLE_CONST

    # rewrite ouput communication hook
    def output_hook(
        self, output: torch.Tensor, async_op: bool = False  # pylint: disable=W0613
    ) -> Tuple[torch.Tensor, AsyncCommHandle]:
        """
        all gather output for head layer if retain_out_sharded is False.
        """

        # gather seq dim and split hidden_states dim when parallel_output is True
        if self._retain_out_sharded:
            if self._seq_world_size <= 1:
                return output, DUMMY_HANDLE_CONST
            else:
                _hid_splited_list = [
                    t.contiguous() for t in torch.tensor_split(output, self._seq_world_size, dim=self._hid_dim)
                ]
                output_list = [torch.empty_like(_hid_splited_list[0]) for _ in range(self._seq_world_size)]
                dist.all_to_all(output_list, _hid_splited_list, group=self.seq_process_group)
                output = torch.cat(output_list, dim=self._seq_dim).contiguous()
                return output, DUMMY_HANDLE_CONST
        # gather seq dim when parallel_output is False
        else:
            if self._seq_world_size <= 1:
                return output, DUMMY_HANDLE_CONST
            else:
                return _gather(output, parallel_mode=self._seq_parallel_mode, dim=self._seq_dim), DUMMY_HANDLE_CONST


class EmbeddingWeightParallelCommunicator:
    """
    Weight parallel communicator for embedding layer.
    """

    def __init__(self, parallel_mode: ParallelMode) -> None:
        self.parallel_mode = parallel_mode
        self.gather_dim = 0

        self._cur_micro_step = 0
        self._num_micro_step = gpc.config.data.micro_num

    def register_module_hook(self, module: Embedding1D) -> None:
        assert isinstance(module, Embedding1D), "Embbeding weight parallel communicator is only support Embedding1D"

        module.weight.evo_tensor = None
        self.gather_dim = 0 if module.vocab_parallel else 1

        class PreModuleWrapper(torch.autograd.Function):
            """
            Wrapper pre module to prefetch module weight for forward pass.
            """

            @staticmethod
            def forward(ctx, inputs: torch.Tensor):  # pylint: disable=W0613
                if module.weight.evo_tensor is None:
                    module.weight.evo_tensor = module.weight.data

                module.weight.data = _gather(module.weight, self.parallel_mode, dim=self.gather_dim)
                inputs = inputs.detach()
                return inputs

            @staticmethod
            def backward(ctx: Any, grad_input: torch.Tensor) -> torch.Tensor:  # pylint: disable=W0613
                # since input of embedding is int64 dtype, requires_grad=False, the backward fn may not be called
                module.weight.data = module.weight.evo_tensor
                return grad_input

        class PostModuleWrapper(torch.autograd.Function):
            """
            Wrapper post module to prefetch module weight for backward pass.
            """

            @staticmethod
            def forward(ctx, output: torch.Tensor):  # pylint: disable=W0613
                module.weight.data = module.weight.evo_tensor
                output = output.detach()
                return output

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:  # pylint: disable=W0613
                module.weight.data = _gather(module.weight, self.parallel_mode, dim=self.gather_dim)
                return grad_output

        def _pre_forward_hook(module, inputs):  # pylint: disable=W0613
            return apply_to_tensors_only(PreModuleWrapper.apply, inputs)

        def _post_forward_hook(module, inputs, output):  # pylint: disable=W0613
            return apply_to_tensors_only(PostModuleWrapper.apply, output)

        module.register_forward_pre_hook(_pre_forward_hook)
        module.register_forward_hook(_post_forward_hook)

        module.weight.register_post_accumulate_grad_hook(self.grad_reduce_hook)

    def grad_reduce_hook(self, param: torch.Tensor):

        _grad, _ = reduce_scatter_raw(
            param.grad, gpc.get_group(self.parallel_mode), op=dist.ReduceOp.AVG, reduce_dim=self.gather_dim
        )
        if param.evo_tensor.grad is None:
            param.evo_tensor.grad = _grad
        else:
            param.evo_tensor.grad += _grad

        param.data = param.evo_tensor
        param.grad = None

        self._cur_micro_step += 1
        if self._cur_micro_step == self._num_micro_step:
            param.grad = param.evo_tensor.grad
            param.evo_tensor.grad = None
            self._cur_micro_step = 0


class ISPCommModelConfig:
    """
    model config for isp communicator.
    """

    def __init__(
        self,
        dtype: torch.dtype = torch.half,
        device: torch.device = None,
        activation_checkpointing: float = 0.0,
    ) -> None:
        self.dtype = dtype
        if device is None:
            self.device = get_current_device()
        else:
            self.device = device
        self.activation_checkpointing = activation_checkpointing


class ISPOverlapState:
    """
    Overlap state for isp.
    """

    def __init__(self) -> None:
        self.num_blocks: int = 0
        self.ckpt_block_num: int = 0
        self.isp_prefetch_launch_module: List[nn.Module] = []
        self.isp_modules: List[nn.Module] = []
        self.index_to_isp_modules: Dict[int, nn.Module] = {}
        self.index_to_block: Dict[int, nn.Module] = {}
        self.module_to_index: Dict[nn.Module, int] = {}
        self.weight_global_handle: Dict[str, Any] = {}
        self.weight_global_output: Dict[str, torch.Tensor] = {}
        self.bias_global_handle: Dict[str, Any] = {}
        self.bias_global_output: Dict[str, torch.Tensor] = {}


class ISPCommunicationContext(ABC):
    """
    Common communication context interface for isp communication overlap.
    """

    @abstractmethod
    def register_overlap_hooks(self, model: nn.Module) -> None:
        """
        register hooks for communication.
        """
        pass

    @abstractmethod
    def switch_forward_backward_phase(self, is_forward: bool) -> None:
        """switch forward/backward phase."""
        pass

    @abstractmethod
    def switch_current_overlap_state(self, overlap_state: ISPOverlapState) -> None:
        """switch current overlap state."""
        pass

    @abstractmethod
    def all_gather(
        self, module: nn.Module, tensor: torch.Tensor, async_op: bool = False, is_bias: bool = False
    ) -> torch.Tensor:
        """
        all gather proxy.
        TODO: The interface should not have an is_bias parameter, but it is temporarily there.
        """

    @abstractmethod
    def reduce_scatter(
        self, key: str, tensor: torch.Tensor, reduce_op: dist.ReduceOp, async_op: bool = False
    ) -> Tuple[torch.Tensor, AsyncCommHandle]:
        """
        reduce scatter proxy.
        """
        pass

    def pop_reduced_grad(self, key: str) -> torch.Tensor:
        """
        return reduce scatter results
        """
        pass


class _WaitOrCommitHandle(AsyncCommHandle):
    """
    commit or wait handle
    """

    def __init__(self, commit_func: Callable, real_handle: AsyncCommHandle = None):
        self._handle = real_handle
        self._commit_func = commit_func

    def set_handle(self, real_handle: AsyncCommHandle):
        self._handle = real_handle

    def wait(self, stream=None):
        if self._handle is None:
            self._commit_func()
            assert self._handle is not None, "should not happend"

        self._handle.wait(stream)


@dataclass
class ReduceScatterResult:
    handle: _WaitOrCommitHandle
    result: Optional[torch.Tensor] = None


@dataclass
class ReduceScatterOperation:
    key: str
    grad: torch.Tensor
    reduce_op: dist.ReduceOp


# Leveraged the implementation of FSDP2
# https://github.com/pytorch/pytorch/issues/114299
class LayerAsyncCommContext(ISPCommunicationContext):
    """
    layer level async communcation context.
    """

    def __init__(self, dtype, device, process_group) -> None:
        self.dtype = dtype
        self.device = device
        self.process_group = process_group

        # streams for communication overlap
        self._allgather_copy_in_stream = internlm_accelerator.Stream(priority=-1)
        self._allgather_comm_stream = internlm_accelerator.Stream(priority=-1)
        self._reduce_scatter_comm_stream = internlm_accelerator.Stream(priority=-1)

        self._is_forward: bool = True
        self._overlap_state: Optional[ISPOverlapState] = None

        self._allgather_result = None
        self._allgather_buffer = None

        self._reduce_scatter_state = None
        self._reduce_scatter_ops: List[ReduceScatterOperation] = []
        self._reduce_scatter_results: Dict[str, ReduceScatterResult] = {}

    def switch_forward_backward_phase(self, is_forward: bool) -> None:
        self._is_forward = is_forward

    def switch_current_overlap_state(self, overlap_state: ISPOverlapState) -> None:
        self._overlap_state = overlap_state

    # Possible future support for communication between embedding and head layers.
    # def parse_model_structure(
    #     self, chunk_id: int, state: ISPOverlapState, model: nn.Module, is_moe: bool = False
    # ) -> None:
    #     """Rewrite the data structures needed for some LayerAsyncCommContext."""

    #     state.index_to_block = {}
    #     state.index_to_isp_modules = {}
    #     state.module_to_index = {}

    #     idx = 0
    #     for name, children in model.named_children():
    #         if isinstance(children, (Embedding1D, ParallelLinearWithCommExt)):
    #             # embedding layer and head layer.
    #             if is_moe:
    #                 continue

    #             state.index_to_block[idx] = children
    #             state.module_to_index[children] = idx
    #             state.index_to_isp_modules[idx] = []

    #             full_name = f"{chunk_id}.{idx}.{name}"
    #             setattr(children.weight, "isp_reduce_scatter_name", f"{full_name}.weight")
    #             if getattr(children, "bias", None) is not None:
    #                 setattr(children.weight, "isp_reduce_scatter_name", f"{full_name}.bias")
    #             idx += 1
    #         elif isinstance(children, nn.ModuleList):
    #             # decoder layers.
    #             for block in children:
    #                 state.index_to_isp_modules[idx] = []
    #                 for name, child in block.named_modules():
    #                     if isinstance(child, (ParallelLinearWithCommExt)):
    #                         if is_moe_param(child.weight) != is_moe:
    #                             continue
    #                         state.index_to_isp_modules[idx].append(child)

    #                 if len(state.index_to_isp_modules[idx]) > 0:
    #                     state.index_to_block[idx] = block
    #                     state.module_to_index[block] = idx
    #                     idx += 1

    #     state.num_blocks = len(state.index_to_block)

    def register_overlap_hooks(self, model: nn.Module) -> None:
        def _clear_all_gather_buffer(module, *args):  # pylint: disable=W0613
            self._overlap_state.bias_global_output.clear()
            self._overlap_state.weight_global_output.clear()

        def _clear_all_gather_result(module: nn.Module, *args):  # pylint: disable=W0613
            self._allgather_result = None

        def _clear_reduce_scatter_result(module, *args):  # pylint: disable=W0613
            self._allgather_result = None
            self._reduce_scatter_ops = []

        # Pre-fetch parameters for the first layer.
        num_blocks = self._overlap_state.num_blocks

        first_block = self._overlap_state.index_to_block[0]
        last_block = self._overlap_state.index_to_block[num_blocks - 1]

        # Pull parameters for the first layer during the forward phase.
        first_block.register_forward_pre_hook(partial(self._pre_forward_for_block, -1))
        # Pull parameters for the first layer during the backward phase.
        last_block.register_full_backward_pre_hook(partial(self._pre_backward_for_block, num_blocks))

        for _block_idx in range(num_blocks):
            _block = self._overlap_state.index_to_block[_block_idx]
            # Pre-fetch parameters for the next layer.
            _block.register_forward_pre_hook(self._pre_forward_for_block)
            _block.register_full_backward_pre_hook(self._pre_backward_for_block)
            # Clean up the parameters that have been used.
            _block.register_forward_hook(_clear_all_gather_buffer)
            _block.register_full_backward_hook(_clear_all_gather_buffer)
            # Reduce scatter gradients
            _block.register_full_backward_hook(self._post_backward_for_block)

        last_block.register_forward_hook(_clear_all_gather_result)
        first_block.register_full_backward_hook(_clear_reduce_scatter_result)

    def all_gather(
        self, module: nn.Module, tensor: torch.Tensor, async_op: bool = False, is_bias: bool = False
    ) -> torch.Tensor:
        """
        all gather proxy.
        """

        already_gathered = (
            self._overlap_state.bias_global_output if is_bias else self._overlap_state.weight_global_output
        )

        if module not in already_gathered:
            result, _ = all_gather_raw(tensor, self.process_group, async_op)
        else:
            result = already_gathered[module]

        return result

    def reduce_scatter(
        self, key: str, tensor: torch.Tensor, reduce_op: dist.ReduceOp, async_op: bool = False
    ) -> Tuple[torch.Tensor, AsyncCommHandle]:
        """
        reduce scatter proxy.
        """
        if not async_op:
            result, handle = reduce_scatter_raw(tensor, self.process_group, op=reduce_op, async_op=async_op)
        else:
            self._reduce_scatter_ops.append(ReduceScatterOperation(key, tensor, reduce_op))
            result, handle = None, _WaitOrCommitHandle(self._post_backward_for_block)
            self._reduce_scatter_results[key] = ReduceScatterResult(handle, result)

            result, handle = (
                torch.zeros(
                    *(
                        tensor.shape[0] // dist.get_world_size(self.process_group),
                        *tensor.shape[1:],
                    ),
                    dtype=self.dtype,
                    device=self.device,
                ).contiguous(),
                DUMMY_HANDLE_CONST,
            )

        return result, handle

    def pop_reduced_grad(self, key: str) -> torch.Tensor:
        # Be cautious here not to directly pop, as _WaitOrCommitHandle might trigger a commit,
        # update the corresponding reduce scatter result.
        rs_result = self._reduce_scatter_results[key]
        rs_result.handle.wait()

        _ = self._reduce_scatter_results.pop(key)

        return rs_result.result

    def _check_reduce_op(self, reduce_ops: List[dist.ReduceOp]) -> dist.ReduceOp:
        _check_reduce_ops = set(reduce_ops)
        assert len(_check_reduce_ops) == 1, f"cannot fuse reduce scatter with different reduce_op {_check_reduce_ops}"

        return _check_reduce_ops.pop()

    # Copied from FSDP2.
    def _all_gather_copy_in(
        self,
        all_gather_inputs: List[torch.Tensor],
        inp_split_sizes: List[int],
        all_gather_input_numel: int,
        world_size: int,
        rank: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        all_gather_output = torch.empty((all_gather_input_numel * world_size,), dtype=dtype, device=device)
        all_gather_input = all_gather_output.narrow(0, all_gather_input_numel * rank, all_gather_input_numel)
        foreach_copy_dsts = torch.split(all_gather_input, inp_split_sizes)

        all_gather_inputs = [t.view(-1) for t in all_gather_inputs]
        torch._foreach_copy_(foreach_copy_dsts, all_gather_inputs)

        return all_gather_input, all_gather_output

    # Copied from FSDP2.
    def _split_with_sizes_copy(
        self,
        all_gather_output: torch.Tensor,
        all_gather_input_split_sizes: List[int],
        dim: int,
        out: List[torch.Tensor],
    ) -> None:
        torch.split_with_sizes_copy(all_gather_output, all_gather_input_split_sizes, dim=dim, out=out)

    def _all_gather_block_params(self, block_idx):
        all_gather_inputs = []

        for module in self._overlap_state.index_to_isp_modules[block_idx]:
            all_gather_inputs.append(module.weight)
            if module.bias is not None:
                all_gather_inputs.append(module.bias)
        inp_split_sizes = [t.numel() for t in all_gather_inputs]
        all_gather_input_numel = sum(inp_split_sizes)
        all_gather_input_shapes = [t.shape for t in all_gather_inputs]

        with internlm_accelerator.stream(self._allgather_copy_in_stream):
            all_gather_input, all_gather_output = self._all_gather_copy_in(
                all_gather_inputs,
                inp_split_sizes,
                all_gather_input_numel,
                dist.get_world_size(self.process_group),
                dist.get_rank(self.process_group),
                self.dtype,
                self.device,
            )

        # 提交allgather通信
        self._allgather_comm_stream.wait_stream(self._allgather_copy_in_stream)

        with internlm_accelerator.stream(self._allgather_comm_stream):
            dist.all_gather_into_tensor(
                output_tensor=all_gather_output,
                input_tensor=all_gather_input,
                group=self.process_group,
                async_op=False,
            )
            all_gather_event = self._allgather_comm_stream.record_event()

        return (all_gather_event, all_gather_output, all_gather_input_shapes)

    def _wait_and_copy_out_params(self, block_index: int) -> None:
        cur_allgather_event, cur_allgather_output, cur_input_shapes = self._allgather_result

        internlm_accelerator.current_stream().wait_event(cur_allgather_event)

        world_size = dist.get_world_size(self.process_group)
        cur_inp_split_sizes = [t.numel() for t in cur_input_shapes]

        allgather_outputs = [
            torch.empty(torch.Size([numel * world_size]), dtype=self.dtype, device=self.device)
            for numel in cur_inp_split_sizes
        ]

        cur_allgather_output = cur_allgather_output.view(world_size, -1)
        out = [t.view(world_size, -1) for t in allgather_outputs]
        self._split_with_sizes_copy(cur_allgather_output, cur_inp_split_sizes, dim=1, out=out)

        _idx = 0
        for module in self._overlap_state.index_to_isp_modules[block_index]:
            self._overlap_state.weight_global_output[module] = out[_idx].view(-1, *cur_input_shapes[_idx][1:])
            _idx += 1

            if module.bias is not None:
                self._overlap_state.bias_global_output[module] = out[_idx].view(-1, *cur_input_shapes[_idx][1:])
                _idx += 1

    @torch.no_grad()
    def _pre_forward_for_block(self, block_or_idx: Union[int, nn.Module], *args):  # pylint: disable=W0613
        if isinstance(block_or_idx, int):
            block_index = block_or_idx
        else:
            block_index = block_or_idx.layer_idx

        self._allgather_copy_in_stream.wait_stream(internlm_accelerator.current_stream())

        # Check if the communication for this layer's parameters is complete and unpack the communication results.
        if self._allgather_result is not None:
            self._wait_and_copy_out_params(block_index)

        # Pre-fetch parameters for the "next layer".
        if self._is_forward and block_index + 1 < self._overlap_state.num_blocks:
            # start the all-gather for next block
            next_all_gather_result = self._all_gather_block_params(block_index + 1)
        else:
            next_all_gather_result = None

        self._allgather_result = next_all_gather_result

    @torch.no_grad()
    def _pre_backward_for_block(self, block_or_idx: Union[int, nn.Module], *args):  # pylint: disable=W0613
        if isinstance(block_or_idx, int):
            block_index = block_or_idx
        else:
            block_index = block_or_idx.layer_idx

        self._allgather_copy_in_stream.wait_stream(internlm_accelerator.current_stream())

        # Check if the communication for this layer's parameters is complete and unpack the communication results.
        if self._allgather_result is not None:
            self._wait_and_copy_out_params(block_index)

        # Pre-fetch parameters for the "next layer".
        if block_index - 1 >= 0:
            next_all_gather_result = self._all_gather_block_params(block_index - 1)
        else:
            next_all_gather_result = None

        self._allgather_result = next_all_gather_result

    @torch.no_grad()
    def _post_backward_for_block(self, *args):  # pylint: disable=W0613
        if len(self._reduce_scatter_ops) == 0:
            return

        if self._reduce_scatter_state is not None:
            internlm_accelerator.current_stream().wait_event(self._reduce_scatter_state[1])
            self._reduce_scatter_state = None

        # Aggregate parameters for reduce scatter.
        world_size = dist.get_world_size(self.process_group)

        reduce_ops = [_i.reduce_op for _i in self._reduce_scatter_ops]
        reduce_op = self._check_reduce_op(reduce_ops)

        unshard_grads = [_i.grad for _i in self._reduce_scatter_ops]
        unshard_grad_sizes = [_grad.size() for _grad in unshard_grads]
        reduce_scatter_input_numel = sum(s.numel() for s in unshard_grad_sizes)

        # wait for compute stream
        self._reduce_scatter_comm_stream.wait_stream(internlm_accelerator.current_stream())

        with internlm_accelerator.stream(self._reduce_scatter_comm_stream):

            reduce_scatter_input = torch.empty((reduce_scatter_input_numel,), dtype=self.dtype, device=self.device)
            reduce_scatter_input = reduce_scatter_input.view(world_size, -1)
            torch._chunk_cat(unshard_grads, dim=0, num_chunks=world_size, out=reduce_scatter_input)

            reduce_output, _ = reduce_scatter_raw(reduce_scatter_input, self.process_group, reduce_op)

            # unack reduce scatter result
            flat_grad_offset = 0

            for _idx, _unshard_size in enumerate(unshard_grad_sizes):
                _shard_size = (_unshard_size[0] // world_size, *_unshard_size[1:])
                _strides = make_contiguous_strides_for(_shard_size)

                _new_sharded_grad = torch.as_strided(
                    reduce_output,
                    size=_shard_size,
                    stride=_strides,
                    storage_offset=flat_grad_offset,
                )

                _key = self._reduce_scatter_ops[_idx].key
                _event = self._reduce_scatter_comm_stream.record_event()

                self._reduce_scatter_results[_key].result = _new_sharded_grad
                self._reduce_scatter_results[_key].handle.set_handle(_event)

                flat_grad_offset += _unshard_size.numel() // world_size

        reduce_scatter_event = self._reduce_scatter_comm_stream.record_event()

        self._reduce_scatter_state = (unshard_grads, reduce_scatter_event)
        self._reduce_scatter_ops = []


class ISPCommunicator(WPCommunicator):
    """
    ISP Communicator for managing the all-gather and reduce_scatter of Intern Sequence Parallel.
    """

    def __init__(
        self,
        model: Union[nn.Module, nn.ModuleList],
        model_conf: ISPCommModelConfig,
        overlap: bool = False,
        process_group: dist.ProcessGroup = None,
        is_moe: bool = False,
        selective_ckpt_offload: bool = False,
        early_reduce_scatter_release: bool = True,
        enable_layer_fuse_isp_comm: bool = False,
    ) -> None:
        self.process_group = process_group
        self.overlap = overlap
        self.model_conf = model_conf
        self.is_moe = is_moe
        self.is_forward = True
        self._reduce_scatter_handlers = {}
        self._forward_prefetch_prerequisites = []
        self._zero_const_pool = {}

        self._enable_early_reduce_scatter_release = early_reduce_scatter_release
        self._early_prev_layer_rs_handles = []
        self._early_curr_layer_rs_handles = []
        self._forward_overlap_per = self._get_forward_overlap_granularity()
        self._launch_before_module = self._get_launch_before_module()
        # As an optimization, do not release weight after forward for the last
        # transformer block since wp would prefetch it immediately
        self.layers_wp_not_release = []  # [gpc.config.isp_num_layers - 1]
        self.layers_fa_not_release = [
            gpc.config.isp_num_layers - 1,
            int(gpc.config.model.checkpoint * gpc.config.isp_num_layers) - 1,
        ]
        self.sc_offload = selective_ckpt_offload

        # real overlap state for each chunk.
        self._overlap_states: Dict[int, ISPOverlapState] = {}

        # inner interface variables of overlap state.
        self._num_blocks = None
        self._ckpt_block_num = None
        self._isp_prefetch_launch_module = None
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

        enable_layer_fuse_isp_comm = overlap and enable_layer_fuse_isp_comm
        if enable_layer_fuse_isp_comm:
            self._layer_level_comm_context = LayerAsyncCommContext(
                dtype=self.model_conf.dtype,
                device=self.model_conf.device,
                process_group=self.process_group,
            )
        else:
            self._layer_level_comm_context = None

        # init overlap states if necessary.
        if self.overlap:
            # build overlap states for every chunk.
            for chunk_id, chunk in enumerate(unwrap_naive_amp(model)):
                self._parse_model_structure(chunk_id, chunk)
                self.switch_current_model_chunk(chunk_id)
                # register overlap hooks for every chunk.
                self._register_sync_parameters_hook(chunk)
            # switch to chunk 0 at first.
            self.switch_current_model_chunk(0)

    def _get_launch_before_module(self):
        if self.is_moe is True:
            _launch_before = gpc.config.parallel.expert_weight.get("launch_allgather_before", "wo")
        else:
            _launch_before = gpc.config.parallel.weight.get("launch_allgather_before", "wo")

        if _launch_before == "wqkv":
            return ["wqkv", "Wqkv", "qkv", "q_a_proj", "q_proj"]
        elif _launch_before == "attn":
            return ["attn"]
        elif _launch_before == "wo":
            return ["out_proj", "wo"]
        elif _launch_before == "w1":
            return ["w1", "fused_w1_w3"]
        else:
            assert False, "launch module should be in ['wqkv', 'attn', 'wo', 'w1']"

    def _get_forward_overlap_granularity(self):
        if self.is_moe is True:
            _overlap_granularity = gpc.config.parallel.expert_weight.get("forward_overlap_per", "layer")
        else:
            _overlap_granularity = gpc.config.parallel.weight.get("forward_overlap_per", "layer")

        assert _overlap_granularity in ["module", "layer"]
        return _overlap_granularity

    def pop_reduced_grad(self, key: str) -> torch.Tensor:
        if self._layer_level_comm_context is not None:
            return self._layer_level_comm_context.pop_reduced_grad(key)

        result, handle = self._reduce_scatter_handlers.pop(key)
        handle.wait()
        return result

    def _parse_model_structure(self, cid: int, model: nn.Module) -> None:
        self._overlap_states[cid] = ISPOverlapState()

        def get_model(obj: nn.Module) -> nn.Module:
            return get_model(obj.model) if hasattr(obj, "model") else obj

        def is_allgather_launch_module(name, module):
            return (
                hasattr(module, "is_attn_cls")
                and getattr(module, "is_attn_cls")
                and self._launch_before_module == ["attn"]
            ) or (name.split(".")[-1] in self._launch_before_module)

        # Important: only works for llama-class models
        children_name = get_model(model).named_children()
        for _, children in children_name:
            if isinstance(children, nn.ModuleList):
                self._overlap_states[cid].ckpt_block_num = int(self.model_conf.activation_checkpointing * len(children))

                for idx, block in enumerate(children):
                    if not hasattr(block, "layer_idx"):
                        setattr(block, "layer_idx", idx)

                    self._overlap_states[cid].index_to_isp_modules[idx] = []
                    self._overlap_states[cid].index_to_block[idx] = block
                    for name, child in block.named_modules():
                        if is_allgather_launch_module(name, child):
                            self._overlap_states[cid].isp_prefetch_launch_module.append(child)
                            self._overlap_states[cid].module_to_index[child] = idx
                        if isinstance(child, (ParallelLinearWithCommExt)):
                            if is_moe_param(child.weight) != self.is_moe:
                                continue

                            self._overlap_states[cid].module_to_index[child] = idx
                            self._overlap_states[cid].isp_modules.append(child)
                            self._overlap_states[cid].index_to_isp_modules[idx].append(child)

                            setattr(child, "isp_name", name)
                            setattr(child, "isp_layer_idx", idx)

                            full_name = f"{cid}.{idx}.{name}"
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
        assert module not in self._bias_global_output and module not in self._weight_global_output
        with_bias = module.bias is not None

        # submit the all-gather communication for weight and bias.
        if with_bias:
            if module not in self._bias_global_output:
                bias_output, bias_handle = all_gather_raw(
                    module.bias,
                    self.process_group,
                    async_op=True,
                )
                self._bias_global_handle[module] = bias_handle
                self._bias_global_output[module] = bias_output

        if module not in self._weight_global_output:
            weight_output, weight_handle = all_gather_raw(
                module.weight,
                self.process_group,
                async_op=True,
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
        if self._forward_overlap_per == "layer" and self.is_forward is True:
            self._all_gather_block_weight(0)

    def _pre_forward_hook_for_prefetch_launch_module(self, module: nn.Module, *args):  # pylint: disable=W0613
        block_index = self._module_to_index[module]

        if self._forward_overlap_per == "layer":
            if (block_index - 1 < self._ckpt_block_num) and self.is_forward is False:
                if block_index - 1 >= 0:
                    self._all_gather_block_weight(block_index - 1)
            else:
                # start the all-gather for next block
                if block_index + 1 < self._num_blocks:
                    self._all_gather_block_weight(block_index + 1)

        # register offload and prefetch hook for selective ckpt with wo linear
        if self.sc_offload is True:
            # move current layer's attn output from GPU to CPU asynchronizely
            if (
                self.is_forward is True
                and gpc.config.selective_checkpoint
                and block_index not in self.layers_fa_not_release
                and block_index < self._ckpt_block_num
            ):
                get_offload_manager().offload_fa_output_with_layer(layer_idx=block_index)

            # load previous layer's attn output from CPU to GPU asynchronizely
            if (
                self.is_forward is False
                and gpc.config.selective_checkpoint
                and (0 <= (block_index - 1) < self._ckpt_block_num)
            ):
                get_offload_manager().preload_fa_output_with_layer(layer_idx=block_index - 1)

    def _pre_forward_hook_for_module(self, module: nn.Module, *args):  # pylint: disable=W0613
        if module not in self._weight_global_handle:
            self._all_gather_module_weight(module)

        self._wait_handle(module)

        if self._forward_overlap_per == "module":
            # start the all-gather for next module
            # 1.forward prefetch for next module
            module_index = self._isp_modules.index(module)
            module_layer_id = self._module_to_index[module]
            if module_index + 1 < len(self._isp_modules) and self.is_forward is True:
                next_module = self._isp_modules[module_index + 1]
                self._all_gather_module_weight(next_module)

            # 2.recompute forward prefetch for next module
            if self.is_forward is False:
                if module_index + 1 < len(self._isp_modules):
                    next_module = self._isp_modules[module_index + 1]
                    next_module_layer_id = self._module_to_index[next_module]
                    if module_layer_id == next_module_layer_id:
                        self._all_gather_module_weight(next_module)
                    # if current module is the last module in current layer, prefetch previous layer's first module
                    elif module_layer_id - 1 >= 0:
                        next_module = self._index_to_isp_modules[module_layer_id - 1][0]
                        self._all_gather_module_weight(next_module)
                else:
                    # if current module is the last module, prefetch previous layer's first module
                    if module_layer_id - 1 >= 0:
                        next_module = self._index_to_isp_modules[module_layer_id - 1][0]
                        self._all_gather_module_weight(next_module)

    def _post_forward_hook_for_module(self, module: nn.Module, *args):  # pylint: disable=W0613
        if int(module.isp_layer_idx) in self.layers_wp_not_release:
            # print(f"the layer {module.isp_layer_idx} after forward not clear weight")
            return
        if not ((self._module_to_index[module] < self._ckpt_block_num) and self.is_forward is False):
            self._clear_handle(module)
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

    def _early_reduce_scatter_release_hook(self, *args):  # pylint: disable=W0613
        for handle in self._early_prev_layer_rs_handles:
            handle.wait()

        self._early_prev_layer_rs_handles = self._early_curr_layer_rs_handles
        self._early_curr_layer_rs_handles = []

    def _register_sync_parameters_hook(self, model) -> None:
        """
        register forward hooks and backward hooks for isp modules.
        """

        if self._layer_level_comm_context is not None:
            self._layer_level_comm_context.register_overlap_hooks(model)
            return

        # register forward hooks
        # 1. register pre_forward_hook @block_0 to prefetch weight for block 0.
        # 2. register pre_forward_hook @prefetch_launch_module to prefetch weight for next block,
        #    when forward overlap granularity is 'layer'.
        # 3. register pre_forward_hook @isp_module to wait handle for current module,
        #    and prefetch weight for next module when forward overlap granularity is 'module'.
        # 4. register post_forward_hook @isp_module to release memory resource.
        self._index_to_block[0].register_forward_pre_hook(self._pre_forward_hook_for_first_block)

        for module in self._isp_prefetch_launch_module:
            module.register_forward_pre_hook(self._pre_forward_hook_for_prefetch_launch_module)

        for module in self._isp_modules:
            module.register_forward_pre_hook(self._pre_forward_hook_for_module)
            module.register_forward_hook(self._post_forward_hook_for_module)

        # register backward hooks
        # 1. register pre_backward_hook @isp_module to wait handle for current module and to prefetch for next module.
        # 2. register post_backward_hook @isp_module to release memory resource.
        if self._ckpt_block_num < self._num_blocks:
            for module in self._isp_modules:
                module.register_full_backward_pre_hook(self._pre_backward_hook_for_module)

        for module in self._isp_modules:
            module.register_full_backward_hook(self._post_backward_hook_for_module)

        if self._enable_early_reduce_scatter_release:
            for block_idx in range(self._num_blocks):
                block = self._index_to_block[block_idx]
                block.register_full_backward_hook(self._early_reduce_scatter_release_hook)

    def _get_constant_zero(self, size: tuple) -> torch.Tensor:
        if size not in self._zero_const_pool:
            self._zero_const_pool[size] = torch.zeros(
                *size, dtype=self.model_conf.dtype, device=self.model_conf.device
            ).contiguous()

        return self._zero_const_pool[size]

    def communication_mode(self) -> str:
        return "wp"

    def switch_current_model_chunk(self, chunk_id: int) -> None:
        self._isp_prefetch_launch_module = self._overlap_states[chunk_id].isp_prefetch_launch_module
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

        if self._layer_level_comm_context is not None:
            self._layer_level_comm_context.switch_current_overlap_state(self._overlap_states[chunk_id])

    def switch_forward_backward_phase(self, is_forward: int) -> None:
        self.is_forward = is_forward

        if self._layer_level_comm_context is not None:
            self._layer_level_comm_context.switch_forward_backward_phase(is_forward)

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

    def weight_hook(
        self, tensor: torch.Tensor, async_op: bool = False, module: nn.Module = None, is_bias: bool = False
    ) -> torch.Tensor:
        if dist.get_world_size(self.process_group) <= 1:
            return tensor

        if not self.overlap:
            result, _ = all_gather_raw(tensor, self.process_group, async_op=async_op)
            return result

        if self._layer_level_comm_context is not None:
            result = self._layer_level_comm_context.all_gather(module, tensor, async_op, is_bias)
            return result
        elif is_bias:
            assert module is not None, "The module parameter must be specified"
            result = self._bias_global_output[module]
        else:
            assert module is not None, "The module parameter must be specified"
            result = self._weight_global_output[module]

        return result

    def grad_hook(
        self,
        tensor: torch.Tensor,
        async_op: bool = False,
        module: nn.Module = None,
        reduce_op: dist.ReduceOp = dist.ReduceOp.AVG,
        is_bias: bool = False,
    ) -> Tuple[torch.Tensor, AsyncCommHandle]:
        if dist.get_world_size(self.process_group) <= 1:
            return tensor, DUMMY_HANDLE_CONST

        if not self.overlap:
            result, handle = reduce_scatter_raw(tensor, self.process_group, op=reduce_op, async_op=async_op)
        else:
            assert module is not None, "The module parameter must be specified"

            if is_bias:
                assert hasattr(module.bias, "isp_reduce_scatter_name")
                key = getattr(module.bias, "isp_reduce_scatter_name")
            else:
                assert hasattr(module.weight, "isp_reduce_scatter_name")
                key = getattr(module.weight, "isp_reduce_scatter_name")

            if self._layer_level_comm_context is not None:
                result, handle = self._layer_level_comm_context.reduce_scatter(key, tensor, reduce_op, async_op)
            else:
                output, handle = reduce_scatter_raw(
                    tensor,
                    self.process_group,
                    op=reduce_op,
                    async_op=async_op,
                )

                if self._enable_early_reduce_scatter_release:
                    self._early_curr_layer_rs_handles.append(handle)

                self._reduce_scatter_handlers[key] = (output, handle)

                result, handle = (
                    self._get_constant_zero(
                        (
                            tensor.shape[0] // dist.get_world_size(self.process_group),
                            *tensor.shape[1:],
                        )
                    ),
                    DUMMY_HANDLE_CONST,
                )

        return result, handle


class ISPCommunicatorSchedulerHook(SchedulerHook):
    """
    SchedulerHook for isp overlap handler
    """

    def __init__(self, overlap_handler: ISPCommunicator, zero_optim) -> None:
        self._isp_communicator = overlap_handler
        self._zero_optim = zero_optim

    def before_forward(self, scheduler, inputs) -> None:  # pylint: disable=W0613
        self._isp_communicator.switch_forward_backward_phase(is_forward=True)
        # switch model chunk before forward
        chunk_id = 0 if gpc.virtual_pipeline_parallel_rank is None else gpc.virtual_pipeline_parallel_rank
        self._isp_communicator.switch_current_model_chunk(chunk_id)

    def after_forward(self, scheduler, outputs) -> None:  # pylint: disable=W0613
        pass

    def before_criterion(self, scheduler, outputs, label) -> None:  # pylint: disable=W0613
        pass

    def after_criterion(self, scheduler, loss) -> None:  # pylint: disable=W0613
        pass

    def before_backward(self, scheduler, outputs, outputs_grad) -> None:  # pylint: disable=W0613
        self._isp_communicator.switch_forward_backward_phase(is_forward=False)
        # switch model chunk before backward
        chunk_id = 0 if gpc.virtual_pipeline_parallel_rank is None else gpc.virtual_pipeline_parallel_rank
        self._isp_communicator.switch_current_model_chunk(chunk_id)

    def after_backward(self, scheduler, inputs_grad) -> None:  # pylint: disable=W0613
        # accumulate left gradients in last bucket after backward.
        if self._isp_communicator and self._isp_communicator.overlap:
            self._zero_optim.accumulate_left_grads_after_backward()

            if (
                getattr(gpc.config.parallel["pipeline"], "mode", "1F1B").upper() in ["ZBV", "ZBH1"]
                and not self._zero_optim.skip_grad_reduce
            ):
                self._zero_optim.reduce_left_grads_after_backward()

        if self._isp_communicator and self._isp_communicator._enable_early_reduce_scatter_release:
            self._isp_communicator._early_prev_layer_rs_handles = []
            self._isp_communicator._early_curr_layer_rs_handles = []

    def post_helper_func(self, scheduler, outputs, label) -> None:  # pylint: disable=W0613
        pass


class ISPCommunicatorWrapper:
    """
    Wrapper for multiple ISPCommunicators.
    TODO: check all isp communicator external interfaces and wrap them.
    """

    def __init__(
        self,
        isp_communicators: List[ISPCommunicator],
    ) -> None:
        self.isp_communicators = isp_communicators

    def pop_reduced_grad(self, key) -> dict:
        for communicator in self.isp_communicators:
            try:
                return communicator.pop_reduced_grad(key)
            except KeyError:
                continue

        # key is not in any communicator
        raise KeyError(f"key {key} is not found")

    def register_prerequisite_for_forward_prefetch_hooks(self, prerequisite_func: Callable) -> None:
        for isp_communicator in self.isp_communicators:
            isp_communicator.register_prerequisite_for_forward_prefetch_hooks(prerequisite_func)


# adpated from https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/sequence/layer.py
class _SeqAllToAll(torch.autograd.Function):
    "sequence alltoall function"

    @staticmethod
    def forward(
        ctx,
        group: dist.ProcessGroup,
        scatter_idx: Optional[Union[List[int], int]],
        gather_idx: Optional[Union[List[int], int]],
        *input_: torch.Tensor,
    ) -> torch.Tensor:

        ctx.group = group
        ctx.scatter_idx = scatter_idx
        ctx.gather_idx = gather_idx
        seq_world_size = dist.get_world_size(group)

        if dist.get_world_size(group) <= 1:
            if len(input_) == 1:
                return input_[0]
            return input_

        if len(input_) == 1:
            input_list = [t.contiguous() for t in torch.tensor_split(input_[0], seq_world_size, scatter_idx)]
            output_list = [torch.empty_like(input_list[0]) for _ in range(seq_world_size)]
            # TODO: use all_to_all_single instead
            dist.all_to_all(output_list, input_list, group=group)
            return torch.cat(output_list, dim=gather_idx).contiguous()

        outputs = []

        assert len(scatter_idx) == len(gather_idx)
        assert len(gather_idx) == len(input_)

        for i in range(len(input_)):

            if i == 0:
                input_list = [t.contiguous() for t in torch.tensor_split(input_[i], seq_world_size, scatter_idx[i])]
                output_list = [torch.empty_like(input_list[0]) for _ in range(seq_world_size)]
                handle_last = dist.all_to_all(output_list, input_list, group=group, async_op=True)

            # conduct the next all2all
            if i + 1 < len(input_):
                input_list_next = [
                    t.contiguous() for t in torch.tensor_split(input_[i + 1], seq_world_size, scatter_idx[i + 1])
                ]
                output_list_next = [torch.empty_like(input_list_next[0]) for _ in range(seq_world_size)]
                handle_next = dist.all_to_all(output_list_next, input_list_next, group=group, async_op=True)

            handle_last.wait()

            outputs.append(torch.cat(output_list, dim=gather_idx[i]).contiguous())

            if i + 1 < len(input_):
                handle_last = handle_next
                input_list = input_list_next
                output_list = output_list_next

        return tuple(outputs)

    @staticmethod
    def backward(ctx, *grad_output: torch.Tensor) -> Tuple[None, torch.Tensor, None, None]:

        if dist.get_world_size(ctx.group) <= 1:
            return (None, None, None, *grad_output)
        res = _SeqAllToAll.apply(ctx.group, ctx.gather_idx, ctx.scatter_idx, *grad_output)
        if len(grad_output) == 1:
            return (None, None, None, res)

        return (None, None, None, *res)


# adpated from https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/sequence/layer.py
class DistributedAttention(nn.Module):
    """Initialization.

    Arguments:
        local_attention (Module): local self-attention module
        sequence_process_group (ProcessGroup): sequence parallel process group
    """

    def __init__(
        self,
        local_attention: Union[nn.Module, Callable],
        sequence_process_group: dist.ProcessGroup,
    ) -> None:
        super().__init__()
        self.local_attn = local_attention
        self.spg = sequence_process_group
        self.sp_size = dist.get_world_size(self.spg)

    @params_dispatch_with_condition(condition=check_attention_argument)
    def forward(self) -> torch.Tensor:
        assert False, "Should never arrive"

    @forward.register(conditions=(str(QKVPackType.QKVPACKED), str(CuSeqlenType.With)))
    @forward.register(conditions=(str(QKVPackType.QKVPACKED), str(CuSeqlenType.WithOut)))
    def _qkv(self, qkv: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """forward

        Arguments:
            qkv (Tensor): packed qkv input to the layer
            kwargs: other args

        Returns:
            * output (Tensor): context output
        """

        # qkv shape: [1, packlen, 3, n_head, head_dim] or [batch, seqlen, 3, n_head, head_dim]
        # scatter in n_head and gather in seqlen(packlen)
        qkv = _SeqAllToAll.apply(self.spg, 3, 1, qkv)

        context = self.local_attn(qkv, *args, **kwargs)

        # context shape: [1, packlen, n_head, head_dim] or [batch, seqlen, n_head, head_dim]
        # scatter in seqlen(packlen) and gather in n_head
        context = _SeqAllToAll.apply(self.spg, 1, 2, context)

        return context

    @forward.register(conditions=(str(QKVPackType.KVPACKED), str(CuSeqlenType.With)))
    @forward.register(conditions=(str(QKVPackType.KVPACKED), str(CuSeqlenType.WithOut)))
    def _q_kv(self, q: torch.Tensor, kv: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """forward

        Arguments:
            q (Tensor): q input to the layer
            kv (Tensor): packed kv input to the layer
            kwargs: other args

        Returns:
            output (Tensor): context output
        """
        # q shpae: [1, packlen, n_head, head_dim] or [batch, seqlen, n_head, head_dim]
        # scatter in n_head and gather in seqlen(packlen)

        # kv shape: [1, packlen, 2, n_head, head_dim] or [batch, seqlen, 2, n_head, head_dim]
        # scatter in n_head and gather in seqlen(packlen)
        num_head_kv = kv.shape[3]
        # if the num head of kv is not enough to be splitted by sp
        # then we could copy the kv head
        if self.sp_size > num_head_kv:
            assert self.sp_size % num_head_kv == 0, "the num_head_kv should be divided by sp size."
            kv = expandKVPacked(kv, self.sp_size // num_head_kv, 3)

        q, kv = _SeqAllToAll.apply(self.spg, [2, 3], [1, 1], q, kv)

        context = self.local_attn(q, kv, *args, **kwargs)

        context = _SeqAllToAll.apply(self.spg, 1, 2, context)

        return context

    @forward.register(conditions=(str(QKVPackType.QKVSPLITED), str(CuSeqlenType.With)))
    @forward.register(conditions=(str(QKVPackType.QKVSPLITED), str(CuSeqlenType.WithOut)))
    def _q_k_v(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """forward

        Arguments:
            q (Tensor): q input to the layer
            k (Tensor): k input to the layer
            v (Tensor): v input to the layer
            kwargs: other args

        Returns:
            * output (Tensor): context output
        """
        # self._scatter_gather_idx["q"] = [1, 0]  # q/k/v shape: [sequence, head, head_dim]
        # q shpae: [1, packlen, n_head, head_dim] or [batch, seqlen, n_head, head_dim]
        # scatter in n_head and gather in seqlen(packlen)
        q = _SeqAllToAll.apply(self.spg, 2, 1, q)
        # k shpae: [1, packlen, n_head, head_dim] or [batch, seqlen, n_head, head_dim]
        # scatter in n_head and gather in seqlen(packlen)
        k = _SeqAllToAll.apply(self.spg, 2, 1, k)
        # v shpae: [1, packlen, n_head, head_dim] or [batch, seqlen, n_head, head_dim]
        # scatter in n_head and gather in seqlen(packlen)
        v = _SeqAllToAll.apply(self.spg, 2, 1, v)

        context = self.local_attn(q, k, v, *args, **kwargs)

        # context shape: [1, packlen, n_head, head_dim] or [batch, seqlen, n_head, head_dim]
        # scatter in seqlen(packlen) and gather in n_head
        context = _SeqAllToAll.apply(self.spg, 1, 2, context)

        return context


def auto_wrap_distributed_attention(attn_impl: nn.Module) -> Callable[[bool, Any, float], nn.Module]:
    """
    Wrap a local attention module to a distributed one, which will be used in the ISP parallelism.
    """

    # should we impl distributed attention as a metaclass?
    def _attetion_constructor(
        attn_impl: type, causal=False, softmax_scale=None, attention_dropout=0.0, layer_idx=0
    ) -> nn.Module:
        tp_mode = gpc.config.parallel["tensor"].get("mode", TensorParallelMode.mtp.name)

        if tp_mode != TensorParallelMode.isp.name:
            return attn_impl(causal, softmax_scale, attention_dropout)
        else:
            if gpc.config.parallel.sequence_2D.enable is True:
                spg = gpc.get_group(ParallelMode.HEAD)
            else:
                spg = gpc.get_group(ParallelMode.TENSOR)
            return DistributedAttention(
                local_attention=attn_impl(causal, softmax_scale, attention_dropout, layer_idx),
                sequence_process_group=spg,
            )

    return partial(_attetion_constructor, attn_impl=attn_impl)


def auto_wrap_func_distributed_attention(attn_impl: Callable) -> Callable[..., Callable]:
    """
    Wrap a local attention function to a distributed one, which will be used in the ISP parallelism.
    """

    # should we impl distributed attention as a metaclass?
    def _attetion_constructor(*args, attn_impl: type, **kwargs) -> Callable:
        tp_mode = gpc.config.parallel["tensor"].get("mode", TensorParallelMode.mtp.name)

        if tp_mode != TensorParallelMode.isp.name:
            return attn_impl(*args, **kwargs)
        else:
            return DistributedAttention(
                local_attention=attn_impl, sequence_process_group=gpc.get_group(ParallelMode.TENSOR)
            )(*args, **kwargs)

    return partial(_attetion_constructor, attn_impl=attn_impl)
