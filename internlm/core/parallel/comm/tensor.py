"""
communication for tensor/sequence parallel.
"""

from typing import Tuple
from enum import Enum
from abc import ABC, abstractmethod

import torch
from torch import distributed as dist

from internlm.core.context import ParallelMode
from internlm.core.context.parallel_context import global_context as gpc
from internlm.core.parallel.comm.utils import (
    _gather,
    _split,
    all_gather_raw,
    all_reduce_raw,
    reduce_scatter_raw,
    DUMMY_HANDLE_CONST,
    AsyncCommHandle,
)

# input gather dim
__INPUT_GATHER_DIM = -2  # shape: [batch, seqlen, dim] or [packlen, dim]


class LinearRole(Enum):
    COLUMN = "column"
    ROW = "row"


# not realy useful, only for code hint.
class TPCommunicator(ABC):
    """
    Common communicator interafce for tensor/sequence parallel.
    """

    @abstractmethod
    def save_total_input(self) -> bool:
        """
        Should linear save total input after all gather as activation in sequence parallel.
        """
        pass

    @abstractmethod
    def input_hook(
        self, _input: torch.Tensor, async_op: bool = False, is_forward: bool = True
    ) -> Tuple[torch.Tensor, AsyncCommHandle]:
        """
        communiction for input when forward/backward.
        """
        pass

    @abstractmethod
    def grad_output_hook(
        self, grad_output: torch.Tensor, async_op: bool = False
    ) -> Tuple[torch.Tensor, AsyncCommHandle]:
        """
        communiction for grad_output when backward.
        """
        pass

    @abstractmethod
    def grad_input_hook(self, grad_input: torch.Tensor, async_op: bool = False) -> Tuple[torch.Tensor, AsyncCommHandle]:
        """
        communiction for grad_input when backward.
        """
        pass

    @abstractmethod
    def output_hook(self, output: torch.Tensor, async_op: bool = False) -> Tuple[torch.Tensor, AsyncCommHandle]:
        """
        communiction for output when forward.
        """
        pass


class TensorParallelCommunicator(TPCommunicator):
    def __init__(self, process_group: dist.ProcessGroup, role: LinearRole) -> None:
        assert role in (LinearRole.COLUMN, LinearRole.ROW), f"Unknown linear role: {role}"

        self._process_group = process_group
        self._role = role

        self._save_total_input = False

    def save_total_input(self) -> bool:
        return self._save_total_input

    def input_hook(
        self, _input: torch.Tensor, async_op: bool = False, is_forward: bool = True
    ) -> Tuple[torch.Tensor, AsyncCommHandle]:
        """
        tensor parallel should do nothing for input.
        """
        return _input, DUMMY_HANDLE_CONST

    def grad_output_hook(
        self, grad_output: torch.Tensor, async_op: bool = False
    ) -> Tuple[torch.Tensor, AsyncCommHandle]:
        """
        tensor parallel should do nothing for grad_output.
        """
        return grad_output, DUMMY_HANDLE_CONST

    def grad_input_hook(self, grad_input: torch.Tensor, async_op: bool = False) -> Tuple[torch.Tensor, AsyncCommHandle]:
        """
        all reduce grad_input only for column parallel linear when backward.
        """
        if dist.get_world_size(self._process_group) <= 1 or self._role == LinearRole.ROW:
            return grad_input, DUMMY_HANDLE_CONST

        return all_reduce_raw(grad_input, process_group=self._process_group, async_op=async_op)

    def output_hook(self, output: torch.Tensor, async_op: bool = False) -> Tuple[torch.Tensor, AsyncCommHandle]:
        """
        all reduce output only for row parallel linear when forward.
        """
        if dist.get_world_size(self._process_group) <= 1 or self._role == LinearRole.COLUMN:
            return output, DUMMY_HANDLE_CONST

        return all_reduce_raw(output, process_group=self._process_group, async_op=async_op)


class HeadTensorParallelCommunicator(TensorParallelCommunicator):
    def __init__(self, parallel_mode: ParallelMode, retain_out_sharded: bool = True) -> None:
        super().__init__(process_group=gpc.get_group(parallel_mode), role=LinearRole.COLUMN)

        self._parallel_mode = parallel_mode
        self._retain_out_sharded = retain_out_sharded

    def grad_output_hook(
        self, grad_output: torch.Tensor, async_op: bool = False
    ) -> Tuple[torch.Tensor, AsyncCommHandle]:
        """
        split grad_output if retain_out_sharded is False.
        """
        if self._retain_out_sharded or dist.get_world_size(self._process_group) <= 1:
            return grad_output, DUMMY_HANDLE_CONST

        return _split(grad_output, parallel_mode=self._parallel_mode, dim=-1)

    def output_hook(self, output: torch.Tensor, async_op: bool = False) -> Tuple[torch.Tensor, AsyncCommHandle]:
        """
        all gather output for head layer if retain_out_sharded is False.
        """
        if self._retain_out_sharded or dist.get_world_size(self._process_group) <= 1:
            return output, DUMMY_HANDLE_CONST

        return _gather(output, parallel_mode=self._parallel_mode, dim=-1)


class SequenceParallelCommunicator(TPCommunicator):
    def __init__(
        self, process_group: dist.ProcessGroup, role: LinearRole, save_total_input_as_activation: bool = False
    ) -> None:
        assert role in (LinearRole.COLUMN, LinearRole.ROW), f"Unknown linear role: {role}"

        self._process_group = process_group
        self._role = role

        self._save_total_input = save_total_input_as_activation

    def save_total_input(self) -> bool:
        return self._save_total_input

    def input_hook(
        self, _input: torch.Tensor, async_op: bool = False, is_forward: bool = True
    ) -> Tuple[torch.Tensor, AsyncCommHandle]:
        """
        all gather input only for column parallel linear when forward/backward.
        """
        # 1. world_size <= 1
        # 2. row parallel linear should not allgather input.
        # 3. column parallel linear should not allgather input if save_total_input_as_activation and backward is True.
        if (
            dist.get_world_size(self._process_group) <= 1
            or self._role == LinearRole.ROW
            or (is_forward is False and self._save_total_input)
        ):
            return _input, DUMMY_HANDLE_CONST

        return all_gather_raw(
            _input, process_group=self._process_group, async_op=async_op, gather_dim=__INPUT_GATHER_DIM
        )

    def grad_output_hook(
        self, grad_output: torch.Tensor, async_op: bool = False
    ) -> Tuple[torch.Tensor, AsyncCommHandle]:
        """
        all gather grad_output only for row parallel linear when backward.
        """
        if dist.get_world_size(self._process_group) <= 1 or self._role == LinearRole.COLUMN:
            return grad_output, DUMMY_HANDLE_CONST

        return all_gather_raw(
            grad_output, process_group=self._process_group, async_op=async_op, gather_dim=__INPUT_GATHER_DIM
        )

    def grad_input_hook(self, grad_input: torch.Tensor, async_op: bool = False) -> Tuple[torch.Tensor, AsyncCommHandle]:
        """
        reduce scatter grad_input only for column parallel linear when backward.
        """
        if dist.get_world_size(self._process_group) <= 1 or self._role == LinearRole.ROW:
            return grad_input, DUMMY_HANDLE_CONST

        return reduce_scatter_raw(grad_input, process_group=self._process_group, async_op=async_op)

    def output_hook(self, output: torch.Tensor, async_op: bool = False) -> Tuple[torch.Tensor, AsyncCommHandle]:
        """
        reduce scatter output only for row parallel linear when forward.
        """
        if dist.get_world_size(self._process_group) <= 1 or self._role == LinearRole.COLUMN:
            return output, DUMMY_HANDLE_CONST

        return reduce_scatter_raw(output, process_group=self._process_group, async_op=async_op)
