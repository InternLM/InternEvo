"""
communication for tensor/sequence parallel.
"""

from typing import Tuple
from enum import Enum
from abc import ABC, abstractmethod

import torch
from torch import distributed as dist

from internlm.core.parallel.comm.utils import (
    all_gather_raw,
    all_reduce_raw,
    reduce_scatter_raw,
    DUMMY_HANDLE_CONST,
    AsyncCommHandle,
)

# input gather dim
__INPUT_GATHER_DIM = -2  # shape: [batch, seqlen, dim] or [packlen, dim]


class CommRole(Enum):
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
    def gather_input(self, _input: torch.Tensor, async_op: bool = False) -> Tuple[torch.Tensor, AsyncCommHandle]:
        """
        gather input only for column parallel linear when forward/backward.
        """
        pass

    @abstractmethod
    def gather_grad_output(
        self, grad_output: torch.Tensor, async_op: bool = False
    ) -> Tuple[torch.Tensor, AsyncCommHandle]:
        """
        gather grad_output only for row parallel linear when backward.
        """
        pass

    @abstractmethod
    def reduce_grad_input(
        self, grad_input: torch.Tensor, async_op: bool = False
    ) -> Tuple[torch.Tensor, AsyncCommHandle]:
        """
        reduce grad_input only for column parallel linear when backward.
        """
        pass

    @abstractmethod
    def reduce_output(self, output: torch.Tensor, async_op: bool = False) -> Tuple[torch.Tensor, AsyncCommHandle]:
        """
        reduce output only for row parallel linear when forward.
        """
        pass



class TensorParallelCommunicator(TPCommunicator):
    def __init__(self, process_group: dist.ProcessGroup, role: CommRole) -> None:
        assert role in (CommRole.COLUMN, CommRole.ROW), f"Unknown sequence parallel role: {role}"

        self._process_group = process_group
        self._role = role

        self._save_total_input = False

    def save_total_input(self) -> bool:
        return self._save_total_input

    def gather_input(self, _input: torch.Tensor, async_op: bool = False) -> Tuple[torch.Tensor, AsyncCommHandle]:
        """
        tensor parallel should do nothing for input.
        """
        return _input, DUMMY_HANDLE_CONST

    def gather_grad_output(
        self, grad_output: torch.Tensor, async_op: bool = False
    ) -> Tuple[torch.Tensor, AsyncCommHandle]:
        """
        tensor parallel should do nothing for grad_output.
        """
        return grad_output, DUMMY_HANDLE_CONST

    def reduce_grad_input(
        self, grad_input: torch.Tensor, async_op: bool = False
    ) -> Tuple[torch.Tensor, AsyncCommHandle]:
        """
        all reduce grad_input only for column parallel linear when backward.
        """
        if dist.get_world_size(self._process_group) <= 1 or self._role == CommRole.ROW:
            return grad_input, DUMMY_HANDLE_CONST

        return all_reduce_raw(grad_input, process_group=self._process_group, async_op=async_op)

    def reduce_output(self, output: torch.Tensor, async_op: bool = False) -> Tuple[torch.Tensor, AsyncCommHandle]:
        """
        all reduce output only for row parallel linear when forward.
        """
        if dist.get_world_size(self._process_group) <= 1 or self._role == CommRole.COLUMN:
            return output, DUMMY_HANDLE_CONST

        return all_reduce_raw(output, process_group=self._process_group, async_op=async_op)


class SequenceParallelCommunicator(TPCommunicator):
    def __init__(
        self, process_group: dist.ProcessGroup, role: CommRole, save_total_input_as_activation: bool = False
    ) -> None:
        assert role in (CommRole.COLUMN, CommRole.ROW), f"Unknown sequence parallel role: {role}"

        self._process_group = process_group
        self._role = role

        self._save_total_input = save_total_input_as_activation

    def save_total_input(self) -> bool:
        return self._save_total_input

    def gather_input(self, _input: torch.Tensor, async_op: bool = False) -> Tuple[torch.Tensor, AsyncCommHandle]:
        """
        all gather input only for column parallel linear when forward/backward.
        """
        # 1. world_size <= 1
        # 2. row parallel linear should not allgather input.
        # 3. column parallel linear should not allgather input if save_total_input_as_activation is True.
        if dist.get_world_size(self._process_group) <= 1 or self._role == CommRole.ROW or self._save_total_input:
            return _input, DUMMY_HANDLE_CONST

        return all_gather_raw(
            _input, process_group=self._process_group, async_op=async_op, gather_dim=__INPUT_GATHER_DIM
        )

    def gather_grad_output(
        self, grad_output: torch.Tensor, async_op: bool = False
    ) -> Tuple[torch.Tensor, AsyncCommHandle]:
        """
        all gather grad_output only for row parallel linear when backward.
        """
        if dist.get_world_size(self._process_group) <= 1 or self._role == CommRole.COLUMN:
            return grad_output, DUMMY_HANDLE_CONST

        return all_gather_raw(
            grad_output, process_group=self._process_group, async_op=async_op, gather_dim=__INPUT_GATHER_DIM
        )

    def reduce_grad_input(
        self, grad_input: torch.Tensor, async_op: bool = False
    ) -> Tuple[torch.Tensor, AsyncCommHandle]:
        """
        reduce scatter grad_input only for column parallel linear when backward.
        """
        if dist.get_world_size(self._process_group) <= 1 or self._role == CommRole.ROW:
            return grad_input, DUMMY_HANDLE_CONST

        return reduce_scatter_raw(grad_input, process_group=self._process_group, async_op=async_op)

    def reduce_output(self, output: torch.Tensor, async_op: bool = False) -> Tuple[torch.Tensor, AsyncCommHandle]:
        """
        reduce scatter output only for row parallel linear when forward.
        """
        if dist.get_world_size(self._process_group) <= 1 or self._role == CommRole.COLUMN:
            return output, DUMMY_HANDLE_CONST

        return reduce_scatter_raw(output, process_group=self._process_group, async_op=async_op)
