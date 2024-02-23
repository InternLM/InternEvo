#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# adopted from https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/context

import math
from abc import ABC, abstractmethod
from enum import Enum

import torch.distributed as dist

from internlm.utils.timeout import LLM_NCCL_TIMEOUT


# parallel modes
class ParallelMode(Enum):
    """This is an enumeration class containing all possible parallel modes."""

    GLOBAL = "global"

    # common parallel
    DATA = "data"

    # model parallel - containing tensor and pipeline parallel groups
    # this is added to facilitate amp and grad clipping in hybrid parallel
    MODEL = "model"

    # pipeline parallel
    PIPELINE = "pipe"

    # containing all ranks in tensor parallel
    TENSOR = "tensor"

    # zero1 parallel
    ZERO1 = "zero1"

    # runntime network test
    NETTEST = "nettest"

    # zero3-dp parallel
    # if fsdp is activated and size of fsdp-parallel-size is less than dp-parallel-size
    # then manual communication only happens between inter-fsdp-modules, while intra-modules reduction is done by fsdp
    ZERO3_DP = "zero3_dp"

    # expert parallel
    EXPERT = "expert"

    # expert data parallel
    EXPERT_DATA = "expert_data"

    # dummy mode, only used during mode construction
    DUMMY = "dummy"

    # weight parallel
    WEIGHT = "weight"

    # weight data parallel
    WEIGHT_DATA = "weight_data"

    # sequence parallel
    SEQUENCE = "sequence"

    # grouped query attention
    GQA = "gqa"


class ProcessGroupInitializer(ABC):
    """An object, knowing the parallelism configuration, that initializes parallel groups.

    Args:
        rank (int): The rank of current process.
        world_size (int): Size of whole communication world.
        weight_parallel_size (int): Size of model weight parallel.
        weight_data_parallel_size (int): Size of data parallel for common weight.
        sequence_parallel_size (int): Size of data sequence parallel.
        data_parallel_size (int): Size of data parallel.
        pipeline_parallel_size (int): Size of pipeline parallel.
        tensor_parallel_size (int): Size of tensor parallel.
        zero1_parallel_size (int): Size of zero1 parallel.
        nettest_parallel_size (int): Size of net testing parallel.
        expert_parallel_size (int): Size of expert parallel.
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        weight_parallel_size: int,
        weight_data_parallel_size: int,
        sequence_parallel_size: int,
        data_parallel_size: int,
        pipeline_parallel_size: int,
        tensor_parallel_size: int,
        zero1_parallel_size: int,
        nettest_parallel_size: int,
        expert_parallel_size: int,
    ):
        self.rank = rank
        self.world_size = world_size
        self.weight_parallel_size = weight_parallel_size
        self.weight_data_parallel_size = weight_data_parallel_size
        self.sequence_parallel_size = sequence_parallel_size
        self.data_parallel_size = data_parallel_size
        self.pipeline_parallel_size = pipeline_parallel_size
        self.tensor_parallel_size = tensor_parallel_size
        self.zero1_parallel_size = zero1_parallel_size
        self.nettest_parallel_size = nettest_parallel_size
        self.expert_parallel_size = expert_parallel_size

        assert sequence_parallel_size == tensor_parallel_size
        super().__init__()

    @abstractmethod
    def init_dist_group(self, use_cpu: bool = False):
        pass


class Initializer_Pipeline(ProcessGroupInitializer):
    """A ProcessGroupInitializer for pipeline parallelism.

    Args:
        rank (int): The rank of current process.
        world_size (int): Size of whole communication world.
        weight_parallel_size (int): Size of model weight parallel.
        weight_data_parallel_size (int): Size of data parallel for common weight.
        sequence_parallel_size (int): Size of data sequence parallel.
        data_parallel_size (int): Size of data parallel.
        pipeline_parallel_size (int): Size of pipeline parallel.
        tensor_parallel_size (int): Size of tensor parallel.
        zero1_parallel_size (int): Size of zero1 parallel.
        nettest_parallel_size (int): Size of net testing parallel.
        expert_parallel_size (int): Size of expert parallel.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_pp_group = self.world_size // self.pipeline_parallel_size

        assert self.world_size % self.pipeline_parallel_size == 0

    def init_dist_group(self, use_cpu: bool = False):
        """Initialize pipeline parallel groups, and assign local_ranks and groups to each gpu.

        Returns:
            List[Tuple (local_rank, group_world_size, process_group, ranks_in_group, mode)]:
                A Pipeline parallelism's information in list of tuples.

        n=16 tp/sp=4 pp=2 dp=2 wp=8
        wp grops: [0-7] [8-15]
        data groups: [0,4] [1,5] [2,6] [3,7]
                     [8,12] [9,13] [10,14] [11,15]
        pp groups: [0,8] [1,9] [2,10] [3,11] [4,12] [5,13] [6,14] [7,15]

        n=16 tp/sp=4 pp=2 dp=2 wp=2
        wp grops: [0-1] [2-3] [4-5] [6-7] [8-9] [10-11] [12-13] [14-15]
        data groups: [0,4] [1,5] [2,6] [3,7]
                     [8,12] [9,13] [10,14] [11,15]
        pp groups: [0,8] [1,9] [2,10] [3,11] [4,12] [5,13] [6,14] [7,15]
        """
        local_rank = None
        ranks_in_group = None
        process_group = None
        cpu_group = None
        group_world_size = None
        mode = ParallelMode.PIPELINE

        for i in range(self.num_pp_group):
            ranks = [i + j * self.num_pp_group for j in range(self.pipeline_parallel_size)]
            pipe_group_size = len(ranks)
            pipe_group = dist.new_group(ranks, timeout=LLM_NCCL_TIMEOUT)
            if use_cpu:
                group_cpu = (
                    dist.new_group(ranks, backend="gloo", timeout=LLM_NCCL_TIMEOUT)
                    if dist.get_backend() != "gloo"
                    else pipe_group
                )
            else:
                group_cpu = None

            if self.rank in ranks:
                local_rank = ranks.index(self.rank)
                group_world_size = pipe_group_size
                process_group = pipe_group
                cpu_group = group_cpu
                ranks_in_group = ranks

        return local_rank, group_world_size, process_group, cpu_group, ranks_in_group, mode


class Initializer_Tensor(ProcessGroupInitializer):
    """A ProcessGroupInitializer for tensor parallelism.

    Args:
        rank (int): The rank of current process.
        world_size (int): Size of whole communication world.
        weight_parallel_size (int): Size of model weight parallel.
        weight_data_parallel_size (int): Size of data parallel for common weight.
        sequence_parallel_size (int): Size of data sequence parallel.
        data_parallel_size (int): Size of data parallel.
        pipeline_parallel_size (int): Size of pipeline parallel.
        tensor_parallel_size (int): Size of tensor parallel.
        zero1_parallel_size (int): Size of zero1 parallel.
        nettest_parallel_size (int): Size of net testing parallel.
        expert_parallel_size (int): Size of expert parallel.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_tensor_parallel_group = self.world_size // self.tensor_parallel_size

        assert self.world_size % self.tensor_parallel_size == 0

    def init_dist_group(self, use_cpu: bool = False):
        """Initialize tensor parallel groups, and assign local_ranks and groups to each gpu.

        Returns:
            Tuple (local_rank, group_world_size, process_group, ranks_in_group, mode):
                A Tensor parallelism's information tuple.
        """
        local_rank = None
        ranks_in_group = None
        process_group = None
        cpu_group = None
        group_world_size = None
        mode = ParallelMode.TENSOR

        for i in range(self.num_tensor_parallel_group):
            ranks = [i * self.tensor_parallel_size + j for j in range(self.tensor_parallel_size)]
            group = dist.new_group(ranks, timeout=LLM_NCCL_TIMEOUT)
            if use_cpu:
                group_cpu = (
                    dist.new_group(ranks, backend="gloo", timeout=LLM_NCCL_TIMEOUT)
                    if dist.get_backend() != "gloo"
                    else group
                )
            else:
                group_cpu = None

            if self.rank in ranks:
                local_rank = ranks.index(self.rank)
                group_world_size = len(ranks)
                process_group = group
                cpu_group = group_cpu
                ranks_in_group = ranks

        return local_rank, group_world_size, process_group, cpu_group, ranks_in_group, mode


class Initializer_Zero1(ProcessGroupInitializer):
    """A ProcessGroupInitializer for zero-1 parallelism.

    Args:
        rank (int): The rank of current process.
        world_size (int): Size of whole communication world.
        weight_parallel_size (int): Size of model weight parallel.
        weight_data_parallel_size (int): Size of data parallel for common weight.
        sequence_parallel_size (int): Size of data sequence parallel.
        data_parallel_size (int): Size of data parallel.
        pipeline_parallel_size (int): Size of pipeline parallel.
        tensor_parallel_size (int): Size of tensor parallel.
        zero1_parallel_size (int): Size of zero1 parallel.
        nettest_parallel_size (int): Size of net testing parallel.
        expert_parallel_size (int): Size of expert parallel.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tensor_zero1_size = self.tensor_parallel_size * self.zero1_parallel_size
        self.ranks_num_per_pp = self.world_size // self.pipeline_parallel_size
        self.num_tensor_zero1_parallel_group = self.ranks_num_per_pp // self.tensor_zero1_size

        assert self.world_size % (self.tensor_parallel_size * self.zero1_parallel_size) == 0
        assert self.world_size % self.pipeline_parallel_size == 0

    def init_dist_group(self, use_cpu: bool = False):
        """Initialize zero1 parallel groups, and assign local_ranks and groups to each gpu.

        Returns:
            Tuple (local_rank, group_world_size, process_group, ranks_in_group, mode):
                A zero1 parallelism's information tuple.

        n=16 tp/sp=4 pp=2 dp=2 zero1=2
        tp/sp grops: [0-3] [4-7] [8-11] [12-15]
        data groups: [0,4] [1,5] [2,6] [3,7]
                     [8,12] [9,13] [10,14] [11,15]
        pp groups: [0,8] [1,9] [2,10] [3,11] [4,12] [5,13] [6,14] [7,15]
        zero1 groups: [0,4] [1,5] [2,6] [3,7]
                      [8,12] [9,13] [10,14] [11,15]

        n=16 tp/sp=2 pp=2 dp=4 zero1=2
        tp/sp grops: [0-1] [2-3] [4-5] [6-7] [8-9] [10-11] [12-13] [14-15]
        data groups: [0,2,4,6] [1,3,5,7]
                     [8,10,12,14] [9,11,13,15]
        pp groups: [0,8] [1,9] [2,10] [3,11] [4,12] [5,13] [6,14] [7,15]
        zero1 groups: [0,2] [1,3] [4,6] [5,7]
                      [8,10] [9,11] [12,14] [13,15]
        """
        local_rank = None
        ranks_in_group = None
        process_group = None
        cpu_group = None
        group_world_size = None
        mode = ParallelMode.ZERO1

        for i in range(self.pipeline_parallel_size):
            for j in range(self.num_tensor_zero1_parallel_group):
                for k in range(self.tensor_parallel_size):
                    ranks = [
                        i * self.ranks_num_per_pp + j * self.tensor_zero1_size + k + m * self.tensor_parallel_size
                        for m in range(self.zero1_parallel_size)
                    ]
                    group = dist.new_group(ranks, timeout=LLM_NCCL_TIMEOUT)
                    if use_cpu:
                        group_cpu = (
                            dist.new_group(ranks, backend="gloo", timeout=LLM_NCCL_TIMEOUT)
                            if dist.get_backend() != "gloo"
                            else group
                        )
                    else:
                        group_cpu = None

                    if self.rank in ranks:
                        local_rank = ranks.index(self.rank)
                        group_world_size = len(ranks)
                        process_group = group
                        cpu_group = group_cpu
                        ranks_in_group = ranks

        return local_rank, group_world_size, process_group, cpu_group, ranks_in_group, mode


class Initializer_Zero1_ISP(ProcessGroupInitializer):
    """A ProcessGroupInitializer for zero-1 parallelism.

    Args:
        rank (int): The rank of current process.
        world_size (int): Size of whole communication world.
        weight_parallel_size (int): Size of model weight parallel.
        weight_data_parallel_size (int): Size of data parallel for common weight.
        sequence_parallel_size (int): Size of data sequence parallel.
        data_parallel_size (int): Size of data parallel.
        pipeline_parallel_size (int): Size of pipeline parallel.
        tensor_parallel_size (int): Size of tensor parallel.
        zero1_parallel_size (int): Size of zero1 parallel.
        nettest_parallel_size (int): Size of net testing parallel.
        expert_parallel_size (int): Size of expert parallel.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_zero1_size = self.weight_parallel_size * self.zero1_parallel_size
        self.ranks_num_per_pp = self.world_size // self.pipeline_parallel_size
        self.num_weight_zero1_parallel_group = self.ranks_num_per_pp // self.weight_zero1_size

        assert self.world_size % (self.pipeline_parallel_size * self.zero1_parallel_size) == 0
        assert self.world_size % self.weight_zero1_size == 0

    def init_dist_group(self, use_cpu: bool = False):
        """Initialize zero1 parallel groups, and assign local_ranks and groups to each gpu.

        Returns:
            Tuple (local_rank, group_world_size, process_group, ranks_in_group, mode):
                A zero1 parallelism's information tuple.

        n=32 wp=8 sp=4 zo1=2
        wp grops: [0-7] [8-15] [16-23] [24-31]
        zo1 groups: [0,8] [1,9] [2,10] [3,11] [4,12] [5,13] [6,14] [7,15]
                    [16,24] [17,25] [18,26] [19,27] [20,28] [21,29] [22,30] [23,31]

        n=16 tp/sp=4 pp=2 dp=2 wp=8 wdp=1 zero1=1
        wp grops: [0-7] [8-15]
        data groups: [0,4] [1,5] [2,6] [3,7]
                     [8,12] [9,13] [10,14] [11,15]
        wdp groups: [...]

        n=16 tp/sp=4 pp=2 dp=2 wp=2 wdp=4 zero1=2
        wp grops: [0-1] [2-3] [4-5] [6-7] [8-9] [10-11] [12-13] [14-15]
        data groups: [0,4] [1,5] [2,6] [3,7]
                     [8,12] [9,13] [10,14] [11,15]
        pp groups: [0,8] [1,9] [2,10] [3,11] [4,12] [5,13] [6,14] [7,15]
        wdp groups: [0,2,4,6] [1,3,5,7]
                    [8,10,12,14] [9,11,13,15]
        zero1 groups: [0,2] [1,3] [4,6] [5,7]
                      [8,10] [9,11] [12,14] [13,15]
        zero1=4: [0,2,4,6] [1,3,5,7] [8,10,12,14] [9,11,13,15]
        """
        local_rank = None
        ranks_in_group = None
        process_group = None
        cpu_group = None
        group_world_size = None
        mode = ParallelMode.ZERO1

        for i in range(self.pipeline_parallel_size):
            for j in range(self.num_weight_zero1_parallel_group):
                for k in range(self.weight_parallel_size):
                    ranks = [
                        i * self.ranks_num_per_pp + j * self.weight_zero1_size + k + m * self.weight_parallel_size
                        for m in range(self.zero1_parallel_size)
                    ]
                    group = dist.new_group(ranks, timeout=LLM_NCCL_TIMEOUT)
                    if use_cpu:
                        group_cpu = (
                            dist.new_group(ranks, backend="gloo", timeout=LLM_NCCL_TIMEOUT)
                            if dist.get_backend() != "gloo"
                            else group
                        )
                    else:
                        group_cpu = None

                    if self.rank in ranks:
                        local_rank = ranks.index(self.rank)
                        group_world_size = len(ranks)
                        process_group = group
                        cpu_group = group_cpu
                        ranks_in_group = ranks

        return local_rank, group_world_size, process_group, cpu_group, ranks_in_group, mode


class Initializer_Nettest(ProcessGroupInitializer):
    """A ProcessGroupInitializer for network test, especailly for NCCL.

    Args:
        rank (int): The rank of current process.
        world_size (int): Size of whole communication world.
        nettest_parallel_size (int): Size of a network test group.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_nettest_group = math.ceil(self.world_size / self.nettest_parallel_size)

    def init_dist_group(self, use_cpu: bool = False):
        """Initialize tensor parallel groups, and assign local_ranks and groups to each gpu.

        Returns:
            Tuple (local_rank, group_world_size, process_group, ranks_in_group, mode):
                A Tensor parallelism's information tuple.
        """
        local_rank = None
        ranks_in_group = None
        process_group = None
        cpu_group = None
        group_world_size = None
        mode = ParallelMode.NETTEST

        for i in range(self.num_nettest_group):
            ranks = []
            for j in range(self.nettest_parallel_size):
                rank = i * self.nettest_parallel_size + j
                if rank < self.world_size:
                    ranks.append(rank)
            group = dist.new_group(ranks, timeout=LLM_NCCL_TIMEOUT)
            if use_cpu:
                group_cpu = (
                    dist.new_group(ranks, backend="gloo", timeout=LLM_NCCL_TIMEOUT)
                    if dist.get_backend() != "gloo"
                    else group
                )
            else:
                group_cpu = None

            if self.rank in ranks:
                local_rank = ranks.index(self.rank)
                group_world_size = len(ranks)
                process_group = group
                cpu_group = group_cpu
                ranks_in_group = ranks

        return local_rank, group_world_size, process_group, cpu_group, ranks_in_group, mode


class Initializer_Expert_Data(ProcessGroupInitializer):
    """A ProcessGroupInitializer for expert data parallelism.

    Args:
        rank (int): The rank of current process.
        world_size (int): Size of whole communication world.
        data_parallel_size (int): Size of data parallel.
        pipeline_parallel_size (int): Size of pipeline parallel.
        tensor_parallel_size (int): Size of tensor parallel.
        zero1_parallel_size (int): Size of zero-1 parallel.
        expert_parallel_size (int): Size of expert parallel.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ranks_num_per_pp = self.world_size // self.pipeline_parallel_size
        assert self.data_parallel_size % self.expert_parallel_size == 0

    def _get_expert_parallel_ranks(self):
        """
        Create expert and data parallel groups
        Example: world_size = 8, tensor_parallel_size = 2, expert_parallel_size = 2
        model_parallel_group = [0,1], [2,3], [4,5], [6,7]
        data_parallel_group = [0,2,4,6],                [1,3,5,7]
        expert_parallel_group = [0,2], [4,6],           [1,3], [5,7]
        expert_data_parallel_group = [0,4], [2,6],      [1,5], [3,7]
        """
        data_parallel_groups = []
        for i in range(self.pipeline_parallel_size):
            for j in range(self.sequence_parallel_size):
                data_parallel_groups.append(
                    [
                        i * self.ranks_num_per_pp + j + k * self.sequence_parallel_size
                        for k in range(self.data_parallel_size)
                    ]
                )

        expert_parallel_groups = []
        expert_data_parallel_groups = []
        for dp_ranks in data_parallel_groups:
            # partition of expert parallel group, e.g. [0,2], [4,6]
            part_ep_group = []
            for i in range(0, self.data_parallel_size, self.expert_parallel_size):
                part_ep_group.append(dp_ranks[i : i + self.expert_parallel_size])
            expert_parallel_groups.extend(part_ep_group)

            for expert_dp_ranks in zip(*part_ep_group):
                expert_data_parallel_groups.append(list(expert_dp_ranks))

        return expert_parallel_groups, expert_data_parallel_groups

    def init_dist_group(self, use_cpu: bool = False):
        """Initialize expert parallel  and expert data groups, and assign local_ranks and groups to each gpu.

        Returns:
            list: [(local_rank, group_world_size, process_group, ranks_in_group, mode), ...]:
                A length 2 list consists of expert parallelism's and expert data parallelism's information tuple.
        """
        local_rank = None
        ranks_in_group = None
        process_group = None
        cpu_group = None
        group_world_size = None
        expert_parallel_groups, expert_data_parallel_groups = self._get_expert_parallel_ranks()

        groups = []
        for ranks in expert_parallel_groups:
            group = dist.new_group(ranks, timeout=LLM_NCCL_TIMEOUT)
            if use_cpu:
                group_cpu = (
                    dist.new_group(ranks, backend="gloo", timeout=LLM_NCCL_TIMEOUT)
                    if dist.get_backend() != "gloo"
                    else group
                )
            else:
                group_cpu = None
            if self.rank in ranks:
                local_rank = ranks.index(self.rank)
                group_world_size = len(ranks)
                process_group = group
                cpu_group = group_cpu
                ranks_in_group = ranks
                groups.append(
                    (local_rank, group_world_size, process_group, cpu_group, ranks_in_group, ParallelMode.EXPERT)
                )

        for ranks in expert_data_parallel_groups:
            group = dist.new_group(ranks, timeout=LLM_NCCL_TIMEOUT)
            if use_cpu:
                group_cpu = (
                    dist.new_group(ranks, backend="gloo", timeout=LLM_NCCL_TIMEOUT)
                    if dist.get_backend() != "gloo"
                    else group
                )
            else:
                group_cpu = None
            if self.rank in ranks:
                local_rank = ranks.index(self.rank)
                group_world_size = len(ranks)
                process_group = group
                cpu_group = group_cpu
                ranks_in_group = ranks
                groups.append(
                    (local_rank, group_world_size, process_group, cpu_group, ranks_in_group, ParallelMode.EXPERT_DATA)
                )

        return groups


class Initializer_Zero3_dp(ProcessGroupInitializer):
    """A ProcessGroupInitializer for data parallelism.

    Args:
        rank (int): The rank of current process.
        world_size (int): Size of whole communication world.
        data_parallel_size (int): Size of data parallel.
        pipeline_parallel_size (int): Size of pipeline parallel.
        tensor_parallel_size (int): Size of tensor parallel.
        zero1_parallel_size (int): Size of zero1 parallel.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.data_parallel_size % self.zero1_parallel_size == 0

        # the only difference between this initializer and DP_initializer
        # when FSDP is enabled, only corresponding pairs are in the same actual DP group due to parameter sharding
        # eg: when zero=4 and dp=8
        #     no fsdp: rank [0-7] share same model paramters, and [0-3], [4-7] are two separate zero group
        #        fsdp: params of (0, 4), (1, 5), (2, 6), (3, 7) are the same actually

        self.data_parallel_size //= self.zero1_parallel_size
        self.rank_num_per_dp_group = self.world_size // self.data_parallel_size

        assert self.world_size % self.data_parallel_size == 0

    def init_dist_group(self, use_cpu: bool = False):
        """Initialize data parallel groups, and assign local_ranks and groups to each gpu.

        Returns:
            Tuple (local_rank, group_world_size, process_group, ranks_in_group, mode):
                A Data parallelism's information tuple.
        """
        local_rank = None
        ranks_in_group = None
        process_group = None
        cpu_group = None
        group_world_size = None
        mode = ParallelMode.ZERO3_DP

        for i in range(self.rank_num_per_dp_group):
            ranks = [i + j * self.rank_num_per_dp_group for j in range(self.data_parallel_size)]
            group = dist.new_group(ranks)
            if use_cpu:
                group_cpu = dist.new_group(ranks, backend="gloo") if dist.get_backend() != "gloo" else group
            else:
                group_cpu = None

            if self.rank in ranks:
                local_rank = ranks.index(self.rank)
                group_world_size = len(ranks)
                process_group = group
                cpu_group = group_cpu
                ranks_in_group = ranks

        return local_rank, group_world_size, process_group, cpu_group, ranks_in_group, mode


class Initializer_Weight(ProcessGroupInitializer):
    """A ProcessGroupInitializer for model weight parallelism.

    Args:
        rank (int): The rank of current process.
        world_size (int): Size of whole communication world.
        weight_parallel_size (int): Size of model weight parallel.
        weight_data_parallel_size (int): Size of data parallel for common weight.
        sequence_parallel_size (int): Size of data sequence parallel.
        data_parallel_size (int): Size of data parallel.
        pipeline_parallel_size (int): Size of pipeline parallel.
        tensor_parallel_size (int): Size of tensor parallel.
        zero1_parallel_size (int): Size of zero1 parallel.
        nettest_parallel_size (int): Size of net testing parallel.
        expert_parallel_size (int): Size of expert parallel.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_weight_parallel_group = self.world_size // self.weight_parallel_size

        assert self.world_size % self.weight_parallel_size == 0

    def init_dist_group(self, use_cpu: bool = False):
        """Initialize model weight parallel groups, and assign local_ranks and groups to each gpu.

        Returns:
            Tuple (local_rank, group_world_size, process_group, ranks_in_group, mode):
                A Weight parallelism's information tuple.
        """
        local_rank = None
        ranks_in_group = None
        process_group = None
        cpu_group = None
        group_world_size = None
        mode = ParallelMode.WEIGHT

        for i in range(self.num_weight_parallel_group):
            ranks = [i * self.weight_parallel_size + j for j in range(self.weight_parallel_size)]
            group = dist.new_group(ranks, timeout=LLM_NCCL_TIMEOUT)
            if use_cpu:
                group_cpu = (
                    dist.new_group(ranks, backend="gloo", timeout=LLM_NCCL_TIMEOUT)
                    if dist.get_backend() != "gloo"
                    else group
                )
            else:
                group_cpu = None

            if self.rank in ranks:
                local_rank = ranks.index(self.rank)
                group_world_size = len(ranks)
                process_group = group
                cpu_group = group_cpu
                ranks_in_group = ranks

        return local_rank, group_world_size, process_group, cpu_group, ranks_in_group, mode


class Initializer_Data(ProcessGroupInitializer):
    """A ProcessGroupInitializer for data parallelism.

    Args:
        rank (int): The rank of current process.
        world_size (int): Size of whole communication world.
        weight_parallel_size (int): Size of model weight parallel.
        weight_data_parallel_size (int): Size of data parallel for common weight.
        sequence_parallel_size (int): Size of data sequence parallel.
        data_parallel_size (int): Size of data parallel.
        pipeline_parallel_size (int): Size of pipeline parallel.
        tensor_parallel_size (int): Size of tensor parallel.
        zero1_parallel_size (int): Size of zero1 parallel.
        nettest_parallel_size (int): Size of net testing parallel.
        expert_parallel_size (int): Size of expert parallel.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_dp_group = self.pipeline_parallel_size * self.sequence_parallel_size
        self.ranks_num_per_pp = self.world_size // self.pipeline_parallel_size

        assert self.world_size % self.data_parallel_size == 0
        assert self.world_size % self.sequence_parallel_size == 0
        assert self.world_size % self.pipeline_parallel_size == 0

    def init_dist_group(self, use_cpu: bool = False):
        """Initialize data parallel groups, and assign local_ranks and groups to each gpu.

        Returns:
            Tuple (local_rank, group_world_size, process_group, ranks_in_group, mode):
                A Data parallelism's information tuple.

        n=16 tp/sp=4 pp=2 dp=2 wp=8
        wp grops: [0-7] [8-15]
        data groups: [0,4] [1,5] [2,6] [3,7]
                     [8,12] [9,13] [10,14] [11,15]
        """
        local_rank = None
        ranks_in_group = None
        process_group = None
        cpu_group = None
        group_world_size = None
        mode = ParallelMode.DATA

        for i in range(self.pipeline_parallel_size):
            for j in range(self.sequence_parallel_size):
                ranks = [
                    i * self.ranks_num_per_pp + j + k * self.sequence_parallel_size
                    for k in range(self.data_parallel_size)
                ]
                group = dist.new_group(ranks, timeout=LLM_NCCL_TIMEOUT)
                if use_cpu:
                    group_cpu = (
                        dist.new_group(ranks, backend="gloo", timeout=LLM_NCCL_TIMEOUT)
                        if dist.get_backend() != "gloo"
                        else group
                    )
                else:
                    group_cpu = None

                if self.rank in ranks:
                    local_rank = ranks.index(self.rank)
                    group_world_size = len(ranks)
                    process_group = group
                    cpu_group = group_cpu
                    ranks_in_group = ranks

        return local_rank, group_world_size, process_group, cpu_group, ranks_in_group, mode


class Initializer_Weight_Data(ProcessGroupInitializer):
    """A ProcessGroupInitializer for common weight's data parallelism.

    Args:
        rank (int): The rank of current process.
        world_size (int): Size of whole communication world.
        weight_parallel_size (int): Size of model weight parallel.
        weight_data_parallel_size (int): Size of data parallel for common weight.
        sequence_parallel_size (int): Size of data sequence parallel.
        data_parallel_size (int): Size of data parallel.
        pipeline_parallel_size (int): Size of pipeline parallel.
        tensor_parallel_size (int): Size of tensor parallel.
        zero1_parallel_size (int): Size of zero1 parallel.
        nettest_parallel_size (int): Size of net testing parallel.
        expert_parallel_size (int): Size of expert parallel.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_wdp_group_per_pp = self.world_size // self.pipeline_parallel_size // self.weight_data_parallel_size
        self.ranks_num_per_pp = self.world_size // self.pipeline_parallel_size

        assert self.world_size % self.pipeline_parallel_size == 0
        assert self.world_size % (self.pipeline_parallel_size * self.weight_data_parallel_size) == 0

    def init_dist_group(self, use_cpu: bool = False):
        """Initialize weight's data parallel groups, and assign local_ranks and groups to each gpu.

        Returns:
            Tuple (local_rank, group_world_size, process_group, ranks_in_group, mode):
                A WEIGHT_DATA parallelism's information tuple.

        n=32 wp=8 sp=4 zo1=2 with nopp
        wp grops: [0-7] [8-15] [16-23] [24-31]
        weight data groups: [0,8,16,24] [1,9,17,25] [2,10,18,26] [3,11,19,27]
                            [4,12,20,28] [5,13,21,29] [6,14,22,30] [7,15,23,31]

        n=16 tp/sp=4 pp=2 dp=2 wp=8 wdp=1
        wp grops: [0-7] [8-15]
        data groups: [0,4] [1,5] [2,6] [3,7]
                     [8,12] [9,13] [10,14] [11,15]
        wdp groups: [...]

        n=16 tp/sp=4 pp=2 dp=2 wp=2 wdp=4
        wp grops: [0-1] [2-3] [4-5] [6-7] [8-9] [10-11] [12-13] [14-15]
        data groups: [0,4] [1,5] [2,6] [3,7]
                     [8,12] [9,13] [10,14] [11,15]
        pp groups: [0,8] [1,9] [2,10] [3,11] [4,12] [5,13] [6,14] [7,15]
        wdp groups: [0,2,4,6] [1,3,5,7]
                    [8,10,12,14] [9,11,13,15]
        """
        local_rank = None
        ranks_in_group = None
        process_group = None
        cpu_group = None
        group_world_size = None
        mode = ParallelMode.WEIGHT_DATA

        for i in range(self.pipeline_parallel_size):
            for j in range(self.num_wdp_group_per_pp):
                ranks = [
                    i * self.ranks_num_per_pp + j + k * self.weight_parallel_size
                    for k in range(self.weight_data_parallel_size)
                ]
                group = dist.new_group(ranks, timeout=LLM_NCCL_TIMEOUT)
                if use_cpu:
                    group_cpu = (
                        dist.new_group(ranks, backend="gloo", timeout=LLM_NCCL_TIMEOUT)
                        if dist.get_backend() != "gloo"
                        else group
                    )
                else:
                    group_cpu = None

                if self.rank in ranks:
                    local_rank = ranks.index(self.rank)
                    group_world_size = len(ranks)
                    process_group = group
                    cpu_group = group_cpu
                    ranks_in_group = ranks

        return local_rank, group_world_size, process_group, cpu_group, ranks_in_group, mode


class Initializer_GQA(ProcessGroupInitializer):
    """A ProcessGroupInitializer for allreduce kv gradients with common attention head.

    Args:
        rank (int): The rank of current process.
        world_size (int): Size of whole communication world.
        weight_parallel_size (int): Size of model weight parallel.
        weight_data_parallel_size (int): Size of data parallel for common weight.
        sequence_parallel_size (int): Size of data sequence parallel.
        data_parallel_size (int): Size of data parallel.
        pipeline_parallel_size (int): Size of pipeline parallel.
        tensor_parallel_size (int): Size of tensor parallel.
        zero1_parallel_size (int): Size of zero1 parallel.
        nettest_parallel_size (int): Size of net testing parallel.
        expert_parallel_size (int): Size of expert parallel.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # TODO: should adapt to general case
        self.num_kv_attention_heads = 8
        self.NUM_ATTENTION_HEAD = 32
        self.kv_head_repeats_num = self.NUM_ATTENTION_HEAD // self.num_kv_attention_heads
        self.num_kv_group_per_tp = self.num_kv_attention_heads
        self.num_kv_groups = self.num_kv_group_per_tp * self.data_parallel_size

        assert self.world_size % self.tensor_parallel_size == 0
        assert self.world_size % (self.pipeline_parallel_size * self.tensor_parallel_size) == 0
        assert self.pipeline_parallel_size == 1

    def init_dist_group(self, use_cpu: bool = False):
        """Initialize weight's data parallel groups, and assign local_ranks and groups to each gpu.

        Returns:
            Tuple (local_rank, group_world_size, process_group, ranks_in_group, mode):
                A WEIGHT_DATA parallelism's information tuple.

        n=128 sp=32 wp=64 zo1=1 with nopp
        sp groups: [0-31] [32-63] [64-95] [96-127]
        wp groups: [0-63] [64-127]
        kv_head groups: [0,8,16,24] [1,9,17,25] [2,10,18,26] [3,11,19,27]
                        [4,12,20,28] [5,13,21,29] [6,14,22,30] [7,15,23,31]
                        [32,40,48,56] [33,41,49,57] [34,42,50,58] [35,43,51,59]
                        [36,44,52,60] [37,45,53,61] [38,46,54,62] [39,47,55,63]
                        ...
                        ...
        """
        local_rank = None
        ranks_in_group = None
        process_group = None
        cpu_group = None
        group_world_size = None
        mode = ParallelMode.GQA

        # TODO: consider PP
        for i in range(self.data_parallel_size):
            for j in range(self.num_kv_group_per_tp):
                ranks = [
                    i * self.tensor_parallel_size + j + k * self.num_kv_attention_heads
                    for k in range(self.kv_head_repeats_num)
                ]
                group = dist.new_group(ranks, timeout=LLM_NCCL_TIMEOUT)
                if use_cpu:
                    group_cpu = (
                        dist.new_group(ranks, backend="gloo", timeout=LLM_NCCL_TIMEOUT)
                        if dist.get_backend() != "gloo"
                        else group
                    )
                else:
                    group_cpu = None

                if self.rank in ranks:
                    local_rank = ranks.index(self.rank)
                    group_world_size = len(ranks)
                    process_group = group
                    cpu_group = group_cpu
                    ranks_in_group = ranks

        return local_rank, group_world_size, process_group, cpu_group, ranks_in_group, mode
