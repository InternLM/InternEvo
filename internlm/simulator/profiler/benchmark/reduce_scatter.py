import torch
import torch.distributed as dist

from internlm.model.registry import benchmark_initializer
from internlm.simulator.common import *

from .base_benchmark import UnitBench

BENCH_TYPE = "reduce_scatter"


# @benchmark_initializer.register_module(module_name=BENCH_TYPE)
class UnitBenchAllReduceScatter(UnitBench):
    test_loop = {
        "global_size": GLOBAL_ELEM_SIZES_LIST,
        "world_size": WORLD_SIZE_LIST,  # 7B, (13B, 20B), 30B, 65B, 123B
        "async_op": [False],  # it is not work!! False,
        "dtype": [torch.bfloat16],
    }

    def __init__(self, world_size, async_op, dtype, global_size=None, unit_size=None) -> None:
        assert global_size is None or unit_size is None

        self.unit_size = unit_size if unit_size else global_size // world_size  # elements_per_gpu
        self.world_size = world_size
        self.dtype = dtype
        self.async_op = async_op
        self.group = sub_process_groups[str(world_size)]
        self.do_it = dist.get_rank() in set(dist.get_process_group_ranks(self.group))

        if dist.get_world_size() < world_size:
            self.input = None
            self.output = None
        else:
            self.output = torch.ones(self.unit_size, dtype=self.dtype).to(f"cuda:{get_local_rank()}")
            self.input = torch.ones(self.world_size, self.unit_size, dtype=self.dtype).to(f"cuda:{get_local_rank()}")
            self.input_buffer_size = self.input.element_size() * self.input.numel()

    def run(self):
        if self.output is None or not self.do_it:
            return

        handler = dist.reduce_scatter_tensor(self.output, self.input, async_op=self.async_op, group=self.group)
        if self.async_op:
            handler.wait()

    def complexity(self):
        return self.input_buffer_size