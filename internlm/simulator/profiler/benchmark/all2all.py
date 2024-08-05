import torch
import torch.distributed as dist

from internlm.model.registry import benchmark_initializer
from internlm.simulator.common import *

from .base_benchmark import UnitBench

BENCH_TYPE = "all2all"


# @benchmark_initializer.register_module(module_name=BENCH_TYPE)
class UnitBenchAll2ALL(UnitBench):
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
            self.output = torch.ones(self.world_size, self.unit_size, dtype=self.dtype).to(f"cuda:{get_local_rank()}")
            self.input = torch.ones(self.world_size, self.unit_size, dtype=self.dtype).to(f"cuda:{get_local_rank()}")
            self.input_buffer_size = self.input.element_size() * self.input.numel()

    def run(self):
        if self.output is None or not self.do_it:
            return

        handler = dist.all_to_all_single(self.output, self.input, async_op=self.async_op, group=self.group)
        if self.async_op:
            handler.wait()

    def complexity(self):
        return self.input_buffer_size


if __name__ == "__main__":
    # data = {
    #     "Latency_ms": [41.746, 62.982, 65.596, 101.968, 138.671, 159.773, 177.197, 190.415, 193.555, 194.056, 194.097,
    #                        193.776, 193.419, 193.679, 194.425, 194.462, 36.732, 55.592, 80.364, 100.85, 116.875, 133.242,
    #                        160.23, 178.519, 189.055, 193.55, 193.752, 193.717, 193.417, 193.686, 194.365, 194.416, 33.096,
    #                        48.456, 72.221, 97.357, 113.762, 125.266, 134.315, 164.453, 178.744, 187.352, 192.915, 193.512,
    #                        192.669, 193.47, 194.342, 194.218],
    #     "Cards": [64] * 16 + [128] * 16 + [256] * 16,
    #     "Data_MB": [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304,
    #                 8388608, 16777216] * 3
    # }
    cards_8_lat = [
        0.035442,
        0.038785,
        0.041076,
        0.063415,
        0.092584,
        0.151337,
        0.259346,
        0.482307,
        0.896747,
        1.737,
        3.255,
        6.431,
    ]
    cards_16_lat = [
        0.086889,
        0.113204,
        0.177494,
        0.271461,
        0.45525,
        0.84743,
        1.641,
        3.103,
        6.125,
        12.177,
        24.724,
        49.03,
    ]
    cards_32_lat = [
        0.102149,
        0.14717,
        0.230115,
        0.382689,
        0.681639,
        1.432,
        2.499,
        4.812,
        9.554,
        18.706,
        37.845,
        73.225,
    ]
    cards_64_lat = [
        0.115658,
        0.16165,
        0.259298,
        0.43826,
        0.822096,
        1.591,
        2.967,
        5.703,
        11.148,
        22.108,
        41.188,
        98.423,
    ]
    assert len(cards_8_lat) == len(cards_16_lat) == len(cards_32_lat) == len(cards_64_lat)
    samples = len(cards_8_lat)
    data = {
        "Latency_ms": cards_8_lat + cards_16_lat + cards_32_lat + cards_64_lat,
        "Cards": [8] * samples + [16] * samples + [32] * samples + [64] * samples,
        "Data_MB": [i * MB for i in [0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]] * 4,
    }
    segments = {
        "small": (64 * KB, 8 * MB),  # 64KB - 8MB, degree =2
        "large": (8 * MB, 1 * GB),  # 8MB - 1GB, degree=1
    }

    segments = {
        "all": (64 * KB, 1 * GB),
    }

    model = PolynomialModel(degree=2, data=data, segments=segments)
    model.predict(35 * MB)
    model.predict(1.2 * MB)
    model.predict(678 * MB)
