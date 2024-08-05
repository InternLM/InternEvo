import torch
from internlm.model.registry import benchmark_initializer
from internlm.simulator.common import *

from .base_benchmark import UnitBench

BENCH_TYPE = "linear"


# @benchmark_initializer.register_module(module_name=BENCH_TYPE)
class UnitBenchLinear(UnitBench):
    test_loop = {
        "seq_len": [int(0.5 * K), 1 * K, 2 * K, 4 * K, 8 * K, 16 * K, 32 * K],
        "hidden_dim": [
            512,
            1024,
            2048,
            4096,
            5120,
            6144,
            8192,
            9216,
            10240,
            11264,
            12288,
        ],  # 7B, (13B, 20B), 30B, 65B, 123B
        "bias": [False],  # it is not work!! False,
        "dtype": [torch.bfloat16],
        "world_size": [1],
    }

    def __init__(self, seq_len, hidden_dim, bias, dtype) -> None:
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.q = torch.nn.Linear(
            hidden_dim, hidden_dim, bias=bias, device=f"cuda:{get_local_rank()}", dtype=dtype
        )  # (hidden_dim, hidden_dim)
        self.dtype = self.q.weight.element_size()
        self.x = torch.rand(1, seq_len, hidden_dim).to(self.q.weight)  # (bsz, seq_len, hidden_dim)

    def run(self):
        self.q(self.x)

    @staticmethod
    def gen_store_key(seq_len, hidden_dim, bias, dtype):
        if dtype in [torch.bfloat16, torch.float16]:
            element_size = 2
        elif dtype is torch.float32:
            element_size = 4
        else:
            assert False
        return element_size * seq_len * hidden_dim * hidden_dim

    def complexity(self):
        return self.dtype * self.seq_len * self.hidden_dim * self.hidden_dim
        # return f"{self.seq_len} * {self.hidden_dim} * {self.hidden_dim}"
