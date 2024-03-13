"""
shard strategies for parallel
"""
from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc


def get_tensor_split_parallel_mode(self) -> ParallelMode:
    tp_mode = self.config.parallel.tensor.mode

    if tp_mode == "isp":
        return ParallelMode.WEIGHT
    else:
        return ParallelMode.TENSOR


def get_parallel_strategies_split_mode(linear_name: str) -> str:
    # TODO: should we move it to a parallel strategy package
    # TODO: 是否存在一种比根据名字来判断的方式？
    tp_mode = gpc.config.parallel.tensor.mode

    if linear_name in ("head", "output"):
        return "column"
    elif linear_name in ("wqkv", "wq", "wk", "wv", "wkv", "w1", "w3"):
        return "column"
    elif linear_name in ("wo", "out_proj", "w2") and tp_mode == "isp":
        return "column"
    elif linear_name in ("wo", "out_proj", "w2"):
        return "row"
    else:
        return "unknown"
