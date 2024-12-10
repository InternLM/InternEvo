import math
import inspect
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple, Optional, Any
import torch
import torch.distributed as dist



class Range:
    """
    A range represents a start and end points for indexing a shard
    from a full tensor.
    """

    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.size = end - start

    def normalize(self, start=0):
        return Range(start, start + self.size)

    def __str__(self):
        return "%d,%d [%d]" % (self.start, self.end, self.size)

    def __len__(self):
        return self.end - self.start

    def __repr__(self) -> str:
        return "Range(%d,%d [%d])" % (self.start, self.end, self.size)

@dataclass
class OptimizerStateSpec:
    """This class represents mapping between local flattened 1D tensor
    and global original DTensor in DOptimzier, it is used for
    loading or saving optimizer states using vescale.checkpoint (PyTorch DCP)
    and load-time checkpoint resharding when changing tp size or dp size.

    For example, a linear layer in Vescale is DTensor(size=[1024, 1024])
    It first divides into two parts along dim=0 with tensor parallel size = 2

    tensor_part_0 = DTensor(size=[512, 1024])
    tensor_part_1 = DTensor(size=[512, 1024])

    Then each part's optimizer states are initalized in DOptimizer sepearately

    Assume dp=2
    For process with dp=0 tp=0, the flatten tensor is torch.Tensor(size=[262144])
    global_shape=(1024, 1024), local_shape=(256, 1024), global_offset=(0, 0) local=torch.Tensor(size=[262144]).view(local_shape)

    For process with dp=1 tp=0, the flatten tensor is torch.Tensor(size=[262144])
    global_shape=(1024, 1024), local_shape=(256, 1024), global_offset=(256, 0) local=torch.Tensor(size=[262144]).view(local_shape)

    For process with dp=0 tp=1, the flatten tensor is torch.Tensor(size=[262144])
    mapping to [512:768, 0:1024] in original DTensor
    global_shape=(1024, 1024), local_shape=(256, 1024), global_offset=(512, 0) local=torch.Tensor(size=[262144]).view(local_shape)

    For process with dp=1 tp=1, the flatten tensor is torch.Tensor(size=[262144])
    global_shape=(1024, 1024), local_shape=(256, 1024), global_offset=(768, 0) local=torch.Tensor(size=[262144]).view(local_shape)
    """

    # The original DTensor shape
    global_shape: Tuple[int]
    # The local tensor shape ***before flattened into 1D tensor***
    local_shape: Tuple[int]
    # The local tensor's offset with respect to origianl DTensor
    global_offset: Tuple[int]
    # The unflattened local tensor after create view using local_shape on the flattened 1D Tensor in DOptimizer
    # NOTE: In order to support TP resharding and state cross dp ranks, we defer the reshaping from 1D to local_shape
    # to generate saving plan using vescale.checkpoint (PyTorch DCP)
    local_tensor: torch.Tensor
    # If the current optimizer state is sharded by multiple dp ranks,
    # we should record all ranks and their ranges
    dp_ranks_ranges: Optional[Dict[int, Range]]

def convert_dict_with_sharded(
    param_state: dict,
    global_shape: Tuple[int],
    local_shape: Tuple[int],
    global_offset: Tuple[int],
    dp_ranks_ranges: Optional[Dict[int, Range]],
):
    new_param_state = {}
    for k, v in param_state.items():
        if isinstance(v, torch.Tensor) and v.dim() >= 1:
            # Don't unflatten tensor here, see the comments above
            if not dp_ranks_ranges:
                if math.prod(local_shape) != math.prod(v.shape):
                    print(f"rank={dist.get_rank()} name={k} global shape={global_shape}\
                    local_shape={local_shape} global_offset={global_offset} real shape={v.shape}")
                    raise AssertionError()
            new_param_state[k] = OptimizerStateSpec(
                global_shape, local_shape, global_offset, v, dp_ranks_ranges
            )  # , process_group)
        else:
            new_param_state[k] = v
    return new_param_state

def convert_dict_sharded_to_tensor(param_state: dict, range_1d: Optional[Range]):
    for k, v in param_state.items():
        if isinstance(v, OptimizerStateSpec):
            # If the state is distributed on multiple dp ranks
            # Get my parts
            if range_1d:
                param_state[k] = v.local_tensor.flatten()[range_1d.start : range_1d.end]
            else:
                param_state[k] = v.local_tensor.flatten()
    return param_state