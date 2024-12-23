################################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
################################################################################
# Modification Copyright 2023 ByteDance Ltd. and/or its affiliates.
################################################################################
import io
import dataclasses
import torch
from typing import Any, Dict, Union, List, Tuple, Optional
from torch.distributed.checkpoint.default_planner import (
    DefaultSavePlanner,
    DefaultLoadPlanner,
)

import mmh3

from .common import P2PTensorsInfo, sort_rank_ranges, PlanLRUCache, custom_dedup_tensors
import math
import torch.distributed as dist
from torch.distributed.checkpoint.planner import SavePlan, LoadPlan, WriteItem, ReadItem
from torch.distributed.checkpoint.metadata import MetadataIndex, Metadata
from .distributed_optimizer import OptimizerStateSpec
# from vescale.dtensor import DTensor

from .vescale_planner_helpers import _create_write_items, _create_read_items, find_state_dict_object
from .devicemesh_api import VESCALE_DEVICE_MESH
from .meta_type import STATE_DICT_STR
from internlm.utils.logger import get_logger
from internlm.core.context import global_context as gpc
from internlm.core.context import ParallelMode
from torch.distributed.checkpoint.planner import WriteItemType


logger = get_logger(__file__)
__all__ = [
    "VeScaleSavePlanner",
    "VeScaleLoadPlanner",
    "create_default_local_load_plan",
    "create_default_local_save_plan",
]

class DTensor:
    pass


class VeScaleLoadPlanner(DefaultLoadPlanner):
    """
    A planner class for loading vescale checkpoint using PyTorch DCP
    """

    def __init__(self):
        super().__init__()

    def create_local_plan(self, is_optimizer=False) -> LoadPlan:
        return create_default_local_load_plan(self.state_dict, self.metadata, is_optimizer)

    def resolve_tensor(self, read_item: ReadItem):
        tensor = self.lookup_tensor(read_item.dest_index)
        return self.transform_tensor(read_item, tensor)

    def lookup_tensor(self, index: MetadataIndex) -> torch.Tensor:
        """
        This is an extension from the planner interface to make it easy to extend the default planner
        """
        return find_state_dict_object(self.state_dict, index)

from internlm.train.pipeline import map_fqn_local_to_global

def create_default_local_load_plan(state_dict: Dict[str, Any], metadata: Metadata, is_optimizer) -> LoadPlan:
    """
    A function for creating local loading plan for loading checkpoint
    """
    # print(f"metadata: {metadata}", flush=True)
    requests = []
    for fqn, obj in state_dict.items():
        global_fqn = fqn
        if not is_optimizer:
            if fqn in map_fqn_local_to_global:
                global_fqn = map_fqn_local_to_global[fqn]
                # print(f"map_fqn_local_to_global {gpc.get_local_rank(ParallelMode.PIPELINE)}: {fqn}, {global_fqn}", flush=True)
        md = metadata.state_dict_metadata[global_fqn]
        if isinstance(obj, DTensor):
            assert False
            if obj.device_mesh.get_coordinate() is not None:
                requests += _create_read_items(fqn, md, obj)
        elif isinstance(obj, OptimizerStateSpec):
            assert False
            # If the state is distributed on multiple dp ranks
            # Read with local_shape, then in DOptimizer then
            # get flaaten to 1D and get the part belonging to current dp rank
            if obj.dp_ranks_ranges:
                obj.local_tensor = torch.zeros(
                    obj.local_shape, dtype=obj.local_tensor.dtype, device=obj.local_tensor.device
                )
                requests += _create_read_items(fqn, md, obj)
            else:
                # If the state is owned by only one dp rank
                # Read directly
                obj.local_tensor = obj.local_tensor.reshape(obj.local_shape)
                requests += _create_read_items(fqn, md, obj)
        else:
            item = _create_read_items(fqn, global_fqn, md, obj)
            # print(f"_create_read_items {gpc.get_global_rank()} {gpc.get_local_rank(ParallelMode.PIPELINE)}: {item}", flush=True)
            requests += item
    return LoadPlan(requests)


class VeScaleSavePlanner(DefaultSavePlanner):
    """
    A planner class for saving vescale checkpoint using PyTorch DCP
    """

    def __init__(self):
        super().__init__()
        self._plan_cache = PlanLRUCache()

    def resolve_data(self, write_item: WriteItem, fqn=None) -> Union[torch.Tensor, io.BytesIO]:
        assert write_item.type != WriteItemType.BYTE_IO
        object = self.lookup_object(write_item.index, fqn)
        return self.transform_object(write_item, object)

    def create_local_plan(self, is_optimizer=False) -> Tuple[SavePlan, P2PTensorsInfo]:
        plan, p2p_tensors_info = create_default_local_save_plan(self.state_dict, self.is_coordinator, is_optimizer)
        # print(f"save before replace local_plan {dist.get_rank()}: {plan}", flush=True)
        if self.flatten_state_dict:
            plan = dataclasses.replace(plan, planner_data=self.mappings)
        self.plan = plan
        return self.plan, p2p_tensors_info

    def lookup_object(self, index: MetadataIndex, fqn=None) -> Any:
        return find_state_dict_object(self.state_dict, index, fqn)

    def lookup_plan_meta(self) -> Optional[Tuple[SavePlan, Metadata]]:
        # if not hasattr(self, STATE_DICT_STR):
        #     return None
        # else:
        #     device_mesh = VESCALE_DEVICE_MESH.get()
        #     plan_key = hash((frozenset(self.state_dict.keys()), self.is_coordinator, device_mesh))
        #     return self._plan_cache.get(plan_key)

        if not hasattr(self, STATE_DICT_STR):
            return None
        else:
            plan_key = hash((frozenset(self.state_dict.keys()), self.is_coordinator))
            return self._plan_cache.get(plan_key)

    def cache_plan_meta(self, new_plan: SavePlan, new_metadata: Metadata) -> None:
        # device_mesh = VESCALE_DEVICE_MESH.get()
        # plan_key = hash((frozenset(self.state_dict.keys()), self.is_coordinator, device_mesh))
        # self._plan_cache.put(plan_key, new_plan, new_metadata)
        
        print(f"new_plan {gpc.get_global_rank()}: {new_plan}", flush=True)
        print(f"new_metadata {gpc.get_global_rank()}: {new_metadata}", flush=True)

        plan_key = hash((frozenset(self.state_dict.keys()), self.is_coordinator))
        print(f"Before GPU Memory Allocated {gpc.get_global_rank()}: {torch.cuda.memory_allocated() /1024/1024} bytes", flush=True)
        print(f"Before GPU Memory Cached {gpc.get_global_rank()}: {torch.cuda.memory_reserved() /1024/1024} bytes", flush=True)
        self._plan_cache.put(plan_key, new_plan, new_metadata)
        print(f"After GPU Memory Allocated {gpc.get_global_rank()}: {torch.cuda.memory_allocated() /1024/1024} bytes", flush=True)
        print(f"After GPU Memory Cached {gpc.get_global_rank()}: {torch.cuda.memory_reserved() /1024/1024} bytes", flush=True)

    def clear_cache(self) -> None:
        self._plan_cache.clear()

    def dedup_plans(self, all_plans: List[SavePlan]) -> List[SavePlan]:
        # Use customized deduplicate function for load balance
        all_plans = custom_dedup_tensors(all_plans)
        return all_plans

    def create_dedup_global_plan(self, all_plans: List[SavePlan]) -> Tuple[List[SavePlan], Metadata]:
        # Disable DCP's dedup replicated tensors function
        self.dedup_replicated_tensors = False
        rst_value = super().create_global_plan(all_plans)
        return rst_value

    def create_global_plan(self, all_plans: List[SavePlan]) -> Tuple[List[SavePlan], Metadata]:
        # Disable DCP's dedup replicated tensors function
        self.dedup_replicated_tensors = False
        # Use customized deduplicate function for load balance
        all_plans = custom_dedup_tensors(all_plans) #att
        rst_value = super().create_global_plan(all_plans)
        return rst_value


def create_default_local_save_plan(state_dict: Dict[str, Any], is_coordinator: bool, is_optimizer=False) -> SavePlan:
    """
    A function for creating local saving plan for saving checkpoint.
    """
    requests = []
    # Key: fqn
    # Value: dictionary (Key is the process rank, value is tensor to receive)
    recv_tensors = {}

    send_p2p_reqs = []
    recv_p2p_reqs = {}
    # if is_optimizer:
    #     state_dict = state_dict["unflatten_fp32_weights"]

    for fqn, obj in state_dict.items():
        # Since DTensor supports submesh, adding extra check to ensure _create_write_items()
        # gets called only when the current rank is part of the mesh for the corresponding DTensor.
        if isinstance(obj, DTensor):
            assert False
            if obj.device_mesh.get_coordinate() is not None:
                requests += _create_write_items(fqn, obj)
        elif isinstance(obj, OptimizerStateSpec):
            assert False #attn
            # Create write requests if the process is the real writer
            if obj.dp_ranks_ranges:
                process_list = []
                for rank, param_range in obj.dp_ranks_ranges.items():
                    process_list.append((rank, len(param_range)))
                sorted_list = sort_rank_ranges(process_list)
                writer_rank = sorted_list[mmh3.hash(fqn) % len(sorted_list)][0]
                send_ops_to_start = []
                recv_ops_to_start = {}
                # Case 1: I am writer
                # Receive tensors
                logger.debug(f"fqn={fqn} is a tensor across dp ranks. writer rank={writer_rank}")
                if dist.get_rank() == writer_rank:
                    recv_tensors[fqn] = {}
                    for k, param_range in obj.dp_ranks_ranges.items():
                        if k != dist.get_rank():
                            recv_tensor = torch.zeros(
                                (len(param_range),), dtype=obj.local_tensor.dtype, device=obj.local_tensor.device
                            )
                            recv_op = dist.P2POp(
                                op=dist.irecv,
                                tensor=recv_tensor,
                                peer=k,
                                group=gpc.get_group(ParallelMode.DATA), #att
                            )
                            recv_tensors[fqn][k] = (recv_tensor, param_range)
                            recv_ops_to_start[k] = recv_op
                else:
                    # Case 2: I am not writer
                    # Send my tensor
                    send_op = dist.P2POp(
                        op=dist.isend,
                        tensor=obj.local_tensor,
                        peer=writer_rank,
                        group=gpc.get_group(ParallelMode.DATA), #att
                    )
                    send_ops_to_start.append(send_op)

                send_reqs = []
                recv_reqs = []
                if send_ops_to_start:
                    send_reqs = dist.batch_isend_irecv(send_ops_to_start)
                if recv_ops_to_start:
                    recv_reqs = dist.batch_isend_irecv(list(recv_ops_to_start.values()))

                if send_reqs:
                    send_p2p_reqs.extend(send_reqs)

                if recv_reqs:
                    recv_p2p_reqs[fqn] = recv_reqs
            else:
                obj.local_tensor = obj.local_tensor.reshape(obj.local_shape)
                requests += _create_write_items(fqn, obj)
        elif isinstance(obj, (torch.Tensor)) or is_coordinator:
            item = _create_write_items(fqn, obj, is_optimizer=is_optimizer)
            # print(f"_create_write_items {gpc.get_global_rank()} {gpc.get_local_rank(ParallelMode.TENSOR)}: {item}", flush=True)
            requests += item
            # if dist.get_rank() == 1:
            #     print(f"requests: {requests}", flush=True)

    # Padding the states across DP ranks
    # Merge the tensors later
    writer_rank = dist.get_rank()
    for fqn in recv_tensors.keys():
        assert False
        obj = state_dict[fqn]
        new_local_tensor = torch.zeros(
            (math.prod(obj.local_shape),), dtype=obj.local_tensor.dtype, device=obj.local_tensor.device
        )
        new_local_tensor[obj.dp_ranks_ranges[writer_rank].start : obj.dp_ranks_ranges[writer_rank].end] = (
            obj.local_tensor
        )
        obj.local_tensor = new_local_tensor

        obj.local_tensor = obj.local_tensor.reshape(obj.local_shape)
        requests += _create_write_items(fqn, obj)
    return SavePlan(requests), P2PTensorsInfo(recv_tensors, send_p2p_reqs, recv_p2p_reqs)
