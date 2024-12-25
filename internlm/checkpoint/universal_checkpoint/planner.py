#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# adopted from https://github.com/volcengine/veScale/tree/main/vescale/checkpoint/planner/vescale

import dataclasses
import io
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch.distributed.checkpoint.default_planner import (
    DefaultLoadPlanner,
    DefaultSavePlanner,
)
from torch.distributed.checkpoint.metadata import Metadata, MetadataIndex
from torch.distributed.checkpoint.planner import (
    LoadPlan,
    ReadItem,
    SavePlan,
    WriteItem,
    WriteItemType,
)

from internlm.train.pipeline import map_fqn_local_to_global
from internlm.utils.logger import get_logger

from .common import STATE_DICT_STR, PlanLRUCache, custom_dedup_tensors
from .planner_helpers import (
    _create_read_items,
    _create_write_items,
    find_state_dict_object,
)

logger = get_logger(__file__)
__all__ = [
    "UniversalSavePlanner",
    "UniversalLoadPlanner",
    "create_default_local_load_plan",
    "create_default_local_save_plan",
]


class UniversalLoadPlanner(DefaultLoadPlanner):
    """
    A planner class for loading checkpoint using PyTorch DCP
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


def create_default_local_load_plan(state_dict: Dict[str, Any], metadata: Metadata, is_optimizer) -> LoadPlan:
    """
    A function for creating local loading plan for loading checkpoint
    """
    requests = []
    for fqn, obj in state_dict.items():
        global_fqn = fqn
        # For local model state_dict, the default is to use local fqn.
        # We need to map it to global fqn for pipeline parallel.
        # As saving ckpt is always using global fqn.
        if not is_optimizer:
            if fqn in map_fqn_local_to_global:  # pylint: disable=R1715
                global_fqn = map_fqn_local_to_global[fqn]

        md = metadata.state_dict_metadata[global_fqn]
        item = _create_read_items(fqn, global_fqn, md, obj)
        requests += item
    return LoadPlan(requests)


class UniversalSavePlanner(DefaultSavePlanner):
    """
    A planner class for saving checkpoint using PyTorch DCP
    """

    def __init__(self):
        super().__init__()
        self._plan_cache = PlanLRUCache()

    def resolve_data(self, write_item: WriteItem, fqn=None) -> Union[torch.Tensor, io.BytesIO]:
        assert write_item.type != WriteItemType.BYTE_IO
        local_object = self.lookup_object(write_item.index, fqn)
        return self.transform_object(write_item, local_object)

    def create_local_plan(self, is_optimizer=False) -> Tuple[SavePlan, None]:
        plan, p2p_tensors_info = create_default_local_save_plan(self.state_dict, self.is_coordinator, is_optimizer)
        if self.flatten_state_dict:
            plan = dataclasses.replace(plan, planner_data=self.mappings)
        self.plan = plan
        return self.plan, p2p_tensors_info

    def lookup_object(self, index: MetadataIndex, fqn=None) -> Any:
        return find_state_dict_object(self.state_dict, index, fqn)

    def lookup_plan_meta(self) -> Optional[Tuple[SavePlan, Metadata]]:
        if not hasattr(self, STATE_DICT_STR):
            return None
        else:
            plan_key = hash((frozenset(self.state_dict.keys()), self.is_coordinator))
            return self._plan_cache.get(plan_key)

    def cache_plan_meta(self, new_plan: SavePlan, new_metadata: Metadata) -> None:
        plan_key = hash((frozenset(self.state_dict.keys()), self.is_coordinator))
        self._plan_cache.put(plan_key, new_plan, new_metadata)

    def clear_cache(self) -> None:
        self._plan_cache.clear()

    def create_dedup_global_plan(self, all_plans: List[SavePlan]) -> Tuple[List[SavePlan], Metadata]:
        # Disable DCP's dedup replicated tensors function
        self.dedup_replicated_tensors = False
        rst_value = super().create_global_plan(all_plans)
        return rst_value

    def create_global_plan(self, all_plans: List[SavePlan]) -> Tuple[List[SavePlan], Metadata]:
        # Disable DCP's dedup replicated tensors function
        self.dedup_replicated_tensors = False
        # Use customized deduplicate function for load balance
        all_plans = custom_dedup_tensors(all_plans)
        rst_value = super().create_global_plan(all_plans)
        return rst_value


def create_default_local_save_plan(
    state_dict: Dict[str, Any], is_coordinator: bool, is_optimizer=False  # pylint: disable=W0613
) -> SavePlan:
    """
    A function for creating local saving plan for saving checkpoint.
    """
    requests = []
    for fqn, obj in state_dict.items():
        assert isinstance(obj, (torch.Tensor))
        item = _create_write_items(fqn, obj, is_optimizer=is_optimizer)
        requests += item

    return SavePlan(requests), None
