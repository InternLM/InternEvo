#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# adopted from https://github.com/volcengine/veScale/blob/main/vescale/checkpoint/planner

import collections
import dataclasses
from collections import OrderedDict
from typing import Any, Dict, Hashable, List, Optional, Tuple, TypeVar

from torch.distributed.checkpoint.metadata import Metadata, MetadataIndex
from torch.distributed.checkpoint.planner import SavePlan
from typing_extensions import Protocol, runtime_checkable

from internlm.utils.logger import get_logger

logger = get_logger(__file__)


MODEL_STR = "model"
OPTIMIZER_STR = "optimizer"
STATE_DICT_STR = "state_dict"
_MAX_CACHE_SIZE = 2  # model ckpt + optm ckpt


@runtime_checkable
class Stateful(Protocol):
    def state_dict(self) -> Dict[str, Any]:
        ...

    def load_state_dict(self, state_dict: Dict[str, Any], *args) -> None:
        ...


T = TypeVar("T", bound=Stateful)
CheckpointState = Dict[str, T]


class PlanLRUCache:
    """
    For saving cache.
    """

    def __init__(self) -> None:
        self._cache: OrderedDict[Hashable, Tuple[SavePlan, Metadata]] = OrderedDict()
        self._capacity = _MAX_CACHE_SIZE

    def get(self, key: Hashable) -> Optional[Tuple[SavePlan, Metadata]]:
        if key in self._cache:
            return self._cache[key]
        else:
            return None

    def put(self, key: Hashable, plan_value: SavePlan, metadata_value: Metadata) -> None:
        if key in self._cache:
            self._cache.move_to_end(key, last=False)
        else:
            self._cache[key] = (plan_value, metadata_value)
            if len(self._cache) > self._capacity:
                self._cache.popitem()

    def clear(self) -> None:
        self._cache.clear()
        self._capacity = _MAX_CACHE_SIZE

    def __repr__(self) -> str:
        return f"PlanLURCache(capacity: {self._capacity}, keys: {tuple(self._cache.keys())})"


def custom_dedup_tensors(all_plans: List[SavePlan]) -> List[SavePlan]:
    """
    A function to remove duplicate tensors to write
    when creating global writing plan for saving checkpoint
    During the deduplication,
    we balance the workloads for duplicated tensors
    """
    key_to_plan: Dict[MetadataIndex, List[int]] = {}
    for plan_idx, plan in enumerate(all_plans):
        for write_item in plan.items:
            key_to_plan.setdefault(write_item.index, []).append(plan_idx)

    replicated_items = {k: v for k, v in key_to_plan.items() if len(v) > 1}
    # Remove duplicates by always keeping the first entry (Not balance).
    # Compute the per-rank remove set.
    plan_to_keys: Dict[int, List[MetadataIndex]] = {}
    # Record the number of non-duplicated tensors assigned to each rank
    assigned_work_load = collections.defaultdict(int)
    for plan_idx, plan in enumerate(all_plans):
        for write_item in plan.items:
            if write_item.index not in replicated_items:
                assigned_work_load[plan_idx] += 1

    for key, plans in replicated_items.items():
        # For duplicated tensors, select the rank assigned with minimum number tensors so far
        writer_id = min(plans, key=lambda k: assigned_work_load[k])
        assigned_work_load[writer_id] += 1
        for plan_idx in plans:
            # If the rank is not writer rank, remove the key in the rank's plan
            if plan_idx != writer_id:
                plan_to_keys.setdefault(plan_idx, []).append(key)
    # logger.info("Duplicate keys to remove: %s", plan_to_keys)

    for plan_idx, keys in plan_to_keys.items():
        # Key Set contains keys to remove
        key_set = set(keys)
        # rewrite items and remove elements
        new_items = [write_item for write_item in all_plans[plan_idx].items if write_item.index not in key_set]
        all_plans[plan_idx] = dataclasses.replace(all_plans[plan_idx], items=new_items)

    return all_plans
