################################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
################################################################################
# Modification Copyright 2023 ByteDance Ltd. and/or its affiliates.
################################################################################

from typing import Optional

import torch.distributed as dist
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from torch.distributed.checkpoint.planner import LoadPlanner
from torch.distributed.checkpoint.utils import _DistWrapper

from internlm.utils.logger import get_logger

from .filesystem import FileSystemReader
from .planner import UniversalLoadPlanner

logger = get_logger(__file__)

META_DATA_FILE = ".metadata"


def load_state_dict(
    state_dict: dict,
    path: str,
    process_group: Optional[dist.ProcessGroup] = None,
    coordinator_rank: int = 0,
    no_dist: bool = False,
    planner: Optional[LoadPlanner] = None,
    broadcast_tensors=False,
    is_optimizer=False,
) -> None:
    """
    Loads a distributed ``state_dict`` in SPMD style. Fix sub-group storage.
    """
    storage_reader = FileSystemReader(
        path,
        broadcast_tensors=broadcast_tensors,
        data_parallel_process_group=process_group,
    )

    # Step 0: create distributed world based on process group and coordinator rank
    distW = _DistWrapper(process_group, not no_dist, coordinator_rank)
    if process_group:
        distW.coordinator_rank = dist.get_global_rank(process_group, distW.coordinator_rank)
    if planner is None:
        planner = DefaultLoadPlanner()

    # Step 1: all processes create local read plan,
    # then coordinator gathers all local plans and create global plan.
    def local_step():
        assert planner is not None
        metadata = storage_reader.read_metadata()
        planner.set_up_planner(state_dict, metadata, distW.is_coordinator)
        storage_reader.set_up_storage_reader(metadata, distW.is_coordinator)

        local_plan = planner.create_local_plan(is_optimizer=is_optimizer)
        local_plan = storage_reader.prepare_local_plan(local_plan)
        return local_plan

    def global_step(all_local_plans):
        assert planner is not None
        all_local_plans = planner.create_global_plan(all_local_plans)
        all_local_plans = storage_reader.prepare_global_plan(all_local_plans)
        return all_local_plans

    if isinstance(planner, UniversalLoadPlanner):
        central_plan = distW.reduce_scatter("plan", local_step, global_step)
    else:
        raise AssertionError("Unsupported planner for saving checkpoint")

    # Step 2: all processes read data from the given path
    def read_data():  # pylint: disable=R1711
        assert planner is not None
        final_local_plan = planner.finish_plan(central_plan)
        all_reads = storage_reader.read_data(final_local_plan, planner)
        all_reads.wait()
        return None

    _ = distW.all_gather("read", read_data)
