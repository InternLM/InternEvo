################################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
################################################################################
# Modification Copyright 2023 ByteDance Ltd. and/or its affiliates.
################################################################################

import os
import pickle
import time
from concurrent.futures import Future
from typing import List, Optional, Tuple

import torch.distributed as dist
from torch.distributed.checkpoint.default_planner import DefaultSavePlanner
from torch.distributed.checkpoint.metadata import Metadata
from torch.distributed.checkpoint.planner import SavePlanner
from torch.distributed.checkpoint.storage import WriteResult
from torch.distributed.checkpoint.utils import _DistWrapper

from internlm.core.context import global_context as gpc
from internlm.utils.logger import get_logger

from .filesystem import FileSystemWriter
from .planner import UniversalSavePlanner

logger = get_logger(__file__)


def save_state_dict(
    state_dict: dict,
    path: str,
    process_group: Optional[dist.ProcessGroup] = None,
    coordinator_rank: int = 0,
    no_dist: bool = False,
    planner: Optional[SavePlanner] = None,
    async_io: bool = True,
    last_write_futures: Future[List[WriteResult]] = None,  # pylint: disable=E1136
    io_workers=None,
    is_optimizer=False,
) -> Tuple[Metadata, Future[List[WriteResult]]]:  # pylint: disable=E1136
    """
    Saves a model in SPMD style. Fix sub-group storage.
    """
    # Step 0: create distributed world based on process group and coordinator rank
    distW = _DistWrapper(process_group, not no_dist, coordinator_rank)
    if process_group:
        distW.coordinator_rank = dist.get_global_rank(process_group, distW.coordinator_rank)
    if planner is None:
        planner = DefaultSavePlanner()
    assert planner is not None

    global_metatadata = None

    storage_writer = FileSystemWriter(path)

    # Step 1: all processes create local write plan,
    # then coordinator gathers all local plans and create global plan.
    def local_step():
        if isinstance(planner, UniversalSavePlanner):
            local_plan, p2p_tensors_info = planner.create_local_plan(is_optimizer=is_optimizer)
            local_plan = storage_writer.prepare_local_plan(local_plan, p2p_tensors_info)
        else:
            raise AssertionError("Unsupported planner for planning")
        return local_plan

    def global_step(all_local_plans):
        nonlocal global_metatadata
        all_local_plans, global_metatadata = planner.create_global_plan(all_local_plans)
        all_local_plans = storage_writer.prepare_global_plan(all_local_plans)
        return all_local_plans

    # Step 2: all processes write data from GPUs to pinned memory pool, then dump to local path
    # then coordinator write meta-data to local path.
    def write_data(async_io: bool = False, io_workers=io_workers):
        final_local_plan = planner.finish_plan(central_plan)
        if isinstance(planner, UniversalSavePlanner):
            # Use pinned memory pool and mult_processing for dumping ckpt to local directory efficiently
            all_write_futures = storage_writer.write_data(final_local_plan, planner, async_io, io_workers, is_optimizer)
            if async_io:
                return all_write_futures
            else:
                # Gather write results.
                values = []
                for fut in all_write_futures:
                    # values += fut.get()
                    values += fut.result()
                return values
        else:
            raise AssertionError("Unsupported planner for writing data")

    def finish_checkpoint(all_results):
        assert global_metatadata is not None, f"rank: {distW.get_rank()} has no global_metadata"
        storage_writer.finish(metadata=global_metatadata, results=all_results)
        return global_metatadata

    assert planner is not None
    planner.set_up_planner(state_dict, distW.is_coordinator)
    storage_writer.set_up_storage_writer(distW.is_coordinator)

    # Wait for last write futures to finish.
    if last_write_futures:
        if gpc.is_rank_for_log():
            logger.info("Start waiting for last write events.")
        last_write_start_time = time.time()
        for fut in last_write_futures:
            fut.result()
        last_write_time = time.time() - last_write_start_time
        if gpc.is_rank_for_log():
            logger.info(f"Finish waiting for last write events. Time cost: {last_write_time}s")

    # Each worker bypass the `reduce_scatter()` and `all_reduce()` if finding cached central_plan and metadata.
    # NOTE: it fails when the plans of partial workers change while others keep the same.
    cached_data = None
    if isinstance(planner, UniversalSavePlanner):
        cached_data = planner.lookup_plan_meta()
        if cached_data:
            if gpc.is_rank_for_log():
                logger.info("Plan cache hit. Reuse existing plan")
            central_plan, _ = cached_data
            # _ = local_step()  # backup for origin
        else:
            if gpc.is_rank_for_log():
                logger.info("Plan cache miss. The model/optimizer appears for the first time.")
            central_plan = distW.reduce_scatter("plan", local_step, global_step)
    else:
        raise AssertionError("Unsupported planner for saving checkpoint")

    write_futures = []
    if isinstance(planner, UniversalSavePlanner):
        if cached_data:
            if gpc.is_rank_for_log():
                logger.info("Metdata cache hit. Reuse existing metadata")
            _, final_storage_metadata = cached_data
            write_results = write_data(async_io=async_io)
            # Be sure to write cache metadata to .metadata file
            # Otherwises only the first checkpoint has .metadata
            # which leads to error when loading other checkpoints
            if distW.is_coordinator:
                with (storage_writer.path / ".metadata.tmp").open("wb") as metadata_file:
                    pickle.dump(final_storage_metadata, metadata_file)
                    os.fsync(metadata_file.fileno())

                (storage_writer.path / ".metadata.tmp").rename(storage_writer.path / ".metadata")

            if async_io:
                write_futures = write_results
        else:
            if gpc.is_rank_for_log():
                logger.info("Metadata cache miss. The model/optimizer appears for the first time.")
            # First time do synchronous storing to get final_storage_metatdata.
            # Determine which communication topology to use.
            final_storage_metadata = distW.all_reduce("write", write_data, finish_checkpoint)
            assert central_plan is not None
            assert final_storage_metadata is not None
            planner.cache_plan_meta(central_plan, final_storage_metadata)  # attn
    else:
        raise AssertionError("Unsupported planner for writing data and metadata")

    return final_storage_metadata, write_futures
