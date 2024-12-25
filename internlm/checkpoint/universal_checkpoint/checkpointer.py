#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# adopted from https://github.com/volcengine/veScale/blob/main/vescale/checkpoint/api

import atexit
import os
from concurrent.futures import Future, ProcessPoolExecutor
from typing import Dict, List

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.storage import WriteResult

from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from internlm.solver.optimizer import HybridZeroOptimizer
from internlm.utils.common import get_current_device
from internlm.utils.logger import get_logger
from internlm.utils.megatron_timers import megatron_timer as timer

from .common import MODEL_STR, OPTIMIZER_STR, CheckpointState
from .load_state_dict import load_state_dict
from .planner import UniversalLoadPlanner, UniversalSavePlanner
from .save_state_dict import save_state_dict

logger = get_logger(__file__)

NUM_IO_WORKER = 1
SUPPORTED_TYPES = {MODEL_STR, OPTIMIZER_STR}


class BaseCheckpointer:
    """
    The Checkpointer class offers APIs that enable users to save and load state dictionarie.
    It is designed for extension across various training frameworks.
    """

    # Async IO related members.
    state_io_workers: Dict[str, ProcessPoolExecutor] = {}
    state_write_futures: Dict[str, Future[List[WriteResult]]] = {}

    @classmethod
    def save(cls, path: str, checkpoint_state: CheckpointState):
        """
        A Method for saving checkpoint
        Args:
            path: Defines the storage path for checkpoint.
            checkpoint_state: A dictionary contains key-value pairs for model and optimizer.
                              - Model: Identified by 'model' key, value should be a model instance.
                              - Optimizer: Identified by 'optimizer' key, value should be an optimizer instance.

        """
        raise NotImplementedError()

    @classmethod
    def load(cls, path: str, checkpoint_state: CheckpointState):
        """
        A Method for loading checkpoint
        Args:
            path: Defines the storage path for checkpoint.
            checkpoint_state: A dictionary contains key-value pairs for model and optimizer.
                              - Model: Identified by 'model' key, value should be a model instance.
                              - Optimizer: Identified by 'optimizer' key, value should be an optimizer instance.

        """
        raise NotImplementedError()

    @classmethod
    def _cleanup_futures(cls):
        """
        Wait for all write futures to finish before exit, then do the cleanup works.

        WARNING: this method cannot be called by the users.
        """
        for key in SUPPORTED_TYPES:
            if key in cls.state_write_futures:
                futures = cls.state_write_futures[key]
                for fut in futures:
                    fut.result()
                cls.state_write_futures[key] = []
                if cls.state_io_workers[key] is not None:
                    cls.state_io_workers[key].shutdown()
                    cls.state_io_workers[key] = None


class UniversalCheckpointer(BaseCheckpointer):
    """
    The Checkpointer class for universal checkpoint, A PyTorch Native Auto Parallelism Framework
    """

    save_planner = UniversalSavePlanner()
    load_planner = UniversalLoadPlanner()

    optim_ckpt_proces_group = None
    for key in SUPPORTED_TYPES:
        BaseCheckpointer.state_io_workers[key] = ProcessPoolExecutor(max_workers=NUM_IO_WORKER)
        BaseCheckpointer.state_write_futures[key] = []

    @classmethod
    def save(
        cls,
        path: str,
        checkpoint_state: CheckpointState,
        async_checkpoint: bool = False,
    ):
        """
        async_checkpoint: A boolean value indicating if saving checkpoint asynchronously,
                                   i.e. after dumping tensors from GPU memory to Host memory,
                                   the training program can continue training immediately.
                                   Then checkpoint will serialize tensors and dumping to
                                   the persistent storage asynchronously.
        """
        # Check if we support saving the components
        for key in checkpoint_state.keys():
            if key not in SUPPORTED_TYPES:
                raise ValueError(f"{key} is not supported by UniversalCheckpointer")

        # Preprocess saving path
        if path.startswith("local:"):
            path = path.split(":")[1]
        assert ":" not in path, f"{path} is not valid for universal checkpoint!"

        # Start saving checkpoint
        for key, value in checkpoint_state.items():
            if key == MODEL_STR:
                # Get model path
                model_path = os.path.join(path, MODEL_STR)
                # Create a "model" folder on under root path
                if gpc.get_global_rank() == 0:
                    os.makedirs(model_path, exist_ok=True)
                dist.barrier()
                # Save model.
                timer("save-model").start()
                _, new_write_futures = save_state_dict(
                    state_dict=value.state_dict(),
                    path=model_path,
                    process_group=None,
                    coordinator_rank=0,
                    no_dist=False,
                    planner=cls.save_planner,
                    async_io=async_checkpoint,
                    last_write_futures=cls.state_write_futures[MODEL_STR],
                    io_workers=cls.state_io_workers[MODEL_STR],
                    is_optimizer=False,
                )
                # Record new write futures.
                cls.state_write_futures[MODEL_STR] = new_write_futures
                dist.barrier()
                timer("save-model").stop()
            elif key == OPTIMIZER_STR:
                # adamW hybrid zero optim
                assert isinstance(value, HybridZeroOptimizer), "unsupported optimizer for universal ckpt"
                optimizer_state = value.state_dict()
                # Create a "optimizer" folder on under root path
                # to save different parts of optimizer
                optimizer_path = os.path.join(path, OPTIMIZER_STR)
                if gpc.get_global_rank() == 0:
                    os.makedirs(optimizer_path, exist_ok=True)
                dist.barrier()
                # Save optimizer
                timer("save-optimizer").start()
                _, new_write_futures = save_state_dict(
                    state_dict=optimizer_state["sharded_optimizer_state"],
                    path=optimizer_path,
                    process_group=None,
                    coordinator_rank=0,
                    no_dist=False,
                    planner=cls.save_planner,
                    async_io=async_checkpoint,
                    last_write_futures=cls.state_write_futures[OPTIMIZER_STR],
                    io_workers=cls.state_io_workers[OPTIMIZER_STR],
                    is_optimizer=True,
                )
                # Record new write futures.
                cls.state_write_futures[OPTIMIZER_STR] = new_write_futures
                # Save the global part of optimizer state
                optimizer_state.pop("sharded_optimizer_state")
                if gpc.get_global_rank() == 0:
                    torch.save(optimizer_state, os.path.join(path, "global_optimizer_state.pt"))
                dist.barrier()
                timer("save-optimizer").stop()

    @classmethod
    def load(
        cls,
        path: str,
        checkpoint_state: CheckpointState,
        broadcast_checkpoint: bool = False,
    ):
        """
        broadcast_checkpoint: A boolean value decides if load a model replica from one data parallel process group
                                 then broadcast tensors to other data parallel process group using GPUs
                                 to reduce the file system access
                                 For example, when data parellel size = 2,
                                 processes with data parallel rank = 0 load model from file system
                                 then broadcast it to processes with data parallel rank = 1
        """
        # Check if we support loading the component.
        for key in checkpoint_state.keys():
            if key not in SUPPORTED_TYPES:
                raise ValueError(f"{key} is not supported by UniversalCheckpointer")

        # Preprocess loading path
        if path.startswith("local:"):
            path = path.split(":")[1]
        assert ":" not in path, f"{path} is not valid for universal checkpoint!"

        # Start loading checkpoint
        for key, value in checkpoint_state.items():
            if key == MODEL_STR:
                # Get model path and state dictionary
                model_path = os.path.join(path, MODEL_STR)
                model_state = value.state_dict()
                # Set process group
                if broadcast_checkpoint:
                    model_load_process_group = gpc.get_group(ParallelMode.DATA)
                else:
                    model_load_process_group = None
                # Load model
                load_state_dict(
                    state_dict=model_state,
                    path=model_path,
                    process_group=model_load_process_group,
                    coordinator_rank=0,
                    no_dist=False,
                    planner=cls.load_planner,
                    broadcast_tensors=broadcast_checkpoint,
                )
                # Load back to model
                value.load_state_dict(model_state)
            elif key == OPTIMIZER_STR:
                # Get optimizer path and state dictionary
                optimizer_path = os.path.join(path, OPTIMIZER_STR)
                optimizer_state = value.state_dict()
                # Load optimizer state dictionary
                load_state_dict(
                    state_dict=optimizer_state["sharded_optimizer_state"],
                    path=optimizer_path,
                    process_group=None,
                    coordinator_rank=0,
                    no_dist=False,
                    planner=cls.load_planner,
                    broadcast_tensors=False,
                    is_optimizer=True,
                )
                # Load back to optimizer
                global_optimizer_state = torch.load(
                    os.path.join(path, "global_optimizer_state.pt"), map_location=get_current_device()
                )
                value.load_state_dict(optimizer_state["sharded_optimizer_state"], global_optimizer_state)
            dist.barrier()

    @classmethod
    def __cleanup(cls):
        """
        Wait for all write futures to finish before exit, then do the cleanup works.

        WARNING: this method cannot be called by the users.
        """
        cls.save_planner.clear_cache()
        BaseCheckpointer._cleanup_futures()

    @classmethod
    def _register_cleanup(cls):
        atexit.register(UniversalCheckpointer.__cleanup)


UniversalCheckpointer._register_cleanup()
