#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# adopted from https://github.com/volcengine/veScale/blob/main/vescale/checkpoint

from .checkpointer import UniversalCheckpointer
from .common import CheckpointState


def universal_save(path: str, checkpoint_state: CheckpointState, async_checkpoint=False):
    """
    Save a checkpoint to a given path
    Args:
        path: Defines the storage path for checkpoint.
        checkpoint_state: A dictionary contains key-value pairs for model and optimizer.
                            - Model: Identified by 'model' key, value should be a model instance.
                            - Optimizer: Identified by 'optimizer' key, value should be an optimizer instance.
        async_checkpoint: A boolean value indicating if saving checkpoint asynchronously,
                                 i.e. after dumping tensors from GPU memory to Host memory,
                                 the training program can continue training immediately.
                                 Then universal_checkpoint will serialize tensors and dumping to
                                 the persistent storage asynchronously.
    Example:
        >>> checkpoint_state = { "model": nn.Module, "optimizer": HybridZeroOptimizer }
        >>> UniversalCheckpointer.save("/ckpt", checkpoint_state)
    """
    UniversalCheckpointer.save(path, checkpoint_state, async_checkpoint=async_checkpoint)


def universal_load(path: str, checkpoint_state: CheckpointState, broadcast_checkpoint=False):
    """
    Load a checkpoint from a given path
    Args:
        path: Defines the storage path for checkpoint.
        checkpoint_state: A dictionary contains key-value pairs for model and optimizer.
                            - Model: Identified by 'model' key, value should be a model instance.
                            - Optimizer: Identified by 'optimizer' key, value should be an optimizer instance.
        broadcast_checkpoint: A boolean value decides if load a model replica from one data parallel process group
                                 then broadcast tensors to other data parallel process group using GPUs
                                 to reduce the file system access
                                 For example, when data parellel size = 2,
                                 processes with data parallel rank = 0 load model from file system
                                 then broadcast it to processes with data parallel rank = 1
    Example:
        >>> checkpoint_state = { "model": nn.Module, "optimizer": HybridZeroOptimizer }
        >>> UniversalCheckpointer.load("/ckpt", checkpoint_state)
    """
    UniversalCheckpointer.load(path, checkpoint_state, broadcast_checkpoint=broadcast_checkpoint)
