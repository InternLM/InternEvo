#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# adopted from https://github.com/volcengine/veScale/tree/main/vescale/checkpoint/planner/vescale

from typing import Any, List

import torch
from torch.distributed._shard.sharded_tensor import TensorProperties
from torch.distributed.checkpoint.metadata import (
    STORAGE_TYPES,
    ChunkStorageMetadata,
    MetadataIndex,
    TensorStorageMetadata,
)
from torch.distributed.checkpoint.planner import (
    LoadItemType,
    ReadItem,
    TensorWriteData,
    WriteItem,
    WriteItemType,
)
from torch.distributed.checkpoint.resharding import (
    _check_shard_metadata_pair_overlap,
    _shards_get_overlap_region_wrt_saved_tensor,
)

from internlm.train.pipeline import map_fqn_local_to_global, map_layer_attr


def _create_write_item_for_tensor(fqn: str, tensor: torch.Tensor, is_optimizer=False) -> WriteItem:
    offsets = torch.Size([0] * len(tensor.size()))
    size = tensor.size()

    # We always save ckpt using global fqn.
    # For optim state_dict, it originally uses global fqn.
    if not is_optimizer:
        if fqn in map_fqn_local_to_global:  # pylint: disable=R1715
            fqn = map_fqn_local_to_global[fqn]

    map_fqn = fqn
    if map_fqn not in map_layer_attr:
        # Deal with exp_avg and exp_avg_sq in base optim
        map_fqn = fqn.rsplit(".", 1)[0]
    assert map_fqn in map_layer_attr
    offsets = torch.Size(map_layer_attr[map_fqn]["offset"])
    size = torch.Size(map_layer_attr[map_fqn]["complete_size"])

    return WriteItem(
        index=MetadataIndex(fqn, offsets),
        type=WriteItemType.SHARD,
        tensor_data=TensorWriteData(
            chunk=ChunkStorageMetadata(offsets=offsets, sizes=tensor.size()),
            properties=TensorProperties.create_from_tensor(tensor),
            size=size,
        ),
    )


def _create_write_items(fqn: str, object: Any, is_optimizer=False) -> List[WriteItem]:  # pylint: disable=W0622
    assert isinstance(object, torch.Tensor)
    return [_create_write_item_for_tensor(fqn, object, is_optimizer=is_optimizer)]


def _create_read_item_for_tensor(dest_index, dest_offsets, storage_index, storage_offsets, lengths):
    return ReadItem(
        type=LoadItemType.TENSOR,
        dest_index=dest_index,
        dest_offsets=torch.Size(dest_offsets),
        storage_index=storage_index,
        storage_offsets=torch.Size(storage_offsets),
        lengths=torch.Size(lengths),
    )


def create_read_items_for_chunk_list(
    fqn: str,
    global_fqn: str,
    checkpoint_md: TensorStorageMetadata,
    local_chunks: List[ChunkStorageMetadata],
) -> List[ReadItem]:
    """
    Creates a list of ``ReadItem`` based on the checkpoint and local chunks.

    This applies the resharding algorithm and computes the reads needed
    to satisfy ``local_chunks`` with a checkpoint described by ``checkpoint_md``.

    Args:
        fqn (str): The local state_dict FQN to pass to ``ReadItem``.
        global_fqn (str): The global FQN in the checkpoint.
        checkpoint_md (TensorStorageMetadata): metadata for a given tensor
            from a checkpoint.
        local_chunks (List[ChunkStorageMetadata]): Local chunks that needs to be
            loaded.

    Returns:
        A list of ``ReadItem`` that will satisfy all input chunks.
    """
    read_items = []
    # this is a naive quadratic algo that can be optimized later
    for idx, shard in enumerate(local_chunks):
        for storage_idx, storage_md in enumerate(checkpoint_md.chunks):
            if not _check_shard_metadata_pair_overlap(shard, storage_md):
                continue

            storage_offsets = []
            dest_offsets = []
            lengths = []
            for (
                _,
                offset_for_saved_tensor,
                offset_for_current_tensor,
                length,
            ) in _shards_get_overlap_region_wrt_saved_tensor(saved_shard=storage_md, current_shard=shard):
                storage_offsets.append(offset_for_saved_tensor)
                dest_offsets.append(offset_for_current_tensor)
                lengths.append(length)

            read_items.append(
                _create_read_item_for_tensor(
                    dest_index=MetadataIndex(fqn, shard.offsets, idx),
                    dest_offsets=dest_offsets,
                    storage_index=MetadataIndex(global_fqn, storage_md.offsets, storage_idx),
                    storage_offsets=storage_offsets,
                    lengths=lengths,
                )
            )
    return read_items


def _create_chunk_from_tensor(global_fqn, tensor: torch.Tensor) -> ChunkStorageMetadata:
    if global_fqn not in map_layer_attr:
        # Deal with exp_avg and exp_avg_sq in base optim
        global_fqn = global_fqn.rsplit(".", 1)[0]
    assert global_fqn in map_layer_attr, f"{global_fqn}"
    offsets = torch.Size(map_layer_attr[global_fqn]["offset"])

    return ChunkStorageMetadata(offsets=offsets, sizes=tensor.size())


def _create_read_items(fqn: str, global_fqn, md: STORAGE_TYPES, obj: Any) -> List[ReadItem]:
    assert isinstance(obj, torch.Tensor)
    local_chunks = [_create_chunk_from_tensor(global_fqn, obj)]
    return create_read_items_for_chunk_list(fqn, global_fqn, md, local_chunks)


def find_state_dict_object(state_dict: dict, index: MetadataIndex, fqn=None) -> Any:
    # Called when real writing happened
    # The filesystem writer calls resolve_data , then it will
    # call find_state_dict_object
    if fqn is None:
        fqn = index.fqn
    if fqn not in state_dict:
        raise ValueError(f"Could not find FQN: '{fqn}'")
    obj = state_dict[fqn]

    return obj
