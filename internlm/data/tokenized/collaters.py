#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os

import numpy as np
import torch


def packed_collate_fn(batch, packed_length):

    """
    Collate function for packed input sequences.

    Args:
        batch (List[Dict]): List of dictionaries representing each sample in batch.
            Each dictionary contains "tokens", "labels", "type_ids", "cu_seqlens", and "indexes" keys.
        packed_length (int): The length of packed sequence.

    Returns:
        Tuple[Dict[str, torch.Tensor], torch.Tensor]: A tuple containing a dictionary of tensors with "input_ids",
            "cu_seqlens", "indexes", and "type_ids" keys, and the tensor of padded "labels".

    Raises:
        AssertionError: If the length of a sample is not equal to packed_length.
        AssertionError: If the shape of the padded "input_ids" tensor does not have the correct shape.
    """

    xs, ys, cu_seqlens, indexes, ts = [], [], [], [], []
    for b in batch:
        assert (
            len(b["tokens"]) == packed_length
        ), f"length of a sample should be equal to packed_length, but got {len(b['tokens'])} and {packed_length})"
        assert (
            len(b["labels"]) == packed_length
        ), f"length of a sample should be equal to packed_length, but got {len(b['labels'])} and {packed_length})"
        assert (
            len(b["type_ids"]) == packed_length
        ), f"length of a sample should be equal to packed_length, but got {len(b['type_ids'])} and {packed_length})"

        tokens = [abs(w) for w in b["tokens"]]
        labels = [w if w > 0 else -100 for w in b["labels"]]

        xs.append(torch.LongTensor(tokens))
        # The labels have been shifted here, so they are aligned with the output corresponding to the token
        ys.append(torch.LongTensor(labels))
        ts.append(torch.LongTensor(b["type_ids"]))
        cu_seqlens.append(torch.IntTensor(b["cu_seqlens"]))
        indexes.append(torch.LongTensor(b["indexes"]))

    xs = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True)
    ys = torch.nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=-100)
    ts = torch.nn.utils.rnn.pad_sequence(ts, batch_first=True, padding_value=0)
    indexes = torch.stack(indexes, dim=0)
    if len(set(map(len, cu_seqlens))) == 1:  # if has uniform length, then stack to save device transfer time
        cu_seqlens = torch.stack(cu_seqlens, dim=0)

    assert xs.shape[1] == packed_length, (xs.shape[1], packed_length)

    return {"input_ids": xs, "cu_seqlens": cu_seqlens, "indexes": indexes, "type_ids": ts}, ys


def get_ltor_masks_and_position_ids(data, eod_token, reset_position_ids, reset_attention_mask, eod_mask_loss):
    """Build masks and position id for left to right model."""

    # Extract batch size and sequence length.
    micro_batch_size, seq_length = data.size()

    # Attention mask (lower triangular).
    if reset_attention_mask:
        att_mask_batch = micro_batch_size
    else:
        att_mask_batch = 1
    attention_mask = torch.tril(torch.ones((att_mask_batch, seq_length, seq_length), device=data.device)).view(
        att_mask_batch, 1, seq_length, seq_length
    )

    # Loss mask.
    loss_mask = torch.ones(data.size(), dtype=torch.float, device=data.device)
    if eod_mask_loss:
        loss_mask[data == eod_token] = 0.0

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long, device=data.device)
    position_ids = position_ids.unsqueeze(0).expand_as(data)
    # We need to clone as the ids will be modifed based on batch index.
    if reset_position_ids:
        position_ids = position_ids.clone()

    if reset_position_ids or reset_attention_mask:
        # Loop through the batches:
        for b in range(micro_batch_size):

            # Find indecies where EOD token is.
            eod_index = position_ids[b, data[b] == eod_token]
            # Detach indecies from positions if going to modify positions.
            if reset_position_ids:
                eod_index = eod_index.clone()

            # Loop through EOD indecies:
            prev_index = 0
            for j in range(eod_index.size()[0]):
                i = eod_index[j]
                # Mask attention loss.
                if reset_attention_mask:
                    attention_mask[b, 0, (i + 1) :, : (i + 1)] = 0
                # Reset positions.
                if reset_position_ids:
                    position_ids[b, (i + 1) :] -= i + 1 - prev_index
                    prev_index = i + 1

    # Convert attention mask to binary:
    attention_mask = attention_mask < 0.5

    return attention_mask, loss_mask, position_ids


def packed_collate_fn_nopack(batch, packed_length, eos_token, gen_attn_mask):

    """
    Collate function for packed input sequences.

    Args:
        batch (List[Dict]): List of dictionaries representing each sample in batch.
            Each dictionary contains "tokens", "labels", "type_ids", "cu_seqlens", and "indexes" keys.
        packed_length (int): The length of packed sequence.

    Returns:
        Tuple[Dict[str, torch.Tensor], torch.Tensor]: A tuple containing a dictionary of tensors with "input_ids",
            "cu_seqlens", "indexes", and "type_ids" keys, and the tensor of padded "labels".

    Raises:
        AssertionError: If the length of a sample is not equal to packed_length.
        AssertionError: If the shape of the padded "input_ids" tensor does not have the correct shape.
    """
    xs, ys, cu_seqlens, ts = [], [], [], []
    for b in batch:  # micro_num
        assert (
            len(b["tokens"]) * len(b["tokens"][0]) == packed_length
        ), f"length of a sample should be equal to packed_length, but got {len(b['tokens'])},{len(b['tokens'][0])},{len(b['tokens'][1])} and {packed_length})"
        assert (
            len(b["labels"]) * len(b["labels"][0]) == packed_length
        ), f"length of a sample should be equal to packed_length, but got {len(b['labels'])},{len(b['labels'][0])} and {packed_length})"
        assert (
            len(b["type_ids"]) * len(b["type_ids"][0]) == packed_length
        ), f"length of a sample should be equal to packed_length, but got {len(b['type_ids'])},{len(b['type_ids'][0])} and {packed_length})"

        tokens = [abs(np.array(w)) for w in b["tokens"]]

        xs.append(torch.LongTensor(tokens))
        # The labels have been shifted here, so they are aligned with the output corresponding to the token
        ys.append(torch.LongTensor(b["labels"]))
        ts.append(torch.LongTensor(b["type_ids"]))
        cu_seqlens.append(torch.IntTensor(b["cu_seqlens"]))  # get metric needs cu_seqlens.

    attention_mask_micro_num_list = []
    for micro_num_dix in range(len(xs)):
        attention_mask, _, _ = get_ltor_masks_and_position_ids(
            data=xs[micro_num_dix],
            eod_token=eos_token,
            reset_position_ids=True,
            reset_attention_mask=True,
            eod_mask_loss=True,
        )

        attention_mask_micro_num_list.append(attention_mask)

    xs = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True)
    ys = torch.nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=-100)
    ts = torch.nn.utils.rnn.pad_sequence(ts, batch_first=True, padding_value=0)

    if len(set(map(len, cu_seqlens))) == 1:  # if has uniform length, then stack to save device transfer time
        cu_seqlens = torch.stack(cu_seqlens, dim=0)

    #     print(f"input_ids: {xs.shape}, mask nums: {len(attention_mask_micro_num_list)}, \
    # one mask shape: {attention_mask.shape}, cu_seqlens: {cu_seqlens.shape}, type_ids: {ts.shape}", flush=True)
    x = {
        "input_ids": xs,
        "type_ids": ts,
        "cu_seqlens": cu_seqlens,
    }

    if gen_attn_mask:
        x.update({"attention_mask": attention_mask_micro_num_list})

    return x, ys  # "cu_seqlens": cu_seqlens, "indexes": indexes,


def jsonl_ds_collate_fn(batch, max_length_per_sample):
    """
    Collate function for json dataset.

    Args:
        batch (List[Dict]): List of dictionaries representing each sample in batch.
            Each dictionary contains "tokens".
        max_length_per_sample (int): The length of output sequence.

    Returns:
        Tuple[Dict[str, torch.Tensor], torch.Tensor]: A tuple containing a dictionary of tensors with "input_ids",
        and the tensor of padded "labels".

    """
    xs, ys = [], []
    for x in batch:
        x["tokens"] = x["tokens"][:max_length_per_sample]
        tokens = [abs(w) for w in x["tokens"]]
        labels = [w if w > 0 else -100 for w in x["tokens"]]
        labels = labels[1:] + [-100]
        xs.append(torch.as_tensor(tokens))
        ys.append(torch.as_tensor(labels))  # y has been shifted
    xs = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True)
    ys = torch.nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=-100)

    xs = torch.cat([xs, xs.new_zeros(len(xs), max_length_per_sample - len(xs[0]))], dim=-1)
    ys = torch.cat([ys, ys.new_full((len(ys), max_length_per_sample - len(ys[0])), fill_value=-100)], dim=-1)

    return {"input_ids": xs}, ys
