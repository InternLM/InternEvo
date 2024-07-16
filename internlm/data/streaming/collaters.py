import torch


def nopack_collate_fn(batch, micro_num, micro_bsz, seq_len, pad_token_id=0):
    input_ids_list = []
    attention_mask_list = []
    labels_list = []

    for b in batch:
        assert len(b["input_ids"]) > 0

        if "attention_mask" in b:
            assert len(b["input_ids"]) == len(
                b["attention_mask"]
            ), "input_ids and attention_mask should be equal length"
        else:
            b["attention_mask"] = [True] * len(b["input_ids"])

        input_ids = b["input_ids"] + [pad_token_id] * (seq_len - len(b["input_ids"]))
        attention_mask = b["attention_mask"] + [False] * (seq_len - len(b["attention_mask"]))
        labels = [w if w > 0 else -100 for w in b["input_ids"]][1:] + [-100]
        labels = labels + [-100] * (seq_len - len(b["input_ids"]))

        input_ids_list.append(torch.LongTensor(input_ids))
        attention_mask_list.append(torch.BoolTensor(attention_mask))
        labels_list.append(torch.LongTensor(labels))

    input_ids = torch.stack(input_ids_list)
    attention_mask = torch.stack(attention_mask_list)
    labels = torch.stack(labels_list)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "type_ids": torch.zeros(micro_num, micro_bsz, seq_len, dtype=torch.int64),
    }, labels


def pack_collate_fn(batch, micro_num, micro_bsz, seq_len):
    packed_length = micro_bsz * seq_len

    input_ids_list = []
    cu_seqlens_list = []
    indexes_list = []
    labels_list = []

    for b in batch:
        assert len(b["input_ids"]) == packed_length
        assert b["cu_seqlens"][0] == 0 and b["cu_seqlens"][-1] == packed_length
        assert len(b["indexes"]) == packed_length
        assert len(b["labels"]) == packed_length

        input_ids_list.append(torch.LongTensor(b["input_ids"]))
        cu_seqlens_list.append(torch.IntTensor(b["cu_seqlens"]))
        indexes_list.append(torch.IntTensor(b["indexes"]))
        labels_list.append(torch.LongTensor(b["labels"]))

    input_ids = torch.stack(input_ids_list)
    indexes = torch.stack(indexes_list)
    labels = torch.stack(labels_list)

    return {
        "input_ids": input_ids,
        "cu_seqlens": cu_seqlens_list,
        "indexes": indexes,
        "type_ids": torch.zeros(micro_num, micro_bsz * seq_len, dtype=torch.int64),
    }, labels
