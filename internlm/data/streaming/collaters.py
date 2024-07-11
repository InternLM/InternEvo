import torch


def nopack_collate_fn(batch, micro_num, micro_bsz, seq_len):
    input_ids_list = []
    attention_mask_list = []
    labels_list = []
    for b in batch:
        attention_mask = torch.tensor(b["attention_mask"])
        input_ids = torch.LongTensor(b["input_ids"])
        input_ids = torch.abs(input_ids * attention_mask)
        input_ids = torch.nn.functional.pad(input_ids, (0, seq_len - len(input_ids)), mode="constant", value=0)
        attention_mask = torch.nn.functional.pad(
            attention_mask, (0, seq_len - len(attention_mask)), mode="constant", value=0
        )
        label = torch.LongTensor([w if w > 0 else -100 for w in input_ids.tolist()][1:] + [-100])
        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        labels_list.append(label)
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
