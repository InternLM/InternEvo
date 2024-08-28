import torch


def streaming_packed_collate_fn(batch):
    input_ids_list = []
    cu_seqlens_list = []
    indexes_list = []
    type_ids_list = []
    labels_list = []

    for b in batch:
        input_ids_list.append(torch.LongTensor(b["input_ids"]))
        cu_seqlens_list.append(torch.IntTensor(b["cu_seqlens"]))
        indexes_list.append(torch.IntTensor(b["indexes"]))
        type_ids_list.append(torch.LongTensor(b["type_ids"]))
        labels_list.append(torch.LongTensor(b["labels"]))

    return {
        "input_ids": torch.stack(input_ids_list),
        "cu_seqlens": cu_seqlens_list,
        "indexes": torch.stack(indexes_list),
        "type_ids": torch.stack(type_ids_list),
    }, torch.stack(labels_list)
