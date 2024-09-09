import copy
from functools import partial
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from internlm.data.collaters import packed_collate_fn

# define the prefix pattern of saved data, e.g., llama2_70B
PATTERN_PREFIX = "llama2_70B"
# define the total saved steps that we want to load, e.g., 500
SAVED_STEPS = 22

def merge_tensors(file_pattern):
    files = sorted(glob.glob(file_pattern))
    tensors = []
    for file in files:
        tensor = torch.load(file)
        tensors.append(tensor)
    merged_tensor = torch.cat(tensors, dim=0)
    return merged_tensor


def process_raw_data(raw_data, micro_bsz):
    num_groups = len(raw_data) // micro_bsz
    result = []
    for i in range(num_groups):
        start_idx = i * micro_bsz
        end_idx = start_idx + micro_bsz
        group = raw_data[start_idx:end_idx]
        concatenated = torch.cat(group, dim=0)
        result.append(concatenated)
    return result


class SequentialSavedSamplesDataset(Dataset):
    def __init__(self, data_dir, micro_bsz, seq_len):
        db_input_ids = []
        db_labels = []
        
        # load all saved data
        for i in range(SAVED_STEPS):
            # define load pattern
            input_ids_pattern = data_dir + PATTERN_PREFIX + f"_tokens_step{i+1}_dp*"
            labels_pattern = data_dir + PATTERN_PREFIX + f"_labels_step{i+1}_dp*"
            # merge input_ids, labels, and then chunk across dp
            input_ids = torch.chunk(merge_tensors(input_ids_pattern), gpc.get_world_size(ParallelMode.DATA))[gpc.get_local_rank(ParallelMode.DATA)]
            labels = torch.chunk(merge_tensors(labels_pattern), gpc.get_world_size(ParallelMode.DATA))[gpc.get_local_rank(ParallelMode.DATA)]
            # load one step
            db_input_ids.append(input_ids)
            db_labels.append(labels)
        
        # transform db
        db_input_ids = torch.concat(db_input_ids, dim=0)
        db_labels = torch.concat(db_labels, dim=0)
        db_input_ids = [db_input_ids[i] for i in range(db_input_ids.size(0))]
        db_labels = [db_labels[i] for i in range(db_labels.size(0))]
        
        # gen data for internevo format
        db_input_ids = process_raw_data(db_input_ids, micro_bsz)
        db_labels = process_raw_data(db_labels, micro_bsz)
        self.db_input_ids = [item.tolist() for item in db_input_ids]
        self.db_labels = [item.tolist() for item in db_labels]
        
        assert len(self.db_input_ids) == len(self.db_labels)
        self.dataset_len = len(self.db_input_ids)
        self.micro_bsz = micro_bsz
        self.seq_len = seq_len

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        tokens = self.db_input_ids[idx]
        cu_seqlens = list(range(self.micro_bsz+1))
        cu_seqlens = [i*self.seq_len for i in cu_seqlens]
        indexes = list(range(self.seq_len)) * self.micro_bsz
        labels = self.db_labels[idx]
        type_ids = [0] * self.micro_bsz * self.seq_len
        
        return {
            "tokens": tokens,
            "cu_seqlens": cu_seqlens,
            "indexes": indexes,
            "labels": labels,
            "type_ids": type_ids,
        }


class SequentialBatchSampler(torch.utils.data.Sampler):
    def __init__(self, data_source, micro_num):
        self.data_source = data_source
        self.micro_num = micro_num

    def __iter__(self):
        num_samples = len(self.data_source)
        for start in range(0, num_samples, self.micro_num):
            end = min(start + self.micro_num, num_samples)
            yield list(range(start, end))

    def __len__(self):
        return (len(self.data_source) + self.micro_num - 1) // self.micro_num
    
    def copy(self):
        return copy.deepcopy(self)


def get_megatron_dataloader_fake():
    data_cfg = gpc.config.data
    
    train_ds = SequentialSavedSamplesDataset(
        data_dir=data_cfg.train_folder, # defined the folder of saved data
        micro_bsz=data_cfg.micro_bsz,
        seq_len=data_cfg.seq_len,
    )
        
    train_sampler = SequentialBatchSampler(train_ds,data_cfg.micro_num)
    train_collate_fn = partial(packed_collate_fn, packed_length=data_cfg.seq_len*data_cfg.micro_bsz)
    
    num_worker = data_cfg.get("num_worker", 0)
    train_dl = DataLoader(
        dataset=train_ds,
        batch_sampler=train_sampler,
        num_workers=num_worker,
        pin_memory=True,
        collate_fn=train_collate_fn,
        persistent_workers=num_worker > 0,
    )
    
    dataset_types = ["en"]
    return train_dl, dataset_types
