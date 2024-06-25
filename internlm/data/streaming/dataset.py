import sys
import datasets

from torch.utils.data import Dataset
from transformers import AutoTokenizer
from datasets.distributed import split_dataset_by_node

from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc

class HuggingFaceStreamingDataset(Dataset):
    def __init__(self, dataset_name, tokenizer_name, model_max_length, split='train', buffer_size=1000):
        self.dataset = datasets.load_dataset(dataset_name, split=split, streaming=True)
        self.dataset = split_dataset_by_node(self.dataset, rank=gpc.get_local_rank(ParallelMode.DATA), world_size=gpc.get_world_size(ParallelMode.DATA))
        self.buffer_size = buffer_size
        self.senior_iterator = iter(self)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        self.tokenizer.padding_side = "right"
        self.tokenizer.truncation_side = "right"
        self.tokenizer.model_max_length = model_max_length

    def __iter__(self):
        buffer = []
        for sample in self.dataset:
            buffer.append(sample)
            if len(buffer) >= self.buffer_size:
                yield from self._tokenize(buffer)
                buffer = []

        if buffer:
            yield from self._tokenize(buffer)
    
    def __len__(self):
        return sys.maxsize
    
    def _tokenize(self, samples):
        texts = [sample['text'] for sample in samples]
        tokenized_outputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        for i in range(len(samples)):
            yield {key: tokenized_outputs[key][i] for key in tokenized_outputs}

    def __getitem__(self, _):
        return next(self.senior_iterator)