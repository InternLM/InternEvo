#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import numpy as np
import torch
from torch.utils.data import Dataset


class RandomDatasetMultimodal(Dataset):
    """
    RandomDataset for generating random dataset.

    Args:
        num_samples (int): The number of samples to generate.
        max_len (int): The maximum length of each sample.
        image_token_id (int): The placeholder of image.
        image_size (int): The image size.
        image_patch_size (int): The patch size of vit.
        image_token_size (int): The number of placeholder of each image.

    """

    def __init__(
        self,
        num_samples=10000,
        max_len=2048,
        image_token_id=200000,
        image_size=336,
        image_token_size=(336 // 14) ** 2,
    ) -> None:
        super().__init__()
        rng = np.random.RandomState(1999)
        max_num = rng.randint(1, 30, size=(num_samples,))
        rep_num = rng.randint(10, 200, size=(num_samples,))
        data = []
        lengths = []
        images = [
            [torch.randn((3, image_size, image_size))] for _ in range(num_samples)
        ]  # num_samples x img_num x tensor(C x H x W)
        for n, r in zip(max_num, rep_num):
            d = list(range(n)) * r
            d = [n, r] + [image_token_id] * image_token_size + d
            d = d[:max_len]
            data.append(d)
            lengths.append(len(d))
        self.data = data
        self.images = images
        self.max_len = max_len
        self.lengths = np.array(lengths, dtype=int)

    def __getitem__(self, index):
        d = self.data[index]
        input_ids = np.array(d, dtype=int)
        images = self.images[index]
        return {"tokens": list(input_ids), "images": images, "type_id": 0}

    def get_dataset_name(self):
        return "dummy_path/dummy_lang/dummy_ds/train.bin"

    def __len__(self):
        return len(self.data)
