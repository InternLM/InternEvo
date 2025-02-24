# adapted from https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/datasets/gpt_dataset.py
# adapted from https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/datasets/indexed_dataset.py

import hashlib
import os
import struct
import time
from functools import lru_cache
from itertools import accumulate

import numpy as np
import torch

from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc

dtypes = {
    1: np.uint8,
    2: np.int8,
    3: np.int16,
    4: np.int32,
    5: np.int64,
    6: np.float64,
    7: np.float32,
    8: np.uint16,
}


def print_rank_0(message):
    """If distributed is initialized, print only on rank 0."""
    if gpc.is_rank_for_log():
        print(message, flush=True)


def code(dtype):
    for k, v in dtypes.items():
        if v == dtype:
            return k
    raise ValueError(dtype)


def index_file_path(prefix_path):
    return prefix_path + ".idx"


def data_file_path(prefix_path):
    return prefix_path + ".bin"


def read_longs(f, n):
    a = np.empty(n, dtype=np.int64)
    f.readinto(a)
    return a


def _warmup_mmap_file(path):
    with open(path, "rb") as stream:
        while stream.read(100 * 1024 * 1024):
            pass


def _build_shuffle_idx(num_samples, total_size, np_rng):
    """Build the range [0, size) and shuffle."""
    dtype_ = np.uint32
    if total_size >= (np.iinfo(np.uint32).max - 1):
        dtype_ = np.int64

    shuffle_idx_first = np.arange(start=0, stop=num_samples, step=1, dtype=dtype_)
    np_rng.shuffle(shuffle_idx_first)
    if num_samples == total_size:
        return shuffle_idx_first

    shuffle_idx_last = np.arange(start=num_samples, stop=total_size, step=1, dtype=dtype_)
    np_rng.shuffle(shuffle_idx_last)

    return np.concatenate((shuffle_idx_first, shuffle_idx_last))


def _build_doc_idx(documents, num_epochs, np_rng, separate_last_epoch):
    """Build an array with length = number-of-epochs * number-of-dcuments.
    Each index is mapped to a corresponding document."""
    if not separate_last_epoch or num_epochs == 1:
        doc_idx = np.mgrid[0:num_epochs, 0 : len(documents)][1]
        doc_idx[:] = documents
        doc_idx = doc_idx.reshape(-1)
        doc_idx = doc_idx.astype(np.int32)
        np_rng.shuffle(doc_idx)
        return doc_idx

    doc_idx_first = _build_doc_idx(documents, num_epochs - 1, np_rng, False)
    doc_idx_last = _build_doc_idx(documents, 1, np_rng, False)
    return np.concatenate((doc_idx_first, doc_idx_last))


def _num_tokens(documents, sizes):
    """Total number of tokens in the dataset."""
    return np.sum(sizes[documents])


def _num_epochs(tokens_per_epoch, seq_length, num_samples):
    """Based on number of samples and sequence lenght, calculate how many
    epochs will be needed."""
    num_epochs = 0
    total_tokens = 0
    while True:
        num_epochs += 1
        total_tokens += tokens_per_epoch
        # -1 is because we need to retrieve seq_length + 1 token each time
        # but the last token will overlap with the first token of the next
        # sample except for the last sample.
        if ((total_tokens - 1) // seq_length) >= num_samples:
            return num_epochs


def _build_index_mappings(
    name, data_prefix, documents, sizes, splits_string, num_samples, seq_length, seed, *, data_cache_path
):
    """Build doc-idx, sample-idx, and shuffle-idx.
    doc-idx: is an array (ordered) of documents to be used in training.
    sample-idx: is the start document index and document offset for each
       training sample.
    shuffle-idx: maps the sample index into a random index into sample-idx.
    """
    # Number of tokens in each epoch and number of required epochs.
    tokens_per_epoch = _num_tokens(documents, sizes)
    num_epochs = _num_epochs(tokens_per_epoch, seq_length, num_samples)

    # rng state
    np_rng = np.random.RandomState(seed=seed)

    # Filename of the index mappings.
    desc = "GPT Dataset\n\n"
    desc += f"Data prefix {data_prefix}\n"
    desc += f"Dataset name {name}\n"
    desc += f"Number of samples {num_samples}\n"
    desc += f"Sequence length {seq_length}\n"
    desc += f"Random seed {seed}\n"
    desc += f"Split {splits_string}\n"
    desc_hash = hashlib.md5(desc.encode("utf-8")).hexdigest()
    desc_filename = desc_hash + ".dsc"
    doc_idx_filename = desc_hash + "_doc_idx.npy"
    sample_idx_filename = desc_hash + "_sample_idx.npy"
    shuffle_idx_filename = desc_hash + "_shuffle_idx.npy"

    # Look for cache in main data dir first to avoid unnecessary
    # duplication, then look in data-cache-path if specified,
    # If nothing is found, use the last path looked in
    build_indices = True
    prefixes = [os.path.join(os.path.dirname(data_prefix), "index-cache")]
    if data_cache_path is not None:
        prefixes.append(data_cache_path)
    for prefix in prefixes:
        idx_path = {
            "desc": os.path.join(prefix, desc_filename),
            "doc": os.path.join(prefix, doc_idx_filename),
            "sample": os.path.join(prefix, sample_idx_filename),
            "shuffle": os.path.join(prefix, shuffle_idx_filename),
        }
        for f in idx_path.values():
            if not os.path.isfile(f):
                break
        else:
            # Found our files!
            build_indices = False
            break
    data_cache_dir = os.path.dirname(idx_path["desc"])
    data_cache_success = True

    # Build the indexed mapping if not exist.
    if build_indices and gpc.is_rank_for_log():
        print_rank_0(" > WARNING: could not find index map files, building " "the indices on rank 0 ...")

        # For the last epoch, decide whether include the entire epoch
        # in the global shuffle or not.

        # If we need only one epoch, then separating last epoch  does
        # not mean anything.
        if num_epochs == 1:
            separate_last_epoch = False
            print(" > only one epoch required, setting " "separate_last_epoch to False", flush=True)

        else:
            # Get the number of samples for the last epoch
            num_samples_from_epochs_minus_one = ((num_epochs - 1) * tokens_per_epoch - 1) // seq_length
            last_epoch_num_samples = num_samples - num_samples_from_epochs_minus_one
            assert last_epoch_num_samples >= 0, "last epoch number of samples should be non-negative."
            num_samples_per_epoch = (tokens_per_epoch - 1) // seq_length
            assert last_epoch_num_samples <= (
                num_samples_per_epoch + 1
            ), "last epoch number of samples exceeded max value."
            # If we have less than 80% of the samples for the last epoch,
            # seperate out the epoch and treat it differently.
            # Note: the 80% number is just based on common sense and can
            # be adjusted if needed.
            separate_last_epoch = last_epoch_num_samples < int(0.80 * num_samples_per_epoch)
            if separate_last_epoch:
                string = (
                    " > last epoch number of samples ({}) is smaller "
                    "than 80% of number of samples per epoch ({}), "
                    "setting separate_last_epoch to True"
                )
            else:
                string = (
                    " > last epoch number of samples ({}) is larger "
                    "than 80% of number of samples per epoch ({}), "
                    "setting separate_last_epoch to False"
                )
            print(string.format(last_epoch_num_samples, num_samples_per_epoch), flush=True)

        try:
            os.makedirs(data_cache_dir, exist_ok=True)

            # description
            with open(idx_path["desc"], "wt") as fd:
                fd.write(desc)

            # doc-idx.
            start_time = time.time()
            doc_idx = _build_doc_idx(documents, num_epochs, np_rng, separate_last_epoch)
            np.save(idx_path["doc"], doc_idx, allow_pickle=True)
            print_rank_0(
                " > elasped time to build and save doc-idx mapping " "(seconds): {:4f}".format(time.time() - start_time)
            )
            # sample-idx.
            start_time = time.time()
            # Use C++ implementation for speed.
            # First compile and then import.
            from internlm.data.megatron import helpers

            assert doc_idx.dtype == np.int32
            assert sizes.dtype == np.int32
            sample_idx = helpers.build_sample_idx(sizes, doc_idx, seq_length, num_epochs, tokens_per_epoch)
            np.save(idx_path["sample"], sample_idx, allow_pickle=True)
            print_rank_0(
                " > elasped time to build and save sample-idx mapping "
                "(seconds): {:4f}".format(time.time() - start_time)
            )
            # shuffle-idx.
            start_time = time.time()
            # -1 is due to data structure used to retieve the index:
            #    sample i --> [sample_idx[i], sample_idx[i+1])
            if separate_last_epoch:
                num_samples_ = num_samples_from_epochs_minus_one
            else:
                num_samples_ = sample_idx.shape[0] - 1
            shuffle_idx = _build_shuffle_idx(num_samples_, sample_idx.shape[0] - 1, np_rng)
            np.save(idx_path["shuffle"], shuffle_idx, allow_pickle=True)
            print_rank_0(
                " > elasped time to build and save shuffle-idx mapping"
                " (seconds): {:4f}".format(time.time() - start_time)
            )
        except OSError:
            print(f"There was an error trying to create the data cache directory ({data_cache_dir})")
            print('or a file in it. This defaults to a directory "index-cache" within the directory')
            print("the data files are in and can be set with the --data-cache-path argument. Please")
            print("ensure you have write access to this directory or specify one that you do have")
            print("write access to.")
            data_cache_success = False

    counts = torch.cuda.LongTensor([data_cache_success])

    if gpc.is_using_parallel_mode(ParallelMode.DATA):
        torch.distributed.all_reduce(counts, group=gpc.get_group(ParallelMode.DATA))

    if gpc.is_using_parallel_mode(ParallelMode.PIPELINE):
        torch.distributed.all_reduce(counts, group=gpc.get_group(ParallelMode.PIPELINE))

    assert counts[0].item() == (
        gpc.get_world_size(ParallelMode.GLOBAL) // gpc.get_world_size(ParallelMode.TENSOR)
    ), "Data index creation unsuccessful!"

    # Load mappings.
    start_time = time.time()
    print_rank_0(f" > loading doc-idx mapping from {idx_path['doc']}")
    doc_idx = np.load(idx_path["doc"], allow_pickle=True, mmap_mode="r")

    print_rank_0(f" > loading sample-idx mapping from {idx_path['sample']}")
    sample_idx = np.load(idx_path["sample"], allow_pickle=True, mmap_mode="r")

    print_rank_0(f" > loading shuffle-idx mapping from {idx_path['shuffle']}")
    shuffle_idx = np.load(idx_path["shuffle"], allow_pickle=True, mmap_mode="r")

    print_rank_0("    loaded indexed file in {:3.3f} seconds".format(time.time() - start_time))
    print_rank_0("    total number of samples: {}".format(sample_idx.shape[0]))
    print_rank_0("    total number of epochs: {}".format(num_epochs))

    return doc_idx, sample_idx, shuffle_idx, desc, desc_hash


class GPTDataset(torch.utils.data.Dataset):
    """
    GPTDataset
    """

    def __init__(
        self,
        name,
        data_prefix,
        documents,
        indexed_dataset,
        splits_string,
        num_samples,
        seq_length,
        seed,
        return_doc_ids=False,
        *,
        data_cache_path=None,
    ):

        self.name = name
        self.indexed_dataset = indexed_dataset
        self.return_doc_ids = return_doc_ids

        # Checks
        assert np.min(documents) >= 0
        assert np.max(documents) < indexed_dataset.sizes.shape[0]

        # Build index mappings.
        self.doc_idx, self.sample_idx, self.shuffle_idx, self.desc, self.desc_hash = _build_index_mappings(
            self.name,
            data_prefix,
            documents,
            self.indexed_dataset.sizes,
            splits_string,
            num_samples,
            seq_length,
            seed,
            data_cache_path=data_cache_path,
        )

    def __len__(self):
        # -1 is due to data structure used to retieve the index:
        #    sample i --> [sample_idx[i], sample_idx[i+1])
        return self.sample_idx.shape[0] - 1

    def __getitem__(self, idx):
        # Get the shuffled index.
        idx = self.shuffle_idx[idx]
        # Start and end documents and offsets.
        doc_index_f = self.sample_idx[idx][0]
        doc_index_l = self.sample_idx[idx + 1][0]
        offset_f = self.sample_idx[idx][1]
        offset_l = self.sample_idx[idx + 1][1]
        # If we are within the same document, just extract the chunk.
        doc_ids = []
        if doc_index_f == doc_index_l:
            doc_ids.append(self.doc_idx[doc_index_f])
            sample = self.indexed_dataset.get(
                self.doc_idx[doc_index_f], offset=offset_f, length=offset_l - offset_f + 1
            )
        else:
            # Otherwise, get the rest of the initial document.
            doc_ids.append(self.doc_idx[doc_index_f])
            sample_list = [self.indexed_dataset.get(self.doc_idx[doc_index_f], offset=offset_f)]
            # Loop over all in between documents and add the entire document.
            for i in range(doc_index_f + 1, doc_index_l):
                doc_ids.append(self.doc_idx[i])
                sample_list.append(self.indexed_dataset.get(self.doc_idx[i]))
            # And finally add the relevant portion of last document.
            doc_ids.append(self.doc_idx[doc_index_l])
            sample_list.append(self.indexed_dataset.get(self.doc_idx[doc_index_l], length=offset_l + 1))
            sample = np.concatenate(sample_list)

        if self.return_doc_ids:  # for retro preprocessing
            return {"text": np.array(sample, dtype=np.int64), "doc_ids": np.array(doc_ids, dtype=np.int64)}
        else:
            return {"text": np.array(sample, dtype=np.int64)}


def infer_dataset_impl(path):
    if IndexedDataset.exists(path):
        with open(index_file_path(path), "rb") as f:
            magic = f.read(8)
            if magic == IndexedDataset._HDR_MAGIC:
                return "cached"
            elif magic == MMapIndexedDataset.Index._HDR_MAGIC[:8]:
                return "mmap"
            else:
                return None
    else:
        print(f"Dataset does not exist: {path}")
        print("Path should be a basename that both .idx and .bin can be appended to get full filenames.")
        return None


def make_dataset(path, impl, skip_warmup=False):
    if not IndexedDataset.exists(path):
        print(f"Dataset does not exist: {path}")
        print("Path should be a basename that both .idx and .bin can be appended to get full filenames.")
        return None
    if impl == "infer":
        impl = infer_dataset_impl(path)
    if impl == "lazy" and IndexedDataset.exists(path):
        return IndexedDataset(path)
    elif impl == "cached" and IndexedDataset.exists(path):
        return IndexedCachedDataset(path)
    elif impl == "mmap" and MMapIndexedDataset.exists(path):
        return MMapIndexedDataset(path, skip_warmup)
    print(f"Unknown dataset implementation: {impl}")
    return None


class IndexedDataset(torch.utils.data.Dataset):
    """Loader for IndexedDataset"""

    _HDR_MAGIC = b"TNTIDX\x00\x00"

    def __init__(self, path):
        super().__init__()
        self.path = path
        self.data_file = None
        self.read_index(path)

    def read_index(self, path):
        with open(index_file_path(path), "rb") as f:
            magic = f.read(8)
            assert magic == self._HDR_MAGIC, (
                "Index file doesn't match expected format. " "Make sure that --dataset-impl is configured properly."
            )
            version = f.read(8)
            assert struct.unpack("<Q", version) == (1,)
            code, self.element_size = struct.unpack("<QQ", f.read(16))
            self.dtype = dtypes[code]
            self._len, self.s = struct.unpack("<QQ", f.read(16))
            self.doc_count = struct.unpack("<Q", f.read(8))
            self.dim_offsets = read_longs(f, self._len + 1)
            self.data_offsets = read_longs(f, self._len + 1)
            self.sizes = read_longs(f, self.s)
            self.doc_idx = read_longs(f, self.doc_count)

    def read_data(self, path):
        with open(data_file_path(path), "rb", buffering=0) as f:
            data_file = f.read()
        self.data_file = data_file

    def check_index(self, i):
        if i < 0 or i >= self._len:
            raise IndexError("index out of range")

    def __del__(self):
        if self.data_file:
            self.data_file.close()

    # @lru_cache(maxsize=8)
    def __getitem__(self, idx):
        if not self.data_file:
            self.read_data(self.path)
        if isinstance(idx, int):
            i = idx
            self.check_index(i)
            tensor_size = self.sizes[self.dim_offsets[i] : self.dim_offsets[i + 1]]
            a = np.empty(tensor_size, dtype=self.dtype)
            self.data_file.seek(self.data_offsets[i] * self.element_size)
            self.data_file.readinto(a)
            return a
        elif isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            if step != 1:
                raise ValueError("Slices into indexed_dataset must be contiguous")
            sizes = self.sizes[self.dim_offsets[start] : self.dim_offsets[stop]]
            size = sum(sizes)
            a = np.empty(size, dtype=self.dtype)
            self.data_file.seek(self.data_offsets[start] * self.element_size)
            self.data_file.readinto(a)
            offsets = list(accumulate(sizes))
            sents = np.split(a, offsets[:-1])
            return sents

    def __len__(self):
        return self._len

    def num_tokens(self, index):
        return self.sizes[index]

    def size(self, index):
        return self.sizes[index]

    @staticmethod
    def exists(path):
        return os.path.exists(index_file_path(path)) and os.path.exists(data_file_path(path))

    @property
    def supports_prefetch(self):
        return False  # avoid prefetching to save memory


class IndexedCachedDataset(IndexedDataset):
    """
    IndexedCachedDataset
    """

    def __init__(self, path):
        super().__init__(path)
        self.cache = None
        self.cache_index = {}

    @property
    def supports_prefetch(self):
        return True

    def prefetch(self, indices):
        if all(i in self.cache_index for i in indices):
            return
        if not self.data_file:
            self.read_data(self.path)
        indices = sorted(set(indices))
        total_size = 0
        for i in indices:
            total_size += self.data_offsets[i + 1] - self.data_offsets[i]
        self.cache = np.empty(total_size, dtype=self.dtype)
        ptx = 0
        self.cache_index.clear()
        for i in indices:
            self.cache_index[i] = ptx
            size = self.data_offsets[i + 1] - self.data_offsets[i]
            a = self.cache[ptx : ptx + size]
            self.data_file.seek(self.data_offsets[i] * self.element_size)
            self.data_file.readinto(a)
            ptx += size
        if self.data_file:
            # close and delete data file after prefetch so we can pickle
            self.data_file.close()
            self.data_file = None

    # @lru_cache(maxsize=8)
    def __getitem__(self, idx):
        if isinstance(idx, int):
            i = idx
            self.check_index(i)
            tensor_size = self.sizes[self.dim_offsets[i] : self.dim_offsets[i + 1]]
            a = np.empty(tensor_size, dtype=self.dtype)
            ptx = self.cache_index[i]
            np.copyto(a, self.cache[ptx : ptx + a.size])
            return a
        elif isinstance(idx, slice):
            # Hack just to make this work, can optimizer later if necessary
            sents = []
            for i in range(*idx.indices(len(self))):
                sents.append(self[i])
            return sents


class MMapIndexedDataset(torch.utils.data.Dataset):
    """
    MMapIndexedDataset
    """

    class Index(object):
        """
        Index
        """

        _HDR_MAGIC = b"MMIDIDX\x00\x00"

        @classmethod
        def writer(cls, path, dtype):
            class _Writer(object):
                """
                _Writer
                """

                def __enter__(self):
                    self._file = open(path, "wb")

                    self._file.write(cls._HDR_MAGIC)
                    self._file.write(struct.pack("<Q", 1))
                    self._file.write(struct.pack("<B", code(dtype)))

                    return self

                @staticmethod
                def _get_pointers(sizes):
                    dtype_size = dtype().itemsize
                    address = 0
                    pointers = []

                    for size in sizes:
                        pointers.append(address)
                        address += size * dtype_size

                    return pointers

                def write(self, sizes, doc_idx):
                    pointers = self._get_pointers(sizes)

                    self._file.write(struct.pack("<Q", len(sizes)))
                    self._file.write(struct.pack("<Q", len(doc_idx)))

                    sizes = np.array(sizes, dtype=np.int32)
                    self._file.write(sizes.tobytes(order="C"))
                    del sizes

                    pointers = np.array(pointers, dtype=np.int64)
                    self._file.write(pointers.tobytes(order="C"))
                    del pointers

                    doc_idx = np.array(doc_idx, dtype=np.int64)
                    self._file.write(doc_idx.tobytes(order="C"))

                def __exit__(self, exc_type, exc_val, exc_tb):
                    self._file.close()

            return _Writer()

        def __init__(self, path, skip_warmup=False):
            with open(path, "rb") as stream:
                magic_test = stream.read(9)
                assert self._HDR_MAGIC == magic_test, (
                    "Index file doesn't match expected format. " "Make sure that --dataset-impl is configured properly."
                )
                version = struct.unpack("<Q", stream.read(8))
                assert (1,) == version

                (dtype_code,) = struct.unpack("<B", stream.read(1))
                self._dtype = dtypes[dtype_code]
                self._dtype_size = self._dtype().itemsize

                self._len = struct.unpack("<Q", stream.read(8))[0]
                self._doc_count = struct.unpack("<Q", stream.read(8))[0]
                offset = stream.tell()

            if not skip_warmup:
                print_rank_0("    warming up index mmap file...")
                _warmup_mmap_file(path)

            self._bin_buffer_mmap = np.memmap(path, mode="r", order="C")
            self._bin_buffer = memoryview(self._bin_buffer_mmap)
            print_rank_0("    reading sizes...")
            self._sizes = np.frombuffer(self._bin_buffer, dtype=np.int32, count=self._len, offset=offset)
            print_rank_0("    reading pointers...")
            self._pointers = np.frombuffer(
                self._bin_buffer, dtype=np.int64, count=self._len, offset=offset + self._sizes.nbytes
            )
            print_rank_0("    reading document index...")
            self._doc_idx = np.frombuffer(
                self._bin_buffer,
                dtype=np.int64,
                count=self._doc_count,
                offset=offset + self._sizes.nbytes + self._pointers.nbytes,
            )

        def __del__(self):
            self._bin_buffer_mmap._mmap.close()
            del self._bin_buffer_mmap

        @property
        def dtype(self):
            return self._dtype

        @property
        def sizes(self):
            return self._sizes

        @property
        def doc_idx(self):
            return self._doc_idx

        @lru_cache(maxsize=8)
        def __getitem__(self, i):
            return self._pointers[i], self._sizes[i]

        def __len__(self):
            return self._len

    def __init__(self, path, skip_warmup=False):
        super().__init__()

        self._path = None
        self._index = None
        self._bin_buffer = None

        self._do_init(path, skip_warmup)

    def __getstate__(self):
        return self._path

    def __setstate__(self, state):
        self._do_init(state, skip_warmup=True)

    def _do_init(self, path, skip_warmup):
        self._path = path
        self._index = self.Index(index_file_path(self._path), skip_warmup)

        if not skip_warmup:
            print_rank_0("    warming up data mmap file...")
            _warmup_mmap_file(data_file_path(self._path))
        print_rank_0("    creating numpy buffer of mmap...")
        self._bin_buffer_mmap = np.memmap(data_file_path(self._path), mode="r", order="C")
        print_rank_0("    creating memory view of numpy buffer...")
        self._bin_buffer = memoryview(self._bin_buffer_mmap)

    def __del__(self):
        self._bin_buffer_mmap._mmap.close()
        del self._bin_buffer_mmap
        del self._index

    def __len__(self):
        return len(self._index)

    # @lru_cache(maxsize=8)
    def __getitem__(self, idx):
        if isinstance(idx, (int, np.integer)):
            ptr, size = self._index[idx]
            np_array = np.frombuffer(self._bin_buffer, dtype=self._index.dtype, count=size, offset=ptr)
            return np_array
        elif isinstance(idx, slice):
            start, _, step = idx.indices(len(self))
            if step != 1:
                raise ValueError("Slices into indexed_dataset must be contiguous")
            ptr = self._index._pointers[start]
            sizes = self._index._sizes[idx]
            offsets = list(accumulate(sizes))
            total_size = sum(sizes)
            np_array = np.frombuffer(self._bin_buffer, dtype=self._index.dtype, count=total_size, offset=ptr)
            sents = np.split(np_array, offsets[:-1])
            return sents
        else:
            raise TypeError("Unexpected type received for idx: {}".format(type(idx)))

    def get(self, idx, offset=0, length=None):
        """Retrieves a single item from the dataset with the option to only
        return a portion of the item.

        get(idx) is the same as [idx] but get() does not support slicing.
        """
        ptr, size = self._index[idx]
        if length is None:
            length = size - offset
        ptr += offset * np.dtype(self._index.dtype).itemsize
        np_array = np.frombuffer(self._bin_buffer, dtype=self._index.dtype, count=length, offset=ptr)
        return np_array

    @property
    def sizes(self):
        return self._index.sizes

    @property
    def doc_idx(self):
        return self._index.doc_idx

    def get_doc_idx(self):
        return self._index._doc_idx

    def set_doc_idx(self, doc_idx_):
        self._index._doc_idx = doc_idx_

    @property
    def supports_prefetch(self):
        return False

    @staticmethod
    def exists(path):
        return os.path.exists(index_file_path(path)) and os.path.exists(data_file_path(path))


def get_indexed_dataset_(data_prefix, data_impl, skip_warmup):
    """Build indexed dataset."""
    print_rank_0(" > building dataset index ...")

    start_time = time.time()
    indexed_dataset = make_dataset(data_prefix, data_impl, skip_warmup)
    print_rank_0(" > finished creating indexed dataset in {:4f} " "seconds".format(time.time() - start_time))
    print_rank_0("    number of documents: {}".format(indexed_dataset.sizes.shape[0]))

    return indexed_dataset


def build_megatron_dataset(
    data_prefix,
    seq_len,
    seed,
):
    # Indexed dataset.
    indexed_dataset = get_indexed_dataset_(data_prefix, data_impl="infer", skip_warmup=True)

    # GPT dataset.
    return GPTDataset(
        name="train",
        data_prefix=data_prefix,
        documents=np.arange(start=0, stop=indexed_dataset.sizes.shape[0], step=1, dtype=np.int32),
        indexed_dataset=indexed_dataset,
        splits_string="1.0, 0.0, 0.0",  # proportion of dataset for train/valid/test, we set 1.0 for train only
        num_samples=gpc.config.data.micro_bsz
        * gpc.config.data.micro_num
        * gpc.get_world_size(ParallelMode.DATA)
        * gpc.config.data.total_steps,  # total number of train samples
        seq_length=seq_len,
        seed=seed,
    )
