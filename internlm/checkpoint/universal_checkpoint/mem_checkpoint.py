#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# adopted from https://github.com/volcengine/veScale/tree/main/vescale/checkpoint/utilities

import io
import pickle
import threading
from typing import DefaultDict

import torch

from internlm.utils.logger import get_logger

logger = get_logger(__file__)

if hasattr(torch.storage, "TypedStorage"):
    TypedStorage = torch.storage.TypedStorage
elif hasattr(torch.storage, "_TypedStorage"):
    TypedStorage = torch.storage._TypedStorage

# TypedStorage changes in pytorch 2.
if torch.__version__ >= "2":

    def untyped_storage(o):
        return o.untyped_storage()

    def location_caster(o):
        return o

elif torch.__version__ >= "1.11":

    def untyped_storage(o):
        return o.storage()._storage

    def location_caster(o):
        return o._storage if isinstance(o, TypedStorage) else o


try:
    lib = torch.cuda.cudart()
except Exception:
    lib = None


class PinnedStoragePool:  # pylint: disable=C0115
    def __init__(self):
        self._l = threading.Lock()
        self._m = DefaultDict(set)

    def allocate(self, nbytes: int):
        with self._l:
            # We don't really need storage to have the exact size. So in theory we can find a
            # bigger storage that may suit here. But so far we keep everything simple here.
            s = self._m[nbytes]
            if not s:
                t = torch.empty([nbytes], dtype=torch.uint8)
                t = t.share_memory_()
                if lib is not None and nbytes != 0:
                    err = lib.cudaHostRegister(t.data_ptr(), t.numel() * t.element_size(), 0)
                    assert err == 0, err
                storage = untyped_storage(t)
                s.add(storage)
            return s.pop()

    def deallocate(self, s):
        # WARNING: Call deallocate when the reference to CPU tensor goes to zero
        # so the memory pool will reuse the memory if possbile
        # Othterwise, the memory pool will allocate memory on the used memory range,
        # leading to cuda error 712 cudaErrorHostMemoryAlreadyRegistered
        with self._l:
            self._m[s.nbytes()].add(s)


GLOBAL_POOL = PinnedStoragePool()

TID = threading.get_ident()


def copy_gpu_tensor_to_cpu_pinned_mem_pool(tensor: torch.Tensor, non_blocking=False) -> torch.Tensor:
    """
    Copy a tensor on GPU to pinned memory pool (host CPU memory).
    The input tensor will not be modified
    Args:
        tensor: a tensor on cuda device
    Return:
        a tensor on cpu, whose data is the same as input tensor
    """
    m = {}
    _old_warning = getattr(torch.storage, "_warn_typed_storage_removal", None)
    torch.storage._warn_typed_storage_removal = lambda *args, **kwags: None

    def persistent_id(o):
        if torch.is_storage(o) or isinstance(o, TypedStorage):
            storage = o
            if storage._cdata in m:
                return storage._cdata
            if storage.device.type != "cpu":
                copied = GLOBAL_POOL.allocate(storage.nbytes())
                copied.copy_(storage, non_blocking=non_blocking)
                if isinstance(storage, TypedStorage):
                    copied = storage._new_wrapped_storage(copied)
            else:
                copied = storage.clone()
            m[storage._cdata] = copied
            return storage._cdata
        return

    b = io.BytesIO()
    p = pickle.Pickler(b)
    p.persistent_id = persistent_id
    p.dump(tensor)
    b.seek(0)
    up = pickle.Unpickler(b)
    up.persistent_load = lambda i: m[i]
    cpu_tensor = up.load()
    """
    assert type(tensor) == torch.Tensor
    storage_obj = tensor.storage()
    cpu_storage = GLOBAL_POOL.allocate(storage_obj.nbytes())

    cpu_storage.copy_(storage_obj, non_blocking=non_blocking)
    cpu_tensor = torch.tensor(cpu_storage)
    """
    torch.storage._warn_typed_storage_removal = _old_warning
    return cpu_tensor


def deallocate_cpu_tensor_in_pinned_mem_pool(tensor: torch.Tensor):
    "Deallocate CPU tensor in the global pinned memory pool"
    GLOBAL_POOL.deallocate(tensor.untyped_storage())
