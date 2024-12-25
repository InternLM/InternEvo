#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# adopted from https://github.com/volcengine/veScale/blob/main/vescale/checkpoint/storage

import collections
import dataclasses
import io
import os
import pickle
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Union, cast

import torch
import torch.distributed as dist
from torch import Tensor
from torch._utils import _get_device_module
from torch.distributed._shard._utils import narrow_tensor_by_index
from torch.distributed.checkpoint.metadata import Metadata, MetadataIndex
from torch.distributed.checkpoint.planner import (
    LoadPlan,
    LoadPlanner,
    ReadItem,
    SavePlan,
    SavePlanner,
    WriteItem,
    WriteItemType,
)
from torch.distributed.checkpoint.storage import (
    StorageReader,
    StorageWriter,
    WriteResult,
)
from torch.distributed.checkpoint.utils import _create_file_view
from torch.futures import Future

from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from internlm.train.pipeline import map_fqn_global_to_local
from internlm.utils.common import get_current_device
from internlm.utils.logger import get_logger

from .mem_checkpoint import (
    copy_gpu_tensor_to_cpu_pinned_mem_pool,
    deallocate_cpu_tensor_in_pinned_mem_pool,
)

logger = get_logger(__file__)

__all__ = [
    "FileSystemWriter",
    "FileSystemReader",
]


@dataclass
class _StorageInfo:
    """
    This is the per entry storage info
    """

    relative_path: str
    offset: int
    length: int


@dataclass
class _StoragePrefix:
    prefix: str


DEFAULT_SUFFIX = ".distcp"


def _result_from_write_item(item: WriteItem, size_in_bytes, storage_data) -> WriteResult:
    return WriteResult(index=item.index, size_in_bytes=size_in_bytes, storage_data=storage_data)


class _TensorLoader(ABC):
    """
    Abstract class
    """

    @abstractmethod
    def add(self, fqn, size, obj):
        pass

    @abstractmethod
    def start_loading(self):
        pass

    @abstractmethod
    def values(self):
        pass


class _SerialCpuLoader(_TensorLoader):
    """
    Abstract class
    currently no use
    """

    def __init__(self, resolve_fun, p2p_tensors_info=None):
        self.resolve_fun = resolve_fun
        self.items = []
        # For HybridZeroOptimizer, p2p_tensors_info is always none.
        self.p2p_tensors_info = p2p_tensors_info

    def add(self, fqn, size, obj):
        self.items.append((fqn, size, obj))

    def start_loading(self):
        pass

    def values(self):
        for _, _, obj in self.items:
            tensor = self.resolve_fun(obj).detach()
            tensor = copy_gpu_tensor_to_cpu_pinned_mem_pool(tensor)
            # Comment the original DCP code
            # When dumping to pinned memory,
            # the memory layout for tensor has been contiguous
            #   if tensor.storage().size() != tensor.numel():
            #       tensor = tensor.clone()
            yield (
                tensor,
                obj,
            )


class _OverlappingCpuLoader(_TensorLoader):
    """
    currently no use
    """

    def __init__(
        self,
        resolve_fun,
        p2p_tensors_info=None,
        stream=None,
        inflight_threshhold=1_000_000,
    ):
        self.resolve_fun = resolve_fun
        self.items = []
        self.inflight_threshhold = inflight_threshhold
        self.in_flight_data = 0
        self.current_items: collections.deque = collections.deque()
        self.idx = 0
        self.started = False
        self.device_type = stream.device_type if stream else torch.device("cuda").type
        self.device_module = _get_device_module(self.device_type)
        # For HybridZeroOptimizer, p2p_tensors_info is always none.
        self.p2p_tensors_info = p2p_tensors_info
        self.stream = stream or self.device_module.current_stream()
        if self.stream != self.device_module.current_stream():
            self.stream.wait_stream(self.device_module.current_stream())

    @property
    def _done(self):
        return self.idx >= len(self.items)

    def _drain(self):
        drained = []
        if self.in_flight_data >= self.inflight_threshhold:
            self.stream.synchronize()
        while self.in_flight_data >= self.inflight_threshhold:
            val = self.current_items.popleft()
            self.in_flight_data -= val[0].numel() * val[0].element_size()
            drained.append(val)
        return drained

    def _refill(self):
        with self.device_module.stream(self.stream):
            while not self._done and self.in_flight_data < self.inflight_threshhold:
                _, _, obj = self.items[self.idx]
                self.idx += 1
                tensor = self.resolve_fun(obj).detach()
                if tensor.device.type == self.device_type:
                    tensor = copy_gpu_tensor_to_cpu_pinned_mem_pool(tensor, non_blocking=True)

                self.current_items.append(
                    (
                        tensor,
                        obj,
                    )
                )
                self.in_flight_data += tensor.numel() * tensor.element_size()

    def _finish(self):
        assert self._done
        if len(self.current_items) > 0:
            self.stream.synchronize()
        return self.current_items

    def add(self, fqn, size, obj):
        if self.started:
            raise RuntimeError("cannot add items after loading started")
        self.items.append((fqn, size, obj))

    def start_loading(self):
        if self.started:
            return
        self.started = True
        self.items.sort(key=lambda x: x[1])
        self._refill()

    def values(self):
        self.start_loading()
        while not self._done:
            drained = self._drain()
            self._refill()
            yield from drained

        yield from self._finish()


def _item_fqn(item: WriteItem) -> str:
    return item.index.fqn


def _item_size(item: WriteItem) -> int:
    size = 1
    assert item.tensor_data is not None
    # can't use math.prod as PT needs to support older python
    for s in item.tensor_data.size:
        size *= s

    dtype = item.tensor_data.properties.dtype
    return size * torch._utils._element_size(dtype)


def _split_by_size_and_type(bins, items: List[WriteItem]) -> List[List[WriteItem]]:
    if bins == 1:
        return [items]

    bytes_w = [wi for wi in items if wi.type == WriteItemType.BYTE_IO]
    tensor_w = [wi for wi in items if wi.type != WriteItemType.BYTE_IO]
    assert len(bytes_w) == 0, "currently no WriteItemType.BYTE_IO"

    buckets: List[List[WriteItem]] = [[] for _ in range(bins)]
    bucket_sizes = [0 for _ in range(bins)]

    tensor_w.sort(key=_item_size, reverse=True)

    for i, wi in enumerate(bytes_w):
        buckets[i % bins].append(wi)

    for wi in tensor_w:
        idx = min(enumerate(bucket_sizes), key=lambda x: x[1])[0]
        buckets[idx].append(wi)
        bucket_sizes[idx] += _item_size(wi)

    return buckets


def _write_item(stream, data, write_item, storage_key):
    offset = stream.tell()
    assert isinstance(data, torch.Tensor)
    assert data.device == torch.device("cpu")
    torch.save(data, stream)
    length = stream.tell() - offset

    return _result_from_write_item(write_item, length, _StorageInfo(storage_key, offset, length))


def _write_files_from_queue(
    file_name,
    storage_key,
    write_items,
    planner: SavePlanner,
    inflight_threshhold: int,
    use_fsync: bool,
    p2p_tensors_info=None,  # currently no use
):
    loader: _TensorLoader

    if torch.cuda.is_available() and inflight_threshhold > 0:
        loader = _OverlappingCpuLoader(
            lambda x: planner.resolve_data(x),  # pylint: disable=W0108
            inflight_threshhold=inflight_threshhold,
            p2p_tensors_info=p2p_tensors_info,
        )
    else:
        loader = _SerialCpuLoader(
            lambda x: planner.resolve_data(x), p2p_tensors_info=p2p_tensors_info  # pylint: disable=W0108
        )

    tensor_w = list(write_items)
    for write_item in tensor_w:
        loader.add(_item_fqn(write_item), _item_size(write_item), write_item)
    loader.start_loading()

    write_results = []
    stream = open(file_name, "wb")  # pylint: disable=R1732

    for tensor, write_item in loader.values():
        assert tensor.is_cpu
        write_results.append(_write_item(stream, tensor, write_item, storage_key))
        # WARNING: Call deallocate_cpu_tensor_in_pinned_mem_pooltensor
        # when the reference to CPU tensor goes to zero
        # so the memory pool will reuse the memory if possbile
        # Othterwise, the memory pool will allocate memory on the used memory range,
        # leading to cuda error 712 cudaErrorHostMemoryAlreadyRegistered
        deallocate_cpu_tensor_in_pinned_mem_pool(tensor)

    if use_fsync:
        os.fsync(stream.fileno())

    stream.close()
    return write_results


def _serialize_tensor(tensor: torch.Tensor) -> bytes:
    bio = io.BytesIO()
    # NOTE: currently use torch.save() to do the serialization.
    torch.save(tensor, bio)
    return bio.getbuffer()


def _write_to_file(stream, content: bytes, write_item: WriteItem, storage_key: str) -> WriteResult:
    offset = stream.tell()
    stream.write(content)
    length = stream.tell() - offset
    return _result_from_write_item(write_item, length, _StorageInfo(storage_key, offset, length))


def _write_files_per_proc_pipe(
    file_path: Path,
    storage_key: str,
    byte_data_item: List[Tuple[io.BytesIO, WriteItem]],
    tensor_data_item: List[Tuple[torch.Tensor, WriteItem]],
    use_fsync: bool,
) -> List[WriteResult]:
    write_futures = []
    write_results = []
    stream = open(file_path, "wb")  # pylint: disable=R1732
    executor = ThreadPoolExecutor(max_workers=1)
    # For byte data, directly write byte data.
    assert len(byte_data_item) == 0
    for write_data, write_item in byte_data_item:
        content = write_data.getbuffer()
        write_futures.append(
            executor.submit(
                _write_to_file,
                stream,
                content,
                write_item,
                storage_key,
            )
        )

    # For tensor data, perform serialization in process then do saving in threadpool.
    for write_data, write_item in tensor_data_item:
        content = _serialize_tensor(write_data)
        write_futures.append(
            executor.submit(
                _write_to_file,
                stream,
                content,
                write_item,
                storage_key,
            )
        )

    for fut in write_futures:
        write_results.append(fut.result())
    if use_fsync:
        os.fsync(stream.fileno())
    executor.shutdown(wait=False)
    return write_results


class FileSystemWriter(StorageWriter):
    """
    Basic implementation of StorageWriter using file IO.

    This implementation makes the following assumptions and simplifications:

    * The checkpoint path is an empty or non-existing directory.
    * File creation is atomic

    The checkpoint consist of one file per write request plus
    a `.metadata` file with the serialized metadata.

    """

    def __init__(
        self,
        path: Union[str, os.PathLike],
        single_file_per_rank: bool = True,
        sync_files: bool = True,
        worker_count: int = 1,
        per_process_copy_ahead: int = 10_000_000,
    ) -> None:
        """
        Initialize the writer pointing to `path`

        Args:
            path: directory where the checkpoint will be written to.
            single_file_per_rank: Produce one file per rank instead of one file per tensor/blob. Default to True.
            sync_files : force files to be synced to permanent storage. Default to True.
            worker_count: Number of IO workers (processes) to use to write. Default to 1.
            per_process_copy_ahead: How many bytes to copy from the GPU ahead of saving then. Default 10Mb.

        N. B. If sync_files is disabled, there's no guarantee that the checkpoint will be
              consistent in the case of a failure.
        """
        super().__init__()
        self.path = Path(path)
        self.single_file_per_rank = single_file_per_rank
        # self.single_file_per_rank = False
        self.sync_files = sync_files
        self.worker_count = worker_count
        self.per_process_copy_ahead = per_process_copy_ahead

    def set_up_storage_writer(self, is_coordinator: bool) -> None:
        pass

    def prepare_local_plan(self, plan: SavePlan, p2p_tensors_info=None) -> SavePlan:
        self.path.mkdir(parents=True, exist_ok=True)
        # For HybridZeroOptimizer, p2p_tensors_info is always none.
        self.p2p_tensors_info = p2p_tensors_info
        return plan

    def prepare_global_plan(self, global_plan: List[SavePlan]) -> List[SavePlan]:  # pylint: disable=W0237
        new_plans = [
            dataclasses.replace(plan, storage_data=_StoragePrefix(f"__{i}_")) for i, plan in enumerate(global_plan)
        ]
        return new_plans

    def prepare_write_data(self, tasks: List[Tuple[Path, str, List[WriteItem]]], planner: SavePlanner, is_optimizer):
        """
        First stage of saving, Perform Copy data to CPU (D2H).

        Args:
            tasks: partitoned tasks for workers to conduct serialization and the actual saving.
            planner: save planner used to resolve the bytes and tensor data.

        NOTE: Currently we do D2H synchronously.
        """

        byte_data_item_writes: List[List[Tuple[io.BytesIO, WriteItem]]] = []
        tensor_data_item_writes: List[List[Tuple[torch.Tensor, WriteItem]]] = []
        file_path_names: List[Tuple[Path, str]] = []

        # Perform D2H in copy stream.
        for task in tasks:
            file_path, file_name, write_items = task
            byte_w = [wi for wi in write_items if wi.type == WriteItemType.BYTE_IO]
            tensor_w = [wi for wi in write_items if wi.type != WriteItemType.BYTE_IO]
            byte_data_item = [(planner.resolve_data(wi), wi) for wi in byte_w]
            assert len(byte_data_item) == 0, "currentlu no WriteItemType.BYTE_IO"
            tensor_data_item = []

            item_list = []
            # Map global fqn to local fqn to search local model state data.
            for item in tensor_w:
                fqn = _item_fqn(item)
                if not is_optimizer:
                    if fqn in map_fqn_global_to_local:  # pylint: disable=R1715
                        fqn = map_fqn_global_to_local[fqn]

                # Use tensor.clone() when saving slices to avoid unexpected memory issues.
                tensor = planner.resolve_data(item, fqn).detach().clone()
                tensor = copy_gpu_tensor_to_cpu_pinned_mem_pool(tensor, non_blocking=True)
                tensor_data_item.append((tensor, item))
                item_list.append(item.index.fqn)
            byte_data_item_writes.append(byte_data_item)
            tensor_data_item_writes.append(tensor_data_item)
            file_path_names.append((file_path, file_name))

        # Deallocate pinned memory.
        # NOTE: when prepare_write_data() is called next time, make sure the previous save event is completed.
        # Otherwise, tensors in pinned memory pool may be overwritten.
        for tensor_data_item in tensor_data_item_writes:
            for tensor, _ in tensor_data_item:
                assert tensor.is_cpu
                deallocate_cpu_tensor_in_pinned_mem_pool(tensor)

        return byte_data_item_writes, tensor_data_item_writes, file_path_names

    def write_data(
        self, plan: SavePlan, planner: SavePlanner, async_io: bool = False, io_workers=False, is_optimizer=False
    ) -> Future[List[WriteResult]]:
        storage_plan: _StoragePrefix = plan.storage_data
        file_count = 0

        def gen_file():
            nonlocal file_count
            file_name = f"{storage_plan.prefix}{file_count}{DEFAULT_SUFFIX}"
            file_count += 1
            return file_name

        tasks: List[Tuple[Path, str, List[WriteItem]]] = []
        # Generate K tasks where K is the number of worker_count.
        if self.single_file_per_rank:
            for bucket in _split_by_size_and_type(self.worker_count, plan.items):
                file_name = gen_file()
                tasks.append((self.path / file_name, file_name, bucket))
        # Generate K tasks where K is the number of write items.
        else:
            for item in plan.items:
                file_name = gen_file()
                tasks.append((self.path / file_name, file_name, [item]))

        futures = []
        if not io_workers:
            executor = ProcessPoolExecutor(max_workers=self.worker_count)
        else:
            executor = io_workers

        # ProcessPool VERSION.
        if isinstance(executor, ProcessPoolExecutor):
            byte_data_item_writes, tensor_data_item_writes, file_path_names = self.prepare_write_data(
                tasks, planner, is_optimizer
            )
            for byte_data_item, tensor_data_item, file_path_name in zip(
                byte_data_item_writes, tensor_data_item_writes, file_path_names
            ):
                file_path, storage_key = file_path_name
                worker_args = (file_path, storage_key, byte_data_item, tensor_data_item, self.sync_files)
                futures.append(executor.submit(_write_files_per_proc_pipe, *worker_args))
            if async_io:
                return futures
            else:
                for fut in futures:
                    fut.result()
                    # fut.wait()
                return futures
        else:
            # ThreadPool VERSION.
            assert False, "unavailable version"
            for task in tasks:
                futures.append(
                    executor.submit(
                        _write_files_from_queue,
                        *task,
                        planner,
                        self.per_process_copy_ahead,
                        self.sync_files,
                    )
                )
            if async_io:
                return futures
            else:
                for fut in futures:
                    fut.result()
                return futures

    def finish(self, metadata: Metadata, results: List[List[WriteResult]]) -> None:
        storage_md = dict()
        for wr_list in results:
            storage_md.update({wr.index: wr.storage_data for wr in wr_list})
        metadata.storage_data = storage_md
        with (self.path / ".metadata.tmp").open("wb") as metadata_file:
            pickle.dump(metadata, metadata_file)
            os.fsync(metadata_file.fileno())

        (self.path / ".metadata.tmp").rename(self.path / ".metadata")

    def reset(self, checkpoint_id):
        pass

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id):
        pass


class FileSystemReader(StorageReader):
    """
    Basic implementation of StorageReader using file IO.
    """

    def __init__(
        self,
        path: Union[str, os.PathLike],
        broadcast_tensors=False,
        data_parallel_process_group=None,
    ) -> None:
        super().__init__()
        self.path = path
        self.storage_data: Dict[MetadataIndex, _StorageInfo] = dict()
        self.broadcast_tensors = broadcast_tensors
        self.data_parallel_process_group = data_parallel_process_group

        # If broadcast_tensors is enabled, the data_parallel_process_group is not none
        if self.broadcast_tensors:
            assert self.data_parallel_process_group

    def _slice_file(self, file, sinfo: _StorageInfo):
        return _create_file_view(file, sinfo.offset, sinfo.length)

    def _get_file_path(self, relative_path):
        file_path = os.path.join(self.path, relative_path)
        return file_path

    def read_data(self, plan: LoadPlan, planner: LoadPlanner) -> Future[None]:
        # group requests by file
        per_file: Dict[str, List[ReadItem]] = dict()
        for read_item in plan.items:
            item_md = self.storage_data[read_item.storage_index]
            path = item_md.relative_path
            per_file.setdefault(path, []).append(read_item)

        # If broadcasting model tensors is enabled,
        # let processes with dp_rank=0 load models and broadcast them to other processes
        if self.broadcast_tensors:
            self.read_data_with_broadcast(per_file=per_file, planner=planner)
        else:
            # Otherwise, let all ranks load tensors from files
            self.read_from_files(per_file=per_file, planner=planner)

        fut: Future = Future()
        fut.set_result(None)

        return fut

    def read_from_files(self, per_file: Dict[str, List[ReadItem]], planner: LoadPlanner):
        for relative_path, reqs in per_file.items():
            file_path = self._get_file_path(relative_path)
            with open(file_path, "rb") as file:
                reqs = sorted(reqs, key=lambda req: self.storage_data[req.storage_index].offset)
                for req in reqs:
                    item_md = self.storage_data[req.storage_index]
                    file_slice = self._slice_file(file, item_md)
                    tensor = cast(Tensor, torch.load(file_slice, map_location="cpu"))  # att
                    tensor = narrow_tensor_by_index(tensor, req.storage_offsets, req.lengths)
                    target_tensor = planner.resolve_tensor(req).detach()

                    assert (
                        target_tensor.size() == tensor.size()
                    ), f"req {req.storage_index} mismatch sizes {target_tensor.size()} vs {tensor.size()}"

                    target_tensor.copy_(tensor)
                    planner.commit_tensor(req, target_tensor)

    def read_data_with_broadcast(self, per_file: Dict[str, List[ReadItem]], planner: LoadPlanner):
        for relative_path, reqs in per_file.items():
            if gpc.get_local_rank(ParallelMode.DATA) == 0:
                file_path = self._get_file_path(relative_path)
                file = open(file_path, "rb")  # pylint: disable=R1732
            dist.barrier(self.data_parallel_process_group)
            reqs = sorted(reqs, key=lambda req: self.storage_data[req.storage_index].offset)
            for req in reqs:
                if gpc.get_local_rank(ParallelMode.DATA) == 0:
                    item_md = self.storage_data[req.storage_index]
                    file_slice = self._slice_file(file, item_md)
                    object_list = [cast(Tensor, torch.load(file_slice, map_location="cuda"))]
                else:
                    object_list = [None]
                dist.broadcast_object_list(
                    object_list,
                    src=dist.get_global_rank(self.data_parallel_process_group, 0),
                    group=self.data_parallel_process_group,
                    device=get_current_device(),
                )
                tensor = object_list[0].cpu()
                tensor = narrow_tensor_by_index(tensor, req.storage_offsets, req.lengths)
                target_tensor = planner.resolve_tensor(req).detach()

                assert (
                    target_tensor.size() == tensor.size()
                ), f"req {req.storage_index} mismatch sizes {target_tensor.size()} vs {tensor.size()}"
                target_tensor.copy_(tensor)
                planner.commit_tensor(req, target_tensor)

    # Implementing the abstract function in StorageReader
    def read_metadata(self) -> Metadata:
        metadata_path = self._get_file_path(".metadata")
        with open(metadata_path, "rb") as metadata_file:
            metadata = pickle.load(metadata_file)
        return metadata

    def set_up_storage_reader(self, metadata: Metadata, is_coordinator: bool) -> None:
        self.storage_data = metadata.storage_data
        assert self.storage_data is not None

    def prepare_local_plan(self, plan: LoadPlan) -> LoadPlan:
        return plan

    def prepare_global_plan(self, global_plan: List[LoadPlan]) -> List[LoadPlan]:  # pylint: disable=W0237
        return global_plan

    def reset(self, checkpoint_id):
        pass

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id):
        pass
