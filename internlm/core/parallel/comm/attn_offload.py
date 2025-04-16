import torch
import torch_npu
from internlm.utils.common import get_current_device
from internlm.core.context import global_context as gpc
import pdb

global_attn_offload = None
global_attn_npu_offload = None

class AttnOffloadManager:
    """
    A manager for attention output CPU offloading and GPU prefetch loading.
    """

    def __init__(self, enable_cpu_offload: bool = False) -> None:
        # cpu offload overlapping
        self.cpu_offload = enable_cpu_offload
        # layer id mapping to flash attn output
        self.fa_output_mapping = {} # 存储各层注意力输出的字典
        self.fa_stream = torch_npu.npu.Stream() # CUDA流用于异步传输
        self.d2h_final_event = torch_npu.npu.Event() # device to host事件
        self.h2d_final_event = torch_npu.npu.Event() # host to device事件
        # prepare for tensor buffer
        self.tensor_id_to_tensor_bufs = {}  # 按层和id缓存的GPU缓存区

    def get_tensor_buf_for_offloaded_tensor(self, tensor, layer_id, tensor_id): # 分配存储缓存区
        """Get tensor buffer for offloaded tensor."""
        layer_id = layer_id % 2 # 分奇偶双缓冲

        # 检查对应层对应id的tensor是否在缓存中，否则分配新缓冲
        if layer_id not in self.tensor_id_to_tensor_bufs:
            self.tensor_id_to_tensor_bufs[layer_id] = {}

        if tensor_id not in self.tensor_id_to_tensor_bufs[layer_id]:
            allocate_new_buf = True
        else:
            tensor_buf = self.tensor_id_to_tensor_bufs[layer_id][tensor_id]
            allocate_new_buf = tensor_buf.size() == tensor.size() and tensor_buf.dtype == tensor.dtype

        if allocate_new_buf:
            # supposed to only execute once
            buffer = torch.empty(
                tensor.size(),
                dtype=tensor.dtype,
                layout=tensor.layout,
                device=tensor.device,
            )

            self.tensor_id_to_tensor_bufs[layer_id][tensor_id] = buffer # 将空间分配给字典，字典定层定序

        return self.tensor_id_to_tensor_bufs[layer_id][tensor_id] # 返回buffer空间

    def insert_fa_output_with_layer(self, layer_idx, output): # 构建一个输出字典
        assert layer_idx not in self.fa_output_mapping
        if self.cpu_offload is False:
            self.fa_output_mapping[layer_idx] = output # 若无需offload，则将输出直接存储
            return

        tensors = []
        for tensor_id, item in enumerate(output):
            if isinstance(item, torch.Tensor):
                tensor_buf = self.get_tensor_buf_for_offloaded_tensor(item, layer_idx, tensor_id)
                tensor_buf.copy_(item)
                tensors.append(tensor_buf)
            elif item is None:
                tensors.append(None)
                continue
            else:
                tensors.append(item)
        self.fa_output_mapping[layer_idx] = tensors # 若需offload，则将输出存储到buf中再给字典
        # if gpc.is_rank_for_log():
        #     breakpoint()
        # print(f"insert: {self.fa_output_mapping}")

    def get_fa_output_with_layer(self, layer_idx): # 取出输出
        assert layer_idx in self.fa_output_mapping
        return self.fa_output_mapping.pop(layer_idx) # 按层id取出输出

    def offload_fa_output_with_layer(self, layer_idx): # 将输出offload至CPU
        # if gpc.is_rank_for_log():
        #     breakpoint()
        # print(f"offload: {self.fa_output_mapping}")
        assert layer_idx in self.fa_output_mapping

        self.fa_stream.wait_stream(torch_npu.npu.current_stream())
        self.fa_stream.wait_event(self.d2h_final_event)

        with torch_npu.npu.stream(self.fa_stream):
            _gpu_tensors = self.fa_output_mapping.pop(layer_idx) # 获取GPU上输出，应该在缓存中
            _cpu_tensors = []
            for _tensor in _gpu_tensors:
                if isinstance(_tensor, torch.Tensor):
                    _cpu_backup = torch.empty(
                    _tensor.size(),
                    dtype=_tensor.dtype,
                    layout=_tensor.layout,
                    device="cpu",
                    pin_memory=True,
                )
                    _cpu_backup.copy_(_tensor, non_blocking=True)
                    _cpu_tensors.append(_cpu_backup)
                elif _tensor is None:
                    _cpu_tensors.append(_tensor)
                    continue
                else:
                    _cpu_tensors.append(_tensor)

                # _cpu_tensors.append(_tensor.to("cpu", non_blocking=False))

            self.fa_output_mapping[layer_idx] = _cpu_tensors # 用cuda流将输出从GPU（buf）放到CPU，字典记录Cpu——tensors
        # if gpc.is_rank_for_log():
        #     breakpoint()
        self.fa_stream.record_event(self.d2h_final_event)

    def preload_fa_output_with_layer(self, layer_idx):# 将输出重新载入gpu
        assert layer_idx in self.fa_output_mapping
        # breakpoint()
        self.fa_stream.wait_stream(torch_npu.npu.current_stream())
        self.fa_stream.wait_event(self.h2d_final_event)

        # Important: get device before with stream, in stream get device is error
        _device = get_current_device()
        print(f"device: {_device}")
        with torch_npu.npu.stream(self.fa_stream):
            _cpu_tensors = self.fa_output_mapping.pop(layer_idx)
            self.fa_output_mapping[layer_idx] = []
            for _tensor in _cpu_tensors:
                if isinstance(_tensor, torch.Tensor):
                    _gpu_backup = torch.empty(
                        _tensor.size(),
                        dtype=_tensor.dtype,
                        layout=_tensor.layout,
                        device=_device,
                        # pin_memory=True,
                    )
                    _gpu_backup.copy_(_tensor, non_blocking=True)
                    self.fa_output_mapping[layer_idx].append(_gpu_backup)
                    
                elif _tensor is None:
                    self.fa_output_mapping[layer_idx].append(_tensor)
                    continue
                else:
                    self.fa_output_mapping[layer_idx].append(_tensor)
        # print(f"preload:{self.fa_output_mapping[layer_idx]}")
        self.fa_stream.record_event(self.h2d_final_event) 
        # breakpoint()


def initialize_offload_manager(enable_cpu_offload: bool = False):
    global global_attn_offload
    if global_attn_offload is None:
        global_attn_offload = AttnOffloadManager(enable_cpu_offload)

    return global_attn_offload


def get_offload_manager():
    assert global_attn_offload is not None
    return global_attn_offload


def initialize_offload_npu_manager(enable_cpu_offload: bool = False):
    global global_attn_npu_offload
    if global_attn_npu_offload is None:
        global_attn_npu_offload = AttnOffloadManager(enable_cpu_offload)

    return global_attn_npu_offload


def get_offload_npu_manager():
    assert global_attn_npu_offload is not None
    return global_attn_npu_offload
