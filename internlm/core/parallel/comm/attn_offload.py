import torch

from internlm.utils.common import get_current_device

global_attn_offload = None


class AttnOffloadManager:
    """
    A manager for attention output CPU offloading and GPU prefetch loading.
    """

    def __init__(self, enable_cpu_offload: bool = False) -> None:
        # cpu offload overlapping
        self.cpu_offload = enable_cpu_offload
        # layer id mapping to flash attn output
        self.fa_output_mapping = {}
        self.fa_stream = torch.cuda.Stream()
        self.d2h_final_event = torch.cuda.Event()
        self.h2d_final_event = torch.cuda.Event()
        # prepare for tensor buffer
        self.tensor_id_to_tensor_bufs = {}

    def get_tensor_buf_for_offloaded_tensor(self, tensor, layer_id, tensor_id):
        """Get tensor buffer for offloaded tensor."""
        layer_id = layer_id % 2
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

            self.tensor_id_to_tensor_bufs[layer_id][tensor_id] = buffer

        return self.tensor_id_to_tensor_bufs[layer_id][tensor_id]

    def insert_fa_output_with_layer(self, layer_idx, output):
        assert layer_idx not in self.fa_output_mapping
        if self.cpu_offload is False:
            self.fa_output_mapping[layer_idx] = output
            return

        tensors = []
        for tensor_id, tensor in enumerate(output):
            if tensor is None:
                tensors.append(None)
                continue
            tensor_buf = self.get_tensor_buf_for_offloaded_tensor(tensor, layer_idx, tensor_id)
            tensor_buf.copy_(tensor)
            tensors.append(tensor_buf)
        self.fa_output_mapping[layer_idx] = tensors

    def get_fa_output_with_layer(self, layer_idx):
        assert layer_idx in self.fa_output_mapping
        return self.fa_output_mapping.pop(layer_idx)

    def offload_fa_output_with_layer(self, layer_idx):
        assert layer_idx in self.fa_output_mapping

        self.fa_stream.wait_stream(torch.cuda.current_stream())
        self.fa_stream.wait_event(self.d2h_final_event)

        with torch.cuda.stream(self.fa_stream):
            _gpu_tensors = self.fa_output_mapping.pop(layer_idx)
            _cpu_tensors = []
            for _tensor in _gpu_tensors:
                if _tensor is None:
                    _cpu_tensors.append(_tensor)
                    continue

                _cpu_backup = torch.empty(
                    _tensor.size(),
                    dtype=_tensor.dtype,
                    layout=_tensor.layout,
                    device="cpu",
                    pin_memory=True,
                )
                _cpu_backup.copy_(_tensor, non_blocking=True)
                _cpu_tensors.append(_cpu_backup)

                # _cpu_tensors.append(_tensor.to("cpu", non_blocking=False))

            self.fa_output_mapping[layer_idx] = _cpu_tensors

        self.fa_stream.record_event(self.d2h_final_event)

    def preload_fa_output_with_layer(self, layer_idx):
        assert layer_idx in self.fa_output_mapping

        self.fa_stream.wait_stream(torch.cuda.current_stream())
        self.fa_stream.wait_event(self.h2d_final_event)

        # Important: get device before with stream, in stream get device is error
        _device = get_current_device()
        with torch.cuda.stream(self.fa_stream):
            _cpu_tensors = self.fa_output_mapping.pop(layer_idx)
            self.fa_output_mapping[layer_idx] = [
                _tensor.to(device=_device, non_blocking=True) if _tensor is not None else _tensor
                for _tensor in _cpu_tensors
            ]

        self.fa_stream.record_event(self.h2d_final_event)


def initialize_offload_manager(enable_cpu_offload: bool = False):
    global global_attn_offload
    if global_attn_offload is None:
        global_attn_offload = AttnOffloadManager(enable_cpu_offload)

    return global_attn_offload


def get_offload_manager():
    assert global_attn_offload is not None
    return global_attn_offload
