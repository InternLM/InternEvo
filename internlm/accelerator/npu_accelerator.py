from .abstract_accelerator import Accelerator, AcceleratorType

try:
    import torch.npu
except ImportError:
    pass


class ASCEND_Accelerator(Accelerator):
    """Accelerator for NPU device.

    Args:
        Accelerator (Accelerator): Repalce torch.npu
    """

    def __init__(self) -> None:
        self._name_str = "npu"
        self._communication_backend_name = "hccl"
        self.amp = self.get_amp()

    def backend_name(self):
        return self._name_str

    def get_accelerator_backend(self):
        return AcceleratorType.NPU

    # Device APIs
    def device_name(self, device_index=None):
        if device_index is None:
            return "npu"
        return "npu:{}".format(device_index)

    def device(self, device_index=None):
        return torch.npu.device(device_index)

    def set_device(self, device_index):
        torch.npu.set_device(device_index)

    def current_device(self):
        return torch.npu.current_device()

    def current_device_name(self):
        return "npu:{}".format(torch.npu.current_device())

    def device_count(self):
        return torch.npu.device_count()

    def synchronize(self, device_index=None):
        return torch.npu.synchronize(device_index)

    # RNG APIs
    def random(self):
        return torch.random

    def set_rng_state(self, new_state, device_index=None):
        if device_index is None:
            return torch.npu.set_rng_state(new_state)

        return torch.npu.set_rng_state(new_state, device_index)

    def get_rng_state(self, device_index=None):
        if device_index is None:
            return torch.npu.get_rng_state()

        return torch.npu.get_rng_state(device_index)

    def manual_seed(self, seed):
        return torch.npu.manual_seed(seed)

    def manual_seed_all(self, seed):
        return torch.npu.manual_seed_all(seed)

    def initial_seed(self):
        return torch.npu.initial_seed()

    def default_generator(self, device_index):
        return torch.npu.default_generators[device_index]

    # Streams/Events
    @property
    def Stream(self):
        return torch.npu.Stream

    def stream(self, stream):
        return torch.npu.stream(stream)

    def current_stream(self, device_index=None):
        return torch.npu.current_stream(device_index)

    def default_stream(self, device_index=None):
        return torch.npu.default_stream(device_index)

    @property
    def Event(self):
        return torch.npu.Event

    # Memory management
    def empty_cache(self):
        return torch.npu.empty_cache()

    def memory_allocated(self, device_index=None):
        return torch.npu.memory_allocated(device_index)

    def max_memory_allocated(self, device_index=None):
        return torch.npu.max_memory_allocated(device_index)

    def reset_max_memory_allocated(self, device_index=None):
        return torch.npu.reset_max_memory_allocated(device_index)

    def memory_cached(self, device_index=None):
        return torch.npu.memory_cached(device_index)

    def max_memory_cached(self, device_index=None):
        return torch.npu.max_memory_cached(device_index)

    def reset_max_memory_cached(self, device_index=None):
        return torch.npu.reset_max_memory_cached(device_index)

    def memory_stats(self, device_index=None):
        if hasattr(torch.npu, "memory_stats"):
            return torch.npu.memory_stats(device_index)

    def reset_peak_memory_stats(self, device_index=None):
        if hasattr(torch.npu, "reset_peak_memory_stats"):
            return torch.npu.reset_peak_memory_stats(device_index)

    def memory_reserved(self, device_index=None):
        if hasattr(torch.npu, "memory_reserved"):
            return torch.npu.memory_reserved(device_index)

    def max_memory_reserved(self, device_index=None):
        if hasattr(torch.npu, "max_memory_reserved"):
            return torch.npu.max_memory_reserved(device_index)

    def total_memory(self, device_index=None):
        return torch.npu.get_device_properties(device_index).total_memory

    # Data types
    def is_bf16_supported(self):
        return torch.npu.is_bf16_supported()

    def is_fp16_supported(self):
        major, _ = torch.npu.get_device_capability()
        return bool(major >= 7)

    # Misc
    def get_amp(self):
        if hasattr(torch.npu, "amp"):
            return torch.npu.amp
        return None

    def is_available(self):
        return torch.npu.is_available()

    def range_push(self, msg):
        if hasattr(torch.npu.nvtx, "range_push"):
            return torch.npu.nvtx.range_push(msg)

    def range_pop(self):
        if hasattr(torch.npu.nvtx, "range_pop"):
            return torch.npu.nvtx.range_pop()

    def lazy_call(self, callback):
        return torch.npu._lazy_call(callback)

    def communication_backend_name(self):
        return self._communication_backend_name

    # Tensor operations

    @property
    def BFloat16Tensor(self):
        return torch.npu.BFloat16Tensor

    @property
    def ByteTensor(self):
        return torch.npu.ByteTensor

    @property
    def DoubleTensor(self):
        return torch.npu.DoubleTensor

    @property
    def FloatTensor(self):
        return torch.npu.FloatTensor

    @property
    def HalfTensor(self):
        return torch.npu.HalfTensor

    @property
    def IntTensor(self):
        return torch.npu.IntTensor

    @property
    def LongTensor(self):
        return torch.npu.LongTensor

    def pin_memory(self, tensor):
        return tensor.pin_memory()

    def on_accelerator(self, tensor):
        device_str = str(tensor.device)
        return bool(device_str.startswith("npu:"))

    def set_allow_tf32(self, enable: bool):
        print(f"Not support tf32 for NPU, {enable}!")

    def return_custom_bwd(self):
        return torch.npu.amp.custom_bwd

    def return_custom_fwd(self):
        return torch.npu.amp.custom_fwd
