import torch

# from internlm.simulator.elements.tensor import FakeTensor


class FakeAllocator:
    def __init__(self, capcity=0) -> None:
        self.init_capcity = capcity
        self.capcity = capcity

    def alloc(self, size):
        if self.capcity - size >= 0:
            self.capcity -= size
        else:
            raise RuntimeError(f"Out of Memory request: {size}, left: {self.capcity}")

    def free(self, size):
        self.capcity += size
        assert self.capcity <= self.init_capcity


global_allocator = FakeAllocator()


def get_global_allocator() -> FakeAllocator:
    return global_allocator


class TensorTracker:
    def __init__(self) -> None:
        self.tensor_map = {}

    def save_tensor(self, tensor: torch.Tensor):
        tid = id(tensor)
        assert tid not in self.tensor_map
        self.tensor_map[tid] = tensor

    def del_tensor(self, tid):
        self.tensor_map.pop(tid).free_self()


global_tensor_manager = TensorTracker()


def get_global_tensor_manager():
    return global_tensor_manager
