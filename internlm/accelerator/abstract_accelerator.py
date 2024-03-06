"""
Universal accelerator interface implementation, inspired by DeepSpeed.
"""
import os
import enum

class AcceleratorType(enum.Enum):
    GPU = 1
    NPU = 2
    CPU = 3
    OTHER = 4


internlm_accelerator = None


class Accelerator:
    def __init__(self) -> None:
        pass

    def get_accelerator_name(self):
        raise NotImplementedError

    # Device APIs
    def device_name(self, device_index=None):
        raise NotImplementedError

    def device(self, device_index=None):
        raise NotImplementedError

    def set_device(self, device_index):
        raise NotImplementedError

    def current_device(self):
        raise NotImplementedError

    def current_device_name(self):
        raise NotImplementedError

    def device_count(self):
        raise NotImplementedError

    def synchronize(self, device_index=None):
        raise NotImplementedError


def get_accelerator():
    global internlm_accelerator
    if internlm_accelerator is not None:
        return internlm_accelerator

    accelerator_name = None
    # 1. Detect whether there is override of DeepSpeed accelerators from environment variable.
    intern_accelerator_LIST = ["cuda", "npu"]
    if "INTERNLM_ACCELERATOR" in os.environ.keys():
        accelerator_name = os.environ["INTERNLM_ACCELERATOR"]
        if accelerator_name == "npu":
            try:
                import torch_npu  # noqa: F401 # type: ignore
            except ImportError as e:
                raise ValueError(f"NPU_Accelerator requires torch_npu, which is not installed on this system.")
            pass
        else:
            raise ValueError(
                f'internlm_accelerator must be one of {intern_accelerator_LIST}.  Value "{accelerator_name}" is not supported'
            )

    # 2. If no override, detect which accelerator to use automatically
    if accelerator_name == None:
        if accelerator_name == None:
            try:
                import torch_npu  # noqa: F401,F811 # type: ignore

                accelerator_name = "npu"
            except ImportError as e:
                pass
        if accelerator_name == None:
            accelerator_name = "cuda"

    # 3. Set internlm_accelerator accordingly
    if accelerator_name == "cuda":
        from .cuda_accelerator import CUDA_Accelerator

        internlm_accelerator = CUDA_Accelerator()
    elif accelerator_name == "npu":
        from .npu_accelerator import ASCEND_Accelerator

        internlm_accelerator = ASCEND_Accelerator()

    return internlm_accelerator
