"""
Universal accelerator interface implementation, inspired by DeepSpeed.
"""
import enum
import os


class AcceleratorType(enum.Enum):
    GPU = 1
    NPU = 2
    CPU = 3
    OTHER = 4


internlm_accelerator = None


class Accelerator:
    """
    Abstract base class for accelerator
    """

    def __init__(self) -> None:
        pass

    def get_backend_name(self):
        """
        Return the name of the accelerator.
        """
        raise NotImplementedError

    def get_accelerator_backend(self):
        """
        Return the name of the backend.
        """
        raise NotImplementedError

    # Device APIs
    def device_name(self, device_index=None):
        """
        Return the name of the device.
        """
        raise NotImplementedError

    def set_device(self, device_index):
        """
        Bind the current process to a device.
        """
        raise NotImplementedError

    def get_device_id(self):
        """
        Return the current device index.
        """
        raise NotImplementedError

    def current_device_name(self):
        """
        Return the name of the current device.
        """
        raise NotImplementedError

    def device_count(self):
        """
        Return the number of devices on the machine.
        """
        raise NotImplementedError

    def synchronize(self, device_index=None):
        """
        Synchronize the current process.
        """
        raise NotImplementedError


def get_accelerator():
    global internlm_accelerator
    if internlm_accelerator is not None:
        return internlm_accelerator

    accelerator_name = None
    # 1. Detect whether there is override of DeepSpeed accelerators from environment variable.
    intern_accelerator_LIST = ["cuda", "npu"]
    if "INTERNLM_ACCELERATOR" in os.environ:
        accelerator_name = os.environ["INTERNLM_ACCELERATOR"]
        if accelerator_name == "npu":
            try:
                import torch_npu  # noqa # pylint: disable=W0611
            except (ImportError, ModuleNotFoundError):
                raise ValueError("NPU_Accelerator requires torch_npu, which is not installed on this system.")
            pass
        elif accelerator_name != "cuda":
            raise ValueError(
                f"accelerator_name must be one of {intern_accelerator_LIST}."
                + " Value '{accelerator_name}' is not supported"
            )

    # 2. If no override, detect which accelerator to use automatically
    if accelerator_name is None:
        try:
            import torch_npu  # noqa: F401,F811 # type: ignore

            accelerator_name = "npu"
        except (ImportError, ModuleNotFoundError):
            pass
    if accelerator_name is None:
        accelerator_name = "cuda"

    # 3. Set internlm_accelerator accordingly
    if accelerator_name == "cuda":
        from .cuda_accelerator import CUDA_Accelerator

        internlm_accelerator = CUDA_Accelerator()
    elif accelerator_name == "npu":
        from .npu_accelerator import ASCEND_Accelerator

        internlm_accelerator = ASCEND_Accelerator()

    return internlm_accelerator
