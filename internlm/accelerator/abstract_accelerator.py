"""
Universal accelerator interface implementation, inspired by DeepSpeed.
"""
import abc
import enum
import os
from abc import ABC


class AcceleratorType(enum.Enum):
    GPU = 1
    NPU = 2
    CPU = 3
    DIPU = 4
    DITORCH = 5
    OTHER = 6


internlm_accelerator = None


class Accelerator(ABC):
    """
    Abstract base class for accelerator
    """

    def __init__(self) -> None:
        self._name_str = None
        self._communication_backend_name = None

    @abc.abstractmethod
    def get_backend_name(self):
        """
        Return the name of the accelerator.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_accelerator_backend(self):
        """
        Return the name of the accelerator backend.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def communication_backend_name(self):
        """
        Return the name of the communication backend.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def device_name(self, device_index=None):
        """
        Return the name of the device.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def set_device(self, device_index):
        """
        Bind the current process to a device.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_device_id(self):
        """
        Return the current device index.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def current_device_name(self):
        """
        Return the name of the current device.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def device_count(self):
        """
        Return the number of devices on the machine.
        """
        raise NotImplementedError

    @abc.abstractmethod
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
    # 2. ditorch: a unified accelerator tools for torch_npu, torch_dipu backend etc.
    #    deeplink_ext: implemented FlashSelfAttention, FlashCrossAttention, RmsNorm and RotaryEmbedding operations
    #           etc. based on torch_dipu and torch_npu backend, respectively.
    #    ditorch, together with deeplink_ext, provides unified APIs for internlm training.
    #    usage:
    #       for torch_dipu backend: export INTERNLM_ACCELERATOR=ditorch; export DEEPLINK_EXT_PLATFORM_TYPE=torch_dipu
    #       for torch_npu backend: export INTERNLM_ACCELERATOR=ditorch; export DEEPLINK_EXT_PLATFORM_TYPE=torch_npu

    intern_accelerator_LIST = ["cuda", "npu", "dipu", "ditorch"]
    if "INTERNLM_ACCELERATOR" in os.environ:
        accelerator_name = os.environ["INTERNLM_ACCELERATOR"]
        if accelerator_name == "npu":
            try:
                import torch_npu  # noqa # pylint: disable=W0611
            except (ImportError, ModuleNotFoundError):
                raise ValueError("NPU_Accelerator requires torch_npu, which is not installed on this system.")
            pass
        elif accelerator_name == "dipu":
            try:
                import deeplink_ext  # noqa # pylint: disable=W0611
                import torch_dipu  # noqa # pylint: disable=W0611
            except (ImportError, ModuleNotFoundError):
                raise ValueError(
                    "DIPU_Accelerator requires torch_dipu and deeplink_ext, which is not installed on this system."
                )
            pass
        elif accelerator_name == "ditorch":
            try:
                import deeplink_ext  # pylint: disable=unused-import
                import ditorch  # pylint: disable=unused-import
            except (ImportError, ModuleNotFoundError):
                raise ValueError(
                    "DIPU_Accelerator requires ditorch and deeplink_ext, which is not installed on this system."
                )
            pass
        elif accelerator_name != "cuda":
            raise ValueError(
                f"accelerator_name must be one of {intern_accelerator_LIST}."
                + " Value '{accelerator_name}' is not supported"
            )

    # 2. If no override, detect which accelerator to use automatically
    if accelerator_name is None:
        try:
            import deeplink_ext  # noqa: F401,F811 # type: ignore
            import ditorch  # noqa: F401,F811 # type: ignore

            accelerator_name = "ditorch"
        except (ImportError, ModuleNotFoundError):
            pass
    if accelerator_name is None:
        try:
            import deeplink_ext  # noqa: F401,F811 # type: ignore
            import torch_dipu  # noqa: F401,F811 # type: ignore

            accelerator_name = "dipu"
        except (ImportError, ModuleNotFoundError):
            pass
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
    elif accelerator_name == "dipu":
        from .dipu_accelerator import DIPU_Accelerator

        internlm_accelerator = DIPU_Accelerator()
    elif accelerator_name == "ditorch":
        from .ditorch_accelerator import DITORCH_Accelerator

        internlm_accelerator = DITORCH_Accelerator()

    return internlm_accelerator
