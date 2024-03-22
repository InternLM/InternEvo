from .abstract_accelerator import get_accelerator, AcceleratorType

get_accelerator()
from .abstract_accelerator import internlm_accelerator

__all__ = [
    "AcceleratorType",
    "internlm_accelerator",
    "get_accelerator",
]
