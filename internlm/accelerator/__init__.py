from .abstract_accelerator import get_accelerator

get_accelerator()
from .abstract_accelerator import internlm_accelerator

__all__ = [
    "internlm_accelerator",
    "get_accelerator",
]
