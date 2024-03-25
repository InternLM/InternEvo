from .abstract_accelerator import AcceleratorType, get_accelerator

get_accelerator()  # noqa: E402  #pylint: disable=wrong-import-position
from .abstract_accelerator import internlm_accelerator

__all__ = [
    "AcceleratorType",
    "internlm_accelerator",
    "get_accelerator",
]
