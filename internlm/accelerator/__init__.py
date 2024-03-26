from .abstract_accelerator import AcceleratorType, get_accelerator

get_accelerator()
from .abstract_accelerator import (  # noqa: E402  #pylint: disable=wrong-import-position
    internlm_accelerator,
)

__all__ = [
    "AcceleratorType",
    "internlm_accelerator",
    "get_accelerator",
]
