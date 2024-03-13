"""
Some hepler functions for ops package.
"""

from typing import Callable, Dict, Optional


class OpsBinding:
    def __init__(self, binding: Dict[str, Optional[Callable]]) -> None:
        self.__initialized = False
        self.__bindings = binding

        for key, value in binding.items():
            setattr(self, key, value)

    @property
    def is_initialized(self) -> bool:
        if self.__initialized:
            return True

        for key in self.__bindings.keys():
            if getattr(self, key) is None:
                return False

        self.__initialized = True
        return True
