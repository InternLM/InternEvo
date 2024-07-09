#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from typing import Callable

from internlm.model.modeling_internlm import InternLM1
from internlm.model.modeling_internlm2 import InternLM2
from internlm.model.modeling_llama import Llama2
from internlm.model.modeling_llava import Llava
from internlm.model.modeling_moe import Internlm1MoE


class Registry:
    """This is a registry class used to register classes and modules so that a universal
    object builder can be enabled.

    Args:
        name (str): The name of the registry.
    """

    def __init__(self, name: str):
        self._name = name
        self._registry = {}

    @property
    def name(self):
        return self._name

    def register_module(self, module_name: str, func: Callable):
        """Registers a module represented in `module_class`.

        Args:
            module_name (str): The name of module to be registered.
        Returns:
            function: The module to be registered, so as to use it normally if via importing.
        Raises:
            AssertionError: Raises an AssertionError if the module has already been registered before.
        """

        assert module_name not in self._registry, f"{module_name} already registered in {self.name}"

        self._registry[module_name] = func

    def get_module(self, module_name: str):
        """Retrieves a module with name `module_name` and returns the module if it has
        already been registered before.

        Args:
            module_name (str): The name of the module to be retrieved.
        Returns:
            :class:`object`: The retrieved module or None.
        Raises:
            NameError: Raises a NameError if the module to be retrieved has neither been
            registered directly nor as third party modules before.
        """
        if module_name in self._registry:
            return self._registry[module_name]
        raise NameError(f"Module {module_name} not found in the registry {self.name}")

    def has(self, module_name: str):
        """Searches for a module with name `module_name` and returns a boolean value indicating
        whether the module has been registered directly or as third party modules before.

        Args:
            module_name (str): The name of the module to be searched for.
        Returns:
            bool: A boolean value indicating whether the module has been registered directly or
            as third party modules before.
        """
        found_flag = module_name in self._registry

        return found_flag


model_initializer = Registry("model_initializer")
hf_config_initializer = Registry("hf_config_initializer")


def register_model_initializer() -> None:
    model_initializer.register_module("INTERNLM", InternLM1)
    model_initializer.register_module("INTERNLM2_PUBLIC", InternLM2)
    model_initializer.register_module("LLAMA2", Llama2)
    model_initializer.register_module("INTERNLM_MoE", Internlm1MoE)
    model_initializer.register_module("LLAVA", Llava)
