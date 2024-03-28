#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from typing import Union, List

from torch import nn

from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from internlm.core.parallel.shard import pipeline_parallel_sharding_wrapper


class Registry:
    """This is a registry class used to register classes and modules so that a universal
    object builder can be enabled.

    Args:
        name (str): The name of the registry.
    """

    def __init__(self, name: str):
        self._name = name
        self._registry = dict()

    @property
    def name(self):
        return self._name

    def register_module(self, module_name: str):
        """Registers a module represented in `module_class`.

        Args:
            module_name (str): The name of module to be registered.
        Returns:
            function: The module to be registered, so as to use it normally if via importing.
        Raises:
            AssertionError: Raises an AssertionError if the module has already been registered before.
        """

        assert module_name not in self._registry, f"{module_name} already registered in {self.name}"

        def decorator_wrapper(original_func):
            self._registry[module_name] = original_func
            return original_func

        return decorator_wrapper

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


# TODO: change to inner var
MODEL_INITIALIZER = Registry("model_initializer")


def create_model(model_type, *args, **kwargs) -> Union[nn.Module, List[nn.Module]]:
    num_chunks = kwargs.pop("num_chunks", 1)
    model_buidler = MODEL_INITIALIZER.get_module(module_name=model_type)

    if not gpc.is_using_parallel_mode(ParallelMode.PIPELINE):
        model_buidler(*args, **kwargs)

    num_layers = kwargs.pop("num_layers")
    return pipeline_parallel_sharding_wrapper(num_layers, num_chunks, model_buidler, *args, **kwargs)
