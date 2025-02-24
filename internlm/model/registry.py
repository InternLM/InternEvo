#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) InternLM. All rights reserved.

from typing import Callable

from internlm.model.modeling_baichuan2 import Baichuan2
from internlm.model.modeling_gemma import Gemma
from internlm.model.modeling_internlm import InternLM1
from internlm.model.modeling_internlm2 import InternLM2
from internlm.model.modeling_llama import Llama2
from internlm.model.modeling_llava import Llava
from internlm.model.modeling_mixtral import MixtralMoE
from internlm.model.modeling_moe import Internlm1MoE
from internlm.model.modeling_qwen2 import Qwen2
from internlm.model.modeling_qwen2_moe import Qwen2Moe
from internlm.utils.common import SingletonMeta
from internlm.utils.utils import ModelType


class Registry(metaclass=SingletonMeta):
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
        """

        if self.has(module_name):
            return
        else:
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
        if self.has(module_name):
            return self._registry[module_name]
        else:
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


def register_model_initializer() -> None:
    model_initializer.register_module(ModelType.INTERNLM.name, InternLM1)
    model_initializer.register_module(ModelType.INTERNLM2.name, InternLM2)
    model_initializer.register_module(ModelType.INTERNLM3.name, InternLM2)
    model_initializer.register_module(ModelType.LLAMA2.name, Llama2)
    model_initializer.register_module(ModelType.INTERNLM_MoE.name, Internlm1MoE)
    model_initializer.register_module(ModelType.LLAVA.name, Llava)
    model_initializer.register_module(ModelType.QWEN2.name, Qwen2)
    model_initializer.register_module(ModelType.BAICHUAN2.name, Baichuan2)
    model_initializer.register_module(ModelType.GEMMA.name, Gemma)
    model_initializer.register_module(ModelType.QWEN2MOE.name, Qwen2Moe)
    model_initializer.register_module(ModelType.MIXTRALMOE.name, MixtralMoE)


register_model_initializer()
