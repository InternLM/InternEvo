# adapted from https://github.com/pytorch/torchtitan/blob/main/torchtitan/float8.py
from typing import List, Union

import torch
import torch.nn as nn

from internlm.utils.logger import get_logger

logger = get_logger(__file__)


def _is_sm89_or_later():
    # Float8 is only supported on SM89 or later (H100+ GPUs)
    return torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 9)


class Float8Handler:
    def __init__(self, float8_config):
        self.enabled = False

        if not float8_config.enable_float8_linear:
            return

        if not _is_sm89_or_later():
            logger.warning(
                "Failed to swap to Float8Linear because float8 is only supported on SM89 or later",
            )
            return
        try:
            from torchao.float8 import CastConfig, Float8LinearConfig, ScalingType
        except ImportError as e:
            raise ImportError("torchao is not installed. Please install it to use float8 linear layers.") from e

        scaling_type_input = float8_config.scaling_type_input
        scaling_type_weight = float8_config.scaling_type_weight
        scaling_type_grad_output = float8_config.scaling_type_grad_output

        self.config = Float8LinearConfig(
            cast_config_input=CastConfig(scaling_type=ScalingType(scaling_type_input)),
            cast_config_weight=CastConfig(scaling_type=ScalingType(scaling_type_weight)),
            cast_config_grad_output=CastConfig(scaling_type=ScalingType(scaling_type_grad_output)),
        )

        self.enabled = True

        # for sync_float8_amax_and_scale_history
        self.delayed_scaling = (
            scaling_type_input == "delayed" or scaling_type_weight == "delayed" or scaling_type_grad_output == "delayed"
        )
        self._sync_float8_amax_and_scale_history = None

        self.compile = float8_config.compile

        logger.info("Float8 training active")

    def convert_to_float8_training(self, model: nn.Module):
        if not self.enabled:
            return

        from torchao.float8 import convert_to_float8_training

        convert_to_float8_training(
            model,
            config=self.config,
            module_filter_fn=lambda mod, fqn: fqn != "head",
        )

    def sync_float8_amax_and_scale_history(self, model: Union[nn.Module, List[nn.Module]]):
        if not self.enabled:
            return

        if not self.delayed_scaling:
            return

        from torchao.float8 import sync_float8_amax_and_scale_history

        if self._sync_float8_amax_and_scale_history is None:
            if self.compile:
                self._sync_float8_amax_and_scale_history = torch.compile(sync_float8_amax_and_scale_history)
            else:
                self._sync_float8_amax_and_scale_history = sync_float8_amax_and_scale_history

        models = [model] if isinstance(model, nn.Module) else model
        for m in models:
            self._sync_float8_amax_and_scale_history(m)
