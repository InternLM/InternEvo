import torch
import torch.nn as nn

from internlm.utils.logger import get_logger
logger = get_logger(__file__)


def _is_sm89_or_later():
    # Float8 is only supported on SM89 or later (H100+ GPUs)
    return torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 9)


class Float8Handler:
    def __init__(self):
        self.enabled = False

        if not _is_sm89_or_later():
            logger.warning(
                "Failed to swap to Float8Linear because float8 is only supported on SM89 or later",
            )
            return
        try:
            from torchao.float8 import CastConfig, Float8LinearConfig, ScalingType
        except ImportError as e:
            raise ImportError(
                "torchao is not installed. Please install it to use float8 linear layers."
            ) from e

        self.config = Float8LinearConfig(
            cast_config_input=CastConfig(scaling_type=ScalingType.DYNAMIC),
            cast_config_weight=CastConfig(scaling_type=ScalingType.DYNAMIC),
            cast_config_grad_output=CastConfig(scaling_type=ScalingType.DYNAMIC),
        )

        self.enabled = True

        logger.info("Float8 training active")

    def convert_to_float8_training(self, model: nn.Module):
        """
        This function converts the linear layers of `model` to `Float8Linear`.
        Note that today, only dynamic tensor scaling is supported.
        This will mutate the model inplace.
        """
        if not self.enabled:
            return

        from torchao.float8 import convert_to_float8_training

        # Mutates the model inplace replacing instances of nn.Linear with Float8Linear
        convert_to_float8_training(
            model,
            config=self.config,
        )