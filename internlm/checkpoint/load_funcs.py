# Copyright (c) InternLM. All rights reserved.

from internlm.model.model_implementations.transformers.modeling_internlm2 import (
    InternLM2,
)
from internlm.model.model_implementations.transformers.modeling_llama import Llama2
from internlm.utils.logger import get_logger

logger = get_logger(__file__)

LOAD_FUNC_DICT = {
    "llama": Llama2.load_llama_pretrained_weights,
    "internlm2_test": InternLM2.load_internlm2_with_dynamic_parallel_size,
}
