# Copyright (c) InternLM. All rights reserved.

from internevo.model.modeling_internlm import InternLM1
from internevo.model.modeling_internlm2 import InternLM2
from internevo.model.modeling_llama import Llama2
from internevo.utils.logger import get_logger

logger = get_logger(__file__)

LOAD_FUNC_DICT = {
    "llama": Llama2.load_llama_pretrained_weights,
    "internlm_test": InternLM1.load_internlm_with_dynamic_parallel_size,
    "internlm2_test": InternLM2.load_internlm2_with_dynamic_parallel_size,
}
