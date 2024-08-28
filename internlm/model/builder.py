from typing import List, Union

from torch import nn

from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from internlm.core.parallel.shard import pipeline_parallel_sharding_wrapper
from internlm.model.base_model import BaseModel
from internlm.model.modules.embedding import Embedding1D
from internlm.model.modules.linear import ParallelLinearWithCommExt, new_linear
from internlm.model.registry import model_initializer
from internlm.utils.common import get_current_device
from internlm.utils.logger import get_logger

logger = get_logger(__file__)


LINEAR2NEWLINEAR_NAME_MAPPING = dict(
    q_proj="wq",
    k_proj="wk",
    v_proj="wv",
    o_proj="wo",
    gate_proj="w1",
    down_proj="w2",
    up_proj="w3",
    lm_head="head",
)


def create_model(model_type) -> Union[nn.Module, List[nn.Module]]:

    kwargs = dict(gpc.config.model)

    num_layers = kwargs.pop("num_layers")
    num_chunks = kwargs.pop("num_chunks", 1)

    # TODO: fix use_flash_attn parameter config
    kwargs.pop("use_flash_attn", False)
    kwargs.pop("apply_post_layer_norm")
    kwargs.pop("embed_split_hidden", True)

    kwargs["checkpoint"] = float(kwargs.get("checkpoint", False))
    kwargs["device"] = get_current_device()

    model_buidler = model_initializer.get_module(module_name=model_type)

    if not gpc.is_using_parallel_mode(ParallelMode.PIPELINE):
        kwargs["first"] = kwargs["last"] = True
        kwargs["start_layer_idx"] = 0
        kwargs["num_layers"] = num_layers
        model = model_buidler(**kwargs).to(kwargs["device"])
        setattr(model, "first_layer", 0)
        setattr(model, "last_layer", num_layers)
    else:
        model = pipeline_parallel_sharding_wrapper(num_layers, num_chunks, model_buidler, **kwargs)

    check_model(model, kwargs.get("hack_mode", None))

    return model


def timeout_input(printout, default, timeout=None, interactive=True):
    if not interactive:
        return default
    import select
    import sys

    if gpc.is_rank_for_log():
        print(printout)

    i, _, _ = select.select([sys.stdin], [], [], timeout)
    if i:
        msg = sys.stdin.readline().strip()
        return default if len(msg) == 0 else msg
    else:
        return default


def check_embed(model: nn.Module, hack=False, interactive=False) -> None:
    def traverse(module):
        for name, child in module.named_children():
            if isinstance(child, nn.Embedding) and not isinstance(child, Embedding1D):
                msg = (
                    f"To get parallel training enabled, module {name} of type {nn.Embedding.__name__} "
                    + f"is suggested to be replaced with {Embedding1D.__name__}."
                )
                if hack:
                    help_msg = f"Do you want to replace {name}? (y/n)"
                    opt = timeout_input(
                        f"{msg}\n{help_msg}",
                        default="y",
                        timeout=60,
                        interactive=interactive,
                    )
                    if opt in ["y", "yes"]:
                        child_new = Embedding1D(
                            num_embeddings=child.num_embeddings,
                            embedding_dim=child.embedding_dim,
                            padding_idx=child.padding_idx,
                        ).to(device=child.weight.device, dtype=child.weight.dtype)
                        setattr(module, name, child_new)
                    else:
                        if gpc.is_rank_for_log():
                            logger.warning(f"Skip replacing {name}")
                else:
                    if gpc.is_rank_for_log():
                        logger.warning(msg)
            else:
                traverse(child)

    traverse(model)


def check_linear(model: nn.Module, hack=False, interactive=False) -> None:
    def traverse(module):
        for name, child in module.named_children():
            if isinstance(child, nn.Linear) and not isinstance(child, ParallelLinearWithCommExt):
                msg = (
                    f"To get parallel training enabled, module {name} of type {nn.Linear.__name__} "
                    + f"is suggested to be replaced with {new_linear.__name__}"
                )
                if hack:
                    help_msg = f"Do you want to replace {name}? (y/n)"
                    opt = timeout_input(
                        f"{msg}\n{help_msg}",
                        default="y",
                        timeout=60,
                        interactive=interactive,
                    )
                    if opt in ["y", "yes"]:
                        child_new = new_linear(
                            name=LINEAR2NEWLINEAR_NAME_MAPPING.get(name, name),
                            in_features=child.in_features,
                            out_features=child.out_features,
                            bias=child.bias is not None,
                        ).to(device=child.weight.device, dtype=child.weight.dtype)
                        setattr(module, name, child_new)
                    else:
                        if gpc.is_rank_for_log():
                            logger.warning(f"Skip replacing {name}")
                else:
                    if gpc.is_rank_for_log():
                        logger.warning(msg)
            else:
                traverse(child)

    traverse(model)


def check_model(model: nn.Module, hack_mode) -> None:
    if not isinstance(model, BaseModel):
        logger.warning(
            f"To get load_hf_weights and convert_internevo2hf_weights enabled, "
            f"model is suggested to be inherited from {BaseModel.__name__}"
        )

    if hack_mode is not None:
        hack = True
        interactive = hack_mode == "interactive"
    else:
        hack = False
        interactive = False

    check_embed(model, hack, interactive)
    check_linear(model, hack, interactive)

    if hack and gpc.is_rank_for_log():
        logger.info(
            f"Hack mode is enabled, please check the model carefully, "
            f"if there are any problems, please report issue to us. "
            f"{model}"
        )
