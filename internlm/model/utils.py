from typing import Any, Dict, List

from internlm.model.modules.mha import MHA


def internlm1_mha_pre_load_convert(
    model: MHA, state_dict: Dict, prefix: str, *args, **kwargs  # pylint: disable=W0613
) -> None:
    if f"{prefix}wqkv.weight" not in state_dict and f"{prefix}Wqkv.weight" in state_dict:
        state_dict[f"{prefix}wqkv.weight"] = state_dict.pop(f"{prefix}Wqkv.weight")

    if f"{prefix}wqkv.bias" not in state_dict and f"{prefix}Wqkv.bias" in state_dict:
        state_dict[f"{prefix}wqkv.bias"] = state_dict.pop(f"{prefix}Wqkv.bias")


def internlm1_mha_save_convert(
    model: MHA, state_dict: Dict, prefix: str, *args, **kwargs  # pylint: disable=W0613
) -> None:
    state_dict[f"{prefix}Wqkv.weight"] = state_dict.pop(f"{prefix}wqkv.weight")

    if f"{prefix}wqkv.bias" in state_dict:
        state_dict[f"{prefix}Wqkv.bias"] = state_dict.pop(f"{prefix}wqkv.bias")


def convert_attn_kwargs_to_args(kwargs) -> List[Any]:
    inference_params = kwargs.get("inference_params", None)
    cu_seqlens = kwargs.get("cu_seqlens", None)
    indexes = kwargs.get("indexes", None)
    max_seqlen = kwargs.get("max_seqlen", None)

    return (inference_params, cu_seqlens, indexes, max_seqlen)


def convert_attn_args_to_kwargs(args, kwargs) -> Dict[str, Any]:
    if len(args) == 0:
        return kwargs

    assert len(args) == 4, "args must be generate by convert_attn_kwargs_to_args function"

    if args[0] is not None:
        assert "inference_params" not in kwargs, "repeated 'inference_params' argument exists both in args and kwargs"
        kwargs["inference_params"] = args[0]
    if args[1] is not None:
        assert "cu_seqlens" not in kwargs, "repeated 'cu_seqlens' argument exists both in args and kwargs"
        kwargs["cu_seqlens"] = args[1]
    if args[2] is not None:
        assert "indexes" not in kwargs, "repeated 'indexes' argument exists both in args and kwargs"
        kwargs["indexes"] = args[2]
    if args[3] is not None:
        assert "max_seqlen" not in kwargs, "repeated 'max_seqlen' argument exists both in args and kwargs"
        kwargs["max_seqlen"] = args[3]

    return kwargs
