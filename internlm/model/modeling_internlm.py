#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import math
import os
from typing import Optional

import torch
from torch import nn
from tqdm import tqdm

from internlm.accelerator import get_accelerator
from internlm.core.context import ParallelMode
from internlm.core.context.parallel_context import global_context as gpc
from internlm.core.naive_amp import set_output_attr_to_module
from internlm.core.parallel.shard import partition_uniform
from internlm.initialize.initialize_tensor import normal_, scaled_init_method_normal
from internlm.model.base_model import BaseModel
from internlm.model.modules.embedding import Embedding1D
from internlm.model.modules.linear import new_linear
from internlm.model.modules.mha import MHA
from internlm.model.modules.mlp import new_feed_forward
from internlm.model.modules.norm import new_layer_norm
from internlm.model.utils import (
    convert_attn_args_to_kwargs,
    convert_attn_kwargs_to_args,
    internlm1_mha_pre_load_convert,
    internlm1_mha_save_convert,
)
from internlm.solver.activation_checkpoint import activation_checkpoint
from internlm.utils.logger import get_logger
from internlm.utils.storage_manager import get_fns, llm_load, llm_save
from transformers.modeling_utils import (
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    shard_checkpoint,
)

internlm_accelerator = get_accelerator()
logger = get_logger(__file__)


class InternLM1Decoder(nn.Module):
    """
    1D Packed Flash Base Layer.

    Args:
        hidden_size (int): The hidden size of model. 768 by default.
        num_attention_heads (int): The number of attention heads. 12 by default.
        mlp_ratio (int): The ratio of MLP layers. 4 by default.
        attn_drop_rate (float): The dropout rate of attention module. 0 by default.
        drop_rate (float): The dropout rate of the input hidden state. 0.0 by default.
        max_position_embeddings (int): The maximum position embeddings. 2048 by default.
        dtype (torch.dtype): Type of data. torch.float by default.
        layer_norm_epsilon (float): A value added to the denominator for numerical stability. 1e-5 by default.
        checkpoint (bool): Whether to use checkpointing to save VRAM. True by default.
        layer_idx (int): The index of current layer. 0 by default.
        use_dynamic_ntk_rope (bool): Whether to use dynamic ntk rope. False by default.
        residual_in_fp32 (bool): Whether to use residual in fp32. False by default.
        device (Optional[Union[str, torch.device]]): The device will be used.
        norm_type (str): Use RMS norm or layernorm."rmsnorm" by default.
        qk_interleaved (bool): Whether the odd and even columns of the wq and wk are normally interleaved.
        dropout_selective_checkpoint (bool): Whether to selectively checkpoint dropout layers only.
        use_scaled_init (bool): Whether to use scaled initialization for weights.
        use_swiglu (bool): Whether to use SwiGLU activation in the mlp module.
        rope_base (int): The value of `base` for rotary position embeddings. 10000 by default.
        mlp_layer_fusion (bool): Whether to fuse layers in the mlp module for optimization.
        multiple_of (int): Ensures mlp dimensions are multiples of this value for efficient hardware utilization.
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        mlp_ratio: int = 4,
        attn_drop_rate: float = 0,
        drop_rate: float = 0.0,
        max_position_embeddings: int = 2048,
        dtype: torch.dtype = torch.float,
        layer_norm_epsilon: float = 1e-6,
        checkpoint: bool = False,
        layer_idx: int = 0,
        use_dynamic_ntk_rope: bool = False,
        residual_in_fp32: bool = False,
        device: Optional[torch.device] = None,
        norm_type: str = "rmsnorm",
        qk_interleaved: bool = False,
        dropout_selective_checkpoint: bool = True,
        use_scaled_init: bool = True,
        use_swiglu: bool = True,
        rope_base: int = 10000,
        mlp_layer_fusion: bool = False,
        multiple_of: int = 256,
    ):
        super().__init__()
        self.checkpoint = checkpoint
        # dropout selective checkpoint can only be enabled when checkpoint is disabled.
        self.dropout_selective_checkpoint = dropout_selective_checkpoint is True and checkpoint is False
        self.layer_idx = layer_idx

        head_dim = hidden_size // num_attention_heads

        self.mixer = MHA(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            dropout=attn_drop_rate,
            bias=True,
            max_position_embeddings=max_position_embeddings,
            softmax_scale=1 / math.sqrt(head_dim),
            causal=True,
            layer_idx=layer_idx,
            use_dynamic_ntk_rope=use_dynamic_ntk_rope,
            rotary_emb_dim=head_dim,
            rotary_emb_scale_base=0,
            rope_base=rope_base,
            device=device,
            dtype=dtype,
            qk_interleaved=qk_interleaved,
            enable_qkv_fusion=True,
        )

        # Compatible with the name of internlm1 Wqkv linear layer
        self.mixer.register_checkpoint_compatibility_hooks(internlm1_mha_pre_load_convert, internlm1_mha_save_convert)

        self.dropout1 = nn.Dropout(drop_rate)
        self.dropout2 = nn.Dropout(drop_rate)

        self.norm1 = new_layer_norm(norm_type, hidden_size, eps=layer_norm_epsilon)
        self.norm2 = new_layer_norm(norm_type, hidden_size, eps=layer_norm_epsilon)

        self.mlp = new_feed_forward(
            hidden_size,
            int(hidden_size * mlp_ratio),
            out_features=hidden_size,
            bias=False,
            device=device,
            dtype=dtype,
            mlp_layer_fusion=mlp_layer_fusion,
            multiple_of=multiple_of,
            # TODO: to support more activation functions
            activation_type="swiglu" if use_swiglu else "swiglu",
        )

        self.use_swiglu = use_swiglu
        self.use_scaled_init = use_scaled_init
        self.residual_in_fp32 = residual_in_fp32  # only make sense when using prenorm
        self.return_residual = False
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            for name, param in self.mixer.named_parameters():
                if param.ndim == 1:
                    param.data.zero_()
                elif "wqkv" in name:
                    normal_(std=0.006)(param.data)
                elif self.use_scaled_init:
                    scaled_init_method_normal(sigma=0.006, num_layers=self.layer_idx + 1)(param.data)
                else:
                    normal_(std=0.0015)(param.data)

            for name, param in self.mlp.named_parameters():
                if param.ndim == 1 and "bias" in name:
                    param.data.zero_()
                elif self.use_swiglu:
                    if self.use_scaled_init and "w2" in name:
                        scaled_init_method_normal(sigma=0.006, num_layers=self.layer_idx + 1)(param.data)
                    else:
                        # candidate: w1, w3, fused_w1_w3
                        normal_(std=0.006 if "w1" in name or "w3" in name else 0.0015)(param.data)
                else:
                    if self.use_scaled_init and "fc1" not in name:
                        scaled_init_method_normal(sigma=0.006, num_layers=self.layer_idx + 1)(param.data)
                    else:
                        normal_(std=0.006 if "fc1" in name else 0.0015)(param.data)

    def forward(self, hidden_states, **kwargs):
        if self.checkpoint and self.training:
            # NOTICE: activation_checkpiont do not support kwargs when use_reentrant = True.
            args = convert_attn_kwargs_to_args(kwargs)
            return activation_checkpoint(self._forward, False, hidden_states, *args)
        else:
            return self._forward(hidden_states, **kwargs)

    def _forward(self, hidden_states, *args, **kwargs):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Attn/MLP(LN(residual))
            cu_seqlens: 1d LongTensor, len(cu_seqlens) = hidden_states + 1
            indexes: the length of index is same as hidden states, which stand for the current position
        """

        def _dropout_and_norm_attn(_hidden_states):
            _dropped = self.dropout1(_hidden_states)
            _residual = _dropped
            _hidden_states = self.norm1(_residual.float())
            return _residual, _hidden_states

        if self.dropout_selective_checkpoint:
            residual, hidden_states = activation_checkpoint(_dropout_and_norm_attn, False, hidden_states)
        else:
            residual, hidden_states = _dropout_and_norm_attn(hidden_states)

        if self.residual_in_fp32:
            residual = residual.to(torch.float32)

        mixer_kwargs = convert_attn_args_to_kwargs(args, kwargs)
        hidden_states = self.mixer(hidden_states, **mixer_kwargs)

        def _dropout_and_norm_ffn(_residual, _hidden_states):
            _dropped = self.dropout2(_hidden_states)
            _residual = (_dropped + _residual) if _residual is not None else _dropped
            _hidden_states = self.norm2(_residual.float())
            return _residual, _hidden_states

        if self.dropout_selective_checkpoint:
            residual, hidden_states = activation_checkpoint(_dropout_and_norm_ffn, False, residual, hidden_states)
        else:
            residual, hidden_states = _dropout_and_norm_ffn(residual, hidden_states)

        if self.residual_in_fp32:
            residual = residual.to(torch.float32)

        hidden_states = self.mlp(hidden_states)

        return hidden_states + residual


class InternLM1(BaseModel):
    """
    1D Packed Flash InternLm.

    Args:
        num_layers (int): The number of layer. 12 by default.
        hidden_size (int): The size of hidden state. 768 by default.
        num_attention_heads (int): The number of attention head. 12 by default.
        vocab_size (int): The size of vocabulary. 50304 by default.
        mlp_ratio (int): The ratio of MLP layers. 4 by default.
        attn_drop_rate (float): The dropout rate of attention module. 0.0 by default.
        drop_rate (float): The dropout rate of input hidden state. 0.0 by default.
        max_position_embeddings (int): The maximum position embeddings. 2048 by default.
        dtype (torch.dtype): The type of data. torch.float by default.
        checkpoint (float): The proportion of layers that need to be checkpointed compared to the total number
                                    of layers. 0.0 by default.
        layer_norm_epsilon (float): A value added to the denominator for numerical stability. 1e-5 by default.
        first (bool): Whether input embedding layer or not. False by default.
        last (bool): Whether output embedding layer or not. False by default.
        embed_grad_scale (float): Refer to GLM-130B, for training stability. 0.1 by default.
        parallel_output (bool): If it is necessary to collect the output of parallel computing. True by default.
        start_layer_idx (int): The index of start layer in the pipeline. 0 by default.
        use_dynamic_ntk_rope (bool): Whether to use dynamic ntk rope. False by default.
        device (Optional[Union[str, torch.device]]): The device will be used. None by default.
        residual_in_fp32 (bool): Whether to use residual in fp32. False by default.
        norm_type (str): Normalization type. Use RMSNorm or LayerNorm. "rmsnorm" by default.
        qk_interleaved (bool): Whether the odd and even columns of the wq and wk are normally interleaved.
        dropout_selective_checkpoint (bool): Whether to selectively checkpoint dropout and norm layers.
        use_scaled_init (bool): Whether to use scaled initialization for weights.
        use_swiglu (bool): Whether to use SwiGLU activation in the mlp module.
        rope_base (int): The value of `base` for rotary position embeddings. 10000 by default.
        mlp_layer_fusion (bool): Whether to fuse layers in the mlp module for optimization.
        multiple_of (int): Ensures mlp dimensions are multiples of this value for efficient hardware utilization.
    """

    def __init__(
        self,
        num_layers: int = 12,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        vocab_size: int = 50304,
        mlp_ratio: int = 4.0,
        attn_drop_rate: float = 0.0,
        drop_rate: float = 0.0,
        max_position_embeddings: int = 2048,
        dtype: torch.dtype = torch.float,
        checkpoint: float = 0.0,
        layer_norm_epsilon: float = 1e-5,
        first: bool = False,
        last: bool = False,
        embed_grad_scale: float = 0.1,
        parallel_output: bool = True,
        start_layer_idx: int = 0,
        use_dynamic_ntk_rope: bool = False,
        device: Optional[torch.device] = None,
        residual_in_fp32: bool = False,
        norm_type: str = "rmsnorm",
        qk_interleaved: bool = False,
        is_reward: bool = False,
        dropout_selective_checkpoint: bool = True,
        use_scaled_init: bool = True,
        use_swiglu: bool = True,
        rope_base: int = 10000,
        mlp_layer_fusion: bool = False,
        multiple_of: int = 256,
    ):
        super().__init__()

        checkpoint_layer_num = int(num_layers * checkpoint)
        self.embed_grad_scale = embed_grad_scale
        self.parallel_output = parallel_output

        if first:
            self.embedding = Embedding1D(num_embeddings=vocab_size, embedding_dim=hidden_size)
            for _, param in self.embedding.named_parameters():
                normal_(std=0.0052)(param)

        self.blocks = nn.ModuleList(
            [
                InternLM1Decoder(
                    hidden_size=hidden_size,
                    num_attention_heads=num_attention_heads,
                    mlp_ratio=mlp_ratio,
                    attn_drop_rate=attn_drop_rate,
                    drop_rate=drop_rate,
                    max_position_embeddings=max_position_embeddings,
                    dtype=dtype,
                    layer_norm_epsilon=layer_norm_epsilon,
                    checkpoint=lid < checkpoint_layer_num,
                    layer_idx=lid + start_layer_idx,  # This parameter is used for caching during generation
                    use_dynamic_ntk_rope=use_dynamic_ntk_rope,
                    residual_in_fp32=residual_in_fp32,
                    device=device,
                    norm_type=norm_type,
                    dropout_selective_checkpoint=dropout_selective_checkpoint,
                    use_scaled_init=use_scaled_init,
                    use_swiglu=use_swiglu,
                    rope_base=rope_base,
                    qk_interleaved=qk_interleaved,
                    mlp_layer_fusion=mlp_layer_fusion,
                    multiple_of=multiple_of,
                )
                for lid in range(num_layers)
            ]
        )

        if last:
            self.norm = new_layer_norm(norm_type, hidden_size, eps=layer_norm_epsilon)
            self.head = new_linear(
                name="head",
                in_features=hidden_size,
                out_features=gpc.get_world_size(ParallelMode.TENSOR) if is_reward else vocab_size,
                bias=False,
                device=device,
                dtype=dtype,
                is_reward=is_reward,
                weight_scale=embed_grad_scale,
            )
            set_output_attr_to_module(self.head)
            for _, param in self.head.named_parameters():
                normal_(std=0.0052)(param)

    def forward(self, hidden_states=None, input_ids=None, **kwargs):
        # attention_mask: compute attention on the places where the value is 1
        if hasattr(self, "embedding") and input_ids is not None:
            hidden_states = self.embedding(input_ids)
            if self.embed_grad_scale != 1:
                hidden_states = (
                    self.embed_grad_scale * hidden_states + (1 - self.embed_grad_scale) * hidden_states.detach()
                )

        for _, block in enumerate(self.blocks):
            hidden_states = block(hidden_states, **kwargs)

        if hasattr(self, "norm"):
            hidden_states = self.norm(hidden_states.float())
        if hasattr(self, "head"):
            hidden_states = self.head(hidden_states)

        return hidden_states

    @staticmethod
    def load_hf_weights(folder: str, model: nn.Module) -> None:
        """NOTE: when loading huggingface's llama pretrained weights, you should set `adapt_hf=True` in your config."""
        assert folder is not None, "Please specify the folder of the pretrained model"
        if gpc.is_rank_for_log():
            logger.info(f"Loading pretrained model from {folder}")

        fns = get_fns(folder)
        model_fns = [
            os.path.join(folder, fn)
            for fn in fns
            if (fn.endswith(".bin") and fn.startswith("pytorch_model"))
            or (fn.endswith(".safetensors") and fn.startswith("model"))
        ]
        model_fns.sort()

        state_dict = {}
        for model_fn in model_fns:
            state_dict.update(llm_load(model_fn, map_location="cpu"))

        tp_size = gpc.get_world_size(ParallelMode.TENSOR)
        tp_rank = gpc.get_local_rank(ParallelMode.TENSOR)
        wp_size = gpc.get_world_size(ParallelMode.WEIGHT)
        wp_rank = gpc.get_local_rank(ParallelMode.WEIGHT)
        tp_mode = gpc.config.parallel.tensor["mode"]
        split_size = wp_size if tp_mode == "isp" else tp_size
        local_rank = wp_rank if tp_mode == "isp" else tp_rank
        row_dim = 0 if tp_mode == "isp" else 1
        if gpc.config.model.get("embed_split_hidden", True):
            embed_concat_dim = 1
        else:
            embed_concat_dim = 0

        new_state_dict = {}

        for idx, i in enumerate(range(model.first_layer, model.last_layer)):
            layer_ids = i

            # attn
            q_proj_weight = torch.chunk(
                state_dict.pop(f"model.layers.{layer_ids}.self_attn.q_proj.weight"),
                split_size,
                dim=0,
            )[local_rank]
            q_proj_bias = torch.chunk(
                state_dict.pop(f"model.layers.{layer_ids}.self_attn.q_proj.bias"),
                split_size,
            )[local_rank]
            k_proj_weight = torch.chunk(
                state_dict.pop(f"model.layers.{layer_ids}.self_attn.k_proj.weight"),
                split_size,
                dim=0,
            )[local_rank]
            k_proj_bias = torch.chunk(
                state_dict.pop(f"model.layers.{layer_ids}.self_attn.k_proj.bias"),
                split_size,
            )[local_rank]
            v_proj_weight = torch.chunk(
                state_dict.pop(f"model.layers.{layer_ids}.self_attn.v_proj.weight"),
                split_size,
                dim=0,
            )[local_rank]
            v_proj_bias = torch.chunk(
                state_dict.pop(f"model.layers.{layer_ids}.self_attn.v_proj.bias"),
                split_size,
            )[local_rank]
            state_dict[f"blocks.{i}.mixer.wqkv.weight"] = torch.cat(
                [q_proj_weight, k_proj_weight, v_proj_weight], dim=0
            )
            state_dict[f"blocks.{i}.mixer.wqkv.bias"] = torch.cat([q_proj_bias, k_proj_bias, v_proj_bias], dim=0)
            state_dict[f"blocks.{i}.mixer.out_proj.weight"] = torch.chunk(
                state_dict.pop(f"model.layers.{layer_ids}.self_attn.o_proj.weight"),
                split_size,
                dim=row_dim,
            )[local_rank]
            state_dict[f"blocks.{i}.mixer.out_proj.bias"] = torch.chunk(
                state_dict.pop(f"model.layers.{layer_ids}.self_attn.o_proj.bias"),
                split_size,
            )[local_rank]

            # mlp
            state_dict[f"blocks.{i}.mlp.w1.weight"] = torch.chunk(
                state_dict.pop(f"model.layers.{layer_ids}.mlp.gate_proj.weight"),
                split_size,
                dim=0,
            )[local_rank]
            # Be cautious that in InternLM1, down_proj is equivalent to w3
            state_dict[f"blocks.{i}.mlp.w3.weight"] = torch.chunk(
                state_dict.pop(f"model.layers.{layer_ids}.mlp.down_proj.weight"),
                split_size,
                dim=row_dim,
            )[local_rank]
            # Be cautious that in InternLM1, up_proj is equivalent to w2
            state_dict[f"blocks.{i}.mlp.w2.weight"] = torch.chunk(
                state_dict.pop(f"model.layers.{layer_ids}.mlp.up_proj.weight"),
                split_size,
                dim=0,
            )[local_rank]

            # attn norm
            state_dict[f"blocks.{i}.norm1.weight"] = state_dict.pop(f"model.layers.{layer_ids}.input_layernorm.weight")
            # mlp norm
            state_dict[f"blocks.{i}.norm2.weight"] = state_dict.pop(
                f"model.layers.{layer_ids}.post_attention_layernorm.weight"
            )

            # skip rotary_emb inv_freq
            if f"model.layers.{layer_ids}.self_attn.rotary_emb.inv_freq" in state_dict:
                state_dict.pop(f"model.layers.{layer_ids}.self_attn.rotary_emb.inv_freq")

            # replace value within decoder layer
            for name in list(state_dict.keys()):
                if name.startswith(f"blocks.{i}"):
                    new_state_dict[name.replace(f".{i}.", f".{idx}.")] = state_dict.pop(name)

        # embedding
        if (gpc.get_local_rank(ParallelMode.PIPELINE) == 0) or (not gpc.is_using_parallel_mode(ParallelMode.PIPELINE)):
            new_state_dict["embedding.weight"] = torch.chunk(
                state_dict.pop("model.embed_tokens.weight"),
                split_size,
                dim=embed_concat_dim,
            )[local_rank]

        # head
        if gpc.is_last_rank(ParallelMode.PIPELINE):
            new_state_dict["head.weight"] = torch.chunk(
                state_dict.pop("lm_head.weight"),
                split_size,
                dim=0,
            )[local_rank]
            new_state_dict["norm.weight"] = state_dict.pop("model.norm.weight")

        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
        if len(state_dict) > 0:
            logger.warning(f"Be cautious, checkpoint state_dict keys={state_dict.keys()} have not beed loaded.")

        if gpc.get_local_rank(ParallelMode.DATA) == 0:
            pp_rank = 0 if not gpc.is_initialized(ParallelMode.PIPELINE) else gpc.get_local_rank(ParallelMode.PIPELINE)
            logger.info(
                f"Missing keys:{missing_keys}, unexpected keys:{unexpected_keys} in "
                f"tp:{gpc.get_local_rank(ParallelMode.TENSOR)}, pp:{pp_rank}"
            )

        internlm_accelerator.empty_cache()

    @staticmethod
    def load_internlm_with_dynamic_parallel_size(folder: str, model: nn.Module):

        assert folder is not None, "Please specify the folder of the pretrained model"
        if gpc.is_rank_for_log():
            logger.info(f"Loading pretrained model from {folder}")

        fns = get_fns(folder)
        model_fns = []
        for fn in fns:
            # filter with `_t` is for avoiding conflict with model_config.py
            if fn.startswith("model_t") and not fn.endswith("md5"):
                model_fns.append(fn)

        old_tp, old_pp = -1, -1
        for fn in model_fns:
            _, tp, pp = os.path.splitext(fn)[0].split("_")
            old_tp = max(old_tp, int(tp[2:]) + 1)
            old_pp = max(old_pp, int(pp[2:]) + 1)

        assert old_tp > 0 and old_pp > 0, f"ckpt with tp:{old_tp} and pp:{old_pp} is illegal"

        tp = gpc.get_world_size(ParallelMode.TENSOR)
        tp_rank = gpc.get_local_rank(ParallelMode.TENSOR)
        assert old_tp % tp == 0 or tp % old_tp == 0, (
            f"Expected TP size in loaded checkpoint to be fit with TP size in current config, but got {old_tp} in "
            f"checkpoint and {tp} in current config"
        )

        correspond_tps = []

        if old_tp <= tp:
            correspond_tps.append(tp_rank // (tp // old_tp))
            ratio = tp // old_tp
            rank = tp_rank % ratio
        else:
            for i in range(old_tp // tp):
                correspond_tps.append(tp_rank * (old_tp // tp) + i)
            rank = 0
            ratio = 1

        current_states = {}

        pp = gpc.get_world_size(ParallelMode.PIPELINE)

        assert gpc.config.model.num_chunks == 1, "May cause future collisions, ignore this if necessary"

        old_pp_partition = partition_uniform(gpc.config.model.num_layers, old_pp, 1)

        for idx, parts in enumerate(old_pp_partition):
            start, end = parts[0]
            if model.last_layer <= start or model.first_layer >= end:
                continue

            tmp_states = {}

            for correspond_tp in correspond_tps:
                model_name = f"model_tp{correspond_tp}_pp{idx}.pt"
                states = llm_load(os.path.join(folder, model_name), map_location="cpu")
                for i in range(start, end):
                    if i >= model.last_layer:
                        break
                    if i < model.first_layer:
                        continue
                    for name in list(states.keys()):
                        if f".{i-start}." in name:
                            to_name = name.replace(f".{i-start}.", f".{i-model.first_layer}.")
                            if "norm" in name:
                                tmp_states[to_name] = [states.pop(name)]
                            elif any(x in name for x in ("out_proj", "w2")):
                                if "bias" not in name:
                                    tmp_states[to_name] = tmp_states.get(to_name, [])
                                    tmp_states[to_name].append(states.pop(name).chunk(ratio, dim=-1)[rank])
                                else:
                                    tmp_states[to_name] = [states.pop(name)]
                            elif any(x in name for x in ("w1", "w3")):
                                tmp_states[to_name] = tmp_states.get(to_name, [])
                                tmp_states[to_name].append(states.pop(name).chunk(ratio, dim=0)[rank])
                            elif any(x in name for x in ("Wqkv",)):
                                tmp_states[to_name] = tmp_states.get(to_name, [])
                                _wqkv = states.pop(name).chunk(3, dim=0)
                                _wq_splits = _wqkv[0].chunk(ratio, dim=0)
                                _wk_splits = _wqkv[1].chunk(ratio, dim=0)
                                _wv_splits = _wqkv[2].chunk(ratio, dim=0)
                                new_wqkv = torch.concat([_wq_splits[rank], _wk_splits[rank], _wv_splits[rank]], dim=0)
                                tmp_states[to_name].append(new_wqkv)
                            else:
                                raise KeyError(f"Unknown key {name}.")

                if "embedding.weight" in states and model.first_layer == 0:
                    tmp_states["embedding.weight"] = tmp_states.get("embedding.weight", [])
                    tmp_states["embedding.weight"].append(states["embedding.weight"].chunk(ratio, dim=1)[rank])
                if "head.weight" in states and model.last_layer == gpc.config.model.num_layers:
                    tmp_states["norm.weight"] = [states["norm.weight"]]
                    tmp_states["head.weight"] = tmp_states.get("head.weight", [])
                    tmp_states["head.weight"].append(states["head.weight"].chunk(ratio, dim=0)[rank])

                states = {}

            for name in list(tmp_states.keys()):
                data = tmp_states.pop(name)
                if len(data) == 1:
                    current_states[name] = data[0]
                else:
                    current_states[name] = torch.concat(
                        data, dim=1 if name == "embedding.weight" or any(x in name for x in ("out_proj", "w2")) else 0
                    )

        missing_keys, unexpected_keys = model.load_state_dict(current_states, strict=False)

        if gpc.get_local_rank(ParallelMode.DATA) == 0:
            pp_rank = 0 if not gpc.is_initialized(ParallelMode.PIPELINE) else gpc.get_local_rank(ParallelMode.PIPELINE)
            logger.info(
                f"Missing keys:{missing_keys}, unexpected keys:{unexpected_keys} in "
                f"tp:{gpc.get_local_rank(ParallelMode.TENSOR)}, pp:{pp_rank}"
            )

        internlm_accelerator.empty_cache()

    @staticmethod
    def convert_internevo2hf_weights(src: str, tgt: str) -> None:
        model_config = gpc.config.model
        n_heads = model_config["num_attention_heads"]
        h_dim = model_config["hidden_size"]
        tp_mode = gpc.config.parallel.tensor["mode"]
        row_dim = 0 if tp_mode == "isp" else 1

        # load states
        states, num_shards = InternLM1.load_sharded_states(src)

        # convert state_dict
        state_dict = {}
        embedding_key_list = ["embedding.word_embeddings.weight", "embedding.weight", "tok_embeddings.weight", None]
        for layer_i in tqdm(range(model_config["num_layers"])):
            # attn norm, mlp norm
            state_dict.update(
                {
                    f"model.layers.{layer_i}.input_layernorm.weight": states[0][
                        f"blocks.{layer_i}.norm1.weight"
                    ].clone(),
                    f"model.layers.{layer_i}.post_attention_layernorm.weight": states[0][
                        f"blocks.{layer_i}.norm2.weight"
                    ].clone(),
                }
            )
            # attn wqkv weight
            wqkvs = [
                states[tp].pop(f"blocks.{layer_i}.mixer.Wqkv.weight").reshape(3, n_heads // num_shards, -1, h_dim)
                for tp in range(num_shards)
            ]
            # attn wqkv bias
            bqkvs = [
                states[tp].pop(f"blocks.{layer_i}.mixer.Wqkv.bias").reshape(3, n_heads // num_shards, -1)
                for tp in range(num_shards)
            ]
            state_dict[f"model.layers.{layer_i}.self_attn.q_proj.weight"] = torch.cat(
                [wqkvs[i][0] for i in range(num_shards)],
                dim=0,
            ).reshape(h_dim, h_dim)
            state_dict[f"model.layers.{layer_i}.self_attn.q_proj.bias"] = torch.cat(
                [bqkvs[i][0] for i in range(num_shards)],
                dim=0,
            ).reshape(-1)
            state_dict[f"model.layers.{layer_i}.self_attn.k_proj.weight"] = torch.cat(
                [wqkvs[i][1] for i in range(num_shards)],
                dim=0,
            ).reshape(h_dim, h_dim)
            state_dict[f"model.layers.{layer_i}.self_attn.k_proj.bias"] = torch.cat(
                [bqkvs[i][1] for i in range(num_shards)],
                dim=0,
            ).reshape(-1)
            state_dict[f"model.layers.{layer_i}.self_attn.v_proj.weight"] = torch.cat(
                [wqkvs[i][2] for i in range(num_shards)],
                dim=0,
            ).reshape(h_dim, h_dim)
            state_dict[f"model.layers.{layer_i}.self_attn.v_proj.bias"] = torch.cat(
                [bqkvs[i][2] for i in range(num_shards)],
                dim=0,
            ).reshape(-1)
            # attn wo weight
            state_dict[f"model.layers.{layer_i}.self_attn.o_proj.weight"] = torch.cat(
                [states[i][f"blocks.{layer_i}.mixer.out_proj.weight"] for i in range(num_shards)], dim=row_dim
            )
            # attn wo bias
            state_dict[f"model.layers.{layer_i}.self_attn.o_proj.bias"] = torch.cat(
                [states[i][f"blocks.{layer_i}.mixer.out_proj.bias"] for i in range(num_shards)],
                dim=0,
            ).reshape(-1)
            # mlp
            state_dict[f"model.layers.{layer_i}.mlp.gate_proj.weight"] = torch.cat(
                [states[i][f"blocks.{layer_i}.mlp.w1.weight"] for i in range(num_shards)], dim=0
            )
            # Be cautious that in InternLM1, down_proj is equivalent to w3
            state_dict[f"model.layers.{layer_i}.mlp.down_proj.weight"] = torch.cat(
                [states[i][f"blocks.{layer_i}.mlp.w3.weight"] for i in range(num_shards)], dim=row_dim
            )
            # Be cautious that in InternLM1, up_proj is equivalent to w2
            state_dict[f"model.layers.{layer_i}.mlp.up_proj.weight"] = torch.cat(
                [states[i][f"blocks.{layer_i}.mlp.w2.weight"] for i in range(num_shards)], dim=0
            )
        # embedding, head
        for embedding_key in embedding_key_list:
            if embedding_key in states[0]:
                break
        if embedding_key is None:
            raise KeyError("Cannot find embedding key!")
        if model_config["embed_split_hidden"]:
            embed_concat_dim = 1
            tok_emb_list = [states[i][embedding_key] for i in range(num_shards)]
        else:
            embed_concat_dim = 0
            _, size_1 = states[0][embedding_key].shape
            embdim_pertp = size_1 // num_shards
            tok_emb_list = [
                torch.concat(
                    [
                        states[tp][embedding_key][:, embdim_pertp * local_rank : embdim_pertp * (local_rank + 1)]
                        for tp in range(num_shards)
                    ],
                    dim=0,
                )
                for local_rank in range(num_shards)
            ]
        state_dict.update(
            {
                "model.norm.weight": states[0]["norm.weight"],
                "model.embed_tokens.weight": torch.cat(tok_emb_list, dim=embed_concat_dim),
                "lm_head.weight": torch.cat([states[i]["head.weight"] for i in range(num_shards)], dim=0),
            },
        )

        # save state_dict to hf format
        shards, index = shard_checkpoint(state_dict, weights_name=SAFE_WEIGHTS_NAME)
        for shard_file, shard in shards.items():
            llm_save(save_path=os.path.join(tgt, shard_file), saved_obj=shard, metadata={"format": "pt"})
        if index is not None:
            # Save the index as well
            llm_save(save_path=os.path.join(tgt, SAFE_WEIGHTS_INDEX_NAME), saved_obj=index)
