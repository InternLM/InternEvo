# Copyright (c) InternLM. All rights reserved.
import math
import os
from functools import reduce
from typing import Optional

import torch
from einops import rearrange
from torch import nn
from tqdm import tqdm

from internlm.accelerator import get_accelerator
from internlm.core.context import ParallelMode
from internlm.core.context.parallel_context import global_context as gpc
from internlm.core.parallel.shard import partition_uniform
from internlm.initialize.initialize_tensor import (
    normal_,
    scaled_init_method_normal,
    scaled_init_method_uniform,
    uniform_,
)
from internlm.model.base_model import BaseModel
from internlm.model.modules.embedding import Embedding1D
from internlm.model.modules.linear import new_linear
from internlm.model.modules.mha import GQA
from internlm.model.modules.mlp import new_feed_forward
from internlm.model.modules.norm import new_layer_norm
from internlm.model.utils import (
    convert_attn_args_to_kwargs,
    convert_attn_kwargs_to_args,
    get_parallel_size_from_file,
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


class InternLM2Decoder(nn.Module):
    """
    InternLM2 Decoder layer.

    Args:
        hidden_size (int): The hidden size of model. 768 by default.
        num_attention_heads (int): The number of attention heads. 12 by default.
        num_kv_attention_heads (int): The number of key/value attention heads. Defaults to 12.
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
        apply_post_layer_norm (bool): Whether to apply layer normalization after the attention and mlp.
                                        Defaults to False.
        fused_dropout_add_ln (bool): Whether to fuse dropout, residual addition, and layer normalization.
                                        Defaults to True.
        no_bias (bool): Whether to exclude bias in attention and feed-forward networks. Defaults to False.
        norm_type (str): Use RMS norm or layernorm."rmsnorm" by default.
        qk_interleaved (bool): Whether the odd and even columns of the wq and wk are normally interleaved.
        dropout_selective_checkpoint (bool): Whether to selectively checkpoint dropout layers only.
        use_scaled_init (bool): Whether to use scaled initialization for weights.
        use_swiglu (bool): Whether to use SwiGLU activation in the mlp module.
        attn_wqkv_init_std (float): std used to init attn_wqkv weight. 0.02 by default,
        attn_other_init_std (float): std used to init attn_other weight. 0.02 by default,
        ffn_uplayer_init_std (float): std used to init w1, w2 weight in ffn when using glu
            otherwise init fc1 weight in ffn. 0.02 by default,
        ffn_other_init_std (float): std used to init ffn_other weight. 0.02 by default,
        init_type (str): Initialization type. Use uniform or normal. "normal" by default,
        rope_base (int): The value of `base` for rotary position embeddings. 10000 by default.
        mlp_layer_fusion (bool): Whether to fuse layers in the mlp module for optimization.
        multiple_of (int): Ensures mlp dimensions are multiples of this value for efficient hardware utilization.
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        num_kv_attention_heads: int = 12,
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
        apply_post_layer_norm: bool = False,
        fused_dropout_add_ln: bool = True,
        no_bias: bool = False,
        norm_type: str = "rmsnorm",
        qk_interleaved: bool = False,
        dropout_selective_checkpoint: bool = True,
        use_scaled_init: bool = True,
        use_swiglu: bool = True,
        attn_wqkv_init_std: float = 0.02,
        attn_other_init_std: float = 0.02,
        ffn_uplayer_init_std: float = 0.02,
        ffn_other_init_std: float = 0.02,
        init_type: str = "normal",
        rope_base: int = 10000,
        mlp_layer_fusion: bool = False,
        multiple_of: int = 256,
    ):
        super().__init__()
        self.checkpoint = checkpoint
        # dropout selective checkpoint can only be enabled when checkpoint is disabled.
        self.dropout_selective_checkpoint = dropout_selective_checkpoint is True and checkpoint is False
        self.layer_idx = layer_idx
        self.prenorm = not apply_post_layer_norm
        assert not fused_dropout_add_ln, "dropout_add_layer_norm can not be used here"
        self.fused_dropout_add_ln = fused_dropout_add_ln
        self.attn_wqkv_init_std = attn_wqkv_init_std
        self.attn_other_init_std = attn_other_init_std
        self.ffn_uplayer_init_std = ffn_uplayer_init_std
        self.ffn_other_init_std = ffn_other_init_std

        self.max_position_embeddings = max_position_embeddings
        self.use_dynamic_ntk_rope = use_dynamic_ntk_rope

        head_dim = hidden_size // num_attention_heads
        self.attention = GQA(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            num_kv_heads=num_kv_attention_heads,
            dropout=attn_drop_rate,
            max_position_embeddings=max_position_embeddings,
            softmax_scale=1 / math.sqrt(head_dim),
            causal=True,
            layer_idx=layer_idx,
            use_dynamic_ntk_rope=use_dynamic_ntk_rope,
            rotary_emb_dim=head_dim,
            rotary_emb_scale_base=0,
            device=device,
            dtype=dtype,
            qk_interleaved=qk_interleaved,
            bias=not no_bias,
            rope_base=rope_base,
            enable_qkv_fusion=True,
        )

        self.dropout1 = nn.Dropout(drop_rate)
        self.dropout2 = nn.Dropout(drop_rate)
        self.attention_norm = new_layer_norm(norm_type, hidden_size, eps=layer_norm_epsilon)
        self.ffn_norm = new_layer_norm(norm_type, hidden_size, eps=layer_norm_epsilon)

        self.feed_forward = new_feed_forward(
            hidden_size,
            int(hidden_size * mlp_ratio),
            out_features=hidden_size,
            bias=False,
            device=device,
            dtype=dtype,
            mlp_layer_fusion=mlp_layer_fusion,
            multiple_of=multiple_of,
            # TODO: to support more activation functions
            activation_type="swiglu" if use_swiglu else "gelu",
        )

        self.use_swiglu = use_swiglu
        self.use_scaled_init = use_scaled_init
        self.residual_in_fp32 = residual_in_fp32  # only make sense when using prenorm
        self.return_residual = False

        if init_type == "normal":
            self.init_func = normal_
            self.scaled_init_func = scaled_init_method_normal
        else:
            self.init_func = uniform_
            self.scaled_init_func = scaled_init_method_uniform

        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            for name, param in self.attention.named_parameters():
                if param.ndim == 1:
                    param.data.zero_()
                elif "wq" in name or "wk" in name or "wv" in name:
                    self.init_func(std=self.attn_wqkv_init_std)(param.data)
                elif self.use_scaled_init:  # wo
                    self.scaled_init_func(sigma=self.attn_other_init_std, num_layers=self.layer_idx + 1)(param.data)
                else:
                    self.init_func(std=self.attn_other_init_std)(param.data)

            for name, param in self.feed_forward.named_parameters():
                if self.use_swiglu:
                    if self.use_scaled_init and "w2" in name:
                        self.scaled_init_func(sigma=self.ffn_other_init_std, num_layers=self.layer_idx + 1)(param.data)
                    else:
                        # candidate: w1, w3, fused_w1_w3
                        self.init_func(
                            std=self.ffn_uplayer_init_std if "w1" in name or "w3" in name else self.ffn_other_init_std
                        )(param.data)
                else:
                    if self.use_scaled_init and "fc1" not in name:
                        self.scaled_init_func(sigma=self.ffn_other_init_std, num_layers=self.layer_idx + 1)(param.data)
                    else:
                        self.init_func(std=self.ffn_uplayer_init_std if "fc1" in name else self.ffn_other_init_std)(
                            param.data
                        )

    def forward(self, hidden_states, residual=None, **kwargs):
        if self.checkpoint and self.training:
            # NOTICE: activation_checkpiont do not support kwargs when use_reentrant = True.
            args = convert_attn_kwargs_to_args(kwargs)
            return activation_checkpoint(self._forward, False, hidden_states, residual, *args)
        else:
            return self._forward(hidden_states, residual, **kwargs)

    def _forward(self, hidden_states, residual, *args, **kwargs):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Attn/MLP(LN(residual))
            cu_seqlens: 1d LongTensor, len(cu_seqlens) = hidden_states + 1
            indexes: the length of index is same as hidden states, which stand for the current position
        """
        if self.prenorm:

            def _dropout_and_norm_attn(_residual, _hidden_states):
                _dropped = self.dropout1(_hidden_states)
                _residual = (_dropped + _residual) if _residual is not None else _dropped
                _hidden_states = self.attention_norm(_residual.to(dtype=self.attention_norm.weight.dtype))

                return _residual, _hidden_states

            if self.dropout_selective_checkpoint:
                residual, hidden_states = activation_checkpoint(_dropout_and_norm_attn, False, residual, hidden_states)
            else:
                residual, hidden_states = _dropout_and_norm_attn(residual, hidden_states)

            if self.residual_in_fp32:
                residual = residual.to(torch.float32)

            attn_kwargs = convert_attn_args_to_kwargs(args, kwargs)
            hidden_states = self.attention(hidden_states, **attn_kwargs)

            if not isinstance(self.feed_forward, nn.Identity):
                if not self.fused_dropout_add_ln:

                    def _dropout_and_norm_ffn(_residual, _hidden_states):
                        _dropped = self.dropout2(_hidden_states)
                        _residual = (_dropped + _residual) if _residual is not None else _dropped
                        _hidden_states = self.ffn_norm(_residual.to(torch.float32))

                        return _residual, _hidden_states

                    if self.dropout_selective_checkpoint:
                        residual, hidden_states = activation_checkpoint(
                            _dropout_and_norm_ffn, False, residual, hidden_states
                        )
                    else:
                        residual, hidden_states = _dropout_and_norm_ffn(residual, hidden_states)

                    if self.residual_in_fp32:
                        residual = residual.to(torch.float32)
                hidden_states = self.feed_forward(hidden_states)

            return hidden_states + residual
        else:
            assert residual is None

            mixer_out = self.attention(hidden_states, **kwargs)
            if self.return_residual:  # mixer out is actually a pair here
                mixer_out, hidden_states = mixer_out
            hidden_states = self.attention_norm(self.dropout1(mixer_out) + hidden_states).to(
                dtype=self.attention_norm.weight.dtype
            )
            if not isinstance(self.feed_forward, nn.Identity):
                mlp_out = self.feed_forward(hidden_states)
                if self.return_residual:  # mlp out is actually a pair here
                    mlp_out, hidden_states = mlp_out
                hidden_states = self.ffn_norm((self.dropout2(mlp_out)) + hidden_states).to(
                    dtype=self.ffn_norm.weight.dtype
                )
            return hidden_states


class InternLM2(BaseModel):
    """
    InternLM2 Model.

    Args:
        num_layers (int): The number of layer. 48 by default.
        hidden_size (int): The size of hidden state. 2048 by default.
        num_attention_heads (int): The number of attention head. 32 by default.
        num_kv_attention_heads (int): The number of key/value attention heads. Defaults to 32.
        vocab_size (int): The size of vocabulary. 50304 by default.
        mlp_ratio (int): The ratio of MLP layers. 4 by default.
        attn_drop_rate (float): The dropout rate of attention module. 0.0 by default.
        drop_rate (float): The dropout rate of input hidden state. 0.0 by default.
        max_position_embeddings (int): The maximum position embeddings. 2048 by default.
        dtype (torch.dtype): The type of data. torch.float by default.
        checkpoint (float): The proportion of layers that need to be checkpointed compared to the total number
                            of layers. 0.0 by default.
        layer_norm_epsilon (float): A value added to the denominator for numerical stability. 1e-6 by default.
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
        embedding_init_std (float): std used to init embedding weight. 0.02 by default,
        attn_wqkv_init_std (float): std used to init attn_wqkv weight. 0.02 by default,
        attn_other_init_std (float): std used to init attn_other weight. 0.02 by default,
        ffn_uplayer_init_std (float): std used to init w1, w2 weight in ffn when using glu
            otherwise init fc1 weight in ffn. 0.02 by default,
        ffn_other_init_std (float): std used to init ffn_other weight. 0.02 by default,
        out_head_init_std (float): std used to init output lmhead weight. 0.02 by default,
        init_type (str): Initialization type. Use uniform or normal. "normal" by default,
        rope_base (int): The value of `base` for rotary position embeddings. 10000 by default.
        norm_head (bool): Whether to use norm head. False by default.
        mlp_layer_fusion (bool): Whether to fuse layers in the mlp module for optimization.
        multiple_of (int): Ensures mlp dimensions are multiples of this value for efficient hardware utilization.
    """

    def __init__(
        self,
        num_layers: int = 48,
        hidden_size: int = 2048,
        num_attention_heads: int = 32,
        num_kv_attention_heads: int = 32,
        vocab_size: int = 50304,
        mlp_ratio: float = 4.0,
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
        apply_post_layer_norm=False,
        no_bias=False,
        residual_in_fp32: bool = False,
        norm_type: str = "rmsnorm",
        qk_interleaved: bool = False,
        is_reward: bool = False,
        dropout_selective_checkpoint: bool = True,
        use_scaled_init: bool = True,
        use_swiglu: bool = True,
        embedding_init_std: float = 0.02,
        attn_wqkv_init_std: float = 0.02,
        attn_other_init_std: float = 0.02,
        ffn_uplayer_init_std: float = 0.02,
        ffn_other_init_std: float = 0.02,
        out_head_init_std: float = 0.02,
        init_type: str = "normal",
        rope_base: int = 10000,
        norm_head: bool = False,
        mlp_layer_fusion: bool = False,
        multiple_of: int = 256,
    ):
        super().__init__()

        checkpoint_layer_num = int(num_layers * checkpoint)
        self.embed_grad_scale = embed_grad_scale
        self.parallel_output = parallel_output

        if first:
            self.tok_embeddings = Embedding1D(num_embeddings=vocab_size, embedding_dim=hidden_size)

            for _, param in self.tok_embeddings.named_parameters():
                if init_type == "normal":
                    normal_(std=embedding_init_std)(param)
                else:
                    uniform_(std=embedding_init_std)(param)

        self.layers = nn.ModuleList(
            [
                InternLM2Decoder(
                    hidden_size=hidden_size,
                    num_attention_heads=num_attention_heads,
                    num_kv_attention_heads=num_kv_attention_heads,
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
                    apply_post_layer_norm=apply_post_layer_norm,
                    fused_dropout_add_ln=False,
                    no_bias=no_bias,
                    norm_type=norm_type,
                    dropout_selective_checkpoint=dropout_selective_checkpoint,
                    use_scaled_init=use_scaled_init,
                    use_swiglu=use_swiglu,
                    qk_interleaved=qk_interleaved,
                    attn_wqkv_init_std=attn_wqkv_init_std,
                    attn_other_init_std=attn_other_init_std,
                    ffn_uplayer_init_std=ffn_uplayer_init_std,
                    ffn_other_init_std=ffn_other_init_std,
                    init_type=init_type,
                    rope_base=rope_base,
                    mlp_layer_fusion=mlp_layer_fusion,
                    multiple_of=multiple_of,
                )
                for lid in range(num_layers)
            ]
        )

        if last:
            if not apply_post_layer_norm:
                self.norm = new_layer_norm(norm_type, hidden_size, eps=layer_norm_epsilon)

            self.output = new_linear(
                name="output",
                in_features=hidden_size,
                out_features=gpc.get_world_size(ParallelMode.TENSOR) if is_reward else vocab_size,
                bias=False,
                device=device,
                dtype=dtype,
                is_reward=is_reward,
                weight_scale=embed_grad_scale,
                norm_head=norm_head,
            )
            for _, param in self.output.named_parameters():
                if init_type == "normal":
                    normal_(std=out_head_init_std)(param)
                else:
                    uniform_(std=out_head_init_std)(param)

    def forward(self, hidden_states=None, input_ids=None, **kwargs):
        # attention_mask: compute attention on the places where the value is 1
        if hasattr(self, "tok_embeddings") and input_ids is not None:
            hidden_states = self.tok_embeddings(input_ids)
            if self.embed_grad_scale != 1:
                hidden_states = (
                    self.embed_grad_scale * hidden_states + (1 - self.embed_grad_scale) * hidden_states.detach()
                )

        for _, block in enumerate(self.layers):
            hidden_states = block(hidden_states, residual=None, **kwargs)

        if hasattr(self, "norm"):
            hidden_states = self.norm(hidden_states.float())
        if hasattr(self, "output"):
            hidden_states = self.output(hidden_states)

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
            state_dict[f"layers.{i}.attention.wqkv.weight"] = torch.chunk(
                state_dict.pop(f"model.layers.{layer_ids}.attention.wqkv.weight"),
                split_size,
                dim=0,
            )[local_rank]
            state_dict[f"layers.{i}.attention.wo.weight"] = torch.chunk(
                state_dict.pop(f"model.layers.{layer_ids}.attention.wo.weight"),
                split_size,
                dim=row_dim,
            )[local_rank]

            # ffn
            state_dict[f"layers.{i}.feed_forward.w1.weight"] = torch.chunk(
                state_dict.pop(f"model.layers.{layer_ids}.feed_forward.w1.weight"),
                split_size,
                dim=0,
            )[local_rank]
            state_dict[f"layers.{i}.feed_forward.w3.weight"] = torch.chunk(
                state_dict.pop(f"model.layers.{layer_ids}.feed_forward.w3.weight"),
                split_size,
                dim=0,
            )[local_rank]
            state_dict[f"layers.{i}.feed_forward.w2.weight"] = torch.chunk(
                state_dict.pop(f"model.layers.{layer_ids}.feed_forward.w2.weight"),
                split_size,
                dim=row_dim,
            )[local_rank]

            # attn norm
            state_dict[f"layers.{i}.attention_norm.weight"] = state_dict.pop(
                f"model.layers.{layer_ids}.attention_norm.weight"
            )
            # ffn norm
            state_dict[f"layers.{i}.ffn_norm.weight"] = state_dict.pop(f"model.layers.{layer_ids}.ffn_norm.weight")

            # replace value within decoder layer
            for name in list(state_dict.keys()):
                if name.startswith(f"layers.{i}"):
                    new_state_dict[name.replace(f".{i}.", f".{idx}.")] = state_dict.pop(name)

        # embedding
        if (gpc.get_local_rank(ParallelMode.PIPELINE) == 0) or (not gpc.is_using_parallel_mode(ParallelMode.PIPELINE)):
            new_state_dict["tok_embeddings.weight"] = torch.chunk(
                state_dict.pop("model.tok_embeddings.weight"),
                split_size,
                dim=embed_concat_dim,
            )[local_rank]

        # output
        if gpc.is_last_rank(ParallelMode.PIPELINE):
            new_state_dict["output.weight"] = torch.chunk(
                state_dict.pop("output.weight"),
                split_size,
                dim=0,
            )[local_rank]
            new_state_dict["norm.weight"] = state_dict.pop("model.norm.weight")

        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)

        if gpc.get_local_rank(ParallelMode.DATA) == 0:
            pp_rank = 0 if not gpc.is_initialized(ParallelMode.PIPELINE) else gpc.get_local_rank(ParallelMode.PIPELINE)
            logger.info(
                f"Missing keys:{missing_keys}, unexpected keys:{unexpected_keys} in "
                f"tp:{gpc.get_local_rank(ParallelMode.TENSOR)}, pp:{pp_rank}"
            )

        internlm_accelerator.empty_cache()

    @staticmethod
    def load_internlm2_with_dynamic_parallel_size(folder, model):
        """Load InternLM2 with dynamic parallel size."""
        assert folder is not None, "Please specify the folder of the pretrained model"
        assert gpc.config.model_type in ["INTERNLM2_PUBLIC"], "dynamic_parallel is only for INTERNLM2_PUBLIC"

        fns = get_fns(folder)
        if gpc.is_rank_for_log():
            logger.info(f"Loading pretrained model from {folder}")
        model_fns, old_tp, old_pp = get_parallel_size_from_file(fns)  # pylint: disable=W0612

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

        pp = gpc.get_world_size(ParallelMode.PIPELINE)  # noqa: F841 # pylint: disable=W0612

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
                states = {k.replace("model.", ""): v for k, v in states.items()}
                for i in range(start, end):
                    if i >= model.last_layer:
                        break
                    if i < model.first_layer:
                        continue

                    for name in list(states.keys()):
                        if f".{i-start}." in name:
                            to_name = name.replace(f".{i-start}.", f".{i-model.first_layer}.")

                            if gpc.config.model_type == "INTERNLM2_PUBLIC":
                                if "norm" in name:
                                    tmp_states[to_name] = [states.pop(name)]
                                elif any(x in name for x in ("wo", "w2")):
                                    tmp_states[to_name] = tmp_states.get(to_name, [])
                                    tmp_states[to_name].append(states.pop(name).chunk(ratio, dim=1)[rank])
                                elif any(x in name for x in ("w1", "w3")):
                                    tmp_states[to_name] = tmp_states.get(to_name, [])
                                    tmp_states[to_name].append(states.pop(name).chunk(ratio, dim=0)[rank])
                                elif any(x in name for x in ("wqkv",)):
                                    tmp_states[to_name] = tmp_states.get(to_name, [])
                                    if tp > gpc.config.model.num_kv_attention_heads:
                                        assert old_tp <= gpc.config.model.num_kv_attention_heads, (
                                            f"`old_tp ({old_tp}) => tp ({tp})` is not supported. "
                                            "At least one of `tp` and `old_tp` should be less than or "
                                            "equal to `num_kv_attention_heads`"
                                        )
                                        # Suitable for cases where the num_kv_attention_head is small,
                                        # but you want to have a large TP Size
                                        q_per_kv = (
                                            gpc.config.model.num_attention_heads
                                            // gpc.config.model.num_kv_attention_heads
                                        )
                                        head_dim = gpc.config.model.hidden_size // gpc.config.model.num_attention_heads
                                        index = torch.concat(
                                            (
                                                torch.arange(q_per_kv).chunk(ratio, dim=0)[tp_rank % ratio],
                                                torch.tensor([q_per_kv, q_per_kv + 1]),
                                            )
                                        )
                                        index = index + (q_per_kv + 2) * (tp_rank // ratio)
                                        index = index % (
                                            (q_per_kv + 2) * (gpc.config.model.num_kv_attention_heads / old_tp)
                                        )
                                        index = index * head_dim
                                        index = index.repeat_interleave(head_dim) + torch.arange(head_dim).repeat(
                                            index.shape[0]
                                        )
                                        tmp_states[to_name].append(
                                            torch.index_select(states.pop(name), 0, index.to(torch.int32))
                                        )
                                    else:
                                        tmp_states[to_name].append(states.pop(name).chunk(ratio, dim=0)[rank])
                                else:
                                    raise KeyError(f"Unknown key {name}.")

                            else:
                                assert False, "unsupported model type"

                if "tok_embeddings.weight" in states and model.first_layer == 0:
                    tmp_states["tok_embeddings.weight"] = tmp_states.get("tok_embeddings.weight", [])
                    tmp_states["tok_embeddings.weight"].append(
                        states["tok_embeddings.weight"].chunk(ratio, dim=1)[rank]
                    )
                if "output.weight" in states and model.last_layer == gpc.config.model.num_layers:
                    tmp_states["norm.weight"] = [states["norm.weight"]]
                    tmp_states["output.weight"] = tmp_states.get("output.weight", [])
                    tmp_states["output.weight"].append(states["output.weight"].chunk(ratio, dim=0)[rank])

                states = {}

            for name in list(tmp_states.keys()):
                data = tmp_states.pop(name)
                if len(data) == 1:
                    current_states[name] = data[0]
                else:
                    current_states[name] = torch.concat(
                        data, dim=1 if name == "tok_embeddings.weight" or any(x in name for x in ("wo", "w2")) else 0
                    )
                    # Merge copied kv heads
                    if "wqkv" in name and old_tp > gpc.config.model.num_kv_attention_heads:
                        assert (
                            tp <= gpc.config.model.num_kv_attention_heads
                        ), "new_tp should be less than or equal to num_kv_attention_heads"
                        head_dim = gpc.config.model.hidden_size // gpc.config.model.num_attention_heads
                        q_per_kv = gpc.config.model.num_attention_heads // gpc.config.model.num_kv_attention_heads
                        copied_times = old_tp // gpc.config.model.num_kv_attention_heads
                        cur_q_per_kv = q_per_kv // copied_times

                        # pylint: disable=all
                        def duplicate_kv_index(i):
                            if i % (cur_q_per_kv + 2) >= cur_q_per_kv:
                                return i
                            else:
                                return -100

                        def unique_kv_index(i):
                            if i // (cur_q_per_kv + 2) == copied_times - 1 or i % (cur_q_per_kv + 2) < cur_q_per_kv:
                                return i
                            else:
                                return -100

                        # pylint: enable=all

                        # Verify
                        duplicate_index = [duplicate_kv_index(i) for i in range((cur_q_per_kv + 2) * copied_times)]
                        duplicate_index = [i for i in duplicate_index if i != -100]
                        duplicate_index = _duplicate_index = torch.tensor(duplicate_index)
                        for i in range(gpc.config.model.num_kv_attention_heads // tp - 1):
                            duplicate_index = torch.concat(
                                (duplicate_index, _duplicate_index + duplicate_index.max() + 1), dim=0
                            )
                        duplicate_kv = []
                        for index in duplicate_index.reshape(-1, copied_times * 2).chunk(copied_times, dim=-1):
                            index = index.reshape(-1) * head_dim
                            index = index.repeat_interleave(head_dim) + torch.arange(head_dim).repeat(index.shape[0])
                            duplicate_kv.append(torch.index_select(current_states[name], 0, index))
                        assert reduce(
                            lambda x, y: x and y,
                            [torch.allclose(duplicate_kv[0], x, atol=1e-5) for x in duplicate_kv[1:]],
                        ), "Copied kv heads are not equal after training!"

                        # Merge
                        unique_index = [unique_kv_index(i) for i in range((cur_q_per_kv + 2) * copied_times)]
                        unique_index = [i for i in unique_index if i != -100]
                        unique_index = _unique_index = torch.tensor(unique_index)
                        for i in range(gpc.config.model.num_kv_attention_heads // tp - 1):
                            unique_index = torch.concat((unique_index, _unique_index + unique_index.max() + 1), dim=0)
                        unique_index = unique_index * head_dim
                        unique_index = unique_index.repeat_interleave(head_dim) + torch.arange(head_dim).repeat(
                            unique_index.shape[0]
                        )
                        current_states[name] = torch.index_select(current_states[name], 0, unique_index)
        missing_keys, unexpected_keys = model.load_state_dict(current_states, strict=False)

        if gpc.get_local_rank(ParallelMode.DATA) == 0:
            pp_rank = 0 if not gpc.is_initialized(ParallelMode.PIPELINE) else gpc.get_local_rank(ParallelMode.PIPELINE)
            logger.info(
                f"Missing keys:{missing_keys}, unexpected keys:{unexpected_keys} in "
                f"tp:{gpc.get_local_rank(ParallelMode.TENSOR)}, pp:{pp_rank}"
            )

    @staticmethod
    def convert_internevo2hf_weights(src: str, tgt: str) -> None:
        def permute(qkv, num_heads, num_kv_heads, head_dim, adapt_hf=True):
            if adapt_hf:
                return qkv
            q_per_kv = num_heads // num_kv_heads
            qkv = rearrange(qkv.T, "o (g n i) -> o g n i", n=q_per_kv + 2, i=head_dim)
            q, k, v = qkv[..., :q_per_kv, :], qkv[..., -2:-1, :], qkv[..., -1:, :]
            q = torch.cat([q[..., ::2], q[..., 1::2]], dim=-1)
            k = torch.cat([k[..., ::2], k[..., 1::2]], dim=-1)
            qkv = torch.cat((q, k, v), dim=2)
            qkv = rearrange(qkv, "o g n i -> o (g n i)").T
            return qkv

        model_config = gpc.config.model
        tp_mode = gpc.config.parallel.tensor["mode"]
        row_dim = 0 if tp_mode == "isp" else 1
        if model_config["embed_split_hidden"]:
            embed_concat_dim = 1
        else:
            embed_concat_dim = 0

        # load states
        states, num_shards = InternLM2.load_sharded_states(src)

        # convert state_dict
        state_dict = {}
        embedding_key_list = ["tok_embeddings.word_embeddings.weight", "tok_embeddings.weight", None]
        for layer_i in tqdm(range(model_config["num_layers"])):
            # attn norm, ffn norm
            state_dict.update(
                {
                    f"model.layers.{layer_i}.attention_norm.weight": states[0][
                        f"layers.{layer_i}.attention_norm.weight"
                    ].clone(),
                    f"model.layers.{layer_i}.ffn_norm.weight": states[0][f"layers.{layer_i}.ffn_norm.weight"].clone(),
                }
            )
            # attn
            state_dict[f"model.layers.{layer_i}.attention.wqkv.weight"] = permute(
                torch.cat([states[i][f"layers.{layer_i}.attention.wqkv.weight"] for i in range(num_shards)], dim=0),
                num_heads=model_config["num_attention_heads"],
                num_kv_heads=model_config["num_kv_attention_heads"],
                head_dim=model_config["hidden_size"] // model_config["num_attention_heads"],
                adapt_hf=model_config.get("adapt_hf", True),
            )
            state_dict[f"model.layers.{layer_i}.attention.wo.weight"] = torch.cat(
                [states[i][f"layers.{layer_i}.attention.wo.weight"] for i in range(num_shards)], dim=row_dim
            )
            # ffn
            state_dict[f"model.layers.{layer_i}.feed_forward.w1.weight"] = torch.cat(
                [states[i][f"layers.{layer_i}.feed_forward.w1.weight"] for i in range(num_shards)], dim=0
            )
            state_dict[f"model.layers.{layer_i}.feed_forward.w2.weight"] = torch.cat(
                [states[i][f"layers.{layer_i}.feed_forward.w2.weight"] for i in range(num_shards)], dim=row_dim
            )
            state_dict[f"model.layers.{layer_i}.feed_forward.w3.weight"] = torch.cat(
                [states[i][f"layers.{layer_i}.feed_forward.w3.weight"] for i in range(num_shards)], dim=0
            )
        # embedding, output
        for embedding_key in embedding_key_list:
            if embedding_key in states[0]:
                break
        if embedding_key is None:
            raise KeyError("Cannot find embedding key!")
        state_dict.update(
            {
                "model.norm.weight": states[0]["norm.weight"],
                "model.tok_embeddings.weight": torch.cat(
                    [states[i][embedding_key] for i in range(num_shards)], dim=embed_concat_dim
                ),
                "output.weight": torch.cat([states[i]["output.weight"] for i in range(num_shards)], dim=0),
            },
        )

        # save state_dict to hf format
        shards, index = shard_checkpoint(state_dict, weights_name=SAFE_WEIGHTS_NAME)
        for shard_file, shard in shards.items():
            llm_save(save_path=os.path.join(tgt, shard_file), saved_obj=shard, metadata={"format": "pt"})
        if index is not None:
            llm_save(save_path=os.path.join(tgt, SAFE_WEIGHTS_INDEX_NAME), saved_obj=index)
