# Copyright (c) InternLM. All rights reserved.
import math
from typing import Optional

import torch
from torch import nn

from internlm.core.context import ParallelMode
from internlm.core.context.parallel_context import global_context as gpc
from internlm.initialize.initialize_tensor import (normal_,
                                                   scaled_init_method_normal,
                                                   scaled_init_method_uniform,
                                                   uniform_)
from internlm.model.modules.embedding import Embedding1D
from internlm.model.modules.linear import new_linear
from internlm.model.modules.mha import QKVPackedGQA
from internlm.model.modules.mlp import new_fead_forward
from internlm.model.modules.norm import new_layer_norm
from internlm.model.modules.utils import split_forward_gather_backward
from internlm.solver.activation_checkpoint import activation_checkpoint
from internlm.solver.pipeline_utils import partition_uniform
from internlm.utils.common import filter_kwargs
from internlm.utils.logger import get_logger
from internlm.utils.registry import MODEL_INITIALIZER

MODEL_TYPE = "INTERNLM2_PUBLIC"

logger = get_logger(__file__)


class PackedFlashLlamaLayer1D(nn.Module):
    """
    InternLM2 layer.

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
        use_flash_attn (bool): Whether use flash-attn. True by default.
        tp_mode (str): The string value of tensor parallel mode, should be in ["mtp", "msp", "fsp", "isp"],
                       "mtp" by default.
        attn_wqkv_init_std (float): std used to init attn_wqkv weight. 0.02 by default,
        attn_other_init_std (float): std used to init attn_other weight. 0.02 by default,
        ffn_uplayer_init_std (float): std used to init w1, w2 weight in ffn when using glu
            otherwise init fc1 weight in ffn. 0.02 by default,
        ffn_other_init_std (float): std used to init ffn_other weight. 0.02 by default,
        init_type (str): Initialization type. Use uniform or normal. "normal" by default,
        rope_base (int): The value of `base` for rotary position embeddings. 10000 by default.
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
        adapt_hf: bool = True,
        dropout_selective_checkpoint: bool = True,
        use_scaled_init: bool = True,
        use_swiglu: bool = True,
        # use_flash_attn: bool = True,
        # tp_mode: str = "mtp",
        attn_wqkv_init_std: float = 0.02,
        attn_other_init_std: float = 0.02,
        ffn_uplayer_init_std: float = 0.02,
        ffn_other_init_std: float = 0.02,
        init_type: str = "normal",
        rope_base: int = 10000,
    ):
        super().__init__()
        self.checkpoint = checkpoint
        # dropout selective checkpoint can only be enabled when checkpoint is disabled.
        self.dropout_selective_checkpoint = dropout_selective_checkpoint is True and checkpoint is False
        self.layer_idx = layer_idx
        # self.use_flash_attn = use_flash_attn
        self.prenorm = not apply_post_layer_norm
        assert not fused_dropout_add_ln, "dropout_add_layer_norm can not be used here"
        self.fused_dropout_add_ln = fused_dropout_add_ln
        self.attn_wqkv_init_std = attn_wqkv_init_std
        self.attn_other_init_std = attn_other_init_std
        self.ffn_uplayer_init_std = ffn_uplayer_init_std
        self.ffn_other_init_std = ffn_other_init_std

        self.max_position_embeddings = max_position_embeddings
        self.use_dynamic_ntk_rope = use_dynamic_ntk_rope
        # self.tp_mode = tp_mode
        # parallel_mode = ParallelMode.WEIGHT if self.tp_mode == "isp" else ParallelMode.TENSOR

        head_dim = hidden_size // num_attention_heads
        self.attention = QKVPackedGQA(
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
            rot_embed_HF_impl=adapt_hf,
            bias=not no_bias,
            rope_base=rope_base,
        )

        self.dropout1 = nn.Dropout(drop_rate)
        self.dropout2 = nn.Dropout(drop_rate)
        self.attention_norm = new_layer_norm(norm_type, hidden_size, eps=layer_norm_epsilon)
        self.ffn_norm = new_layer_norm(norm_type, hidden_size, eps=layer_norm_epsilon)
        # if self.fused_dropout_add_ln and self.use_flash_attn:
        #     from flash_attn.ops.layer_norm import dropout_add_layer_norm

        #     assert dropout_add_layer_norm is not None, "dropout_add_ln is not installed"
        #     assert isinstance(self.attention_norm, nn.LayerNorm) and isinstance(self.dropout1, nn.Dropout)

        if use_swiglu:
            self.feed_forward = new_fead_forward(
                hidden_size,
                int(hidden_size * mlp_ratio),
                out_features=hidden_size,
                bias=False,
                device=device,
                dtype=dtype,
            )
        else:
            # TODO: support gelu and so on.
            raise ValueError("NYI")

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

    def forward(
        self, hidden_states, residual=None, cu_seqlens=None, indexes=None, inference_params=None, max_seqlen=None
    ):
        if self.checkpoint and self.training:
            return activation_checkpoint(
                self._forward, False, hidden_states, residual, cu_seqlens, indexes, inference_params, max_seqlen
            )
        else:
            return self._forward(hidden_states, residual, cu_seqlens, indexes, inference_params, max_seqlen)

    def _forward(
        self, hidden_states=None, residual=None, cu_seqlens=None, indexes=None, inference_params=None, max_seqlen=None
    ):
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
            mixer_kwargs = {
                "cu_seqlens": cu_seqlens,
                "max_seqlen": max_seqlen,
                "indexes": indexes,
                "inference_params": inference_params,
            }
            hidden_states = self.attention(hidden_states, **mixer_kwargs)

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
            mixer_kwargs = {
                "cu_seqlens": cu_seqlens,
                "max_seqlen": max_seqlen,
                "indexes": indexes,
                "inference_params": inference_params,
            }
            mixer_out = self.attention(hidden_states, **mixer_kwargs)
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


class PackedFlashLlama1D(nn.Module):
    """
    1D Packed Flash InternLM2.

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
        checkpoint (bool): Whether to use checkpointing to save VRAM. True by default.
        checkpoint_fraction (float): The proportion of layers that need to be checkpointed compared to the total number
                                    of layers. 1.0 by default.
        layer_norm_epsilon (float): A value added to the denominator for numerical stability. 1e-6 by default.
        first (bool): Whether input embedding layer or not. False by default.
        last (bool): Whether output embedding layer or not. False by default.
        embed_split_hidden (bool): Split the embedding layer in the hidden state dimention or vocabulary dimention.
                                    True by default.
        embed_grad_scale (float): Refer to GLM-130B, for training stability. 0.1 by default.
        parallel_output (bool): If it is necessary to collect the output of parallel computing. True by default.
        start_layer_idx (int): The index of start layer in the pipeline. 0 by default.
        use_dynamic_ntk_rope (bool): Whether to use dynamic ntk rope. False by default.
        device (Optional[Union[str, torch.device]]): The device will be used. None by default.
        residual_in_fp32 (bool): Whether to use residual in fp32. False by default.
        norm_type (str): Normalization type. Use RMSNorm or LayerNorm. "rmsnorm" by default.
        use_flash_attn (bool): Whether to use flash-attn. True by default.
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
        tp_mode (str): The string value of tensor parallel mode, should be in ["mtp", "msp", "fsp", "isp"],
                       "mtp" by default.
    """

    def __init__(
        self,
        num_layers: int = 12,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        num_kv_attention_heads: int = 12,
        vocab_size: int = 50304,
        mlp_ratio: int = 4,
        attn_drop_rate: float = 0.0,
        drop_rate: float = 0.0,
        max_position_embeddings: int = 2048,
        dtype: torch.dtype = torch.float,
        checkpoint: bool = False,
        checkpoint_fraction: float = 1.0,
        layer_norm_epsilon: float = 1e-5,
        first: bool = False,
        last: bool = False,
        embed_split_hidden: bool = False,
        embed_grad_scale: float = 0.1,
        parallel_output: bool = True,
        start_layer_idx: int = 0,
        use_dynamic_ntk_rope: bool = False,
        device: Optional[torch.device] = None,
        apply_post_layer_norm=False,
        no_bias=False,
        residual_in_fp32: bool = False,
        norm_type: str = "rmsnorm",
        adapt_hf: bool = True,
        is_reward: bool = False,
        dropout_selective_checkpoint: bool = True,
        use_scaled_init: bool = True,
        use_swiglu: bool = True,
        # use_flash_attn: bool = True,
        embedding_init_std: float = 0.02,
        attn_wqkv_init_std: float = 0.02,
        attn_other_init_std: float = 0.02,
        ffn_uplayer_init_std: float = 0.02,
        ffn_other_init_std: float = 0.02,
        out_head_init_std: float = 0.02,
        init_type: str = "normal",
        rope_base: int = 10000,
        norm_head: bool = False,
        # tp_mode: str = "mtp",
    ):
        super().__init__()

        # self.use_flash_attn = use_flash_attn

        if checkpoint_fraction <= 0:
            checkpoint = False
        if not checkpoint:
            checkpoint_fraction = 0
        checkpoint_layer_num = num_layers * checkpoint_fraction

        # self.tp_mode = tp_mode
        # if isinstance(gpc.config.parallel["tensor"], dict):
        #     self.tp_mode = gpc.config.parallel["tensor"].get("mode", "mtp")

        # sequence_parallel = gpc.config.parallel.get("sequence_parallel", False)

        if first:
            # if embed_split_hidden or not gpc.config.model.use_flash_attn:
            self.tok_embeddings = Embedding1D(num_embeddings=vocab_size, embedding_dim=hidden_size)
            # else:
            #     from flash_attn.modules.embedding import ParallelGPT2Embeddings

            #     self.tok_embeddings = ParallelGPT2Embeddings(
            #         embed_dim=hidden_size,
            #         vocab_size=vocab_size,
            #         max_position_embeddings=-1,
            #         process_group=gpc.get_group(ParallelMode.TENSOR),
            #         padding_idx=None,
            #         sequence_parallel=sequence_parallel,
            #         device=device,
            #         dtype=dtype,
            #     )
            for _, param in self.tok_embeddings.named_parameters():
                if init_type == "normal":
                    normal_(std=embedding_init_std)(param)
                else:
                    uniform_(std=embedding_init_std)(param)

        self.embed_grad_scale = embed_grad_scale

        self.layers = nn.ModuleList(
            [
                PackedFlashLlamaLayer1D(
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
                    # use_flash_attn=use_flash_attn,
                    adapt_hf=adapt_hf,
                    attn_wqkv_init_std=attn_wqkv_init_std,
                    attn_other_init_std=attn_other_init_std,
                    ffn_uplayer_init_std=ffn_uplayer_init_std,
                    ffn_other_init_std=ffn_other_init_std,
                    init_type=init_type,
                    # tp_mode=self.tp_mode,
                    rope_base=rope_base,
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

        self.parallel_output = parallel_output

    def forward(self, hidden_states=None, cu_seqlens=None, input_ids=None, indexes=None, inference_params=None):
        # attention_mask: compute attention on the places where the value is 1
        if hasattr(self, "tok_embeddings"):
            hidden_states = self.tok_embeddings(input_ids)
            if self.embed_grad_scale != 1:
                hidden_states = (
                    self.embed_grad_scale * hidden_states + (1 - self.embed_grad_scale) * hidden_states.detach()
                )
        if isinstance(cu_seqlens, list):
            assert len(cu_seqlens) == 1
            cu_seqlens = cu_seqlens[0].to(hidden_states.device)

        if cu_seqlens is not None:
            cu_seqlens = cu_seqlens.squeeze(0)
            hidden_states = hidden_states.squeeze(0)  # If cu_seqlens is passed in，it indicated a packed state，
            # the batch dimension with a size of 1 should be directly squeezed off.

        if indexes is not None:
            assert len(indexes) == 1
            # The indexes are used to indicate the actual position IDs of each token in the packed input.
            indexes = indexes[0]
            # if the sequence parallel mode is 'isp', the indexes should also be split in sequence dimension.
            # TODO：从模型中移除
            if gpc.config.parallel.sequence_parallel and self.tp_mode == "isp":
                indexes = split_forward_gather_backward(indexes, ParallelMode.TENSOR, dim=0)

        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item() if cu_seqlens is not None else None

        for _, block in enumerate(self.layers):
            hidden_states = block(
                hidden_states,
                residual=None,
                cu_seqlens=cu_seqlens,
                indexes=indexes,
                inference_params=inference_params,
                max_seqlen=max_seqlen,
            )

        if hasattr(self, "norm"):
            hidden_states = self.norm(hidden_states.float())
        if hasattr(self, "output"):
            # Evaluation
            # TODO: 统一并去掉维度
            if gpc.is_evaluating is True:
                hidden_states = self.output(hidden_states, gather_dim=1, tp_mode=self.tp_mode)
            else:  # Training
                hidden_states = self.output(hidden_states, gather_dim=0, tp_mode=self.tp_mode)

        return hidden_states


def _build_generic_model_1d(num_layers, num_chunks, device=torch.device("cuda"), **kwargs):
    """
    build generic model 1d

    Args:
        num_layers (int): The number of layer.
        num_chunks (int): The number of partitions in pipeline parallel.
        device (Optional[Union[str, torch.device]]): The device will be used. torch.device("cuda") by default.

    """
    pipeline_size = gpc.get_world_size(ParallelMode.PIPELINE)
    pipeline_rank = gpc.get_local_rank(ParallelMode.PIPELINE)

    all_parts = partition_uniform(num_layers, pipeline_size, num_chunks)
    parts = all_parts[pipeline_rank]
    if gpc.is_rank_for_log():
        logger.info(f"The layer sharding is {all_parts}.")

    models = []
    kwargs["checkpoint_fraction"] = float(kwargs.get("checkpoint", False))
    start_idx, end_idx = 0, 0
    for start, end in parts:
        start_idx, end_idx = start, end
        kwargs["num_layers"] = end - start
        kwargs["first"] = start == 0
        # If there is no content in the final layer, assign the last layer.
        kwargs["last"] = end == num_layers and len(all_parts[-1]) != 0
        kwargs["device"] = device
        kwargs["start_layer_idx"] = start
        chunk = PackedFlashLlama1D(**filter_kwargs(PackedFlashLlama1D.__init__, kwargs)).to(device)

        models.append(chunk)
    torch.distributed.barrier()
    if len(models) == 1:
        model = models[0]
    else:
        model = nn.ModuleList(models)
    setattr(model, "first_layer", start_idx)
    setattr(model, "last_layer", end_idx)
    return model


@MODEL_INITIALIZER.register_module(module_name=MODEL_TYPE)
def build_model_with_cfg(
    num_chunks=1,
    checkpoint=False,
    dtype=torch.float,
    embed_split_hidden=False,
    num_layers=48,
    hidden_size=2048,
    vocab_size=50304,
    embed_grad_scale=1,
    parallel_output=True,
    num_attention_heads=32,
    num_kv_attention_heads=None,
    mlp_ratio=4.0,
    residual_in_fp32=False,
    norm_type="rmsnorm",
    adapt_hf=True,
    drop_rate=0,
    attn_drop_rate=0,
    apply_post_layer_norm=False,  # pylint: disable=W0613
    no_bias=False,
    deepnorm=False,
    layer_norm_epsilon=1e-5,
    is_reward=False,
    dropout_selective_checkpoint=True,
    use_scaled_init: bool = True,
    use_swiglu: bool = True,
    use_flash_attn: bool = True,
    embedding_init_std: float = 0.02,
    attn_wqkv_init_std: float = 0.02,
    attn_other_init_std: float = 0.02,
    ffn_uplayer_init_std: float = 0.02,
    ffn_other_init_std: float = 0.02,
    out_head_init_std: float = 0.02,
    init_type: str = "normal",
    rope_base: int = 10000,
    norm_head: bool = False,
    max_position_embeddings=2048,
    use_dynamic_ntk_rope=False,
):
    """
    Builde model with config

    Args:
        num_chunks (int): The number of partitions in pipeline parallel. 1 by default.
        checkpoint (bool): Whether to use checkpointing to save VRAM. False by default.
        dtype (torch.dtype): The type of data. torch.float by default.
        embed_split_hidden (bool): Split the embedding layer in the hidden state dimention or vocabulary dimention.
                                    False by default.
        num_layers (int): The number of layer. 48 by default.
        hidden_size (int): The size of hidden state. 2048 by default.
        vocab_size (int): The size of vocabulary. 50304 by default.
        embed_grad_scale (float): Refer to GLM-130B, for training stability. 0.1 by default.
        parallel_output (bool): If it is necessary to collect the output of parallel computing. True by default.
        num_attention_heads (int): The number of attention head. 32 by default.
        mlp_ratio (int): The ratio of MLP layers. 4.0 by default.
        residual_in_fp32 (bool): Whether to use residual in fp32. False by default. It cannot be used temporarily
                                 because this parameter requires inconsistent data types to be passed between pipelines,
                                 which requires significant modifications to internlm.
        norm_type (str): Normalization type. Use RMSNorm or LayerNorm. "rmsnorm" by default.
        drop_rate (float): The dropout rate of input hidden state. 0 by default.
        attn_drop_rate (float): The dropout rate of attention module. 0 by default.
        apply_post_layer_norm (bool): Whether to apply post layer norm. False by default.
        layer_norm_epsilon (float): A value added to the denominator for numerical stability. 1e-5 by default.
        is_reward (bool): Whether to use reward model. False by default.
        dropout_selective_checkpoint (bool): It can only be enabled when checkpoint is disabled. True by default.
        use_scaled_init (bool): Whether to use scaled init. True by default.
        use_swiglu (bool): Whether to use swiglu. True by default.
        use_flash_attn (bool): Whether to use flash-attn. True by default.
        embedding_init_std (float): std used to init embedding weight. 0.02 by default,
        attn_wqkv_init_std (float): std used to init attn_wqkv weight. 0.02 by default,
        attn_other_init_std (float): std used to init attn_other weight. 0.02 by default,
        ffn_uplayer_init_std (float): std used to init w1, w2 weight in ffn when using glu
            otherwise init fc1 weight in ffn. 0.02 by default,
        ffn_other_init_std (float): std used to init ffn_other weight. 0.02 by default,
        out_head_init_std (float): std used to init output lmhead weight. 0.02 by default,
        init_type (str): Initialization type. Use uniform or normal. "normal" by default,
        rope_base (int): The value of `base` for rotary position embeddings. 10000 by default.
        max_position_embeddings (int): The maximum position embeddings. 2048 by default.
        use_dynamic_ntk_rope (bool): Whether to use dynamic ntk rope. False by default.
    """
    if deepnorm:
        raise AssertionError("deepnorm will not be supported in future versions." "Use early versions if necessary.")

    cfg = dict(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_kv_attention_heads=num_kv_attention_heads if num_kv_attention_heads else num_attention_heads,
        checkpoint=checkpoint,
        dtype=dtype,
        embed_split_hidden=embed_split_hidden,
        vocab_size=vocab_size,
        embed_grad_scale=embed_grad_scale,
        parallel_output=parallel_output,
        mlp_ratio=mlp_ratio,
        apply_post_layer_norm=apply_post_layer_norm,
        no_bias=no_bias,
        residual_in_fp32=residual_in_fp32,
        norm_type=norm_type,
        adapt_hf=adapt_hf,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        layer_norm_epsilon=layer_norm_epsilon,
        is_reward=is_reward,
        dropout_selective_checkpoint=dropout_selective_checkpoint,
        use_scaled_init=use_scaled_init,
        use_swiglu=use_swiglu,
        use_flash_attn=use_flash_attn,
        embedding_init_std=embedding_init_std,
        attn_wqkv_init_std=attn_wqkv_init_std,
        attn_other_init_std=attn_other_init_std,
        ffn_uplayer_init_std=ffn_uplayer_init_std,
        ffn_other_init_std=ffn_other_init_std,
        out_head_init_std=out_head_init_std,
        init_type=init_type,
        rope_base=rope_base,
        norm_head=norm_head,
        max_position_embeddings=max_position_embeddings,
        use_dynamic_ntk_rope=use_dynamic_ntk_rope,
    )

    return _build_generic_model_1d(num_layers=num_layers, num_chunks=num_chunks, **cfg)
