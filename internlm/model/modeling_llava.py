from typing import Optional

import torch
from torch import nn

from internlm.core.context import ParallelMode
from internlm.core.context.parallel_context import global_context as gpc
from internlm.core.naive_amp import set_output_attr_to_module
from internlm.initialize.initialize_tensor import normal_, uniform_
from internlm.model.modeling_llama import PackedFlashLlamaLayer1D
from internlm.model.modules.embedding import Embedding1D
from internlm.model.ops.linear import RewardModelLinear, ScaleColumnParallelLinear
from internlm.model.utils import (
    gather_forward_split_backward,
    split_forward_gather_backward,
    try_import_RMSNorm,
)
from internlm.solver.pipeline_utils import partition_uniform
from internlm.utils.common import filter_kwargs
from internlm.utils.logger import get_logger
from internlm.utils.registry import MODEL_INITIALIZER

MODEL_TYPE = "LLAVA"

logger = get_logger(__file__)
RMSNorm = try_import_RMSNorm()


class PackedFlashLlava1D(nn.Module):
    """
    1D Packed Flash Llava.

    Args:
        num_layers (int): The number of layer. 12 by default.
        hidden_size (int): The size of hidden state. 768 by default.
        num_attention_heads (int): The number of attention head. 12 by default.
        vocab_size (int): The size of vocabulary. 50304 by default.
        mlp_ratio (int): The ratio of MLP layers. 4 by default.
        attn_drop_rate (float): The dropout rate of attention module. 0.0 by default.
        drop_rate (float): The dropout rate of input hidden state. 0.0 by default.
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
        image_token_id (int): image token id. 200000 by default.
        vit_cfg (dict): The config of vision tower. None by default.
        vision_proj_cfg (dict): The config of vision projector. None by default.
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
        device: Optional[torch.device] = None,
        apply_post_layer_norm=False,
        no_bias=False,
        residual_in_fp32: bool = False,
        norm_type: str = "rmsnorm",
        adapt_hf: bool = False,
        is_reward: bool = False,
        dropout_selective_checkpoint: bool = True,
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
        image_token_id: int = 200000,
        vit_cfg=None,
        vision_proj_cfg=None,
    ):
        super().__init__()

        self.use_flash_attn = use_flash_attn
        if checkpoint_fraction <= 0:
            checkpoint = False
        if not checkpoint:
            checkpoint_fraction = 0
        checkpoint_layer_num = num_layers * checkpoint_fraction
        sequence_parallel = gpc.config.parallel.get("sequence_parallel", False)
        self.tp_mode = "mtp"
        self.dtype = dtype
        self.image_token_id = image_token_id

        if isinstance(gpc.config.parallel["tensor"], dict):
            self.tp_mode = gpc.config.parallel["tensor"].get("mode", "mtp")

        if is_reward:
            head_cls = RewardModelLinear
        else:
            head_cls = ScaleColumnParallelLinear

        if first:
            if embed_split_hidden or not gpc.config.model.use_flash_attn:
                self.tok_embeddings = Embedding1D(num_embeddings=vocab_size, embedding_dim=hidden_size)
            else:
                from flash_attn.modules.embedding import ParallelGPT2Embeddings

                self.tok_embeddings = ParallelGPT2Embeddings(
                    embed_dim=hidden_size,
                    vocab_size=vocab_size,
                    max_position_embeddings=-1,
                    process_group=gpc.get_group(ParallelMode.TENSOR),
                    padding_idx=None,
                    sequence_parallel=sequence_parallel,
                    device=device,
                    dtype=dtype,
                )
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
                    dtype=dtype,
                    layer_norm_epsilon=layer_norm_epsilon,
                    checkpoint=lid < checkpoint_layer_num,
                    layer_idx=lid + start_layer_idx,  # This parameter is used for caching during generation
                    residual_in_fp32=residual_in_fp32,
                    device=device,
                    apply_post_layer_norm=apply_post_layer_norm,
                    fused_dropout_add_ln=False,
                    no_bias=no_bias,
                    norm_type=norm_type,
                    dropout_selective_checkpoint=dropout_selective_checkpoint,
                    use_scaled_init=use_scaled_init,
                    use_swiglu=use_swiglu,
                    use_flash_attn=use_flash_attn,
                    adapt_hf=adapt_hf,
                    attn_wqkv_init_std=attn_wqkv_init_std,
                    attn_other_init_std=attn_other_init_std,
                    ffn_uplayer_init_std=ffn_uplayer_init_std,
                    ffn_other_init_std=ffn_other_init_std,
                    init_type=init_type,
                    rope_base=rope_base,
                    tp_mode=self.tp_mode,
                )
                for lid in range(num_layers)
            ]
        )

        if last:
            if not apply_post_layer_norm:
                if norm_type == "rmsnorm":
                    self.norm = RMSNorm(hidden_size, eps=layer_norm_epsilon)
                else:
                    self.norm = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)

            self.output = head_cls(
                in_features=hidden_size,
                out_features=gpc.get_world_size(ParallelMode.TENSOR) if is_reward else vocab_size,
                process_group=gpc.get_group(ParallelMode.TENSOR),
                bias=False,
                device=device,
                dtype=dtype,
                weight_scale=embed_grad_scale,
            )
            set_output_attr_to_module(self.output)
            for _, param in self.output.named_parameters():
                if init_type == "normal":
                    normal_(std=out_head_init_std)(param)
                else:
                    uniform_(std=out_head_init_std)(param)

        self.parallel_output = parallel_output
        assert vit_cfg is not None
        if first:
            from internlm.model.llava_modules.clip_builder import build_vision_tower

            self.vit = build_vision_tower(vit_cfg)
            self.vit.requires_grad_(False)

        assert vision_proj_cfg is not None
        if first:
            from internlm.model.llava_modules.projector_builder import (
                build_vision_projector,
            )

            self.vision_proj = build_vision_projector(vision_proj_cfg)
            # self.vision_proj.requires_grad_(False)

    def forward(  # pylint: disable=W0102
        self,
        hidden_states=None,
        images=[],
        cu_seqlens=None,
        input_ids=None,
        indexes=None,
        inference_params=None,
    ):
        xs = []
        pure_text = False
        input_ids = input_ids.clone()
        assert hasattr(self, "vit")
        assert hasattr(self, "vision_proj")
        if len(images) == 1 and len(images[0]) == 0:  # make sure grad in Qformer for update
            images = [torch.rand(1, 3, self.vit.image_size, self.vit.image_size).cuda().to(self.dtype)]
            pure_text = True

        for image in images:
            assert len(image) > 0
            if len(image) == 0:
                x = []
            else:
                assert not isinstance(image, list), image
                x = image.to(torch.cuda.current_device()).to(self.dtype)
                x = self.vit(x)
                x = self.vision_proj(x)
            xs.append(x)

        # attention_mask: compute attention on the places where the value is 1
        if hasattr(self, "tok_embeddings"):
            org_ids = input_ids.clone()
            input_ids[input_ids == self.image_token_id] = 0
            hidden_states = self.tok_embeddings(input_ids).clone()
            if pure_text and len(xs) > 0:
                hidden_states = hidden_states + 0 * xs[0].sum()
            else:
                for i in range(len(xs)):
                    hidden_states[i, org_ids[i] == self.image_token_id] = (xs[i].reshape((-1, xs[i].shape[-1]))).to(
                        hidden_states.dtype
                    )

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
            if gpc.is_evaluating is True:
                hidden_states = self.output(hidden_states, gather_dim=1, tp_mode=self.tp_mode)
            else:  # Training
                hidden_states = self.output(hidden_states, gather_dim=0, tp_mode=self.tp_mode)

        if not self.parallel_output:
            hidden_states = gather_forward_split_backward(hidden_states, ParallelMode.TENSOR, dim=-1)

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
        chunk = PackedFlashLlava1D(**filter_kwargs(PackedFlashLlava1D.__init__, kwargs)).to(device)

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
    adapt_hf=False,
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
    image_token_id: int = 200000,
    vit_cfg=None,
    vision_proj_cfg=None,
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
        image_token_id=image_token_id,
        vit_cfg=vit_cfg,
        vision_proj_cfg=vision_proj_cfg,
    )

    return _build_generic_model_1d(num_layers=num_layers, num_chunks=num_chunks, **cfg)
