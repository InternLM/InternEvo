from typing import Optional

import torch
from torch import nn

from internlm.core.context import ParallelMode
from internlm.core.context.parallel_context import global_context as gpc
from internlm.core.naive_amp import set_output_attr_to_module
from internlm.initialize.initialize_tensor import normal_, uniform_
from internlm.model.base_model import BaseModel
from internlm.model.llava.clip_builder import build_vision_tower
from internlm.model.llava.projector_builder import build_vision_projector
from internlm.model.modeling_llama import Llama2Decoder
from internlm.model.modules.embedding import Embedding1D
from internlm.model.modules.linear import new_linear
from internlm.model.modules.norm import new_layer_norm
from internlm.utils.logger import get_logger

logger = get_logger(__file__)


class Llava(BaseModel):
    """
    1D Packed Flash Llava.

    Args:
        num_layers (int): The number of layer. 48 by default.
        hidden_size (int): The size of hidden state. 2048 by default.
        num_attention_heads (int): The number of attention head. 32 by default.
        num_kv_attention_heads (int): The number of key/value attention heads. Defaults to 32.
        vocab_size (int): The size of vocabulary. 50304 by default.
        mlp_ratio (int): The ratio of MLP layers. 4 by default.
        attn_drop_rate (float): The dropout rate of attention module. 0.0 by default.
        drop_rate (float): The dropout rate of input hidden state. 0.0 by default.
        dtype (torch.dtype): The type of data. torch.float by default.
        checkpoint (bool): Whether to use checkpointing to save VRAM. True by default.
        layer_norm_epsilon (float): A value added to the denominator for numerical stability. 1e-6 by default.
        first (bool): Whether input embedding layer or not. False by default.
        last (bool): Whether output embedding layer or not. False by default.
        embed_grad_scale (float): Refer to GLM-130B, for training stability. 0.1 by default.
        parallel_output (bool): If it is necessary to collect the output of parallel computing. True by default.
        start_layer_idx (int): The index of start layer in the pipeline. 0 by default.
        device (Optional[Union[str, torch.device]]): The device will be used. None by default.
        residual_in_fp32 (bool): Whether to use residual in fp32. False by default.
        norm_type (str): Normalization type. Use RMSNorm or LayerNorm. "rmsnorm" by default.
        qk_interleaved (bool): Whether the odd and even columns of the wq and wk are normally interleaved.
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
        num_layers: int = 48,
        hidden_size: int = 2048,
        num_attention_heads: int = 32,
        num_kv_attention_heads: int = 32,
        vocab_size: int = 50304,
        mlp_ratio: int = 4,
        attn_drop_rate: float = 0.0,
        drop_rate: float = 0.0,
        dtype: torch.dtype = torch.float,
        checkpoint: bool = False,
        layer_norm_epsilon: float = 1e-5,
        first: bool = False,
        last: bool = False,
        embed_grad_scale: float = 0.1,
        parallel_output: bool = True,
        start_layer_idx: int = 0,
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
        mlp_layer_fusion: bool = False,
        multiple_of: int = 256,
        image_token_id: int = 200000,
        vit_cfg=None,
        vision_proj_cfg=None,
    ):
        super().__init__()

        checkpoint_layer_num = num_layers * checkpoint

        self.dtype = dtype
        self.image_token_id = image_token_id
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
                Llama2Decoder(
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
            )
            set_output_attr_to_module(self.output)
            for _, param in self.output.named_parameters():
                if init_type == "normal":
                    normal_(std=out_head_init_std)(param)
                else:
                    uniform_(std=out_head_init_std)(param)

        if first:
            assert vit_cfg is not None
            self.vit = build_vision_tower(vit_cfg)
            self.vit.requires_grad_(False)

            assert vision_proj_cfg is not None
            self.vision_proj = build_vision_projector(vision_proj_cfg)
            # self.vision_proj.requires_grad_(False)

    def forward(self, hidden_states=None, images=None, input_ids=None, **kwargs):
        xs = []
        pure_text = False
        images = [] if images is None else images

        if hasattr(self, "vit") and hasattr(self, "vision_proj") and hasattr(self, "tok_embeddings"):
            # vit
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

            # tok embeddings
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

        for _, block in enumerate(self.layers):
            hidden_states = block(hidden_states, residual=None, **kwargs)

        if hasattr(self, "norm"):
            hidden_states = self.norm(hidden_states.float())

        if hasattr(self, "output"):
            hidden_states = self.output(hidden_states)

        return hidden_states

    @staticmethod
    def load_hf_weights(folder: str, model: nn.Module) -> None:
        raise NotImplementedError

    @staticmethod
    def convert_internevo2hf_weights(src: str, tgt: str) -> None:
        raise NotImplementedError
