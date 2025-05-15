"""                 self-written DeepSeek V3 model in PyTorch, trainable with FSDP+EP                """
""" modified from https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/configuration_deepseek.py """
import torch
from transformers.configuration_utils import PretrainedConfig

class DeepseekV3Config(PretrainedConfig):
    model_type = "deepseek_v3"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=129280, # v2 102400, v3 129280
        hidden_size=7168, # Dimension of the hidden representations.
        intermediate_size=18432, # Dimension of the MLP representations.
        moe_intermediate_size = 2048, # Dimension of the MoE representations.
        num_hidden_layers=8, # Number of hidden layers in the Transformer decoder. num_hidden_layers=8 is runnable with FSDP and EP=8 on 32gpus
        num_nextn_predict_layers=1, # Number of MTP layers
        num_attention_heads=128, # Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads=128, # Number of key_value heads for each attention layer in the Transformer decoder.
        n_shared_experts = 1, # Number of shared experts, None means dense model.
        n_routed_experts = 256, # Number of routed experts, None means dense model.
        routed_scaling_factor = 2.5, # Scaling factor or routed experts.
        kv_lora_rank = 512,
        q_lora_rank = 1536,
        qk_rope_head_dim = 64,
        v_head_dim = 128,
        qk_nope_head_dim = 128,
        n_group = 8, # Number of groups for routed experts.
        topk_group = 4, # Number of selected groups for each token(for each token, ensuring the selected experts is only within `topk_group` groups).
        num_experts_per_tok = 8, # Number of selected experts, None means dense model.
        moe_layer_freq = 1, # The frequency of the MoE layer: one expert layer for every `moe_layer_freq - 1` dense layers.
        first_k_dense_replace = 3, # Number of dense layers in shallow layers.
        norm_topk_prob = True, # Whether to normalize the weights of the routed experts.
        scoring_func = 'sigmoid', # Method of computing expert weights.
        hidden_act="silu", # The non-linear activation function (function or string) in the decoder.
        max_position_embeddings=4096,
        initializer_range=0.02, # The standard deviation for initializing all weights.
        rms_norm_eps=1e-6, # The epsilon used by the rms normalization layers.
        use_cache=False,
        pad_token_id=0,
        bos_token_id=0,
        eos_token_id=1,
        tie_word_embeddings=False, # Whether the model's input and output word embeddings should be tied.
        rope_theta=10000.0, # The base period of the RoPE embeddings.
        rope_scaling=None,
        attention_bias=False, # Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout=0.0, # The dropout ratio for the attention probabilities.
        mlp_bias=False, # Whether to use bias in MLP (both FFN and GroupedFFN).
        mlp_layer_fusion=False, # Whether to use layer fusion in MLP (both FFN and GroupedFFN).
        # MoE load balancing
        moe_loss_type="default", # "none", "default", "seq_aux"
        aux_loss_alpha = 0.0001, # Auxiliary loss weight coefficient.
        aux_free = True, # Auxiliary free load balancing via e_score_correction_bias.
        aux_free_update_rate = 0.001, # Update rate for auxiliary free load balancing.
        device="meta",
        dtype=torch.bfloat16,
        multiple_of=1,
        checkpoint=1,
        attn_implementation="flash_attention_2",
        return_dict=False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.num_attention_heads = num_attention_heads
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.n_shared_experts = n_shared_experts
        self.n_routed_experts = n_routed_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.n_group = n_group
        self.topk_group = topk_group
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_layer_freq = moe_layer_freq
        self.first_k_dense_replace = first_k_dense_replace
        self.norm_topk_prob = norm_topk_prob
        self.scoring_func = scoring_func
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias
        self.mlp_layer_fusion = mlp_layer_fusion
        self.moe_loss_type = moe_loss_type
        self.aux_loss_alpha = aux_loss_alpha
        self.aux_free = aux_free
        self.aux_free_update_rate = aux_free_update_rate
        self.device = device
        self.dtype = dtype
        self.multiple_of = multiple_of
        self.checkpoint = checkpoint
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            attn_implementation=attn_implementation,
            return_dict=return_dict,
            **kwargs,
        )
