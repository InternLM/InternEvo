# Copyright (c) InternLM. All rights reserved.

model_type = "INTERNLM2_PUBLIC"

VOCAB_SIZE = 92544
HIDDEN_SIZE = 2048
NUM_ATTENTION_HEAD = 16
NUM_KV_ATTENTION_HEAD = 8
MULTIPLE_OF = 128
MLP_RATIO = 4
NUM_LAYER = 24

model = dict(
    num_chunks=1,  # if num_chunks > 1, interleaved pipeline scheduler is used.
    checkpoint=0.2,  # The proportion of layers for activation aheckpointing, the optional value are True/False/[0-1]
    dtype="torch.bfloat16",  # Support: "torch.float16", "torch.half", "torch.bfloat16", "torch.float32", "torch.tf32"
    embed_split_hidden=True,
    num_layers=NUM_LAYER,
    hidden_size=HIDDEN_SIZE,
    vocab_size=VOCAB_SIZE,
    embed_grad_scale=1,
    parallel_output=True,
    num_attention_heads=NUM_ATTENTION_HEAD,
    num_kv_attention_heads=NUM_KV_ATTENTION_HEAD,
    mlp_ratio=MLP_RATIO,
    multiple_of=MULTIPLE_OF,
    norm_type="rmsnorm",
    qk_interleaved=False,
    apply_post_layer_norm=False,
    no_bias=True,
    layer_norm_epsilon=1e-5,
    rope_base=1000000,
    norm_head=True,
)

hybrid_zero_optimizer = dict(
    # Enable low_level_optimzer overlap_communication
    overlap_sync_grad=True,
    overlap_sync_param=False,
    # bucket size for nccl communication params
    reduce_bucket_size=512 * 1024 * 1024,
    # grad clipping
    clip_grad_norm=1.0,
)

"""
zero1 parallel (dict):
    1. size: int
        * if size <= 0, the size of the zero process group is equal to the size of the dp process group,
            so parameters will be divided within the range of dp.
        * if size == 1, zero is not used, and all dp groups retain the full amount of model parameters.
        * if size > 1 and size <= dp world size, the world size of zero is a subset of dp world size.
        For smaller models, it is usually a better choice to split the parameters within nodes with a setting <= 8.
    2. fsdp: bool, enable/disable torch's fully sharded data parallel, defaults to False.
tensor parallel (dict):
    1. size: int, the size of tensor parallel.
    2. mode: str, the tensor parallel mode, should be in ['mtp', 'msp', 'fsp', 'isp'],
        defaults to 'mtp', means the pure megatron tensor parallel without sequence parallel.
        msp: megatron tensor parallel with sequence parallel, sequence parallel size = tensor parallel size.
        fsp: tensor parallel by flash-attn with sequence parallel, sequence parallel size = tensor parallel size.
        isp: customed intern sequence parallel without tensor parallel, can be used with weight parallel.
pipeline parallel (dict):
    1. size: int, the size of pipeline parallel.
    2. interleaved_overlap: bool, enable/disable communication overlap when using interleaved pipeline scheduler,
        defaults to False.
weight parallel (dict):
    1. size: int, the size of weight parallel.
    2. overlap: bool, enable/disable all_gather/reduce_scatter communication overlap, defaults to False.
"""
parallel = dict(
    zero1=dict(size=8),
    tensor=dict(size=1, mode="mtp"),
    pipeline=dict(size=1, interleaved_overlap=True),
    weight=dict(size=1, overlap=True),
)
