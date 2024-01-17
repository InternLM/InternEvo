# Copyright (c) InternLM. All rights reserved.

model_type = "INTERNLM"

VOCAB_SIZE = 103168
HIDDEN_SIZE = 5120
NUM_ATTENTION_HEAD = 40
MLP_RATIO = 8 / 3
NUM_LAYER = 60

model = dict(
    num_chunks=1,
    checkpoint=False,
    dtype="torch.bfloat16",
    embed_split_hidden=True,
    num_layers=NUM_LAYER,
    hidden_size=HIDDEN_SIZE,
    vocab_size=VOCAB_SIZE,
    embed_grad_scale=1,
    parallel_output=True,
    num_attention_heads=NUM_ATTENTION_HEAD,
    mlp_ratio=MLP_RATIO,
    norm_type="rmsnorm",
    apply_post_layer_norm=False,
    layer_norm_epsilon=1e-5,
)

hybrid_zero_optimizer = dict(
    # Enable overlap_communication
    overlap_sync_grad=True,
    overlap_sync_param=False,
    # bucket size for nccl communication params
    reduce_bucket_size=512 * 1024 * 1024,
    # grad clipping
    clip_grad_norm=1.0,
)

# zero1 parallel:
#     1. if zero1 <= 0, The size of the zero process group is equal to the size of the dp process group,
#         so parameters will be divided within the range of dp.
#     2. if zero1 == 1, zero is not used, and all dp groups retain the full amount of model parameters.
#     3. zero1 > 1 and zero1 <= dp world size, the world size of zero is a subset of dp world size.
#         For smaller models, it is usually a better choice to split the parameters within nodes with a setting <= 8.
#     4. fsdp: bool, whether to use fsdp in pytorch, which can be a subsitution of ZeRO1.
# pipeline parallel (dict):
#     1. size: int, the size of pipeline parallel.
#     2. interleaved_overlap: bool, enable/disable communication overlap when using interleaved pipeline scheduler.
# tensor parallel: tensor parallel size, usually the number of GPUs per node.
parallel = dict(
    zero1=dict(size=8, fsdp=False),
    tensor=4,
    pipeline=dict(size=1, interleaved_overlap=True),
    sequence_parallel=False,
)
