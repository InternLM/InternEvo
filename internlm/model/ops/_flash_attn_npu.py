# Copyright (c) 2024, Huawei Technologies.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

from internlm.accelerator import get_accelerator
from internlm.core.context import global_context as gpc
from internlm.core.parallel.comm import get_offload_npu_manager
try:
    import mindspeed
    from mindspeed.op_builder import FusionAttentionV2OpBuilder

    npu_mindspeed_fa_impl = True
except (ModuleNotFoundError, ImportError):
    npu_mindspeed_fa_impl = False
    
internlm_accelerator = get_accelerator()
device_backend = internlm_accelerator.get_accelerator_backend()

__all__ = ["npu_fusion_attention"]


class FusionAttentionV2Function(torch.autograd.Function):

    @staticmethod
    def forward(ctx, query, key, value, head_num, input_layout, pse, padding_mask, atten_mask, scale, keep_prob,
                pre_tokens, next_tokens, inner_precise, prefix, actual_seq_qlen, actual_seq_kvlen, sparse_mode,
                gen_mask_parallel, sync, pse_type, q_start_idx, kv_start_idx, layer_idx):
        mindspeed_ops = FusionAttentionV2OpBuilder().load()

        _ckpt_block_num = int(gpc.config.model.checkpoint * gpc.config.isp_num_layers)
        _is_ckpt_layer = gpc.config.cpu_offloading.num_layers <= layer_idx < _ckpt_block_num

        if gpc.is_forward is False and gpc.config.selective_checkpoint and _is_ckpt_layer: 
            outputs = get_offload_npu_manager().get_fa_output_with_layer(layer_idx)
            # breakpoint()
            # attention_in, softmax_max, softmax_sum, softmax_in, seed, offset, numels
        else:    
            outputs = mindspeed_ops.npu_fusion_attention_v2(query, key, value, head_num,
                                                                                  input_layout, pse,
                                                                                  padding_mask, atten_mask,
                                                                                  scale, keep_prob, pre_tokens,
                                                                                  next_tokens, inner_precise, prefix,
                                                                                  actual_seq_qlen, actual_seq_kvlen,
                                                                                  sparse_mode, gen_mask_parallel,
                                                                                  sync, pse_type, q_start_idx,
                                                                                  kv_start_idx)
        attention_in, softmax_max, softmax_sum, softmax_in, seed, offset, numels = outputs
        # store attn forward output to avoid re-computation of attn when activation checkpoint is enabled
        if gpc.is_forward and gpc.config.selective_checkpoint and _is_ckpt_layer:
            # breakpoint()
            get_offload_npu_manager().insert_fa_output_with_layer(
                layer_idx=layer_idx, output=(attention_in, softmax_max, softmax_sum, softmax_in, seed, offset, numels)
            )
        ctx.save_for_backward(query, key, value, pse, padding_mask, atten_mask, attention_in,
                              softmax_max, softmax_sum, softmax_in)
        ctx.scale = scale
        ctx.input_layout = input_layout
        ctx.head_num = head_num
        ctx.pre_tokens = pre_tokens
        ctx.next_tokens = next_tokens
        ctx.inner_precise = inner_precise
        ctx.gen_mask_parallel = gen_mask_parallel
        ctx.sync = sync
        ctx.seed = seed
        ctx.offset = offset
        ctx.numels = numels
        ctx.prefix = prefix
        ctx.keep_prob = keep_prob
        ctx.actual_seq_qlen = actual_seq_qlen
        ctx.actual_seq_kvlen = actual_seq_kvlen
        ctx.sparse_mode = sparse_mode
        ctx.pse_type = pse_type
        ctx.q_start_idx = q_start_idx
        ctx.kv_start_idx = kv_start_idx

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs, dq=None, dk=None, dv=None, seed=0, offset=0, numels=0):
        mindspeed_ops = FusionAttentionV2OpBuilder().load()
        query, key, value, pse, padding_mask, atten_mask, attention_in, softmax_max, \
        softmax_sum, softmax_in = ctx.saved_tensors
        results = mindspeed_ops.npu_fusion_attention_grad_v2(
            query, key, value, grad_outputs, ctx.head_num, ctx.input_layout, pse, padding_mask, atten_mask,
            softmax_max, softmax_sum, softmax_in, attention_in, ctx.scale, ctx.keep_prob, ctx.pre_tokens,
            ctx.next_tokens, ctx.inner_precise, ctx.seed, ctx.offset, ctx.numels, ctx.prefix, ctx.actual_seq_qlen,
            ctx.actual_seq_kvlen, ctx.sparse_mode, ctx.gen_mask_parallel, ctx.sync, ctx.pse_type, ctx.q_start_idx,
            ctx.kv_start_idx)

        return results[0], results[1], results[2], None, None, results[3], None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None # 22


def npu_fusion_attention(query, key, value, head_num,
                         input_layout, *, pse=None,
                         padding_mask=None, atten_mask=None,
                         scale=1., keep_prob=1., pre_tokens=2147483647,
                         next_tokens=2147483647, inner_precise=0, prefix=None,
                         actual_seq_qlen=None, actual_seq_kvlen=None,
                         sparse_mode=0, gen_mask_parallel=True,
                         sync=False, pse_type=1, q_start_idx=None,
                         kv_start_idx=None, layer_idx=0):
    return FusionAttentionV2Function.apply(query, key, value, head_num,
                                           input_layout, pse,
                                           padding_mask, atten_mask,
                                           scale, keep_prob, pre_tokens,
                                           next_tokens, inner_precise, prefix,
                                           actual_seq_qlen, actual_seq_kvlen,
                                           sparse_mode, gen_mask_parallel,
                                           sync, pse_type, q_start_idx,
                                           kv_start_idx, layer_idx)


def npu_fusion_attention_grad(query, key, value, grad_outputs,
                              head_num, input_layout, *, pse=None,
                              padding_mask=None, atten_mask=None,
                              softmax_max=None, softmax_sum=None, softmax_in=None, attention_in=None,
                              scale=1., keep_prob=1., pre_tokens=2147483647,
                              next_tokens=2147483647, inner_precise=0,
                              seed=1234, offset=0, numels=0, prefix=None,
                              actual_seq_qlen=None, actual_seq_kvlen=None, sparse_mode=0,
                              gen_mask_parallel=True, sync=False, pse_type=1, q_start_idx=None,
                              kv_start_idx=None):
    mindspeed_ops = FusionAttentionV2OpBuilder().load()
    return mindspeed_ops.npu_fusion_attention_grad_v2(query, key, value, grad_outputs, head_num, input_layout, pse,
                                                      padding_mask, atten_mask, softmax_max, softmax_sum, softmax_in,
                                                      attention_in, scale, keep_prob, pre_tokens, next_tokens,
                                                      inner_precise, seed, offset, numels, prefix, actual_seq_qlen,
                                                      actual_seq_kvlen, sparse_mode, gen_mask_parallel, sync,
                                                      pse_type, q_start_idx, kv_start_idx)
