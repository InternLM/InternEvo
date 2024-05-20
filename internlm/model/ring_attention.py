import torch
import torch.nn as nn

from einops import rearrange

from ring_flash_attn import zigzag_ring_flash_attn_qkvpacked_func, zigzag_ring_flash_attn_varlen_qkvpacked_func, zigzag_ring_flash_attn_kvpacked_func, zigzag_ring_flash_attn_varlen_kvpacked_func , ring_flash_attn_kvpacked_func, ring_flash_attn_qkvpacked_func, ring_flash_attn_varlen_kvpacked_func, ring_flash_attn_varlen_qkvpacked_func

from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from internlm.utils.logger import get_logger

from .globals import PROCESS_GROUP


import torch.distributed as dist
from typing import Any, Optional, Tuple
from torch import Tensor, nn
from torch.nn import Module


logger = get_logger(__file__)

def split_seqlens(cu_seqlens, total_slices, each_seqlen, return_idx=None):
    cu_seqlens_splits = []
    current_split_start = 0
    
    for i in range(1, total_slices + 1):
        current_split_end = i * each_seqlen
        split_indices = (cu_seqlens >= current_split_start) & (cu_seqlens < current_split_end)
        current_split = cu_seqlens[split_indices]
        if len(current_split) == 0 or current_split[0] %  each_seqlen!= 0:
            current_split = torch.cat((torch.tensor([current_split_start], device=cu_seqlens.device), current_split))

        current_split = torch.cat((current_split, torch.tensor([current_split_end], device=cu_seqlens.device)))
        if i == total_slices:
            if current_split_end != cu_seqlens[-1]:
                current_split = torch.cat((current_split, cu_seqlens[-1:]))
        
        cu_seqlens_splits.append(current_split)
        current_split_start = current_split_end
    
    if return_idx is not None:
        new_cu_seqlens = cu_seqlens_splits[return_idx]
        new_cu_seqlens = new_cu_seqlens - each_seqlen * return_idx
        new_cu_seqlens = new_cu_seqlens.to(torch.int32)
        max_seqlen = max([new_cu_seqlens[i+1] - new_cu_seqlens[i] for i in range(len(new_cu_seqlens)-1)])
        
        return new_cu_seqlens, max_seqlen.item()

    split_cu_seqlens_and_max = []
    for idx in range(total_slices):
        new_cu_seqlens = cu_seqlens_splits[idx]
        new_cu_seqlens = new_cu_seqlens - each_seqlen * idx
        new_cu_seqlens = new_cu_seqlens.to(torch.int32)
        max_seqlen = max([new_cu_seqlens[i+1] - new_cu_seqlens[i] for i in range(len(new_cu_seqlens)-1)])
        split_cu_seqlens_and_max.append((new_cu_seqlens, max_seqlen.item()))
    return split_cu_seqlens_and_max



class MySeqAllToAll(torch.autograd.Function):
    "sequence alltoall"

    @staticmethod
    def forward(ctx: Any, group: dist.ProcessGroup, input_: Tensor, scatter_idx: int, gather_idx: int) -> Tensor:
        ctx.group = group
        ctx.scatter_idx = scatter_idx
        ctx.gather_idx = gather_idx

        if dist.get_world_size(group) <= 1:
            return input_

        seq_world_size = dist.get_world_size(group)

        input_list = [t.contiguous() for t in torch.tensor_split(input_, seq_world_size, scatter_idx)]
        output_list = [torch.empty_like(input_list[0]) for _ in range(seq_world_size)]
        # TODO Use all_to_all_single instead
        dist.all_to_all(output_list, input_list, group=group)
        return torch.cat(output_list, dim=gather_idx).contiguous()

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> Tuple[None, Tensor, None, None]:
        if dist.get_world_size(ctx.group) <= 1:
            return (None, *grad_output, None, None)

        return (None, MySeqAllToAll.apply(ctx.group, *grad_output, ctx.gather_idx, ctx.scatter_idx), None, None)


class SP2DFalshAttention(nn.Module):
    def __init__(self, causal=False, softmax_scale=None, attention_dropout=0.0):
        super().__init__()
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.drop = nn.Dropout(attention_dropout)
        rank = dist.get_rank()

        self.r_pg = PROCESS_GROUP.RING_PG
        self.u_pg = PROCESS_GROUP.ULYSSES_PG



    def forward(self, qkv, causal=None, cu_seqlens=None, max_seqlen=None):
        # torch.Size([16384, 3, 32, 128])
        if dist.get_rank()==0:
            print(f'BE::::::::::rank_id:{torch.distributed.get_rank()},qkv_shape:{qkv.shape},ring num:{self.r_pg},u num:{self.u_pg}',flush=True)
        
        qkv=MySeqAllToAll.apply(self.u_pg,qkv,3,1) #  torch.Size([32768, 3, 16, 128])
     
        if dist.get_rank()==0:
            print(f'After:::::::rank_id:{torch.distributed.get_rank()},qkv_shape:{qkv.shape}',flush=True)

        # torch.Size([1, 32768, 16, 128])
        causal=True

        # import pdb;pdb.set_trace()
        out= zigzag_ring_flash_attn_qkvpacked_func(qkv, self.drop.p if self.training else 0.0,
                                            softmax_scale=self.softmax_scale, causal=causal, group=self.r_pg)
        

        out = MySeqAllToAll.apply(
                self.u_pg, out, 1,2
            )
        
        return out
        


class RingFlashSelfAttention(nn.Module):
    def __init__(self, causal=False, softmax_scale=None, attention_dropout=0.0):
        super().__init__()
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.drop = nn.Dropout(attention_dropout)
        self.group = gpc.get_group(ParallelMode.TENSOR)
    

    def forward(self, qkv, causal=None, cu_seqlens=None, max_seqlen=None):
        assert qkv.dtype in [torch.float16, torch.bfloat16]
        assert qkv.is_cuda
        causal = self.causal if causal is None else causal
        unpadded = cu_seqlens is not None
        # import pdb;pdb.set_trace()

        ring_use_zigzag=gpc.config.ring_use_zigzag

        unpadded=False
        use2d=False

        if ring_use_zigzag:
            
            # qkv = qkv.unsqueeze(0)

            # (Pdb) torch.Size([1, 512, 3, 32, 128])
            return zigzag_ring_flash_attn_qkvpacked_func(qkv, self.drop.p if self.training else 0.0,
                                            softmax_scale=self.softmax_scale, causal=causal, group=self.group)

        if unpadded:
            seqlen = qkv.shape[0]
            rank = gpc.get_local_rank(ParallelMode.TENSOR)
            world_size = gpc.get_world_size(ParallelMode.TENSOR)
            cu_seqlens, max_seqlen = split_seqlens(cu_seqlens,world_size, seqlen, return_idx=rank)
            # logger.info(f"cuda memory profiling: max_allocated {torch.cuda.max_memory_allocated()}, allocated {torch.cuda.memory_allocated()}")
            assert cu_seqlens.dtype == torch.int32
            assert max_seqlen is not None
            assert isinstance(max_seqlen, int)
            if gpc.config.data.pack_sample_into_one:
                
                return zigzag_ring_flash_attn_varlen_qkvpacked_func(
                    qkv, cu_seqlens, max_seqlen, self.drop.p if self.training else 0.0,
                    softmax_scale=self.softmax_scale, causal=causal, group=self.group, window_size=(-1, -1), alibi_slopes=None,
                    deterministic=False,
                    return_attn_probs=False)
                # qkv = qkv.unsqueeze(0)
                # return zigzag_ring_flash_attn_qkvpacked_func(qkv, self.drop.p if self.training else 0.0,
                #                             softmax_scale=self.softmax_scale, causal=causal, group=self.group)
                
            else:
          
                return ring_flash_attn_varlen_qkvpacked_func(
                    qkv, cu_seqlens, max_seqlen, self.drop.p if self.training else 0.0,
                    softmax_scale=self.softmax_scale, causal=causal, group=self.group
                )

        else:
            if gpc.config.data.pack_sample_into_one:
                # qkv = qkv.unsqueeze(0)
                return zigzag_ring_flash_attn_qkvpacked_func(qkv, self.drop.p if self.training else 0.0,
                                            softmax_scale=self.softmax_scale, causal=causal, group=self.group)
            else:
                # qkv = qkv.unsqueeze(0)
                return ring_flash_attn_qkvpacked_func(qkv, self.drop.p if self.training else 0.0,
                                            softmax_scale=self.softmax_scale, causal=causal, group=self.group)



class RingFlashCrossAttention(nn.Module):
    def __init__(self, causal=False, softmax_scale=None, attention_dropout=0.0):
        super().__init__()
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.drop = nn.Dropout(attention_dropout)
        self.group = gpc.get_group(ParallelMode.TENSOR)


    def forward(self, q, kv, causal=None, cu_seqlens=None, max_seqlen=None,
                cu_seqlens_k=None, max_seqlen_k=None):
        assert q.dtype in [torch.float16, torch.bfloat16]
        assert q.is_cuda and kv.is_cuda
        causal = self.causal if causal is None else causal
        unpadded = cu_seqlens is not None

        if unpadded:
            seqlen = q.shape[0]
            rank = gpc.get_local_rank(ParallelMode.TENSOR)
            world_size = gpc.get_world_size(ParallelMode.TENSOR)
            cu_seqlens, max_seqlen = split_seqlens(cu_seqlens,world_size, seqlen, return_idx=rank)
            assert cu_seqlens.dtype == torch.int32
            assert max_seqlen is not None
            assert isinstance(max_seqlen, int)
            assert cu_seqlens_k is not None
            assert cu_seqlens_k.dtype == torch.int32
            assert max_seqlen_k is not None
            assert isinstance(max_seqlen, int)
            if gpc.config.data.pack_sample_into_one:
                return zigzag_ring_flash_attn_varlen_kvpacked_func(
                    q, kv, cu_seqlens, cu_seqlens_k, max_seqlen, max_seqlen_k,
                    self.drop.p if self.training else 0.0,
                    softmax_scale=self.softmax_scale, causal=causal, group=self.group
                )
            else:
                return ring_flash_attn_varlen_kvpacked_func(
                    q, kv,cu_seqlens, cu_seqlens_k, max_seqlen, max_seqlen_k,
                    self.drop.p if self.training else 0.0,
                    softmax_scale=self.softmax_scale, causal=causal, group=self.group
                )
        else:
            batch_size, seqlen_q = q.shape[0], q.shape[1]
            seqlen_k = kv.shape[1]
            assert kv.shape[0] == batch_size and kv.shape[4] == q.shape[3]
            if gpc.config.data.pack_sample_into_one:
                return zigzag_ring_flash_attn_kvpacked_func(q, kv, self.drop.p if self.training else 0.0,
                                            causal=causal, softmax_scale=self.softmax_scale, group=self.group)
            else:
                return ring_flash_attn_kvpacked_func(q, kv, self.drop.p if self.training else 0.0,
                                            causal=causal, softmax_scale=self.softmax_scale, group=self.group)


