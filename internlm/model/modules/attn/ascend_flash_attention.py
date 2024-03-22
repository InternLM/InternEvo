import math

import torch
from einops import rearrange

try:
    import torch_npu
except (ModuleNotFoundError, ImportError):
    print("python env don't have Ascend flash attention!")


class AscendFlashSelfAttention(torch.nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """

    def __init__(self, causal=False, softmax_scale=None, attention_dropout=0.0):
        super().__init__()
        assert rearrange is not None, "Please install einops first, e.g., with pip install einops"
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.shape_order = "BSND"
        self.next_tockens = 0  # 0
        self.dropout_p = attention_dropout

    def forward(self, q, k, v, attention_mask):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            q, k, v: The tensor containing the query, key, and value. (B, S, H, D)
        """
        pre_tockens = k.shape[0]  # seq_len
        head_num, head_dim = q.shape[2], q.shape[3]

        if self.shape_order == "BSH":
            q, k, v = [rearrange(x, "b s h d -> b s (h d)") for x in [q, k, v]]
        elif self.shape_order == "SBH":
            q, k, v = [rearrange(x, "b s h d -> s b (h d)") for x in [q, k, v]]
        elif self.shape_order != "BSND":
            raise ValueError("Invalid shape-order: {}, shape-order must be SBH or BSH or BSND".format(self.shape_order))

        try:
            scale = 1.0 / math.sqrt(head_dim) if self.softmax_scale is None else self.softmax_scale
        except Exception as e:
            raise ValueError("Invalid head_dim: {}".format(head_dim)) from e

        output = torch_npu.npu_fusion_attention(
            query=q,
            key=k,
            value=v,
            head_num=head_num,
            input_layout=self.shape_order,
            pse=None,
            padding_mask=None,  # resvered args, is not used for now.
            atten_mask=attention_mask,
            scale=scale,
            sparse_mode=1,  # Represents allMask, which means passing in the complete attendMaskOptional matrix.
            pre_tockens=pre_tockens,  # Use for sparse calculation, representing left boundary of the slides window
            next_tockens=self.next_tockens,
            keep_prob=1 - self.dropout_p,
            inner_precise=0,
        )[0]

        if self.shape_order == "BSH":
            output = rearrange(output, "b s (h d) -> b s h d", h=head_num)
        elif self.shape_order == "SBH":
            output = rearrange(output, "s b (h d) -> b s h d", h=head_num)
        elif self.shape_order != "BSND":
            raise ValueError("Invalid shape-order: {}, shape-order must be SBH or BSH or BSND".format(self.shape_order))

        return output
