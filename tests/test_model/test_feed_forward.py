import pytest
import torch

from internlm.model.modules.mlp import new_feed_forward, split_fused_mlp_weight
from internlm.utils.common import get_current_device

SEQ_LEN = 64
HIDDEN_SIZE = 128
MLP_RATIO = 8 / 3


mlp_args = {
    "in_features": HIDDEN_SIZE,
    "hidden_features": int(HIDDEN_SIZE * MLP_RATIO),
    "out_features": HIDDEN_SIZE,
    "bias": False,
    "device": get_current_device(),
    "dtype": torch.bfloat16,
}


def check_param(a1, a2, b1, b2):
    for key in a1.keys():
        assert torch.equal(a1[key], a2[key])
        assert torch.equal(a1[key], b1[key])
        assert torch.equal(b1[key], b2[key])


def init_mlp():
    mlp_no_fused = new_feed_forward(**mlp_args)
    mlp_fused = new_feed_forward(mlp_layer_fusion=True, **mlp_args)

    for _, param in mlp_fused.named_parameters():
        torch.nn.init.normal_(param.data, std=0.02)

    w1, w3 = split_fused_mlp_weight(mlp_fused.fused_w1_w3.weight)
    mlp_no_fused.w1.weight.data = w1.data
    mlp_no_fused.w3.weight.data = w3.data
    mlp_no_fused.w2.weight.data = mlp_fused.w2.weight.data

    return mlp_no_fused, mlp_fused


@pytest.mark.mlp_layer_fusion_convert
def test_mlp_layer_fusion_convert():
    mlp_no_fused, mlp_fused = init_mlp()

    no_fused_ckpt, fused_ckpt = mlp_no_fused.state_dict(), mlp_fused.state_dict()

    # fuse load fuse
    mlp_fused.load_state_dict(fused_ckpt)

    # nofuse load nofuse
    mlp_no_fused.load_state_dict(no_fused_ckpt)

    a1, b1 = mlp_no_fused.state_dict(), mlp_fused.state_dict()

    # fuse load nofuse
    mlp_fused.load_state_dict(no_fused_ckpt)

    # nofuse load fuse
    mlp_no_fused.load_state_dict(fused_ckpt)

    a2, b2 = mlp_no_fused.state_dict(), mlp_fused.state_dict()

    check_param(a1, a2, b1, b2)


@pytest.mark.mlp_layer_fusion_loss
def test_mlp_layer_fusion_loss():
    mlp_no_fused, mlp_fused = init_mlp()
    x1_intput = torch.rand((2, SEQ_LEN, HIDDEN_SIZE), device=get_current_device(), dtype=torch.bfloat16)
    torch.nn.init.normal_(x1_intput.data, std=0.02)
    x2_input = x1_intput.clone()

    a1 = mlp_no_fused(x1_intput)
    a2 = mlp_fused(x2_input)

    assert torch.allclose(a1, a2, rtol=1e-4, atol=1e-5)

    l1, l2 = a1.sum(), a2.sum()
    l1.backward()
    l2.backward()

    assert torch.allclose(mlp_no_fused.w2.weight.grad, mlp_fused.w2.weight.grad, rtol=1e-4, atol=1e-5)
    w1_g, w3_g = split_fused_mlp_weight(mlp_fused.fused_w1_w3.weight.grad)
    assert torch.allclose(mlp_no_fused.w1.weight.grad, w1_g, rtol=1e-4, atol=1e-5)
    assert torch.allclose(mlp_no_fused.w3.weight.grad, w3_g, rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    pytest.main(["-s", "-q", "test_feed_forward.py"])
