import copy

import torch
from torch import nn

from internlm.accelerator import AcceleratorType, get_accelerator
from internlm.solver.optimizer.npu_fused_adamw import AdamW as NPUAdamW
from internlm.utils.common import get_current_device

internlm_accelerator = get_accelerator()


def check_AdamW():
    class MlpModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(128, 256)
            self.linear2 = nn.Linear(256, 512)

        def forward(self, x):
            x = self.linear1(x)
            x = self.linear2(x)
            return x

    device = get_current_device()
    dtype = torch.bfloat16
    input_data = torch.rand(16, 128, dtype=dtype).to(device)
    torch_model = MlpModel().to(dtype).to(get_current_device())
    npu_model = copy.deepcopy(torch_model)

    adamW_torch = torch.optim.AdamW(
        params=torch_model.parameters(),
        lr=1e-4,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    adamW_npu = NPUAdamW(
        params=npu_model.parameters(),
        lr=1e-4,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    adamW_torch.zero_grad()
    adamW_npu.zero_grad()

    output_torch = torch_model(input_data)
    output_npu = npu_model(input_data)

    output_torch.mean().backward()
    output_npu.mean().backward()

    adamW_torch.step()
    adamW_npu.step()

    params_zip = zip(list(torch_model.parameters()), list(npu_model.parameters()))
    for torch_param, npu_param in params_zip:
        assert torch.allclose(torch_param, npu_param, rtol=1e-5, atol=1e-5)


def test_AdamW():
    if internlm_accelerator.get_accelerator_backend() == AcceleratorType.NPU:
        check_AdamW()
