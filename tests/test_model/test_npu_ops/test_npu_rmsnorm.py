import pytest
import torch

from internlm.accelerator import AcceleratorType, get_accelerator
from internlm.model.ops.norm import _RMSNorm as RMSNormTorch
from internlm.model.ops.norm import _RMSNormNPU as RMSNormNPU
from internlm.utils.common import get_current_device

internlm_accelerator = get_accelerator()


def check_RMSNormNPU():
    device = get_current_device()
    input_data = torch.randn(128).to(torch.float32).to(device)
    input_data_2 = input_data.clone().detach()

    rmsnorm_torch = RMSNormTorch(128, eps=1e-5).to(torch.bfloat16).to(device)
    output_torch = rmsnorm_torch(input_data)

    rmsnorm_npu = RMSNormNPU(128, eps=1e-5).to(torch.bfloat16).to(device)
    output_npu = rmsnorm_npu(input_data_2)

    if torch.equal(output_torch, output_npu):
        print("RMSNorm check passed: totaly equal", flush=True)
    else:
        max_diff, index_max_diff = (output_torch - output_npu).abs().max(dim=0)
        max_diff = max_diff.item()
        index_max_diff = index_max_diff.item()
        rtol = max_diff / abs(output_npu[index_max_diff])
        print(
            f"The relative error is {rtol}. Between {output_torch[index_max_diff]} and {output_npu[index_max_diff]}",
            flush=True,
        )
        assert rtol <= 1e-5, f"RMSNorm check failed: The relative error is {rtol}"
        print("RMSNorm check passed: allclose", flush=True)


def test_RMSNorm():
    if internlm_accelerator.get_accelerator_backend() == AcceleratorType.NPU:
        check_RMSNormNPU()


if __name__ == "__main__":
    pytest.main(["-s", "-q", "test_npu_ops.py"])
