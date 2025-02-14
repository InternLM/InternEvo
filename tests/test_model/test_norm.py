import multiprocessing as mp

import pytest
import torch

from internlm.model.modules.norm import new_layer_norm
from internlm.utils.common import get_current_device
from tests.common_fixture import find_free_port
from tests.test_model.test_model_internlm import build_environment, seed_all


def check_norm(args):
    # init
    rank, world_size, free_port = args
    build_environment(rank, world_size, free_port)
    device = get_current_device()
    rtol, atol = (1e-3, 5e-3)
    hidden_size = 4
    layer_norm_epsilon = 1e-05

    # fix seed
    seed_all(1024)

    # define norm
    norm = new_layer_norm(norm_type="rmsnorm", normalized_shape=hidden_size, eps=layer_norm_epsilon)
    norm = norm.to(device)

    # create input
    hidden_states = torch.tensor(
        [
            [8.3726, 1.9245, 5.5101, 1.0000],
            [3.3474, 2.9582, 1.0000, 1.0000],
            [8.3726, 1.2875, 5.5101, 1.0000],
            [8.3726, 1.2875, 5.5101, 1.0000],
        ],
        requires_grad=True,
    ).to(device)

    # forward
    output_list = []
    for _ in range(10):
        result = norm(hidden_states.float())
        output_list.append(result)

    # check only forward logits
    first_output = output_list[0]
    for i in range(1, 10):
        assert torch.equal(first_output, output_list[i])

    standard = torch.tensor(
        [
            [1.6329, 0.3753, 1.0746, 0.1950],
            [1.4288, 1.2626, 0.4268, 0.4268],
            [1.6490, 0.2536, 1.0852, 0.1970],
            [1.6490, 0.2536, 1.0852, 0.1970],
        ]
    ).to(device)

    # check output
    assert torch.allclose(result, standard, rtol=rtol, atol=atol, equal_nan=True)

    hidden_states.retain_grad()
    loss = torch.randn_like(result)

    # backward
    result.backward(loss)
    grad = hidden_states.grad

    standard_grad = torch.tensor(
        [
            [-0.0193, 0.1248, 0.0324, -0.2573],
            [-0.2140, 0.2010, 0.2901, -0.1683],
            [-0.0815, -0.0689, 0.0850, 0.3027],
            [0.0847, 0.1739, -0.1554, -0.0773],
        ]
    ).to(device)

    # check grad
    assert torch.allclose(grad, standard_grad, rtol=rtol, atol=atol, equal_nan=True)


@pytest.mark.norm
def test_norm():
    ctx = mp.get_context("spawn")
    free_port = str(find_free_port())
    with ctx.Pool(processes=8) as pool:
        pool.map(check_norm, [[rank, 8, free_port] for rank in range(8)])
        pool.close()
        pool.join()


if __name__ == "__main__":
    pytest.main(["-s", "-q", "test_norm.py"])
