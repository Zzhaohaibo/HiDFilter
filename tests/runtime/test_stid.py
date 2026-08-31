from __future__ import annotations

import pytest
import torch

from hidfilter.runtime.stid import build_traffic_only_stid, stid_forward


def test_stid_cpu_protocol_and_train_step():
    model = build_traffic_only_stid(num_nodes=5)
    history = torch.randn(4, 12, 5, 1)
    target = torch.randn(4, 12, 5, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    prediction = stid_forward(model, history)
    loss = torch.nn.functional.l1_loss(prediction, target)
    loss.backward()
    optimizer.step()

    assert prediction.shape == target.shape
    assert torch.isfinite(loss)


@pytest.mark.cuda
def test_stid_cuda_train_step():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    device = torch.device("cuda:0")
    model = build_traffic_only_stid(num_nodes=170).to(device)
    history = torch.randn(2, 12, 170, 1, device=device)
    target = torch.randn(2, 12, 170, 1, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    optimizer.zero_grad(set_to_none=True)
    prediction = stid_forward(model, history)
    loss = torch.nn.functional.l1_loss(prediction, target)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()

    assert prediction.shape == target.shape
    assert torch.isfinite(loss)
