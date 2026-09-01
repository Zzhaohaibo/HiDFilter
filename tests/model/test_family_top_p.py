from __future__ import annotations

import numpy as np
import pytest
import torch

from hidfilter.filtration import family_top_p
from hidfilter.model import HiDFilter
from hidfilter.physical import PhysicalGraphContract, build_physical_candidates
from hidfilter.runtime.phase0 import build_optimizer
from hidfilter.semantic import SemanticCandidateMetadata


def _family_probability(values: list[float], *, requires_grad: bool = False) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32).view(1, 1, 1, 3).requires_grad_(
        requires_grad
    )


def _all_available() -> torch.Tensor:
    return torch.ones((1, 3), dtype=torch.bool)


def _three_family_model(num_nodes: int = 4) -> HiDFilter:
    adjacency = np.ones((num_nodes, num_nodes), dtype=np.float64)
    np.fill_diagonal(adjacency, 0.0)
    physical = build_physical_candidates(
        adjacency,
        PhysicalGraphContract("undirected", "affinity", None),
        kp=8,
    ).candidates
    source_slots = torch.zeros((num_nodes, 8), dtype=torch.int64)
    source_valid = torch.zeros((num_nodes, 8), dtype=torch.bool)
    for target in range(num_nodes):
        choices = [source for source in range(num_nodes) if source != target][:2]
        source_slots[target, : len(choices)] = torch.tensor(choices)
        source_valid[target, : len(choices)] = True
    source_index = source_slots.repeat_interleave(12, dim=1)
    lag_index = torch.arange(12, dtype=torch.int64).repeat(8).view(1, -1)
    lag_index = lag_index.expand(num_nodes, -1).clone()
    valid = source_valid.repeat_interleave(12, dim=1)
    semantic = SemanticCandidateMetadata(
        source_index=source_index,
        lag_index=lag_index,
        flat_index=source_index * 12 + lag_index,
        valid=valid,
        prior=valid.to(torch.float32),
    )
    return HiDFilter(
        num_nodes,
        physical_candidates=physical,
        semantic_candidates=semantic,
    )


@pytest.mark.parametrize(
    ("probability", "expected_keep", "expected_weight"),
    [
        ([0.50, 0.30, 0.20], [True, True, False], [0.625, 0.375, 0.0]),
        ([0.50, 0.29, 0.21], [True, True, True], [0.50, 0.29, 0.21]),
        ([0.90, 0.05, 0.05], [True, False, False], [1.0, 0.0, 0.0]),
    ],
)
def test_family_top_p_basic_crossing_and_single_support(
    probability, expected_keep, expected_weight
):
    keep, weight = family_top_p(
        _family_probability(probability),
        _all_available(),
        rho=0.8,
    )

    assert keep.reshape(-1).tolist() == expected_keep
    torch.testing.assert_close(
        weight.reshape(-1),
        torch.tensor(expected_weight),
        rtol=1.0e-6,
        atol=1.0e-7,
    )


def test_family_top_p_rho_one_retains_all_available_with_exact_weights():
    probability = _family_probability([0.40, 0.35, 0.25])

    keep, weight = family_top_p(probability, _all_available(), rho=1.0)

    assert keep.all()
    assert torch.equal(weight, probability)


def test_family_top_p_respects_availability_and_only_self_rows():
    probability = torch.tensor(
        [[[[0.60, 0.40, 0.00]]], [[[1.00, 0.00, 0.00]]]], dtype=torch.float32
    )
    available = torch.tensor([[True, True, False]])

    keep, weight = family_top_p(probability, available, rho=0.8)

    assert keep[0, 0, 0].tolist() == [True, True, False]
    assert keep[1, 0, 0].tolist() == [True, False, False]
    assert torch.equal(weight[0, 0, 0], torch.tensor([0.60, 0.40, 0.00]))
    assert torch.equal(weight[1, 0, 0], torch.tensor([1.00, 0.00, 0.00]))


def test_family_top_p_uses_canonical_tie_order():
    keep, weight = family_top_p(
        _family_probability([0.40, 0.30, 0.30]),
        _all_available(),
        rho=0.7,
    )

    assert keep.reshape(-1).tolist() == [True, True, False]
    assert weight.reshape(-1)[2].item() == 0.0


def test_family_top_p_uses_frozen_float32_crossing_tolerance():
    tolerance = 4.0 * (3 + 1) * torch.finfo(torch.float32).eps
    probability = _family_probability(
        [0.50, 0.30 - tolerance / 2.0, 0.20 + tolerance / 2.0]
    )

    keep, _ = family_top_p(probability, _all_available(), rho=0.8)

    assert keep.reshape(-1).tolist() == [True, True, False]


def test_family_top_p_exactly_renormalizes_and_keeps_weight_gradient_only():
    probability = _family_probability([0.50, 0.30, 0.20], requires_grad=True)
    keep, weight = family_top_p(probability, _all_available(), rho=0.8)

    assert keep.dtype == torch.bool
    assert not keep.requires_grad
    assert torch.equal(weight.sum(dim=-1), torch.ones_like(weight[..., 0]))
    loss = (weight * torch.tensor([[[[1.0, 3.0, 7.0]]]])).sum()
    loss.backward()

    assert probability.grad is not None
    assert torch.isfinite(probability.grad).all()
    assert probability.grad[..., :2].abs().sum() > 0
    assert probability.grad[..., 2].eq(0.0).all()


@pytest.mark.cuda
@pytest.mark.parametrize(
    "probability",
    [
        [0.40, 0.30, 0.30],
        [0.50, 0.29, 0.21],
        [0.50, 0.30 - 4.0e-7, 0.20 + 4.0e-7],
    ],
)
def test_family_top_p_cpu_cuda_parity(probability):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    cpu_probability = _family_probability(probability)

    cpu_keep, cpu_weight = family_top_p(cpu_probability, _all_available(), rho=0.8)
    cuda_keep, cuda_weight = family_top_p(
        cpu_probability.cuda(), _all_available().cuda(), rho=0.8
    )

    assert cpu_keep.equal(cuda_keep.cpu())
    torch.testing.assert_close(cpu_weight, cuda_weight.cpu(), rtol=1.0e-6, atol=1.0e-7)


def test_family_top_p_off_is_exact_phase4_forward_and_on_is_full_forward():
    torch.manual_seed(53)
    model = _three_family_model().eval()
    history = torch.randn(2, 12, 4, 1)

    phase4_prediction = model(history)
    off_prediction = model(history, family_top_p_enabled=False)
    on_prediction = model(history, family_top_p_enabled=True)

    assert torch.equal(off_prediction, phase4_prediction)
    assert on_prediction.shape == phase4_prediction.shape
    assert torch.isfinite(on_prediction).all()


def test_family_top_p_on_forward_backward_optimizer_smoke():
    torch.manual_seed(59)
    model = _three_family_model()
    optimizer = build_optimizer(model)
    history = torch.randn(2, 12, 4, 1)
    target = torch.randn(2, 12, 4, 1)

    prediction = model(history, family_top_p_enabled=True)
    loss = torch.nn.functional.l1_loss(prediction, target)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    assert torch.isfinite(prediction).all()
    for parameter in model.router_scorer.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_phase5_fixed_batch_learning_reduces_mae_with_family_top_p_on():
    torch.manual_seed(67)
    model = _three_family_model(num_nodes=3).eval()
    history = torch.randn(4, 12, 3, 1)
    target = history[:, -1:, :, :].expand(-1, 12, -1, -1) + 0.5
    optimizer = build_optimizer(model)

    with torch.no_grad():
        initial = torch.nn.functional.l1_loss(
            model(history, family_top_p_enabled=True), target
        ).item()
    for _ in range(60):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(history, family_top_p_enabled=True)
        loss = torch.nn.functional.l1_loss(prediction, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
    with torch.no_grad():
        final = torch.nn.functional.l1_loss(
            model(history, family_top_p_enabled=True), target
        ).item()

    assert final < initial * 0.5


@pytest.mark.cuda
def test_phase5_cuda_forward_backward_optimizer_smoke():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    torch.manual_seed(71)
    device = torch.device("cuda")
    model = _three_family_model().to(device)
    optimizer = build_optimizer(model)
    history = torch.randn(2, 12, 4, 1, device=device)
    target = torch.randn(2, 12, 4, 1, device=device)

    prediction = model(history, family_top_p_enabled=True)
    loss = torch.nn.functional.l1_loss(prediction, target)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    assert torch.isfinite(prediction).all()
