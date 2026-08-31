from __future__ import annotations

import importlib
import importlib.util

import pytest
import torch


def _filtration_module():
    assert importlib.util.find_spec("hidfilter.filtration") is not None
    return importlib.import_module("hidfilter.filtration")


def test_safe_masked_softmax_contract():
    safe_masked_softmax = _filtration_module().safe_masked_softmax
    logits = torch.tensor([[1.0, 2.0, 3.0], [4.0, -2.0, 1.0], [0.0, 0.0, 0.0]])
    valid = torch.tensor([[True, True, True], [True, False, True], [False, False, False]])

    probability = safe_masked_softmax(logits, valid)

    assert probability.dtype == torch.float32
    torch.testing.assert_close(probability[0].sum(), torch.tensor(1.0))
    torch.testing.assert_close(probability[1].sum(), torch.tensor(1.0))
    assert probability[1, 1].item() == 0.0
    assert probability[2].equal(torch.zeros(3))


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), float("-inf")])
def test_safe_masked_softmax_rejects_nonfinite_valid_logits(nonfinite):
    safe_masked_softmax = _filtration_module().safe_masked_softmax
    logits = torch.tensor([[0.0, nonfinite]])

    with pytest.raises(RuntimeError, match="non-finite valid logits"):
        safe_masked_softmax(logits, torch.ones_like(logits, dtype=torch.bool))


def test_edge_top_p_crossing_tie_and_exact_renormalization():
    edge_top_p = _filtration_module().edge_top_p
    probability = torch.tensor([[0.50, 0.30, 0.20], [0.40, 0.30, 0.30]], requires_grad=True)
    valid = torch.ones_like(probability, dtype=torch.bool)

    keep, weight = edge_top_p(probability, valid, rho=0.8)

    assert keep.tolist() == [[True, True, False], [True, True, True]]
    torch.testing.assert_close(weight.sum(dim=-1), torch.ones(2), rtol=0.0, atol=0.0)
    assert weight[0, 2].item() == 0.0
    (weight * torch.tensor([[1.0, 2.0, 4.0], [1.0, 2.0, 4.0]])).sum().backward()
    assert probability.grad is not None
    assert torch.isfinite(probability.grad).all()


def test_edge_top_p_canonical_tie_rho_one_and_all_invalid():
    edge_top_p = _filtration_module().edge_top_p
    probability = torch.tensor(
        [[0.40, 0.30, 0.30], [0.40, 0.30, 0.20], [0.20, 0.30, 0.50]]
    )
    valid = torch.tensor(
        [[True, True, True], [True, True, True], [False, False, False]]
    )

    tie_keep, _ = edge_top_p(probability[:1], valid[:1], rho=0.7)
    full_keep, full_weight = edge_top_p(probability[1:2], valid[1:2], rho=1.0)
    empty_keep, empty_weight = edge_top_p(probability[2:], valid[2:], rho=0.8)

    assert tie_keep.tolist() == [[True, True, False]]
    assert full_keep.tolist() == [[True, True, True]]
    torch.testing.assert_close(full_weight.sum(), torch.tensor(1.0), rtol=0.0, atol=0.0)
    assert not empty_keep.any()
    assert empty_weight.equal(torch.zeros_like(empty_weight))


def test_self_top_p_can_retain_multiple_lags():
    edge_top_p = _filtration_module().edge_top_p
    probability = torch.tensor([[0.45, 0.40, 0.15]])

    keep, _ = edge_top_p(probability, torch.ones_like(probability, dtype=torch.bool), rho=0.8)

    assert keep.tolist() == [[True, True, False]]


@pytest.mark.cuda
def test_top_p_cpu_cuda_support_parity():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    edge_top_p = _filtration_module().edge_top_p
    probability = torch.tensor([[0.41, 0.24, 0.18, 0.10, 0.07]], dtype=torch.float32)
    valid = torch.tensor([[True, True, True, True, False]])

    cpu_keep, _ = edge_top_p(probability, valid, rho=0.8)
    cuda_keep, _ = edge_top_p(probability.cuda(), valid.cuda(), rho=0.8)

    assert cpu_keep.equal(cuda_keep.cpu())


@pytest.mark.cuda
def test_top_p_runs_under_formal_cuda_determinism():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    edge_top_p = _filtration_module().edge_top_p
    probability = torch.tensor([[0.43, 0.27, 0.17, 0.13]], device="cuda")
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        keep, weight = edge_top_p(
            probability,
            torch.ones_like(probability, dtype=torch.bool),
            rho=0.8,
        )
        torch.cuda.synchronize()
    finally:
        torch.use_deterministic_algorithms(previous)

    assert keep.tolist() == [[True, True, True, False]]
    torch.testing.assert_close(weight.sum(), torch.tensor(1.0, device="cuda"), rtol=0.0, atol=1e-6)
