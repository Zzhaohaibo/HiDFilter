from __future__ import annotations

import numpy as np
import pytest
import torch

from hidfilter.filtration import edge_top_p
from hidfilter.model import (
    HiDFilter,
    build_self_candidate_metadata,
    compute_fine_family,
    equal_average_three_family_messages,
)
from hidfilter.physical import PhysicalGraphContract, build_physical_candidates
from hidfilter.runtime.determinism import configure_determinism
from hidfilter.runtime.phase0 import build_optimizer
from hidfilter.semantic import SemanticCandidateMetadata


def _physical_candidates(num_nodes: int = 4):
    adjacency = np.ones((num_nodes, num_nodes), dtype=np.float64)
    np.fill_diagonal(adjacency, 0.0)
    return build_physical_candidates(
        adjacency, PhysicalGraphContract("undirected", "affinity", None), kp=8
    ).candidates


def _semantic_candidates(num_nodes: int = 4, *, available: bool = True):
    source_slots = torch.zeros((num_nodes, 8), dtype=torch.int64)
    source_valid = torch.zeros((num_nodes, 8), dtype=torch.bool)
    source_prior = torch.zeros((num_nodes, 8), dtype=torch.float32)
    if available:
        for target in range(num_nodes):
            choices = [source for source in range(num_nodes) if source != target][:2]
            source_slots[target, : len(choices)] = torch.tensor(choices)
            source_valid[target, : len(choices)] = True
            source_prior[target, : len(choices)] = torch.tensor([0.45, 0.85][: len(choices)])
    source_index = source_slots.repeat_interleave(12, dim=1)
    lag_index = torch.arange(12, dtype=torch.int64).repeat(8).view(1, -1)
    lag_index = lag_index.expand(num_nodes, -1).clone()
    valid = source_valid.repeat_interleave(12, dim=1)
    prior = source_prior.repeat_interleave(12, dim=1)
    return SemanticCandidateMetadata(
        source_index=source_index,
        lag_index=lag_index,
        flat_index=source_index * 12 + lag_index,
        valid=valid,
        prior=prior,
    )


def test_three_family_temporary_fusion_averages_only_available_messages():
    self_message = torch.full((1, 4, 2, 3), 3.0)
    physical_message = torch.full_like(self_message, 6.0)
    semantic_message = torch.full_like(self_message, 12.0)

    fused = equal_average_three_family_messages(
        self_message,
        physical_message,
        torch.tensor([True, True, False, False]),
        semantic_message,
        torch.tensor([True, False, True, False]),
    )

    assert fused[:, 0].eq(7.0).all()
    assert fused[:, 1].eq(4.5).all()
    assert fused[:, 2].eq(7.5).all()
    assert fused[:, 3].eq(3.0).all()


def test_self_physical_semantic_probability_rows_are_independently_normalized():
    physical = _physical_candidates(4)
    semantic = _semantic_candidates(4)
    self_metadata = build_self_candidate_metadata(4)
    query = torch.zeros(1, 4, 2, 64)
    key_global = torch.zeros(1, 4, 12, 64)
    value_global = torch.randn(1, 4, 12, 64)

    outputs = [
        compute_fine_family(
            query,
            key_global,
            value_global,
            metadata.flat_index,
            metadata.valid,
            metadata.prior,
            torch.tensor(1.0),
        )
        for metadata in (self_metadata, physical, semantic)
    ]

    for output in outputs:
        torch.testing.assert_close(output.dense_probability.sum(-1), torch.ones(1, 4, 2))
    semantic_valid = semantic.valid.view(1, 4, 1, 96).expand_as(outputs[2].edge_weight)
    assert outputs[2].dense_probability.masked_select(~semantic_valid).eq(0).all()
    assert outputs[2].edge_weight.masked_select(~semantic_valid).eq(0).all()
    assert outputs[2].edge_keep[0, 0, 0, :12].sum().item() >= 2


def test_phase3_forward_shares_qkv_reaches_all_family_rows_and_global_alpha():
    torch.manual_seed(31)
    model = HiDFilter(
        4,
        physical_candidates=_physical_candidates(4),
        semantic_candidates=_semantic_candidates(4),
    ).train()
    history = torch.randn(3, 12, 4, 1)
    target = torch.randn(3, 12, 4, 1)
    wq_calls = 0
    wv_calls = 0

    def count_wq_calls(_module, _inputs, _output):
        nonlocal wq_calls
        wq_calls += 1

    def count_wv_calls(_module, _inputs, _output):
        nonlocal wv_calls
        wv_calls += 1

    wq_handle = model.wq.register_forward_hook(count_wq_calls)
    wv_handle = model.wv.register_forward_hook(count_wv_calls)
    prediction = model(history)
    wq_handle.remove()
    wv_handle.remove()
    loss = torch.nn.functional.l1_loss(prediction, target)
    loss.backward()

    assert prediction.shape == (3, 12, 4, 1)
    assert torch.isfinite(prediction).all()
    assert wq_calls == 1
    assert wv_calls == 1
    linear_names = {name for name, child in model.named_modules() if isinstance(child, torch.nn.Linear)}
    assert {"wq", "wk", "wv"}.issubset(linear_names)
    assert not any(name.startswith(("semantic_w", "semantic_decoder")) for name in linear_names)
    family_gradient = model.fine_family_embedding.weight.grad
    assert family_gradient is not None
    assert all(family_gradient[row].abs().sum() > 0 for row in range(3))
    assert model.alpha_raw.grad is not None
    assert torch.isfinite(model.alpha_raw.grad)
    assert model.alpha_raw.grad.abs() > 0


def test_unavailable_semantic_path_is_exact_phase2_and_message_is_zero():
    phase2 = HiDFilter(4, physical_candidates=_physical_candidates(4)).eval()
    phase3_empty = HiDFilter(
        4,
        physical_candidates=_physical_candidates(4),
        semantic_candidates=_semantic_candidates(4, available=False),
    ).eval()
    phase3_empty.load_state_dict(phase2.state_dict(), strict=True)
    history = torch.randn(2, 12, 4, 1)

    with torch.no_grad():
        expected = phase2(history)
        actual = phase3_empty(history)
        query = torch.randn(2, 4, 12, 64)
        global_values = torch.randn(2, 4, 12, 64)
        empty = _semantic_candidates(4, available=False)
        output = compute_fine_family(
            query,
            global_values,
            global_values,
            empty.flat_index,
            empty.valid,
            empty.prior,
            torch.tensor(1.0),
        )

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    assert output.dense_probability.eq(0).all()
    assert output.edge_weight.eq(0).all()
    assert output.message.eq(0).all()


def test_phase3_fixed_batch_learning_reduces_mae():
    torch.manual_seed(37)
    model = HiDFilter(
        3,
        physical_candidates=_physical_candidates(3),
        semantic_candidates=_semantic_candidates(3),
    ).eval()
    history = torch.randn(4, 12, 3, 1)
    target = history[:, -1:, :, :].expand(-1, 12, -1, -1) + 0.5
    optimizer = build_optimizer(model)

    with torch.no_grad():
        initial = torch.nn.functional.l1_loss(model(history), target).item()
    for _ in range(60):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.l1_loss(model(history), target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
    with torch.no_grad():
        final = torch.nn.functional.l1_loss(model(history), target).item()

    assert final < initial * 0.5


@pytest.mark.cuda
def test_phase3_cuda_forward_backward_optimizer_and_semantic_top_p_parity():
    configure_determinism(41)
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    candidates = _semantic_candidates(4)
    model = HiDFilter(
        4,
        physical_candidates=_physical_candidates(4),
        semantic_candidates=candidates,
    ).cuda().train()
    optimizer = build_optimizer(model)
    history = torch.randn(2, 12, 4, 1, device="cuda")
    target = torch.randn(2, 12, 4, 1, device="cuda")

    prediction = model(history)
    loss = torch.nn.functional.l1_loss(prediction, target)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optimizer.step()

    assert torch.isfinite(prediction).all()
    assert model.alpha_raw.grad is not None and torch.isfinite(model.alpha_raw.grad)
    probability = torch.rand(2, 4, 12, 96, dtype=torch.float32)
    valid = candidates.valid.view(1, 4, 1, 96).expand_as(probability)
    probability = probability * valid
    probability = probability / probability.sum(dim=-1, keepdim=True)
    cpu_keep, cpu_weight = edge_top_p(probability, valid, rho=0.8)
    cuda_keep, cuda_weight = edge_top_p(probability.cuda(), valid.cuda(), rho=0.8)
    assert torch.equal(cuda_keep.cpu(), cpu_keep)
    torch.testing.assert_close(cuda_weight.cpu(), cpu_weight, rtol=3.0e-7, atol=1.0e-8)
