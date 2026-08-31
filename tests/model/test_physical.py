from __future__ import annotations

import numpy as np
import pytest
import torch

import hidfilter.physical as physical_module
from hidfilter.filtration import edge_top_p
from hidfilter.model import (
    HiDFilter,
    build_self_candidate_metadata,
    compute_fine_family,
    equal_average_family_messages,
)
from hidfilter.physical import PhysicalGraphContract, build_physical_candidates
from hidfilter.runtime.determinism import configure_determinism
from hidfilter.runtime.phase0 import build_optimizer


def _physical_candidates(num_nodes: int = 4):
    adjacency = np.ones((num_nodes, num_nodes), dtype=np.float64)
    np.fill_diagonal(adjacency, 0.0)
    adjacency[0, 1] = adjacency[1, 0] = 0.25
    contract = PhysicalGraphContract("undirected", "affinity", None)
    return build_physical_candidates(adjacency, contract, kp=8).candidates


def test_self_and_physical_have_independent_probability_rows_and_padding_is_zero():
    physical = _physical_candidates(num_nodes=3)
    self_metadata = build_self_candidate_metadata(num_nodes=3)
    query = torch.zeros(1, 3, 2, 4)
    key_global = torch.zeros(1, 3, 12, 4)
    value_global = torch.randn(1, 3, 12, 4)
    alpha = torch.tensor(1.0)

    self_output = compute_fine_family(
        query,
        key_global,
        value_global,
        self_metadata.flat_index,
        self_metadata.valid,
        self_metadata.prior,
        alpha,
    )
    physical_output = compute_fine_family(
        query,
        key_global,
        value_global,
        physical.flat_index,
        physical.valid,
        physical.prior,
        alpha,
    )

    torch.testing.assert_close(self_output.dense_probability.sum(-1), torch.ones(1, 3, 2))
    torch.testing.assert_close(physical_output.dense_probability.sum(-1), torch.ones(1, 3, 2))
    expanded_valid = physical.valid.view(1, 3, 1, 96).expand_as(physical_output.edge_weight)
    assert physical_output.dense_probability.masked_select(~expanded_valid).eq(0).all()
    assert physical_output.edge_weight.masked_select(~expanded_valid).eq(0).all()
    assert physical_output.edge_keep[0, 0, 0, :12].sum().item() >= 2


def test_all_invalid_physical_family_has_exact_zero_message():
    query = torch.randn(2, 3, 12, 64)
    key_global = torch.randn(2, 3, 12, 64)
    value_global = torch.randn(2, 3, 12, 64)
    flat_index = torch.zeros(3, 96, dtype=torch.int64)
    valid = torch.zeros(3, 96, dtype=torch.bool)
    prior = torch.zeros(3, 96)

    output = compute_fine_family(
        query, key_global, value_global, flat_index, valid, prior, torch.tensor(1.0)
    )

    assert output.dense_probability.eq(0).all()
    assert output.edge_weight.eq(0).all()
    assert output.message.eq(0).all()


def test_equal_average_fusion_uses_only_available_families():
    self_message = torch.full((2, 3, 12, 4), 2.0)
    physical_message = torch.full_like(self_message, 6.0)
    available = torch.tensor([True, False, True])

    fused = equal_average_family_messages(self_message, physical_message, available)

    assert fused[:, 0].eq(4.0).all()
    assert fused[:, 1].eq(2.0).all()
    assert fused[:, 2].eq(4.0).all()


def test_phase2_forward_shares_qkv_value_projection_decoder_and_reaches_alpha():
    torch.manual_seed(19)
    model = HiDFilter(num_nodes=4, physical_candidates=_physical_candidates()).train()
    history = torch.randn(3, 12, 4, 1)
    target = torch.randn(3, 12, 4, 1)
    wv_calls = 0

    def count_wv_calls(_module, _inputs, _output):
        nonlocal wv_calls
        wv_calls += 1

    handle = model.wv.register_forward_hook(count_wv_calls)
    prediction = model(history)
    handle.remove()
    loss = torch.nn.functional.l1_loss(prediction, target)
    loss.backward()

    assert prediction.shape == (3, 12, 4, 1)
    assert torch.isfinite(prediction).all()
    assert wv_calls == 1
    linear_names = {name for name, child in model.named_modules() if isinstance(child, torch.nn.Linear)}
    assert {"wq", "wk", "wv"}.issubset(linear_names)
    assert not any(name.startswith(("physical_w", "physical_decoder")) for name in linear_names)
    family_gradient = model.fine_family_embedding.weight.grad
    assert family_gradient is not None
    assert family_gradient[0].abs().sum() > 0
    assert family_gradient[1].abs().sum() > 0
    assert family_gradient[2].eq(0).all()
    assert model.alpha_raw.grad is not None
    assert torch.isfinite(model.alpha_raw.grad)
    assert model.alpha_raw.grad.abs() > 0
    for shared in (model.wq, model.wk, model.wv, model.node_embedding, model.decoder):
        gradients = [parameter.grad for parameter in shared.parameters()]
        assert gradients and all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)


def test_unavailable_physical_path_matches_self_only_and_forward_never_builds_graph(monkeypatch):
    invalid = _physical_candidates(num_nodes=3)
    invalid = type(invalid)(
        source_index=torch.zeros_like(invalid.source_index),
        lag_index=invalid.lag_index,
        flat_index=torch.zeros_like(invalid.flat_index),
        valid=torch.zeros_like(invalid.valid),
        prior=torch.zeros_like(invalid.prior),
    )
    self_only = HiDFilter(num_nodes=3).eval()
    phase2 = HiDFilter(num_nodes=3, physical_candidates=invalid).eval()
    phase2.load_state_dict(self_only.state_dict(), strict=True)
    history = torch.randn(2, 12, 3, 1)

    monkeypatch.setattr(
        physical_module,
        "build_physical_candidates",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("online graph build")),
    )

    with torch.no_grad():
        expected = self_only(history)
        actual = phase2(history)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_phase2_fixed_batch_learning_reduces_mae():
    torch.manual_seed(23)
    model = HiDFilter(num_nodes=3, physical_candidates=_physical_candidates(3)).eval()
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
def test_phase2_cuda_forward_backward_optimizer_and_c96_top_p_parity():
    configure_determinism(29)
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    torch.manual_seed(29)
    candidates = _physical_candidates(num_nodes=4)
    model = HiDFilter(num_nodes=4, physical_candidates=candidates).cuda().train()
    optimizer = build_optimizer(model)
    history = torch.randn(2, 12, 4, 1, device="cuda")
    target = torch.randn(2, 12, 4, 1, device="cuda")

    prediction = model(history)
    loss = torch.nn.functional.l1_loss(prediction, target)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optimizer.step()

    assert torch.isfinite(prediction).all()
    assert torch.isfinite(loss)
    assert model.alpha_raw.grad is not None and torch.isfinite(model.alpha_raw.grad)

    probability = torch.rand(2, 4, 12, 96, dtype=torch.float32)
    valid = candidates.valid.view(1, 4, 1, 96).expand_as(probability)
    probability = probability * valid
    probability = probability / probability.sum(dim=-1, keepdim=True)
    cpu_keep, cpu_weight = edge_top_p(probability, valid, rho=0.8)
    cuda_keep, cuda_weight = edge_top_p(probability.cuda(), valid.cuda(), rho=0.8)

    assert torch.equal(cuda_keep.cpu(), cpu_keep)
    torch.testing.assert_close(cuda_weight.cpu(), cpu_weight, rtol=3.0e-7, atol=1.0e-8)
    torch.testing.assert_close(
        cuda_weight.sum(dim=-1), torch.ones_like(cuda_weight.sum(dim=-1)), rtol=0.0, atol=1.0e-6
    )
