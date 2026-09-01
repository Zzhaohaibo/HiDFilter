from __future__ import annotations

import numpy as np
import pytest
import torch

from hidfilter.model import (
    FAMILY_COUNT,
    FORECAST_HORIZON,
    HIDDEN_DIM,
    IDENTITY_DIM,
    PHYSICAL_FAMILY_ID,
    ROUTER_HIDDEN_DIM,
    ROUTER_INPUT_DIM,
    SELF_FAMILY_ID,
    SEMANTIC_FAMILY_ID,
    HiDFilter,
    compute_family_evidence,
    compute_router_probability,
    dense_router_fusion,
    masked_router_softmax,
)
from hidfilter.physical import PhysicalGraphContract, build_physical_candidates
from hidfilter.runtime.phase0 import build_optimizer
from hidfilter.semantic import SemanticCandidateMetadata


def _physical_candidates(num_nodes: int = 4):
    adjacency = np.ones((num_nodes, num_nodes), dtype=np.float64)
    np.fill_diagonal(adjacency, 0.0)
    return build_physical_candidates(
        adjacency,
        PhysicalGraphContract("undirected", "affinity", None),
        kp=8,
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
            source_prior[target, : len(choices)] = torch.tensor(
                [0.45, 0.85][: len(choices)]
            )
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


def _three_family_model(num_nodes: int = 4) -> HiDFilter:
    return HiDFilter(
        num_nodes,
        physical_candidates=_physical_candidates(num_nodes),
        semantic_candidates=_semantic_candidates(num_nodes),
    )


def test_family_evidence_is_unweighted_source_mean_and_empty_is_exact_zero():
    lag_mean = torch.arange(4, dtype=torch.float32).view(1, 4, 1).expand(-1, -1, HIDDEN_DIM)
    source_index = torch.tensor(
        [[1, 2, 0], [0, 3, 0], [0, 0, 0], [2, 1, 0]], dtype=torch.int64
    )
    source_valid = torch.tensor(
        [[True, True, False], [True, True, False], [False, False, False], [True, False, False]]
    )

    evidence = compute_family_evidence(lag_mean, source_index, source_valid)
    permuted = compute_family_evidence(
        lag_mean,
        source_index[:, [2, 0, 1]],
        source_valid[:, [2, 0, 1]],
    )

    assert evidence.shape == (1, 4, HIDDEN_DIM)
    assert evidence[:, 0].eq(1.5).all()
    assert evidence[:, 1].eq(1.5).all()
    assert evidence[:, 2].eq(0.0).all()
    assert evidence[:, 3].eq(2.0).all()
    assert torch.equal(evidence, permuted)


def test_model_family_evidence_is_identity_and_prior_free():
    torch.manual_seed(41)
    model = _three_family_model()
    lag_content = torch.randn(2, 4, 12, HIDDEN_DIM)
    before = model.encode_family_evidence(lag_content)

    with torch.no_grad():
        model.node_embedding.weight.normal_()
        model.horizon_embedding.weight.normal_()
        model.fine_family_embedding.weight.normal_()
        model.route_family_embedding.weight.normal_()
        model.physical_prior.uniform_(0.01, 1.0)
        model.semantic_prior.uniform_(0.01, 1.0)
    after = model.encode_family_evidence(lag_content)

    assert torch.equal(before, after)
    assert torch.equal(before[:, :, SELF_FAMILY_ID], lag_content.float().mean(dim=2))


def test_model_physical_and_semantic_evidence_use_each_source_once():
    physical = _physical_candidates()
    semantic = _semantic_candidates()
    model = HiDFilter(
        4,
        physical_candidates=physical,
        semantic_candidates=semantic,
    )
    sensor_values = torch.arange(4, dtype=torch.float32).view(1, 4, 1, 1)
    lag_content = sensor_values.expand(1, 4, 12, HIDDEN_DIM)

    evidence = model.encode_family_evidence(lag_content)

    assert torch.equal(
        model.physical_evidence_source_index,
        physical.source_index.reshape(4, 8, 12)[:, :, 0],
    )
    assert torch.equal(
        model.semantic_evidence_source_index,
        semantic.source_index.reshape(4, 8, 12)[:, :, 0],
    )
    assert evidence[:, 0, PHYSICAL_FAMILY_ID].eq(2.0).all()
    assert evidence[:, 1, PHYSICAL_FAMILY_ID].eq(5.0 / 3.0).all()
    assert evidence[:, 0, SEMANTIC_FAMILY_ID].eq(1.5).all()
    assert evidence[:, 3, SEMANTIC_FAMILY_ID].eq(0.5).all()


def test_router_architecture_and_family_order_are_frozen():
    model = _three_family_model()

    assert (SELF_FAMILY_ID, PHYSICAL_FAMILY_ID, SEMANTIC_FAMILY_ID) == (0, 1, 2)
    assert FAMILY_COUNT == 3
    assert model.router_scorer[0].in_features == ROUTER_INPUT_DIM == 176
    assert model.router_scorer[0].out_features == ROUTER_HIDDEN_DIM == 64
    assert model.router_scorer[2].in_features == ROUTER_HIDDEN_DIM
    assert model.router_scorer[2].out_features == 1
    assert model.route_family_embedding.weight.shape == (FAMILY_COUNT, IDENTITY_DIM)
    assert (
        model.route_family_embedding.weight.data_ptr()
        != model.fine_family_embedding.weight.data_ptr()
    )
    router_modules = [name for name, _ in model.named_modules() if name == "router_scorer"]
    assert router_modules == ["router_scorer"]


def test_router_softmax_masks_unavailable_families_and_rejects_valid_nonfinite():
    logits = torch.tensor(
        [[[[0.0, 1.0, 2.0]], [[2.0, 1.0, float("nan")]], [[3.0, 8.0, -4.0]]]]
    )
    available = torch.tensor(
        [[True, True, True], [True, True, False], [True, False, False]]
    )
    probability = masked_router_softmax(logits, available)

    assert torch.isfinite(probability).all()
    assert torch.allclose(probability.sum(dim=-1), torch.ones_like(probability[..., 0]))
    assert probability[:, 1, :, SEMANTIC_FAMILY_ID].eq(0.0).all()
    assert probability[:, 2, :, PHYSICAL_FAMILY_ID:].eq(0.0).all()
    assert probability[:, 2, :, SELF_FAMILY_ID].eq(1.0).all()

    for nonfinite in (float("nan"), float("inf"), float("-inf")):
        invalid_logits = logits.clone()
        invalid_logits[0, 0, 0, SELF_FAMILY_ID] = nonfinite
        with pytest.raises(RuntimeError, match="non-finite valid logits"):
            masked_router_softmax(invalid_logits, available)


def test_router_uses_only_allowed_inputs_and_gradients_reach_all_router_inputs():
    torch.manual_seed(17)
    model = _three_family_model()
    context = torch.randn(2, 4, HIDDEN_DIM, requires_grad=True)
    evidence = torch.randn(2, 4, FAMILY_COUNT, HIDDEN_DIM, requires_grad=True)
    node_identity = torch.randn(4, IDENTITY_DIM, requires_grad=True)
    horizon_identity = torch.randn(FORECAST_HORIZON, IDENTITY_DIM, requires_grad=True)
    route_identity = model.route_family_embedding.weight
    available = torch.ones(4, FAMILY_COUNT, dtype=torch.bool)

    probability = compute_router_probability(
        context,
        evidence,
        node_identity,
        horizon_identity,
        route_identity,
        available,
        model.router_scorer,
    )
    fine_before = probability.detach().clone()
    with torch.no_grad():
        model.fine_family_embedding.weight.normal_(mean=50.0, std=10.0)
    fine_after = compute_router_probability(
        context,
        evidence,
        node_identity,
        horizon_identity,
        route_identity,
        available,
        model.router_scorer,
    )
    assert torch.equal(fine_before, fine_after.detach())

    family_messages = tuple(
        torch.full((2, 4, FORECAST_HORIZON, HIDDEN_DIM), value)
        for value in (1.0, 2.0, 4.0)
    )
    fused = dense_router_fusion(*family_messages, probability)
    fused.square().mean().backward()

    for tensor in (context, evidence, node_identity, horizon_identity, route_identity):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
        assert tensor.grad.abs().sum() > 0
    for parameter in model.router_scorer.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_memory_efficient_shared_router_matches_explicit_176_input_formula():
    torch.manual_seed(19)
    model = _three_family_model()
    context = torch.randn(2, 4, HIDDEN_DIM)
    evidence = torch.randn(2, 4, FAMILY_COUNT, HIDDEN_DIM)
    node_identity = model.node_embedding.weight
    horizon_identity = model.horizon_embedding.weight
    route_identity = model.route_family_embedding.weight
    available = torch.tensor(
        [[True, True, True], [True, True, False], [True, False, True], [True, False, False]]
    )

    efficient = compute_router_probability(
        context,
        evidence,
        node_identity,
        horizon_identity,
        route_identity,
        available,
        model.router_scorer,
    )
    batch_size = context.shape[0]
    explicit_input = torch.cat(
        (
            context[:, :, None, None, :].expand(-1, -1, FORECAST_HORIZON, FAMILY_COUNT, -1),
            node_identity[None, :, None, None, :].expand(
                batch_size, -1, FORECAST_HORIZON, FAMILY_COUNT, -1
            ),
            horizon_identity[None, None, :, None, :].expand(
                batch_size, 4, -1, FAMILY_COUNT, -1
            ),
            evidence[:, :, None, :, :].expand(-1, -1, FORECAST_HORIZON, -1, -1),
            route_identity[None, None, None, :, :].expand(
                batch_size, 4, FORECAST_HORIZON, -1, -1
            ),
        ),
        dim=-1,
    )
    explicit_logits = model.router_scorer(explicit_input).squeeze(-1)
    explicit = masked_router_softmax(explicit_logits, available)

    assert torch.allclose(efficient, explicit, rtol=1.0e-6, atol=1.0e-7)


def test_dense_router_fusion_matches_weighted_family_sum():
    self_message = torch.full((1, 2, 3, 4), 2.0)
    physical_message = torch.full_like(self_message, 5.0)
    semantic_message = torch.full_like(self_message, 11.0)
    probability = torch.tensor(
        [[[[0.2, 0.3, 0.5]] * 3, [[1.0, 0.0, 0.0]] * 3]], dtype=torch.float32
    )

    fused = dense_router_fusion(
        self_message,
        physical_message,
        semantic_message,
        probability,
    )

    assert fused[:, 0].eq(7.4).all()
    assert torch.equal(fused[:, 1], self_message[:, 1])


def test_full_phase4_forward_backward_optimizer_and_router_probability():
    torch.manual_seed(23)
    model = _three_family_model()
    optimizer = build_optimizer(model)
    call_count = {"wq": 0, "wv": 0}

    def count_call(name):
        def hook(_module, _inputs, _output):
            call_count[name] += 1

        return hook

    handles = [
        model.wq.register_forward_hook(count_call("wq")),
        model.wv.register_forward_hook(count_call("wv")),
    ]
    history = torch.randn(2, 12, 4, 1)
    target = torch.randn(2, 12, 4, 1)
    prediction = model(history)
    for handle in handles:
        handle.remove()
    probability = model.router_probability(history)

    assert prediction.shape == target.shape
    assert probability.shape == (2, 4, FORECAST_HORIZON, FAMILY_COUNT)
    assert torch.isfinite(prediction).all()
    assert torch.isfinite(probability).all()
    assert torch.allclose(probability.sum(dim=-1), torch.ones_like(probability[..., 0]))
    assert call_count == {"wq": 1, "wv": 1}

    loss = (prediction - target).square().mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    for parameter in (
        model.context_encoder.network[0].weight,
        model.lag_content_encoder.mlp[0].weight,
        model.node_embedding.weight,
        model.horizon_embedding.weight,
        model.fine_family_embedding.weight,
        model.route_family_embedding.weight,
        model.router_scorer[0].weight,
        model.wq.weight,
        model.wv.weight,
        model.decoder[0].weight,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_phase4_fixed_batch_learning_reduces_mae():
    torch.manual_seed(43)
    model = _three_family_model(num_nodes=3).eval()
    history = torch.randn(4, 12, 3, 1)
    target = history[:, -1:, :, :].expand(-1, 12, -1, -1) + 0.5
    optimizer = build_optimizer(model)
    route_before = model.route_family_embedding.weight.detach().clone()
    scorer_before = model.router_scorer[0].weight.detach().clone()

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
    assert not torch.equal(route_before, model.route_family_embedding.weight)
    assert not torch.equal(scorer_before, model.router_scorer[0].weight)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_phase4_cuda_forward_backward_optimizer_smoke():
    torch.manual_seed(29)
    device = torch.device("cuda")
    model = _three_family_model().to(device)
    optimizer = build_optimizer(model)
    history = torch.randn(2, 12, 4, 1, device=device)
    target = torch.randn(2, 12, 4, 1, device=device)

    prediction = model(history)
    probability = model.router_probability(history)
    loss = (prediction - target).square().mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    assert torch.isfinite(prediction).all()
    assert torch.isfinite(probability).all()
    assert torch.allclose(probability.sum(dim=-1), torch.ones_like(probability[..., 0]))
