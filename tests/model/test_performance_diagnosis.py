from __future__ import annotations

import importlib
import importlib.util

import pytest
import torch

from hidfilter.model import HIDDEN_DIM, HiDFilter, compute_fine_family
from hidfilter.physical import PhysicalCandidateMetadata
from hidfilter.runtime.diagnostics import DiagnosticAccumulator
from hidfilter.semantic import SemanticCandidateMetadata


def _diagnosis_module():
    if importlib.util.find_spec("hidfilter.runtime.performance_diagnosis") is None:
        pytest.fail("performance diagnosis runtime policy is not implemented")
    return importlib.import_module("hidfilter.runtime.performance_diagnosis")


def _candidate_metadata(
    metadata_type: type[PhysicalCandidateMetadata] | type[SemanticCandidateMetadata],
    num_nodes: int,
    source_offset: int,
):
    source_slots = torch.empty((num_nodes, 8), dtype=torch.int64)
    for target in range(num_nodes):
        source_slots[target] = torch.tensor(
            [(target + source_offset + rank) % num_nodes for rank in range(8)]
        )
    source_index = source_slots.repeat_interleave(12, dim=1)
    lag_index = torch.arange(12, dtype=torch.int64).repeat(8).view(1, -1)
    lag_index = lag_index.expand(num_nodes, -1).clone()
    valid = torch.ones((num_nodes, 96), dtype=torch.bool)
    source_prior = torch.linspace(0.15, 0.95, 8).view(1, -1).expand(num_nodes, -1)
    prior = source_prior.repeat_interleave(12, dim=1).contiguous()
    return metadata_type(
        source_index=source_index,
        lag_index=lag_index,
        flat_index=source_index * 12 + lag_index,
        valid=valid,
        prior=prior,
    )


def _model(num_nodes: int = 4) -> HiDFilter:
    return HiDFilter(
        num_nodes,
        physical_candidates=_candidate_metadata(
            PhysicalCandidateMetadata, num_nodes, 1
        ),
        semantic_candidates=_candidate_metadata(
            SemanticCandidateMetadata, num_nodes, 9
        ),
    )


def test_default_full_path_is_exactly_explicit_edge_top_p_on() -> None:
    torch.manual_seed(801)
    model = _model().eval()
    history = torch.randn(2, 12, 4, 1)

    default = model(history, family_top_p_enabled=True)
    explicit = model(
        history,
        family_top_p_enabled=True,
        edge_top_p_enabled=True,
    )

    assert torch.equal(default, explicit)


def test_edge_top_p_off_is_exact_dense_probability_message() -> None:
    torch.manual_seed(802)
    query = torch.randn(1, 2, 3, HIDDEN_DIM)
    key_global = torch.randn(1, 2, 12, HIDDEN_DIM)
    value_global = torch.randn(1, 2, 12, HIDDEN_DIM)
    flat_index = torch.tensor([[0, 1, 2, 3], [12, 13, 14, 15]])
    valid = torch.tensor([[True, True, False, True], [True, False, True, True]])
    prior = torch.tensor([[1.0, 0.8, 0.0, 0.4], [0.9, 0.0, 0.7, 0.5]])

    output = compute_fine_family(
        query,
        key_global,
        value_global,
        flat_index,
        valid,
        prior,
        torch.tensor(1.0),
        edge_top_p_enabled=False,
    )

    expected_keep = valid.view(1, 2, 1, 4).expand_as(output.edge_keep)
    candidate_value = value_global.reshape(1, 24, HIDDEN_DIM)[:, flat_index]
    expected_message = torch.einsum(
        "bnhc,bncd->bnhd", output.dense_probability, candidate_value
    )
    assert torch.equal(output.edge_keep, expected_keep)
    assert torch.equal(output.edge_weight, output.dense_probability)
    assert torch.equal(output.message, expected_message)


def test_design_audit_additive_fine_family_identity_cancels() -> None:
    torch.manual_seed(803)
    model = HiDFilter(2).eval()
    query = torch.randn(1, 2, 4, HIDDEN_DIM)
    fine_tokens = torch.randn(1, 2, 12, HIDDEN_DIM)
    value_global = model.wv(fine_tokens)
    first_identity = model.fine_family_projection(model.fine_family_embedding.weight[0])
    second_identity = first_identity + 10.0 * torch.randn_like(first_identity)
    first_key = model.wk(fine_tokens + first_identity.view(1, 1, 1, -1))
    second_key = model.wk(fine_tokens + second_identity.view(1, 1, 1, -1))

    first = compute_fine_family(
        query,
        first_key,
        value_global,
        model.self_flat_index,
        model.self_valid,
        model.self_prior,
        model.alpha,
    )
    second = compute_fine_family(
        query,
        second_key,
        value_global,
        model.self_flat_index,
        model.self_valid,
        model.self_prior,
        model.alpha,
    )

    torch.testing.assert_close(
        second.dense_probability,
        first.dense_probability,
        rtol=0.0,
        atol=2.0e-6,
    )
    assert torch.equal(second.edge_keep, first.edge_keep)
    torch.testing.assert_close(second.message, first.message, rtol=0.0, atol=2.0e-6)


def test_design_audit_fine_family_embedding_gradient_is_negligible() -> None:
    torch.manual_seed(804)
    model = _model().eval()
    history = torch.randn(2, 12, 4, 1)
    target = torch.randn(2, 12, 4, 1)

    prediction = model(history, family_top_p_enabled=True)
    torch.nn.functional.l1_loss(prediction, target).backward()

    fine_gradient = model.fine_family_embedding.weight.grad
    assert fine_gradient is not None
    assert torch.linalg.vector_norm(fine_gradient).item() <= 1.0e-8
    for module in (
        model.context_encoder,
        model.lag_content_encoder,
        model.wq,
        model.wk,
        model.wv,
        model.router_scorer,
        model.decoder,
    ):
        gradients = [
            parameter.grad
            for parameter in module.parameters()
            if parameter.grad is not None
        ]
        assert gradients
        assert sum(gradient.abs().sum().item() for gradient in gradients) > 1.0e-8


@pytest.mark.parametrize(
    ("variant", "expected_family_top_p", "expected_edge_top_p"),
    [
        ("no_family_top_p", False, True),
        ("no_edge_top_p", True, False),
        ("no_top_p", False, False),
    ],
)
def test_variant_forward_policies_control_only_the_two_support_operators(
    variant: str,
    expected_family_top_p: bool,
    expected_edge_top_p: bool,
) -> None:
    torch.manual_seed(805)
    module = _diagnosis_module()
    policy = module.diagnosis_epoch_policy(variant, 6)
    model = _model().eval()
    history = torch.randn(1, 12, 4, 1)

    output = model.forward_with_diagnostics(
        history,
        family_top_p_enabled=policy.family_top_p_enabled,
        edge_top_p_enabled=policy.edge_top_p_enabled,
    )

    assert policy.family_top_p_enabled is expected_family_top_p
    assert policy.edge_top_p_enabled is expected_edge_top_p
    available = model.family_available.view(1, 4, 1, 3).expand_as(
        output.state.family_keep
    )
    if not expected_family_top_p:
        assert torch.equal(output.state.family_keep, available)
    if not expected_edge_top_p:
        for fine in output.state.fine:
            expected = fine.candidate_valid.view(1, 4, 1, -1).expand_as(
                fine.edge_keep
            )
            assert torch.equal(fine.edge_keep, expected)


def test_no_top_p_diagnostics_have_three_families_and_204_effective_edges() -> None:
    torch.manual_seed(806)
    module = _diagnosis_module()
    policy = module.diagnosis_epoch_policy("no_top_p", 6)
    model = _model(num_nodes=20).eval()
    history = torch.randn(1, 12, 20, 1)
    output = model.forward_with_diagnostics(
        history,
        family_top_p_enabled=policy.family_top_p_enabled,
        edge_top_p_enabled=policy.edge_top_p_enabled,
    )
    accumulator = DiagnosticAccumulator(device=torch.device("cpu"))
    accumulator.update(
        output.state,
        torch.ones((1, 20, 12), dtype=torch.bool),
    )

    report = accumulator.finalize()
    assert report["family"]["active_family_count"] == {
        "mean": 3.0,
        "minimum": 3,
        "maximum": 3,
        "fractions": {"1": 0.0, "2": 0.0, "3": 1.0},
    }
    assert report["fine"]["overall"]["effective_edge_count"] == {
        "mean": 204.0,
        "minimum": 204,
        "maximum": 204,
    }


@pytest.mark.cuda
@pytest.mark.parametrize(
    "variant", ("no_family_top_p", "no_edge_top_p", "no_top_p")
)
def test_diagnosis_variants_cuda_forward_backward_smoke(variant: str) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    torch.manual_seed(807)
    module = _diagnosis_module()
    policy = module.diagnosis_epoch_policy(variant, 6)
    model = _model().cuda()
    history = torch.randn(1, 12, 4, 1, device="cuda")
    prediction = model(
        history,
        family_top_p_enabled=policy.family_top_p_enabled,
        edge_top_p_enabled=policy.edge_top_p_enabled,
    )
    prediction.square().mean().backward()
    torch.cuda.synchronize()

    assert torch.isfinite(prediction).all()
    assert model.decoder[-1].weight.grad is not None
