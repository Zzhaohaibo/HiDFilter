from __future__ import annotations

import torch

from hidfilter.model import HiDFilter
from hidfilter.physical import PhysicalCandidateMetadata
from hidfilter.runtime.diagnostics import DiagnosticAccumulator, export_case_study
from hidfilter.runtime.phase0 import build_optimizer
from hidfilter.semantic import SemanticCandidateMetadata


def _candidate_metadata(
    num_nodes: int,
    source_offset: int,
) -> tuple[PhysicalCandidateMetadata, SemanticCandidateMetadata]:
    source_slots = torch.zeros((num_nodes, 8), dtype=torch.int64)
    source_valid = torch.zeros((num_nodes, 8), dtype=torch.bool)
    for target in range(num_nodes):
        source_slots[target, 0] = (target + source_offset) % num_nodes
        source_valid[target, 0] = True
    source_index = source_slots.repeat_interleave(12, dim=1)
    lag_index = torch.arange(12, dtype=torch.int64).repeat(8).view(1, -1)
    lag_index = lag_index.expand(num_nodes, -1).clone()
    valid = source_valid.repeat_interleave(12, dim=1)
    metadata = {
        "source_index": source_index,
        "lag_index": lag_index,
        "flat_index": source_index * 12 + lag_index,
        "valid": valid,
        "prior": valid.to(torch.float32),
    }
    return PhysicalCandidateMetadata(**metadata), SemanticCandidateMetadata(**metadata)


def _model() -> HiDFilter:
    physical, _ = _candidate_metadata(3, 1)
    _, semantic = _candidate_metadata(3, 2)
    return HiDFilter(3, physical_candidates=physical, semantic_candidates=semantic)


def test_diagnostic_forward_is_exactly_prediction_invariant() -> None:
    torch.manual_seed(71)
    model = _model().eval()
    history = torch.randn(2, 12, 3, 1)

    plain = model(history, family_top_p_enabled=True)
    observed = model.forward_with_diagnostics(history, family_top_p_enabled=True)

    assert torch.equal(observed.prediction, plain)
    assert observed.state.family_dense_probability.shape == (2, 3, 12, 3)
    assert observed.state.family_keep.shape == (2, 3, 12, 3)
    assert observed.state.family_retained_weight.shape == (2, 3, 12, 3)
    assert observed.state.family_available.shape == (3, 3)
    assert tuple(item.family_name for item in observed.state.fine) == (
        "Self",
        "Physical",
        "Semantic",
    )
    assert observed.state.fine[0].dense_probability.shape == (2, 3, 12, 12)
    assert observed.state.fine[1].dense_probability.shape == (2, 3, 12, 96)
    assert observed.state.fine[2].dense_probability.shape == (2, 3, 12, 96)


def test_diagnostic_forward_preserves_prediction_gradients() -> None:
    torch.manual_seed(72)
    model = _model().train()
    history = torch.randn(1, 12, 3, 1)

    output = model.forward_with_diagnostics(history, family_top_p_enabled=True)
    output.prediction.square().mean().backward()

    assert model.decoder[-1].weight.grad is not None
    assert torch.isfinite(model.decoder[-1].weight.grad).all()
    assert model.decoder[-1].weight.grad.abs().sum() > 0


def test_real_forward_state_supports_cpu_streaming_and_explicit_case_export() -> None:
    torch.manual_seed(73)
    model = _model().eval()
    history = torch.randn(2, 12, 3, 1)
    output = model.forward_with_diagnostics(history, family_top_p_enabled=True)
    valid_query = torch.ones((2, 3, 12), dtype=torch.bool)
    accumulator = DiagnosticAccumulator(device=torch.device("cpu"))
    accumulator.update(output.state, valid_query)

    report = accumulator.finalize()
    case = export_case_study(
        output.state,
        sample_index=0,
        target_sensor=1,
        horizon_index=2,
    )

    assert report["valid_query_count"] == 72
    assert sum(report["fine"]["overall"]["lag_histogram"]["counts"]) == sum(
        sum(report["fine"][family]["lag_histogram"]["counts"])
        for family in ("Self", "Physical", "Semantic")
    )
    assert case["sample_index"] == 0
    assert case["target_sensor"] == 1
    assert case["paper_horizon"] == 3
    assert all(item["family"] in {"Self", "Physical", "Semantic"} for item in case["retained_families"])


@torch.no_grad()
def _support_report(model: HiDFilter, history: torch.Tensor) -> dict[str, object]:
    output = model.forward_with_diagnostics(history, family_top_p_enabled=True)
    accumulator = DiagnosticAccumulator(device=history.device)
    accumulator.update(
        output.state,
        torch.ones((history.shape[0], 3, 12), dtype=torch.bool, device=history.device),
    )
    return accumulator.finalize()


def test_cuda_diagnostic_support_parity_and_optimizer_regression() -> None:
    if not torch.cuda.is_available():
        return
    torch.manual_seed(74)
    cpu_model = _model().eval()
    history = torch.randn(1, 12, 3, 1)
    cpu_report = _support_report(cpu_model, history)

    cuda_model = _model().cuda().eval()
    cuda_model.load_state_dict(cpu_model.state_dict(), strict=True)
    cuda_history = history.cuda()
    cuda_report = _support_report(cuda_model, cuda_history)
    assert cuda_report["family"]["active_family_count"] == cpu_report["family"][
        "active_family_count"
    ]
    for family in ("Self", "Physical", "Semantic", "overall"):
        assert cuda_report["fine"][family]["lag_histogram"]["counts"] == cpu_report[
            "fine"
        ][family]["lag_histogram"]["counts"]

    cuda_model.train()
    optimizer = build_optimizer(cuda_model)
    optimizer.zero_grad(set_to_none=True)
    prediction = cuda_model.forward_with_diagnostics(
        cuda_history, family_top_p_enabled=True
    ).prediction
    prediction.square().mean().backward()
    torch.nn.utils.clip_grad_norm_(cuda_model.parameters(), 5.0)
    optimizer.step()
    assert torch.isfinite(cuda_model.decoder[-1].weight).all()
