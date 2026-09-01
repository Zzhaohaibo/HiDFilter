from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch
from basicts.scaler import ZScoreScaler
from torch import nn
from torch.utils.data import DataLoader

from hidfilter.model import (
    FineDiagnosticState,
    HiDFilter,
    HiDFilterDiagnosticState,
)
from hidfilter.runtime.checkpoint import CheckpointManager, load_model_checkpoint
from hidfilter.runtime.diagnostics import (
    DiagnosticAccumulator,
    effective_edge_support,
    evaluate_with_diagnostics,
)
from hidfilter.runtime.formal import (
    FormalTestOnceGuard,
    WarmupEarlyStopping,
    formal_config_fingerprint,
    formal_learning_rate,
    set_formal_learning_rate,
)
from hidfilter.runtime.family_top_p import phase5_epoch_policy
from hidfilter.runtime.phase0 import build_optimizer


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "formal_pems08.json"
RUNNER_PATH = ROOT / "scripts" / "run_formal_pems08.py"
BENCHMARK_PATH = ROOT / "scripts" / "benchmark_phase6_diagnostics.py"


def _load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fine_state(
    family_name: str,
    source: int,
    kept_lags_by_horizon: dict[int, tuple[int, ...]],
) -> FineDiagnosticState:
    dense = torch.full((1, 1, 12, 12), 1.0 / 12.0)
    edge_keep = torch.zeros((1, 1, 12, 12), dtype=torch.bool)
    for horizon, lags in kept_lags_by_horizon.items():
        edge_keep[0, 0, horizon, list(lags)] = True
    retained_count = edge_keep.sum(dim=-1, keepdim=True).clamp_min(1)
    edge_weight = edge_keep.to(torch.float32) / retained_count
    return FineDiagnosticState(
        family_name=family_name,
        candidate_valid=torch.ones((1, 12), dtype=torch.bool),
        dense_probability=dense,
        edge_keep=edge_keep,
        edge_retained_weight=edge_weight,
        source_index=torch.full((1, 12), source, dtype=torch.int64),
        lag_index=torch.arange(12, dtype=torch.int64).view(1, 12),
    )


def _diagnostic_state(device: torch.device = torch.device("cpu")) -> HiDFilterDiagnosticState:
    dense = torch.zeros((1, 1, 12, 3))
    dense[..., 0] = 1.0
    dense[0, 0, 0] = torch.tensor([0.5, 0.3, 0.2])
    dense[0, 0, 1] = torch.tensor([0.1, 0.2, 0.7])
    keep = torch.zeros_like(dense, dtype=torch.bool)
    keep[..., 0] = True
    keep[0, 0, 0] = torch.tensor([True, True, False])
    keep[0, 0, 1] = torch.tensor([True, False, True])
    weight = torch.zeros_like(dense)
    weight[..., 0] = 1.0
    weight[0, 0, 0] = torch.tensor([0.625, 0.375, 0.0])
    weight[0, 0, 1] = torch.tensor([0.125, 0.0, 0.875])
    fine = (
        _fine_state("Self", 0, {0: (0, 1), 1: (2,)}),
        _fine_state("Physical", 1, {0: (0,), 1: (3,)}),
        _fine_state("Semantic", 2, {0: (4,), 1: (2, 3)}),
    )
    return HiDFilterDiagnosticState(
        family_dense_probability=dense.to(device),
        family_keep=keep.to(device),
        family_retained_weight=weight.to(device),
        family_available=torch.ones((1, 3), dtype=torch.bool, device=device),
        fine=tuple(
            FineDiagnosticState(
                family_name=item.family_name,
                candidate_valid=item.candidate_valid.to(device),
                dense_probability=item.dense_probability.to(device),
                edge_keep=item.edge_keep.to(device),
                edge_retained_weight=item.edge_retained_weight.to(device),
                source_index=item.source_index.to(device),
                lag_index=item.lag_index.to(device),
            )
            for item in fine
        ),
    )


def _summary(device: torch.device = torch.device("cpu")) -> dict[str, object]:
    accumulator = DiagnosticAccumulator(device=device)
    valid_query = torch.zeros((1, 1, 12), dtype=torch.bool, device=device)
    valid_query[..., :2] = True
    accumulator.update(_diagnostic_state(device), valid_query)
    return accumulator.finalize()


def test_effective_edge_is_exact_three_mask_conjunction() -> None:
    valid = torch.tensor([[True, True, False, True]])
    edge_keep = torch.tensor([[[[True, False, True, True]]]])
    family_keep = torch.tensor([[[True]]])
    effective = effective_edge_support(valid, edge_keep, family_keep)
    assert torch.equal(effective, torch.tensor([[[[True, False, False, True]]]]))

    dropped = effective_edge_support(valid, edge_keep, torch.tensor([[[False]]]))
    assert not dropped.any()


def test_family_support_retention_and_per_horizon_summary() -> None:
    report = _summary()
    family = report["family"]

    assert family["retention_frequency"] == {
        "Self": 1.0,
        "Physical": 0.5,
        "Semantic": 0.5,
    }
    assert family["active_family_count"] == {
        "mean": 2.0,
        "minimum": 2,
        "maximum": 2,
        "fractions": {"1": 0.0, "2": 1.0, "3": 0.0},
    }
    assert family["per_horizon"][0]["retention_frequency"] == {
        "Self": 1.0,
        "Physical": 1.0,
        "Semantic": 0.0,
    }
    assert family["per_horizon"][1]["dense_probability_mean"] == pytest.approx(
        {"Self": 0.1, "Physical": 0.2, "Semantic": 0.7}
    )
    assert family["per_horizon"][2]["valid_query_count"] == 0


def test_family_support_distribution_covers_variable_cardinality() -> None:
    state = _diagnostic_state()
    state.family_keep[0, 0, 0] = torch.tensor([True, False, False])
    state.family_retained_weight[0, 0, 0] = torch.tensor([1.0, 0.0, 0.0])
    state.family_keep[0, 0, 1] = torch.tensor([True, True, False])
    state.family_retained_weight[0, 0, 1] = torch.tensor([0.5, 0.5, 0.0])
    state.family_keep[0, 0, 2] = torch.tensor([True, True, True])
    state.family_retained_weight[0, 0, 2] = torch.tensor([1.0, 0.0, 0.0])
    accumulator = DiagnosticAccumulator(device=torch.device("cpu"))
    valid_query = torch.zeros((1, 1, 12), dtype=torch.bool)
    valid_query[..., :3] = True
    accumulator.update(state, valid_query)

    distribution = accumulator.finalize()["family"]["active_family_count"]
    assert distribution["minimum"] == 1
    assert distribution["maximum"] == 3
    assert distribution["fractions"] == pytest.approx(
        {"1": 1 / 3, "2": 1 / 3, "3": 1 / 3}
    )


def test_edge_effective_lag_unique_source_and_multi_lag_statistics() -> None:
    report = _summary()
    fine = report["fine"]

    assert fine["Self"]["edge_retained_count"] == {
        "mean": 1.5,
        "minimum": 1,
        "maximum": 2,
    }
    assert fine["Physical"]["effective_edge_count"] == {
        "mean": 0.5,
        "minimum": 0,
        "maximum": 1,
    }
    assert fine["overall"]["effective_edge_count"] == {
        "mean": 3.0,
        "minimum": 3,
        "maximum": 3,
    }
    assert fine["overall"]["lag_histogram"]["counts"] == [2, 1, 2, 1] + [0] * 8
    assert sum(fine["overall"]["lag_histogram"]["counts"]) == 6
    assert fine["overall"]["mean_unique_source_count"] == 2.0
    assert fine["overall"]["mean_effective_sensor_lag_count"] == 3.0
    assert fine["overall"]["effective_lag_per_unique_source"] == 1.5
    assert fine["overall"]["multi_lag"] == {
        "numerator": 2,
        "denominator": 4,
        "ratio": 0.5,
    }


def test_diagnostic_summary_is_deterministic_and_validates_conservation() -> None:
    assert _summary() == _summary()

    state = _diagnostic_state()
    state.family_retained_weight[0, 0, 0, 0] = 0.5
    accumulator = DiagnosticAccumulator(device=torch.device("cpu"))
    valid_query = torch.zeros((1, 1, 12), dtype=torch.bool)
    valid_query[..., :2] = True
    accumulator.update(state, valid_query)
    with pytest.raises(RuntimeError, match="diagnostic invariant"):
        accumulator.finalize()


@pytest.mark.cuda
def test_cpu_cuda_diagnostic_support_parity() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    cpu = _summary(torch.device("cpu"))
    cuda = _summary(torch.device("cuda:0"))
    assert cuda["family"]["active_family_count"] == cpu["family"]["active_family_count"]
    for family in ("Self", "Physical", "Semantic", "overall"):
        assert cuda["fine"][family]["effective_edge_count"] == cpu["fine"][family][
            "effective_edge_count"
        ]
        assert cuda["fine"][family]["lag_histogram"]["counts"] == cpu["fine"][family][
            "lag_histogram"
        ]["counts"]


def test_cpu_diagnostic_evaluation_smoke() -> None:
    torch.manual_seed(76)
    model = HiDFilter(1).eval()
    samples = [
        {
            "inputs": torch.ones(12, 1, 1),
            "targets": torch.full((12, 1, 1), 2.0),
        }
        for _ in range(2)
    ]
    scaler = ZScoreScaler(
        norm_each_channel=False,
        rescale=True,
        stats={"mean": torch.tensor(0.0), "std": torch.tensor(1.0)},
    )

    result = evaluate_with_diagnostics(
        model,
        DataLoader(samples, batch_size=2),
        scaler,
        torch.device("cpu"),
    )

    assert result.samples == 2
    assert result.metrics.valid == 24
    assert result.diagnostics["valid_query_count"] == 24
    assert result.diagnostics["fine"]["Physical"]["edge_retained_mass_mean"] is None


def test_formal_learning_rate_matches_frozen_formula() -> None:
    assert formal_learning_rate(1) == pytest.approx(1.0e-3, rel=0.0, abs=1.0e-15)
    assert formal_learning_rate(100) == pytest.approx(1.0e-5, rel=0.0, abs=1.0e-15)
    assert formal_learning_rate(50) == pytest.approx(
        1.0e-5 + 0.5 * (1.0e-3 - 1.0e-5) * (1.0 + torch.cos(torch.tensor(torch.pi * 49 / 99)).item())
    )
    with pytest.raises(ValueError, match="1..100"):
        formal_learning_rate(0)


def test_early_stopping_warmup_isolation_and_patience_boundary() -> None:
    state = WarmupEarlyStopping(patience=15, min_delta=0.0, first_eligible_epoch=6)
    for epoch in range(1, 6):
        decision = state.observe(epoch, 100.0 + epoch)
        assert not decision.eligible
        assert state.non_improving_epochs == 0
        assert state.best_metric == float("inf")

    assert state.observe(6, 10.0).improved
    for epoch in range(7, 21):
        assert not state.observe(epoch, 10.0).should_stop
    boundary = state.observe(21, 10.0)
    assert boundary.should_stop
    assert boundary.non_improving_epochs == 15
    assert state.best_epoch == 6


def test_formal_test_once_requires_final_mode_training_and_best_reload() -> None:
    development = FormalTestOnceGuard(mode="development")
    development.mark_training_complete()
    development.mark_best_reloaded()
    assert not development.should_run_test
    with pytest.raises(RuntimeError, match="development"):
        development.begin_test()

    final = FormalTestOnceGuard(mode="final")
    assert not final.should_run_test
    final.mark_training_complete()
    assert not final.should_run_test
    final.mark_best_reloaded()
    assert final.should_run_test
    final.begin_test()
    assert not final.should_run_test
    with pytest.raises(RuntimeError, match="already"):
        final.begin_test()


def test_formal_checkpoint_strict_reload_and_config_fingerprint(tmp_path: Path) -> None:
    model = nn.Linear(2, 1)
    manager = CheckpointManager(tmp_path)
    fingerprint = formal_config_fingerprint({"seed": 2026, "max_epochs": 100})
    metadata = {
        "seed": 2026,
        "epoch": 6,
        "val_mae": 1.25,
        "family_top_p_enabled": True,
        "formal_config_version": 1,
        "formal_config_fingerprint": fingerprint,
    }
    assert manager.maybe_save_best(model, 1.25, metadata)
    expected = {name: value.detach().clone() for name, value in model.state_dict().items()}
    with torch.no_grad():
        model.weight.add_(1.0)

    loaded = load_model_checkpoint(manager.best_path, model, map_location="cpu", strict=True)
    assert loaded == metadata
    for name, value in model.state_dict().items():
        assert torch.equal(value, expected[name])
    assert fingerprint == formal_config_fingerprint({"max_epochs": 100, "seed": 2026})


def test_cpu_formal_protocol_lifecycle_smoke(tmp_path: Path) -> None:
    torch.manual_seed(75)
    model = nn.Linear(2, 1)
    optimizer = build_optimizer(model)
    manager = CheckpointManager(tmp_path)
    stopping = WarmupEarlyStopping(
        patience=15, min_delta=0.0, first_eligible_epoch=6
    )
    guard = FormalTestOnceGuard(mode="development")
    for epoch_number in range(1, 7):
        learning_rate = set_formal_learning_rate(optimizer, epoch_number)
        policy = phase5_epoch_policy(epoch_number)
        prediction = model(torch.ones(2, 2))
        prediction.square().mean().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        metric = 10.0 - epoch_number
        decision = stopping.observe(epoch_number, metric)
        metadata = {
            "epoch": epoch_number,
            "val_mae": metric,
            "family_top_p_enabled": policy.family_top_p_enabled,
        }
        manager.save_last(model, metadata)
        if decision.improved:
            assert manager.maybe_save_best(model, metric, metadata)
        assert learning_rate == formal_learning_rate(epoch_number)
    guard.mark_training_complete()
    loaded = load_model_checkpoint(manager.best_path, model, map_location="cpu", strict=True)
    guard.mark_best_reloaded()
    assert loaded["epoch"] == 6
    assert loaded["family_top_p_enabled"] is True
    assert not guard.should_run_test


def test_formal_config_and_entrypoints_are_frozen() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["phase"] == 6
    assert config["dataset_dir"] == "/root/autodl-tmp/datasets/PEMS08"
    assert config["max_epochs"] == 100
    assert config["batch_size"] == 64
    assert config["num_workers"] == 0
    assert config["optimizer"] == {
        "name": "AdamW",
        "lr": 0.001,
        "betas": [0.9, 0.999],
        "eps": 1e-08,
        "matrix_weight_decay": 0.0001,
        "other_weight_decay": 0.0,
        "grad_clip": 5.0,
    }
    assert config["early_stopping"] == {
        "first_eligible_epoch": 6,
        "patience": 15,
        "min_delta": 0.0,
    }
    assert config["precision"] == "fp32"
    assert config["amp"] is False
    assert config["ddp"] is False
    assert config["training_resume"] is False

    runner = _load_script("run_formal_pems08", RUNNER_PATH)
    benchmark = _load_script("benchmark_phase6_diagnostics", BENCHMARK_PATH)
    assert callable(runner.main)
    assert callable(benchmark.main)


def test_formal_runner_has_explicit_seed_and_no_resume_or_per_epoch_test() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert 'parser.add_argument("--seed"' in source
    assert 'choices=("development", "final")' in source
    assert 'parser.add_argument("--resume"' not in source
    assert "load_training_state" not in source
    loop_body = source.split("for epoch_number in", 1)[1].split("guard.mark_training_complete", 1)[0]
    assert "test_loader" not in loop_body
    assert "load_pems08_test_dataset" not in loop_body
