from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from basicts.scaler import ZScoreScaler
from torch import nn
from torch.utils.data import DataLoader

from hidfilter.model import FAMILY_TOP_P_RHO, HiDFilter
from hidfilter.physical import PhysicalGraphContract, build_physical_candidates
from hidfilter.runtime.checkpoint import CheckpointManager, load_model_checkpoint
from hidfilter.runtime.family_top_p import (
    collect_family_top_p_summary,
    family_top_p_forward,
    phase5_epoch_policy,
)
from hidfilter.semantic import SemanticCandidateMetadata


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "phase5_family_top_p_pems08.json"
RUNNER_PATH = ROOT / "scripts" / "run_phase5_family_top_p.py"
BENCHMARK_PATH = ROOT / "scripts" / "benchmark_phase5_family_top_p.py"


def _load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _model(num_nodes: int = 4) -> HiDFilter:
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
    return HiDFilter(num_nodes, physical_candidates=physical, semantic_candidates=semantic)


@pytest.mark.parametrize(
    ("epoch_number", "enabled"),
    [(1, False), (2, False), (5, False), (6, True), (100, True)],
)
def test_phase5_warmup_boundary(epoch_number, enabled):
    policy = phase5_epoch_policy(epoch_number)

    assert policy.family_top_p_enabled is enabled
    assert policy.best_selection_enabled is enabled
    assert policy.patience_enabled is enabled


@pytest.mark.parametrize("epoch_number", [0, -1])
def test_phase5_warmup_rejects_nonpositive_epoch(epoch_number):
    with pytest.raises(ValueError, match="positive 1-based"):
        phase5_epoch_policy(epoch_number)


def test_phase5_train_and_validation_use_the_same_forward_flag():
    class ForwardSpy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.flags: list[bool] = []

        def forward(
            self, history: torch.Tensor, *, family_top_p_enabled: bool = False
        ) -> torch.Tensor:
            self.flags.append(family_top_p_enabled)
            return history

    model = ForwardSpy()
    history = torch.zeros(1)
    for epoch_number in range(1, 7):
        policy = phase5_epoch_policy(epoch_number)
        forward = family_top_p_forward(policy.family_top_p_enabled)
        forward(model, history)
        forward(model, history)

    assert model.flags == [False] * 10 + [True, True]


def test_phase5_warmup_validation_is_excluded_from_best_and_patience():
    validation_mae = {1: 1.0, 2: 0.9, 3: 0.8, 4: 0.7, 5: 0.5, 6: 2.0}
    eligible = [
        (mae, epoch)
        for epoch, mae in validation_mae.items()
        if phase5_epoch_policy(epoch).best_selection_enabled
    ]

    assert min(eligible) == (2.0, 6)
    assert all(not phase5_epoch_policy(epoch).patience_enabled for epoch in range(1, 6))


def test_phase5_checkpoint_strict_reload_runs_with_top_p_on(tmp_path):
    torch.manual_seed(61)
    model = _model().eval()
    history = torch.randn(2, 12, 4, 1)
    forward = family_top_p_forward(True)
    expected = forward(model, history).detach()
    manager = CheckpointManager(tmp_path)
    metadata = {
        "epoch": 6,
        "val_mae": 2.0,
        "family_top_p_enabled": True,
        "rho_family": FAMILY_TOP_P_RHO,
    }
    assert manager.maybe_save_best(model, 2.0, metadata)

    reloaded = _model().eval()
    loaded = load_model_checkpoint(manager.best_path, reloaded, map_location="cpu", strict=True)
    actual = forward(reloaded, history).detach()

    assert loaded["epoch"] >= 6
    assert loaded["family_top_p_enabled"] is True
    assert loaded["rho_family"] == 0.8
    assert torch.equal(actual, expected)


def test_phase5_family_top_p_summary_is_compact_post_hoc_evidence():
    class FixedRouter(nn.Module):
        num_nodes = 1

        def __init__(self) -> None:
            super().__init__()
            self.register_buffer(
                "family_available", torch.tensor([[True, True, True]])
            )

        def router_probability(self, history: torch.Tensor) -> torch.Tensor:
            row = torch.tensor([0.50, 0.30, 0.20], device=history.device)
            return row.view(1, 1, 1, 3).expand(history.shape[0], 1, 12, 3)

    samples = [
        {
            "inputs": torch.ones(12, 1, 1),
            "targets": torch.ones(12, 1, 1),
        }
        for _ in range(2)
    ]
    scaler = ZScoreScaler(
        norm_each_channel=False,
        rescale=True,
        stats={"mean": torch.tensor(0.0), "std": torch.tensor(1.0)},
    )

    summary = collect_family_top_p_summary(
        FixedRouter(), DataLoader(samples, batch_size=2), scaler, torch.device("cpu")
    )

    assert summary.family_order == ("Self", "Physical", "Semantic")
    assert summary.dense_overall_mean == pytest.approx((0.50, 0.30, 0.20))
    assert summary.retained_weight_overall_mean == pytest.approx((0.625, 0.375, 0.0))
    assert summary.retained_family_count_mean == 2.0
    assert summary.retained_family_count_min == 2
    assert summary.retained_family_count_max == 2
    assert summary.positions == 24


def test_phase5_config_and_entrypoints_are_frozen():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["phase"] == 5
    assert config["dataset_dir"] == "/root/autodl-tmp/datasets/PEMS08"
    assert config["epochs"] == 6
    assert config["batch_size"] == 64
    assert config["num_workers"] == 0
    assert config["precision"] == "fp32"
    assert config["amp"] is False
    assert config["ddp"] is False
    assert config["scheduler"]["t_max"] == 99
    assert config["family_top_p"]["rho"] == 0.8
    assert config["family_top_p"]["warmup_epochs"] == 5
    assert config["family_top_p"]["first_enabled_epoch"] == 6

    runner = _load_script("run_phase5_family_top_p", RUNNER_PATH)
    benchmark = _load_script("benchmark_phase5_family_top_p", BENCHMARK_PATH)
    assert callable(runner.main)
    assert callable(benchmark.main)


def test_phase5_runner_never_reads_test_metrics_or_implements_phase6():
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert "load_pems08_test_dataset" not in source
    assert "test_mae" not in source
    assert "effective_edge" not in source
    assert "lag_histogram" not in source
    assert "family_entropy" not in source
