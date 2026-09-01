from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

from hidfilter.model import HiDFilter
from hidfilter.physical import PhysicalGraphContract, build_physical_candidates
from hidfilter.runtime.checkpoint import CheckpointManager, load_model_checkpoint
from hidfilter.runtime.router import router_forward
from hidfilter.semantic import SemanticCandidateMetadata


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "phase4_router_pems08.json"
RUNNER_PATH = ROOT / "scripts" / "run_phase4_router.py"
BENCHMARK_PATH = ROOT / "scripts" / "benchmark_phase4_router.py"


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
    sources = torch.arange(num_nodes, dtype=torch.int64).view(1, -1).expand(num_nodes, -1)
    source_index = torch.zeros((num_nodes, 8), dtype=torch.int64)
    source_valid = torch.zeros((num_nodes, 8), dtype=torch.bool)
    for target in range(num_nodes):
        choices = sources[target][sources[target] != target][:2]
        source_index[target, : choices.numel()] = choices
        source_valid[target, : choices.numel()] = True
    candidate_source = source_index.repeat_interleave(12, dim=1)
    lag_index = torch.arange(12, dtype=torch.int64).repeat(8).view(1, -1).expand(num_nodes, -1)
    valid = source_valid.repeat_interleave(12, dim=1)
    semantic = SemanticCandidateMetadata(
        source_index=candidate_source,
        lag_index=lag_index,
        flat_index=candidate_source * 12 + lag_index,
        valid=valid,
        prior=valid.float(),
    )
    return HiDFilter(num_nodes, physical_candidates=physical, semantic_candidates=semantic)


def test_phase4_config_and_entrypoints_are_frozen():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["phase"] == 4
    assert config["dataset_dir"] == "/root/autodl-tmp/datasets/PEMS08"
    assert config["epochs"] == 3
    assert config["batch_size"] == 64
    assert config["num_workers"] == 0
    assert config["precision"] == "fp32"
    assert config["amp"] is False
    assert config["ddp"] is False
    assert config["router"]["family_order"] == ["Self", "Physical", "Semantic"]
    assert config["router"]["top_p_enabled"] is False

    runner = _load_script("run_phase4_router", RUNNER_PATH)
    benchmark = _load_script("benchmark_phase4_router", BENCHMARK_PATH)
    assert callable(runner.main)
    assert callable(benchmark.main)


def test_phase4_router_forward_and_checkpoint_reload(tmp_path):
    torch.manual_seed(31)
    model = _model().eval()
    history = torch.randn(2, 12, 4, 1)
    expected = router_forward(model, history).detach()
    manager = CheckpointManager(tmp_path)
    manager.save_last(model, {"epoch": 1, "val_mae": 2.5})

    reloaded = _model().eval()
    state = load_model_checkpoint(manager.last_path, reloaded, map_location="cpu", strict=True)
    actual = router_forward(reloaded, history).detach()

    assert state["epoch"] == 1
    assert state["val_mae"] == 2.5
    assert torch.equal(expected, actual)


def test_phase4_runner_never_reads_test_metrics_for_development():
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert "load_pems08_test_dataset" not in source
    assert "test_mae" not in source
    assert "from hidfilter.filtration import edge_top_p" not in source
