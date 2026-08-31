from __future__ import annotations

import importlib
import importlib.util
import inspect
import json
from pathlib import Path

import numpy as np

from hidfilter.physical import PhysicalGraphContract, build_physical_candidates
from hidfilter.protocol.pems08 import TrafficOnlyForecastingDataset, raw_valid_mask_array


def _phase3_runtime_module():
    assert importlib.util.find_spec("hidfilter.runtime.semantic") is not None
    return importlib.import_module("hidfilter.runtime.semantic")


def _script_module(filename: str):
    path = Path(__file__).resolve().parents[2] / "scripts" / filename
    spec = importlib.util.spec_from_file_location(filename.removesuffix(".py"), path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_semantic_preparation_api_is_train_only_and_cache_is_fingerprinted(tmp_path):
    runtime = _phase3_runtime_module()
    parameters = inspect.signature(runtime.prepare_semantic_candidates).parameters
    assert "raw_train" in parameters
    assert not ({"raw_val", "raw_test", "val_dataset", "test_dataset", "dataset_dir"} & parameters.keys())

    time = np.arange(12, dtype=np.float64)
    raw_train = np.column_stack(
        (time**2 + 1.0, 2.0 * time**2 + 1.0, 300.0 - time**2, time**3 + 1.0)
    )
    adjacency = np.zeros((4, 4), dtype=np.float64)
    physical = build_physical_candidates(
        adjacency, PhysicalGraphContract("undirected", "affinity", None), kp=8
    )
    cache_path = tmp_path / "semantic_candidates.npz"

    first = runtime.prepare_semantic_candidates(
        raw_train,
        raw_valid_mask_array(raw_train),
        adjacency,
        physical.sources,
        cache_path,
        ks=2,
        min_overlap=5,
        variance_threshold=1.0e-12,
    )
    second = runtime.prepare_semantic_candidates(
        raw_train.copy(),
        raw_valid_mask_array(raw_train.copy()),
        adjacency.copy(),
        physical.sources,
        cache_path,
        ks=2,
        min_overlap=5,
        variance_threshold=1.0e-12,
    )
    summary = runtime.summarize_semantic_candidates(first.artifact)

    assert not first.cache_hit
    assert second.cache_hit
    assert first.artifact.fingerprint == second.artifact.fingerprint
    assert first.seconds >= 0.0
    assert summary["ks"] == 2
    assert summary["available_target_count"] >= 1
    assert summary["unavailable_target_count"] >= 0
    assert summary["min_selected_abs_corr"] >= 0.0
    assert summary["max_selected_abs_corr"] <= 1.0
    assert summary["positive_signed_corr_count"] + summary["negative_signed_corr_count"] > 0


def test_pems08_semantic_boundary_rejects_val_and_ignores_val_test_changes(tmp_path):
    runtime = _phase3_runtime_module()
    time = np.arange(300, dtype=np.float32)
    train = np.column_stack(
        (time**2 + 1.0, 2.0 * time**2 + 1.0, 100_000.0 - time**2, time**3 + 1.0)
    ).astype(np.float32)
    np.save(tmp_path / "train_data.npy", train)
    np.save(tmp_path / "val_data.npy", train + 1_000_000.0)
    np.save(tmp_path / "test_data.npy", train + 2_000_000.0)
    adjacency = np.zeros((4, 4), dtype=np.float64)
    physical = build_physical_candidates(
        adjacency, PhysicalGraphContract("undirected", "affinity", None), kp=8
    )

    first = runtime.prepare_pems08_semantic_candidates(
        TrafficOnlyForecastingDataset(tmp_path, "train"),
        adjacency,
        physical.sources,
        tmp_path / "semantic_first.npz",
        ks=2,
        min_overlap=288,
        variance_threshold=1.0e-12,
    )
    np.save(tmp_path / "val_data.npy", train - 3_000_000.0)
    np.save(tmp_path / "test_data.npy", train - 4_000_000.0)
    second = runtime.prepare_pems08_semantic_candidates(
        TrafficOnlyForecastingDataset(tmp_path, "train"),
        adjacency,
        physical.sources,
        tmp_path / "semantic_second.npz",
        ks=2,
        min_overlap=288,
        variance_threshold=1.0e-12,
    )

    assert first.artifact.fingerprint == second.artifact.fingerprint
    val_dataset = TrafficOnlyForecastingDataset(tmp_path, "val")
    with np.testing.assert_raises_regex(ValueError, "train split"):
        runtime.prepare_pems08_semantic_candidates(
            val_dataset,
            adjacency,
            physical.sources,
            tmp_path / "semantic_val.npz",
            ks=2,
            min_overlap=288,
            variance_threshold=1.0e-12,
        )


def test_phase3_config_is_the_frozen_short_sanity_contract():
    config_path = Path(__file__).resolve().parents[2] / "configs" / "phase3_semantic_pems08.json"
    assert config_path.is_file()
    config = json.loads(config_path.read_text(encoding="utf-8"))

    assert config == {
        "dataset_dir": "/root/autodl-tmp/datasets/PEMS08",
        "adjacency_path": "/root/autodl-tmp/datasets/PEMS08/adj_mx.pkl",
        "graph_contract": {
            "graph_mode": "undirected",
            "weight_semantics": "affinity",
            "conversion_scale": None,
        },
        "physical_candidate_cache_path": "artifacts/phase3/semantic/physical_candidates_pems08.npz",
        "semantic_candidate_cache_path": "artifacts/phase3/semantic/semantic_candidates_pems08.npz",
        "checkpoint_dir": "artifacts/phase3/semantic/checkpoints",
        "report_path": "reports/phase3_semantic_cuda.json",
        "seed": 2026,
        "epochs": 3,
        "batch_size": 64,
        "worker_candidates": [0, 2, 4, 8],
        "worker_benchmark_batches": 64,
        "grad_clip": 5.0,
        "physical_kp": 8,
        "semantic_ks": 8,
        "semantic_min_overlap": 288,
        "semantic_variance_threshold": 1.0e-12,
    }


def test_phase3_runner_and_benchmark_entrypoints_are_importable():
    runner = _script_module("run_phase3_semantic.py")
    benchmark = _script_module("benchmark_phase3_semantic.py")
    assert callable(runner.main)
    assert callable(runner.parse_args)
    assert callable(benchmark.main)
    assert callable(benchmark.parse_args)
