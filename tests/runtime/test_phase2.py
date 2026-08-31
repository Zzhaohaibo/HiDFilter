from __future__ import annotations

import importlib
import importlib.util
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from hidfilter.model import HiDFilter
from hidfilter.physical import PhysicalGraphContract, build_physical_candidates
from hidfilter.protocol.pems08 import TrafficOnlyForecastingDataset, fit_train_scaler
from hidfilter.runtime.phase0 import build_dataloader, build_optimizer, evaluate, train_one_epoch


def _phase2_runtime_module():
    assert importlib.util.find_spec("hidfilter.runtime.physical") is not None
    return importlib.import_module("hidfilter.runtime.physical")


def _phase2_runner_module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "run_phase2_physical.py"
    spec = importlib.util.spec_from_file_location("run_phase2_physical", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _phase2_benchmark_module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "benchmark_phase2_physical.py"
    spec = importlib.util.spec_from_file_location("benchmark_phase2_physical", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_phase2_model_reuses_frozen_train_and_validation_runtime(tmp_path):
    time = np.arange(72, dtype=np.float32)[:, None]
    raw = np.concatenate((time + 1.0, time * 0.5 + 2.0, time * 0.25 + 3.0), axis=1)
    np.save(tmp_path / "train_data.npy", raw)
    np.save(tmp_path / "val_data.npy", raw + 1.0)
    np.save(tmp_path / "test_data.npy", raw + 2.0)
    adjacency = np.ones((3, 3), dtype=np.float64) - np.eye(3, dtype=np.float64)
    candidates = build_physical_candidates(
        adjacency, PhysicalGraphContract("undirected", "affinity", None), kp=8
    ).candidates
    train_dataset = TrafficOnlyForecastingDataset(tmp_path, "train")
    val_dataset = TrafficOnlyForecastingDataset(tmp_path, "val")
    scaler = fit_train_scaler(train_dataset)
    train_loader = build_dataloader(
        train_dataset, batch_size=8, num_workers=0, shuffle=True, seed=11, pin_memory=False
    )
    val_loader = build_dataloader(
        val_dataset, batch_size=8, num_workers=0, shuffle=False, seed=11, pin_memory=False
    )
    model = HiDFilter(num_nodes=3, physical_candidates=candidates)
    optimizer = build_optimizer(model)
    forward_fn = _phase2_runtime_module().physical_forward

    training = train_one_epoch(
        model,
        train_loader,
        scaler,
        optimizer,
        torch.device("cpu"),
        grad_clip=5.0,
        forward_fn=forward_fn,
    )
    validation = evaluate(
        model,
        val_loader,
        scaler,
        torch.device("cpu"),
        forward_fn=forward_fn,
    )

    assert training.loss > 0.0
    assert validation.metrics.valid > 0
    assert validation.metrics.mae > 0.0


def test_phase2_candidate_preparation_uses_fingerprinted_cache_and_reports_availability(tmp_path):
    runtime = _phase2_runtime_module()
    adjacency = np.zeros((3, 3), dtype=np.float64)
    adjacency[0, 1] = 1.0
    adjacency_path = tmp_path / "adj_mx.pkl"
    cache_path = tmp_path / "physical_candidates.npz"
    with adjacency_path.open("wb") as handle:
        pickle.dump(adjacency, handle)
    contract = PhysicalGraphContract("directed", "affinity", None)

    first = runtime.prepare_physical_candidates(adjacency_path, cache_path, contract, kp=8)
    second = runtime.prepare_physical_candidates(adjacency_path, cache_path, contract, kp=8)
    summary = runtime.summarize_physical_candidates(first.artifact)

    assert not first.cache_hit
    assert second.cache_hit
    assert first.artifact.fingerprint == second.artifact.fingerprint
    assert first.seconds >= 0.0
    assert summary == {
        "kp": 8,
        "available_target_count": 1,
        "unavailable_target_count": 2,
        "min_real_sources_per_target": 0,
        "mean_real_sources_per_target": 1.0 / 3.0,
        "max_real_sources_per_target": 1,
    }


def test_phase2_config_is_the_frozen_short_sanity_contract():
    config_path = Path(__file__).resolve().parents[2] / "configs" / "phase2_physical_pems08.json"
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
        "candidate_cache_path": "artifacts/phase2/physical/physical_candidates_pems08.npz",
        "checkpoint_dir": "artifacts/phase2/physical/checkpoints",
        "report_path": "reports/phase2_physical_cuda.json",
        "seed": 2026,
        "epochs": 3,
        "batch_size": 64,
        "worker_candidates": [0, 2, 4, 8],
        "worker_benchmark_batches": 64,
        "grad_clip": 5.0,
        "physical_kp": 8,
    }


def test_phase2_runner_entrypoint_is_importable():
    runner = _phase2_runner_module()
    assert callable(runner.main)
    assert callable(runner.parse_args)


def test_phase2_benchmark_entrypoint_is_importable():
    benchmark = _phase2_benchmark_module()
    assert callable(benchmark.main)
    assert callable(benchmark.parse_args)
