from __future__ import annotations

import importlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

from hidfilter.model import SelfOnlyHiDFilter
from hidfilter.protocol.pems08 import TrafficOnlyForecastingDataset, fit_train_scaler
from hidfilter.runtime.phase0 import build_dataloader, build_optimizer, evaluate, train_one_epoch


def _phase1_runtime_module():
    assert importlib.util.find_spec("hidfilter.runtime.self_only") is not None
    return importlib.import_module("hidfilter.runtime.self_only")


def test_self_only_model_reuses_phase0_train_and_validation_runtime(tmp_path):
    time = np.arange(72, dtype=np.float32)[:, None]
    raw = np.concatenate((time + 1.0, time * 0.5 + 2.0, time * 0.25 + 3.0), axis=1)
    np.save(tmp_path / "train_data.npy", raw)
    np.save(tmp_path / "val_data.npy", raw + 1.0)
    np.save(tmp_path / "test_data.npy", raw + 2.0)
    train_dataset = TrafficOnlyForecastingDataset(tmp_path, "train")
    val_dataset = TrafficOnlyForecastingDataset(tmp_path, "val")
    scaler = fit_train_scaler(train_dataset)
    train_loader = build_dataloader(
        train_dataset, batch_size=8, num_workers=0, shuffle=True, seed=11, pin_memory=False
    )
    val_loader = build_dataloader(
        val_dataset, batch_size=8, num_workers=0, shuffle=False, seed=11, pin_memory=False
    )
    model = SelfOnlyHiDFilter(num_nodes=3)
    optimizer = build_optimizer(model)
    forward_fn = _phase1_runtime_module().self_only_forward

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


def test_phase1_config_is_the_frozen_short_sanity_contract():
    config_path = Path(__file__).resolve().parents[2] / "configs" / "phase1_self_pems08.json"
    assert config_path.is_file()
    config = json.loads(config_path.read_text(encoding="utf-8"))

    assert config == {
        "dataset_dir": "/root/autodl-tmp/datasets/PEMS08",
        "checkpoint_dir": "artifacts/phase1/self/checkpoints",
        "report_path": "reports/phase1_self_cuda.json",
        "seed": 2026,
        "epochs": 3,
        "batch_size": 64,
        "worker_candidates": [0, 2, 4, 8],
        "worker_benchmark_batches": 64,
        "grad_clip": 5.0,
    }
