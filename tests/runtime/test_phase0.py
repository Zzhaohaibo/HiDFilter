from __future__ import annotations

import numpy as np
import torch

from hidfilter.protocol.pems08 import TrafficOnlyForecastingDataset, fit_train_scaler
from hidfilter.runtime.phase0 import (
    benchmark_worker_counts,
    build_dataloader,
    build_optimizer,
    evaluate,
    train_one_epoch,
)
from hidfilter.runtime.stid import build_traffic_only_stid


def test_cpu_phase0_vertical_slice(tmp_path):
    time = np.arange(72, dtype=np.float32)[:, None]
    raw = np.concatenate((time + 1.0, time * 0.5 + 2.0, time * 0.25 + 3.0), axis=1)
    np.save(tmp_path / "train_data.npy", raw)
    np.save(tmp_path / "val_data.npy", raw + 1.0)
    np.save(tmp_path / "test_data.npy", raw + 2.0)
    train_dataset = TrafficOnlyForecastingDataset(tmp_path, "train")
    val_dataset = TrafficOnlyForecastingDataset(tmp_path, "val")
    scaler = fit_train_scaler(train_dataset)
    train_loader = build_dataloader(train_dataset, batch_size=8, num_workers=0, shuffle=True, seed=7, pin_memory=False)
    val_loader = build_dataloader(val_dataset, batch_size=8, num_workers=0, shuffle=False, seed=7, pin_memory=False)
    model = build_traffic_only_stid(num_nodes=3)
    optimizer = build_optimizer(model)

    training = train_one_epoch(model, train_loader, scaler, optimizer, torch.device("cpu"), grad_clip=5.0)
    validation = evaluate(model, val_loader, scaler, torch.device("cpu"))
    worker_results = benchmark_worker_counts(train_dataset, batch_size=8, candidates=(0,), max_batches=3)

    assert training.steps > 0
    assert training.samples > 0
    assert training.loss > 0
    assert validation.metrics.valid > 0
    assert validation.metrics.mae > 0
    assert validation.persistence.valid == validation.metrics.valid
    assert worker_results[0].num_workers == 0
    assert worker_results[0].samples_per_second > 0
