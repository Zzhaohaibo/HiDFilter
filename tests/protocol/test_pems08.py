from __future__ import annotations

import numpy as np
import torch
from basicts.utils.mask import null_val_mask
from torch.utils.data import DataLoader

from hidfilter.protocol.pems08 import (
    EXPECTED_WINDOWS,
    TrafficOnlyForecastingDataset,
    fit_train_scaler,
    prepare_batch,
    raw_valid_mask,
    validate_pems08_protocol,
)


def test_exact_window_counts_and_split_isolation(synthetic_pems08_dir):
    counts = validate_pems08_protocol(synthetic_pems08_dir)
    assert counts == EXPECTED_WINDOWS

    train = TrafficOnlyForecastingDataset(synthetic_pems08_dir, "train")
    val = TrafficOnlyForecastingDataset(synthetic_pems08_dir, "val")
    test = TrafficOnlyForecastingDataset(synthetic_pems08_dir, "test")

    assert train[len(train) - 1]["targets"][-1, 0, 0] < 10_000_000.0
    assert val[0]["inputs"][0, 0, 0] >= 10_000_000.0
    assert test[0]["inputs"][0, 0, 0] >= 20_000_000.0


def test_train_only_global_scalar_scaler_and_inverse(tmp_path):
    train_values = np.arange(1, 25, dtype=np.float32).reshape(24, 1)
    np.save(tmp_path / "train_data.npy", train_values)
    np.save(tmp_path / "val_data.npy", train_values + 1_000.0)
    np.save(tmp_path / "test_data.npy", train_values + 2_000.0)
    train = TrafficOnlyForecastingDataset(tmp_path, "train")
    scaler = fit_train_scaler(train)
    train_raw = train.data

    assert scaler.stats["mean"].item() == np.mean(train_raw).item()
    assert scaler.stats["std"].item() == np.std(train_raw, ddof=0).item()

    raw = torch.tensor([[[[1.0], [2.0], [3.0]]]])
    restored = scaler.inverse_transform(scaler.transform(raw))
    torch.testing.assert_close(restored, raw)


def test_raw_validity_precedes_normalization_and_near_zero_mask(tmp_path):
    train = np.array([[1.0], [2.0], [3.0]] * 8, dtype=np.float32)
    for split in ("train", "val", "test"):
        np.save(tmp_path / f"{split}_data.npy", train)

    dataset = TrafficOnlyForecastingDataset(tmp_path, "train")
    scaler = fit_train_scaler(dataset)
    raw = torch.tensor([[[[2.0], [0.0], [4.0e-5], [5.0e-5], [5.1e-5]]]])
    prepared = prepare_batch({"inputs": raw, "targets": raw}, scaler)

    assert prepared.inputs[0, 0, 0, 0].item() == 0.0
    assert prepared.inputs_valid.flatten().tolist() == [True, False, False, False, True]
    assert raw_valid_mask(raw).equal(prepared.inputs_valid)
    assert raw_valid_mask(raw).equal(null_val_mask(raw, null_val=0.0))


def test_traffic_only_shape_and_no_timestamp_keys(synthetic_pems08_dir):
    dataset = TrafficOnlyForecastingDataset(synthetic_pems08_dir, "train")
    item = dataset[0]
    assert item["inputs"].shape == (12, 170, 1)
    assert item["targets"].shape == (12, 170, 1)
    assert set(item) == {"inputs", "targets"}

    batch = next(iter(DataLoader(dataset, batch_size=2, num_workers=0)))
    assert batch["inputs"].shape == (2, 12, 170, 1)
    assert batch["targets"].shape == (2, 12, 170, 1)
    assert batch["inputs"].device.type == "cpu"
