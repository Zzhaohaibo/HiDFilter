from __future__ import annotations

import numpy as np
import pytest
import torch
from basicts.utils.mask import null_val_mask
from torch.utils.data import DataLoader

from hidfilter.protocol.pems08 import (
    EXPECTED_WINDOWS,
    TrafficOnlyForecastingDataset,
    fit_train_scaler,
    prepare_batch,
    raw_valid_mask_array,
    raw_valid_mask,
    validate_pems08_connectivity_adjacency,
    validate_pems08_protocol,
)


def _valid_pems08_connectivity() -> np.ndarray:
    adjacency = np.zeros((170, 170), dtype=np.float32)
    adjacency[0, 1] = adjacency[1, 0] = 1.0
    adjacency[1, 169] = adjacency[169, 1] = 1.0
    return adjacency


def test_valid_pems08_binary_connectivity_reports_formal_evidence():
    evidence = validate_pems08_connectivity_adjacency(_valid_pems08_connectivity())

    assert evidence == {
        "adjacency_shape": [170, 170],
        "binary_connectivity_valid": True,
        "symmetric": True,
        "zero_diagonal": True,
        "directed_positive_entry_count": 4,
    }


def test_pems08_connectivity_requires_exact_170_node_shape():
    with pytest.raises(ValueError, match="shape"):
        validate_pems08_connectivity_adjacency(np.zeros((169, 169), dtype=np.float32))


@pytest.mark.parametrize(
    ("mutate", "error"),
    [
        (lambda adjacency: adjacency.__setitem__(([0, 1], [1, 0]), 0.5), "binary"),
        (lambda adjacency: adjacency.__setitem__((0, 0), 1.0), "diagonal"),
        (lambda adjacency: adjacency.__setitem__((0, 1), 0.0), "symmetric"),
        (lambda adjacency: adjacency.__setitem__((0, 1), np.inf), "non-finite"),
        (lambda adjacency: adjacency.fill(0.0), "positive off-diagonal edge"),
    ],
)
def test_invalid_pems08_connectivity_is_a_hard_failure(mutate, error):
    adjacency = _valid_pems08_connectivity()
    mutate(adjacency)

    with pytest.raises(ValueError, match=error):
        validate_pems08_connectivity_adjacency(adjacency)


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


@pytest.mark.parametrize("nonfinite", [np.nan, np.inf, -np.inf])
def test_nonfinite_raw_artifact_is_a_hard_failure(tmp_path, nonfinite):
    raw = np.ones((24, 1), dtype=np.float32)
    raw[0, 0] = nonfinite
    for split in ("train", "val", "test"):
        np.save(tmp_path / f"{split}_data.npy", raw)

    for split in ("train", "val", "test"):
        with pytest.raises(ValueError, match=rf"{split}_data\.npy contains non-finite"):
            TrafficOnlyForecastingDataset(tmp_path, split)


def test_raw_valid_mask_uses_frozen_finite_isclose_semantics():
    raw = torch.tensor([0.0, 4.0e-5, -4.0e-5, 5.0e-5, -5.0e-5, 5.1e-5, -5.1e-5, 1.0])

    assert raw_valid_mask(raw).tolist() == [False, False, False, False, False, True, True, True]


def test_raw_valid_array_uses_the_same_semantics_for_semantic_statistics():
    raw = np.array([0.0, 4.0e-5, -5.0e-5, 5.1e-5, -5.1e-5, 1.0])

    assert raw_valid_mask_array(raw).tolist() == [False, False, False, True, True, True]


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
