from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from basicts.data import BasicTSForecastingDataset
from basicts.scaler import ZScoreScaler


DATASET_NAME = "PEMS08"
HISTORY_LENGTH = 12
FORECAST_HORIZON = 12
NUM_NODES = 170
EXPECTED_WINDOWS = {"train": 10_690, "val": 3_548, "test": 3_549}


class TrafficOnlyForecastingDataset(BasicTSForecastingDataset):
    """BasicTS forecasting windows with an explicit traffic-only channel."""

    def __init__(
        self,
        data_dir: str | Path,
        mode: str,
        *,
        memmap: bool = False,
        expected_num_nodes: int | None = None,
    ) -> None:
        super().__init__(
            dataset_name=DATASET_NAME,
            input_len=HISTORY_LENGTH,
            output_len=FORECAST_HORIZON,
            mode=mode,
            use_timestamps=False,
            data_file_path=str(data_dir),
            memmap=memmap,
        )
        if self._data.ndim == 2:
            num_nodes = self._data.shape[1]
        elif self._data.ndim == 3 and self._data.shape[-1] == 1:
            num_nodes = self._data.shape[1]
        else:
            raise ValueError(
                f"{mode}_data.npy must have shape [T,N] or [T,N,1], got {self._data.shape}"
            )
        if expected_num_nodes is not None and num_nodes != expected_num_nodes:
            raise ValueError(f"PEMS08 must contain {expected_num_nodes} nodes, got {num_nodes}")
        if not np.isfinite(self._data).all():
            raise ValueError(f"{mode}_data.npy contains non-finite raw traffic")

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        item = super().__getitem__(index)
        return {
            "inputs": _with_traffic_channel(item["inputs"]),
            "targets": _with_traffic_channel(item["targets"]),
        }

    @property
    def data(self) -> np.ndarray:
        return _with_traffic_channel(self._data)


def _with_traffic_channel(data: np.ndarray) -> np.ndarray:
    return data[..., None] if data.ndim == 2 else data


def validate_pems08_protocol(data_dir: str | Path) -> dict[str, int]:
    """Validate the three pre-split official files without joining or resplitting."""

    counts = {
        split: len(
            TrafficOnlyForecastingDataset(
                data_dir,
                split,
                memmap=True,
                expected_num_nodes=NUM_NODES,
            )
        )
        for split in EXPECTED_WINDOWS
    }
    if counts != EXPECTED_WINDOWS:
        raise ValueError(f"unexpected PEMS08 window counts: {counts}, expected {EXPECTED_WINDOWS}")
    return counts


def raw_valid_mask(raw: torch.Tensor) -> torch.Tensor:
    """BasicTS null-value semantics for finite raw traffic, before normalization."""

    zero = torch.zeros((), dtype=raw.dtype, device=raw.device)
    return ~torch.isclose(raw, zero, atol=5e-5)


def fit_train_scaler(train_dataset: TrafficOnlyForecastingDataset) -> ZScoreScaler:
    """Fit the frozen global scalar Z-score using raw training traffic only."""

    if str(train_dataset.mode) != "train":
        raise ValueError("scaler statistics must be fitted on the train split")
    raw_train = train_dataset.data
    if not np.isfinite(raw_train).all():
        raise ValueError("train_data.npy contains non-finite raw traffic; scaler fitting aborted")
    mean = float(np.mean(raw_train))
    std = float(np.std(raw_train, ddof=0))
    if std == 0.0:
        std = 1.0
    return ZScoreScaler(
        norm_each_channel=False,
        rescale=True,
        stats={
            "mean": torch.tensor(mean, dtype=torch.float32),
            "std": torch.tensor(std, dtype=torch.float32),
        },
    )


def move_scaler_to_device(scaler: ZScoreScaler, device: torch.device) -> ZScoreScaler:
    """Move the two fitted scalars once so transform has no per-batch host copy."""

    scaler.stats = {name: value.to(device) for name, value in scaler.stats.items()}
    return scaler


@dataclass(frozen=True)
class PreparedBatch:
    inputs: torch.Tensor
    targets: torch.Tensor
    inputs_valid: torch.Tensor
    targets_valid: torch.Tensor


def prepare_batch(batch: Mapping[str, torch.Tensor], scaler: ZScoreScaler) -> PreparedBatch:
    """Mask raw values first, then normalize and zero invalid model positions."""

    raw_inputs = batch["inputs"]
    raw_targets = batch["targets"]
    inputs_valid = raw_valid_mask(raw_inputs)
    targets_valid = raw_valid_mask(raw_targets)
    inputs = scaler.transform(raw_inputs, inputs_valid)
    targets = scaler.transform(raw_targets, targets_valid)
    inputs = torch.where(inputs_valid, inputs, torch.zeros((), dtype=inputs.dtype, device=inputs.device))
    targets = torch.where(targets_valid, targets, torch.zeros((), dtype=targets.dtype, device=targets.device))
    return PreparedBatch(inputs, targets, inputs_valid, targets_valid)
