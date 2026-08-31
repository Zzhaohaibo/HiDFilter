from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MetricSummary:
    mae: float
    rmse: float
    mape: float
    valid: int


@dataclass(frozen=True)
class RawMetricReport:
    mae: float
    rmse: float
    mape: float
    valid: int
    per_horizon: tuple[MetricSummary, ...]


class RawMetricAccumulator:
    """Accumulate global raw-scale error sums on device and reduce once."""

    def __init__(self, horizons: int) -> None:
        self.horizons = horizons
        self._overall = torch.zeros(4, dtype=torch.float64)
        self._per_horizon = torch.zeros(horizons, 4, dtype=torch.float64)
        self._has_updates = False

    def update(self, prediction: torch.Tensor, targets: torch.Tensor, valid: torch.Tensor) -> None:
        if prediction.shape != targets.shape or targets.shape != valid.shape:
            raise ValueError("prediction, targets, and valid must have identical shapes")
        if prediction.ndim != 4 or prediction.shape[1] != self.horizons:
            raise ValueError(f"expected [B,{self.horizons},N,1], got {prediction.shape}")
        if not self._has_updates:
            self._overall = self._overall.to(prediction.device)
            self._per_horizon = self._per_horizon.to(prediction.device)
            self._has_updates = True
        elif self._overall.device != prediction.device:
            raise ValueError("all metric updates must use the same device")

        prediction64 = prediction.detach().to(torch.float64)
        targets64 = targets.detach().to(torch.float64)
        valid64 = valid.detach().to(torch.float64)
        error = prediction64 - targets64
        absolute_error = error.abs() * valid64
        squared_error = error.square() * valid64
        safe_targets = torch.where(valid, targets64, torch.ones_like(targets64))
        percentage_error = (error / safe_targets).abs() * valid64

        self._overall += torch.stack(
            (
                absolute_error.sum(),
                squared_error.sum(),
                percentage_error.sum(),
                valid64.sum(),
            )
        )
        reduce_dims = (0, 2, 3)
        self._per_horizon += torch.stack(
            (
                absolute_error.sum(dim=reduce_dims),
                squared_error.sum(dim=reduce_dims),
                percentage_error.sum(dim=reduce_dims),
                valid64.sum(dim=reduce_dims),
            ),
            dim=1,
        )

    def compute(self) -> RawMetricReport:
        overall = self._overall.detach().cpu()
        per_horizon = self._per_horizon.detach().cpu()
        summary = _summarize(overall)
        horizons = tuple(_summarize(row) for row in per_horizon)
        return RawMetricReport(
            mae=summary.mae,
            rmse=summary.rmse,
            mape=summary.mape,
            valid=summary.valid,
            per_horizon=horizons,
        )


def _summarize(stats: torch.Tensor) -> MetricSummary:
    sae, sse, sape, valid_value = stats.tolist()
    valid = int(valid_value)
    if valid == 0:
        return MetricSummary(mae=0.0, rmse=0.0, mape=0.0, valid=0)
    values = torch.tensor((sae, sse, sape), dtype=torch.float64)
    if not torch.isfinite(values).all().item():
        raise FloatingPointError("non-finite value reached raw metric reduction")
    return MetricSummary(
        mae=sae / valid,
        rmse=(sse / valid) ** 0.5,
        mape=sape / valid,
        valid=valid,
    )
