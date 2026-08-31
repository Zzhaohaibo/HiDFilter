from __future__ import annotations

import math

import torch
from basicts.metrics import masked_mae, masked_mape, masked_rmse

from hidfilter.protocol.metrics import RawMetricAccumulator
from hidfilter.protocol.pems08 import raw_valid_mask


def test_raw_metrics_match_basicts_on_same_tensor_and_mask():
    target = torch.tensor(
        [[[[10.0], [0.0]], [[20.0], [40.0]]], [[[30.0], [50.0]], [[0.0], [80.0]]]]
    )
    prediction = target + torch.tensor(
        [[[[1.0], [99.0]], [[-2.0], [4.0]]], [[[3.0], [-5.0]], [[99.0], [8.0]]]]
    )
    valid = raw_valid_mask(target)
    metrics = RawMetricAccumulator(horizons=2)
    metrics.update(prediction, target, valid)
    report = metrics.compute()

    assert math.isclose(report.mae, masked_mae(prediction, target, valid).item(), rel_tol=1e-6)
    assert math.isclose(report.rmse, masked_rmse(prediction, target, valid).item(), rel_tol=1e-6)
    assert math.isclose(report.mape, masked_mape(prediction, target, valid).item(), rel_tol=1e-6)


def test_global_accumulation_does_not_average_batch_rmse():
    accumulator = RawMetricAccumulator(horizons=1)
    accumulator.update(torch.tensor([[[[1.0]]]]), torch.tensor([[[[0.1]]]]), torch.ones(1, 1, 1, 1, dtype=torch.bool))
    accumulator.update(
        torch.tensor([[[[3.0], [3.0], [3.0]]]]),
        torch.tensor([[[[1.0], [1.0], [1.0]]]]),
        torch.ones(1, 1, 3, 1, dtype=torch.bool),
    )
    report = accumulator.compute()
    expected = math.sqrt((0.9**2 + 3 * 2.0**2) / 4)
    assert math.isclose(report.rmse, expected, rel_tol=1e-6)
    assert not math.isclose(report.rmse, (0.9 + 2.0) / 2, rel_tol=1e-6)


def test_zero_valid_is_explicit_and_matches_basicts_zero_semantics():
    prediction = torch.ones(2, 2, 3, 1)
    target = torch.zeros_like(prediction)
    valid = torch.zeros_like(target, dtype=torch.bool)
    accumulator = RawMetricAccumulator(horizons=2)
    accumulator.update(prediction, target, valid)
    report = accumulator.compute()

    assert report.valid == 0
    assert (report.mae, report.rmse, report.mape) == (0.0, 0.0, 0.0)
    assert masked_mae(prediction, target, valid).item() == 0.0
    assert masked_rmse(prediction, target, valid).item() == 0.0
    assert masked_mape(prediction, target, valid).item() == 0.0
    assert [item.valid for item in report.per_horizon] == [0, 0]
