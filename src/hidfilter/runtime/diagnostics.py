from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable

import torch
from basicts.scaler import ZScoreScaler
from torch import nn
from torch.utils.data import DataLoader

from hidfilter.model import (
    FAMILY_COUNT,
    FORECAST_HORIZON,
    HISTORY_LENGTH,
    FineDiagnosticState,
    HiDFilterDiagnosticState,
)
from hidfilter.protocol.metrics import RawMetricAccumulator, RawMetricReport
from hidfilter.protocol.pems08 import prepare_batch


FAMILY_NAMES = ("Self", "Physical", "Semantic")
DIAGNOSTICS_VERSION = 1
_MIN_SENTINEL = 2**60


def effective_edge_support(
    candidate_valid: torch.Tensor,
    edge_keep: torch.Tensor,
    family_keep: torch.Tensor,
) -> torch.Tensor:
    """Return valid & Edge-Top-p & Family-Top-p support."""

    valid = candidate_valid.view(1, candidate_valid.shape[0], 1, -1)
    return valid & edge_keep & family_keep.unsqueeze(-1)


@dataclass(frozen=True)
class DiagnosticEvaluation:
    metrics: RawMetricReport
    diagnostics: dict[str, object]
    samples: int
    seconds: float
    samples_per_second: float


class DiagnosticAccumulator:
    """Device-side streaming reductions for the frozen Phase 6 definitions."""

    def __init__(self, *, device: torch.device) -> None:
        self.device = device
        float_type = torch.float64
        int_type = torch.int64

        self.query_count = torch.zeros((), dtype=int_type, device=device)
        self.horizon_query_count = torch.zeros(
            FORECAST_HORIZON, dtype=int_type, device=device
        )
        self.family_available_count = torch.zeros(
            FAMILY_COUNT, dtype=int_type, device=device
        )
        self.family_keep_count = torch.zeros(
            FAMILY_COUNT, dtype=int_type, device=device
        )
        self.family_dense_sum = torch.zeros(
            FAMILY_COUNT, dtype=float_type, device=device
        )
        self.family_sparse_sum = torch.zeros_like(self.family_dense_sum)
        self.family_retained_contribution_sum = torch.zeros_like(self.family_dense_sum)
        self.family_entropy_sum = torch.zeros((), dtype=float_type, device=device)
        self.family_support_sum = torch.zeros((), dtype=int_type, device=device)
        self.family_support_min = torch.full(
            (), _MIN_SENTINEL, dtype=int_type, device=device
        )
        self.family_support_max = torch.zeros((), dtype=int_type, device=device)
        self.family_support_histogram = torch.zeros(4, dtype=int_type, device=device)

        horizon_family_shape = (FORECAST_HORIZON, FAMILY_COUNT)
        self.horizon_family_available_count = torch.zeros(
            horizon_family_shape, dtype=int_type, device=device
        )
        self.horizon_family_keep_count = torch.zeros_like(
            self.horizon_family_available_count
        )
        self.horizon_family_dense_sum = torch.zeros(
            horizon_family_shape, dtype=float_type, device=device
        )
        self.horizon_family_sparse_sum = torch.zeros_like(
            self.horizon_family_dense_sum
        )
        self.horizon_family_contribution_sum = torch.zeros_like(
            self.horizon_family_dense_sum
        )
        self.horizon_family_entropy_sum = torch.zeros(
            FORECAST_HORIZON, dtype=float_type, device=device
        )
        self.horizon_family_support_sum = torch.zeros(
            FORECAST_HORIZON, dtype=int_type, device=device
        )

        family_horizon_shape = (FAMILY_COUNT, FORECAST_HORIZON)
        self.dense_valid_count_sum = torch.zeros(
            FAMILY_COUNT, dtype=int_type, device=device
        )
        self.dense_valid_count_min = torch.full(
            (FAMILY_COUNT,), _MIN_SENTINEL, dtype=int_type, device=device
        )
        self.dense_valid_count_max = torch.zeros(
            FAMILY_COUNT, dtype=int_type, device=device
        )
        self.edge_count_sum = torch.zeros(FAMILY_COUNT, dtype=int_type, device=device)
        self.edge_count_min = torch.full(
            (FAMILY_COUNT,), _MIN_SENTINEL, dtype=int_type, device=device
        )
        self.edge_count_max = torch.zeros(FAMILY_COUNT, dtype=int_type, device=device)
        self.effective_count_sum = torch.zeros(
            FAMILY_COUNT, dtype=int_type, device=device
        )
        self.effective_count_min = torch.full(
            (FAMILY_COUNT,), _MIN_SENTINEL, dtype=int_type, device=device
        )
        self.effective_count_max = torch.zeros(
            FAMILY_COUNT, dtype=int_type, device=device
        )
        self.edge_retained_mass_sum = torch.zeros(
            FAMILY_COUNT, dtype=float_type, device=device
        )
        self.edge_mass_denominator = torch.zeros(
            FAMILY_COUNT, dtype=int_type, device=device
        )
        self.lag_histogram = torch.zeros(
            (FAMILY_COUNT, HISTORY_LENGTH), dtype=int_type, device=device
        )
        self.unique_source_sum = torch.zeros(
            FAMILY_COUNT, dtype=int_type, device=device
        )
        self.multi_lag_numerator = torch.zeros(
            FAMILY_COUNT, dtype=int_type, device=device
        )
        self.multi_lag_denominator = torch.zeros(
            FAMILY_COUNT, dtype=int_type, device=device
        )

        self.horizon_edge_count_sum = torch.zeros(
            family_horizon_shape, dtype=int_type, device=device
        )
        self.horizon_effective_count_sum = torch.zeros_like(
            self.horizon_edge_count_sum
        )
        self.horizon_edge_mass_sum = torch.zeros(
            family_horizon_shape, dtype=float_type, device=device
        )
        self.horizon_edge_mass_denominator = torch.zeros_like(
            self.horizon_edge_count_sum
        )
        self.horizon_lag_histogram = torch.zeros(
            (FAMILY_COUNT, FORECAST_HORIZON, HISTORY_LENGTH),
            dtype=int_type,
            device=device,
        )
        self.horizon_unique_source_sum = torch.zeros_like(
            self.horizon_edge_count_sum
        )
        self.horizon_multi_lag_numerator = torch.zeros_like(
            self.horizon_edge_count_sum
        )
        self.horizon_multi_lag_denominator = torch.zeros_like(
            self.horizon_edge_count_sum
        )

        self.overall_effective_sum = torch.zeros((), dtype=int_type, device=device)
        self.overall_effective_min = torch.full(
            (), _MIN_SENTINEL, dtype=int_type, device=device
        )
        self.overall_effective_max = torch.zeros((), dtype=int_type, device=device)
        self.overall_lag_histogram = torch.zeros(
            HISTORY_LENGTH, dtype=int_type, device=device
        )
        self.overall_unique_source_sum = torch.zeros((), dtype=int_type, device=device)
        self.overall_multi_lag_numerator = torch.zeros((), dtype=int_type, device=device)
        self.overall_multi_lag_denominator = torch.zeros((), dtype=int_type, device=device)
        self.horizon_overall_effective_sum = torch.zeros(
            FORECAST_HORIZON, dtype=int_type, device=device
        )
        self.horizon_overall_lag_histogram = torch.zeros(
            (FORECAST_HORIZON, HISTORY_LENGTH), dtype=int_type, device=device
        )
        self.horizon_overall_unique_source_sum = torch.zeros(
            FORECAST_HORIZON, dtype=int_type, device=device
        )
        self.horizon_overall_multi_lag_numerator = torch.zeros(
            FORECAST_HORIZON, dtype=int_type, device=device
        )
        self.horizon_overall_multi_lag_denominator = torch.zeros(
            FORECAST_HORIZON, dtype=int_type, device=device
        )
        self.invariant_violations = torch.zeros((), dtype=int_type, device=device)

    def update(
        self,
        state: HiDFilterDiagnosticState,
        valid_query: torch.Tensor,
    ) -> None:
        if valid_query.dtype != torch.bool or valid_query.ndim != 3:
            raise ValueError("valid_query must be bool [B,N,H]")
        dense = state.family_dense_probability
        if dense.shape[:-1] != valid_query.shape or dense.shape[-1] != FAMILY_COUNT:
            raise ValueError("diagnostic family tensors and valid_query have incompatible shapes")
        if valid_query.shape[-1] != FORECAST_HORIZON:
            raise ValueError("diagnostics require 12 forecast horizons")

        query = valid_query
        query_family = query.unsqueeze(-1)
        availability = state.family_available.view(
            1, state.family_available.shape[0], 1, FAMILY_COUNT
        )
        availability = availability.expand_as(dense)
        keep = state.family_keep
        sparse = state.family_retained_weight

        self._accumulate_family_invariants(dense, keep, sparse, availability)
        query_count = query.sum(dtype=torch.int64)
        horizon_query_count = query.sum(dim=(0, 1), dtype=torch.int64)
        self.query_count += query_count
        self.horizon_query_count += horizon_query_count

        available_query = query_family & availability
        kept_query = query_family & keep
        self.family_available_count += available_query.sum(
            dim=(0, 1, 2), dtype=torch.int64
        )
        self.family_keep_count += kept_query.sum(dim=(0, 1, 2), dtype=torch.int64)
        self.family_dense_sum += torch.where(
            query_family, dense, torch.zeros_like(dense)
        ).sum(dim=(0, 1, 2), dtype=torch.float64)
        self.family_sparse_sum += torch.where(
            query_family, sparse, torch.zeros_like(sparse)
        ).sum(dim=(0, 1, 2), dtype=torch.float64)
        contribution = dense * keep.to(dense.dtype)
        self.family_retained_contribution_sum += torch.where(
            query_family, contribution, torch.zeros_like(contribution)
        ).sum(dim=(0, 1, 2), dtype=torch.float64)
        entropy = -torch.where(
            dense > 0,
            dense * torch.log(dense),
            torch.zeros_like(dense),
        ).sum(dim=-1)
        self.family_entropy_sum += torch.where(
            query, entropy, torch.zeros_like(entropy)
        ).sum(dtype=torch.float64)

        support_count = keep.sum(dim=-1, dtype=torch.int64)
        self.family_support_sum += torch.where(
            query, support_count, torch.zeros_like(support_count)
        ).sum(dtype=torch.int64)
        self.family_support_min = torch.minimum(
            self.family_support_min,
            torch.where(
                query,
                support_count,
                torch.full_like(support_count, _MIN_SENTINEL),
            ).amin(),
        )
        self.family_support_max = torch.maximum(
            self.family_support_max,
            torch.where(query, support_count, torch.zeros_like(support_count)).amax(),
        )
        for count in range(1, FAMILY_COUNT + 1):
            self.family_support_histogram[count] += (
                query & (support_count == count)
            ).sum(dtype=torch.int64)

        self.horizon_family_available_count += available_query.sum(
            dim=(0, 1), dtype=torch.int64
        )
        self.horizon_family_keep_count += kept_query.sum(
            dim=(0, 1), dtype=torch.int64
        )
        self.horizon_family_dense_sum += torch.where(
            query_family, dense, torch.zeros_like(dense)
        ).sum(dim=(0, 1), dtype=torch.float64)
        self.horizon_family_sparse_sum += torch.where(
            query_family, sparse, torch.zeros_like(sparse)
        ).sum(dim=(0, 1), dtype=torch.float64)
        self.horizon_family_contribution_sum += torch.where(
            query_family, contribution, torch.zeros_like(contribution)
        ).sum(dim=(0, 1), dtype=torch.float64)
        self.horizon_family_entropy_sum += torch.where(
            query, entropy, torch.zeros_like(entropy)
        ).sum(dim=(0, 1), dtype=torch.float64)
        self.horizon_family_support_sum += torch.where(
            query, support_count, torch.zeros_like(support_count)
        ).sum(dim=(0, 1), dtype=torch.int64)

        overall_effective_count = torch.zeros_like(support_count)
        overall_source_lag: list[torch.Tensor] = []
        source_slot_indices: list[torch.Tensor] = []
        source_slot_validities: list[torch.Tensor] = []
        for family_id, fine in enumerate(state.fine):
            effective, source_lag = self._accumulate_fine(
                family_id,
                fine,
                keep[..., family_id],
                availability[..., family_id],
                query,
            )
            overall_effective_count += effective.sum(dim=-1, dtype=torch.int64)
            overall_source_lag.append(source_lag)
            source_slot_indices.append(fine.source_index[:, ::HISTORY_LENGTH])
            source_slot_validities.append(fine.candidate_valid[:, ::HISTORY_LENGTH])

        self._accumulate_overall(
            overall_effective_count,
            overall_source_lag,
            source_slot_indices,
            source_slot_validities,
            query,
        )

    def _accumulate_family_invariants(
        self,
        dense: torch.Tensor,
        keep: torch.Tensor,
        sparse: torch.Tensor,
        availability: torch.Tensor,
    ) -> None:
        row_has_available = availability.any(dim=-1)
        weight_sum = sparse.sum(dim=-1)
        self.invariant_violations += (
            row_has_available & ~torch.isclose(weight_sum, torch.ones_like(weight_sum), atol=1e-6)
        ).sum(dtype=torch.int64)
        self.invariant_violations += (keep & ~availability).sum(dtype=torch.int64)
        self.invariant_violations += (sparse[~availability] != 0).sum(dtype=torch.int64)
        self.invariant_violations += (dense[~availability] != 0).sum(dtype=torch.int64)
        self.invariant_violations += (~torch.isfinite(dense)).sum(dtype=torch.int64)
        self.invariant_violations += (~torch.isfinite(sparse)).sum(dtype=torch.int64)

    def _accumulate_fine(
        self,
        family_id: int,
        fine: FineDiagnosticState,
        family_keep: torch.Tensor,
        family_available: torch.Tensor,
        query: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        valid = fine.candidate_valid.view(1, fine.candidate_valid.shape[0], 1, -1)
        effective = effective_edge_support(
            fine.candidate_valid, fine.edge_keep, family_keep
        )
        self.invariant_violations += (fine.edge_keep & ~valid).sum(dtype=torch.int64)
        self.invariant_violations += (
            effective & ~(valid & fine.edge_keep & family_keep.unsqueeze(-1))
        ).sum(dtype=torch.int64)
        self.invariant_violations += (
            effective & ~family_keep.unsqueeze(-1)
        ).sum(dtype=torch.int64)

        valid_count_static = fine.candidate_valid.sum(dim=-1, dtype=torch.int64)
        valid_count = valid_count_static.view(1, -1, 1).expand_as(query)
        edge_count = (fine.edge_keep & valid).sum(dim=-1, dtype=torch.int64)
        effective_count = effective.sum(dim=-1, dtype=torch.int64)
        self.invariant_violations += (effective_count > edge_count).sum(dtype=torch.int64)

        self._accumulate_count_summary(
            family_id,
            valid_count,
            self.dense_valid_count_sum,
            self.dense_valid_count_min,
            self.dense_valid_count_max,
            query,
        )
        self._accumulate_count_summary(
            family_id,
            edge_count,
            self.edge_count_sum,
            self.edge_count_min,
            self.edge_count_max,
            query,
        )
        self._accumulate_count_summary(
            family_id,
            effective_count,
            self.effective_count_sum,
            self.effective_count_min,
            self.effective_count_max,
            query,
        )
        self.horizon_edge_count_sum[family_id] += torch.where(
            query, edge_count, torch.zeros_like(edge_count)
        ).sum(dim=(0, 1), dtype=torch.int64)
        self.horizon_effective_count_sum[family_id] += torch.where(
            query, effective_count, torch.zeros_like(effective_count)
        ).sum(dim=(0, 1), dtype=torch.int64)

        available_query = query & family_available
        retained_mass = (
            fine.dense_probability * fine.edge_keep.to(fine.dense_probability.dtype)
        ).sum(dim=-1)
        self.edge_retained_mass_sum[family_id] += torch.where(
            available_query, retained_mass, torch.zeros_like(retained_mass)
        ).sum(dtype=torch.float64)
        self.edge_mass_denominator[family_id] += available_query.sum(dtype=torch.int64)
        self.horizon_edge_mass_sum[family_id] += torch.where(
            available_query, retained_mass, torch.zeros_like(retained_mass)
        ).sum(dim=(0, 1), dtype=torch.float64)
        self.horizon_edge_mass_denominator[family_id] += available_query.sum(
            dim=(0, 1), dtype=torch.int64
        )

        query_edge = query.unsqueeze(-1) & effective
        for lag in range(HISTORY_LENGTH):
            lag_mask = fine.lag_index.view(
                1, fine.lag_index.shape[0], 1, -1
            ) == lag
            selected = query_edge & lag_mask
            self.lag_histogram[family_id, lag] += selected.sum(dtype=torch.int64)
            self.horizon_lag_histogram[family_id, :, lag] += selected.sum(
                dim=(0, 1, 3), dtype=torch.int64
            )

        candidate_count = fine.candidate_valid.shape[-1]
        if candidate_count % HISTORY_LENGTH != 0:
            raise ValueError("Fine candidates must contain complete 12-lag source blocks")
        source_slots = candidate_count // HISTORY_LENGTH
        source_lag = effective.reshape(
            *effective.shape[:-1], source_slots, HISTORY_LENGTH
        )
        if source_slots:
            source_blocks = fine.source_index.reshape(
                fine.source_index.shape[0], source_slots, HISTORY_LENGTH
            )
            lag_blocks = fine.lag_index.reshape(
                fine.lag_index.shape[0], source_slots, HISTORY_LENGTH
            )
            valid_blocks = fine.candidate_valid.reshape(
                fine.candidate_valid.shape[0], source_slots, HISTORY_LENGTH
            )
            expected_lags = torch.arange(
                HISTORY_LENGTH, dtype=lag_blocks.dtype, device=lag_blocks.device
            ).view(1, 1, HISTORY_LENGTH)
            self.invariant_violations += (
                source_blocks != source_blocks[..., :1]
            ).sum(dtype=torch.int64)
            self.invariant_violations += (lag_blocks != expected_lags).sum(dtype=torch.int64)
            self.invariant_violations += (
                valid_blocks != valid_blocks[..., :1]
            ).sum(dtype=torch.int64)
        lag_count = source_lag.sum(dim=-1, dtype=torch.int64)
        source_active = lag_count > 0
        unique_count = source_active.sum(dim=-1, dtype=torch.int64)
        self.unique_source_sum[family_id] += torch.where(
            query, unique_count, torch.zeros_like(unique_count)
        ).sum(dtype=torch.int64)
        self.horizon_unique_source_sum[family_id] += torch.where(
            query, unique_count, torch.zeros_like(unique_count)
        ).sum(dim=(0, 1), dtype=torch.int64)
        multi = source_active & (lag_count > 1)
        query_source = query.unsqueeze(-1)
        self.multi_lag_numerator[family_id] += (query_source & multi).sum(
            dtype=torch.int64
        )
        self.multi_lag_denominator[family_id] += (query_source & source_active).sum(
            dtype=torch.int64
        )
        self.horizon_multi_lag_numerator[family_id] += (
            query_source & multi
        ).sum(dim=(0, 1, 3), dtype=torch.int64)
        self.horizon_multi_lag_denominator[family_id] += (
            query_source & source_active
        ).sum(dim=(0, 1, 3), dtype=torch.int64)
        return effective, source_lag

    @staticmethod
    def _accumulate_count_summary(
        family_id: int,
        count: torch.Tensor,
        total: torch.Tensor,
        minimum: torch.Tensor,
        maximum: torch.Tensor,
        query: torch.Tensor,
    ) -> None:
        total[family_id] += torch.where(query, count, torch.zeros_like(count)).sum(
            dtype=torch.int64
        )
        minimum[family_id] = torch.minimum(
            minimum[family_id],
            torch.where(
                query, count, torch.full_like(count, _MIN_SENTINEL)
            ).amin(),
        )
        maximum[family_id] = torch.maximum(
            maximum[family_id],
            torch.where(query, count, torch.zeros_like(count)).amax(),
        )

    def _accumulate_overall(
        self,
        effective_count: torch.Tensor,
        source_lag_parts: Iterable[torch.Tensor],
        source_index_parts: Iterable[torch.Tensor],
        source_valid_parts: Iterable[torch.Tensor],
        query: torch.Tensor,
    ) -> None:
        self.overall_effective_sum += torch.where(
            query, effective_count, torch.zeros_like(effective_count)
        ).sum(dtype=torch.int64)
        self.overall_effective_min = torch.minimum(
            self.overall_effective_min,
            torch.where(
                query, effective_count, torch.full_like(effective_count, _MIN_SENTINEL)
            ).amin(),
        )
        self.overall_effective_max = torch.maximum(
            self.overall_effective_max,
            torch.where(query, effective_count, torch.zeros_like(effective_count)).amax(),
        )
        self.horizon_overall_effective_sum += torch.where(
            query, effective_count, torch.zeros_like(effective_count)
        ).sum(dim=(0, 1), dtype=torch.int64)

        source_lag = torch.cat(tuple(source_lag_parts), dim=-2)
        source_index = torch.cat(tuple(source_index_parts), dim=-1)
        source_valid = torch.cat(tuple(source_valid_parts), dim=-1)
        if source_index.shape[-1] > 1:
            same = source_index.unsqueeze(-1) == source_index.unsqueeze(-2)
            both_valid = source_valid.unsqueeze(-1) & source_valid.unsqueeze(-2)
            upper = torch.triu(
                torch.ones(
                    (source_index.shape[-1], source_index.shape[-1]),
                    dtype=torch.bool,
                    device=source_index.device,
                ),
                diagonal=1,
            )
            self.invariant_violations += (same & both_valid & upper).sum(dtype=torch.int64)
        lag_count = source_lag.sum(dim=-1, dtype=torch.int64)
        source_active = lag_count > 0
        unique_count = source_active.sum(dim=-1, dtype=torch.int64)
        query_source = query.unsqueeze(-1)
        multi = source_active & (lag_count > 1)
        self.overall_unique_source_sum += torch.where(
            query, unique_count, torch.zeros_like(unique_count)
        ).sum(dtype=torch.int64)
        self.overall_multi_lag_numerator += (query_source & multi).sum(dtype=torch.int64)
        self.overall_multi_lag_denominator += (
            query_source & source_active
        ).sum(dtype=torch.int64)
        self.horizon_overall_unique_source_sum += torch.where(
            query, unique_count, torch.zeros_like(unique_count)
        ).sum(dim=(0, 1), dtype=torch.int64)
        self.horizon_overall_multi_lag_numerator += (
            query_source & multi
        ).sum(dim=(0, 1, 3), dtype=torch.int64)
        self.horizon_overall_multi_lag_denominator += (
            query_source & source_active
        ).sum(dim=(0, 1, 3), dtype=torch.int64)
        self.overall_lag_histogram.copy_(self.lag_histogram.sum(dim=0))
        self.horizon_overall_lag_histogram.copy_(
            self.horizon_lag_histogram.sum(dim=0)
        )

    def finalize(self) -> dict[str, object]:
        float_fields = self._float_fields()
        int_fields = self._int_fields()
        floats = _compact_host_snapshot(float_fields)
        integers = _compact_host_snapshot(int_fields)
        query_count = int(integers["query_count"])
        if query_count == 0:
            raise RuntimeError("diagnostic loader contains no valid evaluation queries")
        if int(integers["invariant_violations"]) != 0:
            raise RuntimeError("diagnostic invariant violation")
        if int(integers["overall_lag_histogram"].sum()) != int(
            integers["overall_effective_sum"]
        ):
            raise RuntimeError("diagnostic invariant violation: lag histogram is not conserved")
        for family_id in range(FAMILY_COUNT):
            if int(integers["lag_histogram"][family_id].sum()) != int(
                integers["effective_count_sum"][family_id]
            ):
                raise RuntimeError(
                    "diagnostic invariant violation: family lag histogram is not conserved"
                )
        return self._build_report(floats, integers, query_count)

    def _float_fields(self) -> dict[str, torch.Tensor]:
        return {
            "family_dense_sum": self.family_dense_sum,
            "family_sparse_sum": self.family_sparse_sum,
            "family_retained_contribution_sum": self.family_retained_contribution_sum,
            "family_entropy_sum": self.family_entropy_sum,
            "horizon_family_dense_sum": self.horizon_family_dense_sum,
            "horizon_family_sparse_sum": self.horizon_family_sparse_sum,
            "horizon_family_contribution_sum": self.horizon_family_contribution_sum,
            "horizon_family_entropy_sum": self.horizon_family_entropy_sum,
            "edge_retained_mass_sum": self.edge_retained_mass_sum,
            "horizon_edge_mass_sum": self.horizon_edge_mass_sum,
        }

    def _int_fields(self) -> dict[str, torch.Tensor]:
        names = (
            "query_count",
            "horizon_query_count",
            "family_available_count",
            "family_keep_count",
            "family_support_sum",
            "family_support_min",
            "family_support_max",
            "family_support_histogram",
            "horizon_family_available_count",
            "horizon_family_keep_count",
            "horizon_family_support_sum",
            "dense_valid_count_sum",
            "dense_valid_count_min",
            "dense_valid_count_max",
            "edge_count_sum",
            "edge_count_min",
            "edge_count_max",
            "effective_count_sum",
            "effective_count_min",
            "effective_count_max",
            "edge_mass_denominator",
            "lag_histogram",
            "unique_source_sum",
            "multi_lag_numerator",
            "multi_lag_denominator",
            "horizon_edge_count_sum",
            "horizon_effective_count_sum",
            "horizon_edge_mass_denominator",
            "horizon_lag_histogram",
            "horizon_unique_source_sum",
            "horizon_multi_lag_numerator",
            "horizon_multi_lag_denominator",
            "overall_effective_sum",
            "overall_effective_min",
            "overall_effective_max",
            "overall_lag_histogram",
            "overall_unique_source_sum",
            "overall_multi_lag_numerator",
            "overall_multi_lag_denominator",
            "horizon_overall_effective_sum",
            "horizon_overall_lag_histogram",
            "horizon_overall_unique_source_sum",
            "horizon_overall_multi_lag_numerator",
            "horizon_overall_multi_lag_denominator",
            "invariant_violations",
        )
        return {name: getattr(self, name) for name in names}

    def _build_report(
        self,
        floats: dict[str, torch.Tensor],
        integers: dict[str, torch.Tensor],
        query_count: int,
    ) -> dict[str, object]:
        family_dense = _named_ratio(floats["family_dense_sum"], query_count)
        family_sparse = _named_ratio(floats["family_sparse_sum"], query_count)
        retention = _named_ratio(integers["family_keep_count"], query_count)
        availability = _named_ratio(integers["family_available_count"], query_count)
        activation = _named_optional_ratio(
            integers["family_keep_count"], integers["family_available_count"]
        )
        contributions = _named_ratio(
            floats["family_retained_contribution_sum"], query_count
        )
        support_hist = integers["family_support_histogram"]
        family_report = {
            "order": list(FAMILY_NAMES),
            "availability_rate": availability,
            "activation_conditioned_on_availability": activation,
            "dense_probability_mean": family_dense,
            "retained_weight_mean": family_sparse,
            "retention_frequency": retention,
            "retained_mass_mean": float(
                floats["family_retained_contribution_sum"].sum() / query_count
            ),
            "retained_contribution_mean": contributions,
            "dense_router_entropy_mean": float(floats["family_entropy_sum"] / query_count),
            "active_family_count": {
                "mean": float(integers["family_support_sum"] / query_count),
                "minimum": int(integers["family_support_min"]),
                "maximum": int(integers["family_support_max"]),
                "fractions": {
                    str(count): float(support_hist[count] / query_count)
                    for count in range(1, FAMILY_COUNT + 1)
                },
            },
            "per_horizon": self._family_horizon_report(floats, integers),
        }

        fine_report: dict[str, object] = {}
        for family_id, family_name in enumerate(FAMILY_NAMES):
            fine_report[family_name] = self._fine_family_report(
                family_id, floats, integers, query_count
            )
        fine_report["overall"] = self._overall_report(integers, query_count)
        return {
            "diagnostics_version": DIAGNOSTICS_VERSION,
            "valid_query_count": query_count,
            "family": family_report,
            "fine": fine_report,
            "definitions": {
                "valid_query": "raw evaluator target-valid query mask",
                "effective_edge": "candidate_valid & edge_keep & family_keep",
                "lag_index_0": "paper tau=1 (latest history)",
                "lag_index_11": "paper tau=12 (oldest history)",
                "multi_lag_group": "(batch,target,horizon,source), OR across families",
            },
        }

    def _family_horizon_report(
        self,
        floats: dict[str, torch.Tensor],
        integers: dict[str, torch.Tensor],
    ) -> list[dict[str, object]]:
        rows = []
        for horizon in range(FORECAST_HORIZON):
            count = int(integers["horizon_query_count"][horizon])
            if count == 0:
                rows.append({"horizon": horizon + 1, "valid_query_count": 0})
                continue
            rows.append(
                {
                    "horizon": horizon + 1,
                    "valid_query_count": count,
                    "availability_rate": _named_ratio(
                        integers["horizon_family_available_count"][horizon], count
                    ),
                    "activation_conditioned_on_availability": _named_optional_ratio(
                        integers["horizon_family_keep_count"][horizon],
                        integers["horizon_family_available_count"][horizon],
                    ),
                    "dense_probability_mean": _named_ratio(
                        floats["horizon_family_dense_sum"][horizon], count
                    ),
                    "retained_weight_mean": _named_ratio(
                        floats["horizon_family_sparse_sum"][horizon], count
                    ),
                    "retention_frequency": _named_ratio(
                        integers["horizon_family_keep_count"][horizon], count
                    ),
                    "retained_mass_mean": float(
                        floats["horizon_family_contribution_sum"][horizon].sum()
                        / count
                    ),
                    "retained_contribution_mean": _named_ratio(
                        floats["horizon_family_contribution_sum"][horizon], count
                    ),
                    "dense_router_entropy_mean": float(
                        floats["horizon_family_entropy_sum"][horizon] / count
                    ),
                    "mean_retained_family_count": float(
                        integers["horizon_family_support_sum"][horizon] / count
                    ),
                }
            )
        return rows

    def _fine_family_report(
        self,
        family_id: int,
        floats: dict[str, torch.Tensor],
        integers: dict[str, torch.Tensor],
        query_count: int,
    ) -> dict[str, object]:
        lag_counts = integers["lag_histogram"][family_id]
        lag_total = int(lag_counts.sum())
        numerator = int(integers["multi_lag_numerator"][family_id])
        denominator = int(integers["multi_lag_denominator"][family_id])
        edge_mass_denominator = int(integers["edge_mass_denominator"][family_id])
        return {
            "dense_valid_candidate_count": _count_summary(
                integers["dense_valid_count_sum"][family_id],
                integers["dense_valid_count_min"][family_id],
                integers["dense_valid_count_max"][family_id],
                query_count,
            ),
            "edge_retained_count": _count_summary(
                integers["edge_count_sum"][family_id],
                integers["edge_count_min"][family_id],
                integers["edge_count_max"][family_id],
                query_count,
            ),
            "edge_retained_mass_mean": (
                float(floats["edge_retained_mass_sum"][family_id] / edge_mass_denominator)
                if edge_mass_denominator
                else None
            ),
            "effective_edge_count": _count_summary(
                integers["effective_count_sum"][family_id],
                integers["effective_count_min"][family_id],
                integers["effective_count_max"][family_id],
                query_count,
            ),
            "lag_histogram": _lag_histogram(lag_counts),
            "mean_unique_source_count": float(
                integers["unique_source_sum"][family_id] / query_count
            ),
            "mean_effective_sensor_lag_count": float(
                integers["effective_count_sum"][family_id] / query_count
            ),
            "effective_lag_per_unique_source": _optional_ratio(
                int(integers["effective_count_sum"][family_id]),
                int(integers["unique_source_sum"][family_id]),
            ),
            "within_family_multi_lag": {
                "numerator": numerator,
                "denominator": denominator,
                "ratio": _optional_ratio(numerator, denominator),
            },
            "per_horizon": self._fine_horizon_report(
                family_id, floats, integers
            ),
        }

    def _fine_horizon_report(
        self,
        family_id: int,
        floats: dict[str, torch.Tensor],
        integers: dict[str, torch.Tensor],
    ) -> list[dict[str, object]]:
        rows = []
        for horizon in range(FORECAST_HORIZON):
            query_count = int(integers["horizon_query_count"][horizon])
            if query_count == 0:
                rows.append({"horizon": horizon + 1, "valid_query_count": 0})
                continue
            mass_denominator = int(
                integers["horizon_edge_mass_denominator"][family_id, horizon]
            )
            numerator = int(
                integers["horizon_multi_lag_numerator"][family_id, horizon]
            )
            denominator = int(
                integers["horizon_multi_lag_denominator"][family_id, horizon]
            )
            rows.append(
                {
                    "horizon": horizon + 1,
                    "valid_query_count": query_count,
                    "mean_edge_retained_count": float(
                        integers["horizon_edge_count_sum"][family_id, horizon]
                        / query_count
                    ),
                    "mean_effective_edge_count": float(
                        integers["horizon_effective_count_sum"][family_id, horizon]
                        / query_count
                    ),
                    "edge_retained_mass_mean": (
                        float(
                            floats["horizon_edge_mass_sum"][family_id, horizon]
                            / mass_denominator
                        )
                        if mass_denominator
                        else None
                    ),
                    "lag_histogram": _lag_histogram(
                        integers["horizon_lag_histogram"][family_id, horizon]
                    ),
                    "mean_unique_source_count": float(
                        integers["horizon_unique_source_sum"][family_id, horizon]
                        / query_count
                    ),
                    "within_family_multi_lag": {
                        "numerator": numerator,
                        "denominator": denominator,
                        "ratio": _optional_ratio(numerator, denominator),
                    },
                }
            )
        return rows

    def _overall_report(
        self,
        integers: dict[str, torch.Tensor],
        query_count: int,
    ) -> dict[str, object]:
        numerator = int(integers["overall_multi_lag_numerator"])
        denominator = int(integers["overall_multi_lag_denominator"])
        effective_total = int(integers["overall_effective_sum"])
        unique_total = int(integers["overall_unique_source_sum"])
        rows = []
        for horizon in range(FORECAST_HORIZON):
            horizon_queries = int(integers["horizon_query_count"][horizon])
            if horizon_queries == 0:
                rows.append({"horizon": horizon + 1, "valid_query_count": 0})
                continue
            horizon_numerator = int(
                integers["horizon_overall_multi_lag_numerator"][horizon]
            )
            horizon_denominator = int(
                integers["horizon_overall_multi_lag_denominator"][horizon]
            )
            rows.append(
                {
                    "horizon": horizon + 1,
                    "valid_query_count": horizon_queries,
                    "mean_effective_edge_count": float(
                        integers["horizon_overall_effective_sum"][horizon]
                        / horizon_queries
                    ),
                    "lag_histogram": _lag_histogram(
                        integers["horizon_overall_lag_histogram"][horizon]
                    ),
                    "mean_unique_source_count": float(
                        integers["horizon_overall_unique_source_sum"][horizon]
                        / horizon_queries
                    ),
                    "multi_lag": {
                        "numerator": horizon_numerator,
                        "denominator": horizon_denominator,
                        "ratio": _optional_ratio(
                            horizon_numerator, horizon_denominator
                        ),
                    },
                }
            )
        return {
            "effective_edge_count": _count_summary(
                integers["overall_effective_sum"],
                integers["overall_effective_min"],
                integers["overall_effective_max"],
                query_count,
            ),
            "lag_histogram": _lag_histogram(integers["overall_lag_histogram"]),
            "mean_unique_source_count": float(unique_total / query_count),
            "mean_effective_sensor_lag_count": float(effective_total / query_count),
            "effective_lag_per_unique_source": _optional_ratio(
                effective_total, unique_total
            ),
            "multi_lag": {
                "numerator": numerator,
                "denominator": denominator,
                "ratio": _optional_ratio(numerator, denominator),
            },
            "per_horizon": rows,
        }


@torch.inference_mode()
def evaluate_with_diagnostics(
    model: nn.Module,
    loader: DataLoader,
    scaler: ZScoreScaler,
    device: torch.device,
    *,
    family_top_p_enabled: bool = True,
    edge_top_p_enabled: bool = True,
) -> DiagnosticEvaluation:
    """Accumulate raw metrics and diagnostics in one configured eval pass."""

    model.eval()
    metrics = RawMetricAccumulator(horizons=FORECAST_HORIZON)
    diagnostics = DiagnosticAccumulator(device=device)
    samples = 0
    started = time.perf_counter()
    for cpu_batch in loader:
        raw_batch = {
            "inputs": cpu_batch["inputs"].to(
                device=device, dtype=torch.float32, non_blocking=True
            ),
            "targets": cpu_batch["targets"].to(
                device=device, dtype=torch.float32, non_blocking=True
            ),
        }
        prepared = prepare_batch(raw_batch, scaler)
        output = model.forward_with_diagnostics(
            prepared.inputs,
            family_top_p_enabled=family_top_p_enabled,
            edge_top_p_enabled=edge_top_p_enabled,
        )
        prediction = scaler.inverse_transform(output.prediction)
        metrics.update(prediction, raw_batch["targets"], prepared.targets_valid)
        valid_query = prepared.targets_valid[..., 0].permute(0, 2, 1)
        diagnostics.update(output.state, valid_query)
        samples += int(prepared.inputs.shape[0])
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    if samples == 0:
        raise RuntimeError("diagnostic loader is empty")
    return DiagnosticEvaluation(
        metrics=metrics.compute(),
        diagnostics=diagnostics.finalize(),
        samples=samples,
        seconds=seconds,
        samples_per_second=samples / seconds,
    )


def export_case_study(
    state: HiDFilterDiagnosticState,
    *,
    sample_index: int,
    target_sensor: int,
    horizon_index: int,
) -> dict[str, object]:
    """Export one explicitly requested position; never called by default."""

    dense = state.family_dense_probability[sample_index, target_sensor, horizon_index]
    keep = state.family_keep[sample_index, target_sensor, horizon_index]
    weight = state.family_retained_weight[sample_index, target_sensor, horizon_index]
    reduced = torch.cat(
        (dense.to(torch.float64), keep.to(torch.float64), weight.to(torch.float64))
    ).detach().cpu()
    families = []
    for family_id, fine in enumerate(state.fine):
        if not bool(reduced[FAMILY_COUNT + family_id]):
            continue
        candidate = torch.stack(
            (
                fine.source_index[target_sensor].to(torch.float64),
                fine.lag_index[target_sensor].to(torch.float64),
                fine.dense_probability[
                    sample_index, target_sensor, horizon_index
                ].to(torch.float64),
                fine.edge_keep[sample_index, target_sensor, horizon_index].to(
                    torch.float64
                ),
                fine.edge_retained_weight[
                    sample_index, target_sensor, horizon_index
                ].to(torch.float64),
                (
                    fine.candidate_valid[target_sensor]
                    & fine.edge_keep[sample_index, target_sensor, horizon_index]
                    & state.family_keep[
                        sample_index, target_sensor, horizon_index, family_id
                    ]
                ).to(torch.float64),
            ),
            dim=-1,
        ).detach().cpu()
        valid = fine.candidate_valid[target_sensor].detach().cpu()
        families.append(
            {
                "family": fine.family_name,
                "candidates": [
                    {
                        "source_index": int(row[0]),
                        "lag_index": int(row[1]),
                        "edge_dense_probability": float(row[2]),
                        "edge_keep": bool(row[3]),
                        "edge_retained_weight": float(row[4]),
                        "effective_edge": bool(row[5]),
                    }
                    for row, is_valid in zip(candidate, valid)
                    if bool(is_valid)
                ],
            }
        )
    return {
        "sample_index": sample_index,
        "target_sensor": target_sensor,
        "horizon_index": horizon_index,
        "paper_horizon": horizon_index + 1,
        "family_dense_probability": [float(value) for value in reduced[:FAMILY_COUNT]],
        "family_keep": [bool(value) for value in reduced[FAMILY_COUNT : 2 * FAMILY_COUNT]],
        "family_retained_weight": [
            float(value) for value in reduced[2 * FAMILY_COUNT :]
        ],
        "retained_families": families,
    }


def _compact_host_snapshot(fields: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    names = tuple(fields)
    shapes = {name: fields[name].shape for name in names}
    lengths = {name: fields[name].numel() for name in names}
    packed = torch.cat([fields[name].reshape(-1) for name in names]).detach().cpu()
    result: dict[str, torch.Tensor] = {}
    offset = 0
    for name in names:
        result[name] = packed[offset : offset + lengths[name]].reshape(shapes[name])
        offset += lengths[name]
    return result


def _named_ratio(values: torch.Tensor, denominator: int) -> dict[str, float]:
    return {
        family: float(values[index] / denominator)
        for index, family in enumerate(FAMILY_NAMES)
    }


def _named_optional_ratio(
    numerators: torch.Tensor, denominators: torch.Tensor
) -> dict[str, float | None]:
    return {
        family: _optional_ratio(int(numerators[index]), int(denominators[index]))
        for index, family in enumerate(FAMILY_NAMES)
    }


def _optional_ratio(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def _count_summary(
    total: torch.Tensor,
    minimum: torch.Tensor,
    maximum: torch.Tensor,
    denominator: int,
) -> dict[str, float | int]:
    return {
        "mean": float(total / denominator),
        "minimum": int(minimum),
        "maximum": int(maximum),
    }


def _lag_histogram(counts: torch.Tensor) -> dict[str, list[int] | list[float]]:
    total = int(counts.sum())
    return {
        "counts": [int(value) for value in counts],
        "fractions": (
            [float(value / total) for value in counts]
            if total
            else [0.0] * HISTORY_LENGTH
        ),
    }
