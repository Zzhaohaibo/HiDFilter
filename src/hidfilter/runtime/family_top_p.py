from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader

from hidfilter.filtration import family_top_p
from hidfilter.model import FAMILY_COUNT, FAMILY_TOP_P_RHO, FORECAST_HORIZON
from hidfilter.protocol.pems08 import ZScoreScaler, prepare_batch


@dataclass(frozen=True)
class Phase5EpochPolicy:
    epoch_number: int
    family_top_p_enabled: bool
    best_selection_enabled: bool
    patience_enabled: bool


@dataclass(frozen=True)
class FamilyTopPProbabilitySummary:
    family_order: tuple[str, str, str]
    dense_overall_mean: tuple[float, float, float]
    dense_minimum: tuple[float, float, float]
    dense_maximum: tuple[float, float, float]
    retained_weight_overall_mean: tuple[float, float, float]
    retained_family_count_mean: float
    retained_family_count_min: int
    retained_family_count_max: int
    positions: int
    seconds: float


def phase5_epoch_policy(epoch_number: int) -> Phase5EpochPolicy:
    """Return the frozen 1-based Family Top-p and checkpoint policy."""

    if epoch_number <= 0:
        raise ValueError("epoch_number must be a positive 1-based epoch")
    enabled = epoch_number >= 6
    return Phase5EpochPolicy(
        epoch_number=epoch_number,
        family_top_p_enabled=enabled,
        best_selection_enabled=enabled,
        patience_enabled=enabled,
    )


def family_top_p_forward(
    enabled: bool,
) -> Callable[[nn.Module, torch.Tensor], torch.Tensor]:
    """Bind one epoch's shared train/validation Family Top-p flag."""

    def forward(model: nn.Module, history: torch.Tensor) -> torch.Tensor:
        return model(history, family_top_p_enabled=enabled)

    return forward


@torch.inference_mode()
def collect_family_top_p_summary(
    model: nn.Module,
    loader: DataLoader,
    scaler: ZScoreScaler,
    device: torch.device,
) -> FamilyTopPProbabilitySummary:
    """Reduce minimal dense/sparse family evidence before one final host transfer."""

    model.eval()
    dense_sum = torch.zeros(FAMILY_COUNT, dtype=torch.float64, device=device)
    dense_minimum = torch.full(
        (FAMILY_COUNT,), float("inf"), dtype=torch.float32, device=device
    )
    dense_maximum = torch.full(
        (FAMILY_COUNT,), float("-inf"), dtype=torch.float32, device=device
    )
    retained_sum = torch.zeros(FAMILY_COUNT, dtype=torch.float64, device=device)
    support_sum = torch.zeros((), dtype=torch.float64, device=device)
    support_minimum = torch.full(
        (), FAMILY_COUNT + 1, dtype=torch.int64, device=device
    )
    support_maximum = torch.zeros((), dtype=torch.int64, device=device)
    positions = 0
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
        dense_probability = model.router_probability(prepared.inputs)
        family_keep, family_weight = family_top_p(
            dense_probability,
            model.family_available,
            rho=FAMILY_TOP_P_RHO,
        )
        support_count = family_keep.sum(dim=-1)
        dense_sum += dense_probability.sum(dim=(0, 1, 2), dtype=torch.float64)
        dense_minimum = torch.minimum(
            dense_minimum, dense_probability.amin(dim=(0, 1, 2))
        )
        dense_maximum = torch.maximum(
            dense_maximum, dense_probability.amax(dim=(0, 1, 2))
        )
        retained_sum += family_weight.sum(dim=(0, 1, 2), dtype=torch.float64)
        support_sum += support_count.sum(dtype=torch.float64)
        support_minimum = torch.minimum(support_minimum, support_count.amin())
        support_maximum = torch.maximum(support_maximum, support_count.amax())
        positions += int(dense_probability.shape[0]) * int(model.num_nodes) * FORECAST_HORIZON

    if positions == 0:
        raise RuntimeError("Family Top-p diagnostic loader is empty")
    reduced = torch.cat(
        (
            dense_sum / positions,
            dense_minimum.to(torch.float64),
            dense_maximum.to(torch.float64),
            retained_sum / positions,
            (support_sum / positions).view(1),
            support_minimum.to(torch.float64).view(1),
            support_maximum.to(torch.float64).view(1),
        )
    ).cpu()
    seconds = time.perf_counter() - started
    if int(reduced[-2]) < 1 or int(reduced[-1]) > FAMILY_COUNT:
        raise RuntimeError("Family Top-p retained an invalid family count")
    return FamilyTopPProbabilitySummary(
        family_order=("Self", "Physical", "Semantic"),
        dense_overall_mean=tuple(float(value) for value in reduced[0:3]),
        dense_minimum=tuple(float(value) for value in reduced[3:6]),
        dense_maximum=tuple(float(value) for value in reduced[6:9]),
        retained_weight_overall_mean=tuple(float(value) for value in reduced[9:12]),
        retained_family_count_mean=float(reduced[12]),
        retained_family_count_min=int(reduced[13]),
        retained_family_count_max=int(reduced[14]),
        positions=positions,
        seconds=seconds,
    )
