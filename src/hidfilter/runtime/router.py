from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader

from hidfilter.model import FAMILY_COUNT, FORECAST_HORIZON, HiDFilter
from hidfilter.protocol.pems08 import ZScoreScaler, prepare_batch


@dataclass(frozen=True)
class RouterProbabilitySummary:
    family_order: tuple[str, str, str]
    overall_mean: tuple[float, float, float]
    minimum: tuple[float, float, float]
    maximum: tuple[float, float, float]
    positions: int
    seconds: float


def router_forward(model: nn.Module, history: torch.Tensor) -> torch.Tensor:
    """Thin runtime adapter for the shared dense-Router HiDFilter core."""

    return model(history)


@torch.inference_mode()
def collect_router_probability_summary(
    model: HiDFilter,
    loader: DataLoader,
    scaler: ZScoreScaler,
    device: torch.device,
) -> RouterProbabilitySummary:
    """Collect minimal post-training Router evidence with one final device synchronization."""

    model.eval()
    probability_sum = torch.zeros(FAMILY_COUNT, dtype=torch.float64, device=device)
    probability_min = torch.full(
        (FAMILY_COUNT,), float("inf"), dtype=torch.float32, device=device
    )
    probability_max = torch.full(
        (FAMILY_COUNT,), float("-inf"), dtype=torch.float32, device=device
    )
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
        probability = model.router_probability(prepared.inputs)
        probability_sum += probability.sum(dim=(0, 1, 2), dtype=torch.float64)
        probability_min = torch.minimum(
            probability_min, probability.amin(dim=(0, 1, 2))
        )
        probability_max = torch.maximum(
            probability_max, probability.amax(dim=(0, 1, 2))
        )
        positions += int(probability.shape[0]) * model.num_nodes * FORECAST_HORIZON
    if positions == 0:
        raise RuntimeError("Router diagnostic loader is empty")
    reduced = torch.stack(
        (
            probability_sum / positions,
            probability_min.to(torch.float64),
            probability_max.to(torch.float64),
        )
    ).cpu()
    seconds = time.perf_counter() - started
    return RouterProbabilitySummary(
        family_order=("Self", "Physical", "Semantic"),
        overall_mean=tuple(float(value) for value in reduced[0]),
        minimum=tuple(float(value) for value in reduced[1]),
        maximum=tuple(float(value) for value in reduced[2]),
        positions=positions,
        seconds=seconds,
    )
