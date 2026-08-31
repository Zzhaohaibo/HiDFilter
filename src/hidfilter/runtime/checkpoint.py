from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn


def save_model_checkpoint(path: str | Path, model: nn.Module, metadata: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "metadata": dict(metadata)}, destination)


def load_model_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    map_location: str | torch.device,
    strict: bool = True,
) -> dict[str, Any]:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="TypedStorage is deprecated.*",
            category=UserWarning,
        )
        payload = torch.load(path, map_location=map_location, weights_only=True)
    model.load_state_dict(payload["model_state"], strict=strict)
    return payload["metadata"]


class CheckpointManager:
    """Best/last model checkpoints only; deliberately contains no resume state."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.best_path = self.directory / "best.pt"
        self.last_path = self.directory / "last.pt"
        self.best_metric = float("inf")

    def save_last(self, model: nn.Module, metadata: Mapping[str, Any]) -> None:
        save_model_checkpoint(self.last_path, model, metadata)

    def maybe_save_best(self, model: nn.Module, metric: float, metadata: Mapping[str, Any]) -> bool:
        if metric >= self.best_metric:
            return False
        save_model_checkpoint(self.best_path, model, metadata)
        self.best_metric = metric
        return True
