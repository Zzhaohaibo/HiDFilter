from __future__ import annotations

import torch
from torch import nn


def self_only_forward(model: nn.Module, history: torch.Tensor) -> torch.Tensor:
    """Thin runtime adapter for the independent Self-only HiDFilter module."""

    return model(history)
