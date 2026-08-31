from __future__ import annotations

import torch
from basicts.models.STID import STID, STIDConfig


def build_traffic_only_stid(num_nodes: int) -> STID:
    """Build BasicTS STID with traffic and spatial identity only."""

    config = STIDConfig(
        input_len=12,
        output_len=12,
        num_features=num_nodes,
        if_spatial=True,
        if_time_in_day=False,
        if_day_in_week=False,
    )
    return STID(config)


def stid_forward(model: STID, history: torch.Tensor) -> torch.Tensor:
    """Adapt the frozen [B,12,N,1] traffic contract to BasicTS STID."""

    if history.ndim != 4 or history.shape[1] != 12 or history.shape[-1] != 1:
        raise ValueError(f"history must have shape [B,12,N,1], got {history.shape}")
    timestamps = history.new_empty((history.shape[0], history.shape[1], 0))
    return model(history[..., 0], timestamps).unsqueeze(-1)
