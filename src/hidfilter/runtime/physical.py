from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from hidfilter.physical import (
    PhysicalCandidateArtifact,
    PhysicalGraphContract,
    build_physical_candidates,
    convert_graph_weights,
    load_adjacency_artifact,
    load_physical_candidate_artifact,
    physical_candidate_fingerprint,
    save_physical_candidate_artifact,
)


@dataclass(frozen=True)
class PhysicalCandidatePreparation:
    artifact: PhysicalCandidateArtifact
    seconds: float
    cache_hit: bool


def physical_forward(model: nn.Module, history: torch.Tensor) -> torch.Tensor:
    """Thin runtime adapter for the shared Self + Physical HiDFilter core."""

    return model(history)


def prepare_physical_candidates(
    adjacency_path: str | Path,
    cache_path: str | Path,
    contract: PhysicalGraphContract,
    *,
    kp: int,
) -> PhysicalCandidatePreparation:
    """Load or rebuild a fingerprinted offline Physical candidate artifact."""

    started = time.perf_counter()
    adjacency = load_adjacency_artifact(adjacency_path)
    convert_graph_weights(adjacency, contract)
    fingerprint = physical_candidate_fingerprint(adjacency, contract, kp=kp)
    cache_path = Path(cache_path)
    cache_hit = False
    if cache_path.is_file():
        try:
            artifact = load_physical_candidate_artifact(
                cache_path, expected_fingerprint=fingerprint
            )
            cache_hit = True
        except ValueError:
            artifact = build_physical_candidates(adjacency, contract, kp=kp)
            save_physical_candidate_artifact(cache_path, artifact)
    else:
        artifact = build_physical_candidates(adjacency, contract, kp=kp)
        save_physical_candidate_artifact(cache_path, artifact)
    return PhysicalCandidatePreparation(
        artifact=artifact,
        seconds=time.perf_counter() - started,
        cache_hit=cache_hit,
    )


def summarize_physical_candidates(
    artifact: PhysicalCandidateArtifact,
) -> dict[str, int | float]:
    real_sources = artifact.sources.valid.sum(dim=-1)
    available = real_sources > 0
    return {
        "kp": artifact.kp,
        "available_target_count": int(available.sum().item()),
        "unavailable_target_count": int((~available).sum().item()),
        "min_real_sources_per_target": int(real_sources.min().item()),
        "mean_real_sources_per_target": float(real_sources.to(torch.float64).mean().item()),
        "max_real_sources_per_target": int(real_sources.max().item()),
    }
