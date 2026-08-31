from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from hidfilter.physical import PhysicalSourceMetadata
from hidfilter.protocol.pems08 import TrafficOnlyForecastingDataset, raw_valid_mask_array
from hidfilter.semantic import (
    SemanticCandidateArtifact,
    build_semantic_candidates,
    load_semantic_candidate_artifact,
    save_semantic_candidate_artifact,
    semantic_candidate_fingerprint,
)


@dataclass(frozen=True)
class SemanticCandidatePreparation:
    artifact: SemanticCandidateArtifact
    seconds: float
    cache_hit: bool


def semantic_forward(model: nn.Module, history: torch.Tensor) -> torch.Tensor:
    """Thin runtime adapter for the shared three-family HiDFilter core."""

    return model(history)


def prepare_semantic_candidates(
    raw_train: np.ndarray,
    raw_valid: np.ndarray,
    adjacency: np.ndarray,
    physical_sources: PhysicalSourceMetadata,
    cache_path: str | Path,
    *,
    ks: int,
    min_overlap: int,
    variance_threshold: float,
) -> SemanticCandidatePreparation:
    """Build or load Semantic candidates from raw training traffic only."""

    started = time.perf_counter()
    raw_train = np.asarray(raw_train)
    if not np.isfinite(raw_train).all():
        raise ValueError("raw training traffic contains non-finite values")
    adjacency = np.asarray(adjacency)
    num_nodes = raw_train.shape[1]
    if adjacency.shape != (num_nodes, num_nodes):
        raise ValueError("adjacency shape does not match raw training traffic")
    one_hop_exclusion = adjacency > 0
    np.fill_diagonal(one_hop_exclusion, False)
    physical_index = physical_sources.source_index.cpu().numpy()
    physical_valid = physical_sources.valid.cpu().numpy()
    fingerprint = semantic_candidate_fingerprint(
        raw_train,
        raw_valid,
        one_hop_exclusion,
        physical_index,
        physical_valid,
        ks=ks,
        min_overlap=min_overlap,
        variance_threshold=variance_threshold,
    )
    cache_path = Path(cache_path)
    cache_hit = False
    if cache_path.is_file():
        try:
            artifact = load_semantic_candidate_artifact(
                cache_path, expected_fingerprint=fingerprint
            )
            cache_hit = True
        except ValueError:
            artifact = build_semantic_candidates(
                raw_train,
                raw_valid,
                one_hop_exclusion,
                physical_index,
                physical_valid,
                ks=ks,
                min_overlap=min_overlap,
                variance_threshold=variance_threshold,
            )
            save_semantic_candidate_artifact(cache_path, artifact)
    else:
        artifact = build_semantic_candidates(
            raw_train,
            raw_valid,
            one_hop_exclusion,
            physical_index,
            physical_valid,
            ks=ks,
            min_overlap=min_overlap,
            variance_threshold=variance_threshold,
        )
        save_semantic_candidate_artifact(cache_path, artifact)
    return SemanticCandidatePreparation(
        artifact=artifact,
        seconds=time.perf_counter() - started,
        cache_hit=cache_hit,
    )


def prepare_pems08_semantic_candidates(
    train_dataset: TrafficOnlyForecastingDataset,
    adjacency: np.ndarray,
    physical_sources: PhysicalSourceMetadata,
    cache_path: str | Path,
    *,
    ks: int,
    min_overlap: int,
    variance_threshold: float,
) -> SemanticCandidatePreparation:
    """PEMS08 boundary that prevents val/test traffic from entering Semantic statistics."""

    if str(train_dataset.mode) != "train":
        raise ValueError("Semantic candidates must be built from the train split")
    raw_train = train_dataset.data
    return prepare_semantic_candidates(
        raw_train,
        raw_valid_mask_array(raw_train),
        adjacency,
        physical_sources,
        cache_path,
        ks=ks,
        min_overlap=min_overlap,
        variance_threshold=variance_threshold,
    )


def summarize_semantic_candidates(
    artifact: SemanticCandidateArtifact,
) -> dict[str, bool | int | float | str]:
    real_sources = artifact.sources.valid.sum(dim=-1)
    available = real_sources > 0
    selected_prior = artifact.sources.prior[artifact.sources.valid]
    selected_signed = artifact.sources.signed_corr[artifact.sources.valid]
    if selected_prior.numel() == 0:
        minimum = average = maximum = 0.0
    else:
        minimum = float(selected_prior.min().item())
        average = float(selected_prior.to(torch.float64).mean().item())
        maximum = float(selected_prior.max().item())
    return {
        "train_split_only": True,
        "ks": artifact.ks,
        "min_overlap": artifact.min_overlap,
        "variance_threshold": artifact.variance_threshold,
        "correlation_input": "first_difference",
        "prior": "abs_pearson",
        "available_target_count": int(available.sum().item()),
        "unavailable_target_count": int((~available).sum().item()),
        "min_real_sources_per_target": int(real_sources.min().item()),
        "mean_real_sources_per_target": float(real_sources.to(torch.float64).mean().item()),
        "max_real_sources_per_target": int(real_sources.max().item()),
        "eligible_pair_count_after_exclusions": artifact.eligible_pair_count,
        "min_selected_abs_corr": minimum,
        "mean_selected_abs_corr": average,
        "max_selected_abs_corr": maximum,
        "positive_signed_corr_count": int((selected_signed > 0).sum().item()),
        "negative_signed_corr_count": int((selected_signed < 0).sum().item()),
    }
