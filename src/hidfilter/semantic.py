from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


HISTORY_LENGTH = 12
SEMANTIC_KS = 8
MIN_OVERLAP = 288
VARIANCE_THRESHOLD = 1.0e-12
_ARTIFACT_VERSION = 1
_RAW_VALIDITY_CONTRACT = "finite and not isclose(raw,0,atol=5e-5)"


@dataclass(frozen=True)
class SemanticSourceMetadata:
    source_index: torch.Tensor
    valid: torch.Tensor
    prior: torch.Tensor
    signed_corr: torch.Tensor
    overlap_count: torch.Tensor


@dataclass(frozen=True)
class SemanticCandidateMetadata:
    source_index: torch.Tensor
    lag_index: torch.Tensor
    flat_index: torch.Tensor
    valid: torch.Tensor
    prior: torch.Tensor


@dataclass(frozen=True)
class SemanticCandidateArtifact:
    ks: int
    min_overlap: int
    variance_threshold: float
    fingerprint: str
    eligible_pair_count: int
    sources: SemanticSourceMetadata
    candidates: SemanticCandidateMetadata


def compute_first_differences(
    raw_traffic: np.ndarray,
    raw_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute adjacent first differences without filling invalid positions."""

    raw = _traffic_matrix(raw_traffic)
    valid = np.asarray(raw_valid, dtype=np.bool_)
    if valid.ndim == 3 and valid.shape[-1] == 1:
        valid = valid[..., 0]
    if valid.shape != raw.shape:
        raise ValueError(f"raw validity shape {valid.shape} does not match traffic {raw.shape}")
    if not np.isfinite(raw).all():
        raise ValueError("raw training traffic contains non-finite values")

    difference = np.full(raw.shape, np.nan, dtype=np.float64)
    difference_valid = np.zeros(raw.shape, dtype=np.bool_)
    adjacent_valid = valid[1:] & valid[:-1]
    adjacent_difference = raw[1:] - raw[:-1]
    difference[1:][adjacent_valid] = adjacent_difference[adjacent_valid]
    difference_valid[1:] = adjacent_valid
    return difference, difference_valid


def compute_pairwise_pearson(
    difference: np.ndarray,
    difference_valid: np.ndarray,
    *,
    min_overlap: int = MIN_OVERLAP,
    variance_threshold: float = VARIANCE_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute CPU float64 Pearson statistics on each pair's common-valid samples."""

    values = np.asarray(difference, dtype=np.float64)
    valid = np.asarray(difference_valid, dtype=np.bool_)
    if values.ndim != 2 or valid.shape != values.shape:
        raise ValueError("difference and validity must have matching [T,N] shapes")
    if min_overlap <= 1:
        raise ValueError("min_overlap must be greater than one")
    if not math.isfinite(variance_threshold) or variance_threshold < 0.0:
        raise ValueError("variance_threshold must be finite and non-negative")

    num_nodes = values.shape[1]
    correlation = np.full((num_nodes, num_nodes), np.nan, dtype=np.float64)
    overlap_count = np.zeros((num_nodes, num_nodes), dtype=np.int64)
    for left in range(num_nodes):
        for right in range(left + 1, num_nodes):
            common = valid[:, left] & valid[:, right]
            overlap = int(np.count_nonzero(common))
            overlap_count[left, right] = overlap_count[right, left] = overlap
            if overlap < min_overlap:
                continue
            x = values[common, left]
            y = values[common, right]
            x_centered = x - x.mean(dtype=np.float64)
            y_centered = y - y.mean(dtype=np.float64)
            x_variance = float(np.mean(np.square(x_centered), dtype=np.float64))
            y_variance = float(np.mean(np.square(y_centered), dtype=np.float64))
            if x_variance <= variance_threshold or y_variance <= variance_threshold:
                continue
            denominator = math.sqrt(
                float(np.dot(x_centered, x_centered))
                * float(np.dot(y_centered, y_centered))
            )
            corr = float(np.dot(x_centered, y_centered)) / denominator
            if not math.isfinite(corr):
                continue
            corr = float(np.clip(corr, -1.0, 1.0))
            correlation[left, right] = correlation[right, left] = corr
    return correlation, overlap_count


def select_semantic_sources(
    correlation: np.ndarray,
    overlap_count: np.ndarray,
    one_hop_exclusion: np.ndarray,
    physical_source_index: np.ndarray,
    physical_source_valid: np.ndarray,
    *,
    ks: int = SEMANTIC_KS,
) -> SemanticSourceMetadata:
    """Apply explicit exclusions and deterministic abs-correlation ranking."""

    corr = np.asarray(correlation, dtype=np.float64)
    overlap = np.asarray(overlap_count, dtype=np.int64)
    one_hop = np.asarray(one_hop_exclusion, dtype=np.bool_)
    physical_index = np.asarray(physical_source_index, dtype=np.int64)
    physical_valid = np.asarray(physical_source_valid, dtype=np.bool_)
    if corr.ndim != 2 or corr.shape[0] != corr.shape[1]:
        raise ValueError("correlation must be square")
    num_nodes = corr.shape[0]
    if overlap.shape != corr.shape or one_hop.shape != corr.shape:
        raise ValueError("correlation, overlap, and one-hop exclusion shapes must match")
    if physical_index.ndim != 2 or physical_valid.shape != physical_index.shape:
        raise ValueError("Physical source index/validity shapes must match")
    if physical_index.shape[0] != num_nodes:
        raise ValueError("Physical source metadata node count does not match correlation")
    if ks <= 0:
        raise ValueError("ks must be positive")

    source_index = np.zeros((num_nodes, ks), dtype=np.int64)
    source_valid = np.zeros((num_nodes, ks), dtype=np.bool_)
    source_prior = np.zeros((num_nodes, ks), dtype=np.float32)
    signed_corr = np.zeros((num_nodes, ks), dtype=np.float32)
    selected_overlap = np.zeros((num_nodes, ks), dtype=np.int64)
    all_sources = np.arange(num_nodes, dtype=np.int64)
    for target in range(num_nodes):
        excluded = one_hop[target].copy()
        excluded[target] = True
        selected_physical = physical_index[target, physical_valid[target]]
        excluded[selected_physical] = True
        eligible = (~excluded) & np.isfinite(corr[target])
        candidates = all_sources[eligible]
        if candidates.size == 0:
            continue
        order = np.lexsort((candidates, -np.abs(corr[target, candidates])))
        selected = candidates[order[:ks]]
        count = selected.size
        selected_corr = corr[target, selected]
        source_index[target, :count] = selected
        source_valid[target, :count] = True
        source_prior[target, :count] = np.abs(selected_corr).astype(np.float32)
        signed_corr[target, :count] = selected_corr.astype(np.float32)
        selected_overlap[target, :count] = overlap[target, selected]
    return SemanticSourceMetadata(
        source_index=torch.from_numpy(source_index),
        valid=torch.from_numpy(source_valid),
        prior=torch.from_numpy(source_prior),
        signed_corr=torch.from_numpy(signed_corr),
        overlap_count=torch.from_numpy(selected_overlap),
    )


def semantic_candidate_fingerprint(
    raw_train: np.ndarray,
    raw_valid: np.ndarray,
    one_hop_exclusion: np.ndarray,
    physical_source_index: np.ndarray,
    physical_source_valid: np.ndarray,
    *,
    ks: int,
    min_overlap: int,
    variance_threshold: float,
) -> str:
    raw = np.ascontiguousarray(_traffic_matrix(raw_train), dtype=np.float64)
    valid = np.ascontiguousarray(_validity_matrix(raw_valid, raw.shape), dtype=np.bool_)
    one_hop = np.ascontiguousarray(one_hop_exclusion, dtype=np.bool_)
    physical_index = np.ascontiguousarray(physical_source_index, dtype=np.int64)
    physical_valid = np.ascontiguousarray(physical_source_valid, dtype=np.bool_)
    metadata = {
        "artifact_version": _ARTIFACT_VERSION,
        "raw_validity_contract": _RAW_VALIDITY_CONTRACT,
        "ks": ks,
        "min_overlap": min_overlap,
        "variance_threshold": variance_threshold,
        "raw_shape": list(raw.shape),
    }
    digest = hashlib.sha256()
    digest.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    for array in (raw, valid, one_hop, physical_index, physical_valid):
        digest.update(array.tobytes())
    return digest.hexdigest()


def build_semantic_candidates(
    raw_train: np.ndarray,
    raw_valid: np.ndarray,
    one_hop_exclusion: np.ndarray,
    physical_source_index: np.ndarray,
    physical_source_valid: np.ndarray,
    *,
    ks: int = SEMANTIC_KS,
    min_overlap: int = MIN_OVERLAP,
    variance_threshold: float = VARIANCE_THRESHOLD,
) -> SemanticCandidateArtifact:
    """Build deterministic train-only Semantic source and sensor-lag metadata."""

    raw = _traffic_matrix(raw_train)
    valid = _validity_matrix(raw_valid, raw.shape)
    one_hop = np.asarray(one_hop_exclusion, dtype=np.bool_)
    if one_hop.shape != (raw.shape[1], raw.shape[1]):
        raise ValueError("one-hop exclusion must have shape [N,N]")
    difference, difference_valid = compute_first_differences(raw, valid)
    correlation, overlap_count = compute_pairwise_pearson(
        difference,
        difference_valid,
        min_overlap=min_overlap,
        variance_threshold=variance_threshold,
    )
    sources = select_semantic_sources(
        correlation,
        overlap_count,
        one_hop,
        physical_source_index,
        physical_source_valid,
        ks=ks,
    )
    candidates = _expand_source_metadata(sources)
    eligible_pair_count = _eligible_pair_count(
        correlation,
        one_hop,
        np.asarray(physical_source_index, dtype=np.int64),
        np.asarray(physical_source_valid, dtype=np.bool_),
    )
    fingerprint = semantic_candidate_fingerprint(
        raw,
        valid,
        one_hop,
        physical_source_index,
        physical_source_valid,
        ks=ks,
        min_overlap=min_overlap,
        variance_threshold=variance_threshold,
    )
    return SemanticCandidateArtifact(
        ks=ks,
        min_overlap=min_overlap,
        variance_threshold=variance_threshold,
        fingerprint=fingerprint,
        eligible_pair_count=eligible_pair_count,
        sources=sources,
        candidates=candidates,
    )


def save_semantic_candidate_artifact(
    path: str | Path,
    artifact: SemanticCandidateArtifact,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "artifact_version": _ARTIFACT_VERSION,
        "ks": artifact.ks,
        "min_overlap": artifact.min_overlap,
        "variance_threshold": artifact.variance_threshold,
        "fingerprint": artifact.fingerprint,
        "eligible_pair_count": artifact.eligible_pair_count,
    }
    np.savez_compressed(
        path,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        source_index=artifact.sources.source_index.cpu().numpy(),
        source_valid=artifact.sources.valid.cpu().numpy(),
        source_prior=artifact.sources.prior.cpu().numpy(),
        source_signed_corr=artifact.sources.signed_corr.cpu().numpy(),
        source_overlap_count=artifact.sources.overlap_count.cpu().numpy(),
        candidate_source_index=artifact.candidates.source_index.cpu().numpy(),
        candidate_lag_index=artifact.candidates.lag_index.cpu().numpy(),
        candidate_flat_index=artifact.candidates.flat_index.cpu().numpy(),
        candidate_valid=artifact.candidates.valid.cpu().numpy(),
        candidate_prior=artifact.candidates.prior.cpu().numpy(),
    )


def load_semantic_candidate_artifact(
    path: str | Path,
    *,
    expected_fingerprint: str | None = None,
) -> SemanticCandidateArtifact:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Semantic candidate artifact does not exist: {path}")
    with np.load(path, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata_json"].item()))
        if metadata.get("artifact_version") != _ARTIFACT_VERSION:
            raise ValueError("unsupported Semantic candidate artifact version")
        fingerprint = str(metadata["fingerprint"])
        if expected_fingerprint is not None and fingerprint != expected_fingerprint:
            raise ValueError(
                f"Semantic candidate fingerprint mismatch: {fingerprint} != {expected_fingerprint}"
            )
        sources = SemanticSourceMetadata(
            source_index=torch.from_numpy(payload["source_index"].astype(np.int64, copy=True)),
            valid=torch.from_numpy(payload["source_valid"].astype(np.bool_, copy=True)),
            prior=torch.from_numpy(payload["source_prior"].astype(np.float32, copy=True)),
            signed_corr=torch.from_numpy(
                payload["source_signed_corr"].astype(np.float32, copy=True)
            ),
            overlap_count=torch.from_numpy(
                payload["source_overlap_count"].astype(np.int64, copy=True)
            ),
        )
        candidates = SemanticCandidateMetadata(
            source_index=torch.from_numpy(
                payload["candidate_source_index"].astype(np.int64, copy=True)
            ),
            lag_index=torch.from_numpy(payload["candidate_lag_index"].astype(np.int64, copy=True)),
            flat_index=torch.from_numpy(
                payload["candidate_flat_index"].astype(np.int64, copy=True)
            ),
            valid=torch.from_numpy(payload["candidate_valid"].astype(np.bool_, copy=True)),
            prior=torch.from_numpy(payload["candidate_prior"].astype(np.float32, copy=True)),
        )
    artifact = SemanticCandidateArtifact(
        ks=int(metadata["ks"]),
        min_overlap=int(metadata["min_overlap"]),
        variance_threshold=float(metadata["variance_threshold"]),
        fingerprint=fingerprint,
        eligible_pair_count=int(metadata["eligible_pair_count"]),
        sources=sources,
        candidates=candidates,
    )
    _validate_cached_artifact(artifact)
    return artifact


def _traffic_matrix(raw_traffic: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw_traffic, dtype=np.float64)
    if raw.ndim == 3 and raw.shape[-1] == 1:
        raw = raw[..., 0]
    if raw.ndim != 2:
        raise ValueError(f"raw training traffic must have shape [T,N] or [T,N,1], got {raw.shape}")
    return raw


def _validity_matrix(raw_valid: np.ndarray, expected_shape: tuple[int, int]) -> np.ndarray:
    valid = np.asarray(raw_valid, dtype=np.bool_)
    if valid.ndim == 3 and valid.shape[-1] == 1:
        valid = valid[..., 0]
    if valid.shape != expected_shape:
        raise ValueError(f"raw validity shape {valid.shape} does not match traffic {expected_shape}")
    return valid


def _expand_source_metadata(sources: SemanticSourceMetadata) -> SemanticCandidateMetadata:
    num_nodes, ks = sources.source_index.shape
    source_index = sources.source_index.repeat_interleave(HISTORY_LENGTH, dim=1)
    lag_index = torch.arange(HISTORY_LENGTH, dtype=torch.int64).repeat(ks).view(1, -1)
    lag_index = lag_index.expand(num_nodes, -1).clone()
    valid = sources.valid.repeat_interleave(HISTORY_LENGTH, dim=1)
    prior = sources.prior.repeat_interleave(HISTORY_LENGTH, dim=1)
    return SemanticCandidateMetadata(
        source_index=source_index,
        lag_index=lag_index,
        flat_index=source_index * HISTORY_LENGTH + lag_index,
        valid=valid,
        prior=prior,
    )


def _eligible_pair_count(
    correlation: np.ndarray,
    one_hop: np.ndarray,
    physical_index: np.ndarray,
    physical_valid: np.ndarray,
) -> int:
    eligible_count = 0
    for target in range(correlation.shape[0]):
        excluded = one_hop[target].copy()
        excluded[target] = True
        excluded[physical_index[target, physical_valid[target]]] = True
        eligible_count += int(np.count_nonzero(np.isfinite(correlation[target]) & ~excluded))
    return eligible_count


def _validate_cached_artifact(artifact: SemanticCandidateArtifact) -> None:
    sources = artifact.sources
    candidates = artifact.candidates
    if sources.source_index.ndim != 2 or sources.source_index.shape[1] != artifact.ks:
        raise ValueError("invalid Semantic source metadata shape")
    num_nodes = sources.source_index.shape[0]
    source_shape = (num_nodes, artifact.ks)
    if any(
        tensor.shape != source_shape
        for tensor in (
            sources.valid,
            sources.prior,
            sources.signed_corr,
            sources.overlap_count,
        )
    ):
        raise ValueError("inconsistent Semantic source metadata shapes")
    candidate_shape = (num_nodes, artifact.ks * HISTORY_LENGTH)
    if any(
        tensor.shape != candidate_shape
        for tensor in (
            candidates.source_index,
            candidates.lag_index,
            candidates.flat_index,
            candidates.valid,
            candidates.prior,
        )
    ):
        raise ValueError("inconsistent Semantic candidate metadata shapes")
    if not torch.isfinite(sources.prior).all() or not torch.isfinite(sources.signed_corr).all():
        raise ValueError("Semantic artifact contains non-finite selected correlation metadata")
    if not torch.isfinite(candidates.prior).all():
        raise ValueError("Semantic artifact contains non-finite candidate prior")
    if (sources.prior[sources.valid] < 0).any() or (sources.prior[sources.valid] > 1).any():
        raise ValueError("valid Semantic source prior must lie in [0,1]")
    if (sources.signed_corr[sources.valid].abs() > 1).any():
        raise ValueError("valid signed Semantic correlation must lie in [-1,1]")
    if sources.prior[~sources.valid].ne(0).any() or sources.signed_corr[~sources.valid].ne(0).any():
        raise ValueError("invalid Semantic source metadata must be zero")
    if candidates.prior[~candidates.valid].ne(0).any():
        raise ValueError("invalid Semantic candidate prior must be zero")
