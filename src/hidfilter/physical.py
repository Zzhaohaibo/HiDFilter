from __future__ import annotations

import hashlib
import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


HISTORY_LENGTH = 12
PHYSICAL_KP = 8
_ARTIFACT_VERSION = 1


@dataclass(frozen=True)
class PhysicalGraphContract:
    graph_mode: str
    weight_semantics: str
    conversion_scale: float | None

    def __post_init__(self) -> None:
        if self.graph_mode not in {"directed", "undirected"}:
            raise ValueError(f"graph_mode must be directed or undirected, got {self.graph_mode}")
        if self.weight_semantics not in {"affinity", "distance", "cost"}:
            raise ValueError(
                "weight_semantics must be affinity, distance, or cost, "
                f"got {self.weight_semantics}"
            )
        if self.weight_semantics == "affinity":
            if self.conversion_scale is not None:
                raise ValueError("conversion_scale must be null for affinity weights")
        elif (
            self.conversion_scale is None
            or not math.isfinite(float(self.conversion_scale))
            or float(self.conversion_scale) <= 0.0
        ):
            raise ValueError("conversion_scale must be a positive finite scalar for distance/cost")

    def as_dict(self) -> dict[str, str | float | None]:
        return {
            "graph_mode": self.graph_mode,
            "weight_semantics": self.weight_semantics,
            "conversion_scale": self.conversion_scale,
        }


@dataclass(frozen=True)
class PhysicalSourceMetadata:
    source_index: torch.Tensor
    valid: torch.Tensor
    prior: torch.Tensor
    shortest_hop: torch.Tensor
    path_strength: torch.Tensor


@dataclass(frozen=True)
class PhysicalCandidateMetadata:
    source_index: torch.Tensor
    lag_index: torch.Tensor
    flat_index: torch.Tensor
    valid: torch.Tensor
    prior: torch.Tensor


@dataclass(frozen=True)
class PhysicalCandidateArtifact:
    contract: PhysicalGraphContract
    kp: int
    fingerprint: str
    sources: PhysicalSourceMetadata
    candidates: PhysicalCandidateMetadata


def load_adjacency_artifact(path: str | Path) -> np.ndarray:
    """Load BasicTS ndarray or legacy (sensor_ids, mapping, adjacency) pickle."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"adjacency artifact does not exist: {path}")
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    except UnicodeDecodeError:
        with path.open("rb") as handle:
            payload = pickle.load(handle, encoding="latin1")
    if isinstance(payload, (tuple, list)) and len(payload) == 3:
        payload = payload[2]
    adjacency = np.asarray(payload)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError(f"adjacency must be a square matrix, got shape {adjacency.shape}")
    return adjacency


def convert_graph_weights(
    adjacency: np.ndarray,
    contract: PhysicalGraphContract,
) -> np.ndarray:
    """Apply the explicit graph weight contract without inferring its semantics."""

    raw = np.asarray(adjacency, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError(f"adjacency must be a square matrix, got shape {raw.shape}")
    if not np.isfinite(raw).all():
        raise ValueError("adjacency contains non-finite values")
    if contract.graph_mode == "undirected" and not np.allclose(
        raw, raw.T, rtol=0.0, atol=1.0e-12
    ):
        raise ValueError("undirected graph contract requires a symmetric adjacency matrix")

    positive = raw > 0.0
    np.fill_diagonal(positive, False)
    converted = np.zeros_like(raw, dtype=np.float64)
    if not positive.any():
        return converted

    if contract.weight_semantics == "affinity":
        converted[positive] = raw[positive] / raw[positive].max()
    else:
        scale = float(contract.conversion_scale)
        converted[positive] = np.exp(-np.square(raw[positive] / scale))
        positive_converted = converted[positive]
        if not np.isfinite(positive_converted).all() or np.any(positive_converted <= 0.0):
            raise ValueError("distance/cost conversion produced non-positive or non-finite weights")
        converted[positive] /= positive_converted.max()

    positive_converted = converted[positive]
    if (
        not np.isfinite(positive_converted).all()
        or np.any(positive_converted <= 0.0)
        or np.any(positive_converted > 1.0)
    ):
        raise ValueError("converted positive graph weights must lie in (0,1]")
    return converted


def physical_candidate_fingerprint(
    adjacency: np.ndarray,
    contract: PhysicalGraphContract,
    *,
    kp: int,
) -> str:
    canonical = np.ascontiguousarray(np.asarray(adjacency, dtype=np.float64))
    metadata = {
        "artifact_version": _ARTIFACT_VERSION,
        "contract": contract.as_dict(),
        "kp": kp,
        "shape": list(canonical.shape),
    }
    digest = hashlib.sha256()
    digest.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    digest.update(canonical.tobytes())
    return digest.hexdigest()


def build_physical_candidates(
    adjacency: np.ndarray,
    contract: PhysicalGraphContract,
    *,
    kp: int = PHYSICAL_KP,
) -> PhysicalCandidateArtifact:
    """Build deterministic source and sensor-lag metadata entirely on CPU."""

    if kp <= 0:
        raise ValueError(f"kp must be positive, got {kp}")
    raw = np.asarray(adjacency, dtype=np.float64)
    weights = convert_graph_weights(raw, contract)
    num_nodes = weights.shape[0]
    source_hop = np.full((num_nodes, num_nodes), -1, dtype=np.int64)
    source_log_strength = np.full((num_nodes, num_nodes), -np.inf, dtype=np.float64)
    for source in range(num_nodes):
        hop, log_strength = _shortest_paths_from_source(weights, source)
        source_hop[source] = hop
        source_log_strength[source] = log_strength

    source_index = np.zeros((num_nodes, kp), dtype=np.int64)
    source_valid = np.zeros((num_nodes, kp), dtype=np.bool_)
    source_prior = np.zeros((num_nodes, kp), dtype=np.float32)
    shortest_hop = np.full((num_nodes, kp), -1, dtype=np.int64)
    path_strength = np.zeros((num_nodes, kp), dtype=np.float32)

    all_sources = np.arange(num_nodes, dtype=np.int64)
    for target in range(num_nodes):
        eligible = (source_hop[:, target] > 0) & (all_sources != target)
        candidates = all_sources[eligible]
        if candidates.size == 0:
            continue
        hops = source_hop[candidates, target]
        log_strengths = source_log_strength[candidates, target]
        order = np.lexsort((candidates, -log_strengths, hops))
        selected = candidates[order[:kp]]
        count = selected.size
        selected_strength = np.exp(source_log_strength[selected, target]).astype(np.float32)
        if not np.isfinite(selected_strength).all() or np.any(selected_strength <= 0.0):
            raise ValueError("shortest-path strength is not representable in positive float32")
        source_index[target, :count] = selected
        source_valid[target, :count] = True
        source_prior[target, :count] = selected_strength
        shortest_hop[target, :count] = source_hop[selected, target]
        path_strength[target, :count] = selected_strength

    sources = PhysicalSourceMetadata(
        source_index=torch.from_numpy(source_index),
        valid=torch.from_numpy(source_valid),
        prior=torch.from_numpy(source_prior),
        shortest_hop=torch.from_numpy(shortest_hop),
        path_strength=torch.from_numpy(path_strength),
    )
    candidates = _expand_source_metadata(sources)
    return PhysicalCandidateArtifact(
        contract=contract,
        kp=kp,
        fingerprint=physical_candidate_fingerprint(raw, contract, kp=kp),
        sources=sources,
        candidates=candidates,
    )


def save_physical_candidate_artifact(
    path: str | Path,
    artifact: PhysicalCandidateArtifact,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "artifact_version": _ARTIFACT_VERSION,
        "contract": artifact.contract.as_dict(),
        "kp": artifact.kp,
        "fingerprint": artifact.fingerprint,
    }
    np.savez_compressed(
        path,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        source_index=artifact.sources.source_index.cpu().numpy(),
        source_valid=artifact.sources.valid.cpu().numpy(),
        source_prior=artifact.sources.prior.cpu().numpy(),
        shortest_hop=artifact.sources.shortest_hop.cpu().numpy(),
        path_strength=artifact.sources.path_strength.cpu().numpy(),
        candidate_source_index=artifact.candidates.source_index.cpu().numpy(),
        candidate_lag_index=artifact.candidates.lag_index.cpu().numpy(),
        candidate_flat_index=artifact.candidates.flat_index.cpu().numpy(),
        candidate_valid=artifact.candidates.valid.cpu().numpy(),
        candidate_prior=artifact.candidates.prior.cpu().numpy(),
    )


def load_physical_candidate_artifact(
    path: str | Path,
    *,
    expected_fingerprint: str | None = None,
) -> PhysicalCandidateArtifact:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Physical candidate artifact does not exist: {path}")
    with np.load(path, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata_json"].item()))
        if metadata.get("artifact_version") != _ARTIFACT_VERSION:
            raise ValueError("unsupported Physical candidate artifact version")
        fingerprint = str(metadata["fingerprint"])
        if expected_fingerprint is not None and fingerprint != expected_fingerprint:
            raise ValueError(
                f"Physical candidate fingerprint mismatch: {fingerprint} != {expected_fingerprint}"
            )
        contract = PhysicalGraphContract(**metadata["contract"])
        sources = PhysicalSourceMetadata(
            source_index=torch.from_numpy(payload["source_index"].astype(np.int64, copy=True)),
            valid=torch.from_numpy(payload["source_valid"].astype(np.bool_, copy=True)),
            prior=torch.from_numpy(payload["source_prior"].astype(np.float32, copy=True)),
            shortest_hop=torch.from_numpy(payload["shortest_hop"].astype(np.int64, copy=True)),
            path_strength=torch.from_numpy(payload["path_strength"].astype(np.float32, copy=True)),
        )
        candidates = PhysicalCandidateMetadata(
            source_index=torch.from_numpy(
                payload["candidate_source_index"].astype(np.int64, copy=True)
            ),
            lag_index=torch.from_numpy(payload["candidate_lag_index"].astype(np.int64, copy=True)),
            flat_index=torch.from_numpy(payload["candidate_flat_index"].astype(np.int64, copy=True)),
            valid=torch.from_numpy(payload["candidate_valid"].astype(np.bool_, copy=True)),
            prior=torch.from_numpy(payload["candidate_prior"].astype(np.float32, copy=True)),
        )
    artifact = PhysicalCandidateArtifact(
        contract=contract,
        kp=int(metadata["kp"]),
        fingerprint=fingerprint,
        sources=sources,
        candidates=candidates,
    )
    _validate_cached_artifact(artifact)
    return artifact


def _shortest_paths_from_source(
    weights: np.ndarray,
    source: int,
) -> tuple[np.ndarray, np.ndarray]:
    num_nodes = weights.shape[0]
    hop = np.full(num_nodes, -1, dtype=np.int64)
    log_strength = np.full(num_nodes, -np.inf, dtype=np.float64)
    hop[source] = 0
    log_strength[source] = 0.0
    frontier = [source]
    next_hop = 1
    while frontier:
        next_frontier: set[int] = set()
        for node in frontier:
            for neighbor in np.flatnonzero(weights[node] > 0.0).tolist():
                if hop[neighbor] == -1:
                    hop[neighbor] = next_hop
                    next_frontier.add(neighbor)
                if hop[neighbor] == next_hop:
                    candidate_strength = log_strength[node] + math.log(weights[node, neighbor])
                    if candidate_strength > log_strength[neighbor]:
                        log_strength[neighbor] = candidate_strength
        frontier = sorted(next_frontier)
        next_hop += 1
    return hop, log_strength


def _expand_source_metadata(sources: PhysicalSourceMetadata) -> PhysicalCandidateMetadata:
    num_nodes, kp = sources.source_index.shape
    source_index = sources.source_index.repeat_interleave(HISTORY_LENGTH, dim=1)
    lag_index = torch.arange(HISTORY_LENGTH, dtype=torch.int64).repeat(kp).view(1, -1)
    lag_index = lag_index.expand(num_nodes, -1).clone()
    valid = sources.valid.repeat_interleave(HISTORY_LENGTH, dim=1)
    prior = sources.prior.repeat_interleave(HISTORY_LENGTH, dim=1)
    flat_index = source_index * HISTORY_LENGTH + lag_index
    return PhysicalCandidateMetadata(
        source_index=source_index,
        lag_index=lag_index,
        flat_index=flat_index,
        valid=valid,
        prior=prior,
    )


def _validate_cached_artifact(artifact: PhysicalCandidateArtifact) -> None:
    sources = artifact.sources
    candidates = artifact.candidates
    if sources.source_index.ndim != 2 or sources.source_index.shape[1] != artifact.kp:
        raise ValueError("invalid Physical source metadata shape")
    num_nodes = sources.source_index.shape[0]
    source_shape = (num_nodes, artifact.kp)
    if any(
        tensor.shape != source_shape
        for tensor in (
            sources.valid,
            sources.prior,
            sources.shortest_hop,
            sources.path_strength,
        )
    ):
        raise ValueError("inconsistent Physical source metadata shapes")
    candidate_shape = (num_nodes, artifact.kp * HISTORY_LENGTH)
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
        raise ValueError("inconsistent Physical candidate metadata shapes")
    if not torch.isfinite(sources.prior).all() or not torch.isfinite(candidates.prior).all():
        raise ValueError("Physical candidate artifact contains non-finite prior")
    if (sources.prior[sources.valid] <= 0).any() or (sources.prior[sources.valid] > 1).any():
        raise ValueError("valid Physical source prior must lie in (0,1]")
    if sources.prior[~sources.valid].ne(0).any() or candidates.prior[~candidates.valid].ne(0).any():
        raise ValueError("invalid Physical candidate prior must be zero")
