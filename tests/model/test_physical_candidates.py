from __future__ import annotations

import math
import pickle

import numpy as np
import pytest
import torch

from hidfilter.physical import (
    PhysicalGraphContract,
    build_physical_candidates,
    convert_graph_weights,
    load_adjacency_artifact,
    load_physical_candidate_artifact,
    save_physical_candidate_artifact,
)


def _contract(mode: str = "directed") -> PhysicalGraphContract:
    return PhysicalGraphContract(
        graph_mode=mode,
        weight_semantics="affinity",
        conversion_scale=None,
    )


def test_directed_reachability_uses_source_to_target_and_never_refills():
    adjacency = np.zeros((4, 4), dtype=np.float64)
    adjacency[0, 1] = 1.0
    adjacency[1, 2] = 1.0

    artifact = build_physical_candidates(adjacency, _contract(), kp=8)

    assert artifact.sources.source_index[2, :2].tolist() == [1, 0]
    assert artifact.sources.shortest_hop[2, :2].tolist() == [1, 2]
    assert artifact.sources.valid[2].sum().item() == 2
    assert not artifact.sources.valid[0].any()
    assert 2 not in artifact.sources.source_index[2, artifact.sources.valid[2]].tolist()


def test_undirected_reachability_self_exclusion_and_padding():
    adjacency = np.zeros((4, 4), dtype=np.float64)
    adjacency[0, 1] = adjacency[1, 0] = 1.0
    adjacency[1, 2] = adjacency[2, 1] = 1.0

    artifact = build_physical_candidates(adjacency, _contract("undirected"), kp=8)

    assert artifact.sources.source_index[0, :2].tolist() == [1, 2]
    assert artifact.sources.valid[0].sum().item() == 2
    assert not artifact.sources.valid[3].any()
    assert artifact.sources.source_index[0, 2:].eq(0).all()
    assert artifact.sources.prior[0, 2:].eq(0).all()


def test_ranking_is_hop_then_strength_then_sensor_index():
    adjacency = np.zeros((8, 8), dtype=np.float64)
    adjacency[0, 7] = 0.1
    adjacency[1, 7] = 0.8
    adjacency[2, 7] = 0.8
    adjacency[3, 4] = 1.0
    adjacency[4, 7] = 1.0

    artifact = build_physical_candidates(adjacency, _contract(), kp=8)
    ranked = artifact.sources.source_index[7, artifact.sources.valid[7]].tolist()

    assert ranked == [4, 1, 2, 0, 3]
    assert artifact.sources.shortest_hop[7, :5].tolist() == [1, 1, 1, 1, 2]


def test_path_strength_uses_best_shortest_path_and_ignores_longer_stronger_path():
    adjacency = np.zeros((5, 5), dtype=np.float64)
    adjacency[0, 1] = 0.8
    adjacency[1, 4] = 0.5
    adjacency[0, 2] = 0.9
    adjacency[2, 4] = 0.9
    adjacency[0, 3] = 1.0
    adjacency[3, 2] = 1.0

    artifact = build_physical_candidates(adjacency, _contract(), kp=8)
    row_valid = artifact.sources.valid[4]
    row_sources = artifact.sources.source_index[4, row_valid]
    source_zero_slot = torch.nonzero(row_sources.eq(0), as_tuple=False).item()

    assert artifact.sources.shortest_hop[4, source_zero_slot].item() == 2
    assert artifact.sources.path_strength[4, source_zero_slot].item() == pytest.approx(0.81)
    assert artifact.sources.prior[4, source_zero_slot].item() == pytest.approx(0.81)


def test_weight_conversion_contracts_and_hard_failures():
    affinity = np.array(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 2.0], [0.0, 2.0, 0.0]], dtype=np.float64
    )
    converted_affinity = convert_graph_weights(affinity, _contract("undirected"))
    np.testing.assert_allclose(
        converted_affinity,
        np.array([[0.0, 0.5, 0.0], [0.5, 0.0, 1.0], [0.0, 1.0, 0.0]]),
    )

    distances = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    for semantics in ("distance", "cost"):
        contract = PhysicalGraphContract("undirected", semantics, 2.0)
        converted = convert_graph_weights(distances, contract)
        assert converted[0, 1] == pytest.approx(1.0)
        assert converted[0, 2] == pytest.approx(math.exp(-0.75))

    for invalid_scale in (None, 0.0, -1.0):
        with pytest.raises(ValueError, match="conversion_scale"):
            PhysicalGraphContract("directed", "distance", invalid_scale)
    nonfinite = affinity.copy()
    nonfinite[0, 1] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        convert_graph_weights(nonfinite, _contract("undirected"))


def test_physical_flatten_is_sensor_major_lag_minor_with_invalid_padding():
    adjacency = np.zeros((3, 3), dtype=np.float64)
    adjacency[0, 2] = 1.0
    adjacency[1, 2] = 1.0
    artifact = build_physical_candidates(adjacency, _contract(), kp=8)
    candidates = artifact.candidates

    assert candidates.source_index.shape == (3, 96)
    assert candidates.lag_index[2].tolist() == list(range(12)) * 8
    assert candidates.source_index[2, :12].eq(0).all()
    assert candidates.source_index[2, 12:24].eq(1).all()
    assert candidates.flat_index[2, :12].tolist() == list(range(12))
    assert candidates.flat_index[2, 12:24].tolist() == list(range(12, 24))
    assert candidates.valid[2, :24].all()
    assert not candidates.valid[2, 24:].any()
    assert candidates.prior[2, 24:].eq(0).all()


def test_candidate_build_and_serialized_cache_are_deterministic(tmp_path):
    adjacency = np.zeros((4, 4), dtype=np.float64)
    adjacency[0, 1] = adjacency[1, 0] = 0.5
    adjacency[1, 2] = adjacency[2, 1] = 1.0
    first = build_physical_candidates(adjacency, _contract("undirected"), kp=8)
    second = build_physical_candidates(adjacency.copy(), _contract("undirected"), kp=8)
    cache_path = tmp_path / "physical_candidates.npz"

    assert first.fingerprint == second.fingerprint
    assert torch.equal(first.candidates.flat_index, second.candidates.flat_index)
    assert torch.equal(first.sources.source_index, second.sources.source_index)
    save_physical_candidate_artifact(cache_path, first)
    loaded = load_physical_candidate_artifact(cache_path, expected_fingerprint=first.fingerprint)

    assert loaded.fingerprint == first.fingerprint
    assert torch.equal(loaded.sources.source_index, first.sources.source_index)
    assert torch.equal(loaded.sources.valid, first.sources.valid)
    torch.testing.assert_close(loaded.sources.prior, first.sources.prior, rtol=0.0, atol=0.0)
    assert torch.equal(loaded.candidates.flat_index, first.candidates.flat_index)
    with pytest.raises(ValueError, match="fingerprint"):
        load_physical_candidate_artifact(cache_path, expected_fingerprint="wrong")


def test_basic_ts_adjacency_loader_accepts_array_and_legacy_triple(tmp_path):
    adjacency = np.eye(3, dtype=np.float32)
    array_path = tmp_path / "array.pkl"
    triple_path = tmp_path / "triple.pkl"
    with array_path.open("wb") as handle:
        pickle.dump(adjacency, handle)
    with triple_path.open("wb") as handle:
        pickle.dump((["a", "b", "c"], {"a": 0}, adjacency), handle)

    np.testing.assert_array_equal(load_adjacency_artifact(array_path), adjacency)
    np.testing.assert_array_equal(load_adjacency_artifact(triple_path), adjacency)
