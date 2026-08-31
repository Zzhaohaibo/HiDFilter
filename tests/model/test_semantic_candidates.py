from __future__ import annotations

import numpy as np
import torch

from hidfilter.semantic import (
    build_semantic_candidates,
    compute_first_differences,
    compute_pairwise_pearson,
    load_semantic_candidate_artifact,
    save_semantic_candidate_artifact,
    select_semantic_sources,
)


def test_first_difference_uses_adjacent_raw_validity_only():
    raw = np.array(
        [[1.0, 10.0], [3.0, 12.0], [6.0, 16.0], [10.0, 20.0]], dtype=np.float64
    )
    raw_valid = np.array(
        [[True, True], [True, False], [True, True], [True, True]], dtype=np.bool_
    )

    difference, valid = compute_first_differences(raw, raw_valid)

    np.testing.assert_allclose(difference[1:, 0], [2.0, 3.0, 4.0])
    assert valid[:, 0].tolist() == [False, True, True, True]
    assert valid[:, 1].tolist() == [False, False, False, True]
    assert np.isnan(difference[0]).all()
    assert np.isnan(difference[1:3, 1]).all()
    assert difference[3, 1] == 4.0


def test_pairwise_pearson_uses_pair_specific_common_validity():
    difference = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 2.0, -1.0],
            [2.0, 4.0, -2.0],
            [3.0, 6.0, -3.0],
            [4.0, 8.0, -4.0],
            [5.0, 10.0, -5.0],
        ],
        dtype=np.float64,
    )
    valid = np.array(
        [
            [False, False, False],
            [True, True, True],
            [True, True, False],
            [True, False, True],
            [True, True, True],
            [False, True, False],
        ],
        dtype=np.bool_,
    )

    correlation, overlap = compute_pairwise_pearson(
        difference, valid, min_overlap=2, variance_threshold=1.0e-12
    )

    assert overlap[0, 1] == 3
    assert overlap[0, 2] == 3
    assert overlap[1, 2] == 2
    assert correlation[0, 1] > 0.999
    assert correlation[0, 2] < -0.999
    assert np.isfinite(correlation[1, 2])


def test_min_overlap_boundary_and_near_zero_variance_are_unavailable():
    x = np.arange(288, dtype=np.float64)
    difference = np.column_stack((x, 2.0 * x, np.tile([-1.0e-7, 1.0e-7], 144)))
    valid = np.ones_like(difference, dtype=np.bool_)

    correlation, overlap = compute_pairwise_pearson(
        difference, valid, min_overlap=288, variance_threshold=1.0e-12
    )
    assert overlap[0, 1] == 288
    assert correlation[0, 1] > 0.999999
    assert np.isnan(correlation[0, 2])

    valid[-1, 1] = False
    correlation_287, overlap_287 = compute_pairwise_pearson(
        difference, valid, min_overlap=288, variance_threshold=1.0e-12
    )
    assert overlap_287[0, 1] == 287
    assert np.isnan(correlation_287[0, 1])


def test_pearson_preserves_sign_and_clips_finite_results():
    x = np.linspace(-3.0, 3.0, 300, dtype=np.float64)
    difference = np.column_stack((x, 7.0 * x, -2.0 * x))
    valid = np.ones_like(difference, dtype=np.bool_)

    correlation, _ = compute_pairwise_pearson(
        difference, valid, min_overlap=288, variance_threshold=1.0e-12
    )

    assert correlation[0, 1] == 1.0
    assert correlation[0, 2] == -1.0
    finite = correlation[np.isfinite(correlation)]
    assert np.all(finite >= -1.0)
    assert np.all(finite <= 1.0)


def test_semantic_exclusions_abs_ranking_tie_and_padding_no_refill():
    correlation = np.full((7, 7), np.nan, dtype=np.float64)
    correlation[0, 1:] = [0.99, -0.95, -0.90, 0.80, -0.70, 0.70]
    correlation[1:, 0] = correlation[0, 1:]
    overlap = np.full((7, 7), 300, dtype=np.int64)
    one_hop = np.zeros((7, 7), dtype=np.bool_)
    one_hop[0, 1] = True
    physical_source_index = np.zeros((7, 8), dtype=np.int64)
    physical_source_valid = np.zeros((7, 8), dtype=np.bool_)
    physical_source_index[0, :2] = [1, 2]
    physical_source_valid[0, :2] = True

    sources = select_semantic_sources(
        correlation,
        overlap,
        one_hop,
        physical_source_index,
        physical_source_valid,
        ks=8,
    )

    selected = sources.source_index[0, sources.valid[0]].tolist()
    assert selected == [3, 4, 5, 6]
    torch.testing.assert_close(
        sources.prior[0, :4], torch.tensor([0.9, 0.8, 0.7, 0.7]), rtol=0.0, atol=1.0e-7
    )
    torch.testing.assert_close(
        sources.signed_corr[0, :4],
        torch.tensor([-0.9, 0.8, -0.7, 0.7]),
        rtol=0.0,
        atol=1.0e-7,
    )
    assert not sources.valid[0, 4:].any()
    assert sources.source_index[0, 4:].eq(0).all()
    assert sources.prior[0, 4:].eq(0).all()


def test_semantic_candidate_cache_is_deterministic_and_binds_train_and_exclusions(tmp_path):
    time = np.arange(12, dtype=np.float64)
    raw = np.column_stack((time + 1.0, 2.0 * time + 1.0, -time + 20.0, time**2 + 1.0))
    raw_valid = np.ones_like(raw, dtype=np.bool_)
    one_hop = np.zeros((4, 4), dtype=np.bool_)
    physical_index = np.zeros((4, 8), dtype=np.int64)
    physical_valid = np.zeros((4, 8), dtype=np.bool_)

    first = build_semantic_candidates(
        raw,
        raw_valid,
        one_hop,
        physical_index,
        physical_valid,
        ks=2,
        min_overlap=5,
        variance_threshold=1.0e-12,
    )
    second = build_semantic_candidates(
        raw.copy(),
        raw_valid.copy(),
        one_hop.copy(),
        physical_index.copy(),
        physical_valid.copy(),
        ks=2,
        min_overlap=5,
        variance_threshold=1.0e-12,
    )
    changed_raw = raw.copy()
    changed_raw[3, 0] += 0.25
    changed_train = build_semantic_candidates(
        changed_raw,
        raw_valid,
        one_hop,
        physical_index,
        physical_valid,
        ks=2,
        min_overlap=5,
        variance_threshold=1.0e-12,
    )
    changed_physical_index = physical_index.copy()
    changed_physical_index[0, 0] = 1
    changed_physical_valid = physical_valid.copy()
    changed_physical_valid[0, 0] = True
    changed_exclusion = build_semantic_candidates(
        raw,
        raw_valid,
        one_hop,
        changed_physical_index,
        changed_physical_valid,
        ks=2,
        min_overlap=5,
        variance_threshold=1.0e-12,
    )
    changed_raw_valid = raw_valid.copy()
    changed_raw_valid[4, 1] = False
    changed_validity = build_semantic_candidates(
        raw,
        changed_raw_valid,
        one_hop,
        physical_index,
        physical_valid,
        ks=2,
        min_overlap=5,
        variance_threshold=1.0e-12,
    )

    assert first.fingerprint == second.fingerprint
    assert first.fingerprint != changed_train.fingerprint
    assert first.fingerprint != changed_exclusion.fingerprint
    assert first.fingerprint != changed_validity.fingerprint
    assert torch.equal(first.sources.source_index, second.sources.source_index)
    assert first.candidates.lag_index[0].tolist() == list(range(12)) * 2
    assert first.candidates.flat_index.eq(
        first.candidates.source_index * 12 + first.candidates.lag_index
    ).all()

    cache_path = tmp_path / "semantic_candidates.npz"
    save_semantic_candidate_artifact(cache_path, first)
    loaded = load_semantic_candidate_artifact(
        cache_path, expected_fingerprint=first.fingerprint
    )
    assert torch.equal(loaded.sources.source_index, first.sources.source_index)
    assert torch.equal(loaded.sources.valid, first.sources.valid)
    assert torch.equal(loaded.sources.overlap_count, first.sources.overlap_count)
    torch.testing.assert_close(loaded.sources.prior, first.sources.prior, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        loaded.sources.signed_corr, first.sources.signed_corr, rtol=0.0, atol=0.0
    )
    assert torch.equal(loaded.candidates.flat_index, first.candidates.flat_index)
