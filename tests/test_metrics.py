"""RBF-MMD, bandwidth and null standardization (CPU, no model)."""

import numpy as np
import pytest

from chord.api import ChordScorer
from chord.metrics.distribution import median_bandwidth, rbf_kernel, rbf_mmd, squared_mmd


def _gaussian(n: int, shift: float, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(n, 16)) + shift


def test_mmd_is_zero_for_identical_sets():
    x = _gaussian(64, 0.0, 0)
    assert squared_mmd(x, x, rbf_kernel(1.0), unbiased=False) == pytest.approx(0.0, abs=1e-12)


def test_mmd_grows_with_distribution_shift():
    ref = _gaussian(200, 0.0, 0)
    sigma = median_bandwidth(ref)
    near = rbf_mmd(_gaussian(200, 0.1, 1), ref, sigma, unbiased=True)
    far = rbf_mmd(_gaussian(200, 1.0, 2), ref, sigma, unbiased=True)
    assert far > near


def test_blocked_sum_matches_single_block():
    x, y = _gaussian(70, 0.0, 0), _gaussian(90, 0.5, 1)
    kernel = rbf_kernel(3.0)
    full = squared_mmd(x, y, kernel, unbiased=False, block_size=1024)
    blocked = squared_mmd(x, y, kernel, unbiased=False, block_size=16)
    assert blocked == pytest.approx(full, rel=1e-10)


def test_median_bandwidth_is_positive_and_scale_equivariant():
    x = _gaussian(100, 0.0, 0)
    assert median_bandwidth(x) > 0
    assert median_bandwidth(3 * x) == pytest.approx(3 * median_bandwidth(x), rel=1e-9)


def _scorer_without_model() -> ChordScorer:
    # score_embeddings needs no encoder; skip loading one
    scorer = object.__new__(ChordScorer)
    scorer.model_key, scorer.model = "qwen3.5-27b", "none"
    return scorer


def test_z_score_separates_shifted_corpus_from_null():
    scorer = _scorer_without_model()
    pool = _gaussian(600, 0.0, 0)
    same = scorer.score_embeddings(
        _gaussian(100, 0.0, 1), pool[:300], null_reference_embeddings=pool, null_draws=50
    )
    shifted = scorer.score_embeddings(
        _gaussian(100, 0.5, 2), pool[:300], null_reference_embeddings=pool, null_draws=50
    )
    assert shifted.z_score > 10 * max(abs(same.z_score), 1.0)
    assert shifted.raw_mmd > same.raw_mmd
