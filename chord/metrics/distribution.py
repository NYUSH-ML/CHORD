from __future__ import annotations

from typing import Callable

import numpy as np

Kernel = Callable[[np.ndarray, np.ndarray], np.ndarray]


def _as_float64(matrix: np.ndarray) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float64)
    if value.ndim != 2:
        raise ValueError("embeddings must be a two-dimensional array")
    return value


def _sqrt_psd(matrix: np.ndarray) -> np.ndarray:
    symmetric = (matrix + matrix.T) / 2
    values, vectors = np.linalg.eigh(symmetric)
    tolerance = max(np.max(np.abs(values)), 1.0) * 1e-12
    if np.min(values) < -tolerance:
        raise ValueError("covariance matrix is not positive semidefinite")
    values = np.maximum(values, 0.0)
    return (vectors * np.sqrt(values)) @ vectors.T


def frechet_distance(left: np.ndarray, right: np.ndarray) -> float:
    """Fréchet distance for the unified encoder lane.

    This is the standard Gaussian Fréchet formula over cached embeddings. It is
    not, by itself, a prior-art FBD reproduction because the original score also
    fixes encoder, layer, pooling, and preprocessing choices.
    """
    left = _as_float64(left)
    right = _as_float64(right)
    if len(left) < 2 or len(right) < 2:
        raise ValueError("FBD requires at least two samples per corpus")
    mean_left = left.mean(axis=0)
    mean_right = right.mean(axis=0)
    covariance_left = np.cov(left, rowvar=False, ddof=1)
    covariance_right = np.cov(right, rowvar=False, ddof=1)
    covariance_left = np.atleast_2d(covariance_left)
    covariance_right = np.atleast_2d(covariance_right)
    sqrt_left = _sqrt_psd(covariance_left)
    middle = sqrt_left @ covariance_right @ sqrt_left
    trace_cross = np.trace(_sqrt_psd(middle))
    distance = (
        np.dot(mean_left - mean_right, mean_left - mean_right)
        + np.trace(covariance_left)
        + np.trace(covariance_right)
        - 2 * trace_cross
    )
    if abs(distance) < 1e-10:
        return 0.0
    return float(distance)


def rbf_kernel(sigma: float) -> Kernel:
    if sigma <= 0:
        raise ValueError("RBF bandwidth must be positive")

    def kernel(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        left_norm = np.sum(left * left, axis=1)[:, None]
        right_norm = np.sum(right * right, axis=1)[None, :]
        squared = np.maximum(left_norm + right_norm - 2 * left @ right.T, 0.0)
        return np.exp(-squared / (2 * sigma * sigma))

    return kernel


def kernel_sum(
    left: np.ndarray,
    right: np.ndarray,
    kernel: Kernel,
    block_size: int = 512,
) -> float:
    total = 0.0
    for left_start in range(0, len(left), block_size):
        left_block = left[left_start : left_start + block_size]
        for right_start in range(0, len(right), block_size):
            right_block = right[right_start : right_start + block_size]
            total += float(kernel(left_block, right_block).sum(dtype=np.float64))
    return total


def kernel_diagonal_sum(matrix: np.ndarray, kernel: Kernel, block_size: int = 512) -> float:
    total = 0.0
    for start in range(0, len(matrix), block_size):
        block = matrix[start : start + block_size]
        total += float(np.diag(kernel(block, block)).sum(dtype=np.float64))
    return total


def squared_mmd(
    left: np.ndarray,
    right: np.ndarray,
    kernel: Kernel,
    unbiased: bool,
    block_size: int = 512,
) -> float:
    left = _as_float64(left)
    right = _as_float64(right)
    m, n = len(left), len(right)
    if unbiased and (m < 2 or n < 2):
        raise ValueError("unbiased MMD requires at least two samples per corpus")
    sum_xx = kernel_sum(left, left, kernel, block_size)
    sum_yy = kernel_sum(right, right, kernel, block_size)
    sum_xy = kernel_sum(left, right, kernel, block_size)
    if unbiased:
        diagonal_x = kernel_diagonal_sum(left, kernel, block_size)
        diagonal_y = kernel_diagonal_sum(right, kernel, block_size)
        return (
            (sum_xx - diagonal_x) / (m * (m - 1))
            + (sum_yy - diagonal_y) / (n * (n - 1))
            - 2 * sum_xy / (m * n)
        )
    return sum_xx / (m * m) + sum_yy / (n * n) - 2 * sum_xy / (m * n)


def rbf_mmd(
    left: np.ndarray,
    right: np.ndarray,
    sigma: float,
    unbiased: bool,
    block_size: int = 512,
) -> float:
    """Squared MMD with an RBF kernel under the configured encoder protocol."""
    return squared_mmd(left, right, rbf_kernel(sigma), unbiased=unbiased, block_size=block_size)


def median_bandwidth(embeddings: np.ndarray, max_samples: int = 2000, seed: int = 0) -> float:
    embeddings = _as_float64(embeddings)
    if len(embeddings) > max_samples:
        rng = np.random.default_rng(seed)
        embeddings = embeddings[rng.choice(len(embeddings), max_samples, replace=False)]
    norms = np.sum(embeddings * embeddings, axis=1)[:, None]
    distances = np.sqrt(np.maximum(norms + norms.T - 2 * embeddings @ embeddings.T, 0.0))
    values = distances[np.triu_indices(len(embeddings), k=1)]
    nonzero = values[values > 0]
    if not len(nonzero):
        raise ValueError("cannot estimate bandwidth from identical embeddings")
    return float(np.median(nonzero))
