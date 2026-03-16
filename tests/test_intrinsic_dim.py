"""Tests for intrinsic dimensionality estimation."""

import numpy as np
import pytest

from src.geometry.intrinsic_dim import (
    mle_intrinsic_dim,
    twonn_intrinsic_dim,
    compute_intrinsic_dim,
    compute_local_intrinsic_dims,
)


def make_linear_subspace(n=500, intrinsic_d=5, ambient_d=100, seed=42):
    """Generate points on a d-dimensional linear subspace in D dimensions."""
    rng = np.random.RandomState(seed)
    # Random basis for the subspace
    basis = np.linalg.qr(rng.randn(ambient_d, intrinsic_d))[0][:, :intrinsic_d]
    coords = rng.randn(n, intrinsic_d)
    return coords @ basis.T


def make_sphere(n=1000, d=2, ambient_d=3, seed=42):
    """Generate points on a d-sphere embedded in ambient_d dimensions."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d + 1)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    if ambient_d > d + 1:
        padding = np.zeros((n, ambient_d - d - 1))
        X = np.concatenate([X, padding], axis=1)
    return X


class TestMLEIntrinsicDim:
    def test_known_linear_subspace(self):
        """MLE should recover dimension of a linear subspace."""
        X = make_linear_subspace(n=1000, intrinsic_d=5, ambient_d=100)
        d = mle_intrinsic_dim(X, k=20)
        # Should be close to 5, allow some tolerance
        assert 3.5 < d < 7.0, f"Expected ~5, got {d}"

    def test_1d_line(self):
        """MLE on a 1D line embedded in high-D."""
        rng = np.random.RandomState(0)
        t = rng.randn(500, 1)
        direction = rng.randn(1, 50)
        direction = direction / np.linalg.norm(direction)
        X = t @ direction
        d = mle_intrinsic_dim(X, k=10)
        assert 0.5 < d < 2.0, f"Expected ~1, got {d}"

    def test_too_few_points(self):
        """Should raise if N <= k."""
        X = np.random.randn(5, 10)
        with pytest.raises(ValueError):
            mle_intrinsic_dim(X, k=5)

    def test_returns_positive(self):
        """ID should always be positive."""
        X = np.random.randn(100, 10)
        d = mle_intrinsic_dim(X, k=10)
        assert d > 0

    def test_k_sensitivity(self):
        """Different k values should give roughly similar results."""
        X = make_linear_subspace(n=1000, intrinsic_d=10, ambient_d=100)
        d10 = mle_intrinsic_dim(X, k=10)
        d30 = mle_intrinsic_dim(X, k=30)
        # Should be within factor of 2
        assert abs(d10 - d30) / max(d10, d30) < 0.5


class TestTwoNNIntrinsicDim:
    def test_known_linear_subspace(self):
        X = make_linear_subspace(n=1000, intrinsic_d=5, ambient_d=100)
        d = twonn_intrinsic_dim(X)
        assert 3.0 < d < 8.0, f"Expected ~5, got {d}"

    def test_2d_sphere(self):
        """2-sphere should have ID ~2."""
        X = make_sphere(n=2000, d=2, ambient_d=3)
        d = twonn_intrinsic_dim(X)
        assert 1.0 < d < 3.5, f"Expected ~2, got {d}"

    def test_too_few_points(self):
        X = np.random.randn(2, 10)
        with pytest.raises(ValueError):
            twonn_intrinsic_dim(X)

    def test_returns_positive(self):
        X = np.random.randn(100, 10)
        d = twonn_intrinsic_dim(X)
        assert d > 0


class TestComputeIntrinsicDim:
    def test_dispatch_mle(self):
        X = np.random.randn(200, 10)
        d = compute_intrinsic_dim(X, method="mle", k=10)
        assert d > 0

    def test_dispatch_twonn(self):
        X = np.random.randn(200, 10)
        d = compute_intrinsic_dim(X, method="twonn")
        assert d > 0

    def test_invalid_method(self):
        X = np.random.randn(200, 10)
        with pytest.raises(ValueError):
            compute_intrinsic_dim(X, method="invalid")


class TestLocalIntrinsicDims:
    def test_per_class_dims(self):
        rng = np.random.RandomState(42)
        # Two classes with different intrinsic dimensions
        X1 = make_linear_subspace(n=200, intrinsic_d=3, ambient_d=50, seed=0)
        X2 = make_linear_subspace(n=200, intrinsic_d=8, ambient_d=50, seed=1)
        X2 = X2 + 10  # shift to separate clusters
        X = np.vstack([X1, X2])
        labels = np.array([0] * 200 + [1] * 200)

        result = compute_local_intrinsic_dims(X, labels, method="mle", k=15)
        assert len(result) == 2
        # Class 0 should have lower ID than class 1
        assert result["0"] < result["1"]

    def test_skip_small_classes(self):
        X = np.random.randn(100, 10)
        labels = np.array([0] * 95 + [1] * 5)
        result = compute_local_intrinsic_dims(X, labels, min_samples=10)
        assert "0" in result
        assert "1" not in result
