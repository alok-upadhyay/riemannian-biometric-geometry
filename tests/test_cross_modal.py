"""Tests for cross-modal geometry metrics."""

import numpy as np
import pytest

from src.geometry.cross_modal import (
    gromov_wasserstein_distance,
    spectral_gap_divergence,
    linear_cka,
    intrinsic_dim_mismatch,
    compute_cross_modal_metrics,
)


class TestGromovWassersteinDistance:
    def test_same_space_zero(self):
        """GW distance of identical spaces should be near zero."""
        X = np.random.randn(50, 10)
        d = gromov_wasserstein_distance(X, X.copy(), max_points=50)
        assert d < 0.1, f"GW distance of identical spaces should be ~0, got {d}"

    def test_different_spaces_positive(self):
        """GW distance of different spaces should be positive."""
        rng = np.random.RandomState(42)
        X = rng.randn(50, 10)
        Y = rng.randn(50, 10) * 5 + 10
        d = gromov_wasserstein_distance(X, Y, max_points=50)
        assert d > 0

    def test_subsampling(self):
        """Should work with subsampling for large inputs."""
        X = np.random.randn(1000, 10)
        Y = np.random.randn(1000, 10)
        d = gromov_wasserstein_distance(X, Y, max_points=100)
        assert np.isfinite(d)

    def test_different_dimensions(self):
        """Should work with different ambient dimensions."""
        X = np.random.randn(50, 10)
        Y = np.random.randn(50, 20)
        d = gromov_wasserstein_distance(X, Y, max_points=50)
        assert np.isfinite(d)


class TestSpectralGapDivergence:
    def test_same_space_small(self):
        """Spectral divergence of identical spaces should be small."""
        X = np.random.randn(100, 10)
        d = spectral_gap_divergence(X, X.copy(), k=10, n_eigenvalues=20)
        assert d < 0.5, f"Spectral gap of identical spaces should be small, got {d}"

    def test_different_spaces_larger(self):
        """Different spaces should have larger spectral divergence."""
        rng = np.random.RandomState(42)
        X = rng.randn(100, 10)
        Y = rng.exponential(1, (100, 10))  # Very different distribution
        d_same = spectral_gap_divergence(X, X.copy(), k=10, n_eigenvalues=20)
        d_diff = spectral_gap_divergence(X, Y, k=10, n_eigenvalues=20)
        assert d_diff > d_same * 0.5  # At least somewhat different

    def test_returns_nonnegative(self):
        X = np.random.randn(100, 10)
        Y = np.random.randn(100, 10)
        d = spectral_gap_divergence(X, Y, k=10, n_eigenvalues=20)
        assert d >= 0


class TestLinearCKA:
    def test_identical_representations(self):
        """CKA of identical representations should be 1."""
        X = np.random.randn(50, 10)
        cka = linear_cka(X, X.copy())
        assert cka == pytest.approx(1.0, abs=1e-6)

    def test_scaled_representations(self):
        """CKA should be invariant to scaling."""
        X = np.random.randn(50, 10)
        Y = X * 3.0
        cka = linear_cka(X, Y)
        assert cka == pytest.approx(1.0, abs=1e-6)

    def test_orthogonal_representations(self):
        """CKA of orthogonal representations should be near 0."""
        rng = np.random.RandomState(42)
        X = rng.randn(100, 10)
        # Create orthogonal representation
        Y = rng.randn(100, 10)
        # Make Y orthogonal to X (approximately)
        for i in range(10):
            Y[:, i] -= X @ (X.T @ Y[:, i]) / (np.linalg.norm(X, axis=0) ** 2 + 1e-8).sum()

        cka = linear_cka(X, Y)
        # Not necessarily 0 but should be low
        assert cka < 0.5

    def test_bounded_zero_one(self):
        """CKA should be in [0, 1] for reasonable inputs."""
        X = np.random.randn(50, 10)
        Y = np.random.randn(50, 8)
        cka = linear_cka(X, Y)
        assert -0.1 <= cka <= 1.1  # Allow small numerical error

    def test_different_dimensions(self):
        """CKA should work with different feature dimensions."""
        X = np.random.randn(50, 10)
        Y = np.random.randn(50, 20)
        cka = linear_cka(X, Y)
        assert np.isfinite(cka)


class TestIntrinsicDimMismatch:
    def test_same_dim(self):
        assert intrinsic_dim_mismatch(5.0, 5.0) == 0.0

    def test_positive_difference(self):
        assert intrinsic_dim_mismatch(3.0, 7.0) == 4.0

    def test_symmetric(self):
        assert intrinsic_dim_mismatch(3.0, 7.0) == intrinsic_dim_mismatch(7.0, 3.0)


class TestComputeCrossModalMetrics:
    def test_all_metrics_computed(self):
        rng = np.random.RandomState(42)
        X = rng.randn(100, 10)
        Y = rng.randn(100, 8)
        result = compute_cross_modal_metrics(
            X, Y, id_x=5.0, id_y=3.0, gh_max_points=100
        )
        assert "gw_distance" in result
        assert "spectral_gap" in result
        assert "cka" in result
        assert "id_mismatch" in result
        assert result["id_mismatch"] == 2.0

    def test_without_id_values(self):
        X = np.random.randn(100, 10)
        Y = np.random.randn(100, 8)
        result = compute_cross_modal_metrics(X, Y, gh_max_points=50)
        assert "id_mismatch" not in result

    def test_all_values_finite(self):
        X = np.random.randn(100, 10)
        Y = np.random.randn(100, 8)
        result = compute_cross_modal_metrics(X, Y, gh_max_points=50)
        for key in ["gw_distance", "spectral_gap", "cka"]:
            assert np.isfinite(result[key]), f"{key} is not finite: {result[key]}"
