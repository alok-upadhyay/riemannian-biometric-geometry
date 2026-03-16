"""Tests for local curvature estimation."""

import numpy as np
import pytest

from src.geometry.curvature import estimate_local_curvature, curvature_summary


def make_sphere_points(n=500, radius=1.0, seed=42):
    """Points on a sphere (positive constant curvature)."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 3)
    X = radius * X / np.linalg.norm(X, axis=1, keepdims=True)
    return X


def make_plane_points(n=500, ambient_d=10, seed=42):
    """Points on a flat 2D plane (zero curvature)."""
    rng = np.random.RandomState(seed)
    coords = rng.randn(n, 2)
    # Embed in higher dimensions
    X = np.zeros((n, ambient_d))
    X[:, 0] = coords[:, 0]
    X[:, 1] = coords[:, 1]
    return X


class TestEstimateLocalCurvature:
    def test_output_shapes(self):
        X = np.random.randn(200, 10)
        result = estimate_local_curvature(X, n_sample_points=50, k=20)
        assert "mean_curvatures" in result
        assert "gaussian_curvatures" in result
        assert "principal_curvatures" in result
        assert len(result["mean_curvatures"]) == 50
        assert len(result["gaussian_curvatures"]) == 50
        assert result["principal_curvatures"].shape == (50, 2)

    def test_flat_plane_low_curvature(self):
        """A flat plane should have near-zero curvature."""
        X = make_plane_points(n=500, ambient_d=10)
        result = estimate_local_curvature(X, n_sample_points=100, k=30)
        median_mc = np.median(np.abs(result["mean_curvatures"]))
        # Should be close to 0
        assert median_mc < 0.5, f"Flat plane curvature too high: {median_mc}"

    def test_sphere_nonzero_curvature(self):
        """A sphere should have nonzero curvature."""
        X = make_sphere_points(n=1000, radius=1.0)
        result = estimate_local_curvature(X, n_sample_points=200, k=30)
        median_gc = np.median(np.abs(result["gaussian_curvatures"]))
        assert median_gc > 0.01, f"Sphere Gaussian curvature too low: {median_gc}"

    def test_reproducible_with_seed(self):
        X = np.random.randn(200, 10)
        r1 = estimate_local_curvature(X, n_sample_points=50, k=20, seed=123)
        r2 = estimate_local_curvature(X, n_sample_points=50, k=20, seed=123)
        np.testing.assert_array_equal(r1["mean_curvatures"], r2["mean_curvatures"])

    def test_sample_points_capped(self):
        """If n_sample_points > N, should sample N points."""
        X = np.random.randn(30, 5)
        result = estimate_local_curvature(X, n_sample_points=100, k=10)
        assert len(result["mean_curvatures"]) == 30

    def test_no_nans_with_ridge(self):
        """Ridge regularization should prevent NaNs."""
        X = np.random.randn(100, 20)
        result = estimate_local_curvature(X, n_sample_points=50, k=20, ridge_alpha=1.0)
        assert not np.any(np.isnan(result["mean_curvatures"]))


class TestCurvatureSummary:
    def test_summary_keys(self):
        result = {
            "mean_curvatures": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            "gaussian_curvatures": np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
            "principal_curvatures": np.array([[1, 2]] * 5),
        }
        summary = curvature_summary(result)
        assert "mean_curvatures_median" in summary
        assert "mean_curvatures_mean" in summary
        assert "gaussian_curvatures_std" in summary
        assert summary["mean_curvatures_median"] == 3.0
        assert summary["gaussian_curvatures_mean"] == pytest.approx(0.3)

    def test_summary_with_negative_values(self):
        result = {
            "mean_curvatures": np.array([-2.0, -1.0, 0.0, 1.0, 2.0]),
            "gaussian_curvatures": np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            "principal_curvatures": np.array([[0, 0]] * 5),
        }
        summary = curvature_summary(result)
        assert summary["mean_curvatures_median"] == 0.0
        assert summary["mean_curvatures_mean"] == 0.0
