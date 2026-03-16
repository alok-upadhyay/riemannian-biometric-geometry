"""Tests for cluster topology metrics."""

import numpy as np
import pytest

from src.geometry.cluster_topology import compute_cluster_metrics


def make_well_separated_clusters(n_per_class=50, n_classes=5, d=10, separation=10.0, seed=42):
    """Create well-separated Gaussian clusters."""
    rng = np.random.RandomState(seed)
    X = []
    labels = []
    for c in range(n_classes):
        center = rng.randn(d) * separation
        points = rng.randn(n_per_class, d) * 0.1 + center
        X.append(points)
        labels.extend([c] * n_per_class)
    X = np.vstack(X)
    # L2-normalize
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    return X, np.array(labels)


def make_overlapping_clusters(n_per_class=50, n_classes=5, d=10, seed=42):
    """Create heavily overlapping clusters."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n_per_class * n_classes, d)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    labels = np.repeat(np.arange(n_classes), n_per_class)
    return X, labels


class TestComputeClusterMetrics:
    def test_output_keys(self):
        X, labels = make_well_separated_clusters()
        result = compute_cluster_metrics(X, labels)
        assert "intra_compactness" in result
        assert "inter_separation" in result
        assert "compactness_gap" in result
        assert "silhouette_approx" in result

    def test_well_separated_positive_gap(self):
        """Well-separated clusters should have positive compactness gap."""
        X, labels = make_well_separated_clusters(separation=20.0)
        result = compute_cluster_metrics(X, labels)
        assert result["compactness_gap"] > 0, f"Gap should be positive: {result['compactness_gap']}"
        assert result["silhouette_approx"] > 0

    def test_overlapping_lower_gap(self):
        """Overlapping clusters should have lower gap than separated ones."""
        X_sep, labels_sep = make_well_separated_clusters(separation=20.0)
        X_ovl, labels_ovl = make_overlapping_clusters()

        result_sep = compute_cluster_metrics(X_sep, labels_sep)
        result_ovl = compute_cluster_metrics(X_ovl, labels_ovl)

        assert result_sep["compactness_gap"] > result_ovl["compactness_gap"]

    def test_intra_compactness_positive(self):
        X, labels = make_well_separated_clusters()
        result = compute_cluster_metrics(X, labels)
        assert result["intra_compactness"] >= 0

    def test_min_samples_filter(self):
        """Classes with too few samples should be filtered out."""
        X = np.random.randn(100, 10)
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
        labels = np.array([0] * 50 + [1] * 50)
        result = compute_cluster_metrics(X, labels, min_samples_per_id=100)
        # All classes filtered, should return defaults
        assert result["intra_compactness"] == 0.0

    def test_single_class_insufficient(self):
        """Need at least 2 classes."""
        X = np.random.randn(100, 10)
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
        labels = np.zeros(100)
        result = compute_cluster_metrics(X, labels)
        assert result["intra_compactness"] == 0.0

    def test_gap_equals_difference(self):
        X, labels = make_well_separated_clusters()
        result = compute_cluster_metrics(X, labels)
        expected_gap = result["inter_separation"] - result["intra_compactness"]
        assert result["compactness_gap"] == pytest.approx(expected_gap, abs=1e-10)
