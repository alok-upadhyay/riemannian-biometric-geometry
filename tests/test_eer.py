"""Tests for EER computation and CCA alignment."""

import numpy as np
import pytest

from src.evaluation.eer import (
    compute_eer,
    aggregate_embeddings_by_identity,
    cca_align,
    compute_cross_modal_eer,
)


class TestComputeEER:
    def test_perfect_separation(self):
        """Perfect separation should give EER = 0."""
        labels = np.array([1, 1, 1, 0, 0, 0])
        scores = np.array([0.9, 0.8, 0.7, 0.1, 0.2, 0.3])
        eer, threshold = compute_eer(labels, scores)
        assert eer == pytest.approx(0.0, abs=0.01)

    def test_random_scores(self):
        """Random scores should give EER ~ 0.5."""
        rng = np.random.RandomState(42)
        labels = np.array([1] * 500 + [0] * 500)
        scores = rng.rand(1000)
        eer, _ = compute_eer(labels, scores)
        assert 0.3 < eer < 0.7, f"Random EER should be ~0.5, got {eer}"

    def test_eer_range(self):
        """EER should be in [0, 1]."""
        rng = np.random.RandomState(42)
        labels = np.array([1] * 100 + [0] * 100)
        scores = rng.rand(200)
        eer, _ = compute_eer(labels, scores)
        assert 0 <= eer <= 1

    def test_returns_threshold(self):
        labels = np.array([1, 1, 0, 0])
        scores = np.array([0.9, 0.6, 0.4, 0.1])
        eer, threshold = compute_eer(labels, scores)
        assert np.isfinite(threshold)


class TestAggregateEmbeddingsByIdentity:
    def test_basic_aggregation(self):
        embeddings = np.array([
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ])
        labels = np.array(["a", "a", "b", "b"])
        centroids, unique_labels = aggregate_embeddings_by_identity(embeddings, labels)
        assert centroids.shape == (2, 2)
        assert unique_labels == ["a", "b"]

    def test_l2_normalized(self):
        embeddings = np.random.randn(100, 10)
        labels = np.array([0] * 50 + [1] * 50)
        centroids, _ = aggregate_embeddings_by_identity(embeddings, labels)
        norms = np.linalg.norm(centroids, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=0.01)

    def test_single_sample_per_id(self):
        embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
        labels = np.array(["a", "b"])
        centroids, unique_labels = aggregate_embeddings_by_identity(embeddings, labels)
        assert centroids.shape == (2, 2)

    def test_many_classes(self):
        embeddings = np.random.randn(500, 20)
        labels = np.repeat(np.arange(50), 10)
        centroids, unique_labels = aggregate_embeddings_by_identity(embeddings, labels)
        assert centroids.shape == (50, 20)
        assert len(unique_labels) == 50


class TestCCAAlign:
    def test_output_shapes(self):
        X = np.random.randn(100, 10)
        Y = np.random.randn(100, 8)
        X_cca, Y_cca, cca = cca_align(X, Y, n_components=5)
        assert X_cca.shape == (100, 5)
        assert Y_cca.shape == (100, 5)

    def test_n_components_capped(self):
        """n_components should be capped at min dimension."""
        X = np.random.randn(50, 3)
        Y = np.random.randn(50, 5)
        X_cca, Y_cca, cca = cca_align(X, Y, n_components=128)
        assert X_cca.shape[1] <= 3

    def test_correlated_inputs(self):
        """CCA should find high correlation for related inputs."""
        rng = np.random.RandomState(42)
        Z = rng.randn(100, 5)
        X = Z @ rng.randn(5, 10) + rng.randn(100, 10) * 0.1
        Y = Z @ rng.randn(5, 8) + rng.randn(100, 8) * 0.1
        X_cca, Y_cca, _ = cca_align(X, Y, n_components=5)
        # First CCA component should be correlated
        corr = np.corrcoef(X_cca[:, 0], Y_cca[:, 0])[0, 1]
        assert abs(corr) > 0.5


class TestComputeCrossModalEER:
    def test_basic_computation(self):
        rng = np.random.RandomState(42)
        n_ids = 50
        n_per_id = 10
        labels = np.repeat(np.arange(n_ids), n_per_id)

        # Create face and voice embeddings with some shared structure
        Z = rng.randn(n_ids, 5)
        face_emb = np.zeros((n_ids * n_per_id, 10))
        voice_emb = np.zeros((n_ids * n_per_id, 8))
        for i in range(n_ids):
            face_emb[i * n_per_id:(i + 1) * n_per_id] = (
                Z[i] @ rng.randn(5, 10) + rng.randn(n_per_id, 10) * 0.5
            )
            voice_emb[i * n_per_id:(i + 1) * n_per_id] = (
                Z[i] @ rng.randn(5, 8) + rng.randn(n_per_id, 8) * 0.5
            )

        result = compute_cross_modal_eer(
            face_emb, voice_emb, labels,
            train_fraction=0.8, n_cca_components=5, seed=42,
        )
        assert "eer" in result
        assert 0 <= result["eer"] <= 1
        assert result["n_train_ids"] > 0
        assert result["n_test_ids"] > 0

    def test_eer_below_chance_with_signal(self):
        """With shared identity signal, CCA-aligned EER should be below 0.5."""
        rng = np.random.RandomState(42)
        n_ids = 200
        n_per_id = 10
        labels = np.repeat(np.arange(n_ids), n_per_id)

        # Strong shared signal with fixed projections
        Z = rng.randn(n_ids, 10)
        face_proj = rng.randn(10, 20)
        voice_proj = rng.randn(10, 15)
        face_emb = np.zeros((n_ids * n_per_id, 20))
        voice_emb = np.zeros((n_ids * n_per_id, 15))
        for i in range(n_ids):
            face_emb[i * n_per_id:(i + 1) * n_per_id] = (
                Z[i] @ face_proj + rng.randn(n_per_id, 20) * 0.05
            )
            voice_emb[i * n_per_id:(i + 1) * n_per_id] = (
                Z[i] @ voice_proj + rng.randn(n_per_id, 15) * 0.05
            )

        result = compute_cross_modal_eer(
            face_emb, voice_emb, labels,
            train_fraction=0.8, n_cca_components=10, seed=42,
        )
        assert result["eer"] < 0.5, f"EER should be < 0.5 with shared signal, got {result['eer']}"

    def test_too_few_ids(self):
        """Should handle gracefully when too few test IDs."""
        face_emb = np.random.randn(10, 5)
        voice_emb = np.random.randn(10, 5)
        labels = np.array([0] * 10)  # Only 1 identity
        result = compute_cross_modal_eer(
            face_emb, voice_emb, labels, train_fraction=0.8,
        )
        # Should return nan or error
        assert "error" in result or np.isnan(result.get("eer", float("nan")))
