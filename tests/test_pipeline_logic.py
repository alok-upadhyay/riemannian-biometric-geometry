"""Tests for pipeline logic (scripts) using synthetic data.

Tests the end-to-end flow: create synthetic embeddings -> run geometry -> run EER -> correlate.
"""

import json
import os
import tempfile

import numpy as np
import pytest

from src.geometry.intrinsic_dim import compute_intrinsic_dim
from src.geometry.curvature import estimate_local_curvature, curvature_summary
from src.geometry.cluster_topology import compute_cluster_metrics
from src.geometry.cross_modal import compute_cross_modal_metrics
from src.evaluation.eer import compute_cross_modal_eer, aggregate_embeddings_by_identity


def make_synthetic_embeddings(n_ids=50, n_per_id=10, face_dim=768, voice_dim=1024, seed=42):
    """Create synthetic face/voice embeddings with identity structure."""
    rng = np.random.RandomState(seed)
    shared_dim = 20

    # Shared identity representation
    Z = rng.randn(n_ids, shared_dim)

    face_proj = rng.randn(shared_dim, face_dim)
    voice_proj = rng.randn(shared_dim, voice_dim)

    face_emb = np.zeros((n_ids * n_per_id, face_dim))
    voice_emb = np.zeros((n_ids * n_per_id, voice_dim))
    labels = np.repeat(np.arange(n_ids), n_per_id)

    for i in range(n_ids):
        noise_face = rng.randn(n_per_id, face_dim) * 0.3
        noise_voice = rng.randn(n_per_id, voice_dim) * 0.3
        face_emb[i * n_per_id:(i + 1) * n_per_id] = Z[i] @ face_proj + noise_face
        voice_emb[i * n_per_id:(i + 1) * n_per_id] = Z[i] @ voice_proj + noise_voice

    # L2-normalize
    face_emb = face_emb / (np.linalg.norm(face_emb, axis=1, keepdims=True) + 1e-8)
    voice_emb = voice_emb / (np.linalg.norm(voice_emb, axis=1, keepdims=True) + 1e-8)

    return face_emb, voice_emb, labels


class TestEndToEndPipeline:
    """Test the full analysis pipeline on synthetic data."""

    @pytest.fixture
    def synthetic_data(self):
        return make_synthetic_embeddings()

    def test_intrinsic_dim_on_embeddings(self, synthetic_data):
        face_emb, voice_emb, _ = synthetic_data
        face_id = compute_intrinsic_dim(face_emb, method="mle", k=15)
        voice_id = compute_intrinsic_dim(voice_emb, method="mle", k=15)
        assert face_id > 0
        assert voice_id > 0
        # Both should be much less than ambient dim
        assert face_id < face_emb.shape[1]
        assert voice_id < voice_emb.shape[1]

    def test_curvature_on_embeddings(self, synthetic_data):
        face_emb, _, _ = synthetic_data
        result = estimate_local_curvature(face_emb, n_sample_points=50, k=20)
        summary = curvature_summary(result)
        assert np.isfinite(summary["mean_curvatures_median"])

    def test_cluster_topology_on_embeddings(self, synthetic_data):
        face_emb, _, labels = synthetic_data
        result = compute_cluster_metrics(face_emb, labels)
        assert result["intra_compactness"] >= 0
        assert result["inter_separation"] >= 0

    def test_cross_modal_metrics(self, synthetic_data):
        face_emb, voice_emb, labels = synthetic_data
        face_centroids, _ = aggregate_embeddings_by_identity(face_emb, labels)
        voice_centroids, _ = aggregate_embeddings_by_identity(voice_emb, labels)

        result = compute_cross_modal_metrics(
            face_centroids, voice_centroids,
            id_x=20.0, id_y=25.0,
            gh_max_points=50,
        )
        assert np.isfinite(result["gw_distance"])
        assert np.isfinite(result["cka"])
        assert result["id_mismatch"] == 5.0

    def test_cross_modal_eer(self, synthetic_data):
        face_emb, voice_emb, labels = synthetic_data
        result = compute_cross_modal_eer(
            face_emb, voice_emb, labels,
            train_fraction=0.8, n_cca_components=10, seed=42,
        )
        assert 0 <= result["eer"] <= 1
        # With shared identity signal, EER should be well below chance
        assert result["eer"] < 0.5

    def test_full_pipeline_produces_valid_json(self, synthetic_data):
        """Simulate the full pipeline output format."""
        face_emb, voice_emb, labels = synthetic_data

        # Step 1: ID
        face_id = compute_intrinsic_dim(face_emb, method="mle", k=15)
        voice_id = compute_intrinsic_dim(voice_emb, method="mle", k=15)

        # Step 2: Curvature
        curv = estimate_local_curvature(face_emb, n_sample_points=30, k=15)
        summary = curvature_summary(curv)

        # Step 3: Cluster topology
        cluster = compute_cluster_metrics(face_emb, labels)

        # Step 4: Cross-modal
        face_c, _ = aggregate_embeddings_by_identity(face_emb, labels)
        voice_c, _ = aggregate_embeddings_by_identity(voice_emb, labels)
        cm = compute_cross_modal_metrics(face_c, voice_c, id_x=face_id, id_y=voice_id, gh_max_points=50)

        # Step 5: EER
        eer = compute_cross_modal_eer(face_emb, voice_emb, labels, n_cca_components=10)

        # Verify all are JSON-serializable
        output = {
            "intrinsic_dim": {"face": face_id, "voice": voice_id},
            "curvature": summary,
            "cluster_topology": {k: v for k, v in cluster.items() if k != "per_class_compactness"},
            "cross_modal": cm,
            "eer": eer,
        }
        json_str = json.dumps(output)
        assert len(json_str) > 0

        # Reload and verify
        reloaded = json.loads(json_str)
        assert reloaded["intrinsic_dim"]["face"] > 0


class TestAggregation:
    def test_centroids_preserve_identity_count(self):
        n_ids = 30
        n_per_id = 8
        emb = np.random.randn(n_ids * n_per_id, 64)
        labels = np.repeat(np.arange(n_ids), n_per_id)
        centroids, unique_labels = aggregate_embeddings_by_identity(emb, labels)
        assert centroids.shape[0] == n_ids
        assert len(unique_labels) == n_ids

    def test_centroids_are_normalized(self):
        emb = np.random.randn(200, 32)
        labels = np.repeat(np.arange(20), 10)
        centroids, _ = aggregate_embeddings_by_identity(emb, labels)
        norms = np.linalg.norm(centroids, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=0.01)
