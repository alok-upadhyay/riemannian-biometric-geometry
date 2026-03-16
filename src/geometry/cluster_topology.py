"""Cluster topology metrics: intra-class compactness, inter-class separation."""

import numpy as np
from sklearn.metrics.pairwise import cosine_distances


def compute_cluster_metrics(
    X: np.ndarray,
    labels: np.ndarray,
    min_samples_per_id: int = 5,
) -> dict:
    """Compute intra-class compactness and inter-class separation.

    Args:
        X: (N, D) L2-normalized embedding matrix
        labels: (N,) identity labels
        min_samples_per_id: skip identities with fewer samples

    Returns:
        dict with keys:
            intra_compactness: mean intra-class cosine distance
            inter_separation: mean inter-class centroid cosine distance
            compactness_gap: inter_separation - intra_compactness
            per_class_compactness: dict of label -> mean intra-class cosine distance
            silhouette_approx: approximate silhouette score
    """
    unique_labels = np.unique(labels)

    # Filter classes with enough samples
    valid_labels = []
    for label in unique_labels:
        mask = labels == label
        if mask.sum() >= min_samples_per_id:
            valid_labels.append(label)

    if len(valid_labels) < 2:
        return {
            "intra_compactness": 0.0,
            "inter_separation": 0.0,
            "compactness_gap": 0.0,
            "per_class_compactness": {},
            "silhouette_approx": 0.0,
        }

    # Compute centroids
    centroids = {}
    per_class_compactness = {}
    intra_distances = []

    for label in valid_labels:
        mask = labels == label
        X_class = X[mask]
        centroid = X_class.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
        centroids[label] = centroid

        # Intra-class: mean cosine distance to centroid
        dists = cosine_distances(X_class, centroid.reshape(1, -1)).flatten()
        per_class_compactness[str(label)] = float(np.mean(dists))
        intra_distances.extend(dists.tolist())

    # Inter-class: mean cosine distance between centroids
    centroid_matrix = np.stack([centroids[l] for l in valid_labels])
    centroid_dists = cosine_distances(centroid_matrix)
    # Upper triangle (exclude diagonal)
    n_classes = len(valid_labels)
    upper_tri_idx = np.triu_indices(n_classes, k=1)
    inter_dists = centroid_dists[upper_tri_idx]

    intra_compact = float(np.mean(intra_distances))
    inter_sep = float(np.mean(inter_dists))
    gap = inter_sep - intra_compact

    # Approximate silhouette: (inter - intra) / max(inter, intra)
    silhouette = gap / max(inter_sep, intra_compact, 1e-8)

    return {
        "intra_compactness": intra_compact,
        "inter_separation": inter_sep,
        "compactness_gap": gap,
        "per_class_compactness": per_class_compactness,
        "silhouette_approx": float(silhouette),
    }
