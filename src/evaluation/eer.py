"""Cross-modal EER evaluation with CCA alignment."""

import numpy as np
from sklearn.cross_decomposition import CCA
from sklearn.metrics import roc_curve


def compute_eer(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Compute Equal Error Rate.

    Args:
        labels: binary labels (1=same, 0=different)
        scores: similarity scores

    Returns:
        (eer, threshold)
    """
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2
    threshold = thresholds[idx]
    return float(eer), float(threshold)


def aggregate_embeddings_by_identity(
    embeddings: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, list]:
    """Average embeddings per identity and L2-normalize.

    Args:
        embeddings: (N, D) embedding matrix
        labels: (N,) identity labels

    Returns:
        (centroids, unique_labels) where centroids is (n_ids, D)
    """
    unique_labels = sorted(set(labels.tolist() if isinstance(labels, np.ndarray) else labels))
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}
    D = embeddings.shape[1]

    centroids = np.zeros((len(unique_labels), D))
    counts = np.zeros(len(unique_labels))

    for i, label in enumerate(labels):
        idx = label_to_idx[label if not isinstance(label, np.generic) else label.item()]
        centroids[idx] += embeddings[i]
        counts[idx] += 1

    centroids = centroids / counts[:, np.newaxis]
    # L2-normalize
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroids = centroids / (norms + 1e-8)

    return centroids, unique_labels


def cca_align(
    X: np.ndarray,
    Y: np.ndarray,
    n_components: int = 128,
) -> tuple[np.ndarray, np.ndarray, CCA]:
    """Align two embedding spaces via CCA.

    Args:
        X: (N, D1) first embedding space
        Y: (N, D2) second embedding space
        n_components: number of CCA components

    Returns:
        (X_cca, Y_cca, cca_model)
    """
    n_components = min(n_components, X.shape[1], Y.shape[1], X.shape[0])
    cca = CCA(n_components=n_components, max_iter=1000)
    X_cca, Y_cca = cca.fit_transform(X, Y)
    return X_cca, Y_cca, cca


def compute_cross_modal_eer(
    face_embeddings: np.ndarray,
    voice_embeddings: np.ndarray,
    labels: np.ndarray,
    train_fraction: float = 0.8,
    n_cca_components: int = 128,
    seed: int = 42,
) -> dict:
    """Compute cross-modal EER with CCA alignment.

    1. Split identities into train/test
    2. CCA-align on train identities
    3. Cosine EER on test identities

    Args:
        face_embeddings: (N, D1) face embeddings
        voice_embeddings: (N, D2) voice embeddings
        labels: (N,) identity labels
        train_fraction: fraction of identities for CCA fitting
        n_cca_components: CCA dimensionality
        seed: random seed

    Returns:
        dict with eer, threshold, n_train_ids, n_test_ids, raw_eer
    """
    rng = np.random.RandomState(seed)

    # Aggregate by identity
    face_centroids, unique_labels = aggregate_embeddings_by_identity(face_embeddings, labels)
    voice_centroids, _ = aggregate_embeddings_by_identity(voice_embeddings, labels)

    n_ids = len(unique_labels)
    perm = rng.permutation(n_ids)
    n_train = int(n_ids * train_fraction)

    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    if len(test_idx) < 2:
        return {"eer": float("nan"), "error": "Not enough test identities"}

    # CCA alignment on train set with PCA preprocessing for numerical stability
    from sklearn.decomposition import PCA

    face_train = face_centroids[train_idx]
    voice_train = voice_centroids[train_idx]

    # PCA to reduce dimensionality before CCA (prevents SVD convergence issues)
    pca_dim = min(50, face_train.shape[1], voice_train.shape[1], len(train_idx) - 1)
    pca_face = PCA(n_components=pca_dim, random_state=seed)
    pca_voice = PCA(n_components=pca_dim, random_state=seed)
    face_train = pca_face.fit_transform(face_train)
    voice_train = pca_voice.fit_transform(voice_train)

    n_comp = min(n_cca_components, pca_dim, len(train_idx) - 1)
    cca = CCA(n_components=n_comp, max_iter=2000)
    cca.fit(face_train, voice_train)

    # Transform test set using PCA then CCA projections
    face_test = pca_face.transform(face_centroids[test_idx])
    voice_test = pca_voice.transform(voice_centroids[test_idx])
    # CCA.transform only handles X; for Y we project manually using y_rotations_
    face_test_centered = face_test - cca._x_mean
    voice_test_centered = voice_test - cca._y_mean
    face_test_cca = face_test_centered @ cca.x_rotations_
    voice_test_cca = voice_test_centered @ cca.y_rotations_

    # L2-normalize
    face_test_cca = face_test_cca / (np.linalg.norm(face_test_cca, axis=1, keepdims=True) + 1e-8)
    voice_test_cca = voice_test_cca / (np.linalg.norm(voice_test_cca, axis=1, keepdims=True) + 1e-8)

    # Generate verification pairs: all same-id + sampled different-id
    n_test = len(test_idx)
    scores = []
    pair_labels = []

    # Positive pairs: face_i matched with voice_i
    for i in range(n_test):
        score = float(face_test_cca[i] @ voice_test_cca[i])
        scores.append(score)
        pair_labels.append(1)

    # Negative pairs: face_i matched with voice_j (i != j)
    for i in range(n_test):
        for j in range(n_test):
            if i != j:
                score = float(face_test_cca[i] @ voice_test_cca[j])
                scores.append(score)
                pair_labels.append(0)

    scores = np.array(scores)
    pair_labels = np.array(pair_labels)

    eer, threshold = compute_eer(pair_labels, scores)

    # Also compute raw (unaligned) EER for sanity check
    face_test_raw = face_centroids[test_idx]
    voice_test_raw = voice_centroids[test_idx]
    # For raw EER, project to same dim via simple truncation/padding for comparison
    # Just use CCA-aligned scores as the main metric
    raw_scores = []
    raw_labels = []
    # Use cosine distance in original spaces (won't be meaningful if dims differ)
    # Skip raw EER if dimensions don't match
    raw_eer = float("nan")
    if face_test_raw.shape[1] == voice_test_raw.shape[1]:
        f_norm = face_test_raw / (np.linalg.norm(face_test_raw, axis=1, keepdims=True) + 1e-8)
        v_norm = voice_test_raw / (np.linalg.norm(voice_test_raw, axis=1, keepdims=True) + 1e-8)
        for i in range(n_test):
            raw_scores.append(float(f_norm[i] @ v_norm[i]))
            raw_labels.append(1)
            for j in range(n_test):
                if i != j:
                    raw_scores.append(float(f_norm[i] @ v_norm[j]))
                    raw_labels.append(0)
        raw_eer, _ = compute_eer(np.array(raw_labels), np.array(raw_scores))

    return {
        "eer": eer,
        "threshold": threshold,
        "n_train_ids": n_train,
        "n_test_ids": len(test_idx),
        "raw_eer": raw_eer,
    }
