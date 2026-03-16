"""Intrinsic dimensionality estimation: MLE and TwoNN methods."""

import numpy as np
from sklearn.neighbors import NearestNeighbors


def mle_intrinsic_dim(X: np.ndarray, k: int = 20) -> float:
    """Maximum Likelihood Estimation of intrinsic dimensionality (Levina & Bickel 2004).

    Args:
        X: (N, D) data matrix
        k: number of nearest neighbors

    Returns:
        Estimated intrinsic dimensionality
    """
    N = X.shape[0]
    if N <= k:
        raise ValueError(f"Need N > k, got N={N}, k={k}")

    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto")
    nn.fit(X)
    distances, _ = nn.kneighbors(X)
    # distances[:, 0] is self (0), distances[:, 1:] are k nearest neighbors
    distances = distances[:, 1:]  # (N, k)

    # Avoid log(0) and handle duplicate points
    distances = np.maximum(distances, 1e-10)

    # MLE estimate per point
    T_k = distances[:, -1:]  # (N, 1) - distance to k-th neighbor
    ratios = T_k / distances[:, :-1]  # (N, k-1)
    # Clip ratios to avoid log(1) = 0 which causes division by zero
    ratios = np.maximum(ratios, 1.0 + 1e-10)
    log_ratios = np.log(ratios)  # (N, k-1)
    log_sums = log_ratios.sum(axis=1)  # (N,)

    # Filter out points where log_sum is too small (near-duplicate neighborhoods)
    valid = log_sums > 1e-8
    if valid.sum() == 0:
        # All points have near-duplicate neighbors; return ambient dim as fallback
        return float(X.shape[1])

    m_hat = (k - 1) / log_sums[valid]  # per valid point

    # Use median instead of mean for robustness against outliers
    return float(np.median(m_hat))


def twonn_intrinsic_dim(X: np.ndarray) -> float:
    """TwoNN intrinsic dimensionality estimator (Facco et al. 2017).

    Uses the ratio of distances to the first and second nearest neighbors.

    Args:
        X: (N, D) data matrix

    Returns:
        Estimated intrinsic dimensionality
    """
    N = X.shape[0]
    if N < 3:
        raise ValueError(f"Need N >= 3, got N={N}")

    nn = NearestNeighbors(n_neighbors=3, algorithm="auto")
    nn.fit(X)
    distances, _ = nn.kneighbors(X)
    # distances[:, 0] is self, [:, 1] is 1-NN, [:, 2] is 2-NN
    r1 = distances[:, 1]
    r2 = distances[:, 2]

    # Avoid division by zero
    valid = r1 > 1e-10
    mu = r2[valid] / r1[valid]

    # Sort mu values
    mu_sorted = np.sort(mu)
    n = len(mu_sorted)

    # Empirical CDF
    F = np.arange(1, n + 1) / n

    # Linear regression of log(1 - F) vs log(mu) to estimate d
    # F(mu) = 1 - mu^{-d}, so log(1 - F) = -d * log(mu)
    valid_F = F < 1.0 - 1e-10
    log_mu = np.log(mu_sorted[valid_F])
    log_1_minus_F = np.log(1.0 - F[valid_F])

    # Least squares: log(1-F) = -d * log(mu)
    d = -np.sum(log_1_minus_F * log_mu) / np.sum(log_mu**2)

    return float(d)


def compute_intrinsic_dim(
    X: np.ndarray, method: str = "mle", k: int = 20
) -> float:
    """Compute intrinsic dimensionality.

    Args:
        X: (N, D) data matrix
        method: "mle" or "twonn"
        k: neighbors for MLE method

    Returns:
        Estimated intrinsic dimensionality
    """
    if method == "mle":
        return mle_intrinsic_dim(X, k=k)
    elif method == "twonn":
        return twonn_intrinsic_dim(X)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'mle' or 'twonn'.")


def compute_local_intrinsic_dims(
    X: np.ndarray,
    labels: np.ndarray,
    method: str = "mle",
    k: int = 20,
    min_samples: int = 30,
) -> dict[str, float]:
    """Compute per-class local intrinsic dimensionality.

    Args:
        X: (N, D) data matrix
        labels: (N,) class labels
        method: "mle" or "twonn"
        k: neighbors for MLE method
        min_samples: skip classes with fewer samples

    Returns:
        Dict mapping label -> intrinsic dim
    """
    unique_labels = np.unique(labels)
    results = {}
    for label in unique_labels:
        mask = labels == label
        X_class = X[mask]
        if X_class.shape[0] < min_samples:
            continue
        k_local = min(k, X_class.shape[0] - 1)
        if k_local < 2:
            continue
        try:
            results[str(label)] = compute_intrinsic_dim(X_class, method=method, k=k_local)
        except Exception:
            continue
    return results
