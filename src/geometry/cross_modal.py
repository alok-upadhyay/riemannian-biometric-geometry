"""Cross-modal geometry metrics: GH distance, spectral gap, CKA, ID mismatch."""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import laplacian
from sklearn.neighbors import kneighbors_graph


def gromov_wasserstein_distance(
    X: np.ndarray, Y: np.ndarray, max_points: int = 500, seed: int = 42
) -> float:
    """Compute Gromov-Wasserstein distance between two metric spaces.

    Uses the POT library for GW computation on distance matrices.

    Args:
        X: (N, D1) first embedding space
        Y: (M, D2) second embedding space
        max_points: subsample if N or M exceeds this
        seed: random seed

    Returns:
        GW distance (float)
    """
    import ot

    rng = np.random.RandomState(seed)

    if X.shape[0] > max_points:
        idx = rng.choice(X.shape[0], max_points, replace=False)
        X = X[idx]
    if Y.shape[0] > max_points:
        idx = rng.choice(Y.shape[0], max_points, replace=False)
        Y = Y[idx]

    # Compute intra-space distance matrices
    C1 = cdist(X, X, metric="cosine")
    C2 = cdist(Y, Y, metric="cosine")

    # Normalize
    C1 = C1 / (C1.max() + 1e-10)
    C2 = C2 / (C2.max() + 1e-10)

    # Uniform distributions
    p = np.ones(C1.shape[0]) / C1.shape[0]
    q = np.ones(C2.shape[0]) / C2.shape[0]

    gw_dist, log = ot.gromov.gromov_wasserstein2(
        C1, C2, p, q, loss_fun="square_loss", log=True
    )

    return float(gw_dist)


def spectral_gap_divergence(
    X: np.ndarray,
    Y: np.ndarray,
    k: int = 20,
    n_eigenvalues: int = 50,
) -> float:
    """Compute spectral gap divergence between two embedding spaces.

    Builds kNN graph Laplacians, computes eigenvalues, and measures
    L2 distance between normalized spectra.

    Args:
        X: (N, D1) first embedding space
        Y: (N, D2) second embedding space (same N assumed — identity-matched)
        k: number of neighbors for kNN graph
        n_eigenvalues: number of eigenvalues to compare

    Returns:
        L2 distance between normalized spectra
    """
    def _spectrum(Z, k, n_eig):
        k_actual = min(k, Z.shape[0] - 1)
        adj = kneighbors_graph(Z, n_neighbors=k_actual, mode="connectivity", include_self=False)
        adj = ((adj + adj.T) > 0).astype(float)  # symmetrize
        L = laplacian(adj, normed=True)
        n_eig_actual = min(n_eig, L.shape[0] - 1)
        from scipy.sparse.linalg import eigsh
        eigenvalues, _ = eigsh(L, k=n_eig_actual, which="SM")
        eigenvalues = np.sort(np.abs(eigenvalues))
        # Normalize
        eigenvalues = eigenvalues / (eigenvalues.max() + 1e-10)
        return eigenvalues

    spec_x = _spectrum(X, k, n_eigenvalues)
    spec_y = _spectrum(Y, k, n_eigenvalues)

    # Pad to same length
    max_len = max(len(spec_x), len(spec_y))
    spec_x = np.pad(spec_x, (0, max_len - len(spec_x)))
    spec_y = np.pad(spec_y, (0, max_len - len(spec_y)))

    return float(np.linalg.norm(spec_x - spec_y))


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute linear Centered Kernel Alignment (CKA).

    Args:
        X: (N, D1) first representation
        Y: (N, D2) second representation

    Returns:
        CKA similarity in [0, 1]
    """
    # Center
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)

    # Gram matrices
    XtX = X @ X.T  # (N, N)
    YtY = Y @ Y.T  # (N, N)

    # HSIC
    hsic_xy = np.sum(XtX * YtY)
    hsic_xx = np.sum(XtX * XtX)
    hsic_yy = np.sum(YtY * YtY)

    cka = hsic_xy / (np.sqrt(hsic_xx * hsic_yy) + 1e-10)
    return float(cka)


def intrinsic_dim_mismatch(id_x: float, id_y: float) -> float:
    """Compute absolute intrinsic dimension mismatch."""
    return abs(id_x - id_y)


def compute_cross_modal_metrics(
    X: np.ndarray,
    Y: np.ndarray,
    id_x: float | None = None,
    id_y: float | None = None,
    gh_max_points: int = 500,
    spectral_k: int = 20,
    spectral_n_eigenvalues: int = 50,
    seed: int = 42,
) -> dict:
    """Compute all cross-modal geometry metrics.

    Args:
        X: (N, D1) face embeddings (identity-averaged)
        Y: (N, D2) voice embeddings (identity-averaged)
        id_x: intrinsic dim of X (precomputed)
        id_y: intrinsic dim of Y (precomputed)
        gh_max_points: max points for GW computation
        spectral_k: k for spectral gap
        spectral_n_eigenvalues: eigenvalues for spectral gap
        seed: random seed

    Returns:
        dict with all cross-modal metrics
    """
    results = {}

    # Gromov-Wasserstein distance
    try:
        results["gw_distance"] = gromov_wasserstein_distance(
            X, Y, max_points=gh_max_points, seed=seed
        )
    except Exception as e:
        results["gw_distance"] = float("nan")
        results["gw_error"] = str(e)

    # Spectral gap divergence
    try:
        results["spectral_gap"] = spectral_gap_divergence(
            X, Y, k=spectral_k, n_eigenvalues=spectral_n_eigenvalues
        )
    except Exception as e:
        results["spectral_gap"] = float("nan")
        results["spectral_error"] = str(e)

    # Linear CKA
    try:
        results["cka"] = linear_cka(X, Y)
    except Exception as e:
        results["cka"] = float("nan")
        results["cka_error"] = str(e)

    # ID mismatch
    if id_x is not None and id_y is not None:
        results["id_mismatch"] = intrinsic_dim_mismatch(id_x, id_y)

    return results
