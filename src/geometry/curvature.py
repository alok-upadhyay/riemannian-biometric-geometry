"""Local curvature estimation via kNN + local PCA + quadratic fit."""

import numpy as np
from sklearn.neighbors import NearestNeighbors


def estimate_local_curvature(
    X: np.ndarray,
    n_sample_points: int = 1000,
    k: int = 50,
    ridge_alpha: float = 1e-3,
    seed: int = 42,
) -> dict:
    """Estimate local sectional curvatures at sampled points.

    For each sample point:
    1. Find k nearest neighbors
    2. Local PCA to get tangent plane
    3. Fit quadratic surface (second fundamental form)
    4. Extract principal curvatures from shape operator

    Args:
        X: (N, D) data matrix
        n_sample_points: number of points to sample
        k: number of nearest neighbors
        ridge_alpha: regularization for quadratic fit
        seed: random seed for point sampling

    Returns:
        dict with keys:
            principal_curvatures: (n_sample, n_curvature_pairs) array
            mean_curvatures: (n_sample,) array
            gaussian_curvatures: (n_sample,) array
    """
    rng = np.random.RandomState(seed)
    N, D = X.shape
    n_sample_points = min(n_sample_points, N)

    sample_idx = rng.choice(N, size=n_sample_points, replace=False)

    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto")
    nn.fit(X)
    distances, indices = nn.kneighbors(X[sample_idx])

    all_mean_curvatures = []
    all_gaussian_curvatures = []
    all_principal_curvatures = []

    for i in range(n_sample_points):
        nbr_idx = indices[i, 1:]  # exclude self
        neighbors = X[nbr_idx]  # (k, D)
        center = X[sample_idx[i]]

        # Center the neighborhood
        centered = neighbors - center  # (k, D)

        # Local PCA: get d-dimensional tangent plane
        # Use at most min(k, D) components
        try:
            U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        except np.linalg.LinAlgError:
            all_mean_curvatures.append(0.0)
            all_gaussian_curvatures.append(0.0)
            all_principal_curvatures.append([0.0, 0.0])
            continue

        # Determine effective local dimension (retain 95% variance)
        var_explained = np.cumsum(S**2) / (np.sum(S**2) + 1e-10)
        d_local = max(2, int(np.searchsorted(var_explained, 0.95) + 1))
        d_local = min(d_local, len(S))

        if d_local < 2:
            all_mean_curvatures.append(0.0)
            all_gaussian_curvatures.append(0.0)
            all_principal_curvatures.append([0.0, 0.0])
            continue

        # Project onto tangent plane (first d_local PCA directions)
        tangent_basis = Vt[:d_local]  # (d_local, D)
        normal_basis = Vt[d_local:]  # (D - d_local, D)

        # Tangent coordinates
        t_coords = centered @ tangent_basis.T  # (k, d_local)

        if normal_basis.shape[0] == 0:
            all_mean_curvatures.append(0.0)
            all_gaussian_curvatures.append(0.0)
            all_principal_curvatures.append([0.0, 0.0])
            continue

        # Normal coordinates (use first normal direction)
        n_coords = centered @ normal_basis[0]  # (k,)

        # Fit quadratic: n = 0.5 * t^T H t
        # Build design matrix for quadratic terms
        n_quad_terms = d_local * (d_local + 1) // 2
        quad_features = np.zeros((k, n_quad_terms))
        col = 0
        for a in range(d_local):
            for b in range(a, d_local):
                if a == b:
                    quad_features[:, col] = t_coords[:, a] ** 2
                else:
                    quad_features[:, col] = 2.0 * t_coords[:, a] * t_coords[:, b]
                col += 1

        # Ridge regression: n_coords = quad_features @ h
        A = quad_features.T @ quad_features + ridge_alpha * np.eye(n_quad_terms)
        b_vec = quad_features.T @ n_coords
        try:
            h = np.linalg.solve(A, b_vec)
        except np.linalg.LinAlgError:
            all_mean_curvatures.append(0.0)
            all_gaussian_curvatures.append(0.0)
            all_principal_curvatures.append([0.0, 0.0])
            continue

        # Reconstruct Hessian matrix H (second fundamental form)
        H = np.zeros((d_local, d_local))
        col = 0
        for a in range(d_local):
            for b in range(a, d_local):
                H[a, b] = h[col]
                H[b, a] = h[col]
                col += 1

        # Principal curvatures = eigenvalues of H
        try:
            eigenvalues = np.linalg.eigvalsh(H)
        except np.linalg.LinAlgError:
            eigenvalues = np.zeros(d_local)

        # Clip extreme values for numerical stability
        eigenvalues = np.clip(eigenvalues, -100, 100)

        mean_curv = float(np.mean(eigenvalues))
        # Gaussian curvature = product of first two principal curvatures
        sorted_eigs = np.sort(np.abs(eigenvalues))[::-1]
        gauss_curv = float(sorted_eigs[0] * sorted_eigs[1]) if len(sorted_eigs) >= 2 else 0.0

        all_mean_curvatures.append(mean_curv)
        all_gaussian_curvatures.append(gauss_curv)
        all_principal_curvatures.append(sorted_eigs[:2].tolist() if len(sorted_eigs) >= 2 else [0.0, 0.0])

    return {
        "mean_curvatures": np.array(all_mean_curvatures),
        "gaussian_curvatures": np.array(all_gaussian_curvatures),
        "principal_curvatures": np.array(all_principal_curvatures),
    }


def curvature_summary(curvature_result: dict) -> dict:
    """Compute summary statistics from curvature estimation results.

    Returns:
        dict with median, mean, std for each curvature type
    """
    summary = {}
    for key in ["mean_curvatures", "gaussian_curvatures"]:
        vals = curvature_result[key]
        summary[f"{key}_median"] = float(np.median(vals))
        summary[f"{key}_mean"] = float(np.mean(vals))
        summary[f"{key}_std"] = float(np.std(vals))
        summary[f"{key}_q25"] = float(np.percentile(vals, 25))
        summary[f"{key}_q75"] = float(np.percentile(vals, 75))
    return summary
