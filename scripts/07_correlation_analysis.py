"""Step 7: Correlation analysis — geometry metrics vs. EER.

Usage:
    PYTHONPATH=. python scripts/07_correlation_analysis.py --config configs/geometry.yaml
"""

import argparse
import json
import logging
import os

import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import LinearRegression
from omegaconf import OmegaConf

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def bootstrap_correlation(x, y, func, n_bootstrap=1000, seed=42):
    """Bootstrap 95% CI for a correlation function."""
    rng = np.random.RandomState(seed)
    boot_vals = []
    for _ in range(n_bootstrap):
        idx = rng.choice(len(x), size=len(x), replace=True)
        try:
            val, _ = func(x[idx], y[idx])
            if np.isfinite(val):
                boot_vals.append(val)
        except Exception:
            continue
    if not boot_vals:
        return float("nan"), float("nan")
    return float(np.percentile(boot_vals, 2.5)), float(np.percentile(boot_vals, 97.5))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/geometry.yaml")
    parser.add_argument("--dataset", default="voxceleb1")
    parser.add_argument("--pool-datasets", action="store_true",
                        help="Pool voxceleb1 + mavceleb for more data points")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    cm_dir = os.path.join(cfg.results_dir, "cross_modal")
    out_dir = os.path.join(cfg.results_dir, "tables")
    os.makedirs(out_dir, exist_ok=True)

    datasets = [args.dataset]
    if args.pool_datasets:
        datasets = ["voxceleb1", "mavceleb"]

    # Load and merge results
    pairs = []
    for ds in datasets:
        cm_path = os.path.join(cm_dir, f"cross_modal_{ds}.json")
        eer_path = os.path.join(cm_dir, f"eer_{ds}.json")

        if not os.path.exists(cm_path) or not os.path.exists(eer_path):
            logger.warning(f"Missing results for {ds}")
            continue

        with open(cm_path) as f:
            cm_results = json.load(f)
        with open(eer_path) as f:
            eer_results = json.load(f)

        for pair_name in cm_results:
            if pair_name not in eer_results:
                continue
            cm = cm_results[pair_name]
            eer = eer_results[pair_name]
            if "error" in cm or "error" in eer:
                continue
            if not np.isfinite(eer.get("eer", float("nan"))):
                continue

            pairs.append({
                "pair": pair_name,
                "dataset": ds,
                "gw_distance": cm.get("gw_distance"),
                "spectral_gap": cm.get("spectral_gap"),
                "cka": cm.get("cka"),
                "id_mismatch": cm.get("id_mismatch"),
                "eer": eer["eer"],
            })

    if len(pairs) < 4:
        logger.error(f"Only {len(pairs)} valid pairs — not enough for correlation analysis")
        return

    logger.info(f"Analyzing {len(pairs)} encoder pairs")

    # Extract arrays
    eer_vals = np.array([p["eer"] for p in pairs])

    metric_names = ["gw_distance", "spectral_gap", "cka", "id_mismatch"]
    correlations = {}

    for metric in metric_names:
        vals = np.array([p.get(metric, float("nan")) for p in pairs])
        valid = np.isfinite(vals) & np.isfinite(eer_vals)
        if valid.sum() < 4:
            logger.warning(f"  {metric}: not enough valid values ({valid.sum()})")
            continue

        x = vals[valid]
        y = eer_vals[valid]

        spearman_r, spearman_p = spearmanr(x, y)
        pearson_r, pearson_p = pearsonr(x, y)

        ci_low, ci_high = bootstrap_correlation(x, y, spearmanr)

        correlations[metric] = {
            "spearman_r": float(spearman_r),
            "spearman_p": float(spearman_p),
            "pearson_r": float(pearson_r),
            "pearson_p": float(pearson_p),
            "bootstrap_ci_95": [ci_low, ci_high],
            "n_pairs": int(valid.sum()),
        }

        logger.info(f"  {metric}: Spearman r={spearman_r:.3f} (p={spearman_p:.4f}), "
                    f"Pearson r={pearson_r:.3f} (p={pearson_p:.4f}), "
                    f"95% CI=[{ci_low:.3f}, {ci_high:.3f}]")

    # Multivariate regression: EER ~ all metrics
    feature_names = []
    feature_matrix = []
    for metric in metric_names:
        vals = np.array([p.get(metric, float("nan")) for p in pairs])
        if np.all(np.isfinite(vals)):
            feature_names.append(metric)
            feature_matrix.append(vals)

    if feature_matrix:
        X_reg = np.stack(feature_matrix, axis=1)
        # Standardize
        X_mean = X_reg.mean(axis=0)
        X_std = X_reg.std(axis=0) + 1e-8
        X_norm = (X_reg - X_mean) / X_std

        reg = LinearRegression()
        reg.fit(X_norm, eer_vals)

        regression = {
            "features": feature_names,
            "coefficients": reg.coef_.tolist(),
            "intercept": float(reg.intercept_),
            "r_squared": float(reg.score(X_norm, eer_vals)),
        }
        logger.info(f"\n  Multivariate regression R²={regression['r_squared']:.3f}")
        for name, coef in zip(feature_names, reg.coef_):
            logger.info(f"    {name}: β={coef:.4f}")
    else:
        regression = None

    # Save results
    output = {
        "n_pairs": len(pairs),
        "datasets": datasets,
        "correlations": correlations,
        "regression": regression,
        "pairs": pairs,
    }

    out_path = os.path.join(out_dir, "correlation_analysis.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
