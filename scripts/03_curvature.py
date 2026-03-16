"""Step 3: Estimate local curvature for all encoders.

Usage:
    PYTHONPATH=. python scripts/03_curvature.py --config configs/geometry.yaml
"""

import argparse
import json
import logging
import os

import numpy as np
from omegaconf import OmegaConf

from src.geometry.curvature import estimate_local_curvature, curvature_summary

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/geometry.yaml")
    parser.add_argument("--dataset", default="voxceleb1")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    emb_dir = os.path.join(cfg.results_dir, "embeddings")
    out_dir = os.path.join(cfg.results_dir, "curvature")
    os.makedirs(out_dir, exist_ok=True)

    all_encoders = list(cfg.encoders.face) + list(cfg.encoders.voice)

    results = {}

    for enc_name in all_encoders:
        emb_path = os.path.join(emb_dir, f"{enc_name}_{args.dataset}.npz")
        if not os.path.exists(emb_path):
            logger.warning(f"Missing {emb_path}, skipping")
            continue

        logger.info(f"Estimating curvature for {enc_name}...")
        data = np.load(emb_path)
        X = data["embeddings"]

        try:
            curv = estimate_local_curvature(
                X,
                n_sample_points=cfg.geometry.curvature.n_sample_points,
                k=cfg.geometry.curvature.k_neighbors,
                ridge_alpha=cfg.geometry.curvature.ridge_alpha,
            )

            summary = curvature_summary(curv)
            summary["encoder"] = enc_name
            summary["n_samples"] = X.shape[0]
            results[enc_name] = summary

            # Save raw curvature values
            np.savez(
                os.path.join(out_dir, f"{enc_name}_{args.dataset}_raw.npz"),
                mean_curvatures=curv["mean_curvatures"],
                gaussian_curvatures=curv["gaussian_curvatures"],
                principal_curvatures=curv["principal_curvatures"],
            )

            logger.info(f"  {enc_name}: mean_curv={summary['mean_curvatures_median']:.4f}, "
                        f"gauss_curv={summary['gaussian_curvatures_median']:.6f}")

        except Exception as e:
            logger.error(f"  {enc_name} curvature failed: {e}")
            results[enc_name] = {"encoder": enc_name, "error": str(e)}

    out_path = os.path.join(out_dir, f"curvature_{args.dataset}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved {out_path}")


if __name__ == "__main__":
    main()
