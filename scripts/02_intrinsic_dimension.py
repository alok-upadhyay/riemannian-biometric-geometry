"""Step 2: Compute intrinsic dimensionality for all encoders.

Usage:
    PYTHONPATH=. python scripts/02_intrinsic_dimension.py --config configs/geometry.yaml
"""

import argparse
import json
import logging
import os

import numpy as np
from omegaconf import OmegaConf

from src.geometry.intrinsic_dim import compute_intrinsic_dim, compute_local_intrinsic_dims

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/geometry.yaml")
    parser.add_argument("--dataset", default="voxceleb1")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    emb_dir = os.path.join(cfg.results_dir, "embeddings")
    out_dir = os.path.join(cfg.results_dir, "intrinsic_dim")
    os.makedirs(out_dir, exist_ok=True)

    all_encoders = list(cfg.encoders.face) + list(cfg.encoders.voice)
    k = cfg.geometry.intrinsic_dim.k_neighbors
    methods = list(cfg.geometry.intrinsic_dim.methods)

    results = {}

    for enc_name in all_encoders:
        emb_path = os.path.join(emb_dir, f"{enc_name}_{args.dataset}.npz")
        if not os.path.exists(emb_path):
            logger.warning(f"Missing {emb_path}, skipping")
            continue

        logger.info(f"Computing intrinsic dim for {enc_name}...")
        data = np.load(emb_path)
        X = data["embeddings"]
        labels = data["labels"]

        # Deduplicate embeddings (face encoders produce duplicates for same image)
        X_unique, unique_idx = np.unique(X, axis=0, return_index=True)
        labels_unique = labels[unique_idx]
        logger.info(f"  {enc_name}: {X.shape[0]} total -> {X_unique.shape[0]} unique embeddings")
        X = X_unique
        labels = labels_unique

        enc_results = {"encoder": enc_name, "n_samples": X.shape[0], "ambient_dim": X.shape[1]}

        for method in methods:
            try:
                global_id = compute_intrinsic_dim(X, method=method, k=k)
                enc_results[f"global_{method}"] = global_id
                logger.info(f"  {enc_name} global {method}: {global_id:.2f}")
            except Exception as e:
                logger.warning(f"  {enc_name} global {method} failed: {e}")
                enc_results[f"global_{method}"] = None

            try:
                local_ids = compute_local_intrinsic_dims(X, labels, method=method, k=k)
                if local_ids:
                    values = list(local_ids.values())
                    enc_results[f"local_{method}_mean"] = float(np.mean(values))
                    enc_results[f"local_{method}_std"] = float(np.std(values))
                    enc_results[f"local_{method}_median"] = float(np.median(values))
                    enc_results[f"local_{method}_n_classes"] = len(values)
            except Exception as e:
                logger.warning(f"  {enc_name} local {method} failed: {e}")

        results[enc_name] = enc_results

    out_path = os.path.join(out_dir, f"intrinsic_dim_{args.dataset}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved {out_path}")


if __name__ == "__main__":
    main()
