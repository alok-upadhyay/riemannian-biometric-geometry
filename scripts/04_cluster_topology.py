"""Step 4: Compute cluster topology metrics for all encoders.

Usage:
    PYTHONPATH=. python scripts/04_cluster_topology.py --config configs/geometry.yaml
"""

import argparse
import json
import logging
import os

import numpy as np
from omegaconf import OmegaConf

from src.geometry.cluster_topology import compute_cluster_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/geometry.yaml")
    parser.add_argument("--dataset", default="voxceleb1")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    emb_dir = os.path.join(cfg.results_dir, "embeddings")
    out_dir = os.path.join(cfg.results_dir, "cluster_topology")
    os.makedirs(out_dir, exist_ok=True)

    all_encoders = list(cfg.encoders.face) + list(cfg.encoders.voice)

    results = {}

    for enc_name in all_encoders:
        emb_path = os.path.join(emb_dir, f"{enc_name}_{args.dataset}.npz")
        if not os.path.exists(emb_path):
            logger.warning(f"Missing {emb_path}, skipping")
            continue

        logger.info(f"Computing cluster topology for {enc_name}...")
        data = np.load(emb_path)
        X = data["embeddings"]
        labels = data["labels"]

        # L2-normalize embeddings
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        X_norm = X / (norms + 1e-8)

        try:
            metrics = compute_cluster_metrics(
                X_norm, labels,
                min_samples_per_id=cfg.geometry.cluster_topology.min_samples_per_id,
            )
            # Remove per-class details for JSON summary
            per_class = metrics.pop("per_class_compactness")
            metrics["encoder"] = enc_name
            metrics["n_classes_with_enough_samples"] = len(per_class)
            results[enc_name] = metrics

            logger.info(f"  {enc_name}: intra={metrics['intra_compactness']:.4f}, "
                        f"inter={metrics['inter_separation']:.4f}, "
                        f"gap={metrics['compactness_gap']:.4f}")

        except Exception as e:
            logger.error(f"  {enc_name} failed: {e}")
            results[enc_name] = {"encoder": enc_name, "error": str(e)}

    out_path = os.path.join(out_dir, f"cluster_topology_{args.dataset}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved {out_path}")


if __name__ == "__main__":
    main()
