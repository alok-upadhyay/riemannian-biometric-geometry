"""Step 5: Compute cross-modal geometry metrics for all face-voice pairs.

Usage:
    PYTHONPATH=. python scripts/05_cross_modal_geometry.py --config configs/geometry.yaml
"""

import argparse
import json
import logging
import os

import numpy as np
from omegaconf import OmegaConf

from src.geometry.cross_modal import compute_cross_modal_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/geometry.yaml")
    parser.add_argument("--dataset", default="voxceleb1")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    emb_dir = os.path.join(cfg.results_dir, "embeddings")
    id_dir = os.path.join(cfg.results_dir, "intrinsic_dim")
    out_dir = os.path.join(cfg.results_dir, "cross_modal")
    os.makedirs(out_dir, exist_ok=True)

    # Load intrinsic dim results for ID mismatch
    id_path = os.path.join(id_dir, f"intrinsic_dim_{args.dataset}.json")
    id_results = {}
    if os.path.exists(id_path):
        with open(id_path) as f:
            id_results = json.load(f)

    face_encoders = list(cfg.encoders.face)
    voice_encoders = list(cfg.encoders.voice)

    all_results = {}

    for face_enc in face_encoders:
        face_path = os.path.join(emb_dir, f"{face_enc}_{args.dataset}.npz")
        if not os.path.exists(face_path):
            logger.warning(f"Missing {face_path}")
            continue
        face_data = np.load(face_path)

        for voice_enc in voice_encoders:
            voice_path = os.path.join(emb_dir, f"{voice_enc}_{args.dataset}.npz")
            if not os.path.exists(voice_path):
                logger.warning(f"Missing {voice_path}")
                continue
            voice_data = np.load(voice_path)

            pair_name = f"{face_enc}__{voice_enc}"
            logger.info(f"Computing cross-modal metrics: {pair_name}")

            # Use identity-averaged centroids
            face_centroids = face_data["centroids"]
            voice_centroids = voice_data["centroids"]
            face_labels = list(face_data["centroid_labels"])
            voice_labels = list(voice_data["centroid_labels"])

            # Align to common identities
            common_ids = sorted(set(face_labels) & set(voice_labels))
            if len(common_ids) < 10:
                logger.warning(f"  Only {len(common_ids)} common identities, skipping")
                continue

            face_idx = [face_labels.index(l) for l in common_ids]
            voice_idx = [voice_labels.index(l) for l in common_ids]
            X = face_centroids[face_idx]
            Y = voice_centroids[voice_idx]

            # Get intrinsic dims
            id_face = id_results.get(face_enc, {}).get("global_mle")
            id_voice = id_results.get(voice_enc, {}).get("global_mle")

            try:
                metrics = compute_cross_modal_metrics(
                    X, Y,
                    id_x=id_face,
                    id_y=id_voice,
                    gh_max_points=cfg.geometry.cross_modal.gh_max_points,
                    spectral_k=cfg.geometry.cross_modal.spectral_k_neighbors,
                    spectral_n_eigenvalues=cfg.geometry.cross_modal.spectral_n_eigenvalues,
                    seed=cfg.evaluation.seed,
                )
                metrics["face_encoder"] = face_enc
                metrics["voice_encoder"] = voice_enc
                metrics["n_common_ids"] = len(common_ids)
                all_results[pair_name] = metrics

                logger.info(f"  GW={metrics.get('gw_distance', 'N/A'):.4f}, "
                            f"CKA={metrics.get('cka', 'N/A'):.4f}, "
                            f"spectral_gap={metrics.get('spectral_gap', 'N/A'):.4f}")

            except Exception as e:
                logger.error(f"  {pair_name} failed: {e}")
                all_results[pair_name] = {"error": str(e)}

    out_path = os.path.join(out_dir, f"cross_modal_{args.dataset}.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Saved {out_path}")


if __name__ == "__main__":
    main()
