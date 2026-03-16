"""Step 6: Compute cross-modal EER with CCA alignment for all face-voice pairs.

Usage:
    PYTHONPATH=. python scripts/06_cross_modal_eer.py --config configs/geometry.yaml
"""

import argparse
import json
import logging
import os

import numpy as np
from omegaconf import OmegaConf

from src.evaluation.eer import compute_cross_modal_eer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/geometry.yaml")
    parser.add_argument("--dataset", default="voxceleb1")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    emb_dir = os.path.join(cfg.results_dir, "embeddings")
    out_dir = os.path.join(cfg.results_dir, "cross_modal")
    os.makedirs(out_dir, exist_ok=True)

    face_encoders = list(cfg.encoders.face)
    voice_encoders = list(cfg.encoders.voice)

    all_results = {}

    for face_enc in face_encoders:
        face_path = os.path.join(emb_dir, f"{face_enc}_{args.dataset}.npz")
        if not os.path.exists(face_path):
            continue
        face_data = np.load(face_path)

        for voice_enc in voice_encoders:
            voice_path = os.path.join(emb_dir, f"{voice_enc}_{args.dataset}.npz")
            if not os.path.exists(voice_path):
                continue
            voice_data = np.load(voice_path)

            pair_name = f"{face_enc}__{voice_enc}"
            logger.info(f"Computing EER: {pair_name}")

            # Use raw embeddings (not centroids) for EER
            face_emb = face_data["embeddings"]
            voice_emb = voice_data["embeddings"]
            face_labels = face_data["labels"]
            voice_labels = voice_data["labels"]

            # Align samples to common identities
            common_ids = set(face_labels) & set(voice_labels)
            if len(common_ids) < 10:
                logger.warning(f"  Only {len(common_ids)} common IDs, skipping")
                continue

            face_mask = np.isin(face_labels, list(common_ids))
            voice_mask = np.isin(voice_labels, list(common_ids))

            try:
                eer_result = compute_cross_modal_eer(
                    face_emb[face_mask],
                    voice_emb[voice_mask],
                    face_labels[face_mask],  # labels for face (used for aggregation)
                    train_fraction=cfg.evaluation.cca_train_fraction,
                    n_cca_components=cfg.evaluation.n_cca_components,
                    seed=cfg.evaluation.seed,
                )
                eer_result["face_encoder"] = face_enc
                eer_result["voice_encoder"] = voice_enc
                all_results[pair_name] = eer_result

                logger.info(f"  EER={eer_result['eer']:.4f}")

            except Exception as e:
                logger.error(f"  {pair_name} failed: {e}")
                all_results[pair_name] = {"error": str(e)}

    out_path = os.path.join(out_dir, f"eer_{args.dataset}.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Saved {out_path}")


if __name__ == "__main__":
    main()
