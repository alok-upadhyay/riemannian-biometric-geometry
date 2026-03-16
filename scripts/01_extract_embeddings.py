"""Step 1: Extract and cache embeddings from all encoders.

For each encoder, extracts embeddings on VoxCeleb1 (and optionally MAV-Celeb),
saves per-encoder .npz files with raw embeddings, identity labels, and centroids.

Usage:
    PYTHONPATH=. python scripts/01_extract_embeddings.py --config configs/geometry.yaml
    PYTHONPATH=. python scripts/01_extract_embeddings.py --config configs/geometry.yaml --encoder wavlm
    PYTHONPATH=. python scripts/01_extract_embeddings.py --config configs/geometry.yaml --dataset mavceleb
"""

import argparse
import logging
import os
import glob
import random
from collections import defaultdict

import numpy as np
import torch
import torchaudio
from PIL import Image
from omegaconf import OmegaConf
from tqdm import tqdm

from src.encoders.registry import get_encoder, FACE_ENCODERS, VOICE_ENCODERS
from src.data.transforms import get_audio_transform, get_image_transform

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def discover_voxceleb1(data_root, max_per_id=20, seed=42):
    """Discover VoxCeleb1 identities and subsample."""
    wav_root = os.path.join(data_root, "wav")
    face_root = os.path.join(data_root, "faces")
    rng = random.Random(seed)

    samples = []
    for identity in sorted(os.listdir(wav_root)):
        id_wav = os.path.join(wav_root, identity)
        id_face = os.path.join(face_root, identity)
        if not os.path.isdir(id_wav) or not os.path.isdir(id_face):
            continue
        audios = glob.glob(os.path.join(id_wav, "**", "*.wav"), recursive=True)
        faces = glob.glob(os.path.join(id_face, "**", "*.jpg"), recursive=True)
        if not audios or not faces:
            continue
        rng.shuffle(audios)
        rng.shuffle(faces)
        n = min(max_per_id, len(audios), len(faces))
        for i in range(n):
            samples.append((audios[i], faces[i], identity))

    return samples


def discover_mavceleb(data_root, max_per_id=20, seed=42):
    """Discover MAV-Celeb identities and subsample."""
    voice_root = os.path.join(data_root, "v1", "voices")
    face_root = os.path.join(data_root, "v1", "faces")
    rng = random.Random(seed)

    samples = []
    for identity in sorted(os.listdir(voice_root)):
        id_voice = os.path.join(voice_root, identity)
        id_face = os.path.join(face_root, identity)
        if not os.path.isdir(id_voice) or not os.path.isdir(id_face):
            continue
        audios = glob.glob(os.path.join(id_voice, "**", "*.wav"), recursive=True)
        faces = glob.glob(os.path.join(id_face, "**", "*.jpg"), recursive=True)
        if not audios or not faces:
            continue
        rng.shuffle(audios)
        rng.shuffle(faces)
        n = min(max_per_id, len(audios), len(faces))
        for i in range(n):
            samples.append((audios[i], faces[i], identity))

    return samples


def extract_voice_embeddings(encoder, samples, device, cfg):
    """Extract voice embeddings from audio files."""
    audio_transform = get_audio_transform(
        sample_rate=cfg.data.audio_sample_rate,
        max_length_sec=cfg.data.audio_max_length_sec,
    )

    all_embeddings = []
    all_labels = []
    batch_size = cfg.data.batch_size

    for i in tqdm(range(0, len(samples), batch_size), desc="Voice embeddings"):
        batch_samples = samples[i : i + batch_size]
        waveforms = []
        masks = []
        labels = []

        for audio_path, _, identity in batch_samples:
            try:
                waveform, sr = torchaudio.load(audio_path)
                waveform, mask = audio_transform(waveform, sr)
                waveforms.append(waveform)
                masks.append(mask)
                labels.append(identity)
            except Exception as e:
                logger.warning(f"Failed to load {audio_path}: {e}")
                continue

        if not waveforms:
            continue

        waveforms = torch.stack(waveforms).to(device)
        masks = torch.stack(masks).to(device)

        emb = encoder(waveforms, masks)
        all_embeddings.append(emb.cpu().numpy())
        all_labels.extend(labels)

    return np.concatenate(all_embeddings, axis=0), all_labels


def extract_face_embeddings(encoder, encoder_name, samples, device, cfg, image_size):
    """Extract face embeddings from image files."""
    if encoder_name == "arcface":
        return extract_arcface_embeddings(encoder, samples)

    if encoder_name == "clip":
        preprocess = encoder.preprocess
    else:
        preprocess = get_image_transform(image_size=image_size, is_train=False)

    all_embeddings = []
    all_labels = []
    batch_size = cfg.data.batch_size

    for i in tqdm(range(0, len(samples), batch_size), desc="Face embeddings"):
        batch_samples = samples[i : i + batch_size]
        pixels = []
        labels = []

        for _, face_path, identity in batch_samples:
            try:
                img = Image.open(face_path).convert("RGB")
                pv = preprocess(img)
                pixels.append(pv)
                labels.append(identity)
            except Exception as e:
                logger.warning(f"Failed to load {face_path}: {e}")
                continue

        if not pixels:
            continue

        pixels = torch.stack(pixels).to(device)
        emb = encoder(pixels)
        all_embeddings.append(emb.cpu().numpy())
        all_labels.extend(labels)

    return np.concatenate(all_embeddings, axis=0), all_labels


def extract_arcface_embeddings(encoder, samples):
    """Extract ArcFace embeddings."""
    import cv2

    images = []
    labels = []
    for _, face_path, identity in tqdm(samples, desc="Loading images for ArcFace"):
        img = cv2.imread(face_path)
        if img is not None:
            images.append(img)
            labels.append(identity)

    batch_size = 64
    all_embeddings = []
    for i in tqdm(range(0, len(images), batch_size), desc="ArcFace embeddings"):
        batch = images[i : i + batch_size]
        emb = encoder.encode_images(batch)
        all_embeddings.append(emb)

    return np.concatenate(all_embeddings, axis=0), labels


def compute_centroids(embeddings, labels):
    """Compute identity-averaged centroids."""
    unique_labels = sorted(set(labels))
    D = embeddings.shape[1]
    centroids = np.zeros((len(unique_labels), D))
    counts = np.zeros(len(unique_labels))

    label_to_idx = {l: i for i, l in enumerate(unique_labels)}
    for i, label in enumerate(labels):
        idx = label_to_idx[label]
        centroids[idx] += embeddings[i]
        counts[idx] += 1

    centroids = centroids / counts[:, np.newaxis]
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroids = centroids / (norms + 1e-8)

    return centroids, unique_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/geometry.yaml")
    parser.add_argument("--encoder", default=None, help="Extract only this encoder")
    parser.add_argument("--dataset", default="voxceleb1", choices=["voxceleb1", "mavceleb", "both"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    device = args.device
    results_dir = os.path.join(cfg.results_dir, "embeddings")
    os.makedirs(results_dir, exist_ok=True)

    # Determine which encoders to run
    if args.encoder:
        encoders_to_run = [args.encoder]
    else:
        encoders_to_run = cfg.encoders.face + cfg.encoders.voice

    # Determine datasets
    datasets = []
    if args.dataset in ("voxceleb1", "both"):
        datasets.append(("voxceleb1", cfg.data.voxceleb1_root))
    if args.dataset in ("mavceleb", "both"):
        datasets.append(("mavceleb", cfg.data.mavceleb_root))

    for dataset_name, data_root in datasets:
        logger.info(f"\n{'='*60}")
        logger.info(f"Dataset: {dataset_name}")
        logger.info(f"{'='*60}")

        if dataset_name == "voxceleb1":
            samples = discover_voxceleb1(data_root, max_per_id=cfg.data.max_samples_per_id)
        else:
            samples = discover_mavceleb(data_root, max_per_id=cfg.data.max_samples_per_id)

        logger.info(f"Found {len(samples)} samples")

        for enc_name in encoders_to_run:
            out_path = os.path.join(results_dir, f"{enc_name}_{dataset_name}.npz")
            if os.path.exists(out_path):
                logger.info(f"Skipping {enc_name} (already exists: {out_path})")
                continue

            logger.info(f"\nExtracting {enc_name}...")
            try:
                encoder, image_size = get_encoder(enc_name, device=device)
            except Exception as e:
                logger.error(f"Failed to load {enc_name}: {e}")
                continue

            is_face = enc_name in FACE_ENCODERS
            try:
                if is_face:
                    embeddings, labels = extract_face_embeddings(
                        encoder, enc_name, samples, device, cfg, image_size
                    )
                else:
                    embeddings, labels = extract_voice_embeddings(
                        encoder, samples, device, cfg
                    )
            except Exception as e:
                logger.error(f"Failed to extract {enc_name}: {e}")
                continue

            centroids, unique_labels = compute_centroids(embeddings, labels)

            np.savez(
                out_path,
                embeddings=embeddings,
                labels=np.array(labels),
                centroids=centroids,
                centroid_labels=np.array(unique_labels),
                encoder=enc_name,
                dataset=dataset_name,
            )
            logger.info(f"Saved {out_path}: {embeddings.shape}, {len(unique_labels)} identities")

            # Free memory
            del encoder, embeddings
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
