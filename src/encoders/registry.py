"""Encoder registry: instantiate encoders by name."""

import torch

FACE_ENCODERS = ["arcface", "siglip", "dinov2", "clip"]
VOICE_ENCODERS = ["wavlm", "ecapa_tdnn", "hubert", "wav2vec2"]


def get_encoder(name: str, device: str = "cuda"):
    """Instantiate an encoder by name and move to device.

    Returns:
        (encoder, image_size_or_None)
        For face encoders, image_size is set appropriately.
        For voice encoders, image_size is None.
    """
    name = name.lower()

    if name == "arcface":
        from src.encoders.face_encoders import ArcFaceEncoder
        enc = ArcFaceEncoder(device=device)
        return enc, 112

    if name == "siglip":
        from src.encoders.face_encoders import SigLIPEncoder
        enc = SigLIPEncoder().to(device)
        return enc, 384

    if name == "dinov2":
        from src.encoders.face_encoders import DINOv2Encoder
        enc = DINOv2Encoder().to(device)
        return enc, 224

    if name == "clip":
        from src.encoders.face_encoders import CLIPEncoder
        enc = CLIPEncoder().to(device)
        return enc, 224

    if name == "wavlm":
        from src.encoders.voice_encoders import WavLMEncoder
        enc = WavLMEncoder().to(device)
        return enc, None

    if name == "ecapa_tdnn":
        from src.encoders.voice_encoders import ECAPATDNNEncoder
        enc = ECAPATDNNEncoder()
        # ECAPA-TDNN handles device internally via SpeechBrain
        return enc, None

    if name == "hubert":
        from src.encoders.voice_encoders import HuBERTEncoder
        enc = HuBERTEncoder().to(device)
        return enc, None

    if name == "wav2vec2":
        from src.encoders.voice_encoders import Wav2Vec2Encoder
        enc = Wav2Vec2Encoder().to(device)
        return enc, None

    raise ValueError(f"Unknown encoder: {name}. Available: {FACE_ENCODERS + VOICE_ENCODERS}")
