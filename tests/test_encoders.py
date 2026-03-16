"""Tests for encoder registry and wrappers.

These tests verify the registry dispatching and basic encoder interface.
They use mocking to avoid downloading large models.
"""

import pytest
import numpy as np

from src.encoders.registry import get_encoder, FACE_ENCODERS, VOICE_ENCODERS


class TestRegistry:
    def test_face_encoders_list(self):
        assert "arcface" in FACE_ENCODERS
        assert "siglip" in FACE_ENCODERS
        assert "dinov2" in FACE_ENCODERS
        assert "clip" in FACE_ENCODERS

    def test_voice_encoders_list(self):
        assert "wavlm" in VOICE_ENCODERS
        assert "ecapa_tdnn" in VOICE_ENCODERS
        assert "hubert" in VOICE_ENCODERS
        assert "wav2vec2" in VOICE_ENCODERS

    def test_unknown_encoder_raises(self):
        with pytest.raises(ValueError, match="Unknown encoder"):
            get_encoder("nonexistent_encoder")

    def test_case_insensitive(self):
        """Should handle case-insensitive names."""
        # This tests the lowering logic. We can't actually instantiate
        # without downloading models, so just check it doesn't raise ValueError
        # for valid names (it may raise import/download errors instead).
        for name in ["ARCFACE", "Arcface", "arcFace"]:
            try:
                get_encoder(name, device="cpu")
            except ValueError:
                pytest.fail(f"get_encoder should accept {name}")
            except Exception:
                pass  # Import/download errors are OK in test


class TestEncoderInterfaces:
    """Test that encoder classes have the right interface.

    These import the classes but don't instantiate them (avoids model downloads).
    """

    def test_face_encoder_classes_exist(self):
        from src.encoders.face_encoders import (
            ArcFaceEncoder,
            SigLIPEncoder,
            DINOv2Encoder,
            CLIPEncoder,
        )

    def test_voice_encoder_classes_exist(self):
        from src.encoders.voice_encoders import (
            WavLMEncoder,
            ECAPATDNNEncoder,
            HuBERTEncoder,
            Wav2Vec2Encoder,
        )

    def test_arcface_has_encode_images(self):
        from src.encoders.face_encoders import ArcFaceEncoder
        assert hasattr(ArcFaceEncoder, "encode_images")
        assert hasattr(ArcFaceEncoder, "output_dim")
