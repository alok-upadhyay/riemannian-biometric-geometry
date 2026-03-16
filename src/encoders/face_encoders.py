"""Face encoder wrappers: ArcFace, SigLIP, DINOv2, CLIP."""

import torch
import torch.nn as nn
import numpy as np


class ArcFaceEncoder(nn.Module):
    """ArcFace (insightface buffalo_l) face encoder. Outputs 512-d embeddings."""

    def __init__(self, device="cuda"):
        super().__init__()
        self._device = device
        self._output_dim = 512
        self._app = None
        self._rec_model = None

    def _lazy_init(self):
        if self._app is not None:
            return
        from insightface.app import FaceAnalysis

        providers = (
            ["CUDAExecutionProvider"] if self._device == "cuda" else ["CPUExecutionProvider"]
        )
        self._app = FaceAnalysis(name="buffalo_l", providers=providers)
        self._app.prepare(
            ctx_id=0 if self._device == "cuda" else -1, det_size=(640, 640)
        )
        self._rec_model = self._app.models["recognition"]

    @torch.no_grad()
    def encode_images(self, images_bgr: list[np.ndarray]) -> np.ndarray:
        """Encode a list of BGR numpy images to embeddings.

        Args:
            images_bgr: list of (H, W, 3) uint8 BGR arrays

        Returns:
            (N, 512) float32 array of L2-normalized embeddings
        """
        self._lazy_init()
        embeddings = []
        for img_bgr in images_bgr:
            faces = self._app.get(img_bgr)
            if len(faces) > 0:
                face = max(
                    faces,
                    key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                )
                embeddings.append(face.normed_embedding)
            else:
                # Fallback: center-crop + resize without alignment
                h, w = img_bgr.shape[:2]
                s = min(h, w)
                y0, x0 = (h - s) // 2, (w - s) // 2
                crop = img_bgr[y0 : y0 + s, x0 : x0 + s]
                import cv2

                img_112 = cv2.resize(crop, (112, 112))
                face_input = (img_112.astype(np.float32) - 127.5) / 127.5
                face_input = face_input.transpose(2, 0, 1)[np.newaxis, ...]
                emb = self._rec_model.forward(face_input).flatten()
                emb = emb / (np.linalg.norm(emb) + 1e-8)
                embeddings.append(emb)
        return np.stack(embeddings, axis=0)

    @property
    def output_dim(self) -> int:
        return self._output_dim


class SigLIPEncoder(nn.Module):
    """Frozen SigLIP Vision encoder. Outputs 768-d embeddings."""

    def __init__(self, model_name: str = "google/siglip-base-patch16-384"):
        super().__init__()
        from transformers import SiglipVisionModel

        self.model = SiglipVisionModel.from_pretrained(model_name)
        for param in self.model.parameters():
            param.requires_grad = False
        self._output_dim = self.model.config.hidden_size

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.model(pixel_values=pixel_values)
        return outputs.pooler_output

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @property
    def image_size(self) -> int:
        return 384


class DINOv2Encoder(nn.Module):
    """Frozen DINOv2 ViT-B/14 encoder. Outputs 768-d embeddings."""

    def __init__(self, model_name: str = "dinov2_vitb14"):
        super().__init__()
        self.model = torch.hub.load("facebookresearch/dinov2", model_name)
        for param in self.model.parameters():
            param.requires_grad = False
        self._output_dim = 768

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        features = self.model(pixel_values)
        if features.dim() == 3:
            return features[:, 0, :]
        return features

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @property
    def image_size(self) -> int:
        return 224


class CLIPEncoder(nn.Module):
    """Frozen CLIP ViT-B/16 visual encoder via open_clip. Outputs 512-d embeddings."""

    def __init__(self, model_name: str = "ViT-B-16", pretrained: str = "openai"):
        super().__init__()
        import open_clip

        model, _, self._preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = model.visual
        for param in self.model.parameters():
            param.requires_grad = False
        # CLIP ViT-B/16 output dim
        self._output_dim = 512

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values)

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @property
    def image_size(self) -> int:
        return 224

    @property
    def preprocess(self):
        return self._preprocess
