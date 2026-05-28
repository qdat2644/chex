from __future__ import annotations

import base64
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

from app.config import DEFAULT_IMAGE_SIZE, DEFAULT_LABELS, DEFAULT_THRESHOLD


@dataclass(frozen=True)
class Prediction:
    label: str
    probability: float
    positive: bool


@dataclass(frozen=True)
class Heatmap:
    label: str
    probability: float
    image_data_url: str


class CheXpertPredictor:
    def __init__(
        self,
        checkpoint_path: str | Path | None,
        labels: list[str] | None = None,
        threshold: float = DEFAULT_THRESHOLD,
        image_size: int = DEFAULT_IMAGE_SIZE,
    ) -> None:
        self.labels = labels or DEFAULT_LABELS
        self.threshold = threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        self.model: torch.nn.Module | None = None
        self.metadata: dict[str, object] = {}

        if checkpoint_path:
            self.load(checkpoint_path)

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def load(self, checkpoint_path: str | Path) -> None:
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        except (TypeError, pickle.UnpicklingError):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
        labels = checkpoint.get("labels") if isinstance(checkpoint, dict) else None
        if labels:
            self.labels = list(labels)
        metadata = checkpoint.get("metadata") if isinstance(checkpoint, dict) else None
        if isinstance(metadata, dict):
            self.metadata = metadata

        model = models.densenet121(weights=None)
        in_features = model.classifier.in_features
        model.classifier = torch.nn.Linear(in_features, len(self.labels))

        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        self.model = model

    @torch.inference_mode()
    def predict(self, image: Image.Image) -> list[Prediction]:
        if self.model is None:
            raise RuntimeError("Model checkpoint is not loaded.")

        tensor = self.transform(image).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        probabilities = torch.sigmoid(logits).squeeze(0).detach().cpu().tolist()

        return [
            Prediction(
                label=label,
                probability=float(probability),
                positive=float(probability) >= self.threshold,
            )
            for label, probability in zip(self.labels, probabilities, strict=True)
        ]

    def explain_top_finding(self, image: Image.Image) -> Heatmap:
        if self.model is None:
            raise RuntimeError("Model checkpoint is not loaded.")
        if not (hasattr(self.model, "features") and hasattr(self.model, "classifier")):
            raise RuntimeError("Grad-CAM is only implemented for DenseNet checkpoints.")

        original = image.convert("RGB")
        tensor = self.transform(original).unsqueeze(0).to(self.device)
        self.model.zero_grad(set_to_none=True)

        features = self.model.features(tensor)
        features.retain_grad()
        pooled = F.adaptive_avg_pool2d(F.relu(features, inplace=False), (1, 1))
        logits = self.model.classifier(torch.flatten(pooled, 1))
        probabilities = torch.sigmoid(logits).squeeze(0)
        label_index = int(torch.argmax(probabilities).detach().cpu())
        logits[0, label_index].backward()

        gradients = features.grad.detach()[0]
        activations = features.detach()[0]
        weights = gradients.mean(dim=(1, 2))
        cam = torch.relu((weights[:, None, None] * activations).sum(dim=0))
        cam = self._normalize_cam(cam)
        overlay = self._render_overlay(original, cam)

        return Heatmap(
            label=self.labels[label_index],
            probability=float(probabilities[label_index].detach().cpu()),
            image_data_url=self._encode_png_data_url(overlay),
        )

    @staticmethod
    def _normalize_cam(cam: torch.Tensor) -> np.ndarray:
        cam = cam.detach().cpu()
        cam_min = float(cam.min())
        cam_max = float(cam.max())
        if cam_max <= cam_min:
            return np.zeros(tuple(cam.shape), dtype=np.float32)
        return ((cam - cam_min) / (cam_max - cam_min)).numpy().astype(np.float32)

    @staticmethod
    def _render_overlay(image: Image.Image, cam: np.ndarray) -> Image.Image:
        heatmap = Image.fromarray(np.uint8(cam * 255), mode="L").resize(
            image.size,
            resample=Image.Resampling.BILINEAR,
        )
        heat = np.asarray(heatmap, dtype=np.float32) / 255.0
        base = np.asarray(image.convert("RGB"), dtype=np.float32)

        color = np.zeros_like(base)
        color[..., 0] = 255.0
        color[..., 1] = 190.0 * heat
        alpha = (0.42 * heat)[..., None]
        overlay = np.clip((base * (1.0 - alpha)) + (color * alpha), 0, 255).astype(np.uint8)
        return Image.fromarray(overlay, mode="RGB")

    @staticmethod
    def _encode_png_data_url(image: Image.Image) -> str:
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"
import pickle
