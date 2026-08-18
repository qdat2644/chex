from __future__ import annotations

import base64
import pickle
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
    threshold: float | None = None


@dataclass(frozen=True)
class Heatmap:
    label: str
    probability: float
    image_data_url: str


def build_model(arch: str, num_labels: int, pretrained: bool = False) -> torch.nn.Module:
    arch = arch.lower().replace("-", "_")
    if arch == "densenet121":
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.densenet121(weights=weights)
        in_features = model.classifier.in_features
        model.classifier = torch.nn.Linear(in_features, num_labels)
    elif arch in ("convnext_small", "convnext"):
        weights = models.ConvNeXt_Small_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.convnext_small(weights=weights)
        in_features = model.classifier[2].in_features
        model.classifier[2] = torch.nn.Linear(in_features, num_labels)
    elif arch in ("efficientnet_v2_m", "efficientnet"):
        weights = models.EfficientNet_V2_M_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_v2_m(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = torch.nn.Linear(in_features, num_labels)
    elif arch in ("resnet50", "resnet"):
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, num_labels)
    else:
        # Default fallback to DenseNet121
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.densenet121(weights=weights)
        in_features = model.classifier.in_features
        model.classifier = torch.nn.Linear(in_features, num_labels)
    return model


def get_target_layer_for_gradcam(model: torch.nn.Module, arch: str) -> torch.nn.Module:
    arch = arch.lower().replace("-", "_")
    if "convnext" in arch:
        return model.features[-1]
    if "efficientnet" in arch:
        return model.features[-1]
    if "resnet" in arch:
        return model.layer4[-1]
    if "densenet" in arch:
        return getattr(model.features, "denseblock4", model.features)
    if hasattr(model, "features"):
        return model.features
    return list(model.children())[-2]


class CheXpertPredictor:
    def __init__(
        self,
        checkpoint_path: str | Path | None,
        labels: list[str] | None = None,
        threshold: float = DEFAULT_THRESHOLD,
        thresholds: dict[str, float] | None = None,
        image_size: int = DEFAULT_IMAGE_SIZE,
    ) -> None:
        self.labels = labels or DEFAULT_LABELS
        self.threshold = threshold
        self.thresholds = thresholds or {}
        self.image_size = image_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: torch.nn.Module | None = None
        self.metadata: dict[str, object] = {}
        self.architecture = "DenseNet121"
        self._update_transforms(image_size)

        if checkpoint_path:
            self.load(checkpoint_path)

    def _update_transforms(self, image_size: int) -> None:
        self.image_size = image_size
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

    def set_thresholds(self, thresholds: dict[str, float]) -> None:
        self.thresholds = {label: float(value) for label, value in thresholds.items()}

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
            self.architecture = str(metadata.get("architecture", "DenseNet121"))
            ckpt_size = metadata.get("image_size")
            if ckpt_size and isinstance(ckpt_size, int):
                self._update_transforms(ckpt_size)

        model = build_model(self.architecture, len(self.labels), pretrained=False)
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

        results: list[Prediction] = []
        for label, probability in zip(self.labels, probabilities, strict=True):
            threshold = self.thresholds.get(label)
            is_positive = bool(threshold is not None and float(probability) >= threshold)
            results.append(
                Prediction(
                    label=label,
                    probability=float(probability),
                    positive=is_positive,
                    threshold=threshold,
                )
            )
        return results

    def explain_top_finding(self, image: Image.Image) -> Heatmap:
        if self.model is None:
            raise RuntimeError("Model checkpoint is not loaded.")

        original = image.convert("RGB")
        tensor = self.transform(original).unsqueeze(0).to(self.device)

        # Hook-based Grad-CAM works across any architecture
        target_layer = get_target_layer_for_gradcam(self.model, self.architecture)
        activations_list = []
        gradients_list = []

        def forward_hook(module, input, output):
            activations_list.append(output)

        def backward_hook(module, grad_in, grad_out):
            gradients_list.append(grad_out[0])

        h_fwd = target_layer.register_forward_hook(forward_hook)
        h_bwd = target_layer.register_full_backward_hook(backward_hook)

        self.model.zero_grad(set_to_none=True)
        logits = self.model(tensor)
        probabilities = torch.sigmoid(logits).squeeze(0)
        label_index = int(torch.argmax(probabilities).detach().cpu())

        logits[0, label_index].backward()

        h_fwd.remove()
        h_bwd.remove()

        if not activations_list or not gradients_list:
            raise RuntimeError("Failed to capture Grad-CAM activations/gradients.")

        activations = activations_list[0].detach()[0]
        gradients = gradients_list[0].detach()[0]
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
