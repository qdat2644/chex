from __future__ import annotations

import base64
import os
import pickle
import threading
import urllib.request
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

from app.config import DEFAULT_IMAGE_SIZE, DEFAULT_LABELS, DEFAULT_THRESHOLD, PROJECT_ROOT

MODEL_LOCK = threading.RLock()


@dataclass(frozen=True)
class Prediction:
    label: str
    probability: float
    positive: bool
    threshold: float | None = None
    suspicion_level: str = "Low suspicion"


@dataclass(frozen=True)
class Heatmap:
    label: str
    probability: float
    image_data_url: str
    pure_heatmap_url: str | None = None


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
        self.architecture = "ConvNeXt-Small"
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
        path = Path(checkpoint_path)
        if not path.exists():
            return

        if path.suffix.lower() == ".safetensors":
            try:
                from safetensors.torch import load_file
                state_dict = load_file(path, device=str(self.device))
                checkpoint = {"model_state_dict": state_dict}
            except ImportError:
                raise RuntimeError("safetensors package is required to load .safetensors checkpoints.")
        else:
            try:
                checkpoint = torch.load(path, map_location=self.device, weights_only=True)
            except Exception:
                checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        labels = checkpoint.get("labels") if isinstance(checkpoint, dict) else None
        if labels:
            self.labels = list(labels)
        metadata = checkpoint.get("metadata") if isinstance(checkpoint, dict) else None
        if isinstance(metadata, dict):
            self.metadata = metadata
            self.architecture = str(metadata.get("architecture", "ConvNeXt-Small"))
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

        with MODEL_LOCK:
            tensor = self.transform(image).unsqueeze(0).to(self.device)
            logits = self.model(tensor)
            probabilities = torch.sigmoid(logits).squeeze(0).detach().cpu().tolist()

            results: list[Prediction] = []
            for label, probability in zip(self.labels, probabilities, strict=True):
                prob = float(probability)
                threshold = self.thresholds.get(label, self.threshold)
                is_positive = bool(threshold is not None and prob >= threshold)

                # Determine clinical suspicion level
                if threshold is not None:
                    if prob >= min(1.0, threshold + 0.15):
                        suspicion = "High suspicion"
                    elif prob >= threshold:
                        suspicion = "Moderate suspicion"
                    else:
                        suspicion = "Low suspicion"
                else:
                    suspicion = "Probability only"

                results.append(
                    Prediction(
                        label=label,
                        probability=prob,
                        positive=is_positive,
                        threshold=threshold,
                        suspicion_level=suspicion,
                    )
                )
            return results

    @torch.inference_mode()
    def predict_batch(self, images: list[Image.Image], chunk_size: int = 16) -> list[list[Prediction]]:
        if self.model is None:
            raise RuntimeError("Model checkpoint is not loaded.")
        if not images:
            return []

        all_results: list[list[Prediction]] = []
        with MODEL_LOCK:
            # Chunked batch processing to prevent GPU/CPU memory overflow (OOM)
            for i in range(0, len(images), chunk_size):
                chunk = images[i : i + chunk_size]
                tensors = torch.stack([self.transform(img.convert("RGB")) for img in chunk]).to(self.device)
                logits = self.model(tensors)
                batch_probs = torch.sigmoid(logits).detach().cpu().numpy()

                for probs in batch_probs:
                    img_results: list[Prediction] = []
                    for label, prob_val in zip(self.labels, probs, strict=True):
                        prob = float(prob_val)
                        threshold = self.thresholds.get(label, self.threshold)
                        is_positive = bool(threshold is not None and prob >= threshold)

                        if threshold is not None:
                            if prob >= min(1.0, threshold + 0.15):
                                suspicion = "High suspicion"
                            elif prob >= threshold:
                                suspicion = "Moderate suspicion"
                            else:
                                suspicion = "Low suspicion"
                        else:
                            suspicion = "Probability only"

                        img_results.append(
                            Prediction(
                                label=label,
                                probability=prob,
                                positive=is_positive,
                                threshold=threshold,
                                suspicion_level=suspicion,
                            )
                        )
                    all_results.append(img_results)
            return all_results

    def explain_finding(self, image: Image.Image, target_label: str | None = None) -> Heatmap:
        if self.model is None:
            raise RuntimeError("Model checkpoint is not loaded.")

        if target_label is not None and target_label not in self.labels:
            raise ValueError(f"Invalid target label '{target_label}'. Valid labels are: {self.labels}")

        with MODEL_LOCK:
            original = image.convert("RGB")
            tensor = self.transform(original).unsqueeze(0).to(self.device)

            target_layer = get_target_layer_for_gradcam(self.model, self.architecture)
            activations_list = []
            gradients_list = []

            def forward_hook(module, inp, output):
                activations_list.append(output)

            def backward_hook(module, grad_in, grad_out):
                gradients_list.append(grad_out[0])

            h_fwd = target_layer.register_forward_hook(forward_hook)
            h_bwd = target_layer.register_full_backward_hook(backward_hook)

            try:
                self.model.zero_grad(set_to_none=True)
                logits = self.model(tensor)
                probabilities = torch.sigmoid(logits).squeeze(0)

                # Select target label index
                if target_label and target_label in self.labels:
                    label_index = self.labels.index(target_label)
                else:
                    label_index = int(torch.argmax(probabilities).detach().cpu())

                label_name = self.labels[label_index]
                label_prob = float(probabilities[label_index].detach().cpu())

                logits[0, label_index].backward()

                if not activations_list or not gradients_list:
                    raise RuntimeError("Failed to capture Grad-CAM activations/gradients.")

                activations = activations_list[0].detach()[0]
                gradients = gradients_list[0].detach()[0]
                weights = gradients.mean(dim=(1, 2))
                cam = torch.relu((weights[:, None, None] * activations).sum(dim=0))
                cam_norm = self._normalize_cam(cam, confidence=label_prob)

                overlay = self._render_overlay(original, cam_norm)
                pure_heat = self._render_pure_heatmap_rgba(original.size, cam_norm, confidence=label_prob)

                return Heatmap(
                    label=label_name,
                    probability=label_prob,
                    image_data_url=self._encode_png_data_url(overlay),
                    pure_heatmap_url=self._encode_png_data_url(pure_heat),
                )
            finally:
                h_fwd.remove()
                h_bwd.remove()
                self.model.zero_grad(set_to_none=True)

    def explain_all_findings(self, image: Image.Image) -> dict[str, Heatmap]:
        results = {}
        for label in self.labels:
            results[label] = self.explain_finding(image, target_label=label)
        return results

    @staticmethod
    def _normalize_cam(cam: torch.Tensor, confidence: float = 1.0) -> np.ndarray:
        cam = cam.detach().cpu()
        cam_min = float(cam.min())
        cam_max = float(cam.max())
        if cam_max <= cam_min:
            return np.zeros(tuple(cam.shape), dtype=np.float32)
        norm = ((cam - cam_min) / (cam_max - cam_min)).numpy().astype(np.float32)

        # Suppress ambient background noise (< 15% activation)
        noise_floor = 0.15
        norm = np.maximum(0.0, norm - noise_floor) / (1.0 - noise_floor)

        # Scale intensity by prediction probability / confidence
        # High prob (>0.6) -> Full vivid heat (1.0); Low prob (<0.4) -> Soft subdued heat
        intensity_scale = float(np.clip(confidence * 1.5, 0.25, 1.0))
        return norm * intensity_scale

    @staticmethod
    def _render_overlay(image: Image.Image, cam: np.ndarray) -> Image.Image:
        heatmap = Image.fromarray(np.uint8(np.clip(cam * 255, 0, 255)), mode="L").resize(
            image.size,
            resample=Image.Resampling.BILINEAR,
        )
        heat = np.asarray(heatmap, dtype=np.float32) / 255.0
        base = np.asarray(image.convert("RGB"), dtype=np.float32)

        # Standard red-yellow clinical thermal gradient
        color = np.zeros_like(base)
        color[..., 0] = 255.0
        color[..., 1] = 200.0 * heat
        color[..., 2] = 20.0 * (1.0 - heat)
        alpha = (0.45 * heat)[..., None]
        overlay = np.clip((base * (1.0 - alpha)) + (color * alpha), 0, 255).astype(np.uint8)
        return Image.fromarray(overlay, mode="RGB")

    @staticmethod
    def _render_pure_heatmap_rgba(size: tuple[int, int], cam: np.ndarray, confidence: float = 1.0) -> Image.Image:
        heatmap = Image.fromarray(np.uint8(np.clip(cam * 255, 0, 255)), mode="L").resize(
            size,
            resample=Image.Resampling.BILINEAR,
        )
        heat = np.asarray(heatmap, dtype=np.float32) / 255.0
        h, w = heat.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)

        # Vibrant thermal colormap: Blue -> Cyan -> Yellow -> Red
        rgba[..., 0] = np.clip(255.0 * np.sin(heat * np.pi / 2), 0, 255).astype(np.uint8)
        rgba[..., 1] = np.clip(255.0 * np.sin(heat * np.pi), 0, 255).astype(np.uint8)
        rgba[..., 2] = np.clip(255.0 * np.cos(heat * np.pi / 2), 0, 255).astype(np.uint8)

        # Adaptive Alpha based on heat and prediction confidence
        alpha_scale = float(np.clip(confidence * 1.4, 0.25, 1.0))
        rgba[..., 3] = np.clip(255.0 * (heat ** 1.4) * alpha_scale, 0, 255).astype(np.uint8)

        return Image.fromarray(rgba, mode="RGBA")

    @staticmethod
    def _encode_png_data_url(image: Image.Image) -> str:
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"
