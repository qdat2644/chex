"""Export CheXpert ConvNeXt-Small checkpoint to ONNX format with dynamic batching."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from app.config import DEFAULT_CHECKPOINT_PATH, DEFAULT_IMAGE_SIZE, DEFAULT_LABELS
from app.model import build_model


def export_to_onnx(
    checkpoint_path: Path,
    output_path: Path,
    image_size: int = DEFAULT_IMAGE_SIZE,
    opset_version: int = 17,
) -> Path:
    print(f"[ONNX Export] Loading PyTorch checkpoint: {checkpoint_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metadata = checkpoint.get("metadata", {}) if isinstance(checkpoint, dict) else {}
    arch = metadata.get("architecture", "convnext_small")
    labels = metadata.get("labels", DEFAULT_LABELS)
    num_labels = len(labels)

    print(f"[ONNX Export] Architecture: {arch} | Labels: {num_labels}")
    model = build_model(arch=arch, num_labels=num_labels, pretrained=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise ValueError("Invalid checkpoint structure.")

    model.load_state_dict(state_dict)
    model.eval()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dummy_input = torch.randn(1, 3, image_size, image_size, dtype=torch.float32)

    print(f"[ONNX Export] Exporting to {output_path} (Opset {opset_version})...")
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        dynamo=False,
    )

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"[ONNX Export] Successfully exported ONNX model ({size_mb:.2f} MB) to: {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export PyTorch CheXpert model to ONNX.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH, help="Path to .pt checkpoint")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("checkpoints/chexpert_convnext_small.onnx"),
        help="Path for exported .onnx file",
    )
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE, help="Input resolution")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        export_to_onnx(
            checkpoint_path=args.checkpoint,
            output_path=args.output,
            image_size=args.image_size,
            opset_version=args.opset,
        )
    except Exception as exc:
        print(f"Error during ONNX export: {exc}", file=sys.stderr)
        sys.exit(1)
