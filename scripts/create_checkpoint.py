import argparse
import os
import sys
from pathlib import Path

# Ensure TORCH_HOME is strictly local to workspace
PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ["TORCH_HOME"] = str(PROJECT_ROOT / ".torch")
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torchvision import models
from app.config import DEFAULT_LABELS, DEFAULT_CHECKPOINT_PATH


def generate_checkpoint(output_path: Path, arch: str = "convnext_small"):
    print(f"Creating checkpoint for labels: {DEFAULT_LABELS}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    arch_norm = arch.lower().replace("-", "_")
    if arch_norm in ("convnext_small", "convnext"):
        print("Loading ConvNeXt-Small backbone...")
        model = models.convnext_small(weights=models.ConvNeXt_Small_Weights.IMAGENET1K_V1)
        in_features = model.classifier[2].in_features
        model.classifier[2] = torch.nn.Linear(in_features, len(DEFAULT_LABELS))
        torch.nn.init.xavier_uniform_(model.classifier[2].weight)
        torch.nn.init.zeros_(model.classifier[2].bias)
        arch_name = "convnext_small"
    else:
        print("Loading DenseNet-121 backbone...")
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        in_features = model.classifier.in_features
        model.classifier = torch.nn.Linear(in_features, len(DEFAULT_LABELS))
        torch.nn.init.xavier_uniform_(model.classifier.weight)
        torch.nn.init.zeros_(model.classifier.bias)
        arch_name = "DenseNet121"
    
    checkpoint_data = {
        "model_state_dict": model.state_dict(),
        "labels": DEFAULT_LABELS,
        "metadata": {
            "architecture": arch_name,
            "backbone": "ImageNet1k-V1",
            "label_preset": "competition",
            "labels": DEFAULT_LABELS,
            "description": f"Pretrained {arch_name} initialized for CheXpert 5-competition labels"
        }
    }
    
    torch.save(checkpoint_data, output_path)
    print(f"Saved checkpoint successfully to {output_path} ({output_path.stat().st_size / (1024*1024):.2f} MB)")


def parse_args():
    parser = argparse.ArgumentParser(description="Create initialized baseline checkpoint.")
    parser.add_argument("--output", type=Path, required=True, help="Destination path for .pt checkpoint (required)")
    parser.add_argument("--arch", type=str, default="convnext_small", choices=["convnext_small", "densenet121"], help="Model architecture")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_checkpoint(args.output, args.arch)
