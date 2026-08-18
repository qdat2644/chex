import os
import sys
from pathlib import Path

# Ensure TORCH_HOME is strictly local to c:\test
PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ["TORCH_HOME"] = str(PROJECT_ROOT / ".torch")
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torchvision import models
from app.config import DEFAULT_LABELS, DEFAULT_CHECKPOINT_PATH

def generate_checkpoint():
    print(f"Creating checkpoint for labels: {DEFAULT_LABELS}")
    checkpoint_dir = DEFAULT_CHECKPOINT_PATH.parent
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Download ImageNet pretrained backbone into local TORCH_HOME
    print("Loading pretrained DenseNet-121 backbone...")
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    in_features = model.classifier.in_features
    model.classifier = torch.nn.Linear(in_features, len(DEFAULT_LABELS))
    
    # Initialize classifier weights nicely
    torch.nn.init.xavier_uniform_(model.classifier.weight)
    torch.nn.init.zeros_(model.classifier.bias)
    
    checkpoint_data = {
        "model_state_dict": model.state_dict(),
        "labels": DEFAULT_LABELS,
        "metadata": {
            "architecture": "DenseNet121",
            "backbone": "ImageNet1k-V1",
            "label_preset": "competition",
            "labels": DEFAULT_LABELS,
            "description": "Pretrained DenseNet-121 initialized for CheXpert 5-competition labels demo"
        }
    }
    
    torch.save(checkpoint_data, DEFAULT_CHECKPOINT_PATH)
    print(f"Saved checkpoint successfully to {DEFAULT_CHECKPOINT_PATH} ({DEFAULT_CHECKPOINT_PATH.stat().st_size / (1024*1024):.2f} MB)")

if __name__ == "__main__":
    generate_checkpoint()
