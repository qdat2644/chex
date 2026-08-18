from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Hugging Face Repository & Pinned Revision URLs (Task 3)
HUGGINGFACE_REPO = "qdat264/chexpert-convnext-small"
HUGGINGFACE_REVISION = os.getenv("HUGGINGFACE_REVISION", "d9b23e7f4c0b6b25ea958d0421295b9c0587b1c3")
HUGGINGFACE_CHECKPOINT_URL = f"https://huggingface.co/{HUGGINGFACE_REPO}/resolve/{HUGGINGFACE_REVISION}/chexpert_convnext_small.pt"
HUGGINGFACE_THRESHOLDS_URL = f"https://huggingface.co/{HUGGINGFACE_REPO}/resolve/{HUGGINGFACE_REVISION}/thresholds.json"

# Checksum & Governance Policies (SHA-256)
EXPECTED_CHECKPOINT_SHA256 = os.getenv(
    "CHEXPERT_CHECKPOINT_SHA256",
    "b8b1884a911f6ff9408de141c48034a39f4515e2c8deaadd25312e07601c9bc0",
)
EXPECTED_THRESHOLDS_SHA256 = os.getenv(
    "CHEXPERT_THRESHOLDS_SHA256",
    "7ad370d14f7941fae476d0f2ba2038fcfeedbbf70dcfd1e45cb083baa28965ea",
)
AUTO_DOWNLOAD_ENABLED = os.getenv("CHEXPERT_AUTO_DOWNLOAD", "true").lower() == "true"

# Paths
_convnext_path = PROJECT_ROOT / "checkpoints" / "chexpert_convnext_small.pt"
_densenet_path = PROJECT_ROOT / "checkpoints" / "chexpert_densenet121_v2.pt"
DEFAULT_CHECKPOINT_PATH = _convnext_path if _convnext_path.exists() else _densenet_path
DEFAULT_THRESHOLDS_PATH = PROJECT_ROOT / "outputs" / "evaluation" / "thresholds.json"
MANIFEST_PATH = PROJECT_ROOT / "outputs" / "manifest.json"

# Security & Authentication Configurations (Task 1 & Task 5)
API_KEY = os.getenv("API_KEY") or os.getenv("CHEXPERT_API_KEY")
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))


def get_phi_secret() -> str | None:
    """
    Retrieves the PHI HMAC secret from environment variable or Docker secret mount.
    """
    secret_file = os.getenv("PHI_HMAC_SECRET_FILE", "/run/secrets/phi_hmac_secret")
    p = Path(secret_file)
    if p.exists() and p.is_file():
        try:
            content = p.read_text(encoding="utf-8").strip()
            if content:
                return content
        except Exception:
            pass
    return os.getenv("PHI_HMAC_SECRET")


SUPPORTED_ARCHITECTURES = [
    "densenet121",
    "convnext_small",
    "efficientnet_v2_m",
    "resnet50",
]

MODEL_INFO = {
    "version": "1.0.0",
    "architecture": "ConvNeXt-Small",
    "revision": HUGGINGFACE_REVISION,
    "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
    "thresholds_sha256": EXPECTED_THRESHOLDS_SHA256,
    "label_count": 5,
    "mean_auc": 0.8944,
    "mean_auc_display": "0.8944",
    "valid_rows": 202,
    "checkpoint": "checkpoints/chexpert_convnext_small.pt",
    "huggingface_repo": HUGGINGFACE_REPO,
    "huggingface_url": f"https://huggingface.co/{HUGGINGFACE_REPO}",
    "validation_csv": "archive/valid.csv",
    "view": "frontal",
}

DEFAULT_LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
]

ALL_CHEXPERT_LABELS = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]

LABEL_PRESETS = {
    "competition": DEFAULT_LABELS,
    "all": ALL_CHEXPERT_LABELS,
}

STANFORD_U_ONES_LABELS = {"Atelectasis", "Edema", "Pleural Effusion"}

DEFAULT_IMAGE_SIZE = 224
DEFAULT_THRESHOLD = 0.5


def resolve_label_preset(name: str) -> list[str]:
    try:
        return list(LABEL_PRESETS[name])
    except KeyError as exc:
        valid = ", ".join(sorted(LABEL_PRESETS))
        raise ValueError(f"Unknown label preset {name!r}. Expected one of: {valid}") from exc
