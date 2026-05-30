from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "chexpert_densenet121_v2.pt"
DEFAULT_THRESHOLDS_PATH = PROJECT_ROOT / "outputs" / "evaluation" / "thresholds.json"

MODEL_INFO = {
    "architecture": "DenseNet121",
    "label_count": 5,
    "mean_auc": 0.8763721175369092,
    "mean_auc_display": "0.8764",
    "valid_rows": 202,
    "checkpoint": "checkpoints/chexpert_densenet121_v2.pt",
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

DEFAULT_IMAGE_SIZE = 224
DEFAULT_THRESHOLD = 0.5


def resolve_label_preset(name: str) -> list[str]:
    try:
        return list(LABEL_PRESETS[name])
    except KeyError as exc:
        valid = ", ".join(sorted(LABEL_PRESETS))
        raise ValueError(f"Unknown label preset {name!r}. Expected one of: {valid}") from exc
