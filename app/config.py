from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

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
