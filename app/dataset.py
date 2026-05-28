from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from app.config import DEFAULT_LABELS


class CheXpertDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        csv_path: str | Path,
        data_root: str | Path,
        transform,
        labels: list[str] | None = None,
        uncertain_policy: str = "zero",
        view: str = "all",
    ) -> None:
        self.csv_path = Path(csv_path)
        self.data_root = Path(data_root)
        self.transform = transform
        self.labels = labels or DEFAULT_LABELS
        self.uncertain_policy = uncertain_policy
        self.view = view
        self.frame = pd.read_csv(self.csv_path)

        missing = [label for label in self.labels if label not in self.frame.columns]
        if missing:
            raise ValueError(f"Missing label columns in {self.csv_path}: {missing}")
        if "Path" not in self.frame.columns:
            raise ValueError(f"Missing Path column in {self.csv_path}")
        if self.view != "all":
            if "Frontal/Lateral" not in self.frame.columns:
                raise ValueError(f"Missing Frontal/Lateral column in {self.csv_path}")
            expected = "Frontal" if self.view == "frontal" else "Lateral"
            self.frame = self.frame[self.frame["Frontal/Lateral"] == expected].reset_index(drop=True)
            if self.frame.empty:
                raise ValueError(f"No {expected} rows found in {self.csv_path}")

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.frame.iloc[index]
        image_path = self._resolve_image_path(row["Path"])
        image = Image.open(image_path).convert("RGB")
        target = torch.tensor(
            [self._normalize_label(row[label]) for label in self.labels],
            dtype=torch.float32,
        )
        return self.transform(image), target

    def _resolve_image_path(self, path_value: str) -> Path:
        path = Path(str(path_value))
        if path.is_absolute():
            return path

        candidate = self.data_root / path
        if candidate.exists():
            return candidate

        parts = path.parts
        if parts and parts[0].startswith("CheXpert"):
            candidate = self.data_root / Path(*parts[1:])
            if candidate.exists():
                return candidate

        return self.data_root / path

    def _normalize_label(self, value) -> float:
        if pd.isna(value):
            return 0.0
        value = float(value)
        if value == -1.0:
            if self.uncertain_policy == "one":
                return 1.0
            if self.uncertain_policy == "ignore":
                return 0.0
            return 0.0
        return 1.0 if value == 1.0 else 0.0
