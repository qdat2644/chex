from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from app.config import DEFAULT_LABELS, STANFORD_U_ONES_LABELS


class CheXpertDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        data_root: str | Path,
        transform,
        labels: list[str] | None = None,
        uncertain_policy: str = "u_ones_zeros",
        view: str = "all",
        return_mask: bool = False,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.data_root = Path(data_root)
        self.transform = transform
        self.labels = labels or DEFAULT_LABELS
        self.uncertain_policy = str(uncertain_policy).lower()
        self.view = str(view).lower()
        self.return_mask = return_mask or (self.uncertain_policy == "ignore")
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
            self.frame = self.frame[self.frame["Frontal/Lateral"].astype(str).str.lower() == expected.lower()].reset_index(drop=True)
            if self.frame.empty:
                raise ValueError(f"No {expected} rows found in {self.csv_path}")

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.frame.iloc[index]
        image_path = self._resolve_image_path(row["Path"])
        image = Image.open(image_path).convert("RGB")

        targets = []
        masks = []
        for label in self.labels:
            t_val, m_val = self._normalize_label_and_mask(row[label], label)
            targets.append(t_val)
            masks.append(m_val)

        target_tensor = torch.tensor(targets, dtype=torch.float32)
        mask_tensor = torch.tensor(masks, dtype=torch.float32)
        img_tensor = self.transform(image)

        if self.return_mask:
            return img_tensor, target_tensor, mask_tensor
        return img_tensor, target_tensor

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

    def _normalize_label_and_mask(self, value, label_name: str = "") -> tuple[float, float]:
        if pd.isna(value):
            return 0.0, 1.0
        val = float(value)
        if val == -1.0:
            if self.uncertain_policy == "one":
                return 1.0, 1.0
            if self.uncertain_policy in ("u_ones_zeros", "stanford"):
                return (1.0 if label_name in STANFORD_U_ONES_LABELS else 0.0), 1.0
            if self.uncertain_policy == "smooth":
                return 0.6, 1.0
            if self.uncertain_policy == "ignore":
                # Mask out uncertain label so loss/metric does not penalize or coerce to zero!
                return 0.0, 0.0
            if self.uncertain_policy == "zero":
                return 0.0, 1.0
            return 0.0, 1.0
        return (1.0 if val == 1.0 else 0.0), 1.0
