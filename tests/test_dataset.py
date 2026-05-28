from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd
from PIL import Image
from torchvision import transforms

from app.dataset import CheXpertDataset


class CheXpertDatasetTest(unittest.TestCase):
    def test_resolves_kaggle_layout_and_normalizes_uncertain_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "train" / "patient00001" / "study1" / "view1_frontal.jpg"
            image_path.parent.mkdir(parents=True)
            Image.new("L", (8, 8), color=128).save(image_path)

            csv_path = root / "train.csv"
            pd.DataFrame(
                [
                    {
                        "Path": "CheXpert-v1.0-small/train/patient00001/study1/view1_frontal.jpg",
                        "Atelectasis": -1,
                        "Cardiomegaly": 1,
                        "Consolidation": 0,
                        "Edema": None,
                        "Pleural Effusion": -1,
                    }
                ]
            ).to_csv(csv_path, index=False)

            dataset = CheXpertDataset(csv_path, root, transforms.ToTensor(), uncertain_policy="one")
            image, target = dataset[0]

            self.assertEqual(tuple(image.shape), (3, 8, 8))
            self.assertEqual(target.tolist(), [1.0, 1.0, 0.0, 0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
