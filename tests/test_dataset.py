from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

from app.dataset import CheXpertDataset


class TestCheXpertDatasetPolicies(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        
        # Create a dummy image
        img_path = self.root / "test_cxr.png"
        Image.new("RGB", (64, 64), color="gray").save(img_path)

        # Create a dummy CSV with various labels: 1, 0, -1, NaN
        self.csv_path = self.root / "test.csv"
        df = pd.DataFrame([
            {
                "Path": "test_cxr.png",
                "Frontal/Lateral": "Frontal",
                "Atelectasis": 1.0,
                "Cardiomegaly": -1.0, # In Stanford policy: 0.0
                "Consolidation": 0.0,
                "Edema": -1.0,        # In Stanford policy: 1.0
                "Pleural Effusion": float("nan"),
            }
        ])
        df.to_csv(self.csv_path, index=False)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_policy_one(self):
        ds = CheXpertDataset(
            csv_path=self.csv_path,
            data_root=self.root,
            transform=lambda x: torch.zeros(3, 64, 64),
            uncertain_policy="one",
        )
        _, target = ds[0]
        # Atelectasis=1.0, Cardiomegaly=-1->1.0, Consolidation=0.0, Edema=-1->1.0, Pleural Effusion=nan->0.0
        expected = torch.tensor([1.0, 1.0, 0.0, 1.0, 0.0])
        self.assertTrue(torch.allclose(target, expected))

    def test_policy_zero(self):
        ds = CheXpertDataset(
            csv_path=self.csv_path,
            data_root=self.root,
            transform=lambda x: torch.zeros(3, 64, 64),
            uncertain_policy="zero",
        )
        _, target = ds[0]
        # Atelectasis=1.0, Cardiomegaly=-1->0.0, Consolidation=0.0, Edema=-1->0.0, Pleural Effusion=0.0
        expected = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0])
        self.assertTrue(torch.allclose(target, expected))

    def test_policy_stanford(self):
        ds = CheXpertDataset(
            csv_path=self.csv_path,
            data_root=self.root,
            transform=lambda x: torch.zeros(3, 64, 64),
            uncertain_policy="stanford",
        )
        _, target = ds[0]
        # Atelectasis=1.0, Cardiomegaly=-1->0.0, Consolidation=0.0, Edema=-1->1.0, Pleural Effusion=0.0
        expected = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0])
        self.assertTrue(torch.allclose(target, expected))

    def test_policy_smooth(self):
        ds = CheXpertDataset(
            csv_path=self.csv_path,
            data_root=self.root,
            transform=lambda x: torch.zeros(3, 64, 64),
            uncertain_policy="smooth",
        )
        _, target = ds[0]
        # Atelectasis=1.0, Cardiomegaly=-1->0.6, Consolidation=0.0, Edema=-1->0.6, Pleural Effusion=0.0
        expected = torch.tensor([1.0, 0.6, 0.0, 0.6, 0.0])
        self.assertTrue(torch.allclose(target, expected))

    def test_policy_ignore_masking(self):
        ds = CheXpertDataset(
            csv_path=self.csv_path,
            data_root=self.root,
            transform=lambda x: torch.zeros(3, 64, 64),
            uncertain_policy="ignore",
        )
        _, target, mask = ds[0]
        # Atelectasis: valid(1.0), Cardiomegaly: -1 (mask=0.0), Consolidation: valid(1.0), Edema: -1 (mask=0.0), Effusion: nan(valid=1.0)
        expected_mask = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0])
        self.assertTrue(torch.allclose(mask, expected_mask))


if __name__ == "__main__":
    unittest.main()
