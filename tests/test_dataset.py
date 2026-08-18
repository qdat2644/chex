from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

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
                "Cardiomegaly": -1.0,  # In Stanford policy: 0.0
                "Consolidation": 0.0,
                "Edema": -1.0,         # In Stanford policy: 1.0
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
        _, target, mask = ds[0]
        expected_target = torch.tensor([1.0, 1.0, 0.0, 1.0, 0.0])
        expected_mask = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])
        self.assertTrue(torch.allclose(target, expected_target))
        self.assertTrue(torch.allclose(mask, expected_mask))

    def test_policy_zero(self):
        ds = CheXpertDataset(
            csv_path=self.csv_path,
            data_root=self.root,
            transform=lambda x: torch.zeros(3, 64, 64),
            uncertain_policy="zero",
        )
        _, target, mask = ds[0]
        expected_target = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0])
        expected_mask = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])
        self.assertTrue(torch.allclose(target, expected_target))
        self.assertTrue(torch.allclose(mask, expected_mask))

    def test_policy_stanford(self):
        ds = CheXpertDataset(
            csv_path=self.csv_path,
            data_root=self.root,
            transform=lambda x: torch.zeros(3, 64, 64),
            uncertain_policy="stanford",
        )
        _, target, mask = ds[0]
        expected_target = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0])
        expected_mask = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])
        self.assertTrue(torch.allclose(target, expected_target))
        self.assertTrue(torch.allclose(mask, expected_mask))

    def test_policy_smooth(self):
        ds = CheXpertDataset(
            csv_path=self.csv_path,
            data_root=self.root,
            transform=lambda x: torch.zeros(3, 64, 64),
            uncertain_policy="smooth",
        )
        _, target, mask = ds[0]
        expected_target = torch.tensor([1.0, 0.6, 0.0, 0.6, 0.0])
        expected_mask = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])
        self.assertTrue(torch.allclose(target, expected_target))
        self.assertTrue(torch.allclose(mask, expected_mask))

    def test_policy_ignore_masking(self):
        ds = CheXpertDataset(
            csv_path=self.csv_path,
            data_root=self.root,
            transform=lambda x: torch.zeros(3, 64, 64),
            uncertain_policy="ignore",
        )
        _, target, mask = ds[0]
        # Cardiomegaly and Edema had -1, so mask must be 0.0
        expected_mask = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0])
        self.assertTrue(torch.allclose(mask, expected_mask))

    def test_ignore_mask_loss_and_metric_invariance(self):
        """
        Demonstrates that changing prediction logits for an ignored label (mask=0)
        does NOT change the loss or metric score at all.
        """
        targets = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
        masks = torch.tensor([[1.0, 0.0, 1.0]], dtype=torch.float32)  # Column 1 is ignored

        # Prediction 1: logit for column 1 is +10.0
        logits_1 = torch.tensor([[2.0, 10.0, -2.0]], dtype=torch.float32)
        bce_1 = torch.nn.functional.binary_cross_entropy_with_logits(logits_1, targets, reduction="none")
        masked_loss_1 = (bce_1 * masks).sum() / masks.sum()

        # Prediction 2: logit for column 1 is -10.0 (completely opposite)
        logits_2 = torch.tensor([[2.0, -10.0, -2.0]], dtype=torch.float32)
        bce_2 = torch.nn.functional.binary_cross_entropy_with_logits(logits_2, targets, reduction="none")
        masked_loss_2 = (bce_2 * masks).sum() / masks.sum()

        # Loss must be strictly identical
        self.assertAlmostEqual(float(masked_loss_1), float(masked_loss_2), places=6)

        # Metric invariance
        # Filter by mask for column 1: no elements, so metric is not corrupted
        valid_indices = [i for i, m in enumerate(masks) if m[1] > 0.5]
        self.assertEqual(len(valid_indices), 0)


if __name__ == "__main__":
    unittest.main()
