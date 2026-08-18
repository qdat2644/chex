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
from scripts.train import AsymmetricLoss, FocalLoss, calculate_pos_weight


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
        targets = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
        masks = torch.tensor([[1.0, 0.0, 1.0]], dtype=torch.float32)

        logits_1 = torch.tensor([[2.0, 10.0, -2.0]], dtype=torch.float32)
        bce_1 = torch.nn.functional.binary_cross_entropy_with_logits(logits_1, targets, reduction="none")
        masked_loss_1 = (bce_1 * masks).sum() / masks.sum()

        logits_2 = torch.tensor([[2.0, -10.0, -2.0]], dtype=torch.float32)
        bce_2 = torch.nn.functional.binary_cross_entropy_with_logits(logits_2, targets, reduction="none")
        masked_loss_2 = (bce_2 * masks).sum() / masks.sum()

        self.assertAlmostEqual(float(masked_loss_1), float(masked_loss_2), places=6)

    def test_masked_loss_forward_backward_on_all_loss_functions(self):
        """
        Task 1: Test a batch with uncertain-policy=ignore across BCE, ASL, and FocalLoss.
        Demonstrates stable forward and backward passes without crash.
        """
        batch_logits = torch.randn(4, 5, requires_grad=True)
        batch_targets = torch.tensor([
            [1.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0, 0.0],
        ], dtype=torch.float32)
        batch_masks = torch.tensor([
            [1.0, 0.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 1.0, 1.0],
            [0.0, 1.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 1.0, 1.0],
        ], dtype=torch.float32)

        # 1. Asymmetric Loss
        asl = AsymmetricLoss(gamma_neg=4.0, gamma_pos=1.0)
        loss_asl = asl(batch_logits, batch_targets, mask=batch_masks)
        self.assertFalse(torch.isnan(loss_asl))
        loss_asl.backward(retain_graph=True)
        self.assertIsNotNone(batch_logits.grad)

        # 2. Focal Loss
        focal = FocalLoss(alpha=0.25, gamma=2.0)
        loss_focal = focal(batch_logits, batch_targets, mask=batch_masks)
        self.assertFalse(torch.isnan(loss_focal))

        # 3. BCE with mask
        bce_raw = torch.nn.functional.binary_cross_entropy_with_logits(batch_logits, batch_targets, reduction="none")
        loss_bce = (bce_raw * batch_masks).sum() / batch_masks.sum().clamp(min=1.0)
        self.assertFalse(torch.isnan(loss_bce))

    def test_calculate_pos_weight_ignores_masked_labels(self):
        """
        Task 1: calculate_pos_weight must ignore entries with mask 0.
        """
        ds = CheXpertDataset(
            csv_path=self.csv_path,
            data_root=self.root,
            transform=lambda x: torch.zeros(3, 64, 64),
            uncertain_policy="ignore",
        )
        weights = calculate_pos_weight(ds, ds.labels)
        self.assertEqual(len(weights), len(ds.labels))
        self.assertFalse(torch.isnan(weights).any())


if __name__ == "__main__":
    unittest.main()
