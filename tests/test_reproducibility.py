from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader

from app.dataset import CheXpertDataset
from scripts.train import (
    AsymmetricLoss,
    run_epoch,
    set_seed,
)


class TinyDummyClassifier(torch.nn.Module):
    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.fc = torch.nn.Linear(8 * 16 * 16, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.conv(x))
        h = h.view(h.size(0), -1)
        return self.fc(h)


class TestTrainingReproducibility(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

        # Create 16 dummy images
        rows = []
        for i in range(16):
            fname = f"cxr_{i}.png"
            img_path = self.root / fname
            Image.new("RGB", (16, 16), color=(i * 10 % 255, i * 15 % 255, 100)).save(img_path)
            rows.append({
                "Path": fname,
                "Frontal/Lateral": "Frontal",
                "Atelectasis": float(i % 2),
                "Cardiomegaly": float((i + 1) % 2),
                "Consolidation": 0.0,
                "Edema": float(i % 3 == 0),
                "Pleural Effusion": 0.0,
            })
        self.csv_path = self.root / "train.csv"
        pd.DataFrame(rows).to_csv(self.csv_path, index=False)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_continuous_vs_resume_training_determinism(self):
        """
        Task 6: Test that training 2 epochs continuously matches
        training 1 epoch -> saving checkpoint (model, optimizer, scheduler, RNG) -> resuming for epoch 2.
        """
        labels = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]
        device = torch.device("cpu")

        # ----------------------------------------------------
        # Run A: Continuous 2 Epochs
        # ----------------------------------------------------
        set_seed(42, deterministic=True)
        ds_a = CheXpertDataset(
            self.csv_path,
            self.root,
            lambda img: torch.tensor(np.array(img).transpose(2, 0, 1), dtype=torch.float32) / 255.0,
            labels=labels,
            uncertain_policy="u_ones_zeros",
        )
        loader_a = DataLoader(ds_a, batch_size=4, shuffle=True, generator=torch.Generator().manual_seed(42))

        model_a = TinyDummyClassifier(num_classes=5).to(device)
        # Fix initial weights
        torch.manual_seed(100)
        for p in model_a.parameters():
            p.data.normal_(0, 0.1)

        opt_a = torch.optim.AdamW(model_a.parameters(), lr=1e-3)
        sched_a = torch.optim.lr_scheduler.CosineAnnealingLR(opt_a, T_max=2, eta_min=1e-5)
        crit_a = AsymmetricLoss()

        # Epoch 1
        run_epoch(model_a, loader_a, crit_a, device, opt_a)
        sched_a.step()

        # Epoch 2
        run_epoch(model_a, loader_a, crit_a, device, opt_a)
        sched_a.step()

        weights_continuous = [p.clone().detach() for p in model_a.parameters()]

        # ----------------------------------------------------
        # Run B: 1 Epoch -> Checkpoint with RNG & Scheduler -> Resume Epoch 2
        # ----------------------------------------------------
        set_seed(42, deterministic=True)
        ds_b = CheXpertDataset(
            self.csv_path,
            self.root,
            lambda img: torch.tensor(np.array(img).transpose(2, 0, 1), dtype=torch.float32) / 255.0,
            labels=labels,
            uncertain_policy="u_ones_zeros",
        )
        loader_b = DataLoader(ds_b, batch_size=4, shuffle=True, generator=torch.Generator().manual_seed(42))

        model_b = TinyDummyClassifier(num_classes=5).to(device)
        torch.manual_seed(100)
        for p in model_b.parameters():
            p.data.normal_(0, 0.1)

        opt_b = torch.optim.AdamW(model_b.parameters(), lr=1e-3)
        sched_b = torch.optim.lr_scheduler.CosineAnnealingLR(opt_b, T_max=2, eta_min=1e-5)
        crit_b = AsymmetricLoss()

        # Epoch 1
        run_epoch(model_b, loader_b, crit_b, device, opt_b)
        sched_b.step()

        # Save Checkpoint State
        ckpt_path = self.root / "ckpt_epoch1.pt"
        payload = {
            "model_state_dict": model_b.state_dict(),
            "optimizer_state_dict": opt_b.state_dict(),
            "scheduler_state_dict": sched_b.state_dict(),
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
            },
        }
        torch.save(payload, ckpt_path)

        # Re-load into fresh instances
        model_resume = TinyDummyClassifier(num_classes=5).to(device)
        loaded = torch.load(ckpt_path, map_location=device, weights_only=False)
        model_resume.load_state_dict(loaded["model_state_dict"])

        opt_resume = torch.optim.AdamW(model_resume.parameters(), lr=1e-3)
        opt_resume.load_state_dict(loaded["optimizer_state_dict"])

        sched_resume = torch.optim.lr_scheduler.CosineAnnealingLR(opt_resume, T_max=2, eta_min=1e-5)
        sched_resume.load_state_dict(loaded["scheduler_state_dict"])

        # Restore RNG
        rng = loaded["rng_state"]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])

        # Run Epoch 2 on resumed model
        run_epoch(model_resume, loader_b, crit_b, device, opt_resume)
        sched_resume.step()

        weights_resumed = [p.clone().detach() for p in model_resume.parameters()]

        # Compare weights
        for w_c, w_r in zip(weights_continuous, weights_resumed, strict=True):
            self.assertTrue(torch.allclose(w_c, w_r, atol=1e-5), "Model weights did not match after resume!")


if __name__ == "__main__":
    unittest.main()
