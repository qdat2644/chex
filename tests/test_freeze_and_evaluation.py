from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from scripts.ensemble_predictions import ensemble_prediction_files
from scripts.freeze_experiment import compute_file_sha256, freeze_experiment
from scripts.sync_reports import check_file, generate_tbd_latex, generate_tbd_readme, normalize_text


class TestFreezeAndEvaluationIntegrity(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temp_dir.name)

        # Create dummy valid config
        self.config_path = self.tmp_path / "protocol_v0_1.yaml"
        self.config_path.write_text(
            "protocol_version: '0.1'\n"
            "model:\n"
            "  architectures:\n"
            "    - convnext_small\n"
            "    - densenet121\n"
            "  labels:\n"
            "    - Atelectasis\n"
            "    - Cardiomegaly\n"
            "seeds:\n"
            "  - 42\n"
            "  - 43\n"
            "  - 44\n"
            "  - 45\n"
            "  - 46\n",
            encoding="utf-8",
        )

        # Create dummy manifests
        self.dev_manifest_path = self.tmp_path / "manifest.json"
        self.dev_manifest_path.write_text('{"schema_version": "1.0", "role": "development"}', encoding="utf-8")

        self.locked_manifest_path = self.tmp_path / "locked_test_manifest.json"
        self.locked_manifest_path.write_text('{"schema_version": "1.0", "role": "locked_test", "locked": true}', encoding="utf-8")

        self.runs_dir = self.tmp_path / "runs"
        self.calib_dir = self.tmp_path / "calibration"
        self.ledger_path = self.tmp_path / "frozen_ledger.json"

        # Create all 10 checkpoints and 10 matching threshold artifacts
        dev_manifest_sha = compute_file_sha256(self.dev_manifest_path)
        for arch in ["convnext_small", "densenet121"]:
            for seed in [42, 43, 44, 45, 46]:
                ckpt_dir = self.runs_dir / arch / f"seed_{seed}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                ckpt_file = ckpt_dir / "best.pt"
                torch.save({"model_state_dict": {"w": torch.tensor([1.0])}}, ckpt_file)
                ckpt_sha = compute_file_sha256(ckpt_file)

                self.calib_dir.mkdir(parents=True, exist_ok=True)
                th_file = self.calib_dir / f"{arch}_seed{seed}.json"
                th_file.write_text(
                    json.dumps({
                        "checkpoint_sha256": ckpt_sha,
                        "split_manifest_sha256": dev_manifest_sha,
                        "thresholds": {"Atelectasis": 0.5, "Cardiomegaly": 0.5},
                    }),
                    encoding="utf-8",
                )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_freeze_succeeds_with_complete_artifacts(self) -> None:
        ledger = freeze_experiment(
            config_path=self.config_path,
            development_manifest_path=self.dev_manifest_path,
            locked_test_manifest_path=self.locked_manifest_path,
            runs_dir=self.runs_dir,
            calibration_dir=self.calib_dir,
            output_ledger_path=self.ledger_path,
            allow_dirty=True,
        )
        self.assertIn("convnext_small", ledger["models"])
        self.assertIn("densenet121", ledger["models"])
        self.assertEqual(len(ledger["models"]["convnext_small"]), 5)
        self.assertEqual(len(ledger["models"]["densenet121"]), 5)
        self.assertTrue(self.ledger_path.exists())

    def test_freeze_fails_when_checkpoint_missing(self) -> None:
        missing_ckpt = self.runs_dir / "convnext_small" / "seed_44" / "best.pt"
        missing_ckpt.unlink()

        with self.assertRaises(FileNotFoundError):
            freeze_experiment(
                config_path=self.config_path,
                development_manifest_path=self.dev_manifest_path,
                locked_test_manifest_path=self.locked_manifest_path,
                runs_dir=self.runs_dir,
                calibration_dir=self.calib_dir,
                output_ledger_path=self.ledger_path,
                allow_dirty=True,
            )

    def test_freeze_fails_when_threshold_artifact_missing(self) -> None:
        missing_th = self.calib_dir / "densenet121_seed46.json"
        missing_th.unlink()

        with self.assertRaises(FileNotFoundError):
            freeze_experiment(
                config_path=self.config_path,
                development_manifest_path=self.dev_manifest_path,
                locked_test_manifest_path=self.locked_manifest_path,
                runs_dir=self.runs_dir,
                calibration_dir=self.calib_dir,
                output_ledger_path=self.ledger_path,
                allow_dirty=True,
            )

    def test_freeze_fails_when_threshold_checkpoint_sha_mismatched(self) -> None:
        th_file = self.calib_dir / "convnext_small_seed42.json"
        th_file.write_text(
            json.dumps({
                "checkpoint_sha256": "tampered_fake_sha256_hash",
                "split_manifest_sha256": compute_file_sha256(self.dev_manifest_path),
                "thresholds": {"Atelectasis": 0.5},
            }),
            encoding="utf-8",
        )

        with self.assertRaises(RuntimeError):
            freeze_experiment(
                config_path=self.config_path,
                development_manifest_path=self.dev_manifest_path,
                locked_test_manifest_path=self.locked_manifest_path,
                runs_dir=self.runs_dir,
                calibration_dir=self.calib_dir,
                output_ledger_path=self.ledger_path,
                allow_dirty=True,
            )

    def test_ensemble_fails_when_fewer_than_5_seeds(self) -> None:
        csv_paths = []
        for seed in [42, 43, 44, 45]:  # only 4 seeds
            p = self.tmp_path / f"preds_seed_{seed}.csv"
            df = pd.DataFrame({
                "study_id": ["s1", "s2"],
                "patient_id": ["p1", "p2"],
                "Atelectasis_prob": [0.5, 0.6],
                "Atelectasis_target": [1.0, 0.0],
            })
            df.to_csv(p, index=False)
            csv_paths.append(p)

        out_ens = self.tmp_path / "ens.csv"
        with self.assertRaises(ValueError):
            ensemble_prediction_files(csv_paths, out_ens, labels=["Atelectasis"])

    def test_ensemble_fails_when_patient_ids_mismatched(self) -> None:
        csv_paths = []
        for seed in [42, 43, 44, 45, 46]:
            p = self.tmp_path / f"preds_seed_{seed}.csv"
            # Seed 46 has mismatched patient_id for study s2
            pid_s2 = "p2_different" if seed == 46 else "p2"
            df = pd.DataFrame({
                "study_id": ["s1", "s2"],
                "patient_id": ["p1", pid_s2],
                "Atelectasis_prob": [0.5, 0.6],
                "Atelectasis_target": [1.0, 0.0],
            })
            df.to_csv(p, index=False)
            csv_paths.append(p)

        out_ens = self.tmp_path / "ens.csv"
        with self.assertRaises(ValueError):
            ensemble_prediction_files(csv_paths, out_ens, labels=["Atelectasis"])

    def test_ensemble_averages_5_seeds_correctly(self) -> None:
        csv_paths = []
        for idx, seed in enumerate([42, 43, 44, 45, 46]):
            p = self.tmp_path / f"preds_seed_{seed}.csv"
            df = pd.DataFrame({
                "study_id": ["s1"],
                "patient_id": ["p1"],
                "Atelectasis_prob": [0.1 * (idx + 1)],  # 0.1, 0.2, 0.3, 0.4, 0.5 -> mean is 0.3
                "Atelectasis_target": [1.0],
            })
            df.to_csv(p, index=False)
            csv_paths.append(p)

        out_ens = self.tmp_path / "ens.csv"
        ens_df = ensemble_prediction_files(csv_paths, out_ens, labels=["Atelectasis"])
        self.assertAlmostEqual(ens_df["Atelectasis_prob"].iloc[0], 0.3, places=5)


if __name__ == "__main__":
    unittest.main()
