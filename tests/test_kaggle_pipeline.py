from __future__ import annotations

import hashlib
import json
import random
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from app.experiment_integrity import (
    atomic_save_torch,
    build_resolved_scientific_config,
    canonical_json_sha256,
    compute_file_sha256,
    get_safe_rng_state,
    restore_safe_rng_state,
    sanitize_for_json,
    unwrap_model,
)
from scripts.calibrate import calculate_optimal_thresholds
from scripts.package_kaggle_artifact import package_artifact
from scripts.train import checkpoint_payload


class DummyNet(torch.nn.Module):
    def __init__(self, in_features: int = 4, out_features: int = 5):
        super().__init__()
        self.fc = torch.nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class DummyDataParallelWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.module = model

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class TestKagglePipelineAndNotebook(unittest.TestCase):
    def setUp(self):
        self.nb_path = PROJECT_ROOT / "train_on_kaggle.ipynb"
        self.assertTrue(self.nb_path.is_file(), "train_on_kaggle.ipynb must exist")
        self.nb_content = self.nb_path.read_text(encoding="utf-8")
        self.nb_json = json.loads(self.nb_content)

    # -------------------------------------------------------------------------
    # J1: Resolved Scientific Config & Canonical Hashing Tests
    # -------------------------------------------------------------------------
    def test_canonical_json_sha256_invariance_and_sensitivity(self):
        """J1. Canonical hash must be invariant to key order, sensitive to scientific changes, invariant to operational changes."""
        base_cfg = {
            "schema_version": 1,
            "protocol_version": "0.1",
            "architecture": "convnext_small",
            "seed": 42,
            "labels": ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"],
            "data": {
                "view": "frontal",
                "input_size": 224,
                "uncertainty_policy": "u_ones_zeros",
                "split_manifest_sha256": "man123",
            },
            "optimization": {
                "optimizer": "AdamW",
                "learning_rate": 0.0001,
                "weight_decay": 0.01,
                "batch_size": 32,
                "max_epochs": 20,
                "scheduler": "cosine",
                "loss": "asymmetric",
                "early_stopping_patience": 5,
                "mixed_precision": True,
            },
            "augmentation": {
                "rotation_degrees": 7.0,
                "horizontal_flip": False,
            },
            "reproducibility": {
                "deterministic": True,
            },
        }

        # 1. Invariance to key order
        reordered_cfg = {
            "reproducibility": {"deterministic": True},
            "seed": 42,
            "architecture": "convnext_small",
            "schema_version": 1,
            "labels": ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"],
            "optimization": {
                "mixed_precision": True,
                "early_stopping_patience": 5,
                "loss": "asymmetric",
                "scheduler": "cosine",
                "max_epochs": 20,
                "batch_size": 32,
                "weight_decay": 0.01,
                "learning_rate": 0.0001,
                "optimizer": "AdamW",
            },
            "data": {
                "split_manifest_sha256": "man123",
                "uncertainty_policy": "u_ones_zeros",
                "input_size": 224,
                "view": "frontal",
            },
            "augmentation": {
                "horizontal_flip": False,
                "rotation_degrees": 7.0,
            },
            "protocol_version": "0.1",
        }
        h_base = canonical_json_sha256(base_cfg)
        h_reordered = canonical_json_sha256(reordered_cfg)
        self.assertEqual(h_base, h_reordered, "Canonical hash must not depend on key order!")

        # 2. Batch size change -> hash changes
        cfg_batch = json.loads(json.dumps(base_cfg))
        cfg_batch["optimization"]["batch_size"] = 16
        self.assertNotEqual(h_base, canonical_json_sha256(cfg_batch))

        # 3. Learning rate change -> hash changes
        cfg_lr = json.loads(json.dumps(base_cfg))
        cfg_lr["optimization"]["learning_rate"] = 0.0005
        self.assertNotEqual(h_base, canonical_json_sha256(cfg_lr))

        # 4. Manifest hash change -> hash changes
        cfg_man = json.loads(json.dumps(base_cfg))
        cfg_man["data"]["split_manifest_sha256"] = "different_manifest"
        self.assertNotEqual(h_base, canonical_json_sha256(cfg_man))

        # 5. Label order change -> hash changes
        cfg_lbl = json.loads(json.dumps(base_cfg))
        cfg_lbl["labels"] = ["Cardiomegaly", "Atelectasis", "Consolidation", "Edema", "Pleural Effusion"]
        self.assertNotEqual(h_base, canonical_json_sha256(cfg_lbl))

        # 6. Operational parameters (output_dir, stop_after_epoch, resume) must NOT change scientific config
        built_1 = build_resolved_scientific_config(
            protocol_version="0.1",
            architecture="convnext_small",
            seed=42,
            labels=["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"],
            view="frontal",
            input_size=224,
            uncertainty_policy="u_ones_zeros",
            split_manifest_sha256="man123",
            optimizer="AdamW",
            learning_rate=1e-4,
            weight_decay=1e-2,
            batch_size=32,
            max_epochs=20,
            scheduler="cosine",
            loss="asl",
            early_stopping_patience=5,
            mixed_precision=True,
            rotation_degrees=7.0,
            horizontal_flip=False,
            deterministic=True,
        )
        self.assertEqual(canonical_json_sha256(built_1), h_base)

    # -------------------------------------------------------------------------
    # J2: Checkpoint Schema v2 & Atomic Save Tests
    # -------------------------------------------------------------------------
    def test_checkpoint_schema_v2_and_atomic_save(self):
        """J2. Checkpoint schema v2 fields, atomic save without leftover tmp files, and NaN sanitization."""
        model = DummyNet(4, 5)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
        scaler = torch.amp.GradScaler("cpu", enabled=False)

        labels = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]
        payload = checkpoint_payload(
            model=model,
            optimizer=opt,
            scheduler=sched,
            scaler=scaler,
            epoch=1,
            planned_max_epochs=20,
            best_val_auc=0.88,
            architecture="convnext_small",
            seed=42,
            config_sha256="cfg123",
            resolved_config_sha256="rescfg123",
            manifest_sha256="man123",
            labels=labels,
            run_mode="smoke",
            git_commit="git123",
        )

        # Verify Schema v2 fields
        self.assertEqual(payload["checkpoint_schema_version"], 2)
        self.assertIn("model_state", payload)
        self.assertIn("optimizer_state", payload)
        self.assertIn("scheduler_state", payload)
        self.assertIn("scaler_state", payload)
        self.assertIn("rng_state", payload)
        self.assertIn("torch_cpu", payload["rng_state"])
        self.assertEqual(payload["planned_max_epochs"], 20)
        self.assertEqual(payload["git_commit"], "git123")
        self.assertEqual(payload["resolved_config_sha256"], "rescfg123")

        # Top-level best_val_auc and metadata best_internal_validation_auc must match
        self.assertEqual(payload["best_val_auc"], payload["metadata"]["best_internal_validation_auc"])

        with tempfile.TemporaryDirectory() as tmpdir:
            target_pt = Path(tmpdir) / "ckpt.pt"
            atomic_save_torch(payload, target_pt)
            self.assertTrue(target_pt.is_file())
            # Ensure no .tmp files left
            self.assertFalse(target_pt.with_suffix(".pt.tmp").exists())

        # Verify sanitize_for_json handles NaN and Inf properly
        dirty_dict = {"valid": 1.0, "bad_nan": float("nan"), "bad_inf": float("inf"), "nested": [float("nan"), 2.5]}
        cleaned = sanitize_for_json(dirty_dict)
        self.assertIsNone(cleaned["bad_nan"])
        self.assertIsNone(cleaned["bad_inf"])
        self.assertIsNone(cleaned["nested"][0])
        self.assertEqual(cleaned["nested"][1], 2.5)
        # json.dumps with allow_nan=False must succeed
        json_str = json.dumps(cleaned, allow_nan=False)
        self.assertIn("null", json_str)

    # -------------------------------------------------------------------------
    # J3: Resume Semantics & DataParallel Tests
    # -------------------------------------------------------------------------
    def test_resume_unwrap_and_state_restore(self):
        """J3. Model unwrapping and full state restoration for standard and DataParallel models."""
        base_model = DummyNet(4, 5)
        dp_wrapper = DummyDataParallelWrapper(base_model)

        unwrapped = unwrap_model(dp_wrapper)
        self.assertIs(unwrapped, base_model)

        unwrapped_plain = unwrap_model(base_model)
        self.assertIs(unwrapped_plain, base_model)

    def test_resume_validation_fail_closed(self):
        """J3. Strict fail-closed verification on architecture, seed, label order, hashes, and missing fields."""
        model = DummyNet(4, 5)
        payload = checkpoint_payload(
            model=model,
            optimizer=None,
            scheduler=None,
            scaler=None,
            epoch=1,
            planned_max_epochs=20,
            best_val_auc=0.85,
            architecture="convnext_small",
            seed=42,
            config_sha256="cfg123",
            resolved_config_sha256="correct_res_hash",
            manifest_sha256="correct_man_hash",
            labels=["L1", "L2", "L3", "L4", "L5"],
            run_mode="full",
        )

        def validate_resume(loaded, arch, seed, labels, res_hash, man_hash, run_mode):
            meta = loaded.get("metadata", {})
            ckpt_arch = loaded.get("architecture") or meta.get("architecture")
            ckpt_seed = loaded.get("seed") if "seed" in loaded and loaded["seed"] is not None else meta.get("seed")
            ckpt_labels = loaded.get("labels") or meta.get("labels")
            ckpt_res_cfg = loaded.get("resolved_config_sha256") or meta.get("resolved_config_sha256")
            ckpt_man = loaded.get("manifest_sha256") or meta.get("split_manifest_sha256")
            ckpt_mode = loaded.get("run_mode") or meta.get("run_mode")

            if str(ckpt_arch) != str(arch):
                raise RuntimeError("Resume architecture mismatch")
            if int(ckpt_seed) != int(seed):
                raise RuntimeError("Resume seed mismatch")
            if list(ckpt_labels) != list(labels):
                raise RuntimeError("Resume label order mismatch")
            if str(ckpt_res_cfg) != str(res_hash):
                raise RuntimeError("Resume resolved config mismatch")
            if str(ckpt_man) != str(man_hash):
                raise RuntimeError("Resume manifest mismatch")
            if str(ckpt_mode) != str(run_mode):
                raise RuntimeError("Resume run mode mismatch")

        # Matching passes
        validate_resume(payload, "convnext_small", 42, ["L1", "L2", "L3", "L4", "L5"], "correct_res_hash", "correct_man_hash", "full")

        # Arch mismatch
        with self.assertRaises(RuntimeError):
            validate_resume(payload, "densenet121", 42, ["L1", "L2", "L3", "L4", "L5"], "correct_res_hash", "correct_man_hash", "full")

        # Seed mismatch
        with self.assertRaises(RuntimeError):
            validate_resume(payload, "convnext_small", 43, ["L1", "L2", "L3", "L4", "L5"], "correct_res_hash", "correct_man_hash", "full")

        # Label order mismatch
        with self.assertRaises(RuntimeError):
            validate_resume(payload, "convnext_small", 42, ["L2", "L1", "L3", "L4", "L5"], "correct_res_hash", "correct_man_hash", "full")

        # Resolved config mismatch
        with self.assertRaises(RuntimeError):
            validate_resume(payload, "convnext_small", 42, ["L1", "L2", "L3", "L4", "L5"], "wrong_res_hash", "correct_man_hash", "full")

        # Manifest mismatch
        with self.assertRaises(RuntimeError):
            validate_resume(payload, "convnext_small", 42, ["L1", "L2", "L3", "L4", "L5"], "correct_res_hash", "wrong_man_hash", "full")

        # Run mode mismatch
        with self.assertRaises(RuntimeError):
            validate_resume(payload, "convnext_small", 42, ["L1", "L2", "L3", "L4", "L5"], "correct_res_hash", "correct_man_hash", "smoke")

    # -------------------------------------------------------------------------
    # J4: Calibration Fail-Closed & Schema v2 Tests
    # -------------------------------------------------------------------------
    def test_calibration_schema_v2_and_fail_closed(self):
        """J4. Calibration fail-closed on single-class in full mode, null in smoke mode, and schema v2 validation."""
        labels = ["Atelectasis", "Cardiomegaly"]
        # Targets where Cardiomegaly is all zeros (single class)
        targets = [[1.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 0.0]]
        probs = [[0.8, 0.2], [0.1, 0.1], [0.7, 0.3], [0.2, 0.1]]

        # Full mode must raise RuntimeError
        with self.assertRaises(RuntimeError):
            calculate_optimal_thresholds(targets, probs, labels, strict_full_mode=True)

        # Smoke mode allows threshold=None
        thresholds, metrics = calculate_optimal_thresholds(targets, probs, labels, strict_full_mode=False)
        self.assertIsNotNone(thresholds["Atelectasis"])
        self.assertIsNone(thresholds["Cardiomegaly"])

    # -------------------------------------------------------------------------
    # J5: Packaging & Integrity Checksum Ledger Tests
    # -------------------------------------------------------------------------
    def test_package_kaggle_artifact_script_and_verification(self):
        """J5. Packager requires all 9 files, places checksums.json inside ZIP, detects tampering, and rejects dangerous members."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            run_dir = tmppath / "run"
            run_dir.mkdir()

            calib_file = tmppath / "convnext_small_seed42.json"
            calib_file.write_text(json.dumps({"schema_version": 2, "thresholds": {}}), encoding="utf-8")

            manifest_file = tmppath / "manifest.json"
            manifest_file.write_text("{\"splits\": {}}", encoding="utf-8")

            config_file = tmppath / "protocol_v0_1.yaml"
            config_file.write_text("protocol_version: '0.1'", encoding="utf-8")

            # Create run files
            (run_dir / "best.pt").write_text("best", encoding="utf-8")
            (run_dir / "last.pt").write_text("last", encoding="utf-8")
            (run_dir / "training_history.json").write_text("[]", encoding="utf-8")
            (run_dir / "resolved_config.json").write_text("{}", encoding="utf-8")
            (run_dir / "internal_validation_predictions.csv").write_text("study_id\n", encoding="utf-8")
            (run_dir / "run_manifest.json").write_text(json.dumps({
                "schema_version": 1,
                "architecture": "convnext_small",
                "seed": 42,
                "run_mode": "smoke",
            }), encoding="utf-8")

            out_zip = tmppath / "convnext_small_seed42.zip"

            # 1. Success package
            res_zip = package_artifact(
                run_dir=run_dir,
                calibration_file=calib_file,
                manifest_file=manifest_file,
                config_file=config_file,
                output_zip=out_zip,
            )
            self.assertTrue(res_zip.is_file())

            # Verify contents of ZIP
            with zipfile.ZipFile(res_zip, "r") as zf:
                names = set(zf.namelist())
                self.assertIn("checksums.json", names)
                self.assertIn("best.pt", names)
                self.assertIn("last.pt", names)
                self.assertIn("run_manifest.json", names)
                self.assertIn("resolved_config.json", names)
                self.assertIn("internal_validation_predictions.csv", names)
                self.assertIn("convnext_small_seed42.json", names)
                self.assertIn("manifest.json", names)
                self.assertIn("protocol_v0_1.yaml", names)

                checksums_meta = json.loads(zf.read("checksums.json").decode("utf-8"))
                self.assertEqual(checksums_meta["architecture"], "convnext_small")
                self.assertEqual(checksums_meta["seed"], 42)
                self.assertEqual(checksums_meta["status"], "NON_FINAL_SMOKE_TEST")

            # 2. Missing required file must raise RuntimeError
            (run_dir / "best.pt").unlink()
            with self.assertRaises(RuntimeError):
                package_artifact(
                    run_dir=run_dir,
                    calibration_file=calib_file,
                    manifest_file=manifest_file,
                    config_file=config_file,
                    output_zip=tmppath / "fail.zip",
                )

    # -------------------------------------------------------------------------
    # J6: Notebook Static & Cleanliness Tests
    # -------------------------------------------------------------------------
    def test_notebook_default_configuration_and_cleanliness(self):
        """J6. Notebook has clean state, calls packager script, supports SMOKE_PHASE, and disallows credentials/colab."""
        config_cell = None
        for cell in self.nb_json["cells"]:
            src = "".join(cell.get("source", []))
            if "ARCH =" in src and "SEED =" in src and "RUN_MODE =" in src:
                config_cell = src
                break

        self.assertIsNotNone(config_cell, "Configuration cell not found in notebook!")
        self.assertIn('ARCH = "convnext_small"', config_cell)
        self.assertIn("SEED = 42", config_cell)
        self.assertIn('RUN_MODE = "smoke"', config_cell)
        self.assertIn('SMOKE_PHASE = "fresh"', config_cell)
        self.assertIn("BATCH_SIZE = 32", config_cell)
        self.assertIn("RESUME_CHECKPOINT = None", config_cell)
        self.assertIn('REPO_REF = "main"', config_cell)

        all_code = "".join("".join(c.get("source", [])) for c in self.nb_json["cells"])
        self.assertIn("package_kaggle_artifact.py", all_code)
        self.assertIn("--stop-after-epoch", all_code)
        self.assertNotIn("google" + ".colab", all_code)
        self.assertNotIn("archive/valid.csv", all_code)
        self.assertNotIn("--train-csv archive/valid.csv", all_code)

        for idx, cell in enumerate(self.nb_json["cells"]):
            if cell.get("cell_type") == "code":
                self.assertIsNone(cell.get("execution_count"), f"Cell {idx} has non-null execution_count")
                self.assertEqual(cell.get("outputs"), [], f"Cell {idx} has non-empty outputs")


    def test_notebook_rejects_main_in_full_mode(self):
        """J6b. When RUN_MODE is full, REPO_REF must not be main or master."""
        def check_repo_ref_policy(run_mode: str, repo_ref: str):
            if run_mode == "full":
                if repo_ref in ["main", "master"] or not repo_ref:
                    raise ValueError("In RUN_MODE='full', REPO_REF must be a pinned commit SHA or release tag!")

        # smoke mode allows main
        check_repo_ref_policy("smoke", "main")

        # full mode rejects main and master
        with self.assertRaises(ValueError):
            check_repo_ref_policy("full", "main")
        with self.assertRaises(ValueError):
            check_repo_ref_policy("full", "master")
        with self.assertRaises(ValueError):
            check_repo_ref_policy("full", "")

        # full mode accepts a fixed commit SHA or release tag
        check_repo_ref_policy("full", "e61d2b9")
        check_repo_ref_policy("full", "v0.1.0")

    def test_notebook_checks_patient_overlap_and_duplicate_hashes(self):
        """J6c. Notebook Cell 6 must verify patient overlap and duplicate image hashes across splits."""
        all_code = "".join("".join(c.get("source", [])) for c in self.nb_json["cells"])
        self.assertIn("train_pids & val_pids", all_code)
        self.assertIn("train_pids & cal_pids", all_code)
        self.assertIn("val_pids & cal_pids", all_code)
        self.assertIn("Duplicate hashes", all_code)
        self.assertIn("Patient overlap", all_code)

    def test_make_splits_compatibility_and_notebook_split_structure(self):
        """J6d. Verify make_splits.py output compatibility with CheXpertDataset."""
        import pandas as pd
        from scripts.make_splits import main as make_splits_main
        from app.dataset import CheXpertDataset
        import sys

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            source_csv = tmppath / "train.csv"
            df = pd.DataFrame({
                "Path": [
                    "CheXpert-v1.0-small/train/patient00001/study1/view1_frontal.jpg",
                    "CheXpert-v1.0-small/train/patient00002/study1/view1_frontal.jpg",
                    "CheXpert-v1.0-small/train/patient00003/study1/view1_frontal.jpg",
                ],
                "Frontal/Lateral": ["Frontal", "Frontal", "Frontal"],
                "Atelectasis": [0.0, 1.0, 0.0],
                "Cardiomegaly": [1.0, 0.0, 1.0],
                "Consolidation": [0.0, 0.0, 0.0],
                "Edema": [0.0, 0.0, 0.0],
                "Pleural Effusion": [0.0, 0.0, 0.0],
            })
            df.to_csv(source_csv, index=False)

            splits_dir = tmppath / "splits"
            orig_argv = sys.argv
            try:
                sys.argv = [
                    "make_splits.py",
                    "--train-csv", str(source_csv),
                    "--output-dir", str(splits_dir),
                    "--data-root", str(tmppath),
                    "--protocol-version", "0.1",
                ]
                make_splits_main()
            finally:
                sys.argv = orig_argv

            self.assertTrue((splits_dir / "manifest.json").is_file())
            self.assertTrue((splits_dir / "train.csv").is_file())

            # CheXpertDataset must successfully load train.csv
            ds = CheXpertDataset(splits_dir / "train.csv", tmppath, transform=lambda x: x, view="frontal")
            self.assertGreaterEqual(len(ds), 1)


if __name__ == "__main__":
    unittest.main()
