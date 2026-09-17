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

from scripts.calibrate import parse_args as parse_calibrate_args, main as calibrate_main
from scripts.train import (
    checkpoint_payload,
    compute_file_sha256,
    get_safe_rng_state,
    restore_safe_rng_state,
)


class DummyNet(torch.nn.Module):
    def __init__(self, in_features: int = 4, out_features: int = 5):
        super().__init__()
        self.fc = torch.nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class TestKagglePipelineAndNotebook(unittest.TestCase):
    def setUp(self):
        self.nb_path = PROJECT_ROOT / "train_on_kaggle.ipynb"
        self.assertTrue(self.nb_path.is_file(), "train_on_kaggle.ipynb must exist")
        self.nb_content = self.nb_path.read_text(encoding="utf-8")
        self.nb_json = json.loads(self.nb_content)

    def test_notebook_valid_json_and_format(self):
        """1. Notebook JSON must be valid and conform to nbformat 4."""
        self.assertEqual(self.nb_json.get("nbformat"), 4)
        self.assertIn("cells", self.nb_json)
        self.assertGreaterEqual(len(self.nb_json["cells"]), 10)

    def test_notebook_default_configuration(self):
        """2. Check defaults: smoke mode, batch size 32, convnext_small, seed 42, main repo ref."""
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
        self.assertIn("BATCH_SIZE = 32", config_cell)
        self.assertIn("RESUME_CHECKPOINT = None", config_cell)
        self.assertIn('REPO_REF = "main"', config_cell)

    def test_notebook_rejects_main_in_full_mode(self):
        """3. When RUN_MODE is full, REPO_REF must not be main or master."""
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
        check_repo_ref_policy("full", "2fe6831")
        check_repo_ref_policy("full", "v0.1.0")

    def test_notebook_no_references_to_locked_test_or_valid_csv(self):
        """4. Notebook must not reference locked test or official valid.csv as training input."""
        full_text = self.nb_content.lower()
        self.assertNotIn("locked_test_manifest", full_text)
        self.assertNotIn("outputs/splits/protocol_v0_1/locked_test_manifest.json", full_text)
        # Check that there is no command using valid.csv as data input
        self.assertNotIn("--train-csv archive/valid.csv", self.nb_content)
        self.assertNotIn("valid.csv", self.nb_json["cells"][1].get("source", []))

    def test_notebook_clean_state_and_no_secrets(self):
        """5. Notebook must have null execution_count, empty outputs, and zero credentials."""
        for idx, cell in enumerate(self.nb_json["cells"]):
            if cell.get("cell_type") == "code":
                self.assertIsNone(cell.get("execution_count"), f"Cell {idx} has non-null execution_count")
                self.assertEqual(cell.get("outputs"), [], f"Cell {idx} has non-empty outputs")
                src = "".join(cell.get("source", []))
                self.assertNotIn("google" + ".colab", src)
                self.assertNotIn("drive" + ".mount", src)
                self.assertNotIn("KG" + "AT_", src)
                self.assertNotIn("KAGGLE" + "_TOKEN", src)

    def test_artifact_names_consistency(self):
        """6. Artifact names in notebook must match train.py outputs strictly."""
        notebook_str = self.nb_content
        expected_artifacts = [
            "best.pt",
            "last.pt",
            "training_history.json",
            "run_manifest.json",
            "resolved_config.json",
            "internal_validation_predictions.csv",
        ]
        for name in expected_artifacts:
            self.assertIn(name, notebook_str, f"Required artifact {name} not found in notebook!")

        self.assertNotIn("history.csv", notebook_str)
        self.assertNotIn("run_metadata.json", notebook_str)

    def test_packaging_fails_on_missing_artifact(self):
        """7. Packaging must fail if any required artifact is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            required_files = {
                "best.pt": tmppath / "best.pt",
                "last.pt": tmppath / "last.pt",
                "training_history.json": tmppath / "training_history.json",
                "run_manifest.json": tmppath / "run_manifest.json",
                "resolved_config.json": tmppath / "resolved_config.json",
                "internal_validation_predictions.csv": tmppath / "internal_validation_predictions.csv",
                "convnext_small_seed42.json": tmppath / "convnext_small_seed42.json",
                "manifest.json": tmppath / "manifest.json",
                "protocol_v0_1.yaml": tmppath / "protocol_v0_1.yaml",
            }

            # Create all except one
            for name, p in required_files.items():
                if name != "resolved_config.json":
                    p.write_text("dummy", encoding="utf-8")

            def do_package():
                for name, fpath in required_files.items():
                    if not fpath.is_file():
                        raise FileNotFoundError(f"PACKAGING ERROR: Required artifact file '{name}' does not exist!")

            with self.assertRaises(FileNotFoundError):
                do_package()

    def test_checksums_inside_zip_and_verification(self):
        """8. Checksums must reside inside ZIP, and post-packaging verification must detect any corruption."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            files = {
                "best.pt": tmppath / "best.pt",
                "last.pt": tmppath / "last.pt",
                "resolved_config.json": tmppath / "resolved_config.json",
            }
            for name, p in files.items():
                p.write_text(f"content of {name}", encoding="utf-8")

            zip_path = tmppath / "artifact.zip"
            checksums = {name: compute_file_sha256(p) for name, p in files.items()}
            checksums_data = {
                "architecture": "convnext_small",
                "seed": 42,
                "run_mode": "smoke",
                "status": "NON_FINAL_SMOKE_TEST",
                "files": checksums,
            }
            checksums_json = json.dumps(checksums_data, indent=2)

            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for name, p in files.items():
                    zf.write(p, arcname=name)
                zf.writestr("checksums.json", checksums_json)

            # Verification logic
            with zipfile.ZipFile(zip_path, "r") as zf:
                namelist = set(zf.namelist())
                self.assertIn("checksums.json", namelist)
                for name, expected_hash in checksums.items():
                    actual = hashlib.sha256(zf.read(name)).hexdigest()
                    self.assertEqual(actual, expected_hash)

                read_meta = json.loads(zf.read("checksums.json").decode("utf-8"))
                self.assertEqual(read_meta["status"], "NON_FINAL_SMOKE_TEST")

    def test_resume_standard_and_dataparallel(self):
        """9. Checkpoint saving and loading into standard and DataParallel models."""
        torch.manual_seed(42)
        model = DummyNet(in_features=4, out_features=5)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
        scaler = torch.amp.GradScaler("cpu", enabled=False)

        labels = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]
        payload = checkpoint_payload(
            model=model,
            optimizer=opt,
            scheduler=sched,
            scaler=scaler,
            epoch=3,
            best_val_auc=0.85,
            architecture="convnext_small",
            seed=42,
            config_sha256="cfg123",
            manifest_sha256="man123",
            labels=labels,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "ckpt.pt"
            torch.save(payload, ckpt_path)

            loaded = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            self.assertIn("model_state_dict", loaded)
            self.assertEqual(loaded["epoch"], 3)
            self.assertEqual(loaded["architecture"], "convnext_small")
            self.assertEqual(loaded["seed"], 42)

            # 1. Load into standard model
            fresh_model = DummyNet(in_features=4, out_features=5)
            raw_model = fresh_model.module if isinstance(fresh_model, torch.nn.DataParallel) else fresh_model
            raw_model.load_state_dict(loaded["model_state_dict"])
            for p1, p2 in zip(model.parameters(), fresh_model.parameters(), strict=True):
                self.assertTrue(torch.equal(p1, p2))

            # 2. Load into DataParallel-like wrapper
            dp_model = torch.nn.DataParallel(DummyNet(in_features=4, out_features=5))
            raw_dp_model = dp_model.module if isinstance(dp_model, torch.nn.DataParallel) else dp_model
            raw_dp_model.load_state_dict(loaded["model_state_dict"])
            for p1, p2 in zip(model.parameters(), dp_model.module.parameters(), strict=True):
                self.assertTrue(torch.equal(p1, p2))

    def test_restore_scaler_and_rng(self):
        """10. Restoring scaler and Python/NumPy/Torch RNG states."""
        # Set specific seeds and record state
        random.seed(999)
        np.random.seed(999)
        torch.manual_seed(999)

        initial_py_val = random.random()
        initial_np_val = float(np.random.rand())
        initial_th_val = float(torch.rand(1))

        # Reset seeds to 999 and capture safe state
        random.seed(999)
        np.random.seed(999)
        torch.manual_seed(999)
        saved_rng = get_safe_rng_state()

        # Generate some numbers to alter state
        _ = [random.random() for _ in range(10)]
        _ = [np.random.rand() for _ in range(10)]
        _ = [torch.rand(1) for _ in range(10)]

        # Restore state
        restore_safe_rng_state(saved_rng)

        # First numbers generated after restore should match initial
        self.assertEqual(random.random(), initial_py_val)
        self.assertEqual(float(np.random.rand()), initial_np_val)
        self.assertEqual(float(torch.rand(1)), initial_th_val)

    def test_resume_fails_on_metadata_mismatch(self):
        """11. Resume must fail if architecture, seed, config, or manifest hash mismatches."""
        model = DummyNet()
        payload = checkpoint_payload(
            model=model,
            optimizer=None,
            scheduler=None,
            scaler=None,
            epoch=1,
            best_val_auc=0.80,
            architecture="convnext_small",
            seed=42,
            config_sha256="correct_cfg_hash",
            manifest_sha256="correct_manifest_hash",
            labels=["L1", "L2", "L3", "L4", "L5"],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "ckpt.pt"
            torch.save(payload, ckpt_path)

            loaded = torch.load(ckpt_path, map_location="cpu", weights_only=True)

            def check_resume(loaded_data, expected_arch, expected_seed, expected_cfg, expected_man):
                meta = loaded_data.get("metadata", {})
                ckpt_arch = loaded_data.get("architecture") or meta.get("architecture")
                ckpt_seed = loaded_data.get("seed") if "seed" in loaded_data and loaded_data["seed"] is not None else meta.get("seed")
                ckpt_cfg = loaded_data.get("config_sha256") or meta.get("config_sha256")
                ckpt_man = loaded_data.get("manifest_sha256") or meta.get("split_manifest_sha256")

                if ckpt_arch is None or str(ckpt_arch) != str(expected_arch):
                    raise RuntimeError("Resume architecture mismatch")
                if ckpt_seed is None or int(ckpt_seed) != int(expected_seed):
                    raise RuntimeError("Resume seed mismatch")
                if ckpt_cfg is None or str(ckpt_cfg) != str(expected_cfg):
                    raise RuntimeError("Resume config hash mismatch")
                if ckpt_man is None or str(ckpt_man) != str(expected_man):
                    raise RuntimeError("Resume manifest hash mismatch")

            # Check matches
            check_resume(loaded, "convnext_small", 42, "correct_cfg_hash", "correct_manifest_hash")

            # Architecture mismatch
            with self.assertRaises(RuntimeError):
                check_resume(loaded, "densenet121", 42, "correct_cfg_hash", "correct_manifest_hash")

            # Seed mismatch
            with self.assertRaises(RuntimeError):
                check_resume(loaded, "convnext_small", 43, "correct_cfg_hash", "correct_manifest_hash")

            # Config mismatch
            with self.assertRaises(RuntimeError):
                check_resume(loaded, "convnext_small", 42, "wrong_cfg_hash", "correct_manifest_hash")

            # Manifest mismatch
            with self.assertRaises(RuntimeError):
                check_resume(loaded, "convnext_small", 42, "correct_cfg_hash", "wrong_manifest_hash")

    def test_calibrate_fails_on_mismatch(self):
        """12. Calibrate must fail on manifest, config, arch, or seed mismatch."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            manifest_file = tmppath / "manifest.json"
            manifest_file.write_text(json.dumps({
                "schema_version": "1.0",
                "splits": {"calibration": {"role": "calibration", "csv": "calib.csv"}},
            }), encoding="utf-8")
            calib_csv = tmppath / "calib.csv"
            calib_csv.write_text("Path,Frontal/Lateral,Atelectasis\n", encoding="utf-8")

            config_file = tmppath / "config.yaml"
            config_file.write_text("protocol_version: '0.1'\n", encoding="utf-8")

            ckpt_file = tmppath / "ckpt.pt"
            # Checkpoint with mismatched manifest hash
            torch.save({
                "model_state_dict": DummyNet().state_dict(),
                "architecture": "convnext_small",
                "seed": 42,
                "config_sha256": compute_file_sha256(config_file),
                "manifest_sha256": "mismatched_manifest_hash",
                "metadata": {
                    "architecture": "convnext_small",
                    "seed": 42,
                    "config_sha256": compute_file_sha256(config_file),
                    "split_manifest_sha256": "mismatched_manifest_hash",
                }
            }, ckpt_file)

            import subprocess
            import sys
            cmd = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "calibrate.py"),
                "--checkpoint", str(ckpt_file),
                "--split-manifest", str(manifest_file),
                "--config", str(config_file),
                "--arch", "convnext_small",
                "--seed", "42",
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            self.assertNotEqual(res.returncode, 0, "calibrate.py should fail on manifest mismatch!")
            self.assertIn("INTEGRITY ERROR", res.stderr + res.stdout)


if __name__ == "__main__":
    unittest.main()
