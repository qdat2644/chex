from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path

import pandas as pd
import torch

from app.experiment_integrity import canonical_json_sha256, compute_file_sha256
from scripts.kaggle_preflight import (
    resolve_unique_data_root,
    select_train_csv,
    validate_development_source,
    validate_git_integrity,
    validate_split_image_hashes,
    verify_expected_source_csv,
)
from scripts.package_kaggle_artifact import package_artifact
from scripts.package_resume_bundle import package_resume_bundle
from scripts.train import load_resume_checkpoint
from scripts.verify_resume_bundle import EXPECTED_MEMBERS, verify_resume_bundle


class TestProductionHardening(unittest.TestCase):
    def _checkpoint(self, run_mode: str = "smoke") -> dict[str, object]:
        return {
            "checkpoint_schema_version": 2,
            "architecture": "convnext_small",
            "seed": 42,
            "run_mode": run_mode,
            "protocol_version": "0.1",
            "git_commit": "abc123",
            "git_dirty": True,
            "labels": ["L1"],
            "resolved_config_sha256": "pending",
            "manifest_sha256": "pending",
            "model_state": {"weight": torch.tensor([1.0])},
        }

    def test_resume_requires_expected_checksum_and_rejects_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "last.pt"
            torch.save(self._checkpoint(), checkpoint)
            with self.assertRaises(RuntimeError):
                load_resume_checkpoint(checkpoint, None)
            with self.assertRaises(RuntimeError):
                load_resume_checkpoint(checkpoint, "0" * 64)
            actual = compute_file_sha256(checkpoint)
            loaded, returned = load_resume_checkpoint(checkpoint, actual)
            self.assertEqual(loaded["seed"], 42)
            self.assertEqual(returned, actual)

    def test_resume_bundle_complete_and_tampering_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "last.pt"
            resolved = root / "resolved_config.json"
            resolved.write_text("{}", encoding="utf-8")
            manifest = root / "manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            protocol = root / "protocol_v0_1.yaml"
            protocol.write_text("protocol_version: '0.1'", encoding="utf-8")
            checkpoint_payload = self._checkpoint()
            checkpoint_payload["resolved_config_sha256"] = canonical_json_sha256({})
            checkpoint_payload["manifest_sha256"] = compute_file_sha256(manifest)
            checkpoint_payload["config_sha256"] = compute_file_sha256(protocol)
            torch.save(checkpoint_payload, checkpoint)
            bundle = package_resume_bundle(checkpoint, resolved, manifest, protocol, root / "resume.zip")
            with zipfile.ZipFile(bundle) as archive:
                self.assertEqual(set(archive.namelist()), EXPECTED_MEMBERS)
            verify_resume_bundle(bundle, root / "verified")

            tampered = root / "tampered.zip"
            with zipfile.ZipFile(bundle) as source, zipfile.ZipFile(tampered, "w") as target:
                for name in source.namelist():
                    data = source.read(name)
                    target.writestr(name, data + b"x" if name == "last.pt" else data)
            with self.assertRaises(RuntimeError):
                verify_resume_bundle(tampered)

    def test_notebook_has_no_assert_in_code_cells(self) -> None:
        notebook = json.loads((Path(__file__).parents[1] / "train_on_kaggle.ipynb").read_text(encoding="utf-8"))
        code = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"] if cell["cell_type"] == "code")
        self.assertNotIn("assert ", code)
        self.assertIn("scripts/verify_resume_bundle.py", code)
        self.assertIn("--expected-resume-sha256", code)

    def test_image_hash_validation_fails_closed(self) -> None:
        valid = "a" * 64
        frames = {"train": pd.DataFrame({"image_sha256": [valid]}), "validation": pd.DataFrame({"image_sha256": ["b" * 64]})}
        self.assertEqual(validate_split_image_hashes(frames), 0)
        with self.assertRaises(RuntimeError):
            validate_split_image_hashes({"train": pd.DataFrame({"Path": ["x"]})})
        for invalid in ("", "NOT_FOUND"):
            with self.assertRaises(RuntimeError):
                validate_split_image_hashes({"train": pd.DataFrame({"image_sha256": [invalid]})})

    def test_locked_and_non_training_sources_are_rejected(self) -> None:
        safe_root = Path("/dataset")
        safe_csv = Path("/dataset/train.csv")
        bad_cases = [
            (Path("/dataset/locked_test"), safe_csv, ["train/patient/a.jpg"], {}, {}),
            (safe_root, Path("/dataset/locked-test/train.csv"), ["train/patient/a.jpg"], {}, {}),
            (safe_root, safe_csv, ["valid/patient/a.jpg"], {}, {}),
            (safe_root, safe_csv, ["test/patient/a.jpg"], {}, {}),
            (safe_root, safe_csv, ["images/patient/a.jpg"], {}, {}),
            (safe_root, safe_csv, ["train/patient/a.jpg"], {"role": "locked_test"}, {}),
            (safe_root, safe_csv, ["train/patient/a.jpg"], {"locked": True}, {}),
            (safe_root, safe_csv, ["train/patient/a.jpg"], {}, {"splits": {"locked_test": {}}}),
        ]
        for args in bad_cases:
            with self.subTest(args=args), self.assertRaises(RuntimeError):
                validate_development_source(*args)

    def test_ambiguous_train_csv_and_data_root_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a").mkdir()
            (root / "b").mkdir()
            (root / "a" / "train.csv").write_text("Path\n", encoding="utf-8")
            (root / "b" / "train.csv").write_text("Path\n", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                select_train_csv(root)

            source = root / "source.csv"
            paths = [f"train/patient{i:05d}/image.jpg" for i in range(50)]
            pd.DataFrame({"Path": paths}).to_csv(source, index=False)
            roots = [root / "images1", root / "images2"]
            for candidate in roots:
                for relative in paths:
                    target = candidate / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(b"image")
            with self.assertRaises(RuntimeError):
                resolve_unique_data_root(source, roots)

    def test_dirty_git_is_rejected_in_full_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            subprocess.check_call(["git", "init", "-q"], cwd=repo)
            subprocess.check_call(["git", "config", "user.email", "test@example.com"], cwd=repo)
            subprocess.check_call(["git", "config", "user.name", "Test"], cwd=repo)
            (repo / "tracked.txt").write_text("clean", encoding="utf-8")
            subprocess.check_call(["git", "add", "tracked.txt"], cwd=repo)
            subprocess.check_call(["git", "commit", "-qm", "initial"], cwd=repo)
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
            (repo / "tracked.txt").write_text("dirty", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                validate_git_integrity(repo, commit, "full")
            self.assertTrue(validate_git_integrity(repo, "main", "smoke")["git_dirty"])

    def test_full_source_csv_requires_preregistered_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "train.csv"
            source.write_text("Path\ntrain/patient/a.jpg\n", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                verify_expected_source_csv(source, "full", None)
            with self.assertRaises(RuntimeError):
                verify_expected_source_csv(source, "full", "0" * 64)
            expected = hashlib.sha256(source.read_bytes()).hexdigest()
            self.assertEqual(verify_expected_source_csv(source, "full", expected), expected)

    def _artifact_fixture(self, root: Path) -> tuple[Path, Path, Path, Path]:
        run_dir = root / "run"
        run_dir.mkdir()
        labels = ["L1"]
        manifest = {
            "protocol_version": "0.1", "seed": 42, "labels": labels,
            "source_csv": "/dataset/train.csv", "splits": {"train": {"role": "training"}},
        }
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        manifest_hash = compute_file_sha256(manifest_path)
        resolved = {
            "schema_version": 1, "protocol_version": "0.1", "architecture": "convnext_small",
            "seed": 42, "labels": labels, "data": {"split_manifest_sha256": manifest_hash},
        }
        resolved_path = run_dir / "resolved_config.json"
        resolved_path.write_text(json.dumps(resolved), encoding="utf-8")
        resolved_hash = canonical_json_sha256(resolved)
        checkpoint = self._checkpoint()
        checkpoint["resolved_config_sha256"] = resolved_hash
        checkpoint["manifest_sha256"] = manifest_hash
        best = run_dir / "best.pt"
        last = run_dir / "last.pt"
        torch.save(checkpoint, best)
        torch.save(checkpoint, last)
        predictions = run_dir / "internal_validation_predictions.csv"
        predictions.write_text("study_id\ns1\n", encoding="utf-8")
        (run_dir / "training_history.json").write_text("[]", encoding="utf-8")
        run_manifest = {
            "protocol_version": "0.1", "architecture": "convnext_small", "seed": 42,
            "run_mode": "smoke", "labels": labels, "git_commit": "abc123", "git_dirty": True,
            "resolved_config_sha256": resolved_hash, "manifest_sha256": manifest_hash,
            "best_checkpoint_sha256": compute_file_sha256(best),
            "last_checkpoint_sha256": compute_file_sha256(last),
            "predictions_sha256": compute_file_sha256(predictions),
        }
        (run_dir / "run_manifest.json").write_text(json.dumps(run_manifest), encoding="utf-8")
        calibration = {
            "protocol_version": "0.1", "architecture": "convnext_small", "seed": 42,
            "run_mode": "smoke", "labels": labels, "git_commit": "abc123",
            "resolved_config_sha256": resolved_hash, "manifest_sha256": manifest_hash,
            "checkpoint_sha256": compute_file_sha256(best),
        }
        calibration_path = root / "calibration.json"
        calibration_path.write_text(json.dumps(calibration), encoding="utf-8")
        config_path = root / "protocol_v0_1.yaml"
        config_path.write_text("protocol_version: '0.1'\n", encoding="utf-8")
        return run_dir, calibration_path, manifest_path, config_path

    def test_packager_expected_values_cannot_override_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self._artifact_fixture(root)
            output = package_artifact(*fixture, root / "artifact.zip", "convnext_small", 42, "smoke")
            self.assertTrue(output.is_file())
            with self.assertRaises(RuntimeError):
                package_artifact(*fixture, root / "bad.zip", "densenet121", 42, "smoke")
            with self.assertRaises(RuntimeError):
                package_artifact(*fixture, root / "bad-seed.zip", "convnext_small", 43, "smoke")
            with self.assertRaises(RuntimeError):
                package_artifact(*fixture, root / "bad-mode.zip", "convnext_small", 42, "full")
            run_manifest = json.loads((fixture[0] / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(run_manifest["architecture"], "convnext_small")


if __name__ == "__main__":
    unittest.main()
