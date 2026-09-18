from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from app.experiment_integrity import canonical_json_sha256, compute_file_sha256
from scripts.package_resume_bundle import package_resume_bundle
from scripts.train import (
    checkpoint_payload,
    load_resume_checkpoint,
    restore_training_state,
    run_epoch,
    save_training_checkpoints,
    set_seed,
)
from scripts.verify_resume_bundle import install_verified_best_checkpoint, verify_resume_bundle


class ResumeBundleBestTests(unittest.TestCase):
    def _files(self, root: Path) -> tuple[Path, Path, Path]:
        resolved = root / "resolved_config.json"
        manifest = root / "manifest.json"
        protocol = root / "protocol_v0_1.yaml"
        resolved.write_text(json.dumps({"seed": 42}), encoding="utf-8")
        manifest.write_text(json.dumps({"seed": 42}), encoding="utf-8")
        protocol.write_text("protocol_version: '0.1'\n", encoding="utf-8")
        return resolved, manifest, protocol

    def _checkpoint(
        self,
        root: Path,
        *,
        run_mode: str = "smoke",
        epoch: int = 1,
        best_epoch: int = 1,
        patience_counter: int = 0,
        model: torch.nn.Module | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: object | None = None,
        generator: torch.Generator | None = None,
        history: list[dict[str, object]] | None = None,
    ) -> dict[str, object]:
        resolved, manifest, protocol = self._files(root)
        model = model or torch.nn.Linear(2, 1)
        optimizer = optimizer or torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = scheduler or torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        generator = generator or torch.Generator().manual_seed(42)
        return checkpoint_payload(
            model,
            optimizer,
            scheduler,
            None,
            epoch,
            2,
            0.9,
            "tiny",
            42,
            compute_file_sha256(protocol),
            canonical_json_sha256(json.loads(resolved.read_text(encoding="utf-8"))),
            compute_file_sha256(manifest),
            ["L"],
            run_mode,
            generator=generator,
            git_commit="a" * 40,
            git_dirty=False if run_mode == "full" else True,
            history=history or [{"epoch": 1, "mean_val_auc": 0.9}],
            best_epoch=best_epoch,
            patience_counter=patience_counter,
        )

    def _bundle(self, root: Path, checkpoint: dict[str, object]) -> Path:
        resolved, manifest, protocol = self._files(root)
        best = root / "best.pt"
        last = root / "last.pt"
        torch.save(checkpoint, best)
        torch.save(checkpoint, last)
        return package_resume_bundle(best, last, resolved, manifest, protocol, root / "resume.zip")

    def test_full_checkpoint_bundle_packages_and_verifies(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = self._bundle(root, self._checkpoint(root, run_mode="full"))
            ledger = verify_resume_bundle(bundle, expected_sha256=compute_file_sha256(bundle))
            self.assertEqual(ledger["status"], "COMPLIANT_PROTOCOL_RESUME_BUNDLE")
            self.assertEqual(ledger["run_mode"], "full")

    def test_missing_best_and_best_last_mismatch_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = self._checkpoint(root)
            bundle = self._bundle(root, checkpoint)
            missing = root / "missing-best.zip"
            with zipfile.ZipFile(bundle) as source, zipfile.ZipFile(missing, "w") as target:
                for name in source.namelist():
                    if name != "best.pt":
                        target.writestr(name, source.read(name))
            with self.assertRaisesRegex(RuntimeError, "members mismatch"):
                verify_resume_bundle(missing, expected_sha256=compute_file_sha256(missing))

            best = root / "mismatch-best.pt"
            last = root / "mismatch-last.pt"
            mismatched = dict(checkpoint)
            mismatched["seed"] = 43
            torch.save(mismatched, best)
            torch.save(checkpoint, last)
            resolved, manifest, protocol = self._files(root)
            with self.assertRaisesRegex(RuntimeError, "metadata mismatch"):
                package_resume_bundle(best, last, resolved, manifest, protocol, root / "bad.zip")

    def test_resume_worse_epoch_preserves_verified_best(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            set_seed(42)
            model = torch.nn.Linear(2, 1)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
            generator = torch.Generator().manual_seed(42)
            first = self._checkpoint(
                source,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                generator=generator,
            )
            save_training_checkpoints(first, source, is_best=True)
            resolved, manifest, protocol = self._files(source)
            bundle = package_resume_bundle(
                source / "best.pt",
                source / "last.pt",
                resolved,
                manifest,
                protocol,
                root / "resume.zip",
            )

            extracted = root / "extracted"
            ledger = verify_resume_bundle(
                bundle,
                extracted,
                expected_sha256=compute_file_sha256(bundle),
            )
            resumed_dir = root / "resumed"
            best_target, last_source, expected_best, expected_last = install_verified_best_checkpoint(
                extracted, resumed_dir, ledger
            )
            self.assertEqual(compute_file_sha256(best_target), expected_best)
            self.assertEqual(compute_file_sha256(last_source), expected_last)

            resumed_model = torch.nn.Linear(2, 1)
            resumed_optimizer = torch.optim.SGD(resumed_model.parameters(), lr=0.1)
            resumed_scheduler = torch.optim.lr_scheduler.StepLR(resumed_optimizer, step_size=1)
            resumed_generator = torch.Generator().manual_seed(999)
            loaded, _ = load_resume_checkpoint(last_source, expected_last)
            next_epoch, best_auc, best_epoch, patience, history = restore_training_state(
                loaded,
                resumed_model,
                resumed_optimizer,
                resumed_scheduler,
                generator=resumed_generator,
            )
            dataset = TensorDataset(
                torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
                torch.tensor([[1.0], [0.0]]),
                torch.ones(2, 1),
            )
            loader = DataLoader(dataset, batch_size=2, shuffle=True, generator=resumed_generator)
            run_epoch(
                resumed_model,
                loader,
                torch.nn.BCEWithLogitsLoss(reduction="none"),
                torch.device("cpu"),
                resumed_optimizer,
            )
            resumed_scheduler.step()
            history.append({"epoch": next_epoch, "mean_val_auc": 0.8})
            second = checkpoint_payload(
                resumed_model,
                resumed_optimizer,
                resumed_scheduler,
                None,
                next_epoch,
                2,
                best_auc,
                "tiny",
                42,
                loaded["config_sha256"],
                loaded["resolved_config_sha256"],
                loaded["manifest_sha256"],
                ["L"],
                "smoke",
                generator=resumed_generator,
                git_commit="a" * 40,
                git_dirty=True,
                history=history,
                best_epoch=best_epoch,
                patience_counter=patience + 1,
            )
            save_training_checkpoints(second, resumed_dir, is_best=False)

            preserved = torch.load(resumed_dir / "best.pt", weights_only=True)
            last = torch.load(resumed_dir / "last.pt", weights_only=True)
            self.assertEqual(compute_file_sha256(resumed_dir / "best.pt"), expected_best)
            self.assertEqual(preserved["best_epoch"], 1)
            self.assertEqual(last["best_epoch"], 1)
            self.assertEqual(last["epoch"], 2)
            for key, value in first["model_state"].items():
                torch.testing.assert_close(preserved["model_state"][key], value, rtol=0, atol=0)

    def test_notebook_only_resumes_from_verified_bundle(self) -> None:
        code = (Path(__file__).parents[1] / "train_on_kaggle.ipynb").read_text(encoding="utf-8")
        self.assertNotIn("RESUME_CHECKPOINT", code)
        self.assertNotIn("RESUME_CHECKSUMS_JSON", code)
        self.assertIn("install_verified_best_checkpoint", code)
        self.assertIn("resume_target", code)
        self.assertIn("expected_best_sha256", code)


if __name__ == "__main__":
    unittest.main()
