from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import tempfile
import zipfile
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.experiment_integrity import canonical_json_sha256, compute_file_sha256


EXPECTED_MEMBERS = {
    "best.pt",
    "last.pt",
    "resolved_config.json",
    "manifest.json",
    "protocol_v0_1.yaml",
    "checksums.json",
}

CHECKPOINT_LINK_FIELDS = (
    "architecture",
    "seed",
    "run_mode",
    "labels",
    "resolved_config_sha256",
    "manifest_sha256",
    "config_sha256",
    "git_commit",
)

RESUME_STATE_FIELDS = (
    "epoch",
    "best_epoch",
    "patience_counter",
    "best_val_auc",
    "model_state",
    "optimizer_state",
    "scheduler_state",
    "rng_state",
    "train_loader_generator_state",
)


def _load_checkpoint_bytes(content: bytes, name: str) -> dict[str, object]:
    try:
        checkpoint = torch.load(io.BytesIO(content), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise RuntimeError(f"Resume bundle {name} cannot be parsed") from exc
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"Resume bundle {name} is not a checkpoint dictionary")
    return checkpoint


def validate_checkpoint_pair(best: dict[str, object], last: dict[str, object]) -> dict[str, object]:
    missing: dict[str, list[str]] = {}
    for name, checkpoint in (("best.pt", best), ("last.pt", last)):
        absent = [
            field
            for field in CHECKPOINT_LINK_FIELDS + RESUME_STATE_FIELDS
            if field not in checkpoint or (checkpoint[field] is None and field != "best_val_auc")
        ]
        if absent:
            missing[name] = absent
    if missing:
        raise RuntimeError(f"Resume bundle checkpoints are missing metadata: {missing}")
    mismatched = [field for field in CHECKPOINT_LINK_FIELDS if best[field] != last[field]]
    if mismatched:
        raise RuntimeError(f"Resume bundle best/last metadata mismatch: {mismatched}")
    run_mode = str(last["run_mode"])
    if run_mode not in {"smoke", "full"}:
        raise RuntimeError(f"Unsupported resume bundle run_mode: {run_mode}")
    for name, checkpoint in (("best.pt", best), ("last.pt", last)):
        epoch = checkpoint["epoch"]
        best_epoch = checkpoint["best_epoch"]
        patience_counter = checkpoint["patience_counter"]
        if type(epoch) is not int or epoch < 1:
            raise RuntimeError(f"Resume bundle {name} has invalid epoch")
        if type(best_epoch) is not int or not 1 <= best_epoch <= epoch:
            raise RuntimeError(f"Resume bundle {name} has invalid best_epoch")
        if type(patience_counter) is not int or not 0 <= patience_counter <= epoch:
            raise RuntimeError(f"Resume bundle {name} has invalid patience_counter")
        for field in ("model_state", "optimizer_state", "scheduler_state", "rng_state"):
            if not isinstance(checkpoint[field], dict):
                raise RuntimeError(f"Resume bundle {name} has invalid {field}")
        missing_rng = [
            field
            for field in ("python", "numpy", "torch_cpu", "torch_cuda")
            if field not in checkpoint["rng_state"]
        ]
        if missing_rng:
            raise RuntimeError(f"Resume bundle {name} has incomplete RNG state: {missing_rng}")
        if not isinstance(checkpoint["train_loader_generator_state"], torch.Tensor):
            raise RuntimeError(f"Resume bundle {name} has invalid train_loader_generator_state")
        metadata = checkpoint.get("metadata")
        if not isinstance(metadata, dict) or not isinstance(metadata.get("metrics_history"), list):
            raise RuntimeError(f"Resume bundle {name} is missing metadata.metrics_history")
    if run_mode == "full":
        for name, checkpoint in (("best.pt", best), ("last.pt", last)):
            if checkpoint.get("git_dirty") is not False:
                raise RuntimeError(f"Full resume bundle requires {name} git_dirty=false")
        commit = str(last["git_commit"])
        if commit == "UNKNOWN" or len(commit) < 7 or any(ch not in "0123456789abcdefABCDEF" for ch in commit):
            raise RuntimeError("Full resume bundle requires a valid Git commit SHA")
    return {field: last[field] for field in CHECKPOINT_LINK_FIELDS}


def verify_resume_bundle(path: Path, extract_dir: Path | None = None, *, expected_sha256: str | None = None) -> dict[str, object]:
    bundle = Path(path)
    if not expected_sha256:
        raise RuntimeError("External expected resume bundle SHA-256 is mandatory")
    if not bundle.is_file():
        raise FileNotFoundError(bundle)
    if compute_file_sha256(bundle) != expected_sha256:
        raise RuntimeError("Resume bundle ZIP SHA-256 mismatch")
    try:
        with zipfile.ZipFile(bundle, "r") as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise RuntimeError("Resume bundle contains duplicate ZIP members")
            if set(names) != EXPECTED_MEMBERS:
                raise RuntimeError(
                    f"Resume bundle members mismatch: expected {sorted(EXPECTED_MEMBERS)}, got {sorted(names)}"
                )
            for name in names:
                member = Path(name)
                if member.is_absolute() or ".." in member.parts or len(member.parts) != 1:
                    raise RuntimeError(f"Unsafe resume bundle member: {name}")

            ledger = json.loads(archive.read("checksums.json").decode("utf-8"))
            if ledger.get("schema_version") != 1:
                raise RuntimeError("Unsupported resume checksums schema")
            files = ledger.get("files")
            if not isinstance(files, dict) or set(files) != EXPECTED_MEMBERS - {"checksums.json"}:
                raise RuntimeError("Resume checksums ledger has an invalid file set")
            for name, expected in files.items():
                actual = hashlib.sha256(archive.read(name)).hexdigest()
                if actual != expected:
                    raise RuntimeError(f"Resume bundle checksum mismatch for {name}")

            best_checkpoint = _load_checkpoint_bytes(archive.read("best.pt"), "best.pt")
            checkpoint = _load_checkpoint_bytes(archive.read("last.pt"), "last.pt")
            shared_metadata = validate_checkpoint_pair(best_checkpoint, checkpoint)
            resolved = json.loads(archive.read("resolved_config.json").decode("utf-8"))
            expected_links = {
                "resolved_config_sha256": canonical_json_sha256(resolved),
                "manifest_sha256": files["manifest.json"],
                "config_sha256": files["protocol_v0_1.yaml"],
            }
            for field, actual in expected_links.items():
                expected = checkpoint[field]
                if expected != actual:
                    raise RuntimeError(f"Resume bundle {field} does not match last.pt metadata")
            expected_status = (
                "NON_FINAL_SMOKE_TEST"
                if checkpoint["run_mode"] == "smoke"
                else "COMPLIANT_PROTOCOL_RESUME_BUNDLE"
            )
            for field, expected in {
                "status": expected_status,
                "architecture": shared_metadata["architecture"],
                "seed": shared_metadata["seed"],
                "run_mode": shared_metadata["run_mode"],
            }.items():
                if ledger.get(field) != expected:
                    raise RuntimeError(f"Resume bundle ledger {field} mismatch")

            if extract_dir is not None:
                destination = Path(extract_dir)
                destination.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
                    stage = Path(temporary)
                    for name in EXPECTED_MEMBERS:
                        (stage / name).write_bytes(archive.read(name))
                    for name in EXPECTED_MEMBERS:
                        target = destination / name
                        if target.exists():
                            target.unlink()
                        shutil.move(str(stage / name), str(target))
            return ledger
    except zipfile.BadZipFile as exc:
        raise RuntimeError(f"Invalid resume bundle ZIP: {bundle}") from exc


def install_verified_best_checkpoint(
    extracted_dir: Path,
    output_dir: Path,
    ledger: dict[str, object],
) -> tuple[Path, Path, str, str]:
    """Validate extracted checkpoint bytes against the ledger and atomically install best.pt."""
    files = ledger.get("files")
    if not isinstance(files, dict):
        raise RuntimeError("Resume bundle ledger has no files mapping")
    source_dir = Path(extracted_dir)
    best_source = source_dir / "best.pt"
    last_source = source_dir / "last.pt"
    expected_best = files.get("best.pt")
    expected_last = files.get("last.pt")
    if not isinstance(expected_best, str) or not isinstance(expected_last, str):
        raise RuntimeError("Resume bundle ledger is missing best.pt or last.pt checksum")
    for name, source, expected in (
        ("best.pt", best_source, expected_best),
        ("last.pt", last_source, expected_last),
    ):
        if not source.is_file() or compute_file_sha256(source) != expected:
            raise RuntimeError(f"Extracted resume bundle {name} checksum mismatch")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    best_target = destination / "best.pt"
    with tempfile.NamedTemporaryFile(dir=destination, prefix=".best.pt.", delete=False) as temporary:
        temporary_path = Path(temporary.name)
        with best_source.open("rb") as source:
            shutil.copyfileobj(source, temporary)
    try:
        if compute_file_sha256(temporary_path) != expected_best:
            raise RuntimeError("Copied best.pt checksum mismatch")
        temporary_path.replace(best_target)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return best_target, last_source, expected_best, expected_last


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify and optionally extract a resume bundle")
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--extract-dir", type=Path)
    parser.add_argument("--expected-sha256", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ledger = verify_resume_bundle(args.bundle, args.extract_dir, expected_sha256=args.expected_sha256)
    print(json.dumps(ledger, indent=2))


if __name__ == "__main__":
    main()
