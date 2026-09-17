from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.experiment_integrity import canonical_json_sha256, compute_file_sha256
from scripts.kaggle_preflight import validate_development_source


def _load_json(path: Path, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"PACKAGING ERROR: missing {description}: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"PACKAGING ERROR: {description} must be a JSON object")
    return value


def _checkpoint_metadata(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"PACKAGING ERROR: missing checkpoint: {path}")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise RuntimeError(f"PACKAGING ERROR: cannot parse checkpoint: {path}") from exc
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"PACKAGING ERROR: checkpoint is not a dictionary: {path}")
    nested = checkpoint.get("metadata", {})
    return {
        "architecture": checkpoint.get("architecture") or nested.get("architecture"),
        "seed": checkpoint.get("seed") if checkpoint.get("seed") is not None else nested.get("seed"),
        "run_mode": checkpoint.get("run_mode") or nested.get("run_mode"),
        "protocol_version": checkpoint.get("protocol_version") or nested.get("protocol_version", "0.1"),
        "git_commit": checkpoint.get("git_commit") or nested.get("git_commit"),
        "git_dirty": checkpoint.get("git_dirty") if checkpoint.get("git_dirty") is not None else nested.get("git_dirty"),
        "labels": checkpoint.get("labels") or nested.get("labels"),
        "resolved_config_sha256": checkpoint.get("resolved_config_sha256") or nested.get("resolved_config_sha256"),
        "manifest_sha256": checkpoint.get("manifest_sha256") or nested.get("split_manifest_sha256"),
    }


def _equal(field: str, sources: dict[str, object]) -> object:
    values = list(sources.values())
    if any(value is None for value in values):
        raise RuntimeError(f"PACKAGING ERROR: missing {field}: {sources}")
    first = values[0]
    if any(value != first for value in values[1:]):
        raise RuntimeError(f"PACKAGING ERROR: {field} mismatch: {sources}")
    return first


def validate_artifact_sources(
    run_dir: Path,
    calibration_file: Path,
    manifest_file: Path,
    config_file: Path,
    expected_arch: str | None = None,
    expected_seed: int | None = None,
    expected_run_mode: str | None = None,
) -> tuple[dict[str, Path], dict[str, object]]:
    run_dir = Path(run_dir)
    best_path = run_dir / "best.pt"
    last_path = run_dir / "last.pt"
    run_manifest_path = run_dir / "run_manifest.json"
    resolved_path = run_dir / "resolved_config.json"
    predictions_path = run_dir / "internal_validation_predictions.csv"
    history_path = run_dir / "training_history.json"

    best = _checkpoint_metadata(best_path)
    last = _checkpoint_metadata(last_path)
    run = _load_json(run_manifest_path, "run_manifest.json")
    resolved = _load_json(resolved_path, "resolved_config.json")
    calibration = _load_json(Path(calibration_file), "calibration JSON")
    manifest = _load_json(Path(manifest_file), "development manifest")
    if not Path(config_file).is_file():
        raise RuntimeError(f"PACKAGING ERROR: missing protocol config: {config_file}")
    protocol = yaml.safe_load(Path(config_file).read_text(encoding="utf-8"))
    if not isinstance(protocol, dict):
        raise RuntimeError("PACKAGING ERROR: protocol config must be a mapping")
    if not history_path.is_file() or not predictions_path.is_file():
        raise RuntimeError("PACKAGING ERROR: training history or internal-validation predictions are missing")

    architecture = _equal("architecture", {
        "best.pt": best["architecture"], "last.pt": last["architecture"],
        "run_manifest": run.get("architecture"), "resolved_config": resolved.get("architecture"),
        "calibration": calibration.get("architecture"),
    })
    seed = _equal("seed", {
        "best.pt": best["seed"], "last.pt": last["seed"], "run_manifest": run.get("seed"),
        "resolved_config": resolved.get("seed"), "calibration": calibration.get("seed"),
        "development_manifest": manifest.get("seed"),
    })
    run_mode = _equal("run_mode", {
        "best.pt": best["run_mode"], "last.pt": last["run_mode"],
        "run_manifest": run.get("run_mode"), "calibration": calibration.get("run_mode"),
    })
    protocol_version = _equal("protocol_version", {
        "best.pt": str(best["protocol_version"]), "last.pt": str(last["protocol_version"]),
        "run_manifest": str(run.get("protocol_version")), "calibration": str(calibration.get("protocol_version")),
        "development_manifest": str(manifest.get("protocol_version")), "protocol_yaml": str(protocol.get("protocol_version")),
    })
    labels = _equal("labels", {
        "best.pt": best["labels"], "last.pt": last["labels"], "run_manifest": run.get("labels"),
        "resolved_config": resolved.get("labels"), "calibration": calibration.get("labels"),
        "development_manifest": manifest.get("labels"),
    })
    git_commit = _equal("git_commit", {
        "best.pt": best["git_commit"], "last.pt": last["git_commit"],
        "run_manifest": run.get("git_commit"), "calibration": calibration.get("git_commit"),
    })
    git_dirty = _equal("git_dirty", {
        "best.pt": best["git_dirty"], "last.pt": last["git_dirty"],
        "run_manifest": run.get("git_dirty"),
    })

    resolved_hash = canonical_json_sha256(resolved)
    _equal("resolved_config_sha256", {
        "computed": resolved_hash, "best.pt": best["resolved_config_sha256"],
        "last.pt": last["resolved_config_sha256"], "run_manifest": run.get("resolved_config_sha256"),
        "calibration": calibration.get("resolved_config_sha256"),
    })
    manifest_hash = compute_file_sha256(manifest_file)
    _equal("manifest_sha256", {
        "computed": manifest_hash, "best.pt": best["manifest_sha256"], "last.pt": last["manifest_sha256"],
        "run_manifest": run.get("manifest_sha256"), "calibration": calibration.get("manifest_sha256"),
        "resolved_config": resolved.get("data", {}).get("split_manifest_sha256"),
    })

    best_hash = compute_file_sha256(best_path)
    last_hash = compute_file_sha256(last_path)
    predictions_hash = compute_file_sha256(predictions_path)
    _equal("best checkpoint SHA-256", {
        "computed": best_hash, "run_manifest": run.get("best_checkpoint_sha256"),
        "calibration": calibration.get("checkpoint_sha256"),
    })
    _equal("last checkpoint SHA-256", {"computed": last_hash, "run_manifest": run.get("last_checkpoint_sha256")})
    _equal("prediction SHA-256", {"computed": predictions_hash, "run_manifest": run.get("predictions_sha256")})

    validate_development_source(
        Path(manifest.get("source_csv", "development/train.csv")).parent,
        Path(manifest.get("source_csv", "development/train.csv")),
        ["train/patient00000/study1/view1_frontal.jpg"],
        metadata=manifest,
        manifest=manifest,
    )
    if expected_arch is not None and architecture != expected_arch:
        raise RuntimeError(f"PACKAGING ERROR: expected architecture {expected_arch}, artifact has {architecture}")
    if expected_seed is not None and int(seed) != int(expected_seed):
        raise RuntimeError(f"PACKAGING ERROR: expected seed {expected_seed}, artifact has {seed}")
    if expected_run_mode is not None and run_mode != expected_run_mode:
        raise RuntimeError(f"PACKAGING ERROR: expected run mode {expected_run_mode}, artifact has {run_mode}")

    calibration_name = f"{architecture}_seed{seed}.json"
    sources = {
        "best.pt": best_path, "last.pt": last_path, "training_history.json": history_path,
        "run_manifest.json": run_manifest_path, "resolved_config.json": resolved_path,
        "internal_validation_predictions.csv": predictions_path, calibration_name: Path(calibration_file),
        "manifest.json": Path(manifest_file), "protocol_v0_1.yaml": Path(config_file),
    }
    metadata = {
        "architecture": architecture, "seed": int(seed), "run_mode": run_mode,
        "protocol_version": protocol_version, "labels": labels, "git_commit": git_commit,
        "resolved_config_sha256": resolved_hash, "manifest_sha256": manifest_hash,
        "status": "NON_FINAL_SMOKE_TEST" if run_mode == "smoke" else "COMPLIANT_PROTOCOL_RUN",
        "git_dirty": bool(git_dirty),
    }
    if run_mode == "full" and metadata["git_dirty"]:
        raise RuntimeError("PACKAGING ERROR: full-mode artifact records a dirty Git tree")
    return sources, metadata


def verify_artifact_zip(path: Path) -> dict[str, object]:
    try:
        with zipfile.ZipFile(Path(path), "r") as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise RuntimeError("Artifact ZIP contains duplicate members")
            for name in names:
                member = Path(name)
                if member.is_absolute() or ".." in member.parts or len(member.parts) != 1:
                    raise RuntimeError(f"Unsafe artifact ZIP member: {name}")
            if "checksums.json" not in names:
                raise RuntimeError("Artifact ZIP is missing checksums.json")
            ledger = json.loads(archive.read("checksums.json").decode("utf-8"))
            files = ledger.get("files", {})
            if set(names) != set(files) | {"checksums.json"}:
                raise RuntimeError("Artifact ZIP member set does not match checksums.json")
            for name, expected in files.items():
                if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                    raise RuntimeError(f"Artifact ZIP checksum mismatch for {name}")
            calibration_name = f"{ledger.get('architecture')}_seed{ledger.get('seed')}.json"
            if calibration_name not in files:
                raise RuntimeError("Artifact ZIP calibration member does not match derived metadata")
            with tempfile.TemporaryDirectory() as temporary:
                stage = Path(temporary)
                run_dir = stage / "run"
                run_dir.mkdir()
                run_members = {
                    "best.pt", "last.pt", "training_history.json", "run_manifest.json",
                    "resolved_config.json", "internal_validation_predictions.csv",
                }
                for name in files:
                    destination = (run_dir if name in run_members else stage) / name
                    destination.write_bytes(archive.read(name))
                _, derived = validate_artifact_sources(
                    run_dir,
                    stage / calibration_name,
                    stage / "manifest.json",
                    stage / "protocol_v0_1.yaml",
                    str(ledger.get("architecture")),
                    int(ledger.get("seed")),
                    str(ledger.get("run_mode")),
                )
                for field in (
                    "architecture", "seed", "run_mode", "protocol_version", "labels",
                    "git_commit", "resolved_config_sha256", "manifest_sha256", "status", "git_dirty",
                ):
                    if ledger.get(field) != derived.get(field):
                        raise RuntimeError(f"Artifact ZIP ledger metadata mismatch for {field}")
            return ledger
    except zipfile.BadZipFile as exc:
        raise RuntimeError(f"Invalid artifact ZIP: {path}") from exc


def package_artifact(
    run_dir: Path,
    calibration_file: Path,
    manifest_file: Path,
    config_file: Path,
    output_zip: Path,
    expected_arch: str | None = None,
    expected_seed: int | None = None,
    expected_run_mode: str | None = None,
) -> Path:
    sources, metadata = validate_artifact_sources(
        run_dir, calibration_file, manifest_file, config_file,
        expected_arch, expected_seed, expected_run_mode,
    )
    output_zip = Path(output_zip)
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    temporary_zip = output_zip.with_name(output_zip.name + ".tmp")
    with tempfile.TemporaryDirectory() as temporary:
        stage = Path(temporary)
        hashes: dict[str, str] = {}
        for name, source in sources.items():
            destination = stage / name
            shutil.copy2(source, destination)
            hashes[name] = compute_file_sha256(destination)
        ledger = {"schema_version": 2, **metadata, "files": hashes}
        ledger_path = stage / "checksums.json"
        ledger_path.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
        with zipfile.ZipFile(temporary_zip, "w", zipfile.ZIP_DEFLATED) as archive:
            for name in sources:
                archive.write(stage / name, arcname=name)
            archive.write(ledger_path, arcname="checksums.json")
    verify_artifact_zip(temporary_zip)
    temporary_zip.replace(output_zip)
    return output_zip


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict fail-closed packaging of Kaggle protocol artifacts")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-arch")
    parser.add_argument("--expected-seed", type=int)
    parser.add_argument("--expected-run-mode", choices=["smoke", "full"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(package_artifact(
        args.run_dir, args.calibration, args.manifest, args.config, args.output,
        args.expected_arch, args.expected_seed, args.expected_run_mode,
    ))


if __name__ == "__main__":
    main()
