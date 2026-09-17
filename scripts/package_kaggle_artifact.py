from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.experiment_integrity import compute_file_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict fail-closed packaging of Kaggle protocol artifacts.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Directory containing run outputs (outputs/runs/{ARCH}/seed_{SEED})")
    parser.add_argument("--calibration", type=Path, required=True, help="Path to outputs/calibration/{ARCH}_seed{SEED}.json")
    parser.add_argument("--manifest", type=Path, required=True, help="Path to split manifest.json")
    parser.add_argument("--config", type=Path, required=True, help="Path to protocol YAML config")
    parser.add_argument("--output", type=Path, required=True, help="Target ZIP output path (kaggle_artifacts/{ARCH}_seed{SEED}.zip)")
    parser.add_argument("--arch", type=str, default=None, help="Model architecture override")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override")
    parser.add_argument("--run-mode", choices=["smoke", "full"], default=None, help="Run mode override")
    return parser.parse_args()


def package_artifact(
    run_dir: Path,
    calibration_file: Path,
    manifest_file: Path,
    config_file: Path,
    output_zip: Path,
    arch: str | None = None,
    seed: int | None = None,
    run_mode: str | None = None,
) -> Path:
    run_dir = Path(run_dir)
    calibration_file = Path(calibration_file)
    manifest_file = Path(manifest_file)
    config_file = Path(config_file)
    output_zip = Path(output_zip)

    # 1. Inspect run manifest to infer arch, seed, and run_mode if not passed explicitly
    run_manifest_file = run_dir / "run_manifest.json"
    if not run_manifest_file.is_file():
        raise RuntimeError(f"PACKAGING ERROR: Required file 'run_manifest.json' does not exist in '{run_dir}'!")

    run_manifest_data = json.loads(run_manifest_file.read_text(encoding="utf-8"))
    resolved_arch = arch or run_manifest_data.get("architecture")
    resolved_seed = seed if seed is not None else run_manifest_data.get("seed")
    resolved_run_mode = run_mode or run_manifest_data.get("run_mode", "full")

    if not resolved_arch:
        raise RuntimeError("PACKAGING ERROR: Could not determine architecture for packaging!")
    if resolved_seed is None:
        raise RuntimeError("PACKAGING ERROR: Could not determine seed for packaging!")

    calib_filename = f"{resolved_arch}_seed{resolved_seed}.json"

    # H1: Strict list of 9 required files (fail if any is missing)
    required_source_files: dict[str, Path] = {
        "best.pt": run_dir / "best.pt",
        "last.pt": run_dir / "last.pt",
        "training_history.json": run_dir / "training_history.json",
        "run_manifest.json": run_manifest_file,
        "resolved_config.json": run_dir / "resolved_config.json",
        "internal_validation_predictions.csv": run_dir / "internal_validation_predictions.csv",
        calib_filename: calibration_file,
        "manifest.json": manifest_file,
        "protocol_v0_1.yaml": config_file,
    }

    for name, fpath in required_source_files.items():
        if not fpath.is_file():
            raise RuntimeError(
                f"PACKAGING ERROR: Mandatory artifact component '{name}' not found at '{fpath}'!"
            )

    output_zip.parent.mkdir(parents=True, exist_ok=True)

    # H2: Temporary Staging Directory
    with tempfile.TemporaryDirectory() as staging_dir:
        staging_path = Path(staging_dir)
        checksums: dict[str, str] = {}

        for name, src in required_source_files.items():
            dest = staging_path / name
            shutil.copy2(src, dest)
            sha = compute_file_sha256(dest)
            if not sha or sha == "NOT_FOUND":
                raise RuntimeError(f"PACKAGING ERROR: Failed to compute SHA-256 for '{name}' at '{dest}'!")
            checksums[name] = sha

        # Construct internal checksums.json
        checksums_data = {
            "schema_version": 1,
            "architecture": str(resolved_arch),
            "seed": int(resolved_seed),
            "run_mode": str(resolved_run_mode),
            "status": "NON_FINAL_SMOKE_TEST" if resolved_run_mode == "smoke" else "COMPLIANT_PROTOCOL_RUN",
            "files": checksums,
        }
        checksums_path = staging_path / "checksums.json"
        checksums_path.write_text(json.dumps(checksums_data, indent=2), encoding="utf-8")

        # Build ZIP archive from staging
        tmp_zip = output_zip.with_name(f"{output_zip.name}.tmp")
        with zipfile.ZipFile(tmp_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            for name in required_source_files.keys():
                zf.write(staging_path / name, arcname=name)
            zf.write(checksums_path, arcname="checksums.json")

        # H3: Fail-Closed Post-Packaging Verification
        try:
            with zipfile.ZipFile(tmp_zip, "r") as zf:
                namelist = set(zf.namelist())
                expected_names = set(required_source_files.keys()) | {"checksums.json"}

                # Check missing or unauthorized members
                missing = expected_names - namelist
                if missing:
                    raise RuntimeError(f"PACKAGING VERIFICATION FAILED: Missing members in ZIP: {missing}")

                extra = namelist - expected_names
                if extra:
                    raise RuntimeError(f"PACKAGING VERIFICATION FAILED: Unauthorized extra members in ZIP: {extra}")

                # Security checks: No path traversal, absolute path, or locked test leakage
                for member in namelist:
                    if ".." in member or member.startswith("/") or member.startswith("\\"):
                        raise RuntimeError(f"SECURITY VIOLATION: Dangerous path traversal in ZIP member: '{member}'")
                    if "locked_test" in member.lower():
                        raise RuntimeError(f"SECURITY VIOLATION: Locked-test artifact found inside ZIP: '{member}'")

                # Verify checksums.json integrity
                checksums_raw = zf.read("checksums.json").decode("utf-8")
                loaded_checksums_meta = json.loads(checksums_raw)
                if loaded_checksums_meta.get("architecture") != resolved_arch:
                    raise RuntimeError("PACKAGING VERIFICATION FAILED: Checksums architecture mismatch!")
                if loaded_checksums_meta.get("seed") != resolved_seed:
                    raise RuntimeError("PACKAGING VERIFICATION FAILED: Checksums seed mismatch!")

                member_hashes = loaded_checksums_meta.get("files", {})
                for name, expected_hash in checksums.items():
                    if member_hashes.get(name) != expected_hash:
                        raise RuntimeError(f"PACKAGING VERIFICATION FAILED: Ledger hash mismatch for '{name}'!")
                    actual_member_content = zf.read(name)
                    actual_member_hash = hashlib.sha256(actual_member_content).hexdigest()
                    if actual_member_hash != expected_hash:
                        raise RuntimeError(
                            f"PACKAGING VERIFICATION FAILED: Byte checksum mismatch for member '{name}'!"
                        )

            # Move verified tmp ZIP to final output destination atomically
            tmp_zip.replace(output_zip)

        except Exception:
            if tmp_zip.exists():
                tmp_zip.unlink()
            if output_zip.exists():
                output_zip.unlink()
            raise

    return output_zip


def main():
    args = parse_args()
    zip_path = package_artifact(
        run_dir=args.run_dir,
        calibration_file=args.calibration,
        manifest_file=args.split_manifest if hasattr(args, "split_manifest") else args.manifest,
        config_file=args.config,
        output_zip=args.output,
        arch=args.arch,
        seed=args.seed,
        run_mode=args.run_mode,
    )

    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"\n=======================================================")
    print(f"🎉 ARTIFACT SUCCESSFULLY PACKAGED & VERIFIED: {zip_path}")
    print(f"Size: {size_mb:.2f} MB")
    print(f"=======================================================\n")


if __name__ == "__main__":
    main()
