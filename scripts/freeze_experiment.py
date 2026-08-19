from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def compute_file_sha256(filepath: Path) -> str:
    if not filepath.is_file():
        raise FileNotFoundError(f"Cannot compute SHA-256: file does not exist at '{filepath}'")
    h = hashlib.sha256()
    with filepath.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    digest = h.hexdigest()
    if not digest:
        raise ValueError(f"Computed empty SHA-256 for '{filepath}'")
    return digest


import shutil

def get_git_exe() -> str:
    exe = shutil.which("git")
    if exe:
        return exe
    local_tool = PROJECT_ROOT / "tools" / "git" / "cmd" / "git.exe"
    if local_tool.is_file():
        return str(local_tool)
    return "git"


def get_clean_git_commit(allow_dirty: bool = False) -> str:
    git_bin = get_git_exe()
    try:
        status_output = subprocess.check_output(
            [git_bin, "status", "--porcelain"],
            cwd=str(PROJECT_ROOT),
            text=True,
        ).strip()
        if status_output and not allow_dirty:
            raise RuntimeError(
                f"FREEZE FAIL-CLOSED ERROR: Git working tree is dirty! Uncommitted changes detected:\n{status_output}\n"
                "You must commit all changes before freezing the experiment."
            )

        commit_sha = subprocess.check_output(
            [git_bin, "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            text=True,
        ).strip()

        if not commit_sha or len(commit_sha) < 7:
            raise RuntimeError(f"FREEZE FAIL-CLOSED ERROR: Invalid git commit SHA '{commit_sha}'")

        return commit_sha
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FREEZE FAIL-CLOSED ERROR: Failed to obtain Git commit SHA: {e}") from e


def freeze_experiment(
    config_path: Path,
    development_manifest_path: Path,
    locked_test_manifest_path: Path,
    runs_dir: Path,
    calibration_dir: Path,
    output_ledger_path: Path,
    allow_dirty: bool = False,
) -> dict:
    # 1. Strict Git State Verification
    git_commit = get_clean_git_commit(allow_dirty=allow_dirty)

    # 2. Config & Manifest Verification
    if not config_path.is_file():
        raise FileNotFoundError(f"FREEZE FAIL-CLOSED ERROR: Experiment config not found at '{config_path}'")
    if not development_manifest_path.is_file():
        raise FileNotFoundError(f"FREEZE FAIL-CLOSED ERROR: Development split manifest not found at '{development_manifest_path}'")
    if not locked_test_manifest_path.is_file():
        raise FileNotFoundError(f"FREEZE FAIL-CLOSED ERROR: Locked-test manifest not found at '{locked_test_manifest_path}'")

    config_sha256 = compute_file_sha256(config_path)
    dev_manifest_sha256 = compute_file_sha256(development_manifest_path)
    locked_manifest_sha256 = compute_file_sha256(locked_test_manifest_path)

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Invalid YAML config at '{config_path}'")

    protocol_version = str(config.get("protocol_version", "0.1"))
    architectures = config.get("model", {}).get("architectures", ["convnext_small", "densenet121"])
    seeds = config.get("seeds", [42, 43, 44, 45, 46])

    if not architectures or len(architectures) < 2:
        raise ValueError("FREEZE FAIL-CLOSED ERROR: Config must specify at least 2 architectures (convnext_small, densenet121)")
    if not seeds or len(seeds) != 5:
        raise ValueError(f"FREEZE FAIL-CLOSED ERROR: Protocol strictly requires exactly 5 seeds (42..46), found: {seeds}")

    models_ledger: dict[str, dict[str, dict[str, str]]] = {}
    total_checkpoints = 0
    total_thresholds = 0

    for arch in architectures:
        models_ledger[arch] = {}
        for seed in seeds:
            seed_str = str(seed)
            ckpt_path = runs_dir / arch / f"seed_{seed}" / "best.pt"
            th_path = calibration_dir / f"{arch}_seed{seed}.json"

            if not ckpt_path.is_file():
                raise FileNotFoundError(f"FREEZE FAIL-CLOSED ERROR: Missing checkpoint for {arch} seed {seed} at '{ckpt_path}'")
            if not th_path.is_file():
                raise FileNotFoundError(f"FREEZE FAIL-CLOSED ERROR: Missing threshold artifact for {arch} seed {seed} at '{th_path}'")

            ckpt_sha256 = compute_file_sha256(ckpt_path)
            th_sha256 = compute_file_sha256(th_path)

            # Validate that threshold artifact matches checkpoint SHA
            th_data = json.loads(th_path.read_text(encoding="utf-8"))
            if th_data.get("checkpoint_sha256") != ckpt_sha256:
                raise RuntimeError(
                    f"FREEZE FAIL-CLOSED ERROR: Threshold artifact '{th_path}' records checkpoint SHA "
                    f"'{th_data.get('checkpoint_sha256')}' which does not match actual checkpoint '{ckpt_sha256}'!"
                )
            if th_data.get("split_manifest_sha256") != dev_manifest_sha256:
                raise RuntimeError(
                    f"FREEZE FAIL-CLOSED ERROR: Threshold artifact '{th_path}' records manifest SHA "
                    f"'{th_data.get('split_manifest_sha256')}' which does not match development manifest '{dev_manifest_sha256}'!"
                )

            models_ledger[arch][seed_str] = {
                "checkpoint_path": str(ckpt_path.relative_to(PROJECT_ROOT) if ckpt_path.is_relative_to(PROJECT_ROOT) else ckpt_path),
                "checkpoint_sha256": ckpt_sha256,
                "threshold_path": str(th_path.relative_to(PROJECT_ROOT) if th_path.is_relative_to(PROJECT_ROOT) else th_path),
                "threshold_sha256": th_sha256,
            }
            total_checkpoints += 1
            total_thresholds += 1

    expected_total = len(architectures) * len(seeds)
    if total_checkpoints != expected_total or total_thresholds != expected_total:
        raise RuntimeError(
            f"FREEZE FAIL-CLOSED ERROR: Expected {expected_total} checkpoints and thresholds, got {total_checkpoints} & {total_thresholds}"
        )

    frozen_ledger = {
        "protocol_version": protocol_version,
        "git_commit": git_commit,
        "config_path": str(config_path.relative_to(PROJECT_ROOT) if config_path.is_relative_to(PROJECT_ROOT) else config_path),
        "config_sha256": config_sha256,
        "development_manifest_sha256": dev_manifest_sha256,
        "locked_test_manifest_sha256": locked_manifest_sha256,
        "models": models_ledger,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    # Verify no invalid null/unknown/NOT_FOUND values
    ledger_str = json.dumps(frozen_ledger)
    for forbidden in ["NOT_FOUND", "UNKNOWN", "null"]:
        if f'"{forbidden}"' in ledger_str:
            raise ValueError(f"FREEZE FAIL-CLOSED ERROR: Forbidden value '{forbidden}' found in frozen ledger!")

    output_ledger_path.parent.mkdir(parents=True, exist_ok=True)
    output_ledger_path.write_text(json.dumps(frozen_ledger, indent=2), encoding="utf-8")
    print(f"Experiment successfully frozen (fail-closed verified) -> {output_ledger_path}")
    print(f"Verified {total_checkpoints} checkpoints and {total_thresholds} threshold artifacts across {len(architectures)} architectures.")
    return frozen_ledger


def main():
    parser = argparse.ArgumentParser(description="Freeze experiment checkpoints, thresholds, and configs fail-closed.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "protocol_v0_1.yaml", help="Protocol YAML config")
    parser.add_argument("--development-manifest", type=Path, default=PROJECT_ROOT / "outputs" / "splits" / "protocol_v0_1" / "manifest.json")
    parser.add_argument("--locked-test-manifest", type=Path, default=PROJECT_ROOT / "outputs" / "splits" / "protocol_v0_1" / "locked_test_manifest.json")
    parser.add_argument("--runs", type=Path, default=PROJECT_ROOT / "outputs" / "runs", help="Runs root directory")
    parser.add_argument("--calibration", type=Path, default=PROJECT_ROOT / "outputs" / "calibration", help="Calibration artifacts directory")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "frozen" / "protocol_v0_1.json", help="Output frozen JSON")
    parser.add_argument("--allow-dirty", action="store_true", default=False, help="Allow dirty working tree for development testing only")
    args = parser.parse_args()

    freeze_experiment(
        config_path=args.config,
        development_manifest_path=args.development_manifest,
        locked_test_manifest_path=args.locked_test_manifest,
        runs_dir=args.runs,
        calibration_dir=args.calibration,
        output_ledger_path=args.output,
        allow_dirty=args.allow_dirty,
    )


if __name__ == "__main__":
    main()
