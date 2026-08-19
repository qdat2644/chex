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
    if not filepath.exists():
        return "NOT_FOUND"
    h = hashlib.sha256()
    with filepath.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def get_git_commit_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "UNKNOWN"


def main():
    parser = argparse.ArgumentParser(description="Freeze experiment checkpoints, thresholds, and configurations before test evaluation.")
    parser.add_argument("--runs", type=Path, default=PROJECT_ROOT / "outputs" / "runs", help="Runs root directory")
    parser.add_argument("--calibration", type=Path, default=PROJECT_ROOT / "outputs" / "calibration", help="Calibration artifacts directory")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "outputs" / "splits" / "protocol_v0_1" / "manifest.json", help="Split manifest")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "protocol_v0_1.yaml", help="Protocol YAML config")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "frozen" / "protocol_v0_1.json", help="Output frozen JSON")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) if args.config.exists() else {}
    architectures = config.get("model", {}).get("architectures", ["convnext_small", "densenet121"])
    seeds = config.get("seeds", [42, 43, 44, 45, 46])

    checkpoints_ledger = {}
    thresholds_ledger = {}

    for arch in architectures:
        for seed in seeds:
            ckpt_path = args.runs / arch / f"seed_{seed}" / "best.pt"
            calib_path = args.calibration / f"{arch}_seed{seed}.json"

            key = f"{arch}_seed{seed}"
            checkpoints_ledger[key] = {
                "path": str(ckpt_path),
                "sha256": compute_file_sha256(ckpt_path),
            }
            thresholds_ledger[key] = {
                "path": str(calib_path),
                "sha256": compute_file_sha256(calib_path),
            }

    frozen_manifest = {
        "schema_version": "1.0",
        "protocol_version": "0.1",
        "git_commit": get_git_commit_sha(),
        "config_sha256": compute_file_sha256(args.config),
        "split_manifest_sha256": compute_file_sha256(args.manifest),
        "checkpoints": checkpoints_ledger,
        "threshold_artifacts": thresholds_ledger,
        "frozen_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(frozen_manifest, indent=2), encoding="utf-8")
    print(f"Experiment state frozen successfully: {args.output}")


if __name__ == "__main__":
    main()
