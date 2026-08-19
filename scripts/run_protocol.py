from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description="Multi-Model Multi-Seed Protocol Runner.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "protocol_v0_1.yaml")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "outputs" / "splits" / "protocol_v0_1" / "manifest.json")
    parser.add_argument("--stage", choices=["development", "evaluation"], default="development")
    parser.add_argument("--limit", type=int, help="Optional subset limit for testing")
    args = parser.parse_args()

    if not args.config.exists():
        print(f"Error: Config not found at {args.config}", file=sys.stderr)
        sys.exit(1)
    if not args.manifest.exists():
        print(f"Error: Manifest not found at {args.manifest}", file=sys.stderr)
        sys.exit(1)

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    architectures = config.get("model", {}).get("architectures", ["convnext_small", "densenet121"])
    seeds = config.get("seeds", [42, 43, 44, 45, 46])

    print(f"\n=======================================================")
    print(f"Running Protocol Stage: {args.stage.upper()}")
    print(f"Architectures: {architectures}")
    print(f"Seeds: {seeds}")
    print(f"=======================================================\n")

    python_exe = sys.executable

    if args.stage == "development":
        for arch in architectures:
            for seed in seeds:
                out_dir = PROJECT_ROOT / "outputs" / "runs" / arch / f"seed_{seed}"
                print(f"\n>>> [TRAIN] {arch} (Seed {seed}) -> {out_dir}")
                train_cmd = [
                    python_exe,
                    str(PROJECT_ROOT / "scripts" / "train.py"),
                    "--manifest", str(args.manifest),
                    "--config", str(args.config),
                    "--arch", arch,
                    "--seed", str(seed),
                    "--output-dir", str(out_dir),
                ]
                if args.limit:
                    train_cmd.extend(["--limit", str(args.limit)])
                subprocess.check_call(train_cmd)

                # Run Calibration on the trained best.pt
                ckpt_path = out_dir / "best.pt"
                calib_out = PROJECT_ROOT / "outputs" / "calibration" / f"{arch}_seed{seed}.json"
                print(f">>> [CALIBRATE] {arch} (Seed {seed}) -> {calib_out}")
                calib_cmd = [
                    python_exe,
                    str(PROJECT_ROOT / "scripts" / "calibrate.py"),
                    "--checkpoint", str(ckpt_path),
                    "--split-manifest", str(args.manifest),
                    "--output", str(calib_out),
                    "--seed", str(seed),
                ]
                if args.limit:
                    calib_cmd.extend(["--limit", str(args.limit)])
                subprocess.check_call(calib_cmd)

    print("\nProtocol development runs completed successfully.")


if __name__ == "__main__":
    main()
