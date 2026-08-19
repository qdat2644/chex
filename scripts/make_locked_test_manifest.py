from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import DEFAULT_LABELS

PATIENT_REGEX = re.compile(r"(?:^|[\\/])patient(\d+)(?:[\\/]|$)", re.IGNORECASE)


def compute_file_sha256(filepath: Path) -> str:
    if not filepath.exists():
        return "NOT_FOUND"
    h = hashlib.sha256()
    with filepath.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def extract_patient_id_strict(path_str: str) -> str | None:
    match = PATIENT_REGEX.search(str(path_str))
    if match:
        return f"patient{match.group(1)}"
    return None


def generate_stable_study_id(patient_id: str, path_str: str, index: int) -> str:
    match = re.search(r"[\\/](study\d+)[\\/]", str(path_str), re.IGNORECASE)
    study_token = match.group(1).lower() if match else f"study{index+1}"
    path_hash = hashlib.sha256(str(path_str).encode("utf-8")).hexdigest()[:8]
    return f"{patient_id}_{study_token}_{path_hash}"


def main():
    parser = argparse.ArgumentParser(description="Create official Locked-Test Manifest and frozen test CSV.")
    parser.add_argument("--valid-csv", type=Path, default=PROJECT_ROOT / "archive" / "valid.csv", help="Official CheXpert validation CSV")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "archive", help="Data root directory")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "splits" / "protocol_v0_1", help="Output directory")
    parser.add_argument("--view", choices=["frontal", "all"], default="frontal")
    args = parser.parse_args()

    if not args.valid_csv.exists():
        print(f"Error: Locked test source CSV not found at {args.valid_csv}", file=sys.stderr)
        sys.exit(1)

    print(f"Reading official locked test source: {args.valid_csv}")
    df = pd.read_csv(args.valid_csv)

    if args.view == "frontal" and "Frontal/Lateral" in df.columns:
        df = df[df["Frontal/Lateral"].astype(str).str.lower() == "frontal"].reset_index(drop=True)

    patient_ids = []
    invalid_paths = []
    for idx, row in df.iterrows():
        pid = extract_patient_id_strict(str(row["Path"]))
        if pid is None:
            invalid_paths.append(str(row["Path"]))
        else:
            patient_ids.append(pid)

    if invalid_paths:
        print(f"CRITICAL ERROR: Failed to parse patient ID from {len(invalid_paths)} paths in locked test!", file=sys.stderr)
        sys.exit(1)

    df["patient_id"] = patient_ids
    df["split_role"] = "locked_test"

    study_ids = []
    for idx, row in df.iterrows():
        study_ids.append(generate_stable_study_id(row["patient_id"], str(row["Path"]), idx))
    df["study_id"] = study_ids
    df["image_path"] = df["Path"]

    target_labels = [l for l in DEFAULT_LABELS if l in df.columns]
    output_cols = ["study_id", "patient_id", "image_path", "split_role"] + target_labels
    locked_test_df = df[output_cols].reset_index(drop=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    locked_csv_path = args.output_dir / "locked_test.csv"
    manifest_path = args.output_dir / "locked_test_manifest.json"

    locked_test_df.to_csv(locked_csv_path, index=False)

    manifest = {
        "schema_version": "1.0",
        "protocol_version": "0.1",
        "role": "locked_test",
        "locked": True,
        "source_csv": str(args.valid_csv),
        "source_csv_sha256": compute_file_sha256(args.valid_csv),
        "csv": "locked_test.csv",
        "csv_sha256": compute_file_sha256(locked_csv_path),
        "labels": target_labels,
        "patients": df["patient_id"].nunique(),
        "studies": len(locked_test_df),
        "view": args.view,
    }

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nLocked-Test Manifest created: {manifest_path}")
    print(f"Locked Test Studies: {len(locked_test_df)} ({df['patient_id'].nunique()} patients) -> {locked_csv_path}")


if __name__ == "__main__":
    main()
