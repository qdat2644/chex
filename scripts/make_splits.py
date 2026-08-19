from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import DEFAULT_LABELS


def compute_file_sha256(filepath: Path) -> str:
    if not filepath.exists():
        return "NOT_FOUND"
    h = hashlib.sha256()
    with filepath.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def extract_patient_id(path_str: str) -> str:
    """
    Extracts canonical patient ID from CheXpert image path.
    Example: 'CheXpert-v1.0-small/train/patient00001/study1/view1_frontal.jpg' -> 'patient00001'
    """
    match = re.search(r"(patient\d+)", str(path_str), re.IGNORECASE)
    if match:
        return match.group(1).lower()
    # Fallback to directory structure
    parts = Path(path_str).parts
    for p in parts:
        if p.lower().startswith("patient"):
            return p.lower()
    # Hash fallback if no pattern matched
    return "patient_" + hashlib.sha256(str(path_str).encode()).hexdigest()[:10]


def multi_label_patient_stratification(
    df: pd.DataFrame,
    labels: list[str],
    train_ratio: float = 0.8,
    calib_ratio: float = 0.1,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Patient-level stratification for multi-label data.
    Ensures zero patient leakage while balancing label prevalence across splits.
    """
    assert abs(train_ratio + calib_ratio + val_ratio - 1.0) < 1e-5, "Split ratios must sum to 1.0"
    rng = np.random.RandomState(seed)

    # 1. Aggregate binary positive presence per patient for all target labels
    patient_records = []
    grouped = df.groupby("patient_id")

    for pid, group in grouped:
        label_vector = []
        for lbl in labels:
            # Positive if any study for this patient has label == 1.0
            has_pos = int((group[lbl] == 1.0).any())
            label_vector.append(has_pos)
        patient_records.append({
            "patient_id": pid,
            "label_vector": tuple(label_vector),
            "study_count": len(group),
            "label_sum": sum(label_vector),
        })

    patient_df = pd.DataFrame(patient_records)

    # 2. Sort by label sparsity / rarity so rare combinations are assigned first
    # Sort by label_vector and label_sum
    patient_df = patient_df.sample(frac=1.0, random_state=rng).sort_values(
        by=["label_sum", "study_count"], ascending=False
    ).reset_index(drop=True)

    train_pids: set[str] = set()
    calib_pids: set[str] = set()
    val_pids: set[str] = set()

    # Track target label positive counts in each partition
    num_labels = len(labels)
    train_counts = np.zeros(num_labels, dtype=float)
    calib_counts = np.zeros(num_labels, dtype=float)
    val_counts = np.zeros(num_labels, dtype=float)

    total_patients = len(patient_df)
    target_train = int(total_patients * train_ratio)
    target_calib = int(total_patients * calib_ratio)
    target_val = total_patients - target_train - target_calib

    for _, row in patient_df.iterrows():
        pid = row["patient_id"]
        vec = np.array(row["label_vector"], dtype=float)

        # Candidate partition availability
        can_train = len(train_pids) < target_train
        can_calib = len(calib_pids) < target_calib
        can_val = len(val_pids) < target_val

        if not can_train and not can_calib and not can_val:
            train_pids.add(pid)
            continue

        # Score how much adding this patient balances the ratios
        scores = []
        candidates = []

        if can_train:
            # Projected proportion
            proj_train = (train_counts + vec) / max(1.0, train_ratio)
            scores.append(np.sum(proj_train))
            candidates.append("train")
        if can_calib:
            proj_calib = (calib_counts + vec) / max(1.0, calib_ratio)
            scores.append(np.sum(proj_calib))
            candidates.append("calib")
        if can_val:
            proj_val = (val_counts + vec) / max(1.0, val_ratio)
            scores.append(np.sum(proj_val))
            candidates.append("val")

        # Pick partition with smallest normalized deficit (greedy balanced assignment)
        best_candidate = candidates[int(np.argmin(scores))]

        if best_candidate == "train":
            train_pids.add(pid)
            train_counts += vec
        elif best_candidate == "calib":
            calib_pids.add(pid)
            calib_counts += vec
        else:
            val_pids.add(pid)
            val_counts += vec

    # Verify zero patient overlap
    overlap_tc = train_pids & calib_pids
    overlap_tv = train_pids & val_pids
    overlap_cv = calib_pids & val_pids

    if overlap_tc or overlap_tv or overlap_cv:
        raise ValueError(
            f"CRITICAL LEAKAGE DETECTED: Patient overlap detected between splits! "
            f"Train-Calib: {len(overlap_tc)}, Train-Val: {len(overlap_tv)}, Calib-Val: {len(overlap_cv)}"
        )

    train_df = df[df["patient_id"].isin(train_pids)].reset_index(drop=True)
    calib_df = df[df["patient_id"].isin(calib_pids)].reset_index(drop=True)
    val_df = df[df["patient_id"].isin(val_pids)].reset_index(drop=True)

    return train_df, calib_df, val_df


def main():
    parser = argparse.ArgumentParser(description="Create patient-level leak-free multi-label dataset splits.")
    parser.add_argument("--csv", type=Path, default=PROJECT_ROOT / "archive" / "train.csv", help="Source CSV file")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "archive", help="Data root directory")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "splits", help="Output directory")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train partition ratio (default: 0.80)")
    parser.add_argument("--calib-ratio", type=float, default=0.1, help="Calibration partition ratio (default: 0.10)")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Internal validation partition ratio (default: 0.10)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--view", choices=["frontal", "all"], default="frontal", help="Filter by radiograph view")
    args = parser.parse_args()

    if not args.csv.exists():
        print(f"Error: Input CSV {args.csv} does not exist.")
        sys.exit(1)

    print(f"Reading input dataset: {args.csv}")
    df = pd.read_csv(args.csv)

    # 1. Filter Frontal View only
    if args.view == "frontal" and "Frontal/Lateral" in df.columns:
        initial_len = len(df)
        df = df[df["Frontal/Lateral"].astype(str).str.lower() == "frontal"].reset_index(drop=True)
        print(f"Filtered Frontal views: {len(df)} / {initial_len} studies retained.")

    # 2. Extract and assign patient_id
    df["patient_id"] = df["Path"].apply(extract_patient_id)
    unique_patients = df["patient_id"].nunique()
    print(f"Total Studies: {len(df)} across {unique_patients} unique patients.")

    # 3. Stratified Patient-Level Split
    labels = [l for l in DEFAULT_LABELS if l in df.columns]
    train_df, calib_df, val_df = multi_label_patient_stratification(
        df,
        labels=labels,
        train_ratio=args.train_ratio,
        calib_ratio=args.calib_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_csv = args.output_dir / "train.csv"
    calib_csv = args.output_dir / "calibration.csv"
    val_csv = args.output_dir / "internal_val.csv"
    manifest_json = args.output_dir / "manifest.json"

    train_df.to_csv(train_csv, index=False)
    calib_df.to_csv(calib_csv, index=False)
    val_df.to_csv(val_csv, index=False)

    # 4. Compute split hashes and manifest
    manifest = {
        "dataset_name": "CheXpert Frontal Cohort (Patient-Level Split)",
        "source_csv": str(args.csv),
        "source_sha256": compute_file_sha256(args.csv),
        "seed": args.seed,
        "view": args.view,
        "split_ratios": {
            "train": args.train_ratio,
            "calibration": args.calib_ratio,
            "internal_val": args.val_ratio,
        },
        "splits": {
            "train": {
                "file": "train.csv",
                "sha256": compute_file_sha256(train_csv),
                "num_studies": len(train_df),
                "num_patients": train_df["patient_id"].nunique(),
            },
            "calibration": {
                "file": "calibration.csv",
                "sha256": compute_file_sha256(calib_csv),
                "num_studies": len(calib_df),
                "num_patients": calib_df["patient_id"].nunique(),
            },
            "internal_val": {
                "file": "internal_val.csv",
                "sha256": compute_file_sha256(val_csv),
                "num_studies": len(val_df),
                "num_patients": val_df["patient_id"].nunique(),
            },
        },
        "target_labels": labels,
    }

    manifest_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nManifest successfully created: {manifest_json}")
    print(f"Train:        {len(train_df)} studies ({train_df['patient_id'].nunique()} patients) -> {train_csv}")
    print(f"Calibration:  {len(calib_df)} studies ({calib_df['patient_id'].nunique()} patients) -> {calib_csv}")
    print(f"Internal-Val: {len(val_df)} studies ({val_df['patient_id'].nunique()} patients) -> {val_csv}")


if __name__ == "__main__":
    main()
