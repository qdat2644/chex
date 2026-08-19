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

from app.config import DEFAULT_LABELS, STANFORD_U_ONES_LABELS

PATIENT_REGEX = re.compile(r"(?:^|[\\/])patient(\d+)(?:[\\/]|$)", re.IGNORECASE)


def compute_file_sha256(filepath: Path) -> str:
    if not filepath.exists():
        return "NOT_FOUND"
    h = hashlib.sha256()
    with filepath.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_image_sha256(filepath: Path) -> str:
    if not filepath.exists():
        return "MISSING_FILE"
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
    # Look for study number in path, e.g. study1
    match = re.search(r"[\\/](study\d+)[\\/]", str(path_str), re.IGNORECASE)
    study_token = match.group(1).lower() if match else f"study{index+1}"
    path_hash = hashlib.sha256(str(path_str).encode("utf-8")).hexdigest()[:8]
    return f"{patient_id}_{study_token}_{path_hash}"


def normalize_uncertain_label(val: float | None, label_name: str, policy: str = "u_ones_zeros") -> float:
    if pd.isna(val):
        return 0.0
    v = float(val)
    if v == -1.0:
        if policy == "one":
            return 1.0
        elif policy in ("u_ones_zeros", "stanford"):
            return 1.0 if label_name in STANFORD_U_ONES_LABELS else 0.0
        elif policy == "smooth":
            return 0.6
        elif policy == "zero" or policy == "ignore":
            return 0.0
        return 0.0
    return 1.0 if v >= 0.5 else 0.0


def iterative_multilabel_split(
    patient_ids: list[str],
    label_matrix: np.ndarray,
    ratios: list[float],
    seed: int = 42,
) -> list[list[int]]:
    """
    Iterative multi-label stratification algorithm (Sechidis et al., 2011).
    Partitions patients into K subsets maintaining proportional representation for all labels.
    """
    rng = np.random.RandomState(seed)
    n_samples, n_labels = label_matrix.shape
    k_splits = len(ratios)

    # Normalize ratios
    ratios = [r / sum(ratios) for r in ratios]
    desired_counts = np.zeros((k_splits, n_labels), dtype=float)
    label_totals = np.sum(label_matrix, axis=0)

    for k in range(k_splits):
        desired_counts[k] = ratios[k] * label_totals

    allocated_counts = np.zeros((k_splits, n_labels), dtype=int)
    split_indices: list[list[int]] = [[] for _ in range(k_splits)]
    desired_samples = [int(round(r * n_samples)) for r in ratios]

    # Ensure total samples sum to n_samples
    diff = n_samples - sum(desired_samples)
    desired_samples[0] += diff

    # Shuffle order of patients deterministically
    order = rng.permutation(n_samples).tolist()
    unassigned = set(order)

    # Process labels with fewest positive examples first (rarest label priority)
    while unassigned:
        # Find remaining label with lowest positive count
        unassigned_list = list(unassigned)
        sub_matrix = label_matrix[unassigned_list]
        rem_label_counts = np.sum(sub_matrix, axis=0)

        # Filter to active remaining labels
        active_labels = np.where(rem_label_counts > 0)[0]
        if len(active_labels) == 0:
            # All remaining samples have all 0 labels: distribute by sample capacity
            for idx in unassigned_list:
                # Find subset with highest remaining sample capacity
                capacities = [desired_samples[k] - len(split_indices[k]) for k in range(k_splits)]
                best_split = int(np.argmax(capacities))
                split_indices[best_split].append(idx)
            break

        rarest_label = active_labels[int(np.argmin(rem_label_counts[active_labels]))]

        # Find patients who have this label
        candidate_patients = [idx for idx in unassigned if label_matrix[idx, rarest_label] == 1]
        rng.shuffle(candidate_patients)

        for p_idx in candidate_patients:
            if p_idx not in unassigned:
                continue

            # Determine best split subset for this patient
            deficits = desired_counts - allocated_counts
            # Find splits that still need this label
            candidate_splits = np.where(deficits[:, rarest_label] > 0)[0]
            if len(candidate_splits) == 0:
                # If none need it, pick split with highest sample capacity
                capacities = [desired_samples[k] - len(split_indices[k]) for k in range(k_splits)]
                best_split = int(np.argmax(capacities))
            else:
                # Pick candidate split with highest deficit for this label
                best_split = candidate_splits[int(np.argmax(deficits[candidate_splits, rarest_label]))]

            split_indices[best_split].append(p_idx)
            allocated_counts[best_split] += label_matrix[p_idx]
            unassigned.remove(p_idx)

    return split_indices


def multi_label_patient_stratification(
    df: pd.DataFrame,
    labels: list[str],
    train_ratio: float = 0.8,
    calib_ratio: float = 0.1,
    val_ratio: float = 0.1,
    seed: int = 42,
    uncertainty_policy: str = "u_ones_zeros",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if "patient_id" not in df.columns:
        patient_ids = []
        for idx, row in df.iterrows():
            pid = extract_patient_id_strict(str(row["Path"]))
            patient_ids.append(pid or f"patient_{idx}")
        df = df.copy()
        df["patient_id"] = patient_ids

    unique_patients = sorted(list(set(df["patient_id"])))
    patient_to_studies = df.groupby("patient_id")
    patient_label_vectors = []

    for pid in unique_patients:
        group = patient_to_studies.get_group(pid)
        vec = []
        for label in labels:
            has_pos = 0
            if label in group.columns:
                for val in group[label]:
                    norm_val = normalize_uncertain_label(val, label, policy=uncertainty_policy)
                    if norm_val >= 0.5:
                        has_pos = 1
                        break
            vec.append(has_pos)
        patient_label_vectors.append(vec)

    label_matrix = np.array(patient_label_vectors, dtype=int)
    ratios = [train_ratio, calib_ratio, val_ratio]
    split_indices = iterative_multilabel_split(unique_patients, label_matrix, ratios, seed=seed)

    train_pids = set([unique_patients[i] for i in split_indices[0]])
    calib_pids = set([unique_patients[i] for i in split_indices[1]])
    val_pids = set([unique_patients[i] for i in split_indices[2]])

    train_df = df[df["patient_id"].isin(train_pids)].copy().reset_index(drop=True)
    calib_df = df[df["patient_id"].isin(calib_pids)].copy().reset_index(drop=True)
    val_df = df[df["patient_id"].isin(val_pids)].copy().reset_index(drop=True)

    return train_df, calib_df, val_df


def main():
    parser = argparse.ArgumentParser(description="Create leak-free patient-level dataset splits with strict verification.")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "archive", help="Data root directory")
    parser.add_argument("--train-csv", type=Path, default=PROJECT_ROOT / "archive" / "train.csv", help="Source dataset CSV")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "splits" / "protocol_v0_1", help="Output split directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio (default: 0.80)")
    parser.add_argument("--calibration-ratio", type=float, default=0.1, help="Calibration split ratio (default: 0.10)")
    parser.add_argument("--validation-ratio", type=float, default=0.1, help="Internal validation split ratio (default: 0.10)")
    parser.add_argument("--uncertainty-policy", choices=["u_ones_zeros", "smooth", "zero", "one", "ignore"], default="u_ones_zeros")
    parser.add_argument("--view", choices=["frontal", "all"], default="frontal")
    args = parser.parse_args()

    if not args.train_csv.exists():
        print(f"Error: Training CSV not found at {args.train_csv}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading source dataset from: {args.train_csv}")
    df = pd.read_csv(args.train_csv)
    initial_rows = len(df)

    # 1. Filter by Radiograph View
    if args.view == "frontal" and "Frontal/Lateral" in df.columns:
        df = df[df["Frontal/Lateral"].astype(str).str.lower() == "frontal"].reset_index(drop=True)
        print(f"Filtered Frontal views: {len(df)} / {initial_rows} studies retained.")

    # 2. Parse Patient ID Strictly from Path
    patient_ids = []
    invalid_paths = []

    for idx, row in df.iterrows():
        path_str = str(row["Path"])
        pid = extract_patient_id_strict(path_str)
        if pid is None:
            invalid_paths.append(path_str)
        else:
            patient_ids.append(pid)

    if invalid_paths:
        print(f"\nCRITICAL ERROR: Failed to parse patient ID from {len(invalid_paths)} paths! Regex: (?:^|[\\/])patient(\\d+)(?:[\\/]|$)", file=sys.stderr)
        print("First 20 failing paths:", file=sys.stderr)
        for p in invalid_paths[:20]:
            print(f"  - {p}", file=sys.stderr)
        sys.exit(1)

    df["patient_id"] = patient_ids
    unique_patients = sorted(list(set(patient_ids)))
    print(f"Parsed {len(df)} studies across {len(unique_patients)} unique patients successfully.")

    # 3. Create Patient-Level Aggregated Label Matrix (Boolean OR across studies)
    target_labels = [l for l in DEFAULT_LABELS if l in df.columns]
    print(f"Target competition labels ({len(target_labels)}): {target_labels}")

    patient_to_studies = df.groupby("patient_id")
    patient_label_vectors = []

    for pid in unique_patients:
        group = patient_to_studies.get_group(pid)
        vec = []
        for label in target_labels:
            has_pos = 0
            for val in group[label]:
                norm_val = normalize_uncertain_label(val, label, policy=args.uncertainty_policy)
                if norm_val >= 0.5:
                    has_pos = 1
                    break
            vec.append(has_pos)
        patient_label_vectors.append(vec)

    label_matrix = np.array(patient_label_vectors, dtype=int)

    # 4. Stratified Split (80% Train, 10% Calibration, 10% Internal Validation)
    ratios = [args.train_ratio, args.calibration_ratio, args.validation_ratio]
    split_indices = iterative_multilabel_split(unique_patients, label_matrix, ratios, seed=args.seed)

    train_pids = set([unique_patients[i] for i in split_indices[0]])
    calib_pids = set([unique_patients[i] for i in split_indices[1]])
    val_pids = set([unique_patients[i] for i in split_indices[2]])

    # 5. Strict Zero-Overlap Assertions
    overlap_tc = train_pids & calib_pids
    overlap_tv = train_pids & val_pids
    overlap_cv = calib_pids & val_pids

    if overlap_tc or overlap_tv or overlap_cv:
        print("CRITICAL LEAKAGE DETECTED: Patient overlap across splits!", file=sys.stderr)
        sys.exit(1)

    # 6. Assign Split Roles & Compute Hashes
    df["split_role"] = "unassigned"
    df.loc[df["patient_id"].isin(train_pids), "split_role"] = "training"
    df.loc[df["patient_id"].isin(calib_pids), "split_role"] = "calibration"
    df.loc[df["patient_id"].isin(val_pids), "split_role"] = "internal_validation"

    # Generate stable study IDs and compute image hashes
    study_ids = []
    image_hashes = []

    for idx, row in df.iterrows():
        path_str = str(row["Path"])
        pid = str(row["patient_id"])
        study_ids.append(generate_stable_study_id(pid, path_str, idx))

        full_img_path = args.data_root / path_str
        if full_img_path.exists():
            image_hashes.append(compute_image_sha256(full_img_path))
        else:
            image_hashes.append(hashlib.sha256(path_str.encode()).hexdigest())

    df["study_id"] = study_ids
    df["image_path"] = df["Path"]
    df["image_sha256"] = image_hashes

    # Check for Duplicate Image Hashes across splits
    split_hash_groups = df.groupby("image_sha256")["split_role"].nunique()
    cross_split_duplicates = split_hash_groups[split_hash_groups > 1].index.tolist()
    if cross_split_duplicates:
        print(f"CRITICAL ERROR: {len(cross_split_duplicates)} duplicate image hashes found across different splits!", file=sys.stderr)
        sys.exit(1)

    # Format output CSVs
    output_cols = ["study_id", "patient_id", "image_path", "split_role", "image_sha256"] + target_labels
    train_df = df[df["split_role"] == "training"][output_cols].reset_index(drop=True)
    calib_df = df[df["split_role"] == "calibration"][output_cols].reset_index(drop=True)
    val_df = df[df["split_role"] == "internal_validation"][output_cols].reset_index(drop=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "train.csv"
    calib_path = args.output_dir / "calibration.csv"
    val_path = args.output_dir / "internal_validation.csv"
    manifest_path = args.output_dir / "manifest.json"
    integrity_path = args.output_dir / "integrity_report.json"

    train_df.to_csv(train_path, index=False)
    calib_df.to_csv(calib_path, index=False)
    val_df.to_csv(val_path, index=False)

    # Prevalence Validation Table
    prevalence_table = {}
    print("\n" + "=" * 60)
    print(f"{'Label':<20s} {'Train (80%)':<12s} {'Calib (10%)':<12s} {'Val (10%)':<12s}")
    print("-" * 60)

    for label in target_labels:
        t_pos = int(np.sum([normalize_uncertain_label(v, label, args.uncertainty_policy) for v in train_df[label]]))
        c_pos = int(np.sum([normalize_uncertain_label(v, label, args.uncertainty_policy) for v in calib_df[label]]))
        v_pos = int(np.sum([normalize_uncertain_label(v, label, args.uncertainty_policy) for v in val_df[label]]))

        prevalence_table[label] = {
            "train_positive": t_pos,
            "calib_positive": c_pos,
            "val_positive": v_pos,
            "train_prevalence": float(t_pos / max(1, len(train_df))),
            "calib_prevalence": float(c_pos / max(1, len(calib_df))),
            "val_prevalence": float(v_pos / max(1, len(val_df))),
        }
        print(f"{label:<20s} {t_pos:<12d} {c_pos:<12d} {v_pos:<12d}")

        # Check that rare labels do not completely vanish if present in dataset
        total_pos = t_pos + c_pos + v_pos
        if total_pos >= 10 and (c_pos == 0 or v_pos == 0):
            print(f"WARNING: Label '{label}' has {total_pos} positives but vanishes in calibration or validation!", file=sys.stderr)

    print("=" * 60)

    # Generate Manifest & Integrity Report
    manifest = {
        "schema_version": "1.0",
        "protocol_version": "0.1",
        "seed": args.seed,
        "source_csv": str(args.train_csv),
        "source_csv_sha256": compute_file_sha256(args.train_csv),
        "uncertainty_policy": args.uncertainty_policy,
        "view": args.view,
        "labels": target_labels,
        "splits": {
            "train": {
                "role": "training",
                "csv": "train.csv",
                "csv_sha256": compute_file_sha256(train_path),
                "patients": len(train_pids),
                "studies": len(train_df),
            },
            "calibration": {
                "role": "calibration",
                "csv": "calibration.csv",
                "csv_sha256": compute_file_sha256(calib_path),
                "patients": len(calib_pids),
                "studies": len(calib_df),
            },
            "internal_validation": {
                "role": "internal_validation",
                "csv": "internal_validation.csv",
                "csv_sha256": compute_file_sha256(val_path),
                "patients": len(val_pids),
                "studies": len(val_df),
            },
        },
        "patient_overlap": 0,
        "study_overlap": 0,
        "duplicate_image_hash_overlap": 0,
        "label_prevalence": prevalence_table,
    }

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    integrity_report = {
        "verified_zero_patient_leakage": True,
        "patient_overlap_count": 0,
        "study_overlap_count": 0,
        "duplicate_hash_overlap_count": len(cross_split_duplicates),
        "total_patients": len(unique_patients),
        "total_studies": len(df),
        "created_manifest_sha256": compute_file_sha256(manifest_path),
    }
    integrity_path.write_text(json.dumps(integrity_report, indent=2), encoding="utf-8")

    print(f"\nManifest successfully created at: {manifest_path}")
    print(f"Integrity report written to:     {integrity_path}")


if __name__ == "__main__":
    main()
