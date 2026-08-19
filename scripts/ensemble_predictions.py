from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import DEFAULT_LABELS


def ensemble_prediction_files(csv_paths: list[Path], output_path: Path, labels: list[str] | None = None) -> pd.DataFrame:
    if not csv_paths or len(csv_paths) != 5:
        raise ValueError(f"ENSEMBLE PROTOCOL ERROR: Expected exactly 5 seed prediction CSVs (seeds 42..46), got {len(csv_paths)}!")

    target_labels = labels or DEFAULT_LABELS
    dfs = [pd.read_csv(p) for p in csv_paths]

    base_df = dfs[0].copy()
    base_study_ids = list(base_df["study_id"])
    base_patient_ids = list(base_df["patient_id"])

    # Strict Validation across all 5 seeds
    for idx, df in enumerate(dfs[1:], start=1):
        if "study_id" not in df.columns:
            raise ValueError(f"ENSEMBLE ERROR: File '{csv_paths[idx]}' missing 'study_id' column!")
        if "patient_id" not in df.columns:
            raise ValueError(f"ENSEMBLE ERROR: File '{csv_paths[idx]}' missing 'patient_id' column!")

        if list(df["study_id"]) != base_study_ids:
            if set(df["study_id"]) != set(base_study_ids):
                raise ValueError(
                    f"ENSEMBLE STUDY MISMATCH: Study ID set mismatch between '{csv_paths[0]}' and '{csv_paths[idx]}'!"
                )
            # Re-index to match base_df
            df = df.set_index("study_id").loc[base_study_ids].reset_index()
            dfs[idx] = df

        if list(df["patient_id"]) != base_patient_ids:
            raise ValueError(
                f"ENSEMBLE PATIENT MISMATCH: Patient IDs do not match between '{csv_paths[0]}' and '{csv_paths[idx]}'!"
            )

        # Verify ground truth targets & masks match across all seeds
        for label in target_labels:
            t_col = f"{label}_target"
            if t_col in base_df.columns and t_col in df.columns:
                if not np.allclose(base_df[t_col].fillna(0), df[t_col].fillna(0)):
                    raise ValueError(f"ENSEMBLE TARGET MISMATCH: Ground truth for '{label}' differs in '{csv_paths[idx]}'!")

    ensemble_df = base_df[["study_id", "patient_id"]].copy()

    for label in target_labels:
        prob_matrix = np.zeros((len(base_df), len(dfs)), dtype=float)
        for i, df in enumerate(dfs):
            prob_matrix[:, i] = df[f"{label}_prob"].values

        # Strict Mean probability ensemble across 5 seeds
        mean_prob = np.mean(prob_matrix, axis=1)
        ensemble_df[f"{label}_prob"] = mean_prob

        # Preserve ground truth target & mask
        if f"{label}_target" in base_df.columns:
            ensemble_df[f"{label}_target"] = base_df[f"{label}_target"]
        if f"{label}_mask" in base_df.columns:
            ensemble_df[f"{label}_mask"] = base_df[f"{label}_mask"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ensemble_df.to_csv(output_path, index=False)
    print(f"5-Seed Ensemble predictions successfully saved to: {output_path} ({len(ensemble_df)} studies)")
    return ensemble_df


def main():
    parser = argparse.ArgumentParser(description="Ensemble 5-seed probability predictions via study_id alignment.")
    parser.add_argument("--inputs", type=Path, nargs="+", required=True, help="Prediction CSVs from seeds 42..46")
    parser.add_argument("--output", type=Path, required=True, help="Output ensemble predictions CSV")
    args = parser.parse_args()

    ensemble_prediction_files(args.inputs, args.output)


if __name__ == "__main__":
    main()
