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
    if not csv_paths:
        raise ValueError("No prediction CSV files provided for ensemble!")

    target_labels = labels or DEFAULT_LABELS
    dfs = [pd.read_csv(p) for p in csv_paths]

    base_df = dfs[0].copy()
    base_study_ids = list(base_df["study_id"])

    # Verify all files have identical unique study IDs
    for idx, df in enumerate(dfs[1:], start=1):
        if "study_id" not in df.columns:
            raise ValueError(f"File {csv_paths[idx]} missing 'study_id' column!")
        if list(df["study_id"]) != base_study_ids:
            # Check if sorted set is equal
            if set(df["study_id"]) != set(base_study_ids):
                raise ValueError(f"Study ID set mismatch between {csv_paths[0]} and {csv_paths[idx]}!")
            # Re-index to match base_df
            df = df.set_index("study_id").loc[base_study_ids].reset_index()
            dfs[idx] = df

    ensemble_df = base_df[["study_id", "patient_id"]].copy()

    for label in target_labels:
        prob_matrix = np.zeros((len(base_df), len(dfs)), dtype=float)
        for i, df in enumerate(dfs):
            prob_matrix[:, i] = df[f"{label}_prob"].values

        # Mean probability ensemble across seeds
        mean_prob = np.mean(prob_matrix, axis=1)
        ensemble_df[f"{label}_prob"] = mean_prob

        # Preserve ground truth target & mask
        if f"{label}_target" in base_df.columns:
            ensemble_df[f"{label}_target"] = base_df[f"{label}_target"]
        if f"{label}_mask" in base_df.columns:
            ensemble_df[f"{label}_mask"] = base_df[f"{label}_mask"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ensemble_df.to_csv(output_path, index=False)
    print(f"Ensemble predictions saved to: {output_path} ({len(ensemble_df)} studies)")
    return ensemble_df


def main():
    parser = argparse.ArgumentParser(description="Ensemble multi-seed probability predictions via study_id alignment.")
    parser.add_argument("--inputs", type=Path, nargs="+", required=True, help="Prediction CSVs from seeds 42..46")
    parser.add_argument("--output", type=Path, required=True, help="Output ensemble predictions CSV")
    args = parser.parse_args()

    ensemble_prediction_files(args.inputs, args.output)


if __name__ == "__main__":
    main()
