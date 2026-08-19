from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import DEFAULT_LABELS


def compute_midrank(x: np.ndarray) -> np.ndarray:
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def fast_delong(predictions: np.ndarray, ground_truth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_models = predictions.shape[0]
    pos_mask = (ground_truth == 1)
    neg_mask = (ground_truth == 0)
    m = np.sum(pos_mask)
    n = np.sum(neg_mask)

    if m == 0 or n == 0:
        return np.zeros(n_models), np.zeros((n_models, n_models))

    V10 = np.empty((n_models, m), dtype=float)
    V01 = np.empty((n_models, n), dtype=float)
    aucs = np.empty(n_models, dtype=float)

    for k in range(n_models):
        preds = predictions[k]
        pos_preds = preds[pos_mask]
        neg_preds = preds[neg_mask]

        all_preds = np.concatenate([pos_preds, neg_preds])
        ranks = compute_midrank(all_preds)

        pos_ranks = ranks[:m]
        neg_ranks = ranks[m:]

        V10[k] = (pos_ranks - compute_midrank(pos_preds)) / n
        V01[k] = 1.0 - (neg_ranks - compute_midrank(neg_preds)) / m
        aucs[k] = np.mean(V10[k])

    S10 = np.cov(V10) if m > 1 else np.zeros((n_models, n_models))
    S01 = np.cov(V01) if n > 1 else np.zeros((n_models, n_models))

    if n_models == 1:
        S10 = np.array([[S10]])
        S01 = np.array([[S01]])

    sigma = S10 / m + S01 / n
    return aucs, sigma


def delong_paired_test(
    preds_a: np.ndarray,
    preds_b: np.ndarray,
    ground_truth: np.ndarray,
) -> tuple[float, float, float, float]:
    preds = np.vstack([preds_a, preds_b])
    aucs, sigma = fast_delong(preds, ground_truth)
    auc_diff = aucs[0] - aucs[1]
    variance = sigma[0, 0] + sigma[1, 1] - 2 * sigma[0, 1]

    if variance <= 1e-12:
        return float(aucs[0]), float(aucs[1]), 0.0, 1.0

    z_stat = auc_diff / np.sqrt(variance)
    p_val = 2.0 * (1.0 - stats.norm.cdf(abs(z_stat)))
    return float(aucs[0]), float(aucs[1]), float(z_stat), float(p_val)


def holm_bonferroni_correction(p_values: Sequence[float]) -> list[float]:
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * m
    running_max = 0.0

    for rank, (orig_idx, p) in enumerate(indexed):
        adj_p = p * (m - rank)
        adj_p = min(1.0, max(running_max, adj_p))
        running_max = adj_p
        adjusted[orig_idx] = adj_p

    return adjusted


def paired_bootstrap_delta_auc(
    preds_a: np.ndarray,
    preds_b: np.ndarray,
    ground_truth: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
) -> tuple[float, float, float]:
    rng = np.random.RandomState(seed)
    pos_idx = np.where(ground_truth == 1)[0]
    neg_idx = np.where(ground_truth == 0)[0]
    m, n = len(pos_idx), len(neg_idx)

    if m == 0 or n == 0:
        return 0.0, 0.0, 0.0

    diffs = []
    for _ in range(n_boot):
        sample_pos = rng.choice(pos_idx, size=m, replace=True)
        sample_neg = rng.choice(neg_idx, size=n, replace=True)
        idx = np.concatenate([sample_pos, sample_neg])
        try:
            auc_a = roc_auc_score(ground_truth[idx], preds_a[idx])
            auc_b = roc_auc_score(ground_truth[idx], preds_b[idx])
            diffs.append(auc_a - auc_b)
        except Exception:
            continue

    if not diffs:
        return 0.0, 0.0, 0.0

    lower, upper = np.percentile(diffs, [2.5, 97.5])
    return float(np.mean(diffs)), float(lower), float(upper)


def patient_level_paired_bootstrap(
    patient_ids: list[str],
    merged_df: pd.DataFrame,
    labels: list[str],
    n_boot: int = 2000,
    seed: int = 42,
) -> tuple[dict[str, tuple[float, float, float]], tuple[float, float, float]]:
    rng = np.random.RandomState(seed)
    unique_patients = np.unique(patient_ids)
    n_patients = len(unique_patients)

    patient_to_indices: dict[str, list[int]] = {}
    for idx, pid in enumerate(patient_ids):
        if pid not in patient_to_indices:
            patient_to_indices[pid] = []
        patient_to_indices[pid].append(idx)

    label_deltas: dict[str, list[float]] = {lbl: [] for lbl in labels}
    macro_deltas: list[float] = []

    for _ in range(n_boot):
        sampled_pids = rng.choice(unique_patients, size=n_patients, replace=True)
        sampled_indices = []
        for pid in sampled_pids:
            sampled_indices.extend(patient_to_indices[pid])

        sub_df = merged_df.iloc[sampled_indices]

        boot_aucs_a = []
        boot_aucs_b = []

        for label in labels:
            y_raw = sub_df[f"{label}_target"].values
            mask_raw = sub_df[f"{label}_mask"].values if f"{label}_mask" in sub_df.columns else np.ones(len(sub_df))
            v_idx = np.where(mask_raw > 0.5)[0]

            if len(v_idx) == 0:
                continue

            y_t = np.array([1 if y_raw[i] >= 0.5 else 0 for i in v_idx])
            pa = sub_df[f"{label}_prob_a"].values[v_idx]
            pb = sub_df[f"{label}_prob_b"].values[v_idx]

            if len(np.unique(y_t)) > 1:
                try:
                    auc_a = roc_auc_score(y_t, pa)
                    auc_b = roc_auc_score(y_t, pb)
                    d = auc_a - auc_b
                    label_deltas[label].append(d)
                    boot_aucs_a.append(auc_a)
                    boot_aucs_b.append(auc_b)
                except Exception:
                    pass

        if boot_aucs_a and boot_aucs_b and len(boot_aucs_a) == len(boot_aucs_b):
            macro_deltas.append(float(np.mean(boot_aucs_a) - np.mean(boot_aucs_b)))

    # Compute CIs
    per_label_ci = {}
    for lbl in labels:
        dist = label_deltas[lbl]
        if len(dist) >= 100:
            low, up = np.percentile(dist, [2.5, 97.5])
            per_label_ci[lbl] = (float(np.mean(dist)), float(low), float(up))
        else:
            per_label_ci[lbl] = (0.0, 0.0, 0.0)

    if len(macro_deltas) >= 100:
        low, up = np.percentile(macro_deltas, [2.5, 97.5])
        macro_ci = (float(np.mean(macro_deltas)), float(low), float(up))
    else:
        macro_ci = (0.0, 0.0, 0.0)

    return per_label_ci, macro_ci


def compare_two_prediction_files(
    csv_a: Path,
    csv_b: Path,
    labels: list[str] | None = None,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict[str, object]:
    df_a = pd.read_csv(csv_a)
    df_b = pd.read_csv(csv_b)

    target_labels = labels or DEFAULT_LABELS

    # Strict Validation: study_id must exist and be unique in both files
    if "study_id" not in df_a.columns:
        raise ValueError(f"CRITICAL ERROR: Prediction file A ({csv_a}) is missing 'study_id' column!")
    if "study_id" not in df_b.columns:
        raise ValueError(f"CRITICAL ERROR: Prediction file B ({csv_b}) is missing 'study_id' column!")

    if df_a["study_id"].duplicated().any():
        raise ValueError(f"CRITICAL ERROR: Duplicate study_id found in Model A predictions ({csv_a})!")
    if df_b["study_id"].duplicated().any():
        raise ValueError(f"CRITICAL ERROR: Duplicate study_id found in Model B predictions ({csv_b})!")

    set_a = set(df_a["study_id"])
    set_b = set(df_b["study_id"])
    if set_a != set_b:
        diff_ab = set_a - set_b
        diff_ba = set_b - set_a
        raise ValueError(f"STUDY ALIGNMENT MISMATCH: Model A has {len(diff_ab)} unique studies not in B, B has {len(diff_ba)} not in A!")

    # Merge one-to-one strictly
    merged = pd.merge(df_a, df_b, on="study_id", suffixes=("_a", "_b"), validate="one_to_one")
    if len(merged) != len(df_a):
        raise RuntimeError("Inner join lost rows during study_id alignment!")

    # Verify ground truth targets match perfectly
    for label in target_labels:
        col_ta = f"{label}_target_a" if f"{label}_target_a" in merged.columns else f"{label}_target"
        col_tb = f"{label}_target_b" if f"{label}_target_b" in merged.columns else f"{label}_target"
        if col_ta in merged.columns and col_tb in merged.columns:
            if not np.allclose(merged[col_ta].fillna(0), merged[col_tb].fillna(0)):
                raise ValueError(f"TARGET MISMATCH: Ground truth targets for '{label}' differ between Model A and Model B!")
            merged[f"{label}_target"] = merged[col_ta]
            if f"{label}_mask_a" in merged.columns:
                merged[f"{label}_mask"] = merged[f"{label}_mask_a"]

    # Observed point estimates directly from original data
    results_per_label = {}
    p_values_raw = []
    obs_aucs_a = []
    obs_aucs_b = []

    for label in target_labels:
        prob_a = merged[f"{label}_prob_a"].values
        prob_b = merged[f"{label}_prob_b"].values
        y_raw = merged[f"{label}_target"].values
        mask_raw = merged[f"{label}_mask"].values if f"{label}_mask" in merged.columns else np.ones(len(merged))

        v_idx = np.where(mask_raw > 0.5)[0]
        y_true = np.array([1 if y_raw[i] >= 0.5 else 0 for i in v_idx])
        pa = prob_a[v_idx]
        pb = prob_b[v_idx]

        if len(np.unique(y_true)) > 1:
            auc_a, auc_b, z_stat, p_val = delong_paired_test(pa, pb, y_true)
            obs_delta = float(auc_a - auc_b)
            obs_aucs_a.append(auc_a)
            obs_aucs_b.append(auc_b)
        else:
            auc_a, auc_b, z_stat, p_val = None, None, 0.0, 1.0
            obs_delta = 0.0

        p_values_raw.append(p_val)
        results_per_label[label] = {
            "model_a_auc": auc_a,
            "model_b_auc": auc_b,
            "observed_delta_auc": obs_delta,
            "delong_z": z_stat,
            "delong_p_raw": p_val,
        }

    # Patient-level paired bootstrap for 95% CIs
    patient_ids = list(merged.get("patient_id_a", merged.get("patient_id", [f"p_{i}" for i in range(len(merged))])))
    per_label_ci, macro_ci = patient_level_paired_bootstrap(patient_ids, merged, target_labels, n_boot=n_boot, seed=seed)

    # Apply Holm-Bonferroni correction across 5 labels
    p_adj = holm_bonferroni_correction(p_values_raw)

    for idx, label in enumerate(target_labels):
        ci = per_label_ci.get(label, (0.0, 0.0, 0.0))
        results_per_label[label]["delta_auc_95_ci"] = f"{results_per_label[label]['observed_delta_auc']:+.4f} ({ci[1]:+.4f} to {ci[2]:+.4f})"
        results_per_label[label]["ci_lower"] = ci[1]
        results_per_label[label]["ci_upper"] = ci[2]
        results_per_label[label]["delong_p_holm_adj"] = p_adj[idx]
        results_per_label[label]["statistically_significant"] = bool(p_adj[idx] < 0.05)

    obs_macro_delta = float(np.mean(obs_aucs_a) - np.mean(obs_aucs_b)) if obs_aucs_a and obs_aucs_b else 0.0

    return {
        "schema_version": "1.0",
        "protocol_version": "0.1",
        "model_a_file": str(csv_a),
        "model_b_file": str(csv_b),
        "aligned_studies": len(merged),
        "bootstrap_resamples": n_boot,
        "macro_comparison": {
            "observed_macro_delta_auc": obs_macro_delta,
            "macro_delta_95_ci": f"{obs_macro_delta:+.4f} ({macro_ci[1]:+.4f} to {macro_ci[2]:+.4f})",
            "ci_lower": macro_ci[1],
            "ci_upper": macro_ci[2],
        },
        "results_per_label": results_per_label,
    }


def main():
    parser = argparse.ArgumentParser(description="Strict Paired Model Comparison with Patient-Level Bootstrap and Holm-Corrected DeLong Test.")
    parser.add_argument("--model-a-preds", type=Path, required=True, help="Predictions CSV from Model A")
    parser.add_argument("--model-b-preds", type=Path, required=True, help="Predictions CSV from Model B")
    parser.add_argument("--n-boot", type=int, default=2000, help="Number of bootstrap resamples (>= 2000)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, help="Save statistical comparison JSON")
    args = parser.parse_args()

    comparison = compare_two_prediction_files(args.model_a_preds, args.model_b_preds, n_boot=args.n_boot, seed=args.seed)

    print("\n=== PAIRED STATISTICAL COMPARISON REPORT ===")
    print(f"Aligned Studies: {comparison['aligned_studies']} | Bootstrap Resamples: {comparison['bootstrap_resamples']}")
    print(f"Macro Delta AUROC: {comparison['macro_comparison']['macro_delta_95_ci']}")
    for lbl, res in comparison["results_per_label"].items():
        sig_str = "** SIGNIFICANT **" if res["statistically_significant"] else "Not Significant"
        print(f"  {lbl:20s}: Delta AUC = {res['delta_auc_95_ci']} | p_raw = {res['delong_p_raw']:.4f} | p_adj (Holm) = {res['delong_p_holm_adj']:.4f} [{sig_str}]")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
        print(f"\nSaved statistical comparison to: {args.output}")


if __name__ == "__main__":
    main()
