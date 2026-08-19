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
    """
    Computes midranks for DeLong's test.
    """
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
    """
    Fast DeLong algorithm for calculating AUC covariance.
    predictions: shape (n_models, n_samples)
    ground_truth: shape (n_samples,) binary {0, 1}
    Returns: (aucs, cov_matrix)
    """
    n_models = predictions.shape[0]
    pos_mask = (ground_truth == 1)
    neg_mask = (ground_truth == 0)
    m = np.sum(pos_mask)
    n = np.sum(neg_mask)

    if m == 0 or n == 0:
        return np.zeros(n_models), np.zeros((n_models, n_models))

    # Structural components
    V10 = np.empty((n_models, m), dtype=float)
    V01 = np.empty((n_models, n), dtype=float)
    aucs = np.empty(n_models, dtype=float)

    for k in range(n_models):
        preds = predictions[k]
        pos_preds = preds[pos_mask]
        neg_preds = preds[neg_mask]

        # Combine and compute ranks
        all_preds = np.concatenate([pos_preds, neg_preds])
        ranks = compute_midrank(all_preds)

        pos_ranks = ranks[:m]
        neg_ranks = ranks[m:]

        # Placements
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
    """
    Computes paired DeLong test between two models on the same ground truth.
    Returns: (auc_a, auc_b, z_stat, p_value)
    """
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
    """
    Applies Holm-Bonferroni step-down correction for family-wise error rate control.
    """
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
    """
    Paired bootstrap for difference in AUC (AUC_A - AUC_B).
    """
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

    # Align strictly on study_id
    if "study_id" in df_a.columns and "study_id" in df_b.columns:
        merged = pd.merge(df_a, df_b, on="study_id", suffixes=("_model_a", "_model_b"))
    else:
        # Fallback to index alignment if row count matches
        assert len(df_a) == len(df_b), "Prediction files must contain matching rows or study_id column!"
        merged = df_a.join(df_b, lsuffix="_model_a", rsuffix="_model_b")

    results_per_label = {}
    p_values_raw = []

    for label in target_labels:
        prob_col_a = f"{label}_prob_model_a" if f"{label}_prob_model_a" in merged.columns else f"{label}_prob"
        prob_col_b = f"{label}_prob_model_b" if f"{label}_prob_model_b" in merged.columns else f"{label}_prob"
        target_col = f"{label}_target_model_a" if f"{label}_target_model_a" in merged.columns else f"{label}_target"
        mask_col = f"{label}_mask_model_a" if f"{label}_mask_model_a" in merged.columns else f"{label}_mask"

        if target_col not in merged.columns:
            target_col = label

        y_true_raw = merged[target_col].values
        probs_a = merged[prob_col_a].values
        probs_b = merged[prob_col_b].values

        if mask_col in merged.columns:
            valid_mask = merged[mask_col].values > 0.5
        else:
            valid_mask = ~np.isnan(y_true_raw)

        y_true = np.array([1 if val >= 0.5 else 0 for val in y_true_raw[valid_mask]])
        pa = probs_a[valid_mask]
        pb = probs_b[valid_mask]

        if len(np.unique(y_true)) > 1:
            auc_a, auc_b, z_stat, p_val = delong_paired_test(pa, pb, y_true)
            delta_mean, ci_low, ci_up = paired_bootstrap_delta_auc(pa, pb, y_true, n_boot=n_boot, seed=seed)
        else:
            auc_a, auc_b, z_stat, p_val = None, None, None, 1.0
            delta_mean, ci_low, ci_up = 0.0, 0.0, 0.0

        p_values_raw.append(p_val)
        results_per_label[label] = {
            "model_a_auc": auc_a,
            "model_b_auc": auc_b,
            "delta_auc_mean": delta_mean,
            "delta_auc_95_ci": f"{delta_mean:+.4f} ({ci_low:+.4f} to {ci_up:+.4f})",
            "delong_z": z_stat,
            "delong_p_raw": p_val,
        }

    # Apply Holm-Bonferroni correction
    p_adj = holm_bonferroni_correction(p_values_raw)
    for idx, label in enumerate(target_labels):
        results_per_label[label]["delong_p_holm_adj"] = p_adj[idx]
        results_per_label[label]["statistically_significant"] = bool(p_adj[idx] < 0.05)

    return {
        "model_a_file": str(csv_a),
        "model_b_file": str(csv_b),
        "aligned_studies": len(merged),
        "bootstrap_resamples": n_boot,
        "results_per_label": results_per_label,
    }


def main():
    parser = argparse.ArgumentParser(description="Paired Model Comparison with Paired Bootstrap and Holm-Corrected DeLong Test.")
    parser.add_argument("--model-a-preds", type=Path, required=True, help="Predictions CSV from Model A (e.g. ConvNeXt-Small)")
    parser.add_argument("--model-b-preds", type=Path, required=True, help="Predictions CSV from Model B (e.g. DenseNet-121)")
    parser.add_argument("--n-boot", type=int, default=2000, help="Number of bootstrap resamples (>= 2000)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, help="Save statistical comparison JSON")
    args = parser.parse_args()

    comparison = compare_two_prediction_files(args.model_a_preds, args.model_b_preds, n_boot=args.n_boot, seed=args.seed)

    print("\n=== PAIRED STATISTICAL COMPARISON REPORT (Model A vs Model B) ===")
    print(f"Aligned Studies: {comparison['aligned_studies']} | Bootstrap Resamples: {comparison['bootstrap_resamples']}")
    for lbl, res in comparison["results_per_label"].items():
        sig_str = "** SIGNIFICANT **" if res["statistically_significant"] else "Not Significant"
        print(f"  {lbl:20s}: Delta AUC = {res['delta_auc_95_ci']} | p_raw = {res['delong_p_raw']:.4f} | p_adj (Holm) = {res['delong_p_holm_adj']:.4f} [{sig_str}]")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
        print(f"\nSaved statistical comparison to: {args.output}")


if __name__ == "__main__":
    main()
