from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.dataset import CheXpertDataset
from app.model import CheXpertPredictor


def get_git_commit_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "UNKNOWN"


def compute_file_sha256(filepath: Path) -> str:
    if not filepath.exists():
        return "NOT_FOUND"
    h = hashlib.sha256()
    with filepath.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_stratified_bootstrap_auc_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """
    Stratified bootstrap resampling with 2,000 resamples (95% percentile CI).
    """
    rng = np.random.RandomState(seed)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]

    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return 0.0, 0.0, 0.0

    aucs = []
    for _ in range(n_boot):
        sample_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        sample_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        sample_idx = np.concatenate([sample_pos, sample_neg])
        try:
            score = roc_auc_score(y_true[sample_idx], y_score[sample_idx])
            aucs.append(score)
        except Exception:
            continue

    if not aucs:
        return 0.0, 0.0, 0.0
    lower, upper = np.percentile(aucs, [2.5, 97.5])
    return float(np.mean(aucs)), float(lower), float(upper)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a CheXpert checkpoint on the Locked Test Set.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Trained model checkpoint .pt")
    parser.add_argument("--data-root", type=Path, required=True, help="Data root directory")
    parser.add_argument("--csv", type=Path, help="Evaluation dataset CSV (default: archive/valid.csv)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--uncertain-policy",
        choices=["u_ones_zeros", "smooth", "zero", "one", "ignore"],
        default="u_ones_zeros",
    )
    parser.add_argument("--view", choices=["frontal", "lateral", "all"], default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--frozen-thresholds",
        type=Path,
        help="Path to frozen_thresholds.json artifact (MANDATORY for --locked-test mode)",
    )
    parser.add_argument(
        "--locked-test",
        action="store_true",
        default=False,
        help="Enforce strict locked evaluation protocol (strictly forbids threshold fitting on test data)",
    )
    parser.add_argument("--output-predictions", type=Path, help="Export per-study anonymized predictions CSV")
    parser.add_argument("--output-report", type=Path, help="Export evaluation summary JSON")
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()

    # Reproducibility seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    csv_path = args.csv or args.data_root / "valid.csv"
    if not csv_path.exists():
        csv_path = args.data_root / "train.csv"

    # Strict Locked Test Protocol Guard
    thresholds_dict: dict[str, float] = {}
    threshold_meta: dict[str, object] = {}

    if args.locked_test:
        if not args.frozen_thresholds or not args.frozen_thresholds.exists():
            default_frozen = PROJECT_ROOT / "outputs" / "evaluation" / "frozen_thresholds.json"
            if default_frozen.exists():
                args.frozen_thresholds = default_frozen
            else:
                raise RuntimeError(
                    "LOCKED TEST PROTOCOL VIOLATION: --locked-test mode requires a valid frozen threshold artifact "
                    "calibrated strictly on the calibration set! Provide --frozen-thresholds."
                )

        print(f"Loading frozen thresholds from: {args.frozen_thresholds}")
        th_data = json.loads(args.frozen_thresholds.read_text(encoding="utf-8"))
        thresholds_dict = {k: float(v) for k, v in th_data.get("thresholds", {}).items()}
        threshold_meta = {
            "frozen_thresholds_file": str(args.frozen_thresholds),
            "frozen_thresholds_sha256": compute_file_sha256(args.frozen_thresholds),
            "calibration_method": th_data.get("calibration_method", "optimal_f1"),
        }
    elif args.frozen_thresholds and args.frozen_thresholds.exists():
        th_data = json.loads(args.frozen_thresholds.read_text(encoding="utf-8"))
        thresholds_dict = {k: float(v) for k, v in th_data.get("thresholds", {}).items()}

    predictor = CheXpertPredictor(args.checkpoint, thresholds=thresholds_dict)
    if predictor.model is None:
        raise RuntimeError("Checkpoint failed to load.")

    metadata = getattr(predictor, "metadata", {})
    view = args.view or metadata.get("view", "frontal")

    dataset = CheXpertDataset(
        csv_path,
        args.data_root,
        predictor.transform,
        labels=predictor.labels,
        uncertain_policy=args.uncertain_policy,
        view=view,
    )
    if args.limit:
        dataset = Subset(dataset, range(min(args.limit, len(dataset))))

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    targets_all: list[list[float]] = []
    probs_all: list[list[float]] = []
    masks_all: list[list[float]] = []

    print(f"Evaluating {len(dataset)} studies from {csv_path.name}...")
    for item in loader:
        images = item[0].to(predictor.device)
        targets = item[1].to(predictor.device)
        masks = item[2].to(predictor.device) if len(item) > 2 else torch.ones_like(targets)

        logits = predictor.model(images)
        probs = torch.sigmoid(logits)

        targets_all.extend(targets.detach().cpu().tolist())
        probs_all.extend(probs.detach().cpu().tolist())
        masks_all.extend(masks.detach().cpu().tolist())

    targets_arr = np.array(targets_all)
    probs_arr = np.array(probs_all)
    masks_arr = np.array(masks_all)

    # Compute Comprehensive Metrics
    label_metrics: dict[str, dict[str, object]] = {}
    valid_aucs = []

    for idx, label in enumerate(predictor.labels):
        valid_idx = np.where(masks_arr[:, idx] > 0.5)[0]
        if len(valid_idx) == 0:
            continue

        y_true = np.array([1 if targets_arr[i, idx] >= 0.5 else 0 for i in valid_idx])
        y_prob = probs_arr[valid_idx, idx]
        th = thresholds_dict.get(label, 0.5)
        y_pred = (y_prob >= th).astype(int)

        # AUROC & 95% Bootstrap CI
        if len(np.unique(y_true)) > 1:
            point_auc = float(roc_auc_score(y_true, y_prob))
            _, ci_lower, ci_upper = compute_stratified_bootstrap_auc_ci(y_true, y_prob, n_boot=2000, seed=args.seed)
            auprc = float(average_precision_score(y_true, y_prob))
            valid_aucs.append(point_auc)
        else:
            point_auc, ci_lower, ci_upper, auprc = None, None, None, None

        # Confusion Matrix & Diagnostic Metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        sens = float(tp / max(1, tp + fn))
        spec = float(tn / max(1, tn + fp))
        ppv = float(tp / max(1, tp + fp))
        npv = float(tn / max(1, tn + fn))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        brier = float(brier_score_loss(y_true, y_prob))

        label_metrics[label] = {
            "auroc": point_auc,
            "ci_95": f"{point_auc:.4f} ({ci_lower:.4f}–{ci_upper:.4f})" if point_auc is not None else "N/A",
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "auprc": auprc,
            "threshold": th,
            "sensitivity": sens,
            "specificity": spec,
            "ppv": ppv,
            "npv": npv,
            "f1_score": f1,
            "brier_score": brier,
            "confusion_matrix": {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)},
            "sample_count": len(valid_idx),
            "positive_count": int(tp + fn),
            "negative_count": int(tn + fp),
        }

    mean_auc = float(np.mean(valid_aucs)) if valid_aucs else None

    # Anonymized Predictions Export
    if args.output_predictions:
        pred_records = []
        underlying_df = dataset.dataset.frame if isinstance(dataset, Subset) else dataset.frame
        for i in range(len(dataset)):
            actual_row = underlying_df.iloc[dataset.indices[i]] if isinstance(dataset, Subset) else underlying_df.iloc[i]
            study_path = str(actual_row.get("Path", f"study_{i+1}"))
            study_hash = hashlib.sha256(study_path.encode()).hexdigest()[:12]
            record = {
                "study_id": f"study_{study_hash}",
                "patient_id": f"patient_{study_hash[:8]}",
            }
            for idx, label in enumerate(predictor.labels):
                record[f"{label}_prob"] = float(probs_arr[i, idx])
                record[f"{label}_pred"] = int(probs_arr[i, idx] >= thresholds_dict.get(label, 0.5))
                record[f"{label}_target"] = float(targets_arr[i, idx])
                record[f"{label}_mask"] = float(masks_arr[i, idx])
            pred_records.append(record)

        args.output_predictions.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(pred_records).to_csv(args.output_predictions, index=False)
        print(f"Exported anonymized predictions to: {args.output_predictions}")

    report = {
        "evaluation_mode": "locked_test" if args.locked_test else "standard",
        "git_commit": get_git_commit_sha(),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": compute_file_sha256(args.checkpoint),
        "dataset_csv": str(csv_path),
        "dataset_sha256": compute_file_sha256(csv_path),
        "threshold_metadata": threshold_meta,
        "model_architecture": predictor.architecture,
        "seed": args.seed,
        "num_studies": len(dataset),
        "mean_auroc": mean_auc,
        "metrics_per_label": label_metrics,
    }

    print("\n=== LOCKED EVALUATION BENCHMARK RESULTS ===")
    print(f"Mean AUROC: {mean_auc:.4f}" if mean_auc else "Mean AUROC: N/A")
    for lbl, m in label_metrics.items():
        auc_str = m["ci_95"]
        print(f"  {lbl:20s}: AUROC={auc_str} | Sens={m['sensitivity']:.3f} | Spec={m['specificity']:.3f} | F1={m['f1_score']:.3f}")

    if args.output_report:
        args.output_report.parent.mkdir(parents=True, exist_ok=True)
        args.output_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nSaved evaluation report to: {args.output_report}")


if __name__ == "__main__":
    main()
