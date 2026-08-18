from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a CheXpert checkpoint, calculate stratified bootstrap CIs, and calibrate optimal thresholds.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--csv", type=Path)
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
    parser.add_argument("--output-thresholds", type=Path, help="Save optimal calibrated thresholds and metrics to json")
    return parser.parse_args()


def calculate_optimal_thresholds(
    targets: list[list[float]],
    probs: list[list[float]],
    labels: list[str],
    masks: list[list[float]] | None = None,
) -> dict[str, object]:
    thresholds_dict: dict[str, float] = {}
    metrics_dict: dict[str, object] = {}

    for label_index, label in enumerate(labels):
        if masks is not None:
            valid_idx = [i for i, m in enumerate(masks) if m[label_index] > 0.5]
        else:
            valid_idx = list(range(len(targets)))

        if not valid_idx:
            thresholds_dict[label] = 0.5
            metrics_dict[label] = {"label": label, "threshold": 0.5, "f1": 0.0, "valid_count": 0}
            continue

        y_true = np.array([1 if targets[i][label_index] >= 0.5 else 0 for i in valid_idx])
        y_prob = np.array([probs[i][label_index] for i in valid_idx])

        if len(np.unique(y_true)) < 2:
            thresholds_dict[label] = 0.5
            metrics_dict[label] = {"label": label, "threshold": 0.5, "f1": 0.0, "valid_count": len(valid_idx)}
            continue

        precisions, recalls, candidates = precision_recall_curve(y_true, y_prob)
        f1_scores = (2 * precisions * recalls) / np.clip(precisions + recalls, 1e-8, None)
        best_idx = np.argmax(f1_scores)
        best_threshold = float(candidates[best_idx]) if best_idx < len(candidates) else 0.5
        best_threshold = float(np.clip(best_threshold, 0.1, 0.9))

        y_pred = (y_prob >= best_threshold).astype(int)
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        precision = float(precisions[best_idx])
        recall = float(recalls[best_idx])

        thresholds_dict[label] = best_threshold
        metrics_dict[label] = {
            "label": label,
            "threshold": best_threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "valid_count": len(valid_idx),
            "positive_count": int(np.sum(y_true == 1)),
            "negative_count": int(np.sum(y_true == 0)),
        }

    return {
        "thresholds": thresholds_dict,
        "metrics": metrics_dict,
    }


def compute_stratified_bootstrap_auc_ci(y_true: np.ndarray, y_score: np.ndarray, n_boot: int = 1000, seed: int = 42) -> tuple[float, float, float]:
    """
    Stratified bootstrap resampling (maintains class balance in each bootstrap replicate).
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


def label_aucs(
    targets: list[list[float]],
    probs: list[list[float]],
    labels: list[str],
    masks: list[list[float]] | None = None,
    seed: int = 42,
) -> dict[str, object]:
    scores: dict[str, object] = {}
    for label_index, label in enumerate(labels):
        if masks is not None:
            valid_idx = [i for i, m in enumerate(masks) if m[label_index] > 0.5]
        else:
            valid_idx = list(range(len(targets)))

        if not valid_idx:
            scores[label] = {"auc": None, "ci_lower": None, "ci_upper": None, "valid_count": 0}
            continue

        y_true = np.array([1.0 if targets[i][label_index] >= 0.5 else 0.0 for i in valid_idx])
        y_score = np.array([probs[i][label_index] for i in valid_idx])

        if len(np.unique(y_true)) < 2:
            scores[label] = {"auc": None, "ci_lower": None, "ci_upper": None, "valid_count": len(valid_idx)}
        else:
            try:
                point_auc = float(roc_auc_score(y_true, y_score))
                _, lower, upper = compute_stratified_bootstrap_auc_ci(y_true, y_score, n_boot=1000, seed=seed)
                scores[label] = {
                    "auc": point_auc,
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "ci_95": f"{point_auc:.3f} ({lower:.3f}–{upper:.3f})",
                    "valid_count": len(valid_idx),
                }
            except Exception:
                scores[label] = {"auc": None, "ci_lower": None, "ci_upper": None, "valid_count": len(valid_idx)}
    return scores


@torch.inference_mode()
def main() -> None:
    args = parse_args()

    # Reproducibility seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    predictor = CheXpertPredictor(args.checkpoint)
    if predictor.model is None:
        raise RuntimeError("Checkpoint failed to load.")
    metadata = getattr(predictor, "metadata", {})
    view = args.view or metadata.get("view", "frontal")

    csv_path = args.csv or args.data_root / "valid.csv"
    if not csv_path.exists():
        csv_path = args.data_root / "train.csv"

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
        pin_memory=torch.cuda.is_available(),
    )

    criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
    total_loss = 0.0
    targets_all: list[list[float]] = []
    probs_all: list[list[float]] = []
    masks_all: list[list[float]] = []

    for item in loader:
        images = item[0].to(predictor.device)
        targets = item[1].to(predictor.device)
        masks = item[2].to(predictor.device) if len(item) > 2 else torch.ones_like(targets)

        logits = predictor.model(images)
        bce = criterion(logits, targets)
        loss = (bce * masks).sum() / masks.sum().clamp(min=1.0)
        total_loss += float(loss.detach().cpu()) * images.size(0)

        targets_all.extend(targets.detach().cpu().tolist())
        probs_all.extend(torch.sigmoid(logits).detach().cpu().tolist())
        masks_all.extend(masks.detach().cpu().tolist())

    aucs_dict = label_aucs(targets_all, probs_all, predictor.labels, masks=masks_all, seed=args.seed)
    valid_aucs = [v["auc"] for v in aucs_dict.values() if v["auc"] is not None]
    mean_auc = float(sum(valid_aucs) / len(valid_aucs)) if valid_aucs else None

    calibration = calculate_optimal_thresholds(targets_all, probs_all, predictor.labels, masks=masks_all)

    commit_sha = get_git_commit_sha()
    ckpt_hash = compute_file_sha256(args.checkpoint)
    dataset_hash = compute_file_sha256(csv_path)

    result = {
        "commit_sha": commit_sha,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": ckpt_hash,
        "dataset_csv": str(csv_path),
        "dataset_sha256": dataset_hash,
        "seed": args.seed,
        "uncertain_policy": args.uncertain_policy,
        "view": view,
        "rows": len(dataset),
        "labels": predictor.labels,
        "model": predictor.architecture,
        "loss": total_loss / max(1, len(dataset)),
        "mean_auc": mean_auc,
        "label_auc": {k: v["auc"] for k, v in aucs_dict.items()},
        "label_auc_ci": aucs_dict,
        "thresholds": calibration["thresholds"],
        "metrics": calibration["metrics"],
        "disclaimer": "Research prototype only. Do not use these results for medical decisions.",
    }

    print(json.dumps(result, indent=2))

    if args.output_thresholds:
        args.output_thresholds.parent.mkdir(parents=True, exist_ok=True)
        args.output_thresholds.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nSaved calibrated metrics & SHA-256 manifest to: {args.output_thresholds}")


if __name__ == "__main__":
    main()
