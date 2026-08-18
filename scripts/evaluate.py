from __future__ import annotations

import argparse
import json
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a CheXpert checkpoint and calibrate optimal thresholds.")
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
    parser.add_argument("--output-thresholds", type=Path, help="Save optimal calibrated thresholds to json")
    return parser.parse_args()


def calculate_optimal_thresholds(targets: list[list[float]], probs: list[list[float]], labels: list[str]) -> dict[str, object]:
    thresholds_dict: dict[str, float] = {}
    metrics_dict: dict[str, object] = {}

    for label_index, label in enumerate(labels):
        y_true = np.array([1 if row[label_index] >= 0.5 else 0 for row in targets])
        y_prob = np.array([row[label_index] for row in probs])

        if len(np.unique(y_true)) < 2:
            thresholds_dict[label] = 0.5
            metrics_dict[label] = {"label": label, "threshold": 0.5, "f1": 0.0}
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
            "positive_count": int(np.sum(y_true == 1)),
            "negative_count": int(np.sum(y_true == 0)),
        }

    return {
        "thresholds": thresholds_dict,
        "metrics": metrics_dict,
    }


def label_aucs(targets: list[list[float]], probs: list[list[float]], labels: list[str]) -> dict[str, float | None]:
    scores: dict[str, float | None] = {}
    for label_index, label in enumerate(labels):
        y_true = [1.0 if row[label_index] >= 0.5 else 0.0 for row in targets]
        y_score = [row[label_index] for row in probs]
        if len(set(y_true)) < 2:
            scores[label] = None
        else:
            try:
                scores[label] = float(roc_auc_score(y_true, y_score))
            except Exception:
                scores[label] = None
    return scores


@torch.inference_mode()
def main() -> None:
    args = parse_args()
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

    criterion = torch.nn.BCEWithLogitsLoss()
    total_loss = 0.0
    targets_all: list[list[float]] = []
    probs_all: list[list[float]] = []

    for images, targets in loader:
        images = images.to(predictor.device)
        targets = targets.to(predictor.device)
        logits = predictor.model(images)
        loss = criterion(logits, targets)
        total_loss += float(loss.detach().cpu()) * images.size(0)
        targets_all.extend(targets.detach().cpu().tolist())
        probs_all.extend(torch.sigmoid(logits).detach().cpu().tolist())

    aucs = label_aucs(targets_all, probs_all, predictor.labels)
    valid_aucs = [value for value in aucs.values() if value is not None]
    mean_auc = float(sum(valid_aucs) / len(valid_aucs)) if valid_aucs else None

    calibration = calculate_optimal_thresholds(targets_all, probs_all, predictor.labels)

    result = {
        "checkpoint": str(args.checkpoint),
        "csv": str(csv_path),
        "view": view,
        "rows": len(dataset),
        "labels": predictor.labels,
        "model": predictor.architecture,
        "loss": total_loss / max(1, len(dataset)),
        "mean_auc": mean_auc,
        "label_auc": aucs,
        "thresholds": calibration["thresholds"],
        "metrics": calibration["metrics"],
        "disclaimer": "Research prototype only. Do not use these results for medical decisions.",
    }

    print(json.dumps(result, indent=2))

    if args.output_thresholds:
        args.output_thresholds.parent.mkdir(parents=True, exist_ok=True)
        args.output_thresholds.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nSaved calibrated thresholds to: {args.output_thresholds}")


if __name__ == "__main__":
    main()
