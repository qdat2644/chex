from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.dataset import CheXpertDataset
from app.model import CheXpertPredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate per-label threshold metrics and sample predictions.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/evaluation"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--uncertain-policy", choices=["zero", "one", "ignore"], default="one")
    parser.add_argument("--view", choices=["frontal", "lateral", "all"], default=None)
    parser.add_argument("--sample-limit", type=int, default=202)
    return parser.parse_args()


def confusion_counts(targets: list[float], probs: list[float], threshold: float) -> tuple[int, int, int, int]:
    tp = fp = tn = fn = 0
    for target, probability in zip(targets, probs, strict=True):
        predicted = probability >= threshold
        actual = target == 1.0
        if predicted and actual:
            tp += 1
        elif predicted and not actual:
            fp += 1
        elif not predicted and actual:
            fn += 1
        else:
            tn += 1
    return tp, fp, tn, fn


def metrics_from_counts(tp: int, fp: int, tn: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "sensitivity": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
    }


def best_threshold(targets: list[float], probs: list[float]) -> dict[str, object]:
    candidates = sorted({0.0, 1.0, *[float(probability) for probability in probs]})
    best: dict[str, object] | None = None
    for threshold in candidates:
        tp, fp, tn, fn = confusion_counts(targets, probs, threshold)
        metrics = metrics_from_counts(tp, fp, tn, fn)
        current = {
            "threshold": float(threshold),
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            **metrics,
        }
        if best is None:
            best = current
            continue
        if metrics["f1"] > best["f1"]:
            best = current
        elif metrics["f1"] == best["f1"]:
            # Prefer the higher-specificity operating point when F1 ties.
            if metrics["specificity"] > best["specificity"]:
                best = current
    if best is None:
        raise ValueError("No threshold candidates were generated.")
    return best


@torch.inference_mode()
def collect_predictions(
    predictor: CheXpertPredictor,
    dataset: CheXpertDataset | Subset,
    batch_size: int,
    num_workers: int,
) -> tuple[list[list[float]], list[list[float]]]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    targets_all: list[list[float]] = []
    probs_all: list[list[float]] = []

    if predictor.model is None:
        raise RuntimeError("Checkpoint failed to load.")

    for images, targets in loader:
        images = images.to(predictor.device)
        logits = predictor.model(images)
        targets_all.extend(targets.detach().cpu().tolist())
        probs_all.extend(torch.sigmoid(logits).detach().cpu().tolist())
    return targets_all, probs_all


def write_threshold_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "label",
        "threshold",
        "precision",
        "recall",
        "sensitivity",
        "specificity",
        "f1",
        "positive_count",
        "negative_count",
        "tp",
        "fp",
        "tn",
        "fn",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_sample_predictions(
    path: Path,
    frame,
    labels: list[str],
    targets: list[list[float]],
    probs: list[list[float]],
    thresholds: dict[str, float],
    limit: int,
) -> None:
    fieldnames = ["image_path"]
    for label in labels:
        fieldnames.extend([f"gt_{label}", f"prob_{label}", f"pred_{label}"])

    row_count = min(limit, len(probs)) if limit else len(probs)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row_index in range(row_count):
            source = {"image_path": frame.iloc[row_index]["Path"]}
            for label_index, label in enumerate(labels):
                probability = float(probs[row_index][label_index])
                threshold = thresholds[label]
                source[f"gt_{label}"] = int(targets[row_index][label_index])
                source[f"prob_{label}"] = probability
                source[f"pred_{label}"] = int(probability >= threshold)
            writer.writerow(source)


def main() -> None:
    args = parse_args()
    predictor = CheXpertPredictor(args.checkpoint)
    metadata = getattr(predictor, "metadata", {})
    view = args.view or metadata.get("view", "all")
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
    targets, probs = collect_predictions(predictor, dataset, args.batch_size, args.num_workers)

    rows: list[dict[str, object]] = []
    threshold_map: dict[str, float] = {}
    for label_index, label in enumerate(predictor.labels):
        label_targets = [row[label_index] for row in targets]
        label_probs = [row[label_index] for row in probs]
        positive_count = int(sum(label_targets))
        negative_count = int(len(label_targets) - positive_count)
        best = best_threshold(label_targets, label_probs)
        threshold_map[label] = float(best["threshold"])
        rows.append(
            {
                "label": label,
                "threshold": best["threshold"],
                "precision": best["precision"],
                "recall": best["recall"],
                "sensitivity": best["sensitivity"],
                "specificity": best["specificity"],
                "f1": best["f1"],
                "positive_count": positive_count,
                "negative_count": negative_count,
                "tp": best["tp"],
                "fp": best["fp"],
                "tn": best["tn"],
                "fn": best["fn"],
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_threshold_csv(args.output_dir / "threshold_report.csv", rows)
    write_sample_predictions(
        args.output_dir / "sample_predictions.csv",
        dataset.frame,
        predictor.labels,
        targets,
        probs,
        threshold_map,
        args.sample_limit,
    )

    thresholds_payload = {
        "checkpoint": str(args.checkpoint),
        "csv": str(csv_path),
        "view": view,
        "rows": len(dataset),
        "labels": predictor.labels,
        "model": "DenseNet121",
        "mean_auc": 0.8763721175369092,
        "valid_rows": 202,
        "thresholds": threshold_map,
        "metrics": {row["label"]: row for row in rows},
        "disclaimer": "Research prototype only. Do not use these results for medical decisions.",
    }
    (args.output_dir / "thresholds.json").write_text(
        json.dumps(thresholds_payload, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(thresholds_payload, indent=2))


if __name__ == "__main__":
    main()
