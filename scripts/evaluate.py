from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.dataset import CheXpertDataset
from app.model import CheXpertPredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a CheXpert checkpoint on a CSV split.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--uncertain-policy", choices=["zero", "one", "ignore"], default="zero")
    parser.add_argument("--view", choices=["frontal", "lateral", "all"], default=None)
    return parser.parse_args()


def label_aucs(targets: list[list[float]], probs: list[list[float]], labels: list[str]) -> dict[str, float | None]:
    scores: dict[str, float | None] = {}
    for label_index, label in enumerate(labels):
        y_true = [row[label_index] for row in targets]
        y_score = [row[label_index] for row in probs]
        if len(set(y_true)) < 2:
            scores[label] = None
        else:
            scores[label] = float(roc_auc_score(y_true, y_score))
    return scores


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    predictor = CheXpertPredictor(args.checkpoint)
    if predictor.model is None:
        raise RuntimeError("Checkpoint failed to load.")
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
    result = {
        "checkpoint": str(args.checkpoint),
        "csv": str(csv_path),
        "view": view,
        "rows": len(dataset),
        "loss": total_loss / len(dataset),
        "mean_auc": float(sum(valid_aucs) / len(valid_aucs)) if valid_aucs else None,
        "label_auc": aucs,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
