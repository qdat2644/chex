from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_recall_curve
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.dataset import CheXpertDataset
from app.model import CheXpertPredictor


def compute_file_sha256(filepath: Path) -> str:
    if not filepath.exists():
        return "NOT_FOUND"
    h = hashlib.sha256()
    with filepath.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate optimal per-label decision thresholds on the frozen Calibration partition only.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Trained model checkpoint .pt")
    parser.add_argument("--data-root", type=Path, required=True, help="Data root directory")
    parser.add_argument("--calibration-csv", type=Path, help="Explicit path to calibration.csv")
    parser.add_argument("--split-manifest", type=Path, help="Path to splits manifest.json (auto-resolves calibration.csv)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit", type=int, help="Optional limit for quick testing")
    parser.add_argument(
        "--uncertain-policy",
        choices=["u_ones_zeros", "smooth", "zero", "one", "ignore"],
        default="u_ones_zeros",
        help="Uncertainty policy for calibration",
    )
    parser.add_argument("--view", choices=["frontal", "lateral", "all"], default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "evaluation" / "frozen_thresholds.json",
        help="Target output path for frozen thresholds artifact",
    )
    return parser.parse_args()


def calculate_optimal_thresholds(
    targets: list[list[float]],
    probs: list[list[float]],
    labels: list[str],
    masks: list[list[float]] | None = None,
) -> tuple[dict[str, float], dict[str, object]]:
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
        best_idx = int(np.argmax(f1_scores))
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
            "calib_precision": precision,
            "calib_recall": recall,
            "calib_f1": f1,
            "valid_count": len(valid_idx),
            "positive_count": int(np.sum(y_true == 1)),
            "negative_count": int(np.sum(y_true == 0)),
        }

    return thresholds_dict, metrics_dict


@torch.inference_mode()
def main() -> None:
    args = parse_args()

    # Reject official valid.csv / test dataset to prevent contamination
    calib_csv = args.calibration_csv
    if not calib_csv and args.split_manifest:
        if args.split_manifest.exists():
            manifest_data = json.loads(args.split_manifest.read_text(encoding="utf-8"))
            calib_filename = manifest_data.get("splits", {}).get("calibration", {}).get("file", "calibration.csv")
            calib_csv = args.split_manifest.parent / calib_filename

    if not calib_csv or not calib_csv.exists():
        # Look for default splits folder
        default_calib = PROJECT_ROOT / "outputs" / "splits" / "calibration.csv"
        if default_calib.exists():
            calib_csv = default_calib
        else:
            raise FileNotFoundError("Calibration CSV not found. Provide --calibration-csv or --split-manifest.")

    if "valid.csv" in calib_csv.name.lower() and "internal" not in calib_csv.name.lower():
        raise RuntimeError(
            "LEAKAGE PREVENTION ERROR: Do NOT use official 'valid.csv' for calibration! "
            "Calibration must be performed strictly on the internal 'calibration.csv' split."
        )

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    predictor = CheXpertPredictor(args.checkpoint)
    if predictor.model is None:
        raise RuntimeError(f"Failed to load checkpoint: {args.checkpoint}")

    metadata = getattr(predictor, "metadata", {})
    view = args.view or metadata.get("view", "frontal")

    dataset = CheXpertDataset(
        calib_csv,
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

    print(f"Running threshold calibration on {len(dataset)} studies in {calib_csv.name}...")
    for item in loader:
        images = item[0].to(predictor.device)
        targets = item[1].to(predictor.device)
        masks = item[2].to(predictor.device) if len(item) > 2 else torch.ones_like(targets)

        logits = predictor.model(images)
        probs = torch.sigmoid(logits)

        targets_all.extend(targets.detach().cpu().tolist())
        probs_all.extend(probs.detach().cpu().tolist())
        masks_all.extend(masks.detach().cpu().tolist())

    thresholds_dict, metrics_dict = calculate_optimal_thresholds(
        targets_all, probs_all, predictor.labels, masks=masks_all
    )

    ckpt_hash = compute_file_sha256(args.checkpoint)
    calib_hash = compute_file_sha256(calib_csv)
    manifest_hash = compute_file_sha256(args.split_manifest) if args.split_manifest else "N/A"

    frozen_artifact = {
        "artifact_type": "frozen_thresholds",
        "version": "1.0.0",
        "calibration_method": "optimal_f1_precision_recall_curve",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_architecture": predictor.architecture,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": ckpt_hash,
        "calibration_split_csv": str(calib_csv),
        "calibration_split_sha256": calib_hash,
        "split_manifest_sha256": manifest_hash,
        "uncertain_policy": args.uncertain_policy,
        "seed": args.seed,
        "num_calibration_studies": len(dataset),
        "thresholds": thresholds_dict,
        "calibration_metrics": metrics_dict,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(frozen_artifact, indent=2), encoding="utf-8")

    print("\n=== CALIBRATED FROZEN THRESHOLDS ===")
    for lbl, th in thresholds_dict.items():
        print(f"  {lbl:20s}: {th:.4f} (Calib F1: {metrics_dict[lbl].get('calib_f1', 0.0):.4f})")
    print(f"\nFrozen threshold artifact successfully written to: {args.output}")


if __name__ == "__main__":
    main()
