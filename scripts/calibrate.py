from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_curve
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import DEFAULT_LABELS
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


def calculate_optimal_thresholds(
    targets: list[list[float]] | np.ndarray,
    probs: list[list[float]] | np.ndarray,
    labels: list[str],
    masks: list[list[float]] | np.ndarray | None = None,
    min_bound: float = 0.0,
    max_bound: float = 1.0,
) -> tuple[dict[str, float | None], dict[str, dict[str, object]]]:
    targets_arr = np.array(targets)
    probs_arr = np.array(probs)
    masks_arr = np.array(masks) if masks is not None else np.ones_like(targets_arr)

    thresholds: dict[str, float | None] = {}
    metrics_summary: dict[str, dict[str, object]] = {}

    for idx, label in enumerate(labels):
        valid_idx = np.where(masks_arr[:, idx] > 0.5)[0]
        if len(valid_idx) == 0:
            thresholds[label] = None
            metrics_summary[label] = {"error": "no_valid_samples", "pos_count": 0, "neg_count": 0}
            continue

        y_true = np.array([1 if targets_arr[i, idx] >= 0.5 else 0 for i in valid_idx])
        y_prob = probs_arr[valid_idx, idx]

        pos_count = int(np.sum(y_true == 1))
        neg_count = int(np.sum(y_true == 0))

        if pos_count == 0 or neg_count == 0:
            # If dataset lacks both classes, threshold MUST be null, not fake 0.5
            thresholds[label] = None
            metrics_summary[label] = {
                "error": "insufficient_classes",
                "pos_count": pos_count,
                "neg_count": neg_count,
            }
            continue

        precisions, recalls, candidate_thresholds = precision_recall_curve(y_true, y_prob)

        # Avoid divide-by-zero
        f1_scores = np.where(
            (precisions + recalls) > 0,
            2.0 * (precisions * recalls) / (precisions + recalls + 1e-12),
            0.0,
        )

        if len(candidate_thresholds) == 0:
            thresholds[label] = None
            continue

        # precision_recall_curve returns candidate_thresholds of length len(precisions)-1
        f1_scores_candidates = f1_scores[:-1]
        best_idx = int(np.argmax(f1_scores_candidates))
        best_th = float(candidate_thresholds[best_idx])

        # Apply configurable bounds without silent distortion
        bounded_th = float(np.clip(best_th, min_bound, max_bound))
        thresholds[label] = bounded_th

        best_f1 = float(f1_scores_candidates[best_idx])
        best_prec = float(precisions[best_idx])
        best_rec = float(recalls[best_idx])

        metrics_summary[label] = {
            "optimal_f1": best_f1,
            "precision_at_optimal": best_prec,
            "recall_at_optimal": best_rec,
            "positive_count": pos_count,
            "negative_count": neg_count,
            "threshold": bounded_th,
        }

    return thresholds, metrics_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate decision thresholds on the Calibration split.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Trained model checkpoint .pt")
    parser.add_argument("--split-manifest", type=Path, required=True, help="Path to outputs/splits/protocol_v0_1/manifest.json")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "archive", help="Data root directory")
    parser.add_argument("--output", type=Path, help="Target output path for frozen thresholds artifact")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, help="Optional limit for testing")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


@torch.inference_mode()
def main():
    args = parse_args()

    # 1. Verify Split Manifest
    if not args.split_manifest.exists():
        print(f"Error: Manifest not found at {args.split_manifest}", file=sys.stderr)
        sys.exit(1)

    manifest_data = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    manifest_sha256 = compute_file_sha256(args.split_manifest)

    splits = manifest_data.get("splits", {})
    if "locked_test" in splits or manifest_data.get("role") == "locked_test":
        raise RuntimeError("LEAKAGE INTEGRITY VIOLATION: Calibration cannot run on locked test set!")

    calib_meta = splits.get("calibration", {})
    if not calib_meta or calib_meta.get("role") != "calibration":
        raise RuntimeError("LEAKAGE INTEGRITY VIOLATION: Split role must be 'calibration'!")

    calib_csv = args.split_manifest.parent / calib_meta.get("csv", "calibration.csv")
    if not calib_csv.exists():
        raise FileNotFoundError(f"Calibration CSV not found: {calib_csv}")

    calib_csv_sha256 = compute_file_sha256(calib_csv)

    # 2. Verify Checkpoint Integrity & Linkage
    if not args.checkpoint.exists():
        print(f"Error: Checkpoint not found at {args.checkpoint}", file=sys.stderr)
        sys.exit(1)

    ckpt_sha256 = compute_file_sha256(args.checkpoint)
    loaded_ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    ckpt_meta = loaded_ckpt.get("metadata", {})

    # Check that checkpoint was trained on this split manifest
    if ckpt_meta.get("split_manifest_sha256") and ckpt_meta["split_manifest_sha256"] != manifest_sha256:
        print(f"WARNING: Checkpoint split_manifest_sha256 ({ckpt_meta['split_manifest_sha256']}) does not match manifest ({manifest_sha256})!", file=sys.stderr)

    labels = loaded_ckpt.get("labels") or manifest_data.get("labels") or DEFAULT_LABELS
    unc_policy = ckpt_meta.get("uncertainty_policy") or manifest_data.get("uncertainty_policy", "u_ones_zeros")
    preprocessing_sha256 = ckpt_meta.get("preprocessing_sha256", "UNKNOWN")

    # 3. Load Model
    predictor = CheXpertPredictor(args.checkpoint)
    if predictor.model is None:
        raise RuntimeError("Checkpoint failed to load into predictor.")

    # 4. Load Calibration Dataset
    dataset = CheXpertDataset(
        calib_csv,
        args.data_root,
        predictor.transform,
        labels=labels,
        uncertain_policy=unc_policy,
        view=manifest_data.get("view", "frontal"),
    )
    if args.limit:
        dataset = Subset(dataset, range(min(args.limit, len(dataset))))

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    targets_all: list[list[float]] = []
    probs_all: list[list[float]] = []
    masks_all: list[list[float]] = []

    print(f"Evaluating {len(dataset)} calibration studies from {calib_csv.name}...")
    for item in loader:
        images = item[0].to(predictor.device)
        targets = item[1].to(predictor.device)
        masks = item[2].to(predictor.device) if len(item) > 2 else torch.ones_like(targets)

        logits = predictor.model(images)
        probs = torch.sigmoid(logits)

        targets_all.extend(targets.detach().cpu().tolist())
        probs_all.extend(probs.detach().cpu().tolist())
        masks_all.extend(masks.detach().cpu().tolist())

    # 5. Compute Optimal Thresholds
    thresholds, metrics = calculate_optimal_thresholds(targets_all, probs_all, labels, masks=masks_all)

    # 6. Build Calibration Artifact
    out_path = args.output or PROJECT_ROOT / "outputs" / "calibration" / f"{predictor.architecture}_seed{args.seed}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    artifact = {
        "schema_version": "1.0",
        "role": "calibration_artifact",
        "checkpoint_sha256": ckpt_sha256,
        "split_manifest_sha256": manifest_sha256,
        "calibration_csv_sha256": calib_csv_sha256,
        "labels": labels,
        "uncertainty_policy": unc_policy,
        "preprocessing_sha256": preprocessing_sha256,
        "threshold_selection": "max_f1",
        "thresholds": thresholds,
        "calibration_metrics": metrics,
        "sample_count": len(dataset),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    out_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(f"\nCalibration artifact successfully saved to: {out_path}")
    for lbl, th in thresholds.items():
        m = metrics.get(lbl, {})
        th_display = f"{th:.4f}" if th is not None else "NULL (Insufficient Classes)"
        print(f"  {lbl:20s}: Threshold={th_display} | Calib F1={m.get('optimal_f1', 0.0):.4f}")


if __name__ == "__main__":
    main()
