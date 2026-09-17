from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_curve
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import DEFAULT_LABELS
from app.dataset import CheXpertDataset
from app.experiment_integrity import (
    canonical_json_sha256,
    compute_file_sha256,
    sanitize_for_json,
)
from app.model import CheXpertPredictor


def calculate_optimal_thresholds(
    targets: list[list[float]] | np.ndarray,
    probs: list[list[float]] | np.ndarray,
    labels: list[str],
    masks: list[list[float]] | np.ndarray | None = None,
    min_bound: float = 0.0,
    max_bound: float = 1.0,
    strict_full_mode: bool = False,
) -> tuple[dict[str, float | None], dict[str, dict[str, Any]]]:
    targets_arr = np.array(targets)
    probs_arr = np.array(probs)
    masks_arr = np.array(masks) if masks is not None else np.ones_like(targets_arr)

    thresholds: dict[str, float | None] = {}
    metrics_summary: dict[str, dict[str, Any]] = {}

    for idx, label in enumerate(labels):
        valid_idx = np.where(masks_arr[:, idx] > 0.5)[0]
        if len(valid_idx) == 0:
            if strict_full_mode:
                raise RuntimeError(f"CALIBRATION INTEGRITY ERROR: Label '{label}' has 0 valid samples in full mode!")
            thresholds[label] = None
            metrics_summary[label] = {"error": "no_valid_samples", "pos_count": 0, "neg_count": 0}
            continue

        y_true = np.array([1 if targets_arr[i, idx] >= 0.5 else 0 for i in valid_idx])
        y_prob = probs_arr[valid_idx, idx]

        pos_count = int(np.sum(y_true == 1))
        neg_count = int(np.sum(y_true == 0))

        if pos_count == 0 or neg_count == 0:
            if strict_full_mode:
                raise RuntimeError(
                    f"CALIBRATION INTEGRITY ERROR: Label '{label}' lacks both positive and negative classes "
                    f"in full mode (pos={pos_count}, neg={neg_count})!"
                )
            thresholds[label] = None
            metrics_summary[label] = {
                "error": "insufficient_classes",
                "pos_count": pos_count,
                "neg_count": neg_count,
            }
            continue

        precisions, recalls, candidate_thresholds = precision_recall_curve(y_true, y_prob)

        f1_scores = np.where(
            (precisions + recalls) > 0,
            2.0 * (precisions * recalls) / (precisions + recalls + 1e-12),
            0.0,
        )

        if len(candidate_thresholds) == 0:
            thresholds[label] = None
            continue

        f1_scores_candidates = f1_scores[:-1]
        best_idx = int(np.argmax(f1_scores_candidates))
        best_th = float(candidate_thresholds[best_idx])
        bounded_th = float(np.clip(best_th, min_bound, max_bound))
        thresholds[label] = bounded_th

        metrics_summary[label] = {
            "optimal_f1": float(f1_scores_candidates[best_idx]),
            "precision_at_optimal": float(precisions[best_idx]),
            "recall_at_optimal": float(recalls[best_idx]),
            "positive_count": pos_count,
            "negative_count": neg_count,
            "threshold": bounded_th,
        }

    return thresholds, metrics_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate decision thresholds on the Calibration split.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Trained model checkpoint .pt")
    parser.add_argument("--split-manifest", type=Path, required=True, help="Path to outputs/splits/protocol_v0_1/manifest.json")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "protocol_v0_1.yaml", help="Path to protocol YAML config")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "archive", help="Data root directory")
    parser.add_argument("--output", type=Path, help="Target output path for frozen thresholds artifact")
    parser.add_argument("--arch", type=str, default=None, help="Model architecture")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, help="Optional limit for testing")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-mode", choices=["smoke", "full"], default=None, help="Override or specify run mode")
    return parser.parse_args()


@torch.inference_mode()
def main():
    args = parse_args()

    # 1. Verify Split Manifest
    if not args.split_manifest.exists():
        raise FileNotFoundError(f"Manifest not found at {args.split_manifest}")

    manifest_data = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    manifest_sha256 = compute_file_sha256(args.split_manifest)

    splits = manifest_data.get("splits", {})
    if "locked_test" in splits or manifest_data.get("role") == "locked_test" or manifest_data.get("locked"):
        raise RuntimeError("LEAKAGE INTEGRITY VIOLATION: Calibration cannot run on locked test set!")

    calib_meta = splits.get("calibration", {})
    if not calib_meta or calib_meta.get("role") != "calibration":
        raise RuntimeError("LEAKAGE INTEGRITY VIOLATION: Split role must be 'calibration'!")

    calib_csv = args.split_manifest.parent / calib_meta.get("csv", "calibration.csv")
    if not calib_csv.exists():
        raise FileNotFoundError(f"Calibration CSV not found: {calib_csv}")

    calib_csv_sha256 = compute_file_sha256(calib_csv)

    # 2. Verify Checkpoint Integrity & Extract Metadata
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found at {args.checkpoint}")

    ckpt_sha256 = compute_file_sha256(args.checkpoint)
    loaded_ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(loaded_ckpt, dict):
        raise RuntimeError(f"Invalid checkpoint dictionary format at {args.checkpoint}")

    ckpt_meta = loaded_ckpt.get("metadata", {})

    # F2: Required fields in checkpoint
    ckpt_arch = loaded_ckpt.get("architecture") or ckpt_meta.get("architecture")
    ckpt_seed = loaded_ckpt.get("seed") if "seed" in loaded_ckpt and loaded_ckpt["seed"] is not None else ckpt_meta.get("seed")
    ckpt_labels = loaded_ckpt.get("labels") or ckpt_meta.get("labels")
    ckpt_run_mode = loaded_ckpt.get("run_mode") or ckpt_meta.get("run_mode")
    ckpt_resolved_cfg = loaded_ckpt.get("resolved_config_sha256") or ckpt_meta.get("resolved_config_sha256")
    ckpt_manifest = loaded_ckpt.get("manifest_sha256") or ckpt_meta.get("split_manifest_sha256")
    ckpt_git_commit = loaded_ckpt.get("git_commit") or ckpt_meta.get("git_commit")

    # Fail-closed check: No required field may be missing
    required_fields = {
        "architecture": ckpt_arch,
        "seed": ckpt_seed,
        "labels": ckpt_labels,
        "run_mode": ckpt_run_mode,
        "resolved_config_sha256": ckpt_resolved_cfg,
        "manifest_sha256": ckpt_manifest,
        "git_commit": ckpt_git_commit,
    }
    missing = [k for k, v in required_fields.items() if v is None]
    if missing:
        raise RuntimeError(f"CALIBRATION INTEGRITY ERROR: Checkpoint is missing required metadata fields: {missing}")

    # Determine effective run_mode
    run_mode = args.run_mode or ckpt_run_mode
    strict_full_mode = (run_mode == "full")

    # F3: Absolute Validations
    if args.arch and str(ckpt_arch) != str(args.arch):
        raise RuntimeError(
            f"CALIBRATION INTEGRITY ERROR: Checkpoint architecture ({ckpt_arch}) does not match --arch ({args.arch})!"
        )

    if args.seed is not None and int(ckpt_seed) != int(args.seed):
        raise RuntimeError(
            f"CALIBRATION INTEGRITY ERROR: Checkpoint seed ({ckpt_seed}) does not match --seed ({args.seed})!"
        )

    manifest_labels = manifest_data.get("labels", DEFAULT_LABELS)
    if list(ckpt_labels) != list(manifest_labels):
        raise RuntimeError(
            f"CALIBRATION INTEGRITY ERROR: Checkpoint labels ({ckpt_labels}) do not match manifest labels ({manifest_labels})!"
        )

    if str(ckpt_manifest) != str(manifest_sha256):
        raise RuntimeError(
            f"CALIBRATION INTEGRITY ERROR: Checkpoint manifest hash ({ckpt_manifest}) "
            f"does not match current split manifest hash ({manifest_sha256})!"
        )

    # Check against resolved_config.json if located alongside checkpoint
    resolved_cfg_file = args.checkpoint.parent / "resolved_config.json"
    if resolved_cfg_file.is_file():
        actual_cfg_data = json.loads(resolved_cfg_file.read_text(encoding="utf-8"))
        actual_cfg_hash = canonical_json_sha256(actual_cfg_data)
        if actual_cfg_hash != ckpt_resolved_cfg:
            raise RuntimeError(
                f"CALIBRATION INTEGRITY ERROR: Checkpoint resolved_config_sha256 ({ckpt_resolved_cfg}) "
                f"does not match resolved_config.json ({actual_cfg_hash})!"
            )

    labels = list(ckpt_labels)
    unc_policy = ckpt_meta.get("uncertainty_policy") or manifest_data.get("uncertainty_policy", "u_ones_zeros")
    protocol_version = manifest_data.get("protocol_version", "0.1")

    # 3. Load Predictor Model
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

    # Check zero study ID leakage between calibration and train / internal_validation
    underlying_calib_df = dataset.dataset.frame if isinstance(dataset, Subset) else dataset.frame
    calib_study_ids = set(underlying_calib_df["study_id"].astype(str)) if "study_id" in underlying_calib_df.columns else set()
    train_csv = args.split_manifest.parent / splits.get("train", {}).get("csv", "train.csv")
    val_csv = args.split_manifest.parent / splits.get("internal_validation", {}).get("csv", "val.csv")

    if train_csv.is_file():
        train_df = pd.read_csv(train_csv)
        if "study_id" in train_df.columns:
            train_study_ids = set(train_df["study_id"].astype(str))
            if calib_study_ids & train_study_ids:
                raise RuntimeError("LEAKAGE INTEGRITY VIOLATION: Calibration study IDs overlap with Training split!")

    if val_csv.is_file():
        val_df = pd.read_csv(val_csv)
        if "study_id" in val_df.columns:
            val_study_ids = set(val_df["study_id"].astype(str))
            if calib_study_ids & val_study_ids:
                raise RuntimeError("LEAKAGE INTEGRITY VIOLATION: Calibration study IDs overlap with Internal Validation split!")

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

    # 5. Compute Optimal Decision Thresholds (F4)
    thresholds, metrics = calculate_optimal_thresholds(
        targets_all,
        probs_all,
        labels,
        masks=masks_all,
        strict_full_mode=strict_full_mode,
    )

    # Check for null thresholds in smoke mode vs full mode
    has_null = any(th is None for th in thresholds.values())
    if has_null:
        if strict_full_mode:
            raise RuntimeError("CALIBRATION INTEGRITY ERROR: One or more thresholds is null in full mode!")
        calib_status = "incomplete_smoke_calibration"
    else:
        calib_status = "smoke_calibration_complete" if run_mode == "smoke" else "completed"

    # Export Calibration Predictions CSV & Hash
    out_path = args.output or PROJECT_ROOT / "outputs" / "calibration" / f"{ckpt_arch}_seed{ckpt_seed}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    calib_pred_csv = out_path.parent / f"{ckpt_arch}_seed{ckpt_seed}_calibration_predictions.csv"
    pred_records = []
    for i in range(len(dataset)):
        row_item = underlying_calib_df.iloc[dataset.indices[i]] if isinstance(dataset, Subset) else underlying_calib_df.iloc[i]
        rec = {
            "study_id": str(row_item.get("study_id", f"study_{i+1}")),
            "patient_id": str(row_item.get("patient_id", f"patient_{i+1}")),
            "image_path": str(row_item.get("Path", row_item.get("image_path", ""))),
        }
        for idx, lbl in enumerate(labels):
            safe_lbl = lbl.replace(" ", "_")
            rec[f"{safe_lbl}_target"] = float(targets_all[i][idx])
            rec[f"{safe_lbl}_mask"] = float(masks_all[i][idx])
            rec[f"{safe_lbl}_prob"] = float(probs_all[i][idx])
        pred_records.append(rec)
    pd.DataFrame(pred_records).to_csv(calib_pred_csv, index=False)
    calib_pred_sha256 = compute_file_sha256(calib_pred_csv)

    # 6. Build Calibration Artifact Schema Version 2 (F5)
    artifact = {
        "schema_version": 2,
        "protocol_version": str(protocol_version),
        "run_mode": str(run_mode),
        "status": str(calib_status),
        "architecture": str(ckpt_arch),
        "seed": int(ckpt_seed),
        "labels": labels,
        "thresholds": thresholds,
        "checkpoint_sha256": ckpt_sha256,
        "resolved_config_sha256": str(ckpt_resolved_cfg),
        "manifest_sha256": manifest_sha256,
        "calibration_predictions_sha256": str(calib_pred_sha256),
        "git_commit": str(ckpt_git_commit),
        "threshold_selection": "max_f1",
        "calibration_metrics": metrics,
        "sample_count": len(dataset),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if run_mode == "smoke":
        artifact["note"] = "NON_FINAL_SMOKE_TEST"

    sanitized_artifact = sanitize_for_json(artifact)
    out_path.write_text(json.dumps(sanitized_artifact, indent=2), encoding="utf-8")
    print(f"\nCalibration artifact successfully saved to: {out_path} (status: {calib_status})")
    for lbl, th in thresholds.items():
        m = metrics.get(lbl, {})
        th_display = f"{th:.4f}" if th is not None else "NULL (Incomplete)"
        print(f"  {lbl:20s}: Threshold={th_display} | Calib F1={m.get('optimal_f1', 0.0):.4f}")


if __name__ == "__main__":
    main()
