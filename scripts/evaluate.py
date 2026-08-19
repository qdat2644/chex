from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
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


import shutil

def get_git_exe() -> str:
    exe = shutil.which("git")
    if exe:
        return exe
    local_tool = PROJECT_ROOT / "tools" / "git" / "cmd" / "git.exe"
    if local_tool.is_file():
        return str(local_tool)
    return "git"


def get_git_commit_sha() -> str:
    try:
        return subprocess.check_output([get_git_exe(), "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT), text=True).strip()
    except Exception:
        return "UNKNOWN"


def compute_file_sha256(filepath: Path) -> str:
    if not filepath.is_file():
        raise FileNotFoundError(f"Cannot compute SHA-256: file does not exist at '{filepath}'")
    h = hashlib.sha256()
    with filepath.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total_samples = len(y_true)
    if total_samples == 0:
        return 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        if i == n_bins - 1:
            in_bin = (y_prob >= bin_lower) & (y_prob <= bin_upper)
        else:
            in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)

        bin_count = np.sum(in_bin)
        if bin_count > 0:
            bin_acc = np.mean(y_true[in_bin])
            bin_conf = np.mean(y_prob[in_bin])
            ece += (bin_count / total_samples) * abs(bin_acc - bin_conf)

    return float(ece)


def patient_level_cluster_bootstrap(
    patient_ids: list[str],
    targets_arr: np.ndarray,
    probs_arr: np.ndarray,
    masks_arr: np.ndarray,
    thresholds_dict: dict[str, float],
    labels: list[str],
    n_boot: int = 2000,
    seed: int = 42,
) -> dict[str, dict[str, tuple[float, float]]]:
    rng = np.random.RandomState(seed)
    unique_patients = np.unique(patient_ids)
    n_patients = len(unique_patients)

    patient_to_indices: dict[str, list[int]] = {}
    for idx, pid in enumerate(patient_ids):
        if pid not in patient_to_indices:
            patient_to_indices[pid] = []
        patient_to_indices[pid].append(idx)

    auc_dist: dict[str, list[float]] = {lbl: [] for lbl in labels}
    f1_dist: dict[str, list[float]] = {lbl: [] for lbl in labels}
    sens_dist: dict[str, list[float]] = {lbl: [] for lbl in labels}
    spec_dist: dict[str, list[float]] = {lbl: [] for lbl in labels}
    macro_auc_dist: list[float] = []

    for _ in range(n_boot):
        sampled_pids = rng.choice(unique_patients, size=n_patients, replace=True)
        sampled_indices = []
        for pid in sampled_pids:
            sampled_indices.extend(patient_to_indices[pid])

        sample_t = targets_arr[sampled_indices]
        sample_p = probs_arr[sampled_indices]
        sample_m = masks_arr[sampled_indices]

        boot_aucs = []
        for idx, label in enumerate(labels):
            v_idx = np.where(sample_m[:, idx] > 0.5)[0]
            if len(v_idx) == 0:
                continue

            y_t = np.array([1 if sample_t[i, idx] >= 0.5 else 0 for i in v_idx])
            y_pr = sample_p[v_idx, idx]
            th = thresholds_dict.get(label, 0.5)
            y_pred = (y_pr >= th).astype(int)

            if len(np.unique(y_t)) > 1:
                try:
                    score = roc_auc_score(y_t, y_pr)
                    auc_dist[label].append(score)
                    boot_aucs.append(score)
                except Exception:
                    pass

            try:
                f1 = f1_score(y_t, y_pred, zero_division=0)
                f1_dist[label].append(f1)
                tn, fp, fn, tp = confusion_matrix(y_t, y_pred, labels=[0, 1]).ravel()
                sens_dist[label].append(tp / max(1, tp + fn))
                spec_dist[label].append(tn / max(1, tn + fp))
            except Exception:
                pass

        if boot_aucs:
            macro_auc_dist.append(float(np.mean(boot_aucs)))

    ci_results: dict[str, dict[str, tuple[float, float]]] = {}
    for lbl in labels:
        ci_results[lbl] = {}
        for metric_name, dist in [("auroc", auc_dist[lbl]), ("f1", f1_dist[lbl]), ("sensitivity", sens_dist[lbl]), ("specificity", spec_dist[lbl])]:
            if len(dist) >= 100:
                low, high = np.percentile(dist, [2.5, 97.5])
                ci_results[lbl][metric_name] = (float(low), float(high))
            else:
                ci_results[lbl][metric_name] = (0.0, 0.0)

    if len(macro_auc_dist) >= 100:
        low, high = np.percentile(macro_auc_dist, [2.5, 97.5])
        ci_results["macro"] = {"auroc": (float(low), float(high))}
    else:
        ci_results["macro"] = {"auroc": (0.0, 0.0)}

    return ci_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict Locked Evaluation on CheXpert Test Set.")
    parser.add_argument("--locked-test-manifest", type=Path, required=True, help="Path to locked_test_manifest.json")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Trained model checkpoint .pt")
    parser.add_argument("--threshold-artifact", type=Path, required=True, help="Frozen threshold artifact JSON")
    parser.add_argument("--frozen-ledger", type=Path, help="Path to outputs/frozen/protocol_v0_1.json ledger (MANDATORY for final evaluation)")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "archive", help="Data root directory")
    parser.add_argument("--output-dir", type=Path, help="Output directory for predictions and report")
    parser.add_argument("--arch", type=str, help="Expected architecture name (e.g. convnext_small, densenet121)")
    parser.add_argument("--seed", type=int, default=42, help="Expected seed")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, help="Subset limit (STRICTLY FORBIDDEN in final evaluation)")
    parser.add_argument("--n-boot", type=int, default=2000, help="Number of bootstrap resamples (>= 2000)")
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()

    # 1. Guard against --limit in final evaluation mode
    if args.frozen_ledger and args.limit:
        raise RuntimeError("FAIL-CLOSED ERROR: --limit is strictly forbidden during final frozen evaluation!")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 2. Verify Locked-Test Manifest
    if not args.locked_test_manifest.is_file():
        raise FileNotFoundError(f"Locked test manifest not found: '{args.locked_test_manifest}'")

    locked_manifest_data = json.loads(args.locked_test_manifest.read_text(encoding="utf-8"))
    locked_manifest_sha256 = compute_file_sha256(args.locked_test_manifest)

    if locked_manifest_data.get("role") != "locked_test" or not locked_manifest_data.get("locked"):
        raise RuntimeError("SECURITY VIOLATION: Manifest is not marked as role='locked_test' and locked=True!")

    locked_csv = args.locked_test_manifest.parent / locked_manifest_data.get("csv", "locked_test.csv")
    if not locked_csv.is_file():
        raise FileNotFoundError(f"Locked test CSV not found: '{locked_csv}'")

    # 3. Verify Threshold Artifact & Checkpoint Integrity (Zero Mismatch Tolerance)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: '{args.checkpoint}'")
    if not args.threshold_artifact.is_file():
        raise FileNotFoundError(f"Threshold artifact not found: '{args.threshold_artifact}'")

    ckpt_sha256 = compute_file_sha256(args.checkpoint)
    th_sha256 = compute_file_sha256(args.threshold_artifact)
    th_data = json.loads(args.threshold_artifact.read_text(encoding="utf-8"))

    # FAIL-CLOSED: Checkpoint SHA mismatch
    if th_data.get("checkpoint_sha256") != ckpt_sha256:
        raise RuntimeError(
            f"FAIL-CLOSED MISMATCH ERROR: Threshold artifact records checkpoint SHA '{th_data.get('checkpoint_sha256')}' "
            f"which does not match actual checkpoint '{ckpt_sha256}'!"
        )

    # 4. Verify Frozen Ledger Linkage (if provided)
    if args.frozen_ledger:
        if not args.frozen_ledger.is_file():
            raise FileNotFoundError(f"Frozen ledger not found: '{args.frozen_ledger}'")

        ledger_data = json.loads(args.frozen_ledger.read_text(encoding="utf-8"))
        if ledger_data.get("locked_test_manifest_sha256") != locked_manifest_sha256:
            raise RuntimeError(
                f"FAIL-CLOSED LEDGER ERROR: Ledger locked_test_manifest_sha256 "
                f"'{ledger_data.get('locked_test_manifest_sha256')}' does not match manifest '{locked_manifest_sha256}'!"
            )

        arch_key = args.arch or "convnext_small"
        seed_key = str(args.seed)
        arch_models = ledger_data.get("models", {}).get(arch_key, {})
        seed_entry = arch_models.get(seed_key)

        if not seed_entry:
            raise RuntimeError(f"FAIL-CLOSED LEDGER ERROR: No ledger entry found for {arch_key} seed {seed_key}!")

        if seed_entry.get("checkpoint_sha256") != ckpt_sha256:
            raise RuntimeError(
                f"FAIL-CLOSED LEDGER ERROR: Checkpoint SHA in ledger ({seed_entry.get('checkpoint_sha256')}) "
                f"does not match actual checkpoint SHA ({ckpt_sha256})!"
            )
        if seed_entry.get("threshold_sha256") != th_sha256:
            raise RuntimeError(
                f"FAIL-CLOSED LEDGER ERROR: Threshold SHA in ledger ({seed_entry.get('threshold_sha256')}) "
                f"does not match actual threshold SHA ({th_sha256})!"
            )

    thresholds_dict: dict[str, float] = {}
    for k, v in th_data.get("thresholds", {}).items():
        if v is None:
            raise RuntimeError(f"CANNOT EVALUATE: Label '{k}' has null threshold in calibration artifact!")
        thresholds_dict[k] = float(v)

    # 5. Load Model Predictor
    predictor = CheXpertPredictor(args.checkpoint, thresholds=thresholds_dict)
    if predictor.model is None:
        raise RuntimeError("Checkpoint failed to load into predictor.")

    # 6. Verify Labels & Uncertainty Policy Match
    if th_data.get("labels") and list(th_data["labels"]) != list(predictor.labels):
        raise RuntimeError(f"FAIL-CLOSED ERROR: Label order mismatch between threshold artifact and predictor model!")

    # 7. Load Locked Test Dataset
    dataset = CheXpertDataset(
        locked_csv,
        args.data_root,
        predictor.transform,
        labels=predictor.labels,
        uncertain_policy="u_ones_zeros",
        view=locked_manifest_data.get("view", "frontal"),
    )
    if args.limit and not args.frozen_ledger:
        dataset = Subset(dataset, range(min(args.limit, len(dataset))))

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    targets_all: list[list[float]] = []
    probs_all: list[list[float]] = []
    masks_all: list[list[float]] = []
    patient_ids_all: list[str] = []
    study_ids_all: list[str] = []

    underlying_df = dataset.dataset.frame if isinstance(dataset, Subset) else dataset.frame

    for i in range(len(dataset)):
        row_item = underlying_df.iloc[dataset.indices[i]] if isinstance(dataset, Subset) else underlying_df.iloc[i]
        patient_ids_all.append(str(row_item["patient_id"]))
        study_ids_all.append(str(row_item["study_id"]))

    print(f"Evaluating {len(dataset)} studies from {locked_csv.name} on Locked Test Protocol...")
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

    # 8. Patient-Level Cluster Bootstrap for 95% Confidence Intervals
    print(f"Running Patient-Level Cluster Bootstrap ({args.n_boot} resamples)...")
    ci_results = patient_level_cluster_bootstrap(
        patient_ids_all,
        targets_arr,
        probs_arr,
        masks_arr,
        thresholds_dict,
        predictor.labels,
        n_boot=args.n_boot,
        seed=args.seed,
    )

    # 9. Compute Comprehensive Per-Label and Macro Metrics
    label_metrics = {}
    valid_aucs = []
    valid_f1s = []
    valid_auprcs = []
    valid_briers = []

    for idx, label in enumerate(predictor.labels):
        valid_idx = np.where(masks_arr[:, idx] > 0.5)[0]
        if len(valid_idx) == 0:
            continue

        y_true = np.array([1 if targets_arr[i, idx] >= 0.5 else 0 for i in valid_idx])
        y_prob = probs_arr[valid_idx, idx]
        th = thresholds_dict[label]
        y_pred = (y_prob >= th).astype(int)

        if len(np.unique(y_true)) > 1:
            point_auc = float(roc_auc_score(y_true, y_prob))
            auprc = float(average_precision_score(y_true, y_prob))
            valid_aucs.append(point_auc)
            valid_auprcs.append(auprc)
        else:
            point_auc, auprc = None, None

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        sens = float(tp / max(1, tp + fn))
        spec = float(tn / max(1, tn + fp))
        ppv = float(tp / max(1, tp + fp))
        npv = float(tn / max(1, tn + fn))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        brier = float(brier_score_loss(y_true, y_prob))
        ece = compute_expected_calibration_error(y_true, y_prob, n_bins=15)

        valid_f1s.append(f1)
        valid_briers.append(brier)

        ci_l = ci_results.get(label, {})
        auc_ci = ci_l.get("auroc", (0.0, 0.0))

        label_metrics[label] = {
            "auroc": point_auc,
            "ci_95": f"{point_auc:.4f} ({auc_ci[0]:.4f}–{auc_ci[1]:.4f})" if point_auc is not None else "N/A",
            "ci_lower": auc_ci[0],
            "ci_upper": auc_ci[1],
            "auprc": auprc,
            "threshold": th,
            "sensitivity": sens,
            "specificity": spec,
            "ppv": ppv,
            "npv": npv,
            "f1_score": f1,
            "brier_score": brier,
            "ece_15bins": ece,
            "confusion_matrix": {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)},
            "sample_count": len(valid_idx),
            "positive_count": int(tp + fn),
            "negative_count": int(tn + fp),
        }

    macro_auc = float(np.mean(valid_aucs)) if valid_aucs else 0.0
    macro_auc_ci = ci_results.get("macro", {}).get("auroc", (0.0, 0.0))
    macro_f1 = float(np.mean(valid_f1s)) if valid_f1s else 0.0
    macro_auprc = float(np.mean(valid_auprcs)) if valid_auprcs else 0.0
    macro_brier = float(np.mean(valid_briers)) if valid_briers else 0.0

    out_dir = args.output_dir or PROJECT_ROOT / "outputs" / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)

    arch_name = args.arch or predictor.architecture.lower().replace("-", "_")

    # 10. Export Anonymized Stable Prediction CSV
    pred_csv_path = out_dir / f"{arch_name}_seed{args.seed}_locked_predictions.csv"
    pred_records = []
    for i in range(len(dataset)):
        rec = {
            "study_id": study_ids_all[i],
            "patient_id": patient_ids_all[i],
        }
        for idx, label in enumerate(predictor.labels):
            rec[f"{label}_prob"] = float(probs_arr[i, idx])
            rec[f"{label}_pred"] = int(probs_arr[i, idx] >= thresholds_dict[label])
            rec[f"{label}_target"] = float(targets_arr[i, idx])
            rec[f"{label}_mask"] = float(masks_arr[i, idx])
        pred_records.append(rec)

    pd.DataFrame(pred_records).to_csv(pred_csv_path, index=False)
    print(f"Exported prediction CSV: {pred_csv_path}")

    # 11. Export Evaluation Report
    report = {
        "schema_version": "1.0",
        "protocol_version": "0.1",
        "evaluation_mode": "locked_test",
        "git_commit": get_git_commit_sha(),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": ckpt_sha256,
        "locked_test_manifest_sha256": locked_manifest_sha256,
        "threshold_artifact_sha256": th_sha256,
        "model_architecture": arch_name,
        "seed": args.seed,
        "num_studies": len(dataset),
        "num_patients": len(set(patient_ids_all)),
        "macro_metrics": {
            "mean_auroc": macro_auc,
            "mean_auroc_95_ci": f"{macro_auc:.4f} ({macro_auc_ci[0]:.4f}–{macro_auc_ci[1]:.4f})",
            "mean_f1": macro_f1,
            "mean_auprc": macro_auprc,
            "mean_brier": macro_brier,
        },
        "metrics_per_label": label_metrics,
        "predictions_csv": str(pred_csv_path),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    report_path = out_dir / f"{arch_name}_seed{args.seed}_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved evaluation report: {report_path}")

    print("\n=== LOCKED TEST BENCHMARK RESULTS ===")
    print(f"Architecture: {arch_name} (Seed {args.seed}) | Macro Mean AUROC: {report['macro_metrics']['mean_auroc_95_ci']}")
    for lbl, m in label_metrics.items():
        print(f"  {lbl:20s}: AUROC={m['ci_95']} | Sens={m['sensitivity']:.3f} | Spec={m['specificity']:.3f} | F1={m['f1_score']:.3f} | ECE={m['ece_15bins']:.3f}")


if __name__ == "__main__":
    main()
