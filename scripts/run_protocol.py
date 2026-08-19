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
import yaml
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import DEFAULT_LABELS
from scripts.compare_models import compare_two_prediction_files
from scripts.ensemble_predictions import ensemble_prediction_files
from scripts.evaluate import compute_expected_calibration_error, patient_level_cluster_bootstrap


def compute_file_sha256(filepath: Path) -> str:
    if not filepath.is_file():
        raise FileNotFoundError(f"Cannot compute SHA-256: file does not exist at '{filepath}'")
    h = hashlib.sha256()
    with filepath.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def get_git_commit_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT), text=True).strip()
    except Exception:
        return "UNKNOWN"


def run_development_stage(config_path: Path, manifest_path: Path, limit: int | None = None) -> None:
    print(f"\n=======================================================")
    print(f"RUNNING PROTOCOL: DEVELOPMENT STAGE (Training & Calibration)")
    print(f"Config: {config_path} | Manifest: {manifest_path}")
    print(f"=======================================================\n")

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    architectures = config.get("model", {}).get("architectures", ["convnext_small", "densenet121"])
    seeds = config.get("seeds", [42, 43, 44, 45, 46])

    python_exe = sys.executable

    for arch in architectures:
        for seed in seeds:
            out_dir = PROJECT_ROOT / "outputs" / "runs" / arch / f"seed_{seed}"
            print(f"\n>>> [TRAIN] Architecture: {arch} | Seed: {seed} -> {out_dir}")
            train_cmd = [
                python_exe,
                str(PROJECT_ROOT / "scripts" / "train.py"),
                "--manifest", str(manifest_path),
                "--config", str(config_path),
                "--arch", arch,
                "--seed", str(seed),
                "--output-dir", str(out_dir),
            ]
            if limit:
                train_cmd.extend(["--limit", str(limit)])
            subprocess.check_call(train_cmd)

            # Run Threshold Calibration on calibration set
            ckpt_path = out_dir / "best.pt"
            calib_out = PROJECT_ROOT / "outputs" / "calibration" / f"{arch}_seed{seed}.json"
            print(f">>> [CALIBRATE] Architecture: {arch} | Seed: {seed} -> {calib_out}")
            calib_cmd = [
                python_exe,
                str(PROJECT_ROOT / "scripts" / "calibrate.py"),
                "--checkpoint", str(ckpt_path),
                "--split-manifest", str(manifest_path),
                "--output", str(calib_out),
                "--seed", str(seed),
            ]
            if limit:
                calib_cmd.extend(["--limit", str(limit)])
            subprocess.check_call(calib_cmd)

    print("\n[SUCCESS] Protocol development stage completed (10 models trained and calibrated).")


def run_evaluation_stage(
    config_path: Path,
    locked_manifest_path: Path,
    frozen_ledger_path: Path,
    output_dir: Path,
) -> Path:
    print(f"\n=======================================================")
    print(f"RUNNING PROTOCOL: FINAL LOCKED EVALUATION STAGE")
    print(f"Frozen Ledger: {frozen_ledger_path}")
    print(f"Locked Test Manifest: {locked_manifest_path}")
    print(f"=======================================================\n")

    if not frozen_ledger_path.is_file():
        raise RuntimeError(
            f"FAIL-CLOSED EVALUATION ERROR: Frozen ledger '{frozen_ledger_path}' does not exist!\n"
            "You must run `scripts/freeze_experiment.py` first before running final evaluation."
        )

    ledger_data = json.loads(frozen_ledger_path.read_text(encoding="utf-8"))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    architectures = config.get("model", {}).get("architectures", ["convnext_small", "densenet121"])
    seeds = config.get("seeds", [42, 43, 44, 45, 46])
    labels = config.get("model", {}).get("labels", DEFAULT_LABELS)

    python_exe = sys.executable
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_csvs_per_arch: dict[str, list[Path]] = {arch: [] for arch in architectures}
    per_seed_reports: dict[str, dict[str, dict]] = {arch: {} for arch in architectures}

    # 1. Evaluate all 10 individual checkpoints on Locked Test
    for arch in architectures:
        for seed in seeds:
            seed_str = str(seed)
            model_entry = ledger_data.get("models", {}).get(arch, {}).get(seed_str)
            if not model_entry:
                raise RuntimeError(f"FAIL-CLOSED ERROR: Missing frozen ledger entry for {arch} seed {seed}!")

            ckpt_path = PROJECT_ROOT / model_entry["checkpoint_path"]
            th_path = PROJECT_ROOT / model_entry["threshold_path"]

            print(f">>> [LOCKED EVAL] Architecture: {arch} | Seed: {seed}")
            eval_cmd = [
                python_exe,
                str(PROJECT_ROOT / "scripts" / "evaluate.py"),
                "--locked-test-manifest", str(locked_manifest_path),
                "--checkpoint", str(ckpt_path),
                "--threshold-artifact", str(th_path),
                "--frozen-ledger", str(frozen_ledger_path),
                "--output-dir", str(output_dir / "seeds"),
                "--arch", arch,
                "--seed", str(seed),
            ]
            subprocess.check_call(eval_cmd)

            pred_csv = output_dir / "seeds" / f"{arch}_seed{seed}_locked_predictions.csv"
            report_json = output_dir / "seeds" / f"{arch}_seed{seed}_report.json"

            if not pred_csv.is_file() or not report_json.is_file():
                raise FileNotFoundError(f"Evaluation output not produced: {pred_csv}")

            eval_csvs_per_arch[arch].append(pred_csv)
            per_seed_reports[arch][seed_str] = json.loads(report_json.read_text(encoding="utf-8"))

    # 2. Ensemble 5 seeds per Architecture
    ensemble_csvs: dict[str, Path] = {}
    for arch in architectures:
        ens_csv_path = output_dir / f"predictions_{arch}_ensemble.csv"
        print(f"\n>>> [ENSEMBLE] Ensembling 5 seeds for {arch} -> {ens_csv_path}")
        ensemble_prediction_files(eval_csvs_per_arch[arch], ens_csv_path, labels=labels)
        ensemble_csvs[arch] = ens_csv_path

    # 3. Evaluate Ensemble Metrics
    ensemble_results: dict[str, dict] = {}
    for arch, ens_csv in ensemble_csvs.items():
        ens_df = pd.read_csv(ens_csv)
        targets_arr = np.column_stack([ens_df[f"{l}_target"].values for l in labels])
        probs_arr = np.column_stack([ens_df[f"{l}_prob"].values for l in labels])
        masks_arr = np.column_stack([ens_df[f"{l}_mask"].values if f"{l}_mask" in ens_df.columns else np.ones(len(ens_df)) for l in labels])
        patient_ids = list(ens_df["patient_id"])

        # Use average threshold across the 5 seeds for ensemble binarization
        avg_thresholds = {}
        for l in labels:
            ths = [per_seed_reports[arch][str(s)]["metrics_per_label"][l]["threshold"] for s in seeds]
            avg_thresholds[l] = float(np.mean(ths))

        ci_results = patient_level_cluster_bootstrap(patient_ids, targets_arr, probs_arr, masks_arr, avg_thresholds, labels, n_boot=2000)

        label_metrics = {}
        valid_aucs = []
        valid_auprcs = []
        valid_f1s = []

        for idx, label in enumerate(labels):
            v_idx = np.where(masks_arr[:, idx] > 0.5)[0]
            y_t = np.array([1 if targets_arr[i, idx] >= 0.5 else 0 for i in v_idx])
            y_p = probs_arr[v_idx, idx]
            th = avg_thresholds[label]
            y_pred = (y_p >= th).astype(int)

            if len(np.unique(y_t)) > 1:
                auc_pt = float(roc_auc_score(y_t, y_p))
                auprc_pt = float(average_precision_score(y_t, y_p))
                valid_aucs.append(auc_pt)
                valid_auprcs.append(auprc_pt)
            else:
                auc_pt, auprc_pt = None, None

            tn, fp, fn, tp = confusion_matrix(y_t, y_pred, labels=[0, 1]).ravel()
            sens = float(tp / max(1, tp + fn))
            spec = float(tn / max(1, tn + fp))
            f1 = float(f1_score(y_t, y_pred, zero_division=0))
            brier = float(brier_score_loss(y_t, y_p))
            ece = compute_expected_calibration_error(y_t, y_p, n_bins=15)
            valid_f1s.append(f1)

            auc_ci = ci_results.get(label, {}).get("auroc", (0.0, 0.0))
            label_metrics[label] = {
                "auroc": auc_pt,
                "ci_95": f"{auc_pt:.4f} ({auc_ci[0]:.4f}–{auc_ci[1]:.4f})" if auc_pt is not None else "N/A",
                "ci_lower": auc_ci[0],
                "ci_upper": auc_ci[1],
                "auprc": auprc_pt,
                "sensitivity": sens,
                "specificity": spec,
                "f1_score": f1,
                "brier_score": brier,
                "ece_15bins": ece,
                "threshold": th,
            }

        macro_auc = float(np.mean(valid_aucs)) if valid_aucs else 0.0
        macro_ci = ci_results.get("macro", {}).get("auroc", (0.0, 0.0))

        ensemble_results[arch] = {
            "macro_auroc": macro_auc,
            "macro_auroc_95_ci": f"{macro_auc:.4f} ({macro_ci[0]:.4f}–{macro_ci[1]:.4f})",
            "macro_auprc": float(np.mean(valid_auprcs)) if valid_auprcs else 0.0,
            "macro_f1": float(np.mean(valid_f1s)) if valid_f1s else 0.0,
            "metrics_per_label": label_metrics,
            "predictions_csv": str(ens_csv.relative_to(PROJECT_ROOT) if ens_csv.is_relative_to(PROJECT_ROOT) else ens_csv),
            "predictions_sha256": compute_file_sha256(ens_csv),
        }

    # 4. Paired Statistical Comparison (ConvNeXt vs DenseNet)
    print("\n>>> [COMPARISON] Running Paired DeLong and Patient-Level Bootstrap Comparison...")
    comp_json_path = output_dir / "model_comparison.json"
    comparison = compare_two_prediction_files(
        ensemble_csvs["convnext_small"],
        ensemble_csvs["densenet121"],
        labels=labels,
        n_boot=2000,
    )
    comp_json_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    # 5. Build Final Benchmark Artifact (Single Source of Truth)
    artifact_path = output_dir / "benchmark_artifact.json"
    final_artifact = {
        "schema_version": "1.0",
        "protocol_version": ledger_data.get("protocol_version", "0.1"),
        "artifact_type": "official_scientific_benchmark",
        "git_commit": get_git_commit_sha(),
        "config_sha256": ledger_data.get("config_sha256"),
        "development_manifest_sha256": ledger_data.get("development_manifest_sha256"),
        "locked_test_manifest_sha256": ledger_data.get("locked_test_manifest_sha256"),
        "frozen_ledger_sha256": compute_file_sha256(frozen_ledger_path),
        "architectures": ensemble_results,
        "per_seed_results": per_seed_reports,
        "model_comparison": comparison,
        "multiplicity_correction": "Holm-Bonferroni",
        "bootstrap": {
            "unit": "patient",
            "replicates": 2000,
            "seed": 42,
        },
        "environment": {
            "python": sys.version.split()[0],
            "pytorch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    artifact_path.write_text(json.dumps(final_artifact, indent=2), encoding="utf-8")
    print(f"\n[SUCCESS] Single-Source-of-Truth Artifact created -> {artifact_path}")

    # 6. Synchronize Reports & LaTeX Tables
    print("\n>>> [SYNC] Synchronizing README and LaTeX tables from Final Benchmark Artifact...")
    sync_cmd = [
        python_exe,
        str(PROJECT_ROOT / "scripts" / "sync_reports.py"),
        "--artifact", str(artifact_path),
    ]
    subprocess.check_call(sync_cmd)

    return artifact_path


def main():
    parser = argparse.ArgumentParser(description="Multi-Model Multi-Seed Scientific Protocol Runner.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "protocol_v0_1.yaml")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "outputs" / "splits" / "protocol_v0_1" / "manifest.json")
    parser.add_argument("--locked-test-manifest", type=Path, default=PROJECT_ROOT / "outputs" / "splits" / "protocol_v0_1" / "locked_test_manifest.json")
    parser.add_argument("--frozen-ledger", type=Path, default=PROJECT_ROOT / "outputs" / "frozen" / "protocol_v0_1.json")
    parser.add_argument("--stage", choices=["development", "evaluation"], required=True)
    parser.add_argument("--limit", type=int, help="Optional subset limit for development testing only")
    args = parser.parse_args()

    if args.stage == "development":
        run_development_stage(args.config, args.manifest, limit=args.limit)
    elif args.stage == "evaluation":
        if args.limit:
            raise RuntimeError("FAIL-CLOSED ERROR: --limit is strictly forbidden during final evaluation!")
        out_final = PROJECT_ROOT / "outputs" / "final" / "protocol_v0_1"
        run_evaluation_stage(args.config, args.locked_test_manifest, args.frozen_ledger, out_final)


if __name__ == "__main__":
    main()
