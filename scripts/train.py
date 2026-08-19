from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import DEFAULT_LABELS, SUPPORTED_ARCHITECTURES
from app.dataset import CheXpertDataset
from app.model import build_model


def compute_file_sha256(filepath: Path) -> str:
    if not filepath.exists():
        return "NOT_FOUND"
    h = hashlib.sha256()
    with filepath.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class AsymmetricLoss(torch.nn.Module):
    def __init__(self, gamma_neg: float = 4.0, gamma_pos: float = 1.0, clip: float = 0.05, eps: float = 1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        xs_pos = torch.sigmoid(x)
        xs_neg = 1.0 - xs_pos

        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)

        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1.0 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt0 = xs_pos * y
            pt1 = xs_neg * (1.0 - y)
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1.0 - y)
            one_sided_w = torch.pow(1.0 - pt, one_sided_gamma)
            loss = loss * one_sided_w

        if mask is not None:
            loss = loss * mask
            return -loss.sum() / mask.sum().clamp(min=1.0)
        return -loss.sum() / max(1, x.size(0))


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1.0 - pt) ** self.gamma * bce_loss
        if mask is not None:
            return (focal_loss * mask).sum() / mask.sum().clamp(min=1.0)
        return focal_loss.mean()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Protocol-Compliant CheXpert Multi-Label Training Pipeline.")
    parser.add_argument("--manifest", type=Path, required=True, help="Path to outputs/splits/protocol_v0_1/manifest.json (MANDATORY)")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "protocol_v0_1.yaml", help="Path to protocol YAML config")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "archive", help="Data root directory containing images")
    parser.add_argument("--output-dir", type=Path, help="Directory to save run artifacts (e.g. outputs/runs/convnext_small/seed_42)")
    parser.add_argument("--arch", choices=SUPPORTED_ARCHITECTURES, default="convnext_small", help="Model architecture")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--epochs", type=int, help="Optional training epochs (overrides config)")
    parser.add_argument("--batch-size", type=int, help="Optional batch size (overrides config)")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader worker processes")
    parser.add_argument("--limit", type=int, help="Optional subset limit for fast smoke testing")
    parser.add_argument("--resume", type=Path, help="Resume training from an existing checkpoint .pt")
    return parser.parse_args()


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def calculate_pos_weight(dataset: CheXpertDataset, labels_or_count: int | Sequence[str]) -> torch.Tensor:
    if isinstance(labels_or_count, (list, tuple, Sequence)) and not isinstance(labels_or_count, (str, bytes)):
        num_labels = len(labels_or_count)
    else:
        num_labels = int(labels_or_count)

    pos_counts = np.zeros(num_labels, dtype=np.float32)
    valid_counts = np.zeros(num_labels, dtype=np.float32)

    for i in range(len(dataset)):
        row = dataset.frame.iloc[i]
        for idx, label in enumerate(dataset.labels):
            val, mask = dataset._normalize_label_and_mask(row[label], label)
            if mask > 0.5:
                valid_counts[idx] += 1.0
                if val >= 0.5:
                    pos_counts[idx] += 1.0

    neg_counts = valid_counts - pos_counts
    weights = np.where(pos_counts > 0, neg_counts / np.maximum(pos_counts, 1.0), 1.0)
    weights = np.clip(weights, 0.1, 10.0)
    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> tuple[float, list[list[float]], list[list[float]], list[list[float]]]:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss_numerator = 0.0
    total_mask_denominator = 0.0
    targets_all: list[list[float]] = []
    probs_all: list[list[float]] = []
    masks_all: list[list[float]] = []

    for item in loader:
        images = item[0].to(device, non_blocking=True)
        targets = item[1].to(device, non_blocking=True)
        masks = item[2].to(device, non_blocking=True) if len(item) > 2 else torch.ones_like(targets)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            logits = model(images)
            if isinstance(criterion, (AsymmetricLoss, FocalLoss)):
                loss = criterion(logits, targets, mask=masks)
                batch_loss_val = float(loss.detach().cpu()) * float(masks.sum().clamp(min=1.0).cpu())
            else:
                bce = criterion(logits, targets)
                loss = (bce * masks).sum() / masks.sum().clamp(min=1.0)
                batch_loss_val = float((bce * masks).sum().detach().cpu())

            if is_train:
                if scaler and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        total_loss_numerator += batch_loss_val
        total_mask_denominator += float(masks.sum().cpu())

        targets_all.extend(targets.detach().cpu().tolist())
        probs_all.extend(torch.sigmoid(logits).detach().cpu().tolist())
        masks_all.extend(masks.detach().cpu().tolist())

    epoch_loss = total_loss_numerator / max(1.0, total_mask_denominator)
    return epoch_loss, targets_all, probs_all, masks_all


def checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    labels: list[str],
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    raw_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    return {
        "model_state_dict": raw_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        },
        "labels": labels,
        "metadata": metadata or {},
    }


def main():
    args = parse_args()
    set_seed(args.seed, deterministic=True)

    # 1. Validate and Load Protocol Config
    if not args.config.exists():
        print(f"Error: Config file not found at {args.config}", file=sys.stderr)
        sys.exit(1)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    config_sha256 = compute_file_sha256(args.config)

    # 2. Strict Manifest Verification
    if not args.manifest.exists():
        print(f"Error: Split manifest not found at {args.manifest}", file=sys.stderr)
        sys.exit(1)

    manifest_data = json.loads(args.manifest.read_text(encoding="utf-8"))
    manifest_sha256 = compute_file_sha256(args.manifest)

    # Security Guard: Reject locked test manifest
    if manifest_data.get("role") == "locked_test" or manifest_data.get("locked"):
        raise RuntimeError("SECURITY VIOLATION: train.py cannot accept locked test manifest!")

    train_csv = args.manifest.parent / manifest_data["splits"]["train"]["csv"]
    val_csv = args.manifest.parent / manifest_data["splits"]["internal_validation"]["csv"]

    if not train_csv.is_file():
        raise FileNotFoundError(f"Missing train CSV: {train_csv}")
    if not val_csv.is_file():
        raise FileNotFoundError(f"Missing internal validation CSV: {val_csv}")

    # 3. Setup Hyperparameters & Preprocessing
    out_dir = args.output_dir or PROJECT_ROOT / "outputs" / "runs" / args.arch / f"seed_{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    train_cfg = config.get("training", {})
    aug_cfg = config.get("augmentation", {})
    model_cfg = config.get("model", {})

    epochs = int(args.epochs) if args.epochs else int(train_cfg.get("epochs", 20))
    batch_size = int(args.batch_size) if args.batch_size else int(train_cfg.get("batch_size", 32))
    lr = float(train_cfg.get("learning_rate", 1e-4))
    weight_decay = float(train_cfg.get("weight_decay", 1e-2))
    patience = int(train_cfg.get("early_stopping_patience", 5))
    loss_name = str(train_cfg.get("loss", "asl")).lower()
    unc_policy = str(train_cfg.get("uncertainty_policy", "u_ones_zeros"))
    use_amp = bool(train_cfg.get("amp", True))
    image_size = int(model_cfg.get("image_size", 224))
    pretrained = bool(model_cfg.get("pretrained", True))
    rot_degrees = float(aug_cfg.get("random_rotation_degrees", 7))
    h_flip = bool(aug_cfg.get("horizontal_flip", False))
    labels = model_cfg.get("labels", DEFAULT_LABELS)

    # Preprocessing Pipeline
    train_transform_list = [transforms.Resize((image_size, image_size))]
    if rot_degrees > 0:
        train_transform_list.append(transforms.RandomRotation(degrees=rot_degrees))
    if h_flip:
        train_transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
    train_transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_transform = transforms.Compose(train_transform_list)

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    preprocessing_config = {
        "image_size": image_size,
        "random_rotation_degrees": rot_degrees,
        "horizontal_flip": h_flip,
        "normalization_mean": [0.485, 0.456, 0.406],
        "normalization_std": [0.229, 0.224, 0.225],
    }
    preprocessing_sha256 = hashlib.sha256(json.dumps(preprocessing_config, sort_keys=True).encode()).hexdigest()

    # Datasets & DataLoaders
    train_dataset = CheXpertDataset(
        train_csv,
        args.data_root,
        train_transform,
        labels=labels,
        uncertain_policy=unc_policy,
        view=manifest_data.get("view", "frontal"),
    )
    val_dataset = CheXpertDataset(
        val_csv,
        args.data_root,
        val_transform,
        labels=labels,
        uncertain_policy=unc_policy,
        view=manifest_data.get("view", "frontal"),
    )

    if args.limit:
        train_dataset = Subset(train_dataset, range(min(args.limit, len(train_dataset))))
        val_dataset = Subset(val_dataset, range(min(max(2, args.limit // 4), len(val_dataset))))

    import os
    if args.num_workers is not None:
        workers = args.num_workers
    else:
        workers = min(4, os.cpu_count() or 2) if torch.cuda.is_available() else 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(workers > 0),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(workers > 0),
    )

    # Initialize Model & Loss
    model = build_model(args.arch, len(labels), pretrained=pretrained).to(device)
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"[Multi-GPU] Detected {torch.cuda.device_count()} GPUs. Wrapping model with DataParallel.")
        model = torch.nn.DataParallel(model)

    if loss_name == "asl":
        criterion = AsymmetricLoss(gamma_neg=4.0, gamma_pos=1.0, clip=0.05)
    elif loss_name == "focal":
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
    else:
        criterion = torch.nn.BCEWithLogitsLoss(reduction="none")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and torch.cuda.is_available())

    start_epoch = 1
    best_val_auc = 0.0
    patience_counter = 0
    history = []

    # Handle Resume
    if args.resume and args.resume.exists():
        print(f"Resuming training from checkpoint: {args.resume}")
        loaded = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(loaded["model_state_dict"])
        if loaded.get("optimizer_state_dict"):
            optimizer.load_state_dict(loaded["optimizer_state_dict"])
        if loaded.get("scheduler_state_dict"):
            scheduler.load_state_dict(loaded["scheduler_state_dict"])
        meta = loaded.get("metadata", {})
        start_epoch = int(meta.get("epoch", 0)) + 1
        best_val_auc = float(meta.get("best_internal_validation_auc", 0.0))
        history = list(meta.get("metrics_history", []))

    print(f"\n=======================================================")
    print(f"CheXpert Protocol Training: {args.arch} (Seed {args.seed})")
    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")
    print(f"Epochs: {epochs} | Batch size: {batch_size} | LR: {lr} | Loss: {loss_name}")
    print(f"=======================================================\n")

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()
        train_loss, _, _, _ = run_epoch(model, train_loader, criterion, device, optimizer=optimizer, scaler=scaler)
        scheduler.step()

        # Validation Step
        val_loss, val_targets, val_probs, val_masks = run_epoch(model, val_loader, criterion, device)

        # Compute Internal Validation AUC
        val_targets_arr = np.array(val_targets)
        val_probs_arr = np.array(val_probs)
        val_masks_arr = np.array(val_masks)

        label_aucs = {}
        valid_auc_list = []

        for idx, label in enumerate(labels):
            v_idx = np.where(val_masks_arr[:, idx] > 0.5)[0]
            if len(v_idx) > 0 and len(np.unique(val_targets_arr[v_idx, idx])) > 1:
                try:
                    auc = float(roc_auc_score(val_targets_arr[v_idx, idx], val_probs_arr[v_idx, idx]))
                    label_aucs[label] = auc
                    valid_auc_list.append(auc)
                except Exception:
                    label_aucs[label] = None
            else:
                label_aucs[label] = None

        mean_val_auc = float(np.mean(valid_auc_list)) if valid_auc_list else 0.0
        elapsed = time.time() - t0

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "mean_val_auc": mean_val_auc,
            "label_aucs": label_aucs,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_seconds": elapsed,
        }
        history.append(epoch_record)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {mean_val_auc:.4f} ({elapsed:.1f}s)")

        metadata_dict = {
            "epoch": epoch,
            "architecture": args.arch,
            "seed": args.seed,
            "best_internal_validation_auc": max(best_val_auc, mean_val_auc),
            "config_sha256": config_sha256,
            "split_manifest_sha256": manifest_sha256,
            "preprocessing_sha256": preprocessing_sha256,
            "uncertainty_policy": unc_policy,
            "labels": labels,
            "metrics_history": history,
        }

        # Save last checkpoint
        last_ckpt = out_dir / "last.pt"
        payload_last = checkpoint_payload(model, optimizer, scheduler, labels, metadata=metadata_dict)
        payload_last["scaler_state_dict"] = scaler.state_dict() if scaler else None
        torch.save(payload_last, last_ckpt)

        # Check for improvement & save best checkpoint
        if mean_val_auc > best_val_auc:
            best_val_auc = mean_val_auc
            patience_counter = 0
            best_ckpt = out_dir / "best.pt"
            metadata_dict["best_internal_validation_auc"] = best_val_auc
            payload_best = checkpoint_payload(model, optimizer, scheduler, labels, metadata=metadata_dict)
            payload_best["scaler_state_dict"] = scaler.state_dict() if scaler else None
            torch.save(payload_best, best_ckpt)
            print(f"  -> Best model saved to: {best_ckpt} (AUC: {best_val_auc:.4f})")

            # Export internal validation predictions
            pred_records = []
            underlying_df = val_dataset.dataset.frame if isinstance(val_dataset, Subset) else val_dataset.frame
            for i in range(len(val_dataset)):
                row_item = underlying_df.iloc[val_dataset.indices[i]] if isinstance(val_dataset, Subset) else underlying_df.iloc[i]
                rec = {
                    "study_id": str(row_item.get("study_id", f"study_{i+1}")),
                    "patient_id": str(row_item.get("patient_id", f"patient_{i+1}")),
                }
                for idx, label in enumerate(labels):
                    rec[f"{label}_prob"] = float(val_probs_arr[i, idx])
                    rec[f"{label}_target"] = float(val_targets_arr[i, idx])
                    rec[f"{label}_mask"] = float(val_masks_arr[i, idx])
                pred_records.append(rec)
            pd.DataFrame(pred_records).to_csv(out_dir / "internal_validation_predictions.csv", index=False)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch} (patience={patience}).")
                break

    # Save training history JSON and run manifest
    (out_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    run_manifest = {
        "architecture": args.arch,
        "seed": args.seed,
        "best_val_auc": best_val_auc,
        "epochs_trained": len(history),
        "best_checkpoint_sha256": compute_file_sha256(out_dir / "best.pt"),
        "config_sha256": config_sha256,
        "split_manifest_sha256": manifest_sha256,
        "preprocessing_sha256": preprocessing_sha256,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    print(f"\nRun artifacts successfully saved in: {out_dir}")


if __name__ == "__main__":
    main()
