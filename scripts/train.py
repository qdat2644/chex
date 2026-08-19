from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import DEFAULT_IMAGE_SIZE, SUPPORTED_ARCHITECTURES, resolve_label_preset
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
    """
    Asymmetric Loss for Multi-Label Classification with strict masking.
    Reduces the impact of easy negative examples and emphasizes hard positive examples.
    """
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
    """
    Multi-label Focal Loss with optional masking.
    """
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
    parser = argparse.ArgumentParser(description="Train an optimized leak-free CheXpert multi-label classifier.")
    parser.add_argument("--data-root", type=Path, required=True, help="Data root directory containing images")
    parser.add_argument("--split-manifest", type=Path, help="Path to outputs/splits/manifest.json (preferred)")
    parser.add_argument("--train-csv", type=Path, help="Explicit path to train.csv split")
    parser.add_argument("--val-csv", type=Path, help="Explicit path to internal_val.csv split")
    parser.add_argument("--limit", type=int, help="Use a small subset for quick smoke tests.")
    parser.add_argument("--output", type=Path, default=Path("checkpoints/chexpert_model.pt"))
    parser.add_argument("--resume", type=Path, help="Path to checkpoint .pt to resume training from.")
    parser.add_argument("--backup-dir", type=Path, help="Directory to automatically backup checkpoints to.")
    parser.add_argument(
        "--arch",
        choices=SUPPORTED_ARCHITECTURES,
        default="convnext_small",
        help="Backbone architecture: convnext_small, densenet121, efficientnet_v2_m, resnet50",
    )
    parser.add_argument(
        "--loss",
        choices=["asl", "focal", "bce"],
        default="asl",
        help="Loss function: asl (Asymmetric Loss), focal (Focal Loss), bce (Binary Cross Entropy)",
    )
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE, help="Input resolution (e.g. 224, 384)")
    parser.add_argument("--epochs", type=int, default=20, help="Total training epochs (default protocol: 20)")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience epochs (default: 5)")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="Optimizer weight decay (default: 0.01)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--uncertain-policy",
        choices=["u_ones_zeros", "smooth", "zero", "one", "ignore"],
        default="u_ones_zeros",
        help="Policy for -1 uncertainty labels (u_ones_zeros=Stanford baseline, smooth=0.6, zero=0, one=1, ignore=masked)",
    )
    parser.add_argument("--label-preset", choices=["competition", "all"], default="competition")
    parser.add_argument("--view", choices=["frontal", "lateral", "all"], default="frontal")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pos-weight", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pretrained", action="store_true", help="Initialize backbone with ImageNet pretrained weights.")
    parser.add_argument("--horizontal-flip", action="store_true", default=False, help="Enable horizontal flip augmentation (default: False)")
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


def checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    labels: list[str],
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "model_state_dict": model.state_dict(),
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

    total_loss_sum = 0.0
    total_mask_sum = 0.0
    targets_all: list[list[float]] = []
    probs_all: list[list[float]] = []
    masks_all: list[list[float]] = []

    for item in loader:
        images = item[0].to(device)
        targets = item[1].to(device)
        masks = item[2].to(device) if len(item) > 2 else torch.ones_like(targets)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            logits = model(images)
            try:
                loss = criterion(logits, targets, mask=masks)
                loss_val = float(loss.detach().cpu()) * float(masks.sum().clamp(min=1.0).cpu())
            except TypeError:
                bce = criterion(logits, targets)
                loss = (bce * masks).sum() / masks.sum().clamp(min=1.0)
                loss_val = float((bce * masks).sum().detach().cpu())

            if is_train:
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        total_loss_sum += loss_val
        total_mask_sum += float(masks.sum().cpu())

        targets_all.extend(targets.detach().cpu().tolist())
        probs_all.extend(torch.sigmoid(logits).detach().cpu().tolist())
        masks_all.extend(masks.detach().cpu().tolist())

    epoch_loss = total_loss_sum / max(1.0, total_mask_sum)
    return epoch_loss, targets_all, probs_all, masks_all


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    # 1. Resolve Train & Internal-Val Split Paths
    train_csv = args.train_csv
    val_csv = args.val_csv

    if args.split_manifest and args.split_manifest.exists():
        manifest_data = json.loads(args.split_manifest.read_text(encoding="utf-8"))
        train_file = manifest_data.get("splits", {}).get("train", {}).get("file", "train.csv")
        val_file = manifest_data.get("splits", {}).get("internal_val", {}).get("file", "internal_val.csv")
        train_csv = args.split_manifest.parent / train_file
        val_csv = args.split_manifest.parent / val_file

    if not train_csv or not train_csv.exists():
        default_train = PROJECT_ROOT / "outputs" / "splits" / "train.csv"
        if default_train.exists():
            train_csv = default_train
        else:
            train_csv = args.data_root / "train.csv"

    if not val_csv or not val_csv.exists():
        default_val = PROJECT_ROOT / "outputs" / "splits" / "internal_val.csv"
        if default_val.exists():
            val_csv = default_val

    # 2. Strict Leakage Prevention Guard: Refuse official valid.csv during training
    for check_path in [train_csv, val_csv]:
        if check_path and "valid.csv" in check_path.name.lower() and "internal" not in check_path.name.lower():
            raise RuntimeError(
                f"LEAKAGE PREVENTION ERROR: '{check_path.name}' detected as training/validation source! "
                f"The official Stanford validation set ('valid.csv') is strictly reserved as the LOCKED TEST SET. "
                f"You must use internal validation split generated by scripts/make_splits.py."
            )

    labels = resolve_label_preset(args.label_preset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device} | Architecture: {args.arch} | Epochs: {args.epochs} | Seed: {args.seed}")
    print(f"Train Split: {train_csv}")
    print(f"Internal Val Split: {val_csv}")

    # Transforms
    train_transform_list = [
        transforms.Resize((args.image_size, args.image_size)),
    ]
    if args.horizontal_flip:
        train_transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
    train_transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_transform = transforms.Compose(train_transform_list)

    val_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor],
    )
    val_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = CheXpertDataset(
        train_csv,
        args.data_root,
        train_transform,
        labels=labels,
        uncertain_policy=args.uncertain_policy,
        view=args.view,
    )

    if val_csv and val_csv.exists():
        val_dataset = CheXpertDataset(
            val_csv,
            args.data_root,
            val_transform,
            labels=labels,
            uncertain_policy=args.uncertain_policy,
            view=args.view,
        )
    else:
        # Fallback subset split if no explicit internal val
        total_len = len(train_dataset)
        val_size = max(1, int(total_len * 0.1))
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset,
            [total_len - val_size, val_size],
            generator=torch.Generator().manual_seed(args.seed),
        )

    if args.limit:
        train_dataset = Subset(train_dataset, range(min(args.limit, len(train_dataset))))
        val_dataset = Subset(val_dataset, range(min(max(2, args.limit // 4), len(val_dataset))))

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model(args.arch, len(labels), pretrained=args.pretrained).to(device)

    # Loss Selection
    if args.loss == "asl":
        criterion = AsymmetricLoss(gamma_neg=4.0, gamma_pos=1.0, clip=0.05)
    elif args.loss == "focal":
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
    else:
        if args.pos_weight and hasattr(train_dataset, "frame"):
            pos_weight = calculate_pos_weight(train_dataset, len(labels)).to(device)
            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
        else:
            criterion = torch.nn.BCEWithLogitsLoss(reduction="none")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and torch.cuda.is_available())

    best_val_auc = 0.0
    patience_counter = 0

    print(f"Starting training loop: {len(train_dataset)} train samples, {len(val_dataset)} internal val samples.")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss_numerator = 0.0
        total_mask_denominator = 0.0

        for images, targets, masks in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=args.amp and torch.cuda.is_available()):
                logits = model(images)
                if args.loss in ("asl", "focal"):
                    loss = criterion(logits, targets, mask=masks)
                    loss_batch_sum = float(loss.detach().cpu()) * float(masks.sum().clamp(min=1.0).cpu())
                else:
                    bce = criterion(logits, targets)
                    loss = (bce * masks).sum() / masks.sum().clamp(min=1.0)
                    loss_batch_sum = float((bce * masks).sum().detach().cpu())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss_numerator += loss_batch_sum
            total_mask_denominator += float(masks.sum().cpu())

        scheduler.step()
        epoch_loss = total_loss_numerator / max(1.0, total_mask_denominator)

        # Validation Step
        model.eval()
        val_targets: list[list[float]] = []
        val_probs: list[list[float]] = []
        val_masks: list[list[float]] = []

        with torch.no_grad():
            for images, targets, masks in val_loader:
                images = images.to(device, non_blocking=True)
                logits = model(images)
                probs = torch.sigmoid(logits)

                val_targets.extend(targets.cpu().tolist())
                val_probs.extend(probs.cpu().tolist())
                val_masks.extend(masks.cpu().tolist())

        val_targets_arr = np.array(val_targets)
        val_probs_arr = np.array(val_probs)
        val_masks_arr = np.array(val_masks)

        aucs = []
        for l_idx in range(len(labels)):
            v_idx = np.where(val_masks_arr[:, l_idx] > 0.5)[0]
            if len(v_idx) > 0 and len(np.unique(val_targets_arr[v_idx, l_idx])) > 1:
                try:
                    auc = float(roc_auc_score(val_targets_arr[v_idx, l_idx], val_probs_arr[v_idx, l_idx]))
                    aucs.append(auc)
                except Exception:
                    pass

        mean_val_auc = float(np.mean(aucs)) if aucs else 0.0
        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] Train Loss: {epoch_loss:.4f} | Internal Val AUC: {mean_val_auc:.4f}")

        # Checkpoint Saving & Early Stopping
        if mean_val_auc > best_val_auc:
            best_val_auc = mean_val_auc
            patience_counter = 0
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "labels": labels,
                "metadata": {
                    "architecture": args.arch,
                    "image_size": args.image_size,
                    "best_val_auc": best_val_auc,
                    "epoch": epoch,
                    "uncertain_policy": args.uncertain_policy,
                    "seed": args.seed,
                },
            }, args.output)
            print(f"  -> Best model saved to {args.output} (Val AUC: {best_val_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping triggered after {epoch} epochs (patience={args.patience}).")
                break

    print(f"\nTraining completed! Best Internal Val AUC: {best_val_auc:.4f}")


if __name__ == "__main__":
    main()
