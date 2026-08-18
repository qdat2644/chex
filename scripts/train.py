from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

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


class AsymmetricLoss(torch.nn.Module):
    """
    Asymmetric Loss for Multi-Label Classification.
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
    parser = argparse.ArgumentParser(description="Train an optimized CheXpert multi-label classifier.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--train-csv", type=Path)
    parser.add_argument("--valid-csv", type=Path)
    parser.add_argument("--valid-split", type=float, default=0.1)
    parser.add_argument("--limit", type=int, help="Use a small subset for quick smoke tests.")
    parser.add_argument("--output", type=Path, default=Path("checkpoints/chexpert_model.pt"))
    parser.add_argument("--resume", type=Path, help="Path to checkpoint .pt to resume training from.")
    parser.add_argument("--backup-dir", type=Path, help="Directory to automatically backup checkpoints to (e.g. Google Drive).")
    parser.add_argument(
        "--arch",
        choices=SUPPORTED_ARCHITECTURES,
        default="densenet121",
        help="Backbone architecture: densenet121, convnext_small, efficientnet_v2_m, resnet50",
    )
    parser.add_argument(
        "--loss",
        choices=["asl", "focal", "bce"],
        default="asl",
        help="Loss function: asl (Asymmetric Loss), focal (Focal Loss), bce (Binary Cross Entropy)",
    )
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE, help="Input resolution (e.g. 224, 384)")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
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
    parser.add_argument(
        "--scheduler",
        choices=["cosine", "plateau", "none"],
        default="cosine",
        help="LR scheduler: cosine=CosineAnnealingLR, plateau=ReduceLROnPlateau, none=fixed LR.",
    )
    return parser.parse_args()


def build_transforms(train: bool, image_size: int = DEFAULT_IMAGE_SIZE) -> transforms.Compose:
    steps = [
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(num_output_channels=3),
    ]
    if train:
        steps.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=7),
            ]
        )
    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return transforms.Compose(steps)


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    use_amp: bool = False,
    desc: str = "epoch",
) -> tuple[float, list[list[float]], list[list[float]], list[list[float]]]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    all_targets: list[list[float]] = []
    all_probs: list[list[float]] = []
    all_masks: list[list[float]] = []

    progress = tqdm(loader, desc=desc, unit="batch", leave=True, dynamic_ncols=True)
    for batch_item in progress:
        images = batch_item[0].to(device)
        targets = batch_item[1].to(device)
        masks = batch_item[2].to(device) if len(batch_item) > 2 else torch.ones_like(targets)

        with torch.set_grad_enabled(is_train), torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            if isinstance(criterion, (AsymmetricLoss, FocalLoss)):
                loss = criterion(logits, targets, mask=masks)
            elif isinstance(criterion, torch.nn.BCEWithLogitsLoss):
                bce = F.binary_cross_entropy_with_logits(
                    logits, targets, pos_weight=criterion.pos_weight, reduction="none"
                )
                loss = (bce * masks).sum() / masks.sum().clamp(min=1.0)
            else:
                loss = criterion(logits, targets)

            if optimizer:
                optimizer.zero_grad(set_to_none=True)
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        batch_loss = float(loss.detach().cpu())
        total_loss += batch_loss * images.size(0)
        all_targets.extend(targets.detach().cpu().tolist())
        all_probs.extend(torch.sigmoid(logits).detach().cpu().tolist())
        all_masks.extend(masks.detach().cpu().tolist())
        progress.set_postfix(loss=f"{batch_loss:.4f}")

    return total_loss / max(1, len(loader.dataset)), all_targets, all_probs, all_masks


def label_aucs(
    targets: list[list[float]],
    probs: list[list[float]],
    labels: list[str],
    masks: list[list[float]] | None = None,
) -> dict[str, float | None]:
    scores: dict[str, float | None] = {}
    for label_index, label in enumerate(labels):
        if masks is not None:
            valid_idx = [i for i, m in enumerate(masks) if m[label_index] > 0.5]
        else:
            valid_idx = list(range(len(targets)))

        y_true = [1.0 if targets[i][label_index] >= 0.5 else 0.0 for i in valid_idx]
        y_score = [probs[i][label_index] for i in valid_idx]

        if len(set(y_true)) < 2:
            scores[label] = None
            continue
        try:
            scores[label] = float(roc_auc_score(y_true, y_score))
        except Exception:
            scores[label] = None
    return scores


def mean_auc(scores: dict[str, float | None]) -> float | None:
    aucs = [score for score in scores.values() if score is not None]
    if not aucs:
        return None
    return float(sum(aucs) / len(aucs))


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        else:
            torch.backends.cudnn.benchmark = True


def subset_dataset(dataset: CheXpertDataset, limit: int | None) -> CheXpertDataset | Subset:
    if not limit:
        return dataset
    return Subset(dataset, range(min(limit, len(dataset))))


def target_frame(dataset: CheXpertDataset | Subset) -> tuple[CheXpertDataset, list[int] | None]:
    if isinstance(dataset, Subset):
        base = dataset.dataset
        if not isinstance(base, CheXpertDataset):
            raise TypeError("Expected Subset over CheXpertDataset.")
        return base, list(dataset.indices)
    return dataset, None


def calculate_pos_weight(dataset: CheXpertDataset | Subset, labels: list[str]) -> torch.Tensor:
    base, indices = target_frame(dataset)
    frame = base.frame.iloc[indices] if indices is not None else base.frame
    weights = []
    for label in labels:
        pairs = [base._normalize_label_and_mask(v, label) for v in frame[label]]
        valid_targets = [t for t, m in pairs if m > 0.5]
        positives = sum(1 for t in valid_targets if t >= 0.5)
        negatives = len(valid_targets) - positives
        weight = negatives / max(1.0, float(positives))
        weights.append(weight)
    return torch.tensor(weights, dtype=torch.float32)


def checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    labels: list[str],
    args: argparse.Namespace,
    epoch: int,
    metric: float,
    metrics_history: list[dict[str, object]],
    scheduler: object | None = None,
) -> dict[str, object]:
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "scheduler_state_dict": scheduler.state_dict() if (scheduler and hasattr(scheduler, "state_dict")) else None,
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        "labels": labels,
        "metadata": {
            "epoch": epoch,
            "best_metric": metric,
            "architecture": args.arch,
            "image_size": args.image_size,
            "loss_type": args.loss,
            "label_preset": args.label_preset,
            "uncertain_policy": args.uncertain_policy,
            "view": args.view,
            "seed": args.seed,
            "amp": args.amp,
            "pos_weight": args.pos_weight,
            "pretrained": args.pretrained,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "torch_version": str(torch.__version__),
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": getattr(torch.version, "cuda", None),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        },
        "metrics": metrics_history,
    }


def write_metrics(path: Path, history: list[dict[str, object]], config: dict[str, object]) -> None:
    payload = {
        "config": config,
        "history": history,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def backup_file(src_path: Path, backup_dir: Path | None) -> None:
    if not backup_dir or not src_path.exists():
        return
    try:
        backup_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, backup_dir / src_path.name)
    except Exception as e:
        print(f"Warning: Failed to backup {src_path} to {backup_dir}: {e}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed, deterministic=True)
    labels = resolve_label_preset(args.label_preset)
    train_csv = args.train_csv or args.data_root / "train.csv"
    valid_csv = args.valid_csv or args.data_root / "valid.csv"

    train_transforms = build_transforms(train=True, image_size=args.image_size)
    valid_transforms = build_transforms(train=False, image_size=args.image_size)

    train_dataset = CheXpertDataset(
        train_csv,
        args.data_root,
        train_transforms,
        labels=labels,
        uncertain_policy=args.uncertain_policy,
        view=args.view,
    )

    if valid_csv.exists():
        train_dataset = subset_dataset(train_dataset, args.limit)
        valid_dataset = CheXpertDataset(
            valid_csv,
            args.data_root,
            valid_transforms,
            labels=labels,
            uncertain_policy=args.uncertain_policy,
            view=args.view,
        )
    else:
        valid_source = CheXpertDataset(
            train_csv,
            args.data_root,
            valid_transforms,
            labels=labels,
            uncertain_policy=args.uncertain_policy,
            view=args.view,
        )
        row_count = min(args.limit, len(train_dataset)) if args.limit else len(train_dataset)
        valid_size = max(1, int(row_count * args.valid_split))
        train_size = row_count - valid_size
        if train_size < 1:
            raise ValueError("Not enough rows to create a train/validation split.")
        indices = torch.randperm(row_count, generator=torch.Generator().manual_seed(args.seed)).tolist()
        train_dataset = Subset(train_dataset, indices[:train_size])
        valid_dataset = Subset(valid_source, indices[train_size:])

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.arch, len(labels), pretrained=args.pretrained).to(device)

    # Loss selection
    if args.loss == "asl":
        criterion = AsymmetricLoss(gamma_neg=4.0, gamma_pos=1.0)
    elif args.loss == "focal":
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
    else:
        pos_weight = calculate_pos_weight(train_dataset, labels).to(device) if args.pos_weight else None
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if device.type == "cuda" else None

    start_epoch = 1
    best_auc = -1.0
    metrics_history: list[dict[str, object]] = []

    resumed_scheduler_state = None
    if args.resume and args.resume.exists():
        print(f"Resuming training from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
        if "optimizer_state_dict" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except Exception as e:
                print(f"Warning: Could not load optimizer state: {e}")
        if "scaler_state_dict" in checkpoint and scaler and checkpoint["scaler_state_dict"]:
            try:
                scaler.load_state_dict(checkpoint["scaler_state_dict"])
            except Exception as e:
                print(f"Warning: Could not load scaler state: {e}")
        if "scheduler_state_dict" in checkpoint:
            resumed_scheduler_state = checkpoint["scheduler_state_dict"]
        if "rng_state" in checkpoint:
            try:
                rng = checkpoint["rng_state"]
                if "python" in rng: random.setstate(rng["python"])
                if "numpy" in rng: np.random.set_state(rng["numpy"])
                if "torch" in rng: torch.set_rng_state(rng["torch"])
                if "torch_cuda" in rng and rng["torch_cuda"] and torch.cuda.is_available():
                    torch.cuda.set_rng_state_all(rng["torch_cuda"])
            except Exception as e:
                print(f"Warning: Could not load RNG state: {e}")

        meta = checkpoint.get("metadata", {})
        start_epoch = meta.get("epoch", 0) + 1
        best_auc = meta.get("best_metric", -1.0)
        metrics_history = checkpoint.get("metrics", [])
        print(f"Resumed successfully at Epoch {start_epoch} (Previous Best AUC: {best_auc:.4f})")

    if args.scheduler == "cosine":
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr * 1e-2
        )
    elif args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=2, min_lr=args.lr * 1e-2
        )
    else:
        scheduler = None

    if scheduler and resumed_scheduler_state:
        try:
            scheduler.load_state_dict(resumed_scheduler_state)
        except Exception as e:
            print(f"Warning: Could not load scheduler state: {e}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    last_output = args.output.with_name(f"{args.output.stem}_last{args.output.suffix}")
    metrics_output = args.output.with_suffix(".metrics.json")
    config = {
        "data_root": str(args.data_root),
        "train_csv": str(train_csv),
        "valid_csv": str(valid_csv) if valid_csv.exists() else None,
        "train_rows": len(train_dataset),
        "valid_rows": len(valid_dataset),
        "architecture": args.arch,
        "loss_type": args.loss,
        "image_size": args.image_size,
        "labels": labels,
        "label_preset": args.label_preset,
        "uncertain_policy": args.uncertain_policy,
        "view": args.view,
        "seed": args.seed,
        "amp": args.amp,
        "pos_weight": args.pos_weight,
        "pretrained": args.pretrained,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "scheduler": args.scheduler,
        "num_workers": args.num_workers,
        "device": str(device),
        "backup_dir": str(args.backup_dir) if args.backup_dir else None,
    }
    print("\n" + "="*60)
    print("CheXpert Training Configuration:")
    print(json.dumps(config, indent=2))
    print("="*60 + "\n")

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n>>> Epoch {epoch}/{args.epochs} Starting...")
        train_loss, _, _, _ = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
            scaler,
            use_amp,
            desc=f"Epoch {epoch}/{args.epochs} [Train]",
        )
        valid_loss, valid_targets, valid_probs, valid_masks = run_epoch(
            model,
            valid_loader,
            criterion,
            device,
            use_amp=use_amp,
            desc=f"Epoch {epoch}/{args.epochs} [Valid]",
        )
        auc_scores = label_aucs(valid_targets, valid_probs, labels, masks=valid_masks)
        auc = mean_auc(auc_scores)
        auc_text = "n/a" if auc is None else f"{auc:.4f}"
        current_lr = optimizer.param_groups[0]["lr"]
        
        print("\n" + "-"*50)
        print(f"📊 Summary Epoch {epoch}/{args.epochs}:")
        print(f"   Train Loss: {train_loss:.4f} | Valid Loss: {valid_loss:.4f}")
        print(f"   Mean AUC: {auc_text} | Learning Rate: {current_lr:.2e}")
        for lbl, val in auc_scores.items():
            val_str = f"{val:.4f}" if val is not None else "n/a"
            print(f"   - {lbl}: AUC {val_str}")
        print("-"*50)

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(auc if auc is not None else -valid_loss)
        elif scheduler is not None:
            scheduler.step()

        score = auc if auc is not None else -valid_loss
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "mean_auc": auc,
            "label_auc": auc_scores,
            "lr": current_lr,
            "score": score,
        }
        metrics_history.append(epoch_metrics)

        # Save last checkpoint
        torch.save(
            checkpoint_payload(model, optimizer, scaler, labels, args, epoch, score, metrics_history, scheduler=scheduler),
            last_output,
        )
        write_metrics(metrics_output, metrics_history, config)
        backup_file(last_output, args.backup_dir)
        backup_file(metrics_output, args.backup_dir)

        # Save best checkpoint
        if score > best_auc:
            best_auc = score
            torch.save(
                checkpoint_payload(model, optimizer, scaler, labels, args, epoch, score, metrics_history, scheduler=scheduler),
                args.output,
            )
            print(f"🌟 New Best Model saved to {args.output} (AUC: {auc_text})")
            backup_file(args.output, args.backup_dir)


if __name__ == "__main__":
    main()
