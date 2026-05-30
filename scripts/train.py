from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm
from torchvision import models, transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import DEFAULT_IMAGE_SIZE, resolve_label_preset
from app.dataset import CheXpertDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CheXpert multi-label classifier.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--train-csv", type=Path)
    parser.add_argument("--valid-csv", type=Path)
    parser.add_argument("--valid-split", type=float, default=0.1)
    parser.add_argument("--limit", type=int, help="Use a small subset for quick smoke tests.")
    parser.add_argument("--output", type=Path, default=Path("checkpoints/chexpert_densenet121.pt"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--uncertain-policy", choices=["zero", "one", "ignore"], default="zero")
    parser.add_argument("--label-preset", choices=["competition", "all"], default="competition")
    parser.add_argument("--view", choices=["frontal", "lateral", "all"], default="frontal")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pos-weight", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pretrained", action="store_true", help="Initialize DenseNet121 with ImageNet weights.")
    parser.add_argument(
        "--scheduler",
        choices=["cosine", "plateau", "none"],
        default="cosine",
        help="LR scheduler: cosine=CosineAnnealingLR, plateau=ReduceLROnPlateau, none=fixed LR.",
    )
    return parser.parse_args()


def build_model(num_labels: int, pretrained: bool) -> torch.nn.Module:
    weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.densenet121(weights=weights)
    in_features = model.classifier.in_features
    model.classifier = torch.nn.Linear(in_features, num_labels)
    return model


def build_transforms(train: bool) -> transforms.Compose:
    steps = [
        transforms.Resize((DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)),
        transforms.Grayscale(num_output_channels=3),
    ]
    if train:
        steps.append(transforms.RandomHorizontalFlip())
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
) -> tuple[float, list[list[float]], list[list[float]]]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    all_targets: list[list[float]] = []
    all_probs: list[list[float]] = []

    progress = tqdm(loader, desc=desc, unit="batch", leave=False, dynamic_ncols=True, ascii=True)
    for images, targets in progress:
        images = images.to(device)
        targets = targets.to(device)

        with torch.set_grad_enabled(is_train), torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
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

        total_loss += float(loss.detach().cpu()) * images.size(0)
        all_targets.extend(targets.detach().cpu().tolist())
        all_probs.extend(torch.sigmoid(logits).detach().cpu().tolist())
        progress.set_postfix(loss=f"{float(loss.detach().cpu()):.4f}")

    return total_loss / len(loader.dataset), all_targets, all_probs


def label_aucs(targets: list[list[float]], probs: list[list[float]], labels: list[str]) -> dict[str, float | None]:
    scores: dict[str, float | None] = {}
    for label_index, label in enumerate(labels):
        y_true = [row[label_index] for row in targets]
        y_score = [row[label_index] for row in probs]
        if len(set(y_true)) < 2:
            scores[label] = None
            continue
        scores[label] = float(roc_auc_score(y_true, y_score))
    return scores


def mean_auc(scores: dict[str, float | None]) -> float | None:
    aucs = [score for score in scores.values() if score is not None]
    if not aucs:
        return None
    return float(sum(aucs) / len(aucs))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
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
    targets = []
    for label in labels:
        values = frame[label].map(base._normalize_label)
        targets.append(values.astype(float).to_numpy())
    target_array = np.stack(targets, axis=1)
    positives = target_array.sum(axis=0)
    negatives = target_array.shape[0] - positives
    weights = negatives / np.clip(positives, 1.0, None)
    return torch.tensor(weights, dtype=torch.float32)


def checkpoint_payload(
    model: torch.nn.Module,
    labels: list[str],
    args: argparse.Namespace,
    epoch: int,
    metric: float,
    metrics_history: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "model_state_dict": model.state_dict(),
        "labels": labels,
        "image_size": DEFAULT_IMAGE_SIZE,
        "metadata": {
            "epoch": epoch,
            "best_metric": metric,
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
            "cuda_version": torch.version.cuda,
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


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    labels = resolve_label_preset(args.label_preset)
    train_csv = args.train_csv or args.data_root / "train.csv"
    valid_csv = args.valid_csv or args.data_root / "valid.csv"

    train_dataset = CheXpertDataset(
        train_csv,
        args.data_root,
        build_transforms(train=True),
        labels=labels,
        uncertain_policy=args.uncertain_policy,
        view=args.view,
    )

    if valid_csv.exists():
        train_dataset = subset_dataset(train_dataset, args.limit)
        valid_dataset = CheXpertDataset(
            valid_csv,
            args.data_root,
            build_transforms(train=False),
            labels=labels,
            uncertain_policy=args.uncertain_policy,
            view=args.view,
        )
    else:
        valid_source = CheXpertDataset(
            train_csv,
            args.data_root,
            build_transforms(train=False),
            labels=labels,
            uncertain_policy=args.uncertain_policy,
            view=args.view,
        )
        row_count = min(args.limit, len(train_dataset)) if args.limit else len(train_dataset)
        valid_size = max(1, int(row_count * args.valid_split))
        train_size = row_count - valid_size
        if train_size < 1:
            raise ValueError("Not enough rows to create a train/validation split.")
        indices = torch.randperm(row_count, generator=torch.Generator().manual_seed(42)).tolist()
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
    model = build_model(len(labels), pretrained=args.pretrained).to(device)
    pos_weight = calculate_pos_weight(train_dataset, labels).to(device) if args.pos_weight else None
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler(device.type, enabled=args.amp and device.type == "cuda")
    use_amp = args.amp and device.type == "cuda"

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

    best_auc = -1.0
    metrics_history: list[dict[str, object]] = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    last_output = args.output.with_name(f"{args.output.stem}_last{args.output.suffix}")
    metrics_output = args.output.with_suffix(".metrics.json")
    config = {
        "data_root": str(args.data_root),
        "train_csv": str(train_csv),
        "valid_csv": str(valid_csv) if valid_csv.exists() else None,
        "train_rows": len(train_dataset),
        "valid_rows": len(valid_dataset),
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
    }
    print(json.dumps(config, indent=2))

    for epoch in range(1, args.epochs + 1):
        train_loss, _, _ = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
            scaler,
            use_amp,
            desc=f"epoch {epoch}/{args.epochs} train",
        )
        valid_loss, valid_targets, valid_probs = run_epoch(
            model,
            valid_loader,
            criterion,
            device,
            use_amp=use_amp,
            desc=f"epoch {epoch}/{args.epochs} valid",
        )
        auc_scores = label_aucs(valid_targets, valid_probs, labels)
        auc = mean_auc(auc_scores)
        auc_text = "n/a" if auc is None else f"{auc:.4f}"
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} valid_loss={valid_loss:.4f} "
            f"mean_auc={auc_text} lr={current_lr:.2e}"
        )

        # Step scheduler
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
        torch.save(
            checkpoint_payload(model, labels, args, epoch, score, metrics_history),
            last_output,
        )
        write_metrics(metrics_output, metrics_history, config)
        if score > best_auc:
            best_auc = score
            torch.save(
                checkpoint_payload(model, labels, args, epoch, score, metrics_history),
                args.output,
            )
            print(f"saved={args.output}")


if __name__ == "__main__":
    main()
