from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset
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
    parser.add_argument("--pretrained", action="store_true", help="Initialize DenseNet121 with ImageNet weights.")
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
) -> tuple[float, list[list[float]], list[list[float]]]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    all_targets: list[list[float]] = []
    all_probs: list[list[float]] = []

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss = criterion(logits, targets)
            if optimizer:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        total_loss += float(loss.detach().cpu()) * images.size(0)
        all_targets.extend(targets.detach().cpu().tolist())
        all_probs.extend(torch.sigmoid(logits).detach().cpu().tolist())

    return total_loss / len(loader.dataset), all_targets, all_probs


def mean_auc(targets: list[list[float]], probs: list[list[float]], labels: list[str]) -> float | None:
    aucs = []
    for label_index in range(len(labels)):
        y_true = [row[label_index] for row in targets]
        y_score = [row[label_index] for row in probs]
        if len(set(y_true)) < 2:
            continue
        aucs.append(roc_auc_score(y_true, y_score))
    if not aucs:
        return None
    return float(sum(aucs) / len(aucs))


def main() -> None:
    args = parse_args()
    labels = resolve_label_preset(args.label_preset)
    train_csv = args.train_csv or args.data_root / "train.csv"
    valid_csv = args.valid_csv or args.data_root / "valid.csv"

    train_dataset = CheXpertDataset(
        train_csv,
        args.data_root,
        build_transforms(train=True),
        labels=labels,
        uncertain_policy=args.uncertain_policy,
    )

    if valid_csv.exists():
        if args.limit:
            train_dataset = Subset(train_dataset, range(min(args.limit, len(train_dataset))))
        valid_dataset = CheXpertDataset(
            valid_csv,
            args.data_root,
            build_transforms(train=False),
            labels=labels,
            uncertain_policy=args.uncertain_policy,
        )
    else:
        valid_source = CheXpertDataset(
            train_csv,
            args.data_root,
            build_transforms(train=False),
            labels=labels,
            uncertain_policy=args.uncertain_policy,
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
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(len(labels), pretrained=args.pretrained).to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_auc = -1.0
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, _, _ = run_epoch(model, train_loader, criterion, device, optimizer)
        valid_loss, valid_targets, valid_probs = run_epoch(model, valid_loader, criterion, device)
        auc = mean_auc(valid_targets, valid_probs, labels)
        auc_text = "n/a" if auc is None else f"{auc:.4f}"
        print(f"epoch={epoch} train_loss={train_loss:.4f} valid_loss={valid_loss:.4f} mean_auc={auc_text}")

        score = auc if auc is not None else -valid_loss
        if score > best_auc:
            best_auc = score
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "labels": labels,
                    "image_size": DEFAULT_IMAGE_SIZE,
                },
                args.output,
            )
            print(f"saved={args.output}")


if __name__ == "__main__":
    main()
