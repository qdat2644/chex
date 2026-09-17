from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import DEFAULT_LABELS, SUPPORTED_ARCHITECTURES
from app.dataset import CheXpertDataset
from app.experiment_integrity import (
    atomic_save_torch,
    build_resolved_scientific_config,
    canonical_json_sha256,
    compute_file_sha256,
    get_git_commit,
    get_safe_rng_state,
    restore_safe_rng_state,
    sanitize_for_json,
    seed_worker,
    unwrap_model,
)
from app.model import build_model


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
    parser.add_argument("--stop-after-epoch", type=int, default=None, help="Operational option to interrupt training after N epochs for resume verification")
    parser.add_argument("--run-mode", choices=["smoke", "full"], default=None, help="Execution run mode (smoke or full)")
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
    scaler: torch.amp.GradScaler | None,
    epoch: int,
    planned_max_epochs: int,
    best_val_auc: float | None,
    architecture: str,
    seed: int,
    config_sha256: str,
    resolved_config_sha256: str,
    manifest_sha256: str,
    labels: list[str],
    run_mode: str,
    generator: torch.Generator | None = None,
    git_commit: str = "UNKNOWN",
    history: list[dict[str, Any]] | None = None,
    protocol_version: str = "0.1",
) -> dict[str, object]:
    """
    Constructs Checkpoint Schema Version 2 payload with backward compatibility aliases.
    Always unwraps DataParallel models before serializing weights.
    """
    raw_model = unwrap_model(model)
    raw_weights = raw_model.state_dict()
    clean_weights = {k[7:] if k.startswith("module.") else k: v for k, v in raw_weights.items()}

    scaler_st = scaler.state_dict() if scaler else None
    gen_st = generator.get_state() if generator else None
    rng = get_safe_rng_state()

    safe_auc = best_val_auc if (best_val_auc is not None and not math.isnan(best_val_auc) and not math.isinf(best_val_auc)) else None

    # Checkpoint Schema Version 2
    return {
        "checkpoint_schema_version": 2,
        "protocol_version": str(protocol_version),
        "run_mode": str(run_mode),
        "architecture": str(architecture),
        "seed": int(seed),
        "epoch": int(epoch),
        "planned_max_epochs": int(planned_max_epochs),
        "best_val_auc": safe_auc,
        "labels": list(labels),
        "model_state": clean_weights,
        "optimizer_state": optimizer.state_dict() if optimizer else None,
        "scheduler_state": scheduler.state_dict() if scheduler else None,
        "scaler_state": scaler_st,
        "rng_state": {
            "python": rng["python"],
            "numpy": rng["numpy"],
            "torch_cpu": rng["torch_cpu"],
            "torch_cuda": rng["torch_cuda"],
        },
        "train_loader_generator_state": gen_st,
        "config_sha256": str(config_sha256),
        "resolved_config_sha256": str(resolved_config_sha256),
        "manifest_sha256": str(manifest_sha256),
        "git_commit": str(git_commit),
        # Backwards compatibility aliases
        "model_state_dict": clean_weights,
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "scaler_state_dict": scaler_st,
        "dataloader_generator_state": gen_st,
        "metadata": {
            "checkpoint_schema_version": 2,
            "architecture": str(architecture),
            "seed": int(seed),
            "epoch": int(epoch),
            "planned_max_epochs": int(planned_max_epochs),
            "best_internal_validation_auc": safe_auc,
            "config_sha256": str(config_sha256),
            "resolved_config_sha256": str(resolved_config_sha256),
            "split_manifest_sha256": str(manifest_sha256),
            "labels": list(labels),
            "run_mode": str(run_mode),
            "git_commit": str(git_commit),
            "metrics_history": history or [],
        },
    }


def main():
    started_at_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
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

    # 3. Setup Hyperparameters & Operational Configurations
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
    labels = list(model_cfg.get("labels", DEFAULT_LABELS))

    # Determine execution run mode
    if args.run_mode:
        run_mode = args.run_mode
    else:
        run_mode = "smoke" if (args.limit or (args.epochs and args.epochs <= 2)) else "full"

    git_commit = get_git_commit(PROJECT_ROOT)

    # 4. Construct Resolved Scientific Config & Canonical Hash (Part A)
    resolved_config = build_resolved_scientific_config(
        protocol_version=config.get("protocol_version", "0.1"),
        architecture=args.arch,
        seed=args.seed,
        labels=labels,
        view=manifest_data.get("view", "frontal"),
        input_size=image_size,
        uncertainty_policy=unc_policy,
        split_manifest_sha256=manifest_sha256,
        optimizer=train_cfg.get("optimizer", "AdamW"),
        learning_rate=lr,
        weight_decay=weight_decay,
        batch_size=batch_size,
        max_epochs=epochs,
        scheduler=train_cfg.get("scheduler", "cosine"),
        loss=loss_name,
        early_stopping_patience=patience,
        mixed_precision=use_amp,
        rotation_degrees=rot_degrees,
        horizontal_flip=h_flip,
        deterministic=True,
    )
    resolved_config_sha256 = canonical_json_sha256(resolved_config)
    resolved_config_path = out_dir / "resolved_config.json"
    resolved_config_path.write_text(json.dumps(resolved_config, indent=2), encoding="utf-8")

    # 5. Preprocessing Pipelines
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

    # 6. Datasets & DataLoaders with Deterministic Worker Init (C5)
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

    if args.num_workers is not None:
        workers = args.num_workers
    else:
        workers = min(4, os.cpu_count() or 2) if torch.cuda.is_available() else 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,  # C5: persistent_workers=False for strict deterministic reproducibility
        worker_init_fn=seed_worker,
        generator=train_generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
    )

    # 7. Initialize Model, Loss, Optimizer
    model = build_model(args.arch, len(labels), pretrained=pretrained).to(device)
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"[Multi-GPU] Detected {torch.cuda.device_count()} GPUs. Wrapping model with DataParallel.")
        model = torch.nn.DataParallel(model)

    if loss_name in ["asl", "asymmetric"]:
        criterion = AsymmetricLoss(gamma_neg=4.0, gamma_pos=1.0, clip=0.05)
    elif loss_name == "focal":
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
    else:
        criterion = torch.nn.BCEWithLogitsLoss(reduction="none")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and torch.cuda.is_available())

    start_epoch = 1
    best_val_auc = -float("inf")
    best_epoch = 1
    patience_counter = 0
    history: list[dict[str, Any]] = []

    # 8. Resume Validation & State Restoration (Part C)
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found at: {resume_path}")

        ckpt_sha = compute_file_sha256(resume_path)
        if not ckpt_sha or ckpt_sha == "NOT_FOUND":
            raise RuntimeError(f"Cannot compute SHA-256 for checkpoint {resume_path}")

        print(f"Resuming training from checkpoint: {resume_path} (SHA-256: {ckpt_sha})")
        loaded = torch.load(resume_path, map_location="cpu", weights_only=True)
        if not isinstance(loaded, dict):
            raise RuntimeError(f"Invalid checkpoint format in {resume_path}")

        ckpt_meta = loaded.get("metadata", {})
        ckpt_arch = loaded.get("architecture") or ckpt_meta.get("architecture")
        ckpt_seed = loaded.get("seed") if "seed" in loaded and loaded["seed"] is not None else ckpt_meta.get("seed")
        ckpt_labels = loaded.get("labels") or ckpt_meta.get("labels")
        ckpt_resolved_cfg = loaded.get("resolved_config_sha256") or ckpt_meta.get("resolved_config_sha256")
        ckpt_config_sha = loaded.get("config_sha256") or ckpt_meta.get("config_sha256")
        ckpt_manifest = loaded.get("manifest_sha256") or ckpt_meta.get("split_manifest_sha256")
        ckpt_run_mode = loaded.get("run_mode") or ckpt_meta.get("run_mode")
        ckpt_epoch = loaded.get("epoch", ckpt_meta.get("epoch"))

        model_st = loaded.get("model_state") or loaded.get("model_state_dict")
        opt_st = loaded.get("optimizer_state") or loaded.get("optimizer_state_dict")
        sched_st = loaded.get("scheduler_state") or loaded.get("scheduler_state_dict")
        scaler_st = loaded.get("scaler_state") or loaded.get("scaler_state_dict")
        rng_st = loaded.get("rng_state")
        gen_st = loaded.get("train_loader_generator_state") or loaded.get("dataloader_generator_state")

        # C2: Fail-closed verification before loading state
        if run_mode == "full":
            missing_fields = []
            if ckpt_arch is None: missing_fields.append("architecture")
            if ckpt_seed is None: missing_fields.append("seed")
            if ckpt_labels is None: missing_fields.append("labels")
            if ckpt_resolved_cfg is None: missing_fields.append("resolved_config_sha256")
            if ckpt_manifest is None: missing_fields.append("manifest_sha256")
            if ckpt_epoch is None: missing_fields.append("epoch")
            if model_st is None: missing_fields.append("model_state")
            if opt_st is None: missing_fields.append("optimizer_state")
            if sched_st is None: missing_fields.append("scheduler_state")
            if scaler_st is None and use_amp and torch.cuda.is_available(): missing_fields.append("scaler_state")
            if rng_st is None: missing_fields.append("rng_state")
            if gen_st is None: missing_fields.append("train_loader_generator_state")
            if missing_fields:
                raise RuntimeError(f"Resume checkpoint is missing mandatory fields in full mode: {missing_fields}")

        # Absolute comparisons
        if ckpt_arch is None or str(ckpt_arch) != str(args.arch):
            raise RuntimeError(f"Resume architecture mismatch: checkpoint has {ckpt_arch!r}, expected {args.arch!r}")
        if ckpt_seed is None or int(ckpt_seed) != int(args.seed):
            raise RuntimeError(f"Resume seed mismatch: checkpoint has {ckpt_seed!r}, expected {args.seed!r}")
        if ckpt_labels is None or list(ckpt_labels) != list(labels):
            raise RuntimeError(f"Resume label order mismatch: checkpoint has {ckpt_labels!r}, expected {labels!r}")
        if ckpt_resolved_cfg is not None and str(ckpt_resolved_cfg) != str(resolved_config_sha256):
            raise RuntimeError(
                f"Resume resolved config hash mismatch: checkpoint has {ckpt_resolved_cfg!r}, expected {resolved_config_sha256!r}"
            )
        if ckpt_manifest is None or str(ckpt_manifest) != str(manifest_sha256):
            raise RuntimeError(f"Resume manifest hash mismatch: checkpoint has {ckpt_manifest!r}, expected {manifest_sha256!r}")
        if ckpt_run_mode is not None and str(ckpt_run_mode) != str(run_mode):
            raise RuntimeError(f"Resume run mode mismatch: checkpoint has {ckpt_run_mode!r}, expected {run_mode!r}")

        # C1: Load state into unwrap_model(model), stripping module. prefixes if any
        raw_target = unwrap_model(model)
        clean_model_st = {k[7:] if k.startswith("module.") else k: v for k, v in model_st.items()}
        raw_target.load_state_dict(clean_model_st)

        # C4: Full state restoration
        if opt_st and optimizer:
            optimizer.load_state_dict(opt_st)
        if sched_st and scheduler:
            scheduler.load_state_dict(sched_st)
        if scaler_st and scaler:
            scaler.load_state_dict(scaler_st)
        if rng_st:
            restore_safe_rng_state(rng_st)
        if gen_st is not None and train_generator is not None:
            train_generator.set_state(gen_st)

        start_epoch = int(ckpt_epoch) + 1
        raw_best_auc = loaded.get("best_val_auc", ckpt_meta.get("best_internal_validation_auc"))
        if raw_best_auc is not None and not math.isnan(raw_best_auc) and not math.isinf(raw_best_auc):
            best_val_auc = float(raw_best_auc)
        best_epoch = int(ckpt_epoch)
        history = list(ckpt_meta.get("metrics_history", []))

    print(f"\n=======================================================")
    print(f"CheXpert Protocol Training: {args.arch} (Seed {args.seed}) | Mode: {run_mode}")
    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")
    print(f"Epochs: {epochs} | Batch size: {batch_size} | LR: {lr} | Loss: {loss_name}")
    print(f"Resolved Config SHA-256: {resolved_config_sha256}")
    print(f"=======================================================\n")

    training_status = "completed"

    # 9. Training Loop (B2, B3, B4, C6)
    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()
        train_loss, _, _, _ = run_epoch(model, train_loader, criterion, device, optimizer=optimizer, scaler=scaler)
        scheduler.step()

        # Validation Step
        val_loss, val_targets, val_probs, val_masks = run_epoch(model, val_loader, criterion, device)

        # Compute Internal Validation AUROC
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

        mean_val_auc = float(np.mean(valid_auc_list)) if valid_auc_list else None
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

        auc_display = f"{mean_val_auc:.4f}" if mean_val_auc is not None else "N/A"
        print(f"Epoch [{epoch:02d}/{epochs:02d}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {auc_display} ({elapsed:.1f}s)")

        # B2: Update best_val_auc BEFORE creating payload
        current_auc = mean_val_auc if (mean_val_auc is not None and not math.isnan(mean_val_auc)) else None
        is_best = (current_auc is not None and current_auc > best_val_auc)
        if is_best:
            best_val_auc = current_auc
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        payload = checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            planned_max_epochs=epochs,
            best_val_auc=best_val_auc if best_val_auc != -float("inf") else None,
            architecture=args.arch,
            seed=args.seed,
            config_sha256=config_sha256,
            resolved_config_sha256=resolved_config_sha256,
            manifest_sha256=manifest_sha256,
            labels=labels,
            run_mode=run_mode,
            generator=train_generator,
            git_commit=git_commit,
            history=history,
            protocol_version=config.get("protocol_version", "0.1"),
        )

        # B4: Atomic write for last.pt
        last_ckpt = out_dir / "last.pt"
        atomic_save_torch(payload, last_ckpt)

        # B3: Epoch 1 always creates best.pt, or if improvement detected
        best_ckpt = out_dir / "best.pt"
        if is_best or not best_ckpt.exists():
            best_epoch = epoch
            atomic_save_torch(payload, best_ckpt)
            print(f"  -> Best model saved to: {best_ckpt} (AUC: {auc_display})")

        # Early stopping check (only in full mode)
        if run_mode == "full" and patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch} (patience={patience}).")
            break

        # C6: Operational --stop-after-epoch check
        if args.stop_after_epoch is not None and epoch >= args.stop_after_epoch:
            print(f"Operational stop reached at epoch {epoch} (stop_after_epoch={args.stop_after_epoch}).")
            training_status = "interrupted"
            break

    # Save training history JSON
    (out_dir / "training_history.json").write_text(json.dumps(sanitize_for_json(history), indent=2), encoding="utf-8")

    # 10. Re-inference on Internal Validation split using best.pt weights (Part E)
    best_ckpt = out_dir / "best.pt"
    if best_ckpt.is_file():
        best_payload = torch.load(best_ckpt, map_location=device, weights_only=True)
        best_weights = best_payload.get("model_state") or best_payload.get("model_state_dict")
        clean_weights = {k[7:] if k.startswith("module.") else k: v for k, v in best_weights.items()}
        unwrap_model(model).load_state_dict(clean_weights)

    model.eval()
    val_targets_all, val_probs_all, val_masks_all = [], [], []
    with torch.no_grad():
        for item in val_loader:
            imgs = item[0].to(device, non_blocking=True)
            tgts = item[1].to(device, non_blocking=True)
            msks = item[2].to(device, non_blocking=True) if len(item) > 2 else torch.ones_like(tgts)
            logits = model(imgs)
            probs = torch.sigmoid(logits)
            val_targets_all.extend(tgts.cpu().tolist())
            val_probs_all.extend(probs.cpu().tolist())
            val_masks_all.extend(msks.cpu().tolist())

    underlying_df = val_dataset.dataset.frame if isinstance(val_dataset, Subset) else val_dataset.frame
    pred_records = []
    for i in range(len(val_dataset)):
        row_item = underlying_df.iloc[val_dataset.indices[i]] if isinstance(val_dataset, Subset) else underlying_df.iloc[i]
        rec = {
            "study_id": str(row_item.get("study_id", f"study_{i+1}")),
            "patient_id": str(row_item.get("patient_id", f"patient_{i+1}")),
            "image_path": str(row_item.get("Path", row_item.get("image_path", ""))),
        }
        for idx, lbl in enumerate(labels):
            safe_lbl = lbl.replace(" ", "_")
            rec[f"{safe_lbl}_target"] = float(val_targets_all[i][idx])
            rec[f"{safe_lbl}_mask"] = float(val_masks_all[i][idx])
            rec[f"{safe_lbl}_prob"] = float(val_probs_all[i][idx])
        pred_records.append(rec)

    pred_csv_path = out_dir / "internal_validation_predictions.csv"
    pd.DataFrame(pred_records).to_csv(pred_csv_path, index=False)
    pred_csv_sha256 = compute_file_sha256(pred_csv_path)

    # 11. Run Manifest Schema Version 1 (Part D)
    gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []
    cudnn_version = str(torch.backends.cudnn.version()) if (torch.cuda.is_available() and hasattr(torch.backends, "cudnn")) else "None"
    cuda_version = str(torch.version.cuda) if torch.cuda.is_available() else "None"

    run_manifest = {
        "schema_version": 1,
        "protocol_version": str(config.get("protocol_version", "0.1")),
        "run_mode": str(run_mode),
        "status": str(training_status),
        "architecture": str(args.arch),
        "seed": int(args.seed),
        "git_commit": str(git_commit),
        "resolved_config_sha256": str(resolved_config_sha256),
        "manifest_sha256": str(manifest_sha256),
        "best_checkpoint_sha256": compute_file_sha256(out_dir / "best.pt"),
        "last_checkpoint_sha256": compute_file_sha256(out_dir / "last.pt"),
        "predictions_sha256": str(pred_csv_sha256),
        "best_epoch": int(best_epoch),
        "best_val_auc": best_val_auc if (best_val_auc is not None and best_val_auc != -float("inf") and not math.isnan(best_val_auc) and not math.isinf(best_val_auc)) else None,
        "started_at_utc": started_at_utc,
        "completed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": {
            "python": sys.version.split()[0],
            "pytorch": torch.__version__,
            "torchvision": torchvision.__version__,
            "cuda": cuda_version,
            "cudnn": cudnn_version,
            "gpu_names": gpu_names,
            "gpu_count": len(gpu_names),
        },
    }
    sanitized_manifest = sanitize_for_json(run_manifest)
    (out_dir / "run_manifest.json").write_text(json.dumps(sanitized_manifest, indent=2), encoding="utf-8")
    print(f"\nRun artifacts successfully saved in: {out_dir}")


if __name__ == "__main__":
    main()
