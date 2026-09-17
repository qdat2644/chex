from __future__ import annotations

import hashlib
import json
import math
import os
import random
import subprocess
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch


def canonical_json_sha256(payload: dict[str, Any]) -> str:
    """
    Computes deterministic SHA-256 hash of a JSON payload.
    Uses sorted keys, no whitespace separators, utf-8 encoding, and disallows NaN/Infinity.
    """
    sanitized = sanitize_for_json(payload)
    encoded = json.dumps(
        sanitized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def compute_file_sha256(filepath: Path | str) -> str:
    """Computes SHA-256 hash of a file on disk."""
    p = Path(filepath)
    if not p.exists() or not p.is_file():
        return "NOT_FOUND"
    h = hashlib.sha256()
    with p.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def sanitize_for_json(obj: Any) -> Any:
    """Recursively sanitize data structures for standard JSON serialization without NaN/Inf."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, (np.floating, np.integer)):
        val = obj.item()
        if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
            return None
        return val
    elif isinstance(obj, np.ndarray):
        return [sanitize_for_json(v) for v in obj.tolist()]
    elif isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, set)):
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, Path):
        return str(obj)
    return obj


def build_resolved_scientific_config(
    protocol_version: str,
    architecture: str,
    seed: int,
    labels: Sequence[str],
    view: str,
    input_size: int,
    uncertainty_policy: str,
    split_manifest_sha256: str,
    optimizer: str,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    max_epochs: int,
    scheduler: str,
    loss: str,
    early_stopping_patience: int,
    mixed_precision: bool,
    rotation_degrees: float,
    horizontal_flip: bool,
    deterministic: bool = True,
) -> dict[str, Any]:
    """
    Constructs the canonical resolved scientific config dictionary.
    Strictly differentiates scientific parameters from operational ones
    (output_dir, resume path, stop_after_epoch, num_workers, timestamps, hostnames are excluded).
    """
    loss_canonical = loss.lower()
    if loss_canonical == "asl":
        loss_canonical = "asymmetric"

    return {
        "schema_version": 1,
        "protocol_version": str(protocol_version),
        "architecture": str(architecture),
        "seed": int(seed),
        "labels": list(labels),
        "data": {
            "view": str(view),
            "input_size": int(input_size),
            "uncertainty_policy": str(uncertainty_policy),
            "split_manifest_sha256": str(split_manifest_sha256),
        },
        "optimization": {
            "optimizer": str(optimizer),
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "batch_size": int(batch_size),
            "max_epochs": int(max_epochs),
            "scheduler": str(scheduler),
            "loss": loss_canonical,
            "early_stopping_patience": int(early_stopping_patience),
            "mixed_precision": bool(mixed_precision),
        },
        "augmentation": {
            "rotation_degrees": float(rotation_degrees),
            "horizontal_flip": bool(horizontal_flip),
        },
        "reproducibility": {
            "deterministic": bool(deterministic),
        },
    }


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Unwraps model from DataParallel or DistributedDataParallel wrapper."""
    return model.module if hasattr(model, "module") else model


def atomic_save_torch(payload: object, filepath: Path | str) -> None:
    """Saves a PyTorch object atomically via a temporary file replacement."""
    p = Path(filepath)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = p.with_name(f"{p.name}.tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(p)


def seed_worker(worker_id: int) -> None:
    """Initializes DataLoader worker with deterministic seed derived from PyTorch initial seed."""
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def get_git_commit(cwd: Path | str | None = None) -> str:
    """Attempts to retrieve current Git commit SHA."""
    try:
        res = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd) if cwd else None,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return res if res else "UNKNOWN"
    except Exception:
        return "UNKNOWN"


def get_git_dirty(cwd: Path | str | None = None) -> bool:
    """Return whether the repository has tracked or untracked changes, failing closed."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(cwd) if cwd else None,
            text=True,
            capture_output=True,
            check=True,
        )
    except Exception as exc:
        raise RuntimeError("Unable to determine Git working-tree status") from exc
    return bool(result.stdout.strip())


def get_safe_rng_state() -> dict[str, Any]:
    """Captures Python, NumPy, PyTorch CPU, and PyTorch CUDA RNG states safely."""
    np_state = np.random.get_state()
    np_state_safe = (np_state[0], np_state[1].tolist(), np_state[2], np_state[3], np_state[4])
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    return {
        "python": random.getstate(),
        "numpy": np_state_safe,
        "torch": torch.get_rng_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": cuda_rng,
        "cuda": cuda_rng,
    }


def restore_safe_rng_state(rng: dict[str, Any] | None) -> None:
    """Restores Python, NumPy, PyTorch CPU, and PyTorch CUDA RNG states safely."""
    if not rng or not isinstance(rng, dict):
        return
    if "python" in rng and rng["python"] is not None:
        random.setstate(rng["python"])
    if "numpy" in rng and rng["numpy"] is not None:
        np_st = rng["numpy"]
        if isinstance(np_st, (tuple, list)) and len(np_st) == 5:
            arr_part = np_st[1]
            if isinstance(arr_part, list):
                arr_part = np.array(arr_part, dtype=np.uint32)
            np.random.set_state((np_st[0], arr_part, np_st[2], np_st[3], np_st[4]))
        else:
            np.random.set_state(np_st)
    if "torch_cpu" in rng and rng["torch_cpu"] is not None:
        torch.set_rng_state(rng["torch_cpu"])
    elif "torch" in rng and rng["torch"] is not None:
        torch.set_rng_state(rng["torch"])

    cuda_st = rng.get("torch_cuda") if "torch_cuda" in rng else rng.get("cuda")
    if cuda_st is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(cuda_st)
        except Exception:
            pass
