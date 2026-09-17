from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Iterable

import pandas as pd


LOCKED_MARKERS = ("locked_test", "locked-test", "test-set")
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def require(condition: bool, message: str, exc_type: type[Exception] = RuntimeError) -> None:
    if not condition:
        raise exc_type(message)


def file_sha256(path: Path) -> str:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_path(value: object) -> str:
    return "/" + str(value).replace("\\", "/").strip("/").lower() + "/"


def validate_development_source(
    data_root: Path,
    train_csv: Path,
    csv_paths: Iterable[object],
    metadata: dict[str, object] | None = None,
    manifest: dict[str, object] | None = None,
) -> None:
    """Reject any source that could be validation, test, or locked-test data."""
    metadata = metadata or {}
    manifest = manifest or {}
    for candidate in (data_root, train_csv):
        normalized = _normalized_path(candidate)
        if any(marker in normalized for marker in LOCKED_MARKERS):
            raise RuntimeError(f"Locked/test marker is forbidden in source path: {candidate}")

    if str(metadata.get("role", "")).lower() == "locked_test" or metadata.get("locked") is True:
        raise RuntimeError("Source metadata identifies locked-test data")
    if str(manifest.get("role", "")).lower() == "locked_test" or manifest.get("locked") is True:
        raise RuntimeError("Manifest identifies locked-test data")
    splits = manifest.get("splits", {})
    if isinstance(splits, dict):
        for name, split_metadata in splits.items():
            if str(name).lower() == "locked_test":
                raise RuntimeError("Manifest contains a locked_test split")
            if isinstance(split_metadata, dict):
                if str(split_metadata.get("role", "")).lower() == "locked_test" or split_metadata.get("locked") is True:
                    raise RuntimeError(f"Manifest split '{name}' identifies locked-test data")
                split_path = _normalized_path(split_metadata.get("csv", split_metadata.get("path", "")))
                if any(marker in split_path for marker in LOCKED_MARKERS):
                    raise RuntimeError(f"Manifest split '{name}' contains a locked/test path marker")

    violations: list[str] = []
    for value in csv_paths:
        normalized = _normalized_path(value)
        if (
            "/valid/" in normalized
            or "/test/" in normalized
            or any(marker in normalized for marker in LOCKED_MARKERS)
            or "/train/" not in normalized
        ):
            violations.append(str(value))
            if len(violations) >= 20:
                break
    if violations:
        raise RuntimeError(
            "CSV contains non-training or locked/test image paths (first 20):\n"
            + "\n".join(violations)
        )


def select_train_csv(input_root: Path, data_root: Path | None = None) -> Path:
    input_root = Path(input_root)
    all_candidates = sorted(path.resolve() for path in input_root.glob("**/train.csv") if path.is_file())
    candidates = all_candidates
    if data_root is not None:
        root = Path(data_root).resolve()
        candidates = [path for path in all_candidates if path == root / "train.csv" or root in path.parents]
    if len(candidates) != 1:
        rendered = "\n".join(str(path) for path in all_candidates) or "<none>"
        raise RuntimeError(
            f"Expected exactly one train.csv after DATA_ROOT filtering; found {len(candidates)}. "
            f"All candidates:\n{rendered}"
        )
    return candidates[0]


def _resolve_image(root: Path, path_value: object) -> Path | None:
    path = Path(str(path_value))
    possibilities = [path] if path.is_absolute() else [root / path]
    if not path.is_absolute() and path.parts and path.parts[0].startswith("CheXpert"):
        possibilities.append(root / Path(*path.parts[1:]))
    matches = [candidate for candidate in possibilities if candidate.is_file()]
    return matches[0] if matches else None


def resolve_unique_data_root(
    train_csv: Path,
    root_candidates: Iterable[Path],
    sample_size: int = 50,
) -> Path:
    frame = pd.read_csv(train_csv, usecols=["Path"])
    if len(frame) < sample_size:
        raise RuntimeError(f"Source CSV must contain at least {sample_size} paths for root verification")
    samples = frame["Path"].iloc[:sample_size].tolist()
    roots: list[Path] = []
    seen: set[Path] = set()
    for value in root_candidates:
        root = Path(value).resolve()
        if root in seen or not root.is_dir():
            continue
        seen.add(root)
        if all(_resolve_image(root, path) is not None for path in samples):
            roots.append(root)
    if len(roots) != 1:
        rendered = "\n".join(str(root) for root in roots) or "<none>"
        raise RuntimeError(
            f"Expected exactly one data root resolving all {sample_size} sample paths; "
            f"found {len(roots)}:\n{rendered}"
        )
    return roots[0]


def validate_split_image_hashes(split_frames: dict[str, pd.DataFrame]) -> int:
    hash_sets: dict[str, set[str]] = {}
    invalid: list[str] = []
    for split_name, frame in split_frames.items():
        if "image_sha256" not in frame.columns:
            raise RuntimeError(f"Split '{split_name}' is missing mandatory image_sha256 column")
        values: set[str] = set()
        for index, raw in frame["image_sha256"].items():
            value = "" if pd.isna(raw) else str(raw).strip()
            if not SHA256_RE.fullmatch(value) or value.upper() == "NOT_FOUND":
                invalid.append(f"{split_name}[{index}]={value!r}")
                if len(invalid) >= 20:
                    break
            else:
                values.add(value.lower())
        hash_sets[split_name] = values
    if invalid:
        raise RuntimeError("Invalid image SHA-256 values (first 20):\n" + "\n".join(invalid))

    names = list(hash_sets)
    duplicate_count = 0
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            duplicate_count += len(hash_sets[left] & hash_sets[right])
    if duplicate_count:
        raise RuntimeError(f"Detected {duplicate_count} duplicate image hashes across splits")
    return 0


def validate_git_integrity(repo: Path, repo_ref: str, run_mode: str) -> dict[str, object]:
    repo = Path(repo)
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        resolved = subprocess.check_output(
            ["git", "rev-parse", f"{repo_ref}^{{commit}}"], cwd=repo, text=True
        ).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip())
    except Exception as exc:
        raise RuntimeError("Unable to verify Git integrity") from exc
    if not head or head == "UNKNOWN":
        raise RuntimeError("Git commit is UNKNOWN")
    if run_mode == "full":
        if repo_ref.lower() in {"main", "master"}:
            raise RuntimeError("Full mode requires a pinned commit or release tag")
        if dirty:
            raise RuntimeError("Full mode requires a clean Git working tree")
        if head != resolved:
            raise RuntimeError(f"HEAD {head} does not match resolved REPO_REF {resolved}")
    return {"git_commit": head, "git_dirty": dirty, "resolved_repo_ref": resolved}


def verify_expected_source_csv(train_csv: Path, run_mode: str, expected_sha256: str | None) -> str:
    actual = file_sha256(train_csv)
    if run_mode == "full":
        if not expected_sha256:
            raise RuntimeError("EXPECTED_SOURCE_CSV_SHA256 is mandatory in full mode")
        if actual.lower() != str(expected_sha256).strip().lower():
            raise RuntimeError("Source train.csv SHA-256 does not match the pre-registered value")
    return actual


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fail-closed Kaggle dataset and Git preflight")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--run-mode", choices=["smoke", "full"], required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--repo-ref", required=True)
    parser.add_argument("--expected-source-csv-sha256")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_csv = select_train_csv(args.input_root, args.data_root)
    frame = pd.read_csv(train_csv)
    require("Path" in frame.columns, f"Missing Path column in {train_csv}", ValueError)
    roots = [args.data_root] if args.data_root else [train_csv.parent, train_csv.parent.parent]
    data_root = resolve_unique_data_root(train_csv, [root for root in roots if root is not None])
    validate_development_source(data_root, train_csv, frame["Path"])
    source_sha = verify_expected_source_csv(train_csv, args.run_mode, args.expected_source_csv_sha256)
    git_info = validate_git_integrity(args.repo, args.repo_ref, args.run_mode)
    print(json.dumps({"train_csv": str(train_csv), "data_root": str(data_root), "source_csv_sha256": source_sha, **git_info}))


if __name__ == "__main__":
    main()
