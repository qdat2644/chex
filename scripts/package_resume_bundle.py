from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import zipfile
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.verify_resume_bundle import verify_resume_bundle
from app.experiment_integrity import canonical_json_sha256


def _sha256(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def package_resume_bundle(
    last_checkpoint: Path,
    resolved_config: Path,
    manifest: Path,
    protocol_config: Path,
    output: Path,
) -> Path:
    sources = {
        "last.pt": Path(last_checkpoint),
        "resolved_config.json": Path(resolved_config),
        "manifest.json": Path(manifest),
        "protocol_v0_1.yaml": Path(protocol_config),
    }
    hashes = {name: _sha256(path) for name, path in sources.items()}
    checkpoint = torch.load(sources["last.pt"], map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise RuntimeError("last.pt is not a checkpoint dictionary")
    if checkpoint.get("run_mode") != "smoke":
        raise RuntimeError("Resume bundle creation is restricted to smoke checkpoints")
    resolved_payload = json.loads(sources["resolved_config.json"].read_text(encoding="utf-8"))
    cross_links = {
        "resolved_config_sha256": canonical_json_sha256(resolved_payload),
        "manifest_sha256": hashes["manifest.json"],
        "config_sha256": hashes["protocol_v0_1.yaml"],
    }
    for field, actual in cross_links.items():
        expected = checkpoint.get(field) or checkpoint.get("metadata", {}).get(
            "split_manifest_sha256" if field == "manifest_sha256" else field
        )
        if expected != actual:
            raise RuntimeError(f"Resume bundle {field} does not match last.pt metadata")

    ledger = {
        "schema_version": 1,
        "status": "NON_FINAL_SMOKE_TEST",
        "architecture": checkpoint.get("architecture"),
        "seed": checkpoint.get("seed"),
        "run_mode": checkpoint.get("run_mode"),
        "files": hashes,
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_name(output.name + ".tmp")
    with tempfile.TemporaryDirectory() as temporary:
        ledger_path = Path(temporary) / "checksums.json"
        ledger_path.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
        with zipfile.ZipFile(temporary_output, "w", zipfile.ZIP_DEFLATED) as archive:
            for name, source in sources.items():
                archive.write(source, arcname=name)
            archive.write(ledger_path, arcname="checksums.json")
    # Producer verification; consumers must obtain this digest independently.
    bundle_sha256 = _sha256(temporary_output)
    verify_resume_bundle(temporary_output, expected_sha256=bundle_sha256)
    temporary_output.replace(output)
    print(f"Record externally as EXPECTED_RESUME_BUNDLE_SHA256: {bundle_sha256}")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a verified non-final smoke resume bundle")
    parser.add_argument("--last-checkpoint", type=Path, required=True)
    parser.add_argument("--resolved-config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--protocol-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = package_resume_bundle(
        args.last_checkpoint,
        args.resolved_config,
        args.manifest,
        args.protocol_config,
        args.output,
    )
    print(result)


if __name__ == "__main__":
    main()
