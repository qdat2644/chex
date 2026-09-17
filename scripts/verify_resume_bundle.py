from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import tempfile
import zipfile
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.experiment_integrity import canonical_json_sha256


EXPECTED_MEMBERS = {
    "last.pt",
    "resolved_config.json",
    "manifest.json",
    "protocol_v0_1.yaml",
    "checksums.json",
}


def verify_resume_bundle(path: Path, extract_dir: Path | None = None) -> dict[str, object]:
    bundle = Path(path)
    if not bundle.is_file():
        raise FileNotFoundError(bundle)
    try:
        with zipfile.ZipFile(bundle, "r") as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise RuntimeError("Resume bundle contains duplicate ZIP members")
            if set(names) != EXPECTED_MEMBERS:
                raise RuntimeError(
                    f"Resume bundle members mismatch: expected {sorted(EXPECTED_MEMBERS)}, got {sorted(names)}"
                )
            for name in names:
                member = Path(name)
                if member.is_absolute() or ".." in member.parts or len(member.parts) != 1:
                    raise RuntimeError(f"Unsafe resume bundle member: {name}")

            ledger = json.loads(archive.read("checksums.json").decode("utf-8"))
            if ledger.get("schema_version") != 1:
                raise RuntimeError("Unsupported resume checksums schema")
            files = ledger.get("files")
            if not isinstance(files, dict) or set(files) != EXPECTED_MEMBERS - {"checksums.json"}:
                raise RuntimeError("Resume checksums ledger has an invalid file set")
            for name, expected in files.items():
                actual = hashlib.sha256(archive.read(name)).hexdigest()
                if actual != expected:
                    raise RuntimeError(f"Resume bundle checksum mismatch for {name}")

            checkpoint = torch.load(io.BytesIO(archive.read("last.pt")), map_location="cpu", weights_only=True)
            if not isinstance(checkpoint, dict):
                raise RuntimeError("Resume bundle last.pt is not a checkpoint dictionary")
            resolved = json.loads(archive.read("resolved_config.json").decode("utf-8"))
            nested = checkpoint.get("metadata", {})
            expected_links = {
                "resolved_config_sha256": canonical_json_sha256(resolved),
                "manifest_sha256": files["manifest.json"],
                "config_sha256": files["protocol_v0_1.yaml"],
            }
            for field, actual in expected_links.items():
                expected = checkpoint.get(field) or nested.get(
                    "split_manifest_sha256" if field == "manifest_sha256" else field
                )
                if expected != actual:
                    raise RuntimeError(f"Resume bundle {field} does not match last.pt metadata")

            if extract_dir is not None:
                destination = Path(extract_dir)
                destination.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
                    stage = Path(temporary)
                    for name in EXPECTED_MEMBERS:
                        (stage / name).write_bytes(archive.read(name))
                    for name in EXPECTED_MEMBERS:
                        target = destination / name
                        if target.exists():
                            target.unlink()
                        shutil.move(str(stage / name), str(target))
            return ledger
    except zipfile.BadZipFile as exc:
        raise RuntimeError(f"Invalid resume bundle ZIP: {bundle}") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify and optionally extract a resume bundle")
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--extract-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ledger = verify_resume_bundle(args.bundle, args.extract_dir)
    print(json.dumps(ledger, indent=2))


if __name__ == "__main__":
    main()
