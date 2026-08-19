from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def compute_file_sha256(filepath: Path) -> str:
    if not filepath.exists():
        return "NOT_FOUND"
    h = hashlib.sha256()
    with filepath.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Convert legacy/trusted checkpoint into safe, weights_only=True compatible format.")
    parser.add_argument("--input", type=Path, required=True, help="Input checkpoint .pt")
    parser.add_argument("--output", type=Path, required=True, help="Output safe checkpoint .pt")
    parser.add_argument("--acknowledge-trusted-source", action="store_true", required=True, help="Mandatory safety flag acknowledging trusted checkpoint source")
    args = parser.parse_args()

    if not args.acknowledge_trusted_source:
        print("Error: You must provide --acknowledge-trusted-source to convert this checkpoint.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading checkpoint from: {args.input}")
    raw_data = torch.load(args.input, map_location="cpu", weights_only=False)

    if isinstance(raw_data, dict):
        if "model_state_dict" in raw_data:
            state_dict = raw_data["model_state_dict"]
        elif "state_dict" in raw_data:
            state_dict = raw_data["state_dict"]
        else:
            state_dict = {k: v for k, v in raw_data.items() if isinstance(v, torch.Tensor)}
        metadata = raw_data.get("metadata", {})
        labels = raw_data.get("labels", [])
    else:
        state_dict = raw_data

    # Sanitize state_dict to ensure only pure Tensors
    clean_state_dict = {str(k): v.clone().detach() for k, v in state_dict.items() if isinstance(v, torch.Tensor)}

    clean_payload = {
        "model_state_dict": clean_state_dict,
        "labels": labels,
        "metadata": {
            "source_sha256": compute_file_sha256(args.input),
            "sanitized_safe": True,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(clean_payload, args.output)
    print(f"Safe checkpoint successfully written to: {args.output}")
    print(f"SHA-256: {compute_file_sha256(args.output)}")


if __name__ == "__main__":
    main()
