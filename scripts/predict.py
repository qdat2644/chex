from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.model import CheXpertPredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CheXpert model inference on one X-ray image.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictor = CheXpertPredictor(args.checkpoint, threshold=args.threshold)
    image = Image.open(args.image).convert("RGB")
    predictions = predictor.predict(image)

    print(
        json.dumps(
            {
                "image": str(args.image),
                "checkpoint": str(args.checkpoint),
                "findings": [
                    {
                        "label": item.label,
                        "probability": item.probability,
                        "positive": item.positive,
                    }
                    for item in predictions
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
