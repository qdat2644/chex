from __future__ import annotations

import os
import json
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from app.config import DEFAULT_CHECKPOINT_PATH, DEFAULT_THRESHOLDS_PATH, MODEL_INFO, PROJECT_ROOT
from app.model import CheXpertPredictor
from app.schemas import FindingPrediction, HeatmapExplanation, PredictionResponse


app = FastAPI(title="CheXpert Web Reader")


def resolve_checkpoint_path() -> Path | None:
    configured = os.getenv("CHEXPERT_CHECKPOINT")
    if configured:
        return Path(configured)
    if DEFAULT_CHECKPOINT_PATH.exists():
        return DEFAULT_CHECKPOINT_PATH
    return None


def load_threshold_payload(path: Path = DEFAULT_THRESHOLDS_PATH) -> dict[str, object] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_thresholds() -> dict[str, float]:
    payload = load_threshold_payload()
    thresholds = payload.get("thresholds", {}) if payload else {}
    if not isinstance(thresholds, dict):
        return {}
    return {str(label): float(value) for label, value in thresholds.items()}


checkpoint_path = resolve_checkpoint_path()
threshold_payload = load_threshold_payload()
thresholds = load_thresholds()
predictor = CheXpertPredictor(checkpoint_path, thresholds=thresholds)

static_dir = PROJECT_ROOT / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(static_dir / "index.html")


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "model_loaded": predictor.is_loaded,
        "labels": predictor.labels,
        "checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "thresholds_loaded": bool(predictor.thresholds),
        "model_info": MODEL_INFO,
    }


@app.get("/api/model-info")
def model_info() -> dict[str, object]:
    return {
        "model_loaded": predictor.is_loaded,
        "labels": predictor.labels,
        "checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "thresholds_loaded": bool(predictor.thresholds),
        "thresholds": predictor.thresholds,
        "threshold_report": threshold_payload,
        "model_info": MODEL_INFO,
        "disclaimer": "Research prototype only. Do not use these results for medical decisions.",
    }


@app.post("/api/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    include_heatmap: bool = Query(True),
) -> PredictionResponse:
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        image = Image.open(BytesIO(content))
        image.load()
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Upload a valid image file.") from exc

    image = image.convert("RGB")
    width, height = image.size

    if not predictor.is_loaded:
        return PredictionResponse(
            status="model_not_loaded",
            filename=file.filename or "upload",
            width=width,
            height=height,
            mode=image.mode,
            findings=[],
            message="Set CHEXPERT_CHECKPOINT to a trained checkpoint before clinical-style inference.",
        )

    predictions = predictor.predict(image)
    findings = [
        FindingPrediction(
            label=item.label,
            probability=item.probability,
            positive=item.positive,
            threshold=item.threshold,
        )
        for item in predictions
    ]
    heatmap = None
    if include_heatmap:
        explanation = predictor.explain_top_finding(image)
        heatmap = HeatmapExplanation(
            label=explanation.label,
            probability=explanation.probability,
            image_data_url=explanation.image_data_url,
        )

    return PredictionResponse(
        status="ok",
        filename=file.filename or "upload",
        width=width,
        height=height,
        mode=image.mode,
        findings=findings,
        report=build_report(findings),
        heatmap=heatmap,
    )


def build_report(findings: list[FindingPrediction]) -> str:
    if not findings:
        return "No model findings were generated."

    sorted_findings = sorted(findings, key=lambda item: item.probability, reverse=True)
    if not any(item.threshold is not None for item in sorted_findings):
        top = sorted_findings[0]
        return f"Threshold report is not loaded. Highest probability: {top.label} ({top.probability:.0%})."
    positives = [item for item in sorted_findings if item.positive]
    top = sorted_findings[0]
    if positives:
        labels = ", ".join(f"{item.label} ({item.probability:.0%})" for item in positives[:3])
        return f"Model flagged: {labels}. Highest probability: {top.label} ({top.probability:.0%})."
    return f"No findings crossed the threshold. Highest probability: {top.label} ({top.probability:.0%})."
