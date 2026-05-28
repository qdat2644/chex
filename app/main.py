from __future__ import annotations

import os
from io import BytesIO

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from app.config import PROJECT_ROOT
from app.model import CheXpertPredictor
from app.schemas import FindingPrediction, HeatmapExplanation, PredictionResponse


app = FastAPI(title="CheXpert Web Reader")
predictor = CheXpertPredictor(os.getenv("CHEXPERT_CHECKPOINT"))

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
    positives = [item for item in sorted_findings if item.positive]
    top = sorted_findings[0]
    if positives:
        labels = ", ".join(f"{item.label} ({item.probability:.0%})" for item in positives[:3])
        return f"Model flagged: {labels}. Highest probability: {top.label} ({top.probability:.0%})."
    return f"No findings crossed the threshold. Highest probability: {top.label} ({top.probability:.0%})."
