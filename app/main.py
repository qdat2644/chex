from __future__ import annotations

import json
import os
from io import BytesIO
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

try:
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_voi_lut
    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False

from app.config import (
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_THRESHOLDS_PATH,
    HUGGINGFACE_CHECKPOINT_URL,
    HUGGINGFACE_REPO,
    HUGGINGFACE_THRESHOLDS_URL,
    MODEL_INFO,
    PROJECT_ROOT,
)
from app.model import CheXpertPredictor
from app.schemas import DicomMetadata, FindingPrediction, HeatmapExplanation, PredictionResponse, QualityReport


app = FastAPI(title="CheXpert Web Reader - Medical AI Workstation")


def _download_hf_file(url: str, target: Path) -> bool:
    try:
        import urllib.request
        target.parent.mkdir(parents=True, exist_ok=True)
        headers = {"User-Agent": "Mozilla/5.0"}
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            headers["Authorization"] = f"Bearer {hf_token}"
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as resp, open(target, "wb") as f:
            while chunk := resp.read(1024 * 1024):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"Warning: Could not download {url}: {e}")
        return False


def resolve_checkpoint_path() -> Path | None:
    configured = os.getenv("CHEXPERT_CHECKPOINT")
    if configured:
        return Path(configured)
    ckpt_dir = PROJECT_ROOT / "checkpoints"
    for candidate in ["chexpert_convnext_small.pt", "chexpert_densenet121_v2.pt", "chexpert_model.pt"]:
        p = ckpt_dir / candidate
        if p.exists() and p.stat().st_size > 1000:
            return p
    if DEFAULT_CHECKPOINT_PATH.exists() and DEFAULT_CHECKPOINT_PATH.stat().st_size > 1000:
        return DEFAULT_CHECKPOINT_PATH

    # If missing locally, auto-download from Hugging Face
    target = ckpt_dir / "chexpert_convnext_small.pt"
    print(f"Downloading model checkpoint from Hugging Face ({HUGGINGFACE_CHECKPOINT_URL})...")
    if _download_hf_file(HUGGINGFACE_CHECKPOINT_URL, target):
        print(f"Downloaded model to {target} (Size: {target.stat().st_size} bytes)")
        return target
    return None


def load_threshold_payload(path: Path = DEFAULT_THRESHOLDS_PATH) -> dict[str, object] | None:
    if not path.exists():
        _download_hf_file(HUGGINGFACE_THRESHOLDS_URL, path)
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except Exception:
        return None


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


def decode_image_or_dicom(content: bytes, filename: str) -> tuple[Image.Image, DicomMetadata]:
    """
    Decodes standard image files (PNG, JPG, WebP) or DICOM (.dcm) files.
    """
    # 1. Check if it is a DICOM file (starts with DICM or .dcm extension)
    is_dicom = filename.lower().endswith(".dcm") or (len(content) > 132 and content[128:132] == b"DICM")
    
    if is_dicom and HAS_PYDICOM:
        try:
            dcm = pydicom.dcmread(BytesIO(content), force=True)
            arr = dcm.pixel_array.astype(float)
            
            # Apply VOI LUT (Windowing / Rescaling) if available
            try:
                arr = apply_voi_lut(arr, dcm)
            except Exception:
                slope = float(getattr(dcm, "RescaleSlope", 1.0))
                intercept = float(getattr(dcm, "RescaleIntercept", 0.0))
                arr = (arr * slope) + intercept

            # Invert MONOCHROME1 (where 0 is white and max is black) to MONOCHROME2
            photometric = str(getattr(dcm, "PhotometricInterpretation", "MONOCHROME2")).upper()
            if "MONOCHROME1" in photometric:
                arr = np.amax(arr) - arr

            # Normalize to 0-255 uint8
            arr_min = float(np.min(arr))
            arr_max = float(np.max(arr))
            if arr_max > arr_min:
                arr = ((arr - arr_min) / (arr_max - arr_min) * 255.0).astype(np.uint8)
            else:
                arr = np.zeros(arr.shape, dtype=np.uint8)

            image = Image.fromarray(arr).convert("RGB")
            
            dicom_meta = DicomMetadata(
                is_dicom=True,
                patient_id=str(getattr(dcm, "PatientID", "ANONYMIZED")),
                patient_name=str(getattr(dcm, "PatientName", "ANONYMIZED")),
                study_date=str(getattr(dcm, "StudyDate", "N/A")),
                modality=str(getattr(dcm, "Modality", "CR / DX")),
                view_position=str(getattr(dcm, "ViewPosition", "PA (Posteroanterior)")),
                body_part=str(getattr(dcm, "BodyPartExamined", "CHEST")),
                photometric=photometric,
                rows=int(getattr(dcm, "Rows", image.height)),
                columns=int(getattr(dcm, "Columns", image.width)),
            )
            return image, dicom_meta
        except Exception as e:
            # Fallback to standard Image reader if DICOM parse fails
            pass

    # 2. Standard image reader (JPG, PNG, WebP, BMP)
    try:
        image = Image.open(BytesIO(content))
        image.load()
        image = image.convert("RGB")
        return image, DicomMetadata(is_dicom=False)
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Unsupported or invalid medical image format.") from exc


def assess_cxr_quality(image: Image.Image, dicom_meta: DicomMetadata) -> QualityReport:
    """
    Performs clinical validation checks to ensure the image is a valid Frontal Chest X-ray.
    """
    warnings = []
    quality_score = 1.0

    # DICOM check
    if dicom_meta.is_dicom:
        if dicom_meta.body_part and "CHEST" not in dicom_meta.body_part.upper():
            warnings.append(f"DICOM BodyPart is '{dicom_meta.body_part}', not CHEST.")
            quality_score -= 0.3
        if dicom_meta.view_position and "LATERAL" in dicom_meta.view_position.upper():
            warnings.append("DICOM View is LATERAL. This model is trained on FRONTAL (PA/AP) views only.")
            quality_score -= 0.4

    # Aspect ratio check (Frontal CXR typically 1:1 to 4:5)
    w, h = image.size
    ratio = w / max(1, h)
    if ratio < 0.5 or ratio > 2.0:
        warnings.append(f"Unusual aspect ratio ({ratio:.2f}). Please ensure image is cropped to the thorax.")
        quality_score -= 0.2

    # Color saturation check (X-rays are monochrome/grayscale)
    np_img = np.asarray(image, dtype=np.float32)
    std_channels = np.std(np_img, axis=-1)
    mean_std = float(np.mean(std_channels))
    if mean_std > 18.0:
        warnings.append("Image has noticeable color tint/saturation. Ensure this is a raw chest radiograph.")
        quality_score -= 0.25

    is_likely_frontal = quality_score >= 0.5
    return QualityReport(
        is_likely_frontal_cxr=is_likely_frontal,
        quality_score=float(np.clip(quality_score, 0.0, 1.0)),
        suggested_view="Frontal (PA/AP)" if is_likely_frontal else "Non-Frontal / Advisory Warning",
        warnings=warnings,
    )


def build_report(findings: list[FindingPrediction], quality: QualityReport) -> str:
    positive_findings = [f for f in findings if f.positive]
    high_suspicion = [f.label for f in findings if f.suspicion_level == "High suspicion"]
    moderate_suspicion = [f.label for f in findings if f.suspicion_level == "Moderate suspicion"]

    if not positive_findings:
        text = "No pathological findings exceeded calibrated thresholds. Frontal lung fields appear clear of acute radiopaque consolidation, pulmonary edema, or overt pleural effusion."
    else:
        parts = []
        if high_suspicion:
            parts.append(f"High suspicion of {', '.join(high_suspicion)}")
        if moderate_suspicion:
            parts.append(f"moderate suspicion of {', '.join(moderate_suspicion)}")
        findings_summary = " and ".join(parts)
        text = f"Automated analysis indicates {findings_summary}. Clinical correlation and specialist radiologist overread are recommended."

    if quality.warnings:
        text += f" (Note: {'; '.join(quality.warnings)})"

    return text


def get_current_model_info() -> dict[str, object]:
    info = dict(MODEL_INFO)
    if predictor.is_loaded:
        info["architecture"] = predictor.architecture
        if threshold_payload and "mean_auc" in threshold_payload:
            auc = threshold_payload["mean_auc"]
            info["mean_auc"] = auc
            info["mean_auc_display"] = f"{auc:.4f}"
            info["checkpoint"] = str(checkpoint_path) if checkpoint_path else info["checkpoint"]
    return info


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
        "model_info": get_current_model_info(),
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
        "model_info": get_current_model_info(),
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

    filename = file.filename or "uploaded_xray.png"
    image, dicom_meta = decode_image_or_dicom(content, filename)
    width, height = image.size

    if not predictor.is_loaded:
        return PredictionResponse(
            status="model_not_loaded",
            filename=filename,
            width=width,
            height=height,
            mode=image.mode,
            findings=[],
            quality=QualityReport(),
            dicom=dicom_meta,
            message="No model checkpoint loaded. Place a .pt file in checkpoints/.",
        )

    quality = assess_cxr_quality(image, dicom_meta)
    predictions = predictor.predict(image)
    findings = [
        FindingPrediction(
            label=item.label,
            probability=item.probability,
            positive=item.positive,
            threshold=item.threshold,
            suspicion_level=item.suspicion_level,
        )
        for item in predictions
    ]

    heatmap = None
    if include_heatmap:
        explanation = predictor.explain_finding(image, target_label=None)
        heatmap = HeatmapExplanation(
            label=explanation.label,
            probability=explanation.probability,
            image_data_url=explanation.image_data_url,
            pure_heatmap_url=explanation.pure_heatmap_url,
        )

    return PredictionResponse(
        status="ok",
        filename=filename,
        width=width,
        height=height,
        mode=image.mode,
        findings=findings,
        report=build_report(findings, quality),
        heatmap=heatmap,
        quality=quality,
        dicom=dicom_meta,
    )


@app.post("/api/explain", response_model=HeatmapExplanation)
async def explain_label(
    file: UploadFile = File(...),
    label: str = Form(...),
) -> HeatmapExplanation:
    """
    On-demand interactive Grad-CAM generation for ANY specific finding label.
    """
    if not predictor.is_loaded:
        raise HTTPException(status_code=400, detail="Model is not loaded.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty image data.")

    filename = file.filename or "upload.png"
    image, _ = decode_image_or_dicom(content, filename)

    try:
        explanation = predictor.explain_finding(image, target_label=label)
        return HeatmapExplanation(
            label=explanation.label,
            probability=explanation.probability,
            image_data_url=explanation.image_data_url,
            pure_heatmap_url=explanation.pure_heatmap_url,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Grad-CAM generation failed: {e}")
