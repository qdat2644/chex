from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import os
import tempfile
import uuid
from io import BytesIO
from pathlib import Path
from typing import BinaryIO

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

# Decompression bomb protection against oversized images
Image.MAX_IMAGE_PIXELS = 50_000_000

# Security and ingestion limits
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB max per file
MAX_BATCH_FILES = 50                    # 50 files max per batch request
MAX_DICOM_PIXELS = 50_000_000           # 50 megapixels max per DICOM frame

# PHI Protection Configuration
PHI_HMAC_SECRET = os.getenv("PHI_HMAC_SECRET", "chexpert_default_medical_phi_secret_2026").encode("utf-8")
EXPOSE_PHI = os.getenv("EXPOSE_PHI", "false").lower() == "true"

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
from app.schemas import (
    BatchItemResult,
    BatchPredictionResponse,
    DicomMetadata,
    FindingPrediction,
    HeatmapExplanation,
    PredictionResponse,
    QualityReport,
)


app = FastAPI(title="CheXpert Web Reader - Medical AI Workstation")


def mask_phi(val: str | None) -> str:
    """
    Masks Protected Health Information (PHI) using HMAC-SHA256 with 16-character fingerprint.
    """
    if not val or str(val).strip().upper() in ("ANONYMIZED", "NONE", "N/A", "UNKNOWN"):
        return "ANONYMIZED"
    if EXPOSE_PHI:
        return str(val).strip()
    raw_bytes = str(val).strip().encode("utf-8")
    h = hmac.new(PHI_HMAC_SECRET, raw_bytes, hashlib.sha256).hexdigest()[:16].upper()
    return f"ANONYMIZED_{h}"


def _download_hf_file(url: str, target: Path, expected_sha256: str | None = None) -> bool:
    """
    Downloads file from Hugging Face via temporary file, verifies integrity, and performs atomic rename.
    """
    tmp_target = target.with_suffix(target.suffix + f".{uuid.uuid4().hex[:8]}.tmp")
    try:
        import urllib.request
        target.parent.mkdir(parents=True, exist_ok=True)
        headers = {"User-Agent": "Mozilla/5.0"}
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            headers["Authorization"] = f"Bearer {hf_token}"
        req = urllib.request.Request(url, headers=headers)
        
        hasher = hashlib.sha256()
        with urllib.request.urlopen(req) as resp, open(tmp_target, "wb") as f:
            while chunk := resp.read(1024 * 1024):
                f.write(chunk)
                hasher.update(chunk)

        if not tmp_target.exists() or tmp_target.stat().st_size < 1000:
            if tmp_target.exists():
                tmp_target.unlink(missing_ok=True)
            return False

        if expected_sha256:
            calc_sha256 = hasher.hexdigest()
            if calc_sha256.lower() != expected_sha256.lower():
                print(f"Checksum mismatch for {target}: expected {expected_sha256}, got {calc_sha256}")
                tmp_target.unlink(missing_ok=True)
                return False

        os.replace(tmp_target, target)
        return True
    except Exception as e:
        if tmp_target.exists():
            tmp_target.unlink(missing_ok=True)
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


async def stream_to_spooled_tempfile(file: UploadFile, max_bytes: int = MAX_FILE_SIZE_BYTES) -> tempfile.SpooledTemporaryFile:
    """
    Streams upload file in 1MB chunks into a SpooledTemporaryFile with early size-limit abort.
    """
    spool = tempfile.SpooledTemporaryFile(max_size=2 * 1024 * 1024)
    total_read = 0
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        total_read += len(chunk)
        if total_read > max_bytes:
            spool.close()
            raise HTTPException(
                status_code=413,
                detail=f"File exceeds maximum allowed size of {max_bytes // (1024 * 1024)}MB.",
            )
        spool.write(chunk)

    if total_read == 0:
        spool.close()
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    spool.seek(0)
    return spool


def decode_image_or_dicom(spool: BinaryIO, filename: str) -> tuple[Image.Image, DicomMetadata]:
    """
    Decodes standard image files or DICOM files with streaming, frame validation, and PHI masking.
    """
    header_bytes = spool.read(132)
    spool.seek(0)

    is_dicom = filename.lower().endswith(".dcm") or (len(header_bytes) >= 132 and header_bytes[128:132] == b"DICM")

    if is_dicom and HAS_PYDICOM:
        try:
            dcm = pydicom.dcmread(spool, force=True)
            
            # Frame and dimension guards against massive memory allocations
            num_frames = int(getattr(dcm, "NumberOfFrames", 1))
            rows = int(getattr(dcm, "Rows", 0))
            cols = int(getattr(dcm, "Columns", 0))
            if rows * cols > MAX_DICOM_PIXELS:
                raise HTTPException(status_code=400, detail="DICOM resolution exceeds maximum supported dimensions.")

            arr = dcm.pixel_array
            if num_frames > 1 and len(arr.shape) > 2:
                arr = arr[0]  # Take first frame for multi-frame studies

            arr = arr.astype(float)

            # Apply VOI LUT (Windowing / Rescaling) if available
            try:
                arr = apply_voi_lut(arr, dcm)
            except Exception:
                slope = float(getattr(dcm, "RescaleSlope", 1.0))
                intercept = float(getattr(dcm, "RescaleIntercept", 0.0))
                arr = (arr * slope) + intercept

            # Invert MONOCHROME1 to MONOCHROME2
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

            raw_patient_id = getattr(dcm, "PatientID", None)
            raw_patient_name = getattr(dcm, "PatientName", None)
            raw_study_date = getattr(dcm, "StudyDate", "N/A")

            dicom_meta = DicomMetadata(
                is_dicom=True,
                patient_id=mask_phi(raw_patient_id),
                patient_name=mask_phi(raw_patient_name) if EXPOSE_PHI else "REDACTED",
                study_date=str(raw_study_date) if EXPOSE_PHI else "REDACTED",
                modality=str(getattr(dcm, "Modality", "CR / DX")),
                view_position=str(getattr(dcm, "ViewPosition", "PA (Posteroanterior)")),
                body_part=str(getattr(dcm, "BodyPartExamined", "CHEST")),
                photometric=photometric,
                rows=rows or image.height,
                columns=cols or image.width,
            )
            return image, dicom_meta
        except HTTPException:
            raise
        except Exception:
            spool.seek(0)

    # Standard image reader
    try:
        image = Image.open(spool)
        image.load()
        image = image.convert("RGB")
        return image, DicomMetadata(is_dicom=False)
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Unsupported or invalid medical image format.") from exc


def assess_cxr_quality(image: Image.Image, dicom_meta: DicomMetadata) -> QualityReport:
    warnings = []
    quality_score = 1.0

    if dicom_meta.is_dicom:
        if dicom_meta.body_part and "CHEST" not in dicom_meta.body_part.upper():
            warnings.append(f"DICOM BodyPart is '{dicom_meta.body_part}', not CHEST.")
            quality_score -= 0.3
        if dicom_meta.view_position and "LATERAL" in dicom_meta.view_position.upper():
            warnings.append("DICOM View is LATERAL. This model is trained on FRONTAL (PA/AP) views only.")
            quality_score -= 0.4

    w, h = image.size
    ratio = w / max(1, h)
    if ratio < 0.5 or ratio > 2.0:
        warnings.append(f"Unusual aspect ratio ({ratio:.2f}). Please ensure image is cropped to the thorax.")
        quality_score -= 0.2

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
        text = "No pathological findings exceeded calibrated thresholds among the 5 evaluated target labels (Atelectasis, Cardiomegaly, Consolidation, Edema, Pleural Effusion)."
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
    filename = file.filename or "uploaded_xray.png"
    spool = await stream_to_spooled_tempfile(file, max_bytes=MAX_FILE_SIZE_BYTES)
    try:
        image, dicom_meta = decode_image_or_dicom(spool, filename)
    finally:
        spool.close()

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

    # Offload inference and heatmap to worker thread with unified lock
    def _do_inference():
        quality = assess_cxr_quality(image, dicom_meta)
        predictions = predictor.predict(image)
        heatmap = None
        if include_heatmap:
            explanation = predictor.explain_finding(image, target_label=None)
            heatmap = HeatmapExplanation(
                label=explanation.label,
                probability=explanation.probability,
                image_data_url=explanation.image_data_url,
                pure_heatmap_url=explanation.pure_heatmap_url,
            )
        return quality, predictions, heatmap

    quality, predictions, heatmap = await asyncio.to_thread(_do_inference)

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
    if not predictor.is_loaded:
        raise HTTPException(status_code=400, detail="Model is not loaded.")

    if label not in predictor.labels:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid label '{label}'. Valid labels are: {predictor.labels}",
        )

    filename = file.filename or "upload.png"
    spool = await stream_to_spooled_tempfile(file, max_bytes=MAX_FILE_SIZE_BYTES)
    try:
        image, _ = decode_image_or_dicom(spool, filename)
    finally:
        spool.close()

    try:
        explanation = await asyncio.to_thread(predictor.explain_finding, image, label)
        return HeatmapExplanation(
            label=explanation.label,
            probability=explanation.probability,
            image_data_url=explanation.image_data_url,
            pure_heatmap_url=explanation.pure_heatmap_url,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Grad-CAM generation failed: {e}")


@app.post("/api/predict-batch", response_model=BatchPredictionResponse)
async def predict_batch_cxrs(
    files: list[UploadFile] = File(...),
    client_ids: list[str] = Form(default=[]),
) -> BatchPredictionResponse:
    if not predictor.is_loaded:
        raise HTTPException(status_code=400, detail="Model is not loaded.")
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    if len(files) > MAX_BATCH_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size exceeds maximum limit of {MAX_BATCH_FILES} files per request.",
        )

    valid_images: list[Image.Image] = []
    metadata_list: list[dict] = []
    errors: list[str] = []

    client_id_list: list[str] = []
    if client_ids:
        for item in client_ids:
            if "," in item:
                client_id_list.extend([x.strip() for x in item.split(",") if x.strip()])
            elif item.strip():
                client_id_list.append(item.strip())

    # Stream, decode, and generate previews in chunks of 8 to minimize RAM footprint
    for idx, f in enumerate(files):
        c_id = client_id_list[idx] if idx < len(client_id_list) else f"batch_cxr_{idx+1}_{uuid.uuid4().hex[:8]}"
        fname = f.filename or f"image_{idx}.png"

        try:
            spool = await stream_to_spooled_tempfile(f, max_bytes=MAX_FILE_SIZE_BYTES)
            try:
                img, dcm_meta = decode_image_or_dicom(spool, fname)
            finally:
                spool.close()

            quality = assess_cxr_quality(img, dcm_meta)

            # Generate small thumbnail preview for batch table
            thumb = img.copy()
            thumb.thumbnail((120, 120))
            thumb_buf = BytesIO()
            thumb.save(thumb_buf, format="PNG")
            thumb_b64 = f"data:image/png;base64,{base64.b64encode(thumb_buf.getvalue()).decode('ascii')}"

            valid_images.append(img)
            metadata_list.append({
                "index": idx + 1,
                "client_id": c_id,
                "filename": fname,
                "is_dicom": dcm_meta.is_dicom,
                "patient_id": dcm_meta.patient_id or "ANONYMIZED",
                "quality": quality,
                "preview_url": thumb_b64,
            })
        except HTTPException as he:
            errors.append(f"File {fname}: {he.detail}")
        except Exception as e:
            errors.append(f"File {fname}: {e}")

    if not valid_images:
        return BatchPredictionResponse(
            status="error",
            total=len(files),
            processed=0,
            failed=len(errors),
            results=[],
            errors=errors,
        )

    # Chunked batch forward pass on threadpool
    batch_predictions = await asyncio.to_thread(predictor.predict_batch, valid_images, chunk_size=8)

    results: list[BatchItemResult] = []
    for meta, preds in zip(metadata_list, batch_predictions, strict=True):
        findings = [
            FindingPrediction(
                label=p.label,
                probability=p.probability,
                positive=p.positive,
                threshold=p.threshold,
                suspicion_level=p.suspicion_level,
            )
            for p in preds
        ]

        sorted_findings = sorted(findings, key=lambda x: x.probability, reverse=True)
        top_f = sorted_findings[0] if sorted_findings else None
        pos_labels = [f.label for f in findings if f.positive]

        results.append(
            BatchItemResult(
                index=meta["index"],
                client_id=meta["client_id"],
                filename=meta["filename"],
                is_dicom=meta["is_dicom"],
                patient_id=meta["patient_id"],
                findings=findings,
                top_finding=top_f.label if top_f else "None",
                top_probability=top_f.probability if top_f else 0.0,
                positive_count=len(pos_labels),
                positive_labels=pos_labels,
                report=build_report(findings, meta["quality"]),
                preview_url=meta["preview_url"],
            )
        )

    return BatchPredictionResponse(
        status="ok",
        total=len(files),
        processed=len(results),
        failed=len(errors),
        results=results,
        errors=errors,
    )
