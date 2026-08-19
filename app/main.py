from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import os
import tempfile
import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import BinaryIO

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

# Decompression bomb protection against oversized images
Image.MAX_IMAGE_PIXELS = 50_000_000

# Security and ingestion limits
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB max per file
MAX_BATCH_FILES = 50                    # 50 files max per batch request
MAX_DICOM_PIXELS = 50_000_000           # 50 megapixels max per DICOM frame

# Audit Logging Setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("chexpert.audit")

try:
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_voi_lut
    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False

from app.config import (
    API_KEY,
    AUTO_DOWNLOAD_ENABLED,
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_THRESHOLDS_PATH,
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_THRESHOLDS_SHA256,
    HUGGINGFACE_CHECKPOINT_URL,
    HUGGINGFACE_REPO,
    HUGGINGFACE_REVISION,
    HUGGINGFACE_THRESHOLDS_URL,
    MANIFEST_PATH,
    MODEL_INFO,
    PROJECT_ROOT,
    RATE_LIMIT_PER_MINUTE,
    get_phi_secret,
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

static_dir = PROJECT_ROOT / "static"

# In-memory sliding window rate limiter state
_RATE_LIMIT_STORE: dict[str, list[float]] = defaultdict(list)


def get_active_phi_secret() -> bytes:
    sec = get_phi_secret()
    if sec:
        return sec.encode("utf-8")
    app_env = os.getenv("APP_ENV", "development").lower()
    if app_env == "production":
        raise RuntimeError("CRITICAL SECURITY ERROR: PHI_HMAC_SECRET must be explicitly configured in production.")
    return b"dev_temporary_chexpert_phi_secret_key_2026"


def compute_file_sha256(filepath: Path) -> str:
    if not filepath.exists():
        return ""
    h = hashlib.sha256()
    with filepath.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def mask_phi(val: str | None) -> str:
    """
    Masks Protected Health Information (PHI) using HMAC-SHA256 with 16-character fingerprint.
    """
    expose_phi = os.getenv("EXPOSE_PHI", "false").lower() == "true"
    if not val or str(val).strip().upper() in ("ANONYMIZED", "NONE", "N/A", "UNKNOWN"):
        return "ANONYMIZED"
    if expose_phi:
        return str(val).strip()
    raw_bytes = str(val).strip().encode("utf-8")
    secret = get_active_phi_secret()
    h = hmac.new(secret, raw_bytes, hashlib.sha256).hexdigest()[:16].upper()
    return f"ANONYMIZED_{h}"


def anonymize_filename(filename: str) -> str:
    """
    Generates deterministic anonymized scan filename: scan_<8_hex_hash>.png
    """
    expose_phi = os.getenv("EXPOSE_PHI", "false").lower() == "true"
    if expose_phi or not filename:
        return filename or "scan.png"
    raw_bytes = filename.strip().encode("utf-8")
    secret = get_active_phi_secret()
    h = hmac.new(secret, raw_bytes, hashlib.sha256).hexdigest()[:8].lower()
    return f"scan_{h}.png"


def _download_hf_file(url: str, target: Path, expected_sha256: str | None = None) -> bool:
    """
    Downloads file from Hugging Face via temporary file, verifies integrity, and performs atomic rename.
    """
    if not AUTO_DOWNLOAD_ENABLED:
        return False

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
    app_env = os.getenv("APP_ENV", "development").lower()
    configured = os.getenv("CHEXPERT_CHECKPOINT")
    candidates = [Path(configured)] if configured else [
        PROJECT_ROOT / "checkpoints" / "chexpert_convnext_small.pt",
        DEFAULT_CHECKPOINT_PATH,
    ]

    for p in candidates:
        if p.exists() and p.is_file():
            if EXPECTED_CHECKPOINT_SHA256:
                actual_hash = compute_file_sha256(p)
                if actual_hash.lower() != EXPECTED_CHECKPOINT_SHA256.lower():
                    if app_env == "production":
                        raise RuntimeError(
                            f"Integrity check failed: Checkpoint {p} SHA-256 mismatch.\n"
                            f"Expected: {EXPECTED_CHECKPOINT_SHA256}\nActual:   {actual_hash}"
                        )
                    else:
                        print(f"Notice: Checkpoint {p} SHA-256 is {actual_hash} (production hash is {EXPECTED_CHECKPOINT_SHA256}).")
            return p

    # If missing locally and auto-download allowed, download with SHA-256 validation
    if AUTO_DOWNLOAD_ENABLED:
        target = PROJECT_ROOT / "checkpoints" / "chexpert_convnext_small.pt"
        print(f"Downloading model checkpoint from Hugging Face ({HUGGINGFACE_CHECKPOINT_URL})...")
        if _download_hf_file(HUGGINGFACE_CHECKPOINT_URL, target, expected_sha256=EXPECTED_CHECKPOINT_SHA256):
            print(f"Downloaded model to {target} (Size: {target.stat().st_size} bytes)")
            return target
    return None


def load_threshold_payload(path: Path = DEFAULT_THRESHOLDS_PATH) -> dict[str, object] | None:
    app_env = os.getenv("APP_ENV", "development").lower()
    if not path.exists() and AUTO_DOWNLOAD_ENABLED:
        _download_hf_file(HUGGINGFACE_THRESHOLDS_URL, path, expected_sha256=EXPECTED_THRESHOLDS_SHA256 or None)

    if path.exists():
        if EXPECTED_THRESHOLDS_SHA256:
            actual_hash = compute_file_sha256(path)
            if actual_hash.lower() != EXPECTED_THRESHOLDS_SHA256.lower():
                if app_env == "production":
                    raise RuntimeError(
                        f"Integrity check failed: Thresholds {path} SHA-256 mismatch.\n"
                        f"Expected: {EXPECTED_THRESHOLDS_SHA256}\nActual:   {actual_hash}"
                    )
                else:
                    print(f"Notice: Thresholds SHA-256 is {actual_hash}.")
        try:
            with path.open("r", encoding="utf-8") as file:
                return json.load(file)
        except Exception:
            return None
    return None


def load_thresholds(payload: dict[str, object] | None) -> dict[str, float]:
    thresholds = payload.get("thresholds", {}) if payload else {}
    if not isinstance(thresholds, dict):
        return {}
    return {str(label): float(value) for label, value in thresholds.items()}


@asynccontextmanager
async def lifespan(app_obj: FastAPI):
    app_env = os.getenv("APP_ENV", "development").lower()

    # Fail-closed secret verification in production (Task 1 & Task 2)
    if app_env == "production":
        sec = get_phi_secret()
        if not sec or len(sec) < 16:
            raise RuntimeError(
                "CRITICAL PRODUCTION SECURITY VIOLATION: PHI_HMAC_SECRET (min 16 chars) must be explicitly configured."
            )

    # Initialize model if not already injected by test harness
    if getattr(app_obj.state, "predictor", None) is None:
        ckpt_path = resolve_checkpoint_path()
        threshold_pl = load_threshold_payload()
        thresholds_dict = load_thresholds(threshold_pl)

        # Fail-closed production artifact verification (Task 2)
        if app_env == "production":
            if not ckpt_path or not ckpt_path.exists():
                raise RuntimeError("CRITICAL PRODUCTION ERROR: Missing valid model checkpoint.")
            if not threshold_pl:
                raise RuntimeError("CRITICAL PRODUCTION ERROR: Missing or invalid thresholds.json artifact.")

        pred = CheXpertPredictor(ckpt_path, thresholds=thresholds_dict)
        if app_env == "production" and not pred.is_loaded:
            raise RuntimeError("CRITICAL PRODUCTION ERROR: Model checkpoint failed to load.")

        app_obj.state.predictor = pred
        app_obj.state.threshold_payload = threshold_pl
        app_obj.state.checkpoint_path = ckpt_path
    yield


def create_app(
    predictor_instance: CheXpertPredictor | None = None,
    threshold_payload_instance: dict[str, object] | None = None,
    api_key_override: str | None = None,
    rate_limit_override: int | None = None,
) -> FastAPI:
    app_instance = FastAPI(title="CheXpert Web Reader - Medical AI Workstation", lifespan=lifespan)
    app_instance.state.predictor = predictor_instance
    app_instance.state.threshold_payload = threshold_payload_instance
    app_instance.state.checkpoint_path = getattr(predictor_instance, "checkpoint_path", None) if predictor_instance else None
    app_instance.state.api_key = api_key_override if api_key_override is not None else API_KEY
    app_instance.state.rate_limit = rate_limit_override if rate_limit_override is not None else RATE_LIMIT_PER_MINUTE

    # Middleware: Request ID, Authentication, Rate Limiting, Audit Logging (Task 5)
    @app_instance.middleware("http")
    async def security_and_audit_middleware(request: Request, call_next):
        req_id = uuid.uuid4().hex[:12]
        start_time = time.perf_counter()
        client_ip = request.client.host if request.client else "127.0.0.1"

        # Rate Limiting (Sliding Window 60s)
        limit = request.app.state.rate_limit
        if limit and limit > 0 and request.url.path.startswith("/api/"):
            now = time.time()
            timestamps = _RATE_LIMIT_STORE[client_ip]
            # prune timestamps older than 60s
            _RATE_LIMIT_STORE[client_ip] = [t for t in timestamps if now - t < 60.0]
            if len(_RATE_LIMIT_STORE[client_ip]) >= limit:
                logger.warning(f"[AUDIT] RATE_LIMIT_EXCEEDED req_id={req_id} ip={client_ip} path={request.url.path}")
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Too many requests. Rate limit exceeded."},
                    headers={"Retry-After": "60", "X-Request-ID": req_id},
                )
            _RATE_LIMIT_STORE[client_ip].append(now)

        # Authentication Check on /api/* (Excluding public /health)
        expected_key = request.app.state.api_key
        if expected_key and request.url.path.startswith("/api/"):
            auth_header = request.headers.get("Authorization", "")
            x_api_key = request.headers.get("X-API-Key", "")
            token = ""
            if auth_header.startswith("Bearer "):
                token = auth_header[7:].strip()
            elif x_api_key:
                token = x_api_key.strip()

            if not token or not hmac.compare_digest(token, expected_key):
                logger.warning(f"[AUDIT] AUTH_FAILED req_id={req_id} ip={client_ip} path={request.url.path}")
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Unauthorized: Invalid or missing API key."},
                    headers={"X-Request-ID": req_id},
                )

        response: Response = await call_next(request)
        duration_ms = (time.perf_counter() - start_time) * 1000.0

        response.headers["X-Request-ID"] = req_id
        logger.info(
            f"[AUDIT] req_id={req_id} ip={client_ip} method={request.method} "
            f"path={request.url.path} status={response.status_code} duration_ms={duration_ms:.1f}"
        )
        return response

    app_instance.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app_instance.get("/")
    def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    @app_instance.get("/health")
    def health(request: Request) -> dict[str, object]:
        pred = getattr(request.app.state, "predictor", None)
        ckpt_path = getattr(request.app.state, "checkpoint_path", None)
        return {
            "status": "ok",
            "model_loaded": pred.is_loaded if pred else False,
            "labels": pred.labels if pred else [],
            "checkpoint": str(ckpt_path) if ckpt_path else None,
            "thresholds_loaded": bool(pred.thresholds) if pred else False,
            "model_info": get_current_model_info(pred, getattr(request.app.state, "threshold_payload", None), ckpt_path),
        }

    @app_instance.get("/api/model-info")
    def model_info(request: Request) -> dict[str, object]:
        pred = getattr(request.app.state, "predictor", None)
        ckpt_path = getattr(request.app.state, "checkpoint_path", None)
        pl = getattr(request.app.state, "threshold_payload", None)
        return {
            "version": "1.0.0",
            "revision": HUGGINGFACE_REVISION,
            "model_loaded": pred.is_loaded if pred else False,
            "labels": pred.labels if pred else [],
            "checkpoint": str(ckpt_path) if ckpt_path else None,
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "thresholds_loaded": bool(pred.thresholds) if pred else False,
            "thresholds_sha256": EXPECTED_THRESHOLDS_SHA256,
            "thresholds": pred.thresholds if pred else {},
            "threshold_report": pl,
            "model_info": get_current_model_info(pred, pl, ckpt_path),
            "disclaimer": "Research prototype only. Do not use these results for medical decisions.",
        }

    @app_instance.post("/api/predict", response_model=PredictionResponse)
    async def predict(
        request: Request,
        file: UploadFile = File(...),
        include_heatmap: bool = Query(True),
    ) -> PredictionResponse:
        pred: CheXpertPredictor = request.app.state.predictor
        raw_filename = file.filename or "uploaded_xray.png"
        safe_filename = anonymize_filename(raw_filename)

        spool = await stream_to_spooled_tempfile(file, max_bytes=MAX_FILE_SIZE_BYTES)
        try:
            image, dicom_meta = decode_image_or_dicom(spool, raw_filename)
        finally:
            spool.close()

        width, height = image.size

        if pred is None or not pred.is_loaded:
            return PredictionResponse(
                status="model_not_loaded",
                filename=safe_filename,
                width=width,
                height=height,
                mode=image.mode,
                findings=[],
                quality=QualityReport(),
                dicom=dicom_meta,
                message="No model checkpoint loaded. Place a .pt file in checkpoints/.",
            )

        def _do_inference():
            quality = assess_cxr_quality(image, dicom_meta)
            predictions = pred.predict(image)
            heatmap = None
            if include_heatmap:
                explanation = pred.explain_finding(image, target_label=None)
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
            filename=safe_filename,
            width=width,
            height=height,
            mode=image.mode,
            findings=findings,
            report=build_report(findings, quality),
            heatmap=heatmap,
            quality=quality,
            dicom=dicom_meta,
        )

    @app_instance.post("/api/explain", response_model=HeatmapExplanation)
    async def explain_label(
        request: Request,
        file: UploadFile = File(...),
        label: str = Form(...),
    ) -> HeatmapExplanation:
        pred: CheXpertPredictor = request.app.state.predictor
        if pred is None or not pred.is_loaded:
            raise HTTPException(status_code=400, detail="Model is not loaded.")

        if label not in pred.labels:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid label '{label}'. Valid labels are: {pred.labels}",
            )

        raw_filename = file.filename or "upload.png"
        spool = await stream_to_spooled_tempfile(file, max_bytes=MAX_FILE_SIZE_BYTES)
        try:
            image, _ = decode_image_or_dicom(spool, raw_filename)
        finally:
            spool.close()

        try:
            explanation = await asyncio.to_thread(pred.explain_finding, image, label)
            return HeatmapExplanation(
                label=explanation.label,
                probability=explanation.probability,
                image_data_url=explanation.image_data_url,
                pure_heatmap_url=explanation.pure_heatmap_url,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Grad-CAM generation failed: {e}")

    @app_instance.post("/api/predict-batch", response_model=BatchPredictionResponse)
    async def predict_batch_cxrs(
        request: Request,
        files: list[UploadFile] = File(...),
        client_ids: list[str] = Form(default=[]),
    ) -> BatchPredictionResponse:
        pred: CheXpertPredictor = request.app.state.predictor
        if pred is None or not pred.is_loaded:
            raise HTTPException(status_code=400, detail="Model is not loaded.")
        if not files:
            raise HTTPException(status_code=400, detail="No files provided.")

        if len(files) > MAX_BATCH_FILES:
            raise HTTPException(
                status_code=400,
                detail=f"Batch size exceeds maximum limit of {MAX_BATCH_FILES} files per request.",
            )

        client_id_list: list[str] = []
        if client_ids:
            for item in client_ids:
                if "," in item:
                    client_id_list.extend([x.strip() for x in item.split(",") if x.strip()])
                elif item.strip():
                    client_id_list.append(item.strip())

        results: list[BatchItemResult] = []
        errors: list[str] = []
        CHUNK_SIZE = 8

        for i in range(0, len(files), CHUNK_SIZE):
            chunk_files = files[i : i + CHUNK_SIZE]
            chunk_cids = client_id_list[i : i + CHUNK_SIZE]

            chunk_images: list[Image.Image] = []
            chunk_meta: list[dict] = []

            for j, f in enumerate(chunk_files):
                c_id = chunk_cids[j] if j < len(chunk_cids) else f"batch_cxr_{i+j+1}_{uuid.uuid4().hex[:8]}"
                fname = f.filename or f"image_{i+j+1}.png"
                safe_fname = anonymize_filename(fname)

                try:
                    spool = await stream_to_spooled_tempfile(f, max_bytes=MAX_FILE_SIZE_BYTES)
                    try:
                        img, dcm_meta = decode_image_or_dicom(spool, fname)
                    finally:
                        spool.close()

                    quality = assess_cxr_quality(img, dcm_meta)

                    thumb = img.copy()
                    thumb.thumbnail((120, 120))
                    thumb_buf = BytesIO()
                    thumb.save(thumb_buf, format="PNG")
                    thumb_b64 = f"data:image/png;base64,{base64.b64encode(thumb_buf.getvalue()).decode('ascii')}"

                    chunk_images.append(img)
                    chunk_meta.append({
                        "index": i + j + 1,
                        "client_id": c_id,
                        "filename": safe_fname,
                        "is_dicom": dcm_meta.is_dicom,
                        "patient_id": dcm_meta.patient_id or "ANONYMIZED",
                        "quality": quality,
                        "preview_url": thumb_b64,
                    })
                except HTTPException as he:
                    errors.append(f"File index {i+j+1}: {he.detail}")
                except Exception as e:
                    errors.append(f"File index {i+j+1}: processing error")

            if chunk_images:
                chunk_preds = await asyncio.to_thread(pred.predict_batch, chunk_images, chunk_size=CHUNK_SIZE)
                for meta, preds in zip(chunk_meta, chunk_preds, strict=True):
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

                del chunk_images
                del chunk_meta

        return BatchPredictionResponse(
            status="ok" if results else "error",
            total=len(files),
            processed=len(results),
            failed=len(errors),
            results=results,
            errors=errors,
        )

    return app_instance


app = create_app()


async def stream_to_spooled_tempfile(file: UploadFile, max_bytes: int = MAX_FILE_SIZE_BYTES) -> tempfile.SpooledTemporaryFile:
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
    header_bytes = spool.read(132)
    spool.seek(0)

    is_dicom = filename.lower().endswith(".dcm") or (len(header_bytes) >= 132 and header_bytes[128:132] == b"DICM")

    if is_dicom and HAS_PYDICOM:
        try:
            dcm_header = pydicom.dcmread(spool, stop_before_pixels=True, force=True)
            num_frames = int(getattr(dcm_header, "NumberOfFrames", 1))
            rows = int(getattr(dcm_header, "Rows", 0))
            cols = int(getattr(dcm_header, "Columns", 0))

            if num_frames > 1:
                raise HTTPException(
                    status_code=400,
                    detail="Multi-frame DICOM series are not supported. Only single-frame CXR studies are permitted.",
                )

            total_pixels = num_frames * rows * cols if num_frames > 0 else rows * cols
            if total_pixels > MAX_DICOM_PIXELS or rows <= 0 or cols <= 0:
                raise HTTPException(
                    status_code=400,
                    detail=f"DICOM dimensions ({rows}x{cols}) exceed maximum safety limits.",
                )

            spool.seek(0)
            dcm = pydicom.dcmread(spool, force=True)

            try:
                arr = dcm.pixel_array
            except (NotImplementedError, Exception) as pe:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported DICOM compression or transfer syntax: {pe}",
                )

            # Strict medical DICOM pipeline order:
            # 1. Raw pixels
            arr = arr.astype(np.float32)

            # 2. Modality LUT / Rescale Slope & Intercept
            slope = float(getattr(dcm, "RescaleSlope", 1.0))
            intercept = float(getattr(dcm, "RescaleIntercept", 0.0))
            arr = (arr * slope) + intercept

            # 3. VOI LUT / Windowing
            try:
                arr = apply_voi_lut(arr, dcm)
            except Exception:
                wc = getattr(dcm, "WindowCenter", None)
                ww = getattr(dcm, "WindowWidth", None)
                if wc is not None and ww is not None:
                    try:
                        wc_val = float(wc[0]) if isinstance(wc, (list, tuple, pydicom.multival.MultiValue)) else float(wc)
                        ww_val = float(ww[0]) if isinstance(ww, (list, tuple, pydicom.multival.MultiValue)) else float(ww)
                        if ww_val > 0:
                            min_val = wc_val - 0.5 - (ww_val - 1) / 2.0
                            max_val = wc_val - 0.5 + (ww_val - 1) / 2.0
                            arr = np.clip((arr - min_val) / max(1e-5, (max_val - min_val)), 0.0, 1.0) * 255.0
                    except Exception:
                        pass

            # 4. MONOCHROME1 Inversion
            photometric = str(getattr(dcm, "PhotometricInterpretation", "MONOCHROME2")).upper()
            if "MONOCHROME1" in photometric:
                arr = np.amax(arr) - arr

            # 5. Normalization
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
                patient_name=mask_phi(raw_patient_name) if os.getenv("EXPOSE_PHI", "false").lower() == "true" else "REDACTED",
                study_date=str(raw_study_date) if os.getenv("EXPOSE_PHI", "false").lower() == "true" else "REDACTED",
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

    try:
        image = Image.open(spool)
        if image.width * image.height > Image.MAX_IMAGE_PIXELS:
            raise HTTPException(status_code=400, detail="Image pixel dimensions exceed safety limits.")
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


def get_current_model_info(
    pred: CheXpertPredictor | None,
    threshold_pl: dict[str, object] | None,
    ckpt_path: Path | None,
) -> dict[str, object]:
    info = dict(MODEL_INFO)
    if pred and pred.is_loaded:
        info["architecture"] = pred.architecture
        if threshold_pl and "mean_auc" in threshold_pl:
            auc = threshold_pl["mean_auc"]
            info["mean_auc"] = auc
            info["mean_auc_display"] = f"{auc:.4f}"
            info["checkpoint"] = str(ckpt_path) if ckpt_path else info["checkpoint"]
    return info
