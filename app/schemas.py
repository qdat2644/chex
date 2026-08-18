from __future__ import annotations

from pydantic import BaseModel, Field


class FindingPrediction(BaseModel):
    label: str
    probability: float = Field(ge=0.0, le=1.0)
    positive: bool
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    suspicion_level: str = "Low suspicion"


class HeatmapExplanation(BaseModel):
    label: str
    probability: float = Field(ge=0.0, le=1.0)
    image_data_url: str
    pure_heatmap_url: str | None = None


class QualityReport(BaseModel):
    is_likely_frontal_cxr: bool = True
    quality_score: float = Field(default=1.0, ge=0.0, le=1.0)
    suggested_view: str = "Frontal"
    warnings: list[str] = Field(default_factory=list)


class DicomMetadata(BaseModel):
    is_dicom: bool = False
    patient_id: str | None = None
    patient_name: str | None = None
    study_date: str | None = None
    modality: str | None = None
    view_position: str | None = None
    body_part: str | None = None
    photometric: str | None = None
    rows: int | None = None
    columns: int | None = None


class PredictionResponse(BaseModel):
    status: str
    filename: str
    width: int
    height: int
    mode: str
    findings: list[FindingPrediction]
    report: str | None = None
    heatmap: HeatmapExplanation | None = None
    all_heatmaps: dict[str, HeatmapExplanation] = Field(default_factory=dict)
    quality: QualityReport = Field(default_factory=QualityReport)
    dicom: DicomMetadata = Field(default_factory=DicomMetadata)
    message: str | None = None


class BatchItemResult(BaseModel):
    index: int
    client_id: str | None = None
    filename: str
    is_dicom: bool = False
    patient_id: str | None = None
    findings: list[FindingPrediction]
    top_finding: str
    top_probability: float
    positive_count: int
    positive_labels: list[str]
    report: str
    preview_url: str | None = None


class BatchPredictionResponse(BaseModel):
    status: str
    total: int
    processed: int
    failed: int = 0
    results: list[BatchItemResult]
    errors: list[str] = Field(default_factory=list)

