from pydantic import BaseModel, Field


class FindingPrediction(BaseModel):
    label: str
    probability: float = Field(ge=0.0, le=1.0)
    positive: bool
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)


class HeatmapExplanation(BaseModel):
    label: str
    probability: float = Field(ge=0.0, le=1.0)
    image_data_url: str


class PredictionResponse(BaseModel):
    status: str
    filename: str
    width: int
    height: int
    mode: str
    findings: list[FindingPrediction]
    report: str | None = None
    heatmap: HeatmapExplanation | None = None
    message: str | None = None
