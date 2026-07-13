"""Response schemas — the public contract of the service."""

from __future__ import annotations

from pydantic import BaseModel, Field


class FieldOut(BaseModel):
    text: str
    confidence: float
    format_valid: bool


class DocumentOut(BaseModel):
    kind: str
    fields: dict[str, FieldOut]
    mean_confidence: float
    mrz_valid: bool | None = None
    mrz_checks: dict[str, bool] = Field(default_factory=dict)
    viz_mrz_consistency: dict[str, bool] = Field(default_factory=dict)


class FaceOut(BaseModel):
    similarity: float
    threshold: float
    fmr_level: str
    match: bool


class ForensicsOut(BaseModel):
    tamper_probability: float
    heatmap: list[list[float]] | None = None


class VerifyResponse(BaseModel):
    verdict: str = Field(description="pass | review | fail")
    reasons: list[str]
    corners: list[list[float]] | None = None
    document: DocumentOut | None = None
    face: FaceOut | None = None
    forensics: ForensicsOut | None = None
    timings_ms: dict[str, float]
    stages_available: dict[str, bool]


class HealthResponse(BaseModel):
    status: str
    ready: bool
    stages_available: dict[str, bool]
