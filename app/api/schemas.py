from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str
    classifiers_loaded: list[str]
    segmentation_available: bool


class PredictResponse(BaseModel):
    filename: str
    quality: dict[str, Any]
    prediction: dict[str, Any]


class AnalyzeResponse(BaseModel):
    filename: str
    quality: dict[str, Any]
    prediction: dict[str, Any] | None = None
    size_info: dict[str, Any] | None = None
    shape_info: dict[str, Any] | None = None
    mass_info: dict[str, Any] | None = None
    overlap: dict[str, Any] | None = None
    risk_info: dict[str, Any] | None = None
    progression_info: dict[str, Any] | None = None
    warnings: list[str] = Field(default_factory=list)


class ReportRequest(BaseModel):
    patient_name: str = ""
    analysis: dict[str, Any]


class ReportResponse(BaseModel):
    report: str
