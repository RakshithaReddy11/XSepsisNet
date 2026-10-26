"""Request and response schemas for the XSepsisNet backend."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ClinicalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    patient_id: str | None = None
    hours: list[dict[str, Any]] | None = Field(default=None, min_length=6, max_length=6)
    sequence: list[list[float]] | None = None
    features: list[list[float]] | None = None


class PredictionResponse(BaseModel):
    patient_id: str | None = None
    model: str = "final_bilstm"
    sequence_length: int = 6
    clinical_features_per_hour: int = 39
    input_features_per_hour: int = 78
    raw_probability: float
    calibrated_probability: float
    classification_probability: float = Field(
        description="Probability used for classification; currently raw_probability."
    )
    classification_threshold: float = Field(
        description="Fixed threshold applied to classification_probability (0.50)."
    )
    predicted_class: int
    risk_level: str


class ExplainabilityResponse(BaseModel):
    patient_id: str | None = None
    model: str = "final_bilstm"
    sequence_length: int = 6
    clinical_features_per_hour: int = 39
    input_features_per_hour: int = 78
    raw_probability: float
    calibrated_probability: float
    classification_probability: float
    classification_threshold: float
    predicted_class: int
    clinical_contributions: list[dict[str, Any]]
    hourly_contributions: list[dict[str, Any]]
    explanation_output: str
    warning: str


PredictionRequest = ClinicalRequest
ExplainabilityRequest = ClinicalRequest

__all__ = [
    "ClinicalRequest",
    "ExplainabilityRequest",
    "ExplainabilityResponse",
    "PredictionRequest",
    "PredictionResponse",
]
