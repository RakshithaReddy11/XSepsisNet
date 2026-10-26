"""SHAP explainability endpoints for the XSepsisNet backend."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from backend.schemas.prediction_schema import ExplainabilityRequest, ExplainabilityResponse
from backend.services.preprocessing_service import build_sequence
from backend.services.prediction_service import service

router = APIRouter()


@router.post("/explainability", response_model=ExplainabilityResponse)
def explainability(payload: ExplainabilityRequest) -> dict[str, Any]:
    """Return SHAP values for the exact sequence used by prediction."""
    try:
        sequence = build_sequence(payload.model_dump())
        result = service.explain(sequence)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    result["patient_id"] = payload.patient_id
    return result


__all__ = ["router"]
