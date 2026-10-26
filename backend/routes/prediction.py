"""Prediction endpoints for the XSepsisNet backend."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from backend.schemas.prediction_schema import PredictionRequest, PredictionResponse
from backend.services.preprocessing_service import build_sequence
from backend.services.prediction_service import service

router = APIRouter()


def _request_data(payload: Any) -> dict[str, Any]:
    return payload.model_dump()


@router.get("/health", tags=["health"])
def healthcheck() -> dict[str, str]:
    return {"status": "ok", "model": "final_bilstm"}


@router.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> dict[str, Any]:
    """Predict raw and calibrated risk for six hourly clinical records."""
    try:
        sequence = build_sequence(_request_data(payload))
        result = service.prediction(sequence)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    result["patient_id"] = payload.patient_id
    return result


__all__ = ["router"]
