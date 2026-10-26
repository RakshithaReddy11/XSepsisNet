"""Patient-related endpoints for the XSepsisNet backend."""

from __future__ import annotations

try:
    from fastapi import APIRouter
except ImportError:  # pragma: no cover
    class APIRouter:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            self.routes = []

        def get(self, *args, **kwargs):
            def decorate(func):
                self.routes.append(("GET", args, kwargs, func))
                return func
            return decorate

router = APIRouter()


@router.get("/patients")
def list_patients() -> dict[str, list[str]]:
    return {"patients": []}


@router.get("/patients/{patient_id}")
def get_patient(patient_id: str) -> dict[str, str]:
    return {"patient_id": patient_id, "status": "available"}


__all__ = ["router"]
