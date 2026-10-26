"""FastAPI application entrypoint for XSepsisNet."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.routes.explainability import router as explainability_router
from backend.routes.patients import router as patients_router
from backend.routes.prediction import router as prediction_router
from backend.routes.reports import router as reports_router


app = FastAPI(title="XSepsisNet", version="1.0.0")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPLAINABILITY_DIR = PROJECT_ROOT / "models" / "explainability"
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "http://127.0.0.1:8000",
        "http://localhost:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(prediction_router)
app.include_router(explainability_router)
app.include_router(patients_router)
app.include_router(reports_router)
app.mount(
    "/dashboard-assets",
    StaticFiles(directory=EXPLAINABILITY_DIR),
    name="dashboard-assets",
)


@app.get("/", tags=["health"])
def root() -> dict[str, str]:
    return {"message": "XSepsisNet backend is running."}


@app.get("/browser_dashboard.html", include_in_schema=False)
def browser_dashboard() -> FileResponse:
    return FileResponse(EXPLAINABILITY_DIR / "browser_dashboard.html")


__all__ = ["app"]
