from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from utils.saved_model_inference import predict_all_saved_models

app = FastAPI(
    title="Saved Model Inference API",
    version="1.0.0",
    description="HTTP wrapper around predict_all_saved_models.",
)


class PredictRequest(BaseModel):
    feature_values: list[Any] = Field(..., min_length=1)
    feature_names: list[str] | None = None


class HealthResponse(BaseModel):
    status: str
    models_dir: str


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    models_dir = Path(__file__).resolve().parents[1] / "models"
    return HealthResponse(status="ok", models_dir=str(models_dir))


@app.post("/predict")
def predict(request: PredictRequest) -> dict[str, dict[str, Any]]:
    try:
        return predict_all_saved_models(
            feature_values=request.feature_values,
            feature_names=request.feature_names,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Internal inference error: {exc}") from exc
