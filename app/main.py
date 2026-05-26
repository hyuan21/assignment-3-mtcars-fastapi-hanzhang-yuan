"""
FastAPI application that serves predictions from the trained mtcars model.

Endpoints
---------
GET  /health   - liveness check
GET  /ready    - readiness check (model loaded?)
POST /predict  - predict mpg from wt and hp

Configuration
-------------
MODEL_PATH    - path to the joblib artifact (default: models/model.pkl)
LOG_LEVEL     - python logging level (default: INFO)

Run locally
-----------
    uvicorn app.main:app --host 0.0.0.0 --port 8080
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
)
logger = logging.getLogger("mtcars-api")

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model.pkl"
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))

# `model_state` holds the loaded artifact. We deliberately use a module-level
# dict so the readiness endpoint can flip without restarting the process and
# so tests can monkeypatch it.
model_state: dict[str, Any] = {
    "model": None,
    "features": None,
    "target": None,
    "metrics": None,
    "error": None,
}


def load_model(path: Path = MODEL_PATH) -> None:
    """Load the joblib artifact into module state.

    Failures are caught and recorded in `model_state["error"]` so that the
    /ready endpoint can report them without crashing the whole service.
    """
    try:
        artifact = joblib.load(path)
        model_state["model"] = artifact["model"]
        model_state["features"] = artifact["features"]
        model_state["target"] = artifact["target"]
        model_state["metrics"] = artifact.get("metrics", {})
        model_state["error"] = None
        logger.info(
            "Model loaded from %s  features=%s  target=%s",
            path,
            model_state["features"],
            model_state["target"],
        )
    except FileNotFoundError as exc:
        model_state["error"] = f"Model file not found at {path}"
        logger.error(model_state["error"])
    except Exception as exc:  # noqa: BLE001 - we want to capture everything
        model_state["error"] = f"Failed to load model: {exc}"
        logger.exception("Model loading failed")


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------
class PredictionRequest(BaseModel):
    """Input payload for /predict.

    Predictors match those used during training: wt (1000 lbs) and hp.
    """

    wt: float = Field(
        ...,
        gt=0,
        description="Vehicle weight in 1000 lbs (e.g. 2.62 for a 2620 lb car).",
        examples=[2.62],
    )
    hp: float = Field(
        ...,
        gt=0,
        description="Gross horsepower (e.g. 110).",
        examples=[110],
    )


class PredictionResponse(BaseModel):
    # Disable Pydantic's "model_" protected namespace so we can keep the
    # natural field names model_intercept / model_coefficients without
    # warnings.
    model_config = {"protected_namespaces": ()}

    predicted_mpg: float = Field(..., description="Predicted miles per gallon.")
    features_used: list[str]
    model_intercept: float
    model_coefficients: dict[str, float]


class HealthResponse(BaseModel):
    status: str


class ReadyResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    status: str
    model_loaded: bool
    features: list[str] | None = None
    metrics: dict[str, float] | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Load the model when the app starts; nothing to clean up on shutdown."""
    load_model()
    yield


app = FastAPI(
    title="MTCARS FastAPI",
    description=(
        "Predicts mpg (miles per gallon) from vehicle weight (wt) and "
        "horsepower (hp) using a linear regression trained on the classic "
        "mtcars dataset."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    """Liveness check. Returns 200 whenever the process is up."""
    return HealthResponse(status="ok")


@app.get("/ready", tags=["meta"])
def ready() -> JSONResponse:
    """Readiness check. Returns 503 if the model is not loaded."""
    if model_state["model"] is None:
        body = ReadyResponse(
            status="unavailable",
            model_loaded=False,
            error=model_state["error"] or "Model not loaded",
        ).model_dump()
        return JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content=body)
    body = ReadyResponse(
        status="ready",
        model_loaded=True,
        features=model_state["features"],
        metrics=model_state["metrics"],
    ).model_dump()
    return JSONResponse(status_code=status.HTTP_200_OK, content=body)


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
def predict(payload: PredictionRequest) -> PredictionResponse:
    """Predict mpg from wt and hp."""
    if model_state["model"] is None:
        logger.error("Predict called but model is not loaded: %s", model_state["error"])
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=model_state["error"] or "Model not loaded",
        )

    model = model_state["model"]
    features: list[str] = model_state["features"]

    # Build the row in the exact feature order the model expects.
    row = [[getattr(payload, f) for f in features]]
    try:
        prediction = float(model.predict(row)[0])
    except Exception as exc:  # noqa: BLE001
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {exc}",
        ) from exc

    coefficients = {
        name: float(coef) for name, coef in zip(features, model.coef_)
    }
    logger.info(
        "predict  wt=%.3f  hp=%.3f  -> mpg=%.3f", payload.wt, payload.hp, prediction
    )
    return PredictionResponse(
        predicted_mpg=round(prediction, 4),
        features_used=features,
        model_intercept=float(model.intercept_),
        model_coefficients=coefficients,
    )


if __name__ == "__main__":
    # Convenience: `python -m app.main` runs uvicorn directly.
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8080")),
        reload=False,
    )
