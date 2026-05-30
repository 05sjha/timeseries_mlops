"""
demand_forecast/serving/api.py
───────────────────────────────
FastAPI model serving endpoint.

MLOps KEY CONCEPTS:
  1. Input validation with Pydantic — bad requests rejected before prediction
  2. Model loading at startup (not per-request) — critical for latency
  3. Health + readiness endpoints — required by Kubernetes / Vertex AI
  4. Request logging — enables drift detection and debugging
  5. Graceful error handling — never expose stack traces to clients
"""

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import mlflow.xgboost
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

# ── Request / Response schemas ────────────────────────────────────────────────
# These Pydantic models are your API contract.
# They document what the model expects AND validate it automatically.

class ForecastRequest(BaseModel):
    """Input features for a single demand forecast prediction."""
    product_id: str
    price: float = Field(gt=0, description="Current shelf price in USD")
    promotion_flag: int = Field(ge=0, le=1, description="1 if on promotion")
    store_count: int = Field(gt=0, description="Number of stores carrying the product")

    # Lag features (required — must be computed from historical data upstream)
    lag_1w: float = Field(description="Units sold 1 week ago")
    lag_2w: float = Field(description="Units sold 2 weeks ago")
    lag_4w: float = Field(description="Units sold 4 weeks ago")
    lag_8w: float = Field(description="Units sold 8 weeks ago")
    lag_12w: float = Field(description="Units sold 12 weeks ago")

    # Rolling features
    rolling_mean_4w: float
    rolling_mean_8w: float
    rolling_mean_13w: float
    rolling_std_4w: float

    # Calendar features
    week_of_year: int = Field(ge=1, le=53)
    month: int = Field(ge=1, le=12)
    quarter: int = Field(ge=1, le=4)
    year: int = Field(ge=2020, le=2030)
    week_sin: float
    week_cos: float

    @field_validator("price")
    @classmethod
    def price_sanity(cls, v: float) -> float:
        if v > 1000:
            raise ValueError("Price above $1000 is likely a data error")
        return round(v, 2)


class ForecastResponse(BaseModel):
    """Model prediction response."""
    product_id: str
    predicted_units: float
    prediction_lower: float   # simple uncertainty bound
    prediction_upper: float
    model_version: str
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str


# ── Model state (loaded once at startup) ─────────────────────────────────────

class ModelState:
    """
    Holds the loaded model as application state.

    MLOPS INSIGHT: Loading a model from disk takes 2–10 seconds.
    Loading on every request = unacceptable latency.
    Load ONCE at startup, reuse on every request.
    In Kubernetes, a readiness probe prevents traffic until this completes.
    """
    model = None
    model_version: str = "not_loaded"
    feature_names: list[str] = []
    is_ready: bool = False


model_state = ModelState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle handler (FastAPI's modern approach)."""
    # ── STARTUP ───────────────────────────────────────────────────────────────
    model_uri = os.environ.get("MODEL_URI", "models:/demand-forecast-local-model/1")
    logger.info(f"Loading model from: {model_uri}")

    try:
        model_state.model = mlflow.xgboost.load_model(model_uri)
        model_state.model_version = model_uri.split("/")[-1]
        model_state.is_ready = True
        logger.info(f"Model loaded successfully. Version: {model_state.model_version}")
    except Exception as e:
        # In production, failure to load = container fails health check = restart
        logger.error(f"Failed to load model: {e}")
        # Don't crash — allow health endpoint to report unhealthy state

    yield  # Application runs here

    # ── SHUTDOWN ───────────────────────────────────────────────────────────────
    logger.info("Shutting down — releasing model resources")
    model_state.model = None
    model_state.is_ready = False


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Demand Forecast API",
    description="CPG demand forecasting — predicts weekly units sold per product",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health():
    """
    Health check endpoint.

    Kubernetes and Vertex AI call this every 30s.
    Return 200 = healthy, continue sending traffic.
    Return 503 = unhealthy, stop sending traffic, restart container.

    TWO TYPES in production:
    - Liveness probe: is the process alive? (/health)
    - Readiness probe: is it ready to serve? (/ready)
    Liveness failure → restart container.
    Readiness failure → remove from load balancer pool (no restart).
    """
    return HealthResponse(
        status="healthy" if model_state.is_ready else "unhealthy",
        model_loaded=model_state.is_ready,
        model_version=model_state.model_version,
    )


@app.get("/ready")
async def readiness():
    """Readiness probe — only 200 when model is loaded and ready."""
    if not model_state.is_ready:
        raise HTTPException(status_code=503, detail="Model not ready")
    return {"status": "ready"}


@app.post("/predict", response_model=ForecastResponse)
async def predict(request: ForecastRequest):
    """
    Generate a demand forecast for a single product-week.

    The input must include all features computed upstream (lags, rolling stats).
    In production, a feature store (Vertex AI Feature Store) supplies these.
    """
    if not model_state.is_ready:
        raise HTTPException(status_code=503, detail="Model not ready — try again in a moment")

    start = time.perf_counter()

    try:
        # Build feature vector in the same column order as training
        feature_dict = {
            "price": request.price,
            "promotion_flag": request.promotion_flag,
            "store_count": request.store_count,
            "lag_1w": request.lag_1w,
            "lag_2w": request.lag_2w,
            "lag_4w": request.lag_4w,
            "lag_8w": request.lag_8w,
            "lag_12w": request.lag_12w,
            "rolling_mean_4w": request.rolling_mean_4w,
            "rolling_mean_8w": request.rolling_mean_8w,
            "rolling_mean_13w": request.rolling_mean_13w,
            "rolling_std_4w": request.rolling_std_4w,
            "week_of_year": request.week_of_year,
            "month": request.month,
            "quarter": request.quarter,
            "year": request.year,
            "week_sin": request.week_sin,
            "week_cos": request.week_cos,
        }
        X = pd.DataFrame([feature_dict])
        prediction = float(model_state.model.predict(X)[0])
        prediction = max(0.0, prediction)   # units can't be negative

        # Simple uncertainty bounds (±15% — in production use conformal prediction)
        lower = prediction * 0.85
        upper = prediction * 1.15

        latency_ms = (time.perf_counter() - start) * 1000

        # Log prediction for monitoring (in production → Pub/Sub → BigQuery)
        logger.info(
            f"predict | product={request.product_id} | "
            f"units={prediction:.1f} | latency={latency_ms:.1f}ms"
        )

        return ForecastResponse(
            product_id=request.product_id,
            predicted_units=round(prediction, 1),
            prediction_lower=round(lower, 1),
            prediction_upper=round(upper, 1),
            model_version=model_state.model_version,
            latency_ms=round(latency_ms, 2),
        )

    except Exception as e:
        # Never expose internal errors to the client
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Prediction failed — see server logs")


@app.post("/predict/batch", response_model=list[ForecastResponse])
async def predict_batch(requests: list[ForecastRequest]):
    """
    Batch prediction endpoint — more efficient for multiple products.
    Used by downstream systems that need forecasts for all SKUs at once.
    """
    if not model_state.is_ready:
        raise HTTPException(status_code=503, detail="Model not ready")
    if len(requests) > 1000:
        raise HTTPException(status_code=400, detail="Max 1000 items per batch request")

    results = []
    for req in requests:
        result = await predict(req)
        results.append(result)
    return results
