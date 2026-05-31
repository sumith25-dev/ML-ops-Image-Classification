"""
Adobe ML Engineer Project 1 - MLOps Pipeline
FastAPI serving layer with A/B testing and monitoring hooks
"""

import os
import time
import random
import logging
from contextlib import asynccontextmanager

import mlflow
import mlflow.pytorch
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from app.model_manager import ModelManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Prometheus metrics ────────────────────────────────────────────────────────
REQUEST_COUNT   = Counter("predictions_total", "Total predictions", ["model_version", "label"])
REQUEST_LATENCY = Histogram("prediction_latency_seconds", "Prediction latency", ["model_version"])
DRIFT_SCORE     = Counter("data_drift_detected_total", "Times drift was detected", ["feature"])

# ── App lifespan ─────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading models from MLflow registry...")
    app.state.model_manager = ModelManager()
    app.state.model_manager.load_models()
    yield
    logger.info("Shutting down model manager.")

app = FastAPI(
    title="MLOps Image Classifier - Adobe Project",
    description="Production-grade MLOps pipeline with A/B testing and drift monitoring",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    image_url: str
    user_id: str = "anonymous"
    force_version: str | None = None   # override A/B split for testing


class PredictResponse(BaseModel):
    label: str
    confidence: float
    model_version: str
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_versions: list[str]
    ab_split: dict


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
async def health(request: Request):
    mm: ModelManager = request.app.state.model_manager
    return {
        "status": "ok",
        "model_versions": mm.loaded_versions(),
        "ab_split": mm.ab_config,
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(body: PredictRequest, request: Request):
    mm: ModelManager = request.app.state.model_manager

    # A/B routing: 80% → stable, 20% → canary
    if body.force_version:
        version = body.force_version
    else:
        version = mm.route_ab(body.user_id)

    start = time.perf_counter()
    try:
        label, confidence = mm.predict(version, body.image_url)
    except Exception as exc:
        logger.error(f"Prediction error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

    latency = (time.perf_counter() - start) * 1000

    # Metrics
    REQUEST_COUNT.labels(model_version=version, label=label).inc()
    REQUEST_LATENCY.labels(model_version=version).observe(latency / 1000)

    logger.info(f"[{version}] {label} ({confidence:.2%}) in {latency:.1f}ms")

    return PredictResponse(
        label=label,
        confidence=round(confidence, 4),
        model_version=version,
        latency_ms=round(latency, 2),
    )


@app.get("/metrics")
async def metrics():
    """Prometheus scrape endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/rollback")
async def rollback(request: Request):
    """Instantly shift 100% traffic to stable version (operator action)."""
    mm: ModelManager = request.app.state.model_manager
    mm.ab_config = {"stable": 1.0, "canary": 0.0}
    logger.warning("ROLLBACK triggered — all traffic routed to stable model.")
    return {"message": "Rollback complete", "ab_split": mm.ab_config}
