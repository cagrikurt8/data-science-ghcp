"""FastAPI inference endpoint for bank-churn model.

Endpoints:
    GET  /health    - liveness/readiness probe
    POST /predict   - skor + iş kuralına göre aksiyon

Conditional GO: `<30` segmenti `manual_review` olarak donulur (decision_policy.json).
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

import joblib
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from src.serve.policy import RAW_PASSTHROUGH_COLS, load_policy, score_and_decide

logger = logging.getLogger("bank-churn-serve")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/best_model.joblib"))
PREPROC_PATH = Path(os.getenv("PREPROCESSOR_PATH", "models/preprocessor.joblib"))
POLICY_PATH = Path(os.getenv("POLICY_PATH", "models/decision_policy.json"))


class Customer(BaseModel):
    """Tek musteri kaydi (ham kolonlar). PII (Surname) kabul edilmez."""
    CustomerId: int | None = Field(default=None, description="Banka ici kimlik (opsiyonel)")
    CreditScore: int = Field(ge=300, le=900)
    Geography: Literal["France", "Germany", "Spain"]
    Gender: Literal["Male", "Female"]
    Age: int = Field(ge=18, le=120)
    Tenure: int = Field(ge=0, le=20)
    Balance: float = Field(ge=0)
    NumOfProducts: int = Field(ge=1, le=10)
    HasCrCard: int = Field(ge=0, le=1)
    IsActiveMember: int = Field(ge=0, le=1)
    EstimatedSalary: float = Field(ge=0)


class PredictRequest(BaseModel):
    customers: list[Customer] = Field(min_length=1, max_length=1000)


class Prediction(BaseModel):
    customer_id: int | None
    score: float
    risk_band: Literal["high", "medium", "low"]
    action: str
    reason: str


class PredictResponse(BaseModel):
    model_version: str
    decision: str
    predictions: list[Prediction]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    policy_version: str | None


# Lazy-loaded singletons
_state: dict[str, object] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading model artefacts...")
    if not MODEL_PATH.exists() or not PREPROC_PATH.exists() or not POLICY_PATH.exists():
        logger.error("Missing artefacts: model=%s preproc=%s policy=%s",
                     MODEL_PATH.exists(), PREPROC_PATH.exists(), POLICY_PATH.exists())
        raise RuntimeError("Model artefacts not found")
    _state["model"] = joblib.load(MODEL_PATH)
    _state["preprocessor"] = joblib.load(PREPROC_PATH)
    _state["policy"] = load_policy(POLICY_PATH)
    logger.info("Model loaded. policy_version=%s", _state["policy"].get("version"))
    yield
    _state.clear()


app = FastAPI(title="bank-churn-serve", version="1.0.0", lifespan=lifespan)


def get_state() -> dict[str, object]:
    if not _state:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="model not ready")
    return _state


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    loaded = bool(_state.get("model"))
    policy = _state.get("policy") or {}
    return HealthResponse(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
        policy_version=policy.get("version") if isinstance(policy, dict) else None,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(
    payload: PredictRequest,
    state: Annotated[dict[str, object], Depends(get_state)],
) -> PredictResponse:
    raw_df = pd.DataFrame([c.model_dump() for c in payload.customers])
    missing = [c for c in RAW_PASSTHROUGH_COLS if c not in raw_df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"missing columns: {missing}")

    decisions = score_and_decide(
        raw_df,
        policy=state["policy"],
        model=state["model"],
        preprocessor=state["preprocessor"],
    )
    policy = state["policy"]
    return PredictResponse(
        model_version=policy.get("version", "unknown"),
        decision=policy.get("decision", "unknown"),
        predictions=[Prediction(**row) for row in decisions.to_dict(orient="records")],
    )