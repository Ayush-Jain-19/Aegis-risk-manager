"""
src/api/main.py

FastAPI application exposing POST /v1/predict-fraud.
Features:
  1. In-Memory Feature Store for dynamic 1h/24h velocity computation.
  2. Non-blocking async threadpool execution for LightGBM & SHAP.
  3. Resilient 3-tier defense with rules-based fallback (Layer 2) and observability.
"""
from __future__ import annotations
from fastapi.middleware.cors import CORSMiddleware


import logging
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from fastapi import FastAPI, status
from starlette.concurrency import run_in_threadpool

# --------------------------------------------------------------------------- #
# Root Path & Sys.Path Anchoring
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.data_preprocessing import haversine_distance
from src.feature_encoding import CategoricalEncoder
from src.fallback_rules import apply_fallback_rules

from api.exception_handlers import register_exception_handlers
from api.schemas import PredictionOutput, TransactionInput

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

CONFIG_PATH = BASE_DIR / "config" / "config.yaml"

DEFAULT_THRESHOLD_FALLBACK = 0.37
BLOCK_THRESHOLD = 0.75

# --------------------------------------------------------------------------- #
# IN-MEMORY FEATURE STORE
# Keyed by user identifier (DOB + Customer Coordinates composite key).
# Stores deques of (timestamp_epoch_seconds, transaction_amount).
# --------------------------------------------------------------------------- #
feature_store: dict[str, deque[tuple[int, float]]] = {}


class ModelArtifacts:
    """
    Holds loaded artifacts in memory across requests.
    """

    def __init__(self):
        self.model = None
        self.threshold: float = DEFAULT_THRESHOLD_FALLBACK
        self.explainer = None
        self.feature_names: list[str] = []
        self.encoder: CategoricalEncoder | None = None
        self.config: dict = {}
        self.city_pop_fallback: float = 50000.0
        self.fallback_trigger_count: int = 0

    def load(self) -> None:
        with open(CONFIG_PATH, "r") as f:
            self.config = yaml.safe_load(f)

        model_dir = Path(self.config["paths"]["model_dir"])
        model_dir = (BASE_DIR / model_dir) if not model_dir.is_absolute() else model_dir

        self.model = joblib.load(model_dir / "lgbm_model.joblib")
        self.explainer = joblib.load(model_dir / "shap_explainer.joblib")
        self.feature_names = joblib.load(model_dir / "feature_names.joblib")

        encoder_path = Path(self.config["paths"]["encoder_path"])
        encoder_path = (BASE_DIR / encoder_path) if not encoder_path.is_absolute() else encoder_path
        self.encoder = joblib.load(encoder_path)

        try:
            self.threshold = float(joblib.load(model_dir / "optimal_threshold.joblib"))
        except FileNotFoundError:
            logger.warning(
                "optimal_threshold.joblib not found — using fallback default of %.2f.",
                DEFAULT_THRESHOLD_FALLBACK,
            )
            self.threshold = DEFAULT_THRESHOLD_FALLBACK

        self.city_pop_fallback = float(
            self.config.get("inference", {}).get("city_pop_fallback", 50000.0)
        )

        logger.info(
            "Loaded model artifacts from %s (threshold=%.3f, %d features).",
            model_dir,
            self.threshold,
            len(self.feature_names),
        )


artifacts = ModelArtifacts()


def build_app() -> FastAPI:
    app = FastAPI(
        title="AI Risk Manager — Fraud Detection API",
        description="Cost-aware fraud scoring with SHAP explainability and high-availability fallback.",
        version="1.0.0",
    )
    register_exception_handlers(app)

    @app.on_event("startup")
    def _startup() -> None:
        artifacts.load()

    return app


app = build_app()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For the hackathon, we allow all origins. In production, this would be your frontend URL.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok" if artifacts.model is not None else "not_ready",
        "threshold": artifacts.threshold,
        "n_features": len(artifacts.feature_names),
        "fallback_trigger_count": artifacts.fallback_trigger_count,
    }


def compute_velocity_features(payload: TransactionInput) -> dict[str, float]:
    """
    Queries and updates the in-memory feature store to compute real-time velocity metrics.
    """
    # HACKATHON FIX: Brute-force memory protection. 
    # If we get more than 10,000 unique users in RAM, wipe it so the server doesn't die.
    if len(feature_store) > 10000:
        logger.warning("Feature store memory cap reached. Clearing old state to prevent OOM.")
        feature_store.clear()

    user_key = f"{payload.dob}_{payload.customer_lat:.4f}_{payload.customer_long:.4f}"
    
    if payload.trans_time.tzinfo is None:
        current_time = int(payload.trans_time.replace(tzinfo=timezone.utc).timestamp())
    else:
        current_time = int(payload.trans_time.timestamp())

    current_amt = float(payload.amount)

    if user_key not in feature_store:
        feature_store[user_key] = deque()

    history = feature_store[user_key]

    # Clean transactions older than 24 hours (86,400 seconds)
    cutoff_24h = current_time - 86400
    cutoff_1h = current_time - 3600

    while history and history[0][0] < cutoff_24h:
        history.popleft()

    # Calculate metrics
    txn_count_24h = float(len(history))
    amt_sum_24h = float(sum(txn[1] for txn in history))
    txn_count_1h = float(sum(1 for txn in history if txn[0] >= cutoff_1h))
    seconds_since_last_txn = float(current_time - history[-1][0]) if history else 999999.0

    avg_24h = (amt_sum_24h / txn_count_24h) if txn_count_24h > 0 else current_amt
    amt_vs_24h_avg_ratio = float(current_amt / avg_24h) if avg_24h > 0 else 1.0

    # Append current transaction to user history
    history.append((current_time, current_amt))

    return {
        "txn_count_1h": txn_count_1h,
        "txn_count_24h": txn_count_24h,
        "amt_sum_24h": amt_sum_24h,
        "seconds_since_last_txn": seconds_since_last_txn,
        "amt_vs_24h_avg_ratio": amt_vs_24h_avg_ratio,
    }



def build_feature_row(payload: TransactionInput, artifacts: ModelArtifacts) -> pd.DataFrame:
    cols = artifacts.config["columns"]
    trans_time = payload.trans_time
    if trans_time.tzinfo is None:
        trans_time = trans_time.replace(tzinfo=timezone.utc)

    hour = trans_time.hour
    day_of_week = trans_time.weekday()
    month = trans_time.month
    is_weekend = int(day_of_week >= 5)
    hour_sin = float(np.sin(2 * np.pi * hour / 24))
    hour_cos = float(np.cos(2 * np.pi * hour / 24))

    distance_km = float(
        haversine_distance(
            np.array([payload.customer_lat]),
            np.array([payload.customer_long]),
            np.array([payload.merchant_lat]),
            np.array([payload.merchant_long]),
        )[0]
    )

    dob_dt = datetime.combine(payload.dob, datetime.min.time(), tzinfo=timezone.utc)
    age = (trans_time - dob_dt).days / 365.25

    def encoded_or_fallback(col: str, raw_value: str | None) -> float:
        mapping = artifacts.encoder.mappings_.get(col, pd.Series(dtype=float))
        fallback = artifacts.encoder.global_fallback_.get(col, 0.0)
        if raw_value is None:
            return fallback
        return float(mapping.get(raw_value, fallback))

    velocity_metrics = compute_velocity_features(payload)

    row = {
        "amt": payload.amount,
        "lat": payload.customer_lat,
        "long": payload.customer_long,
        "city_pop": artifacts.city_pop_fallback,
        "merch_lat": payload.merchant_lat,
        "merch_long": payload.merchant_long,
        "txn_count_1h": velocity_metrics["txn_count_1h"],
        "txn_count_24h": velocity_metrics["txn_count_24h"],
        "amt_sum_24h": velocity_metrics["amt_sum_24h"],
        "seconds_since_last_txn": velocity_metrics["seconds_since_last_txn"],
        "amt_vs_24h_avg_ratio": velocity_metrics["amt_vs_24h_avg_ratio"],
        "txn_hour": hour,
        "txn_day_of_week": day_of_week,
        "txn_month": month,
        "txn_is_weekend": is_weekend,
        "txn_hour_sin": hour_sin,
        "txn_hour_cos": hour_cos,
        "distance_from_home_km": distance_km,
        "age": age,
        f"{cols['merchant_col']}_{artifacts.encoder.method}_enc": encoded_or_fallback(
            cols["merchant_col"], payload.merchant
        ),
        f"{cols['category_col']}_{artifacts.encoder.method}_enc": encoded_or_fallback(
            cols["category_col"], payload.category
        ),
        f"job_{artifacts.encoder.method}_enc": encoded_or_fallback("job", None),
        f"city_{artifacts.encoder.method}_enc": encoded_or_fallback("city", None),
        f"state_{artifacts.encoder.method}_enc": encoded_or_fallback("state", None),
        f"gender_{artifacts.encoder.method}_enc": encoded_or_fallback("gender", None),
    }

    df = pd.DataFrame([row])
    df = df.reindex(columns=artifacts.feature_names)

    if df.isnull().any(axis=None):
        missing = df.columns[df.isnull().any()].tolist()
        raise ValueError(f"Could not construct required model features: {missing}")

    return df


def extract_top_shap_features(
    explainer, row: pd.DataFrame, feature_names: list[str], top_n: int = 3
) -> dict[str, float]:
    shap_values = explainer.shap_values(row)

    if isinstance(shap_values, list):
        values_for_row = np.asarray(shap_values[1 if len(shap_values) > 1 else 0])[0]
    else:
        arr = np.asarray(shap_values)
        if arr.ndim == 3:
            values_for_row = arr[0, :, 1]
        else:
            values_for_row = arr[0]

    contributions = list(zip(feature_names, values_for_row.tolist()))
    contributions.sort(key=lambda pair: abs(pair[1]), reverse=True)
    top = contributions[:top_n]
    return {name: round(float(value), 6) for name, value in top}


def decide_action(fraud_probability: float, threshold: float) -> str:
    if fraud_probability >= BLOCK_THRESHOLD:
        return "BLOCK"
    if fraud_probability >= threshold:
        return "REVIEW"
    return "APPROVE"

def run_sync_inference(payload: TransactionInput, artifacts: ModelArtifacts) -> PredictionOutput:
    """
    Encapsulates synchronous ML scoring to run cleanly inside a threadpool.
    """
    feature_row = build_feature_row(payload, artifacts)
    
    fraud_probability = float(artifacts.model.predict_proba(feature_row)[:, 1][0])
    is_fraud = fraud_probability >= artifacts.threshold
    action = decide_action(fraud_probability, artifacts.threshold)

    # FIX: We only burn CPU calculating SHAP if a human actually needs to look at it.
    if action in ["REVIEW", "BLOCK"]:
        shap_explanation = extract_top_shap_features(
            artifacts.explainer, feature_row, artifacts.feature_names, top_n=3
        )
    else:
        shap_explanation = {}

    return PredictionOutput(
        is_fraud=is_fraud,
        fraud_probability=round(fraud_probability, 6),
        action_taken=action,
        shap_explanation=shap_explanation,
        threshold_used=artifacts.threshold,
        fallback_triggered=False,
        reason=None,
    )


@app.post(
    "/v1/predict-fraud",
    response_model=PredictionOutput,
    status_code=status.HTTP_200_OK,
    summary="Score a single transaction for fraud risk.",
)
async def predict_fraud(payload: TransactionInput) -> PredictionOutput:
    """
    Scores transactions using the trained ML booster. If any internal component fails,
    seamlessly routes to the rules-based fallback engine without dropping the transaction.
    """
    try:
        # Offload CPU-bound ML scoring to threadpool so the async event loop stays responsive
        result = await run_in_threadpool(run_sync_inference, payload, artifacts)
        return result

    except Exception as exc:
        logger.critical(
            "ML pipeline failed (%s: %s) — routing to rules-based fallback.",
            type(exc).__name__,
            exc,
            exc_info=True,
        )
        artifacts.fallback_trigger_count += 1

        fallback_result = apply_fallback_rules(payload.model_dump(mode="json"))
        return PredictionOutput(**fallback_result)

if __name__ == "__main__":
    import uvicorn
    # This bypasses the string importer and runs your app directly
    uvicorn.run(app, host="127.0.0.1", port=8000)