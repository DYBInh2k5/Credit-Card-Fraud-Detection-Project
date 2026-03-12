from __future__ import annotations

import os
import time
from threading import Lock
from pathlib import Path
from typing import Any
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    from .audit_store import write_audit_events
    from .fraud_pipeline import assign_risk_levels, prepare_features_for_inference, score_with_bundle
except ImportError:
    from audit_store import write_audit_events
    from fraud_pipeline import assign_risk_levels, prepare_features_for_inference, score_with_bundle


class ScoreOneRequest(BaseModel):
    transaction: dict[str, Any] = Field(..., description="Single transaction feature dictionary")


class ScoreBatchRequest(BaseModel):
    transactions: list[dict[str, Any]] = Field(
        ..., min_length=1, description="List of transaction feature dictionaries"
    )


class ScoreExplainRequest(BaseModel):
    transaction: dict[str, Any] = Field(..., description="Single transaction feature dictionary")
    top_k: int = Field(5, ge=1, le=30, description="How many top feature contributors to return")


def _read_bool_env(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() not in {"0", "false", "no", "off"}


RATE_LIMIT_ENABLED = _read_bool_env("FRAUD_RATE_LIMIT_ENABLED", True)
RATE_LIMIT_REQUESTS = int(os.getenv("FRAUD_RATE_LIMIT_REQUESTS", "120"))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("FRAUD_RATE_LIMIT_WINDOW_SECONDS", "60"))

_RATE_LIMITER_STATE: dict[str, tuple[float, int]] = {}
_RATE_LIMITER_LOCK = Lock()


def _resolve_model_path() -> Path:
    return Path(os.getenv("FRAUD_MODEL_PATH", "models/fraud_detection_pipeline.joblib"))


def _load_bundle(model_path: Path) -> dict[str, Any]:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model bundle not found at {model_path.as_posix()}. Train first with src/train.py"
        )
    return joblib.load(model_path)


def _validate_transaction_schema(transaction: dict[str, Any], expected_columns: list[str]) -> None:
    received_columns = set(transaction.keys())
    expected_set = set(expected_columns)

    missing = sorted(expected_set - received_columns)
    extra = sorted(received_columns - expected_set)

    if missing or extra:
        details: list[str] = []
        if missing:
            details.append(f"missing={missing}")
        if extra:
            details.append(f"extra={extra}")
        detail_text = "; ".join(details)
        raise ValueError(
            "Transaction schema does not match model feature set. "
            f"{detail_text}. Use GET /metadata to inspect required features."
        )


def _validate_transactions_schema(transactions: list[dict[str, Any]], expected_columns: list[str]) -> None:
    for index, transaction in enumerate(transactions):
        try:
            _validate_transaction_schema(transaction, expected_columns)
        except ValueError as exc:
            raise ValueError(f"Invalid transaction at index {index}: {exc}") from exc


def _rate_limit_key(request: Request, endpoint: str) -> str:
    if request.client is not None and request.client.host:
        client_host = request.client.host
    else:
        client_host = "unknown"
    return f"{client_host}:{endpoint}"


def _check_rate_limit(request: Request, endpoint: str) -> int | None:
    if not RATE_LIMIT_ENABLED:
        return None

    now = time.time()
    key = _rate_limit_key(request, endpoint)

    with _RATE_LIMITER_LOCK:
        window_start, count = _RATE_LIMITER_STATE.get(key, (now, 0))
        elapsed = now - window_start

        if elapsed >= RATE_LIMIT_WINDOW_SECONDS:
            window_start = now
            count = 0

        if count >= RATE_LIMIT_REQUESTS:
            retry_after = max(1, int(RATE_LIMIT_WINDOW_SECONDS - elapsed))
            return retry_after

        _RATE_LIMITER_STATE[key] = (window_start, count + 1)

    return None


def _enforce_rate_limit(request: Request, endpoint: str) -> None:
    retry_after = _check_rate_limit(request, endpoint)
    if retry_after is None:
        return

    raise HTTPException(
        status_code=429,
        detail={
            "message": "Rate limit exceeded",
            "retry_after_seconds": retry_after,
            "limit": RATE_LIMIT_REQUESTS,
            "window_seconds": RATE_LIMIT_WINDOW_SECONDS,
            "endpoint": endpoint,
        },
    )


def _predict_rows(bundle: dict[str, Any], dataframe: pd.DataFrame) -> list[dict[str, Any]]:
    features = prepare_features_for_inference(
        dataframe=dataframe,
        expected_columns=bundle["feature_columns"],
        training_medians=bundle.get("training_medians", {}),
    )

    scores = score_with_bundle(bundle, features)
    alert_threshold = float(bundle.get("decision_threshold", 0.5))
    medium_threshold = float(bundle.get("risk_medium_threshold", 0.35))
    high_threshold = float(bundle.get("risk_high_threshold", 0.70))

    risk_levels = assign_risk_levels(scores, medium_threshold, high_threshold)
    alerts = (scores >= alert_threshold).astype(int)

    response_rows: list[dict[str, Any]] = []
    for index, score in enumerate(scores):
        response_rows.append(
            {
                "fraud_score": float(np.round(score, 6)),
                "is_alert": int(alerts[index]),
                "risk_level": risk_levels[index],
            }
        )
    return response_rows


def _build_audit_events(
    transactions: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    model_path_str: str,
    decision_threshold: float,
    source: str,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for index, transaction in enumerate(transactions):
        prediction = predictions[index]
        events.append(
            {
                "source": source,
                "request_id": str(uuid4()),
                "model_path": model_path_str,
                "decision_threshold": decision_threshold,
                "fraud_score": prediction["fraud_score"],
                "is_alert": prediction["is_alert"],
                "risk_level": prediction["risk_level"],
                "transaction": transaction,
            }
        )
    return events


def _explain_rows(bundle: dict[str, Any], features: pd.DataFrame, top_k: int) -> list[dict[str, Any]]:
    if xgb is None:
        raise RuntimeError("xgboost is required for explanation endpoint")

    booster = bundle["xgb_model"].get_booster()
    feature_names = list(features.columns)
    dmatrix = xgb.DMatrix(features, feature_names=feature_names)
    contribution_matrix = booster.predict(dmatrix, pred_contribs=True, validate_features=False)

    all_rows: list[dict[str, Any]] = []
    for row in contribution_matrix:
        contributions: list[dict[str, Any]] = []
        for feature_index, feature_name in enumerate(feature_names):
            value = float(row[feature_index])
            contributions.append(
                {
                    "feature": feature_name,
                    "contribution": float(np.round(value, 6)),
                    "abs_contribution": float(np.round(abs(value), 6)),
                }
            )

        top = sorted(contributions, key=lambda item: item["abs_contribution"], reverse=True)[:top_k]
        for item in top:
            item.pop("abs_contribution", None)

        all_rows.append(
            {
                "bias": float(np.round(row[-1], 6)),
                "top_contributors": top,
            }
        )

    return all_rows


model_path = _resolve_model_path()
app = FastAPI(
    title="Fraud Detection API",
    description="Realtime scoring API for credit card fraud risk",
    version="1.0.0",
)

try:
    MODEL_BUNDLE = _load_bundle(model_path)
except FileNotFoundError:
    MODEL_BUNDLE = None

EXPECTED_FEATURE_COLUMNS = [] if MODEL_BUNDLE is None else list(MODEL_BUNDLE.get("feature_columns", []))


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_loaded": MODEL_BUNDLE is not None,
        "model_path": model_path.as_posix(),
    }


@app.get("/metadata")
def metadata() -> dict[str, Any]:
    if MODEL_BUNDLE is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Train and restart API.")

    return {
        "model_path": model_path.as_posix(),
        "created_at": MODEL_BUNDLE.get("created_at"),
        "target_column": MODEL_BUNDLE.get("target_column", "Class"),
        "feature_columns": MODEL_BUNDLE.get("feature_columns", []),
        "decision_threshold": float(MODEL_BUNDLE.get("decision_threshold", 0.5)),
        "risk_medium_threshold": float(MODEL_BUNDLE.get("risk_medium_threshold", 0.35)),
        "risk_high_threshold": float(MODEL_BUNDLE.get("risk_high_threshold", 0.70)),
        "schema_validation": {
            "strict": True,
            "required_feature_count": len(EXPECTED_FEATURE_COLUMNS),
        },
        "rate_limit": {
            "enabled": RATE_LIMIT_ENABLED,
            "requests": RATE_LIMIT_REQUESTS,
            "window_seconds": RATE_LIMIT_WINDOW_SECONDS,
        },
    }


@app.post("/score")
def score_one(payload: ScoreOneRequest, request: Request) -> dict[str, Any]:
    if MODEL_BUNDLE is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Train and restart API.")

    _enforce_rate_limit(request, "/score")

    try:
        _validate_transaction_schema(payload.transaction, EXPECTED_FEATURE_COLUMNS)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    dataframe = pd.DataFrame([payload.transaction])

    try:
        prediction = _predict_rows(MODEL_BUNDLE, dataframe)[0]
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    write_audit_events(
        _build_audit_events(
            transactions=[payload.transaction],
            predictions=[prediction],
            model_path_str=model_path.as_posix(),
            decision_threshold=float(MODEL_BUNDLE.get("decision_threshold", 0.5)),
            source="api_score",
        )
    )

    return prediction


@app.post("/score-batch")
def score_batch(payload: ScoreBatchRequest, request: Request) -> dict[str, Any]:
    if MODEL_BUNDLE is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Train and restart API.")

    _enforce_rate_limit(request, "/score-batch")

    try:
        _validate_transactions_schema(payload.transactions, EXPECTED_FEATURE_COLUMNS)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    dataframe = pd.DataFrame(payload.transactions)

    try:
        rows = _predict_rows(MODEL_BUNDLE, dataframe)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    write_audit_events(
        _build_audit_events(
            transactions=payload.transactions,
            predictions=rows,
            model_path_str=model_path.as_posix(),
            decision_threshold=float(MODEL_BUNDLE.get("decision_threshold", 0.5)),
            source="api_score_batch",
        )
    )

    return {
        "count": len(rows),
        "results": rows,
    }


@app.post("/score-explain")
def score_explain(payload: ScoreExplainRequest, request: Request) -> dict[str, Any]:
    if MODEL_BUNDLE is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Train and restart API.")

    _enforce_rate_limit(request, "/score-explain")

    try:
        _validate_transaction_schema(payload.transaction, EXPECTED_FEATURE_COLUMNS)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    dataframe = pd.DataFrame([payload.transaction])

    try:
        prediction = _predict_rows(MODEL_BUNDLE, dataframe)[0]
        features = prepare_features_for_inference(
            dataframe=dataframe,
            expected_columns=MODEL_BUNDLE["feature_columns"],
            training_medians=MODEL_BUNDLE.get("training_medians", {}),
        )
        explanation = _explain_rows(MODEL_BUNDLE, features, payload.top_k)[0]
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    write_audit_events(
        _build_audit_events(
            transactions=[payload.transaction],
            predictions=[prediction],
            model_path_str=model_path.as_posix(),
            decision_threshold=float(MODEL_BUNDLE.get("decision_threshold", 0.5)),
            source="api_score_explain",
        )
    )

    return {
        "prediction": prediction,
        "explanation": explanation,
    }
