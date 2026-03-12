from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

try:
    from xgboost import XGBClassifier
except ImportError as exc:  # pragma: no cover - import guard
    XGBClassifier = None
    XGBOOST_IMPORT_ERROR = exc
else:
    XGBOOST_IMPORT_ERROR = None


def build_xgb_classifier(scale_pos_weight: float, random_state: int) -> Any:
    if XGBClassifier is None:
        raise ImportError(
            "xgboost is required. Install dependencies with `pip install -r requirements.txt`."
        ) from XGBOOST_IMPORT_ERROR

    return XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        n_jobs=-1,
        random_state=random_state,
        scale_pos_weight=scale_pos_weight,
    )


def build_isolation_forest(contamination: float, random_state: int) -> IsolationForest:
    return IsolationForest(
        n_estimators=300,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    low = float(np.min(scores))
    high = float(np.max(scores))

    if not np.isfinite(low) or not np.isfinite(high):
        return np.zeros_like(scores)

    if high - low < 1e-12:
        return np.full_like(scores, 0.5)

    return (scores - low) / (high - low)


def compute_iforest_risk(model: IsolationForest, features: pd.DataFrame) -> np.ndarray:
    anomaly_score = -model.decision_function(features)
    return normalize_scores(anomaly_score)


def combine_scores(
    xgb_probability: np.ndarray,
    iforest_risk: np.ndarray,
    xgb_weight: float,
    iforest_weight: float,
) -> np.ndarray:
    total_weight = xgb_weight + iforest_weight
    if total_weight <= 0:
        raise ValueError("xgb_weight + iforest_weight must be > 0")

    combined = ((xgb_probability * xgb_weight) + (iforest_risk * iforest_weight)) / total_weight
    return np.clip(combined, 0.0, 1.0)


def select_best_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, dict[str, float]]:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)

    if thresholds.size == 0:
        return 0.5, {"best_f1": 0.0, "best_precision": 0.0, "best_recall": 0.0}

    f1_values = (2 * precision * recall) / (precision + recall + 1e-12)
    valid_f1 = f1_values[:-1]
    best_idx = int(np.argmax(valid_f1))

    return float(thresholds[best_idx]), {
        "best_f1": float(valid_f1[best_idx]),
        "best_precision": float(precision[best_idx]),
        "best_recall": float(recall[best_idx]),
    }


def apply_threshold(scores: np.ndarray, threshold: float) -> np.ndarray:
    return (scores >= threshold).astype(int)


def compute_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = apply_threshold(scores, threshold)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    try:
        roc_auc = float(roc_auc_score(y_true, scores))
    except ValueError:
        roc_auc = 0.0

    try:
        pr_auc = float(average_precision_score(y_true, scores))
    except ValueError:
        pr_auc = 0.0

    metrics = {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float((2 * tp) / max(2 * tp + fp + fn, 1)),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "alerts": int(int(y_pred.sum())),
    }
    return metrics


def assign_risk_level(score: float, medium_threshold: float, high_threshold: float) -> str:
    if score >= high_threshold:
        return "high"
    if score >= medium_threshold:
        return "medium"
    return "low"


def assign_risk_levels(
    scores: np.ndarray,
    medium_threshold: float,
    high_threshold: float,
) -> list[str]:
    return [assign_risk_level(float(score), medium_threshold, high_threshold) for score in scores]


def validate_feature_columns(dataframe: pd.DataFrame, expected_columns: list[str]) -> pd.DataFrame:
    missing = [column for column in expected_columns if column not in dataframe.columns]
    if missing:
        raise ValueError(f"Input data is missing required columns: {missing}")
    return dataframe[expected_columns]


def prepare_features_for_inference(
    dataframe: pd.DataFrame,
    expected_columns: list[str],
    training_medians: dict[str, float] | None,
) -> pd.DataFrame:
    features = validate_feature_columns(dataframe, expected_columns).copy()
    training_medians = training_medians or {}

    for column in expected_columns:
        fallback = float(training_medians.get(column, 0.0))
        features[column] = pd.to_numeric(features[column], errors="coerce").fillna(fallback)

    return features


def score_with_bundle(bundle: dict[str, Any], features: pd.DataFrame) -> np.ndarray:
    xgb_probability = bundle["xgb_model"].predict_proba(features)[:, 1]
    iforest_risk = compute_iforest_risk(bundle["iforest_model"], features)

    return combine_scores(
        xgb_probability=xgb_probability,
        iforest_risk=iforest_risk,
        xgb_weight=float(bundle.get("xgb_weight", 0.8)),
        iforest_weight=float(bundle.get("iforest_weight", 0.2)),
    )


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
