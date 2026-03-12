from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

try:
    from .fraud_pipeline import (
        assign_risk_levels,
        build_isolation_forest,
        build_xgb_classifier,
        combine_scores,
        compute_iforest_risk,
        compute_metrics,
        select_best_threshold,
        utc_now_iso,
    )
except ImportError:
    from fraud_pipeline import (
        assign_risk_levels,
        build_isolation_forest,
        build_xgb_classifier,
        combine_scores,
        compute_iforest_risk,
        compute_metrics,
        select_best_threshold,
        utc_now_iso,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train fraud detection pipeline")
    parser.add_argument("--data-path", type=Path, required=True, help="Path to credit card fraud CSV")
    parser.add_argument(
        "--target-col",
        type=str,
        default="Class",
        help="Target column in dataset (1 = fraud, 0 = normal)",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/fraud_detection_pipeline.joblib"),
        help="Output path for trained model bundle",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("reports/training_report.md"),
        help="Output markdown report path",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=Path("reports/training_metrics.json"),
        help="Output metrics JSON path",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=Path("configs/risk_thresholds.yaml"),
        help="Path to threshold and score-weight config",
    )
    parser.add_argument("--test-size", type=float, default=0.20, help="Test split ratio")
    parser.add_argument("--val-size", type=float, default=0.20, help="Validation split ratio")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    return parser.parse_args()


def load_config(config_path: Path) -> dict[str, dict[str, float]]:
    config = {
        "risk": {"medium": 0.35, "high": 0.70},
        "scoring": {"xgb_weight": 0.80, "iforest_weight": 0.20},
    }

    if config_path.exists():
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        config["risk"].update(loaded.get("risk", {}))
        config["scoring"].update(loaded.get("scoring", {}))

    medium = float(config["risk"]["medium"])
    high = float(config["risk"]["high"])
    if not (0.0 <= medium < high <= 1.0):
        raise ValueError("Risk thresholds must satisfy 0 <= medium < high <= 1")

    xgb_weight = float(config["scoring"]["xgb_weight"])
    iforest_weight = float(config["scoring"]["iforest_weight"])
    if xgb_weight + iforest_weight <= 0:
        raise ValueError("xgb_weight + iforest_weight must be > 0")

    return config


def load_dataset(data_path: Path, target_col: str) -> tuple[pd.DataFrame, np.ndarray, dict[str, float]]:
    dataframe = pd.read_csv(data_path)

    if target_col not in dataframe.columns:
        raise ValueError(f"Target column '{target_col}' not found in {data_path}")

    labels = pd.to_numeric(dataframe[target_col], errors="coerce").fillna(0).astype(int).to_numpy()
    unique_labels = set(np.unique(labels).tolist())
    if not unique_labels.issubset({0, 1}):
        raise ValueError(f"Target column must be binary with values 0/1, got: {unique_labels}")

    features = dataframe.drop(columns=[target_col])
    numeric_features = features.select_dtypes(include=[np.number]).copy()
    if numeric_features.empty:
        raise ValueError("No numeric features found after removing target column")

    medians = numeric_features.median(numeric_only=True).to_dict()
    for column, value in medians.items():
        numeric_features[column] = numeric_features[column].fillna(float(value))

    return numeric_features, labels, {key: float(value) for key, value in medians.items()}


def write_training_report(
    report_path: Path,
    dataset_rows: int,
    fraud_rows: int,
    val_threshold: float,
    val_best: dict[str, float],
    test_metrics: dict[str, float],
    risk_counts: dict[str, int],
    model_path: Path,
) -> None:
    fraud_rate = (fraud_rows / max(dataset_rows, 1)) * 100

    lines = [
        "# Fraud Detection Training Report",
        "",
        "## Dataset",
        f"- Rows: {dataset_rows}",
        f"- Fraud rows: {fraud_rows}",
        f"- Fraud rate: {fraud_rate:.4f}%",
        "",
        "## Validation Threshold Selection",
        f"- Selected threshold: {val_threshold:.6f}",
        f"- Best F1 (validation): {val_best['best_f1']:.6f}",
        f"- Precision at best threshold: {val_best['best_precision']:.6f}",
        f"- Recall at best threshold: {val_best['best_recall']:.6f}",
        "",
        "## Test Metrics",
        f"- Precision: {test_metrics['precision']:.6f}",
        f"- Recall: {test_metrics['recall']:.6f}",
        f"- F1: {test_metrics['f1']:.6f}",
        f"- ROC AUC: {test_metrics['roc_auc']:.6f}",
        f"- PR AUC: {test_metrics['pr_auc']:.6f}",
        f"- TP: {test_metrics['tp']} | FP: {test_metrics['fp']} | TN: {test_metrics['tn']} | FN: {test_metrics['fn']}",
        "",
        "## Risk Distribution On Test",
        f"- High: {risk_counts['high']}",
        f"- Medium: {risk_counts['medium']}",
        f"- Low: {risk_counts['low']}",
        "",
        "## Output",
        f"- Model bundle: {model_path.as_posix()}",
    ]

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = load_config(args.config_path)

    if not (0.0 < args.test_size < 1.0):
        raise ValueError("test-size must be between 0 and 1")
    if not (0.0 < args.val_size < 1.0):
        raise ValueError("val-size must be between 0 and 1")
    if args.test_size + args.val_size >= 1.0:
        raise ValueError("test-size + val-size must be < 1")

    features, labels, medians = load_dataset(args.data_path, args.target_col)

    x_train_val, x_test, y_train_val, y_test = train_test_split(
        features,
        labels,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=labels,
    )

    val_ratio = args.val_size / (1.0 - args.test_size)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val,
        y_train_val,
        test_size=val_ratio,
        random_state=args.random_state,
        stratify=y_train_val,
    )

    positive = int((y_train == 1).sum())
    negative = int((y_train == 0).sum())
    scale_pos_weight = max(1.0, negative / max(positive, 1))

    xgb_model = build_xgb_classifier(scale_pos_weight=scale_pos_weight, random_state=args.random_state)
    xgb_model.fit(x_train, y_train)

    contamination = float(np.clip(float(y_train.mean()) * 2.0, 0.001, 0.2))
    iforest_model = build_isolation_forest(contamination=contamination, random_state=args.random_state)

    normal_rows = x_train[y_train == 0]
    if len(normal_rows) >= 100:
        iforest_model.fit(normal_rows)
    else:
        iforest_model.fit(x_train)

    xgb_weight = float(config["scoring"]["xgb_weight"])
    iforest_weight = float(config["scoring"]["iforest_weight"])

    val_scores = combine_scores(
        xgb_probability=xgb_model.predict_proba(x_val)[:, 1],
        iforest_risk=compute_iforest_risk(iforest_model, x_val),
        xgb_weight=xgb_weight,
        iforest_weight=iforest_weight,
    )

    decision_threshold, val_best = select_best_threshold(y_val, val_scores)

    test_scores = combine_scores(
        xgb_probability=xgb_model.predict_proba(x_test)[:, 1],
        iforest_risk=compute_iforest_risk(iforest_model, x_test),
        xgb_weight=xgb_weight,
        iforest_weight=iforest_weight,
    )

    test_metrics = compute_metrics(y_test, test_scores, decision_threshold)

    medium_threshold = float(config["risk"]["medium"])
    high_threshold = float(config["risk"]["high"])
    risk_levels = assign_risk_levels(test_scores, medium_threshold, high_threshold)
    risk_counts = {
        "high": int(sum(level == "high" for level in risk_levels)),
        "medium": int(sum(level == "medium" for level in risk_levels)),
        "low": int(sum(level == "low" for level in risk_levels)),
    }

    bundle = {
        "created_at": utc_now_iso(),
        "target_column": args.target_col,
        "feature_columns": list(features.columns),
        "training_medians": medians,
        "xgb_model": xgb_model,
        "iforest_model": iforest_model,
        "xgb_weight": xgb_weight,
        "iforest_weight": iforest_weight,
        "decision_threshold": float(decision_threshold),
        "risk_medium_threshold": medium_threshold,
        "risk_high_threshold": high_threshold,
        "validation_selection": val_best,
        "test_metrics": test_metrics,
    }

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.model_path)

    metrics_payload = {
        "created_at": bundle["created_at"],
        "dataset_rows": int(len(features)),
        "fraud_rows": int((labels == 1).sum()),
        "decision_threshold": float(decision_threshold),
        "validation_selection": val_best,
        "test_metrics": test_metrics,
        "risk_distribution_test": risk_counts,
        "model_path": args.model_path.as_posix(),
    }

    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    write_training_report(
        report_path=args.report_path,
        dataset_rows=len(features),
        fraud_rows=int((labels == 1).sum()),
        val_threshold=float(decision_threshold),
        val_best=val_best,
        test_metrics=test_metrics,
        risk_counts=risk_counts,
        model_path=args.model_path,
    )

    print(f"Model saved to: {args.model_path.as_posix()}")
    print(f"Metrics saved to: {args.metrics_path.as_posix()}")
    print(f"Report saved to: {args.report_path.as_posix()}")


if __name__ == "__main__":
    main()
