from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

try:
    from .fraud_pipeline import (
        assign_risk_levels,
        compute_metrics,
        prepare_features_for_inference,
        score_with_bundle,
    )
except ImportError:
    from fraud_pipeline import (
        assign_risk_levels,
        compute_metrics,
        prepare_features_for_inference,
        score_with_bundle,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained fraud model on labeled dataset")
    parser.add_argument("--data-path", type=Path, required=True, help="Path to labeled CSV")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/fraud_detection_pipeline.joblib"),
        help="Path to trained model bundle",
    )
    parser.add_argument(
        "--target-col",
        type=str,
        default="Class",
        help="Label column in evaluation dataset",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override alert threshold. Uses model threshold if omitted.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("reports/evaluation_metrics.json"),
        help="Where to save evaluation metrics JSON",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("reports/evaluation_report.md"),
        help="Where to save evaluation markdown report",
    )
    return parser.parse_args()


def write_report(markdown_path: Path, metrics: dict[str, float], risk_counts: dict[str, int]) -> None:
    lines = [
        "# Fraud Detection Evaluation Report",
        "",
        "## Metrics",
        f"- Threshold: {metrics['threshold']:.6f}",
        f"- Precision: {metrics['precision']:.6f}",
        f"- Recall: {metrics['recall']:.6f}",
        f"- F1: {metrics['f1']:.6f}",
        f"- ROC AUC: {metrics['roc_auc']:.6f}",
        f"- PR AUC: {metrics['pr_auc']:.6f}",
        f"- TP: {metrics['tp']} | FP: {metrics['fp']} | TN: {metrics['tn']} | FN: {metrics['fn']}",
        "",
        "## Risk Distribution",
        f"- High: {risk_counts['high']}",
        f"- Medium: {risk_counts['medium']}",
        f"- Low: {risk_counts['low']}",
    ]

    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()

    bundle = joblib.load(args.model_path)

    dataframe = pd.read_csv(args.data_path)
    if args.target_col not in dataframe.columns:
        raise ValueError(f"Target column '{args.target_col}' not found in {args.data_path}")

    labels = pd.to_numeric(dataframe[args.target_col], errors="coerce").fillna(0).astype(int).to_numpy()

    features = prepare_features_for_inference(
        dataframe=dataframe,
        expected_columns=bundle["feature_columns"],
        training_medians=bundle.get("training_medians", {}),
    )

    scores = score_with_bundle(bundle, features)
    threshold = float(bundle["decision_threshold"] if args.threshold is None else args.threshold)
    metrics = compute_metrics(labels, scores, threshold)

    medium_threshold = float(bundle.get("risk_medium_threshold", 0.35))
    high_threshold = float(bundle.get("risk_high_threshold", 0.70))
    risk_levels = assign_risk_levels(scores, medium_threshold, high_threshold)
    risk_counts = {
        "high": int(sum(level == "high" for level in risk_levels)),
        "medium": int(sum(level == "medium" for level in risk_levels)),
        "low": int(sum(level == "low" for level in risk_levels)),
    }

    payload = {
        "model_path": args.model_path.as_posix(),
        "data_path": args.data_path.as_posix(),
        "metrics": metrics,
        "risk_distribution": risk_counts,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    write_report(args.output_md, metrics, risk_counts)

    print(f"Evaluation JSON saved to: {args.output_json.as_posix()}")
    print(f"Evaluation markdown saved to: {args.output_md.as_posix()}")


if __name__ == "__main__":
    main()
