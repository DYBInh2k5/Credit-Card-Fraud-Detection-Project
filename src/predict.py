from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

try:
    from .audit_store import write_audit_events
    from .fraud_pipeline import assign_risk_levels, prepare_features_for_inference, score_with_bundle
except ImportError:
    from audit_store import write_audit_events
    from fraud_pipeline import assign_risk_levels, prepare_features_for_inference, score_with_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict fraud risk scores from transaction CSV")
    parser.add_argument("--input-csv", type=Path, required=True, help="Input CSV path")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/fraud_detection_pipeline.joblib"),
        help="Path to trained model bundle",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("reports/predictions.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override alert threshold (model threshold if omitted)",
    )
    parser.add_argument(
        "--medium-threshold",
        type=float,
        default=None,
        help="Override medium risk threshold",
    )
    parser.add_argument(
        "--high-threshold",
        type=float,
        default=None,
        help="Override high risk threshold",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="How many highest-risk rows to print as summary",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    bundle = joblib.load(args.model_path)
    dataframe = pd.read_csv(args.input_csv)

    features = prepare_features_for_inference(
        dataframe=dataframe,
        expected_columns=bundle["feature_columns"],
        training_medians=bundle.get("training_medians", {}),
    )

    scores = score_with_bundle(bundle, features)

    alert_threshold = float(bundle["decision_threshold"] if args.threshold is None else args.threshold)
    medium_threshold = float(
        bundle.get("risk_medium_threshold", 0.35)
        if args.medium_threshold is None
        else args.medium_threshold
    )
    high_threshold = float(
        bundle.get("risk_high_threshold", 0.70)
        if args.high_threshold is None
        else args.high_threshold
    )

    if not (0.0 <= medium_threshold < high_threshold <= 1.0):
        raise ValueError("Thresholds must satisfy 0 <= medium < high <= 1")

    output = dataframe.copy()
    output["fraud_score"] = np.round(scores, 6)
    output["is_alert"] = (scores >= alert_threshold).astype(int)
    output["risk_level"] = assign_risk_levels(scores, medium_threshold, high_threshold)

    output = output.sort_values("fraud_score", ascending=False)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_csv, index=False)

    alert_rows = output[output["is_alert"] == 1]
    audit_events = [
        {
            "source": "predict_cli",
            "fraud_score": float(row.fraud_score),
            "is_alert": int(row.is_alert),
            "risk_level": str(row.risk_level),
            "decision_threshold": float(alert_threshold),
            "model_path": args.model_path.as_posix(),
            "transaction": row.drop(labels=["fraud_score", "is_alert", "risk_level"]).to_dict(),
        }
        for _, row in alert_rows.iterrows()
    ]
    audit_paths = write_audit_events(audit_events)

    alert_count = int(output["is_alert"].sum())
    total_count = len(output)
    high_count = int((output["risk_level"] == "high").sum())
    medium_count = int((output["risk_level"] == "medium").sum())

    print(f"Predictions saved to: {args.output_csv.as_posix()}")
    print(f"Total rows: {total_count}")
    print(f"Alerts (score >= {alert_threshold:.4f}): {alert_count}")
    print(f"High risk rows: {high_count}")
    print(f"Medium risk rows: {medium_count}")
    if audit_paths is not None and alert_count > 0:
        print(f"Audit CSV appended: {audit_paths['csv_path']}")
        print(f"Audit JSONL appended: {audit_paths['jsonl_path']}")

    if args.top_k > 0:
        print("\nTop risk rows:")
        columns_to_show = [column for column in ["fraud_score", "is_alert", "risk_level"] if column in output.columns]
        print(output[columns_to_show].head(args.top_k).to_string(index=False))


if __name__ == "__main__":
    main()
