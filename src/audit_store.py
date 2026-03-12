from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _resolve_audit_dir() -> Path:
    configured = os.getenv("FRAUD_AUDIT_DIR", "reports/audit")
    return Path(configured)


def _audit_enabled() -> bool:
    value = os.getenv("FRAUD_AUDIT_ENABLED", "true").strip().lower()
    return value not in {"0", "false", "no", "off"}


def _alerts_only() -> bool:
    value = os.getenv("FRAUD_AUDIT_ALERTS_ONLY", "true").strip().lower()
    return value not in {"0", "false", "no", "off"}


def write_audit_events(events: list[dict[str, Any]]) -> dict[str, str] | None:
    if not events or not _audit_enabled():
        return None

    records: list[dict[str, Any]] = []
    for event in events:
        is_alert = int(event.get("is_alert", 0))
        if _alerts_only() and is_alert != 1:
            continue

        transaction = event.get("transaction", {})
        if not isinstance(transaction, dict):
            transaction = {"value": transaction}

        records.append(
            {
                "timestamp_utc": _utc_timestamp(),
                "source": str(event.get("source", "unknown")),
                "request_id": str(event.get("request_id", "")),
                "model_path": str(event.get("model_path", "")),
                "decision_threshold": float(event.get("decision_threshold", 0.5)),
                "fraud_score": float(event.get("fraud_score", 0.0)),
                "is_alert": is_alert,
                "risk_level": str(event.get("risk_level", "low")),
                "transaction_id": str(transaction.get("transaction_id", transaction.get("id", ""))),
                "amount": str(transaction.get("Amount", "")),
                "transaction_json": json.dumps(transaction, ensure_ascii=True, separators=(",", ":")),
            }
        )

    if not records:
        return None

    audit_dir = _resolve_audit_dir()
    audit_dir.mkdir(parents=True, exist_ok=True)

    day_key = datetime.now(timezone.utc).strftime("%Y%m%d")
    csv_path = audit_dir / f"alerts_{day_key}.csv"
    jsonl_path = audit_dir / f"alerts_{day_key}.jsonl"

    fieldnames = [
        "timestamp_utc",
        "source",
        "request_id",
        "model_path",
        "decision_threshold",
        "fraud_score",
        "is_alert",
        "risk_level",
        "transaction_id",
        "amount",
        "transaction_json",
    ]

    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(records)

    with jsonl_path.open("a", encoding="utf-8") as jsonl_file:
        for record in records:
            jsonl_file.write(json.dumps(record, ensure_ascii=True) + "\n")

    return {
        "csv_path": csv_path.as_posix(),
        "jsonl_path": jsonl_path.as_posix(),
        "written": str(len(records)),
    }
