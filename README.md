# Credit Card Fraud Detection Project

ML project for transaction fraud detection with imbalanced data.

## Goal
- Detect fraudulent transactions with a risk score in range [0, 1].
- Trigger alerts by threshold.
- Support training, evaluation, and prediction workflows.

## Models
- Supervised model: XGBoost classifier
- Anomaly model: Isolation Forest
- Final score: weighted ensemble between XGBoost probability and Isolation Forest risk score

## Project Structure
- `src/train.py`: train pipeline and choose alert threshold on validation set
- `src/evaluate.py`: evaluate a trained model on labeled data
- `src/predict.py`: generate fraud scores and risk levels for new transactions
- `src/api.py`: realtime scoring API with explanation endpoint
- `src/audit_store.py`: append alert history to CSV/JSONL for audit
- `src/fraud_pipeline.py`: shared scoring, threshold, and metric utilities
- `configs/risk_thresholds.yaml`: risk level and ensemble weight config
- `reports/`: metrics and generated markdown reports
- `models/`: trained model bundle output
- `data/`: dataset folder (put `creditcard.csv` here)

## Dataset
Use the Credit Card Fraud Detection dataset (Kaggle).
Expected file path:
- `data/creditcard.csv`

Expected target column:
- `Class` (0 = normal, 1 = fraud)

## Setup
1. Create and activate virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Train
Run training with default settings:

```bash
python src/train.py --data-path data/creditcard.csv
```

Outputs:
- `models/fraud_detection_pipeline.joblib`
- `reports/training_metrics.json`
- `reports/training_report.md`

## Evaluate (labeled data)

```bash
python src/evaluate.py --data-path data/creditcard.csv
```

Outputs:
- `reports/evaluation_metrics.json`
- `reports/evaluation_report.md`

## Predict (new transactions)

```bash
python src/predict.py --input-csv data/creditcard.csv --output-csv reports/predictions.csv
```

Output columns added:
- `fraud_score`: final risk score [0, 1]
- `is_alert`: 1 if score >= alert threshold, else 0
- `risk_level`: `low`, `medium`, `high`

## Realtime API (FastAPI)
After training a model, run the API service:

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

If your model file is in a custom path, set environment variable:

```bash
set FRAUD_MODEL_PATH=models/fraud_detection_pipeline.joblib
```

Available endpoints:
- `GET /health`: service and model-load status
- `GET /metadata`: model metadata, feature list, thresholds
- `POST /score`: score one transaction
- `POST /score-batch`: score multiple transactions
- `POST /score-explain`: score + top feature contributors from XGBoost

Example request for one transaction:

```bash
curl -X POST "http://127.0.0.1:8000/score" \
  -H "Content-Type: application/json" \
  -d "{\"transaction\":{\"Time\":1000,\"V1\":-1.23,\"V2\":0.12,\"Amount\":149.5}}"
```

Note:
- Input JSON is strictly validated against model feature columns (missing or extra keys are rejected).
- You can inspect required features via `GET /metadata`.

## Rate Limiting
Rate limiting is applied per client IP and per scoring endpoint:
- `POST /score`
- `POST /score-batch`
- `POST /score-explain`

Default policy:
- 120 requests / 60 seconds

Environment flags:
- `FRAUD_RATE_LIMIT_ENABLED` (default: `true`)
- `FRAUD_RATE_LIMIT_REQUESTS` (default: `120`)
- `FRAUD_RATE_LIMIT_WINDOW_SECONDS` (default: `60`)

When limit is exceeded, API returns HTTP 429 with `retry_after_seconds`.

Example request for explanation:

```bash
curl -X POST "http://127.0.0.1:8000/score-explain" \
  -H "Content-Type: application/json" \
  -d "{\"transaction\":{\"Time\":1000,\"V1\":-1.23,\"V2\":0.12,\"Amount\":149.5},\"top_k\":5}"
```

## Audit Logging (CSV/JSONL)
Alert history is appended automatically during:
- `POST /score`
- `POST /score-batch`
- `POST /score-explain`
- `python src/predict.py ...` (only alert rows)

Default output files:
- `reports/audit/alerts_YYYYMMDD.csv`
- `reports/audit/alerts_YYYYMMDD.jsonl`

Environment flags:
- `FRAUD_AUDIT_DIR` (default: `reports/audit`)
- `FRAUD_AUDIT_ENABLED` (default: `true`)
- `FRAUD_AUDIT_ALERTS_ONLY` (default: `true`)

## Run With Docker
Build and run API via Compose:

```bash
docker compose up --build
```

The service starts at `http://127.0.0.1:8000`.
Model path in container is configured as:
- `/app/models/fraud_detection_pipeline.joblib`

Rate-limit and audit envs can be customized in `docker-compose.yml`.

Make sure the model exists locally at:
- `models/fraud_detection_pipeline.joblib`

## Customize Thresholds and Weights
Edit `configs/risk_thresholds.yaml`:

```yaml
risk:
  medium: 0.35
  high: 0.70
scoring:
  xgb_weight: 0.80
  iforest_weight: 0.20
```

Notes:
- Alert threshold is selected automatically during training from validation set (best F1).
- `risk.medium` and `risk.high` are for risk band labels.

## Suggested Improvements
- Add time-based split for production-like validation.
- Add feature engineering for `Time` and `Amount`.
- Add calibration and business-oriented threshold tuning (precision/recall targets).
- Add drift detection dashboard for production traffic.
