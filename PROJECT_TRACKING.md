# Project Tracking - Fraud Detection

## Scope
- Build an ML system to flag risky credit card transactions.
- Handle severe class imbalance.
- Provide threshold-based alerts and risk-level output.

## Progress Log
- [x] Create project structure
- [x] Implement training pipeline (XGBoost + Isolation Forest ensemble)
- [x] Implement evaluation CLI
- [x] Implement prediction CLI
- [x] Implement FastAPI realtime scoring service
- [x] Add alert audit logging (CSV/JSONL)
- [x] Add explanation endpoint (`/score-explain`)
- [x] Enforce strict request schema validation by feature set
- [x] Add per-IP rate limiting for scoring endpoints
- [x] Add Dockerfile and docker-compose for deployment
- [x] Add GitHub Actions CI workflow
- [x] Add GitHub Issue and PR templates
- [x] Add Dependabot configuration
- [x] Add config-based risk thresholds and score weights
- [x] Add README with run instructions
- [ ] Train on real `creditcard.csv`
- [ ] Review metrics and calibrate threshold for production objective

## Current Outputs
- Model artifact path: `models/fraud_detection_pipeline.joblib`
- Training metrics path: `reports/training_metrics.json`
- Training report path: `reports/training_report.md`
- Evaluation report path: `reports/evaluation_report.md`
- Prediction output path: `reports/predictions.csv`
- Realtime scoring API path: `src/api.py`
- Audit logs path: `reports/audit/alerts_YYYYMMDD.csv` and `.jsonl`
- Container files: `Dockerfile`, `docker-compose.yml`
- GitHub workflow: `.github/workflows/ci.yml`
- GitHub templates: `.github/ISSUE_TEMPLATE/*`, `.github/pull_request_template.md`
- Dependency automation: `.github/dependabot.yml`

## Notes
- The dataset is not committed to git (`data/*.csv` is ignored).
- Validation threshold is selected by best F1 from Precision-Recall curve.
- Keep monitoring false positives before production rollout.

## Next Actions
1. Put `creditcard.csv` into `data/`.
2. Run train command.
3. Run evaluate command.
4. Start API server and verify `/health`, `/metadata`, `/score`, `/score-explain`.
5. Check `reports/audit/` for alert history output.
6. Load test rate limit behavior and tune env thresholds.
7. Generate predictions and inspect top-risk transactions.
