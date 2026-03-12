# Changelog

All notable changes to this project are documented in this file.

## [v1.0.0] - 2026-03-12

### Added
- End-to-end fraud detection pipeline with XGBoost and Isolation Forest ensemble.
- CLI commands for training, evaluation, and prediction.
- FastAPI service with endpoints for health, metadata, score, batch score, and score explanation.
- Strict request schema validation based on trained feature columns.
- Per-IP rate limiting for scoring endpoints.
- Alert audit logging to CSV and JSONL files.
- Dockerfile and docker-compose deployment configuration.
- CI workflow for smoke testing on push and pull request.
- GitHub issue templates, pull request template, and Dependabot automation.

### Notes
- This v1.0.0 release establishes the first production-ready baseline for repository structure, model workflow, and deployment automation.
