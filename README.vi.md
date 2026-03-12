# Du An Phat Hien Gian Lan The Tin Dung

Tai lieu tieng Viet cho du an phat hien gian lan giao dich the tin dung bang machine learning.

## Muc tieu
- Cham diem rui ro giao dich trong khoang 0 den 1.
- Canh bao theo nguong de phuc vu van hanh.
- Ho tro day du train, evaluate, predict va API realtime.

## Thanh phan mo hinh
- Mo hinh supervised: XGBoost.
- Mo hinh anomaly detection: Isolation Forest.
- Diem cuoi cung: ket hop co trong so giua xac suat XGBoost va diem bat thuong tu Isolation Forest.

## Cau truc chinh
- src/train.py: huan luyen va chon nguong canh bao tren tap validation.
- src/evaluate.py: danh gia mo hinh tren du lieu co nhan.
- src/predict.py: du doan hang loat va xep muc rui ro.
- src/api.py: API FastAPI realtime.
- src/audit_store.py: luu lich su canh bao ra CSV/JSONL.
- configs/risk_thresholds.yaml: cau hinh nguong rui ro va trong so ensemble.

## Du lieu
- Dataset goi y: Credit Card Fraud Detection tren Kaggle.
- Dat file vao data/creditcard.csv.
- Cot nhan mac dinh: Class (0 la binh thuong, 1 la gian lan).

## Cai dat nhanh
1. Tao moi truong ao.
2. Cai thu vien:

pip install -r requirements.txt

## Chay huan luyen
python src/train.py --data-path data/creditcard.csv

Output:
- models/fraud_detection_pipeline.joblib
- reports/training_metrics.json
- reports/training_report.md

## Chay danh gia
python src/evaluate.py --data-path data/creditcard.csv

Output:
- reports/evaluation_metrics.json
- reports/evaluation_report.md

## Chay du doan
python src/predict.py --input-csv data/creditcard.csv --output-csv reports/predictions.csv

Cot duoc them:
- fraud_score
- is_alert
- risk_level

## Chay API realtime
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

Endpoint chinh:
- GET /health
- GET /metadata
- POST /score
- POST /score-batch
- POST /score-explain

## Gioi han toc do
- Ap dung theo IP cho cac endpoint cham diem.
- Mac dinh 120 request trong 60 giay.
- Cau hinh qua bien moi truong:
  - FRAUD_RATE_LIMIT_ENABLED
  - FRAUD_RATE_LIMIT_REQUESTS
  - FRAUD_RATE_LIMIT_WINDOW_SECONDS

## Audit log
- Luu vao reports/audit/alerts_YYYYMMDD.csv va reports/audit/alerts_YYYYMMDD.jsonl.
- Bien moi truong:
  - FRAUD_AUDIT_DIR
  - FRAUD_AUDIT_ENABLED
  - FRAUD_AUDIT_ALERTS_ONLY

## Docker
- Build va chay:

docker compose up --build

## Tu dong hoa GitHub
- CI: .github/workflows/ci.yml
- Docker publish: .github/workflows/docker-publish.yml
- Release theo tag: .github/workflows/release.yml
- Dependabot: .github/dependabot.yml
