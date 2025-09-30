SafeLend: Credit Risk Assessment System

SafeLend is an end-to-end credit risk assessment system that predicts whether a loan applicant will repay or default.
It combines data science (feature engineering, modeling, explainability) with ML engineering (pipelines, APIs, UI demo).

⸻

Dataset

This project uses the Home Credit Default Risk dataset from Kaggle.
	•	Raw data: ~10 CSV files (application_train, bureau, previous_application, POS_CASH_balance, credit_card_balance, installments_payments, etc.)
	•	Final features: 176 engineered features after cleaning, aggregations, and transformations
	•	Train set size: ~307,000 applicants
	•	Test set size: ~48,000 applicants

The dataset simulates real-world credit scoring, with information about applicants’ demographics, previous credits, payment history, and financial behavior.

⸻

System Architecture

1. Data Pipeline
	•	Cleaning (src/data/preprocess.py)
	•	Handles missing values (for example, sentinel 365243 for dates)
	•	Normalizes categorical and numerical features
	•	SQL Aggregations (src/data/run_sql_aggregations.py)
	•	DuckDB queries for bureau, installments, credit card, POS, etc.
	•	Produces aggregated parquet files
	•	Final Dataset Build (src/data/build_processed.py)
	•	Combines cleaned and aggregated features into train_ready.parquet and test_ready.parquet

2. Model Training
	•	Model: LightGBM Gradient Boosting Classifier
	•	Calibration: CalibratedClassifierCV for probability calibration
	•	Artifacts:
	•	artifacts/safelend_model.joblib → trained model
	•	artifacts/features.json → feature list
	•	artifacts/threshold.json → decision threshold (about 0.707)
	•	artifacts/metrics.json → model performance metrics

3. API Service
	•	FastAPI server (src/service/server.py)
	•	Endpoints:
	•	GET /health → health check
	•	GET /schema → feature schema and threshold
	•	POST /predict → probability, repay/default decision, top feature contributions
	•	POST /predict?decision_only=true → simple YES/NO output

4. Demo UI

A React-based UI to showcase predictions.
	•	Enter applicant features
	•	Get a repay/default prediction
	•	See probability and top influencing factors
	•	Copy generated cURL commands for reproducibility

⸻

How to Run

1. Clone and Install
git clone https://github.com/Nishanth2501/safelend.git
cd safelend
pip install -r requirements.txt
2. Run Data Pipeline
make data
3. Train Model
make train
4. Start API
make serve
# API docs → http://localhost:8000/docs
5. Run Demo UI (React)
cd ui
npm install
npm run dev
# UI → http://localhost:5173

Features
	•	End-to-end ML pipeline (data → model → API → UI)
	•	176 engineered features from applicant history
	•	Explainable predictions with top factor contributions
	•	Decision thresholding for balanced repay/default classification
	•	FastAPI service ready for deployment
	•	React demo UI for easy interaction and showcase
Business Value

SafeLend helps lenders:
	•	Assess applicant risk using 176 financial and behavioral features
	•	Get both probability scores and clear YES/NO decisions
	•	Understand why a decision was made via feature contributions
	•	Simulate real-world fintech credit scoring pipelines

⸻

Tech Stack
	•	Python: pandas, NumPy, scikit-learn, LightGBM, DuckDB
	•	FastAPI for serving predictions
	•	React (Vite) for demo UI
	•	Parquet for efficient data storage
	•	Makefile for reproducible steps

