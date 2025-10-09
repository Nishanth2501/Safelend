# SafeLend: Credit Risk Assistant

## Project Insights
- Delivers calibrated credit-risk predictions with ROC-AUC above 0.78 and precision exceeding 80% on high-risk cases.
- Surfaces the top contributing features for more than 95% of predictions, so loan officers understand every decision.
- Processes over 300,000 historical applications and aggregates data from six auxiliary tables using DuckDB.
- Supports near real-time scoring through a FastAPI service that responds in under 100 milliseconds in local tests.
- Provides a Streamlit Cloud deployment for quick hands-on evaluation at https://nishanth2501-safelend-uisafelend-app-wacmff.streamlit.app/

## Overview
SafeLend is an end-to-end credit risk platform inspired by the Home Credit Default Risk challenge. The project covers data preparation, feature engineering, model training, evaluation, and deployment in a way that mirrors production lending systems. The goal is to help lenders make faster, transparent decisions while retaining the ability to audit every recommendation.

## How the System Works
1. **Data pipeline** – Raw CSV files are cleaned, normalized, and enriched with DuckDB SQL aggregations that summarize bureau reports, past loans, credit cards, point-of-sale records, and installment payments.
2. **Feature engineering** – Domain-specific ratios, temporal signals, categorical encodings, and interaction terms are assembled in `src/features/build_features.py`, aligning train and test sets for modeling.
3. **Model training** – A LightGBM classifier is trained with stratified cross-validation, calibrated with isotonic regression, and tuned to meet a minimum precision target. Artifacts and metrics are written to the `artifacts/` directory.
4. **Inference service** – FastAPI exposes `/predict`, `/health`, and `/api/sample-data` endpoints. The service loads the calibrated model, computes feature contributions, and returns structured explanations.
5. **User interface** – A Streamlit app offers an interactive test bench where users can enter borrower profiles, view approval decisions, inspect SHAP-style explanations, and export payloads for reproducibility.

## Data Sources
The solution uses the Home Credit Default Risk dataset from Kaggle, including application data plus bureau, previous loan, installment, credit card, and POS histories. Each table is aggregated into Parquet files stored under `data/interim/`, then joined with the main application records to produce modeling-ready datasets in `data/processed/`.

## Getting Started
### Prerequisites
- Python 3.8 or later
- Optional: Node.js 18+ for running the Streamlit app or any additional frontend components

### Installation
```bash
git clone https://github.com/Nishanth2501/safelend.git
cd safelend
pip install -r requirements.txt
```

### Build the datasets and model
```bash
make data   # generate processed feature tables
make train  # train the LightGBM model and save artifacts
```

### Serve predictions
```bash
make serve  # launches FastAPI on http://localhost:8000
```
Visit `/docs` for interactive OpenAPI documentation and sample payloads.

### Explore the demo app
```bash
cd ui
streamlit run safelend.app.py
```
The Streamlit interface mirrors the API results, lists the top factors driving each decision, and provides narrative recommendations.

## Key Features
- **Explainable predictions** – Every score is paired with a ranked list of risk drivers and a plain-language summary.
- **Business-aware thresholding** – Decision thresholds are chosen to satisfy minimum precision targets, balancing approvals and risk.
- **Data quality controls** – Pipeline scripts include sanity checks for missing values, outliers, and drift markers.
- **Production readiness** – Logging, health endpoints, Docker support, and pytest-based regression tests are included to ease deployment.

## Technology Stack
- **Data and ML**: Python, pandas, NumPy, scikit-learn, LightGBM, SHAP, DuckDB
- **Model and deployment**: Calibrated LightGBM classifier persisted with joblib and exposed by the FastAPI service, mirrored in the hosted Streamlit demo at https://nishanth2501-safelend-uisafelend-app-wacmff.streamlit.app/
- **Service layer**: FastAPI, Pydantic, Uvicorn
- **Interface**: Streamlit application for scenario testing and storytelling
- **Tooling**: Makefile workflows, pytest, python-dotenv, joblib for model serialization

## Performance Summary
- ROC-AUC (out-of-fold): 0.78+
- Precision on high-risk cohort: above 80%
- Typical API response time: < 100 ms on commodity hardware
- Aggregated feature coverage: more than 170 engineered predictors across groups

## Contributing
Contributions are welcome. Please open an issue describing the change you propose, discuss significant design ideas in advance, and submit pull requests with relevant tests or notebooks when possible.

## Additional Resources
- Data exploration notebooks: `notebooks/`
- API usage example: `src/service/example_request.py`
- SQL aggregations: `sql/`
- Model training workflow: `src/models/train.py`

SafeLend provides a realistic template for delivering transparent credit decisions at scale. Use it as a reference for production ML patterns, feature engineering playbooks, or as a starting point for your own lending applications.
