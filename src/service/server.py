from __future__ import annotations
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
import pandas as pd
import numpy as np
from .mock_model import MockSafeLendModel
from .schemas import PredictRequest, PredictResponse, Factor

app = FastAPI(title="SafeLend API", version="0.1.0")
MODEL = MockSafeLendModel()

@app.get("/", response_class=HTMLResponse)
def root():
    """Serve the UI"""
    try:
        with open("static/index.html", "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="""
        <html>
        <head><title>SafeLend API</title></head>
        <body>
            <h1>SafeLend Credit Risk Assessment API</h1>
            <p>API is running! Available endpoints:</p>
            <ul>
                <li><a href="/health">/health</a> - Health check</li>
                <li><a href="/docs">/docs</a> - API documentation</li>
                <li><a href="/api/sample-data">/api/sample-data</a> - Get sample data</li>
                <li><a href="/predict">/predict</a> - Make prediction (POST)</li>
            </ul>
        </body>
        </html>
        """)

@app.get("/health")
def health():
    return {"status": "ok", "model_features": len(MODEL.feature_names)}

@app.get("/schema")
def schema():
    return {"feature_names": MODEL.feature_names, "threshold": MODEL.threshold}

@app.get("/api/sample-data")
def get_sample_data():
    """Get a sample loan application for testing"""
    try:
        # Load test data and return a random sample
        test_df = pd.read_parquet("Data/processed/test_modeling.parquet")
        sample_row = test_df.sample(1, random_state=42).iloc[0]
        
        # Convert to dict and handle numpy types
        features = {}
        for key, value in sample_row.items():
            if pd.isna(value):
                features[key] = None
            elif isinstance(value, (np.integer, np.floating)):
                features[key] = value.item()
            else:
                features[key] = str(value)
        
        return {"features": features}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading sample data: {str(e)}")

def _top_factors(x_row: pd.Series, contribs: np.ndarray, k: int = 5):
    # contribs includes bias as last column
    feat_names = MODEL.feature_names
    vals = contribs[:-1]  # drop bias
    # rank by absolute contribution
    idx = np.argsort(np.abs(vals))[::-1][:k]
    factors = []
    for i in idx:
        direction = "up_risk" if vals[i] > 0 else "down_risk"
        factors.append(Factor(
            feature=feat_names[i],
            direction=direction,
            contribution=float(vals[i]),
            value=x_row.get(feat_names[i], None),
            note=None
        ))
    return factors

def _reason_summary(factors):
    ups = [f.feature.replace("_", " ") for f in factors if f.direction == "up_risk"][:2]
    downs = [f.feature.replace("_", " ") for f in factors if f.direction == "down_risk"][:1]
    parts = []
    if ups:
        parts.append(f"Risk increased by {', '.join(ups)}")
    if downs:
        parts.append(f"and reduced by {', '.join(downs)}")
    return (", ".join(parts) + ".").strip().replace(" ,", ",")

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        # Build single-row DataFrame
        x = pd.DataFrame([req.features])

        # align to training feature space
        x_aligned = x.reindex(columns=MODEL.feature_names, fill_value=0)

        # probabilities
        prob = float(MODEL.predict_proba(x_aligned)[0])
        decision = "default" if prob >= MODEL.threshold else "repay"

        # contributions & top factors
        contrib = MODEL.contrib(x_aligned)[0]  # 1 x (n_features+1)
        factors = _top_factors(x_aligned.iloc[0], contrib, k=5)
        summary = _reason_summary(factors)

        return PredictResponse(
            model_version=app.version,
            default_probability=prob,
            threshold=float(MODEL.threshold),
            prediction=decision,
            top_factors=factors,
            reason_summary=summary
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))