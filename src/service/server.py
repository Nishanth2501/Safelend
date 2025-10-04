from __future__ import annotations
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
import pandas as pd
import numpy as np
from .model_loader import get_model
from .schemas import PredictRequest, PredictResponse, Factor

app = FastAPI(title="SafeLend API", version="0.1.0")
MODEL = get_model()

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
    health_status = MODEL.health_check()
    return health_status

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

def _top_factors(contributions: list, k: int = 5):
    """Convert model contributions to Factor objects."""
    factors = []
    for i, contrib in enumerate(contributions[:k]):
        direction = "up_risk" if contrib['contribution'] > 0 else "down_risk"
        factors.append(Factor(
            feature=contrib['feature'],
            direction=direction,
            contribution=contrib['contribution'],
            value=contrib['value'],
            note=f"Importance: {contrib['importance']:.4f}"
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
        # Get prediction using real model
        approve_loan, repay_prob, reasoning = MODEL.predict(req.features)
        default_prob = 1 - repay_prob
        decision = "default" if not approve_loan else "repay"

        # Get feature contributions
        contributions = MODEL.get_feature_contributions(req.features, top_k=5)
        factors = _top_factors(contributions, k=5)
        summary = _reason_summary(factors)

        return PredictResponse(
            model_version=app.version,
            default_probability=default_prob,
            threshold=float(MODEL.threshold),
            prediction=decision,
            top_factors=factors,
            reason_summary=summary
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))