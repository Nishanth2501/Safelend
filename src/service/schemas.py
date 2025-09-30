from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Dict, Any, List

class PredictRequest(BaseModel):
    # Accept a loose dict of features → values.
    # We’ll align to the trained feature list on the server side.
    features: Dict[str, Any] = Field(..., description="Dict of feature_name → value")

class Factor(BaseModel):
    feature: str
    direction: str
    contribution: float
    value: Any | None = None
    note: str | None = None

class PredictResponse(BaseModel):
    model_version: str
    default_probability: float
    threshold: float
    prediction: str
    top_factors: List[Factor]
    reason_summary: str