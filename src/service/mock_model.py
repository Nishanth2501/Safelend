#!/usr/bin/env python3
"""Mock model for testing the API without categorical issues."""

import pandas as pd
import numpy as np
from pathlib import Path
import json

class MockSafeLendModel:
    def __init__(self):
        # Load feature names and threshold
        ART = Path("artifacts")
        self.feature_names = json.loads((ART / "features.json").read_text())["feature_names"]
        self.threshold = json.loads((ART / "threshold.json").read_text())["threshold"]
        
        print(f"✅ Mock model loaded with {len(self.feature_names)} features")
    
    def predict_proba(self, X_df):
        """Return mock probabilities based on simple heuristics."""
        # Ensure column order matches training
        X = X_df.reindex(columns=self.feature_names, fill_value=0)
        
        # Simple heuristic: higher credit amount = higher default risk
        if 'AMT_CREDIT' in X.columns:
            credit_amount = X['AMT_CREDIT'].iloc[0] if len(X) > 0 else 0
            # Normalize credit amount to 0-1 range (assuming max ~1M)
            prob = min(0.9, max(0.1, credit_amount / 1000000))
        else:
            # Random probability if no credit amount
            prob = np.random.random()
        
        return np.array([prob] * len(X))
    
    def contrib(self, X_df):
        """Return mock contributions."""
        n_features = len(self.feature_names)
        # Return random contributions
        contribs = np.random.normal(0, 0.1, (len(X_df), n_features + 1))
        return contribs
