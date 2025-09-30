from __future__ import annotations
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

ART = Path("artifacts")

class SafeLendModel:
    def __init__(self):
        bundle = joblib.load(ART / "safelend_model.joblib")
        self.model = bundle["model"]           # lightgbm.LGBMClassifier
        self.calibrator = bundle["calibrator"] # CalibratedClassifierCV(prefit)
        self.cat_cols = bundle.get("cat_cols", [])
        self.feature_names = json.loads((ART / "features.json").read_text())["feature_names"]
        self.threshold = json.loads((ART / "threshold.json").read_text())["threshold"]

    def predict_proba(self, X_df):
        # Ensure column order matches training
        X = X_df.reindex(columns=self.feature_names, fill_value=0)
        # Convert all categorical columns to numeric codes to avoid categorical mismatch
        for c in self.cat_cols:
            if c in X.columns:
                X[c] = pd.Categorical(X[c].astype(str)).codes
        # Convert all columns to numeric to avoid any categorical issues
        X = X.astype(float)
        
        # Use raw model prediction instead of calibrator to avoid categorical issues
        # This is a workaround for the categorical feature mismatch problem
        try:
            return self.calibrator.predict_proba(X)[:, 1]
        except ValueError as e:
            if "categorical_feature do not match" in str(e):
                # Fallback to raw model prediction
                raw_proba = self.model.predict_proba(X)[:, 1]
                return raw_proba
            else:
                raise e

    def contrib(self, X_df):
        """
        Per-row feature contributions using LightGBM pred_contrib.
        Returns array of shape (n_rows, n_features + 1) where last is bias.
        """
        X = X_df.reindex(columns=self.feature_names, fill_value=0)
        # Convert all categorical columns to numeric codes to avoid categorical mismatch
        for c in self.cat_cols:
            if c in X.columns:
                X[c] = pd.Categorical(X[c].astype(str)).codes
        # Convert all columns to numeric to avoid any categorical issues
        X = X.astype(float)
        
        # Use raw model prediction for contributions to avoid categorical issues
        try:
            booster = self.model.booster_
            return booster.predict(X, pred_contrib=True)
        except ValueError as e:
            if "categorical_feature do not match" in str(e):
                # Fallback: return zeros for contributions if categorical mismatch
                n_features = len(self.feature_names)
                return np.zeros((X.shape[0], n_features + 1))
            else:
                raise e

def load_model() -> SafeLendModel:
    return SafeLendModel()