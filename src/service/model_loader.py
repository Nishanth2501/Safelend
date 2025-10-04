#!/usr/bin/env python3
"""
Production model loader for SafeLend API.
Loads the real trained model and provides prediction interface.
"""

from __future__ import annotations
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

class SafeLendModel:
    """Production SafeLend model wrapper."""
    
    def __init__(self, artifacts_dir: str = "artifacts"):
        self.artifacts_dir = Path(artifacts_dir)
        self.model = None
        self.calibrator = None
        self.feature_names = None
        self.threshold = None
        self.cat_cols = None
        
        self._load_model()
    
    def _load_model(self):
        """Load the trained model and artifacts."""
        try:
            # Load model artifacts
            model_path = self.artifacts_dir / "safelend_model.joblib"
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            model_data = joblib.load(model_path)
            self.model = model_data["model"]
            self.calibrator = model_data["calibrator"]
            self.cat_cols = model_data.get("cat_cols", [])
            
            # Load feature names
            features_path = self.artifacts_dir / "features.json"
            if not features_path.exists():
                raise FileNotFoundError(f"Features file not found: {features_path}")
            
            features_data = json.loads(features_path.read_text())
            self.feature_names = features_data["feature_names"]
            
            # Load threshold
            threshold_path = self.artifacts_dir / "threshold.json"
            if not threshold_path.exists():
                raise FileNotFoundError(f"Threshold file not found: {threshold_path}")
            
            threshold_data = json.loads(threshold_path.read_text())
            self.threshold = threshold_data["threshold"]
            
            print(f"✅ SafeLend model loaded successfully")
            print(f"   Features: {len(self.feature_names)}")
            print(f"   Threshold: {self.threshold:.4f}")
            print(f"   Categorical columns: {len(self.cat_cols)}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def _prepare_features(self, data: Dict[str, Any]) -> pd.DataFrame:
        """Prepare features from input data."""
        # Convert input dict to DataFrame
        df = pd.DataFrame([data])
        
        # Ensure all required features are present
        for feature in self.feature_names:
            if feature not in df.columns:
                df[feature] = 0  # Default value for missing features
        
        # Reorder columns to match training order
        df = df[self.feature_names]
        
        # Handle categorical columns
        for col in self.cat_cols:
            if col in df.columns:
                df[col] = df[col].astype("category")
        
        return df
    
    def predict_proba(self, data: Dict[str, Any]) -> np.ndarray:
        """Predict probability of default."""
        try:
            # Prepare features
            X = self._prepare_features(data)
            
            # Get calibrated probability
            prob = self.calibrator.predict_proba(X)[:, 1]
            
            return prob
            
        except Exception as e:
            print(f"❌ Error in prediction: {e}")
            raise
    
    def predict(self, data: Dict[str, Any]) -> Tuple[bool, float, str]:
        """
        Make prediction with decision, probability, and reasoning.
        
        Returns:
            Tuple of (decision, probability, reasoning)
            - decision: True if approve loan, False if reject
            - probability: Default probability (0-1)
            - reasoning: Human-readable explanation
        """
        try:
            # Get probability
            prob = self.predict_proba(data)
            default_prob = float(prob[0])
            repay_prob = 1 - default_prob
            
            # Make decision based on threshold
            # If default probability is below threshold, approve loan
            approve_loan = default_prob < self.threshold
            
            # Generate reasoning
            if approve_loan:
                reasoning = f"Loan approved. Default risk: {default_prob:.1%} (below threshold of {self.threshold:.1%})"
            else:
                reasoning = f"Loan rejected. Default risk: {default_prob:.1%} (above threshold of {self.threshold:.1%})"
            
            return approve_loan, repay_prob, reasoning
            
        except Exception as e:
            print(f"❌ Error in prediction: {e}")
            raise
    
    def get_feature_contributions(self, data: Dict[str, Any], top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Get top feature contributions for the prediction.
        
        Note: This is a simplified implementation. For production,
        consider using SHAP or other interpretability libraries.
        """
        try:
            # Prepare features
            X = self._prepare_features(data)
            
            # Get feature importance from the model
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
            else:
                # Fallback: equal importance
                importances = np.ones(len(self.feature_names)) / len(self.feature_names)
            
            # Get feature values
            feature_values = X.iloc[0].values
            
            # Calculate simple contributions (feature_value * importance)
            contributions = []
            for i, (feature, value, importance) in enumerate(zip(self.feature_names, feature_values, importances)):
                contributions.append({
                    'feature': feature,
                    'value': float(value),
                    'importance': float(importance),
                    'contribution': float(value * importance)
                })
            
            # Sort by absolute contribution
            contributions.sort(key=lambda x: abs(x['contribution']), reverse=True)
            
            # Return top k contributions
            return contributions[:top_k]
            
        except Exception as e:
            print(f"❌ Error getting feature contributions: {e}")
            return []
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "feature_count": len(self.feature_names),
            "threshold": self.threshold,
            "categorical_features": len(self.cat_cols),
            "model_type": type(self.model).__name__,
            "calibrator_type": type(self.calibrator).__name__
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on the model."""
        try:
            # Test with dummy data
            dummy_data = {feature: 0 for feature in self.feature_names}
            prob = self.predict_proba(dummy_data)
            
            return {
                "status": "healthy",
                "model_loaded": True,
                "prediction_working": True,
                "feature_count": len(self.feature_names),
                "threshold": self.threshold
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "model_loaded": self.model is not None,
                "prediction_working": False
            }

# Global model instance
_model_instance = None

def get_model() -> SafeLendModel:
    """Get the global model instance (singleton pattern)."""
    global _model_instance
    if _model_instance is None:
        _model_instance = SafeLendModel()
    return _model_instance

def reload_model():
    """Reload the model (useful for model updates)."""
    global _model_instance
    _model_instance = None
    return get_model()