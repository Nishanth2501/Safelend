# safelend_app.py
# Streamlit UI for a portfolio-ready, explainable credit risk demo
# -----------------------------------------------------------------
# Features
# - Clean form UI with realistic features
# - Approval decision, probability, risk band, and adjustable threshold
# - Top risk drivers chart (uses SHAP values if available, else falls back to simple feature impacts)
# - Plain-language recommendations
# - Exportable JSON payload for reproducibility
# - Demo-safe: works without a trained model by using a calibrated mock model
#
# To use with a real model:
# 1) Save your trained classifier (must implement predict_proba) as `models/model.pkl`
# 2) Optionally save a fitted SHAP explainer for the same model as `models/explainer.pkl`
# 3) Ensure feature order in `FEATURES` matches your training pipeline

import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# Optional imports; app still runs if these are missing
try:
    import shap  # for model explainability
except Exception:  # pragma: no cover
    shap = None

# ---------------------------
# Configuration & constants
# ---------------------------
APP_NAME = "🏦 SafeLend Credit Risk Assessment"
BRAND_TAGLINE = "Demo • Explainable ML • Not financial advice"

FEATURES = [
    # Numeric / encoded features. Order matters if you plug in a real model.
    "credit_amount",
    "annual_income",
    "debt_to_income",                # percent (0-100)
    "credit_history_years",
    "age",
    "children_count",
    "days_registration",            # days since registration (positive number)
    "own_car_age",                   # years
    "living_apartments_mode",       # 0/1 feature for demo
    "flag_document_8",              # 0/1
    # One-hot style demo encodings (simple integers here for brevity)
    "gender_male",                   # 1 if male else 0
    "contract_cash_loans",          # 1 if cash loans else 0
    "owns_car",                      # 1 yes else 0
    "owns_realty",                   # 1 yes else 0
    "live_city_not_work_city",      # 1 if True else 0
    "education_level",              # ordinal for demo: 0=Other,1=Secondary,2=Higher,3=Postgrad
    "marital_status",               # ordinal for demo: 0=Other,1=Single,2=Married,3=Divorced
    "employment_type"               # ordinal for demo: 0=Other,1=Salaried,2=Self-Employed,3=Contract
]

RISK_BANDS = [
    (0.00, 0.15, "Very Low"),
    (0.15, 0.30, "Low"),
    (0.30, 0.50, "Medium"),
    (0.50, 0.70, "High"),
    (0.70, 1.01, "Very High"),
]

# ---------------------------
# Utilities
# ---------------------------
@dataclass
class Decision:
    default_prob: float
    threshold: float
    approved: bool
    risk_band: str


def bucket_risk(p: float) -> str:
    for lo, hi, name in RISK_BANDS:
        if lo <= p < hi:
            return name
    return "Unknown"


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


class MockModel:
    """A demo-friendly, deterministic model that resembles a trained classifier.
    Replace with your real model for production/portfolio depth.
    """

    def __init__(self, seed: int = 7):
        rng = np.random.default_rng(seed)
        # Random but fixed weights per feature for deterministic output
        self.coef_ = rng.normal(loc=0.0, scale=0.2, size=(len(FEATURES),))
        # Encourage realistic directionalities for a few features
        name_to_idx = {f: i for i, f in enumerate(FEATURES)}
        # More debt_to_income => higher default prob
        self.coef_[name_to_idx["debt_to_income"]] = 0.035
        # Longer credit history => lower risk
        self.coef_[name_to_idx["credit_history_years"]] = -0.06
        # Higher income => lower risk
        self.coef_[name_to_idx["annual_income"]] = -0.000002
        # Higher credit amount => slightly higher risk
        self.coef_[name_to_idx["credit_amount"]] = 0.0000015
        # Age (somewhat U-shaped; approximate by positive weight on young)
        self.coef_[name_to_idx["age"]] = -0.005
        self.intercept_ = -0.5

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        z = X @ self.coef_ + self.intercept_
        p = sigmoid(z)
        # Ensure p is a numpy array
        if not isinstance(p, np.ndarray):
            p = np.array(p)
        # Ensure p is 2D for proper stacking
        if p.ndim == 1:
            p = p.reshape(-1, 1)
        return np.column_stack([1 - p, p])


@st.cache_resource(show_spinner=False)
def load_model_and_explainer():
    model_path = "models/model.pkl"
    explainer_path = "models/explainer.pkl"
    model = None
    explainer = None

    if os.path.exists(model_path):
        import joblib
        model = joblib.load(model_path)
    else:
        model = MockModel()

    if shap is not None and os.path.exists(explainer_path):
        try:
            import joblib
            explainer = joblib.load(explainer_path)
        except Exception:
            explainer = None

    return model, explainer


def decision_from_prob(p: float, threshold: float) -> Decision:
    return Decision(
        default_prob=p,
        threshold=threshold,
        approved=(p < threshold),
        risk_band=bucket_risk(p),
    )


def top_feature_impacts(
    x: np.ndarray,
    feature_names: List[str],
    model,
    explainer=None,
    k: int = 6,
) -> List[Tuple[str, float]]:
    """Return top-k feature contributions toward default probability.
    If a SHAP explainer is provided, use it; else compute gradient-like proxy.
    Positive values => increase risk; negative => decrease risk.
    """
    if explainer is not None and shap is not None:
        try:
            shap_values = explainer.shap_values(x.reshape(1, -1))
            # For binary classifier, pick the contribution for the default class
            if isinstance(shap_values, list):
                vals = shap_values[-1][0]
            else:
                vals = shap_values[0]
            pairs = list(zip(feature_names, vals))
            pairs.sort(key=lambda t: abs(t[1]), reverse=True)
            return pairs[:k]
        except Exception:
            pass

    # Fallback: use coefficient * value as a simple impact proxy if model has coef_
    vals = []
    coef = getattr(model, "coef_", np.zeros(len(feature_names)))
    for name, v, c in zip(feature_names, x, coef):
        vals.append((name, float(v * c)))
    vals.sort(key=lambda t: abs(t[1]), reverse=True)
    return vals[:k]


def friendly_label(name: str) -> str:
    mapping = {
        "credit_amount": "Credit Amount ($)",
        "annual_income": "Annual Income ($)",
        "debt_to_income": "Debt-to-Income (%)",
        "credit_history_years": "Credit History (years)",
        "age": "Age",
        "children_count": "Number of Children",
        "days_registration": "Days Since Registration",
        "own_car_age": "Car Ownership (years)",
        "living_apartments_mode": "Apartment Living (flag)",
        "flag_document_8": "Document Flag 8 (flag)",
        "gender_male": "Gender: Male",
        "contract_cash_loans": "Contract: Cash Loans",
        "owns_car": "Owns Car",
        "owns_realty": "Owns Real Estate",
        "live_city_not_work_city": "Live City ≠ Work City",
        "education_level": "Education Level (ordinal)",
        "marital_status": "Marital Status (ordinal)",
        "employment_type": "Employment Type (ordinal)",
    }
    return mapping.get(name, name)


def recs_from_impacts(impacts: List[Tuple[str, float]]) -> List[str]:
    tips = []
    for feat, val in impacts:
        if feat == "debt_to_income" and val > 0:
            tips.append("Lower your debt-to-income ratio (reduce monthly debt or increase reported income).")
        elif feat == "credit_history_years" and val < 0:
            tips.append("Longer credit history helps—keep accounts open and in good standing.")
        elif feat == "annual_income" and val < 0:
            tips.append("Stable or higher income lowers risk—include verifiable income sources.")
        elif feat == "credit_amount" and val > 0:
            tips.append("Consider a smaller credit amount or add collateral to reduce risk.")
        elif feat == "live_city_not_work_city" and val > 0:
            tips.append("Provide proof of residence/work consistency if possible to mitigate address risk.")
        elif feat == "owns_realty" and val < 0:
            tips.append("Real estate ownership supports approval—include property documents if applicable.")
        elif feat == "owns_car" and val < 0:
            tips.append("Vehicle ownership can help—ensure ownership documents are clear and up to date.")
    # De-duplicate while preserving order
    seen = set()
    deduped = []
    for t in tips:
        if t not in seen:
            deduped.append(t)
            seen.add(t)
    return deduped[:5]


# ---------------------------
# UI
# ---------------------------
st.set_page_config(page_title="SafeLend Credit Risk", page_icon="💳", layout="centered")

# Add custom CSS for better text visibility
st.markdown("""
<style>
    .main .block-container {
        color: #ffffff;
        background-color: #1e1e1e;
    }
    .stMarkdown {
        color: #ffffff;
    }
    .stDataFrame {
        color: #ffffff;
    }
    .stText {
        color: #ffffff;
    }
    .stSelectbox label, .stNumberInput label, .stTextInput label {
        color: #ffffff !important;
    }
    .stMarkdown p {
        color: #ffffff !important;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
        color: #ffffff !important;
    }
    .stForm {
        background-color: #2d2d2d;
        padding: 20px;
        border-radius: 10px;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    /* Keep the risk assessment section with black text on white background */
    .risk-assessment {
        color: #000000 !important;
        background-color: #f5f7f9 !important;
    }
</style>
""", unsafe_allow_html=True)

st.title(APP_NAME)
st.caption(BRAND_TAGLINE)

with st.expander("About this demo", expanded=False):
    st.markdown(
        """
        This portfolio demo estimates a **default probability** for a hypothetical credit applicant and
        explains the **top factors** that affected the decision. Replace the mock model with your trained
        model to use real predictions. Nothing here is financial advice.
        """
    )

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    threshold = st.slider("Approval Threshold (probability)", 0.1, 0.9, 0.5, 0.01)
    st.write("Applications with default probability below the threshold are **approved**.")
    st.divider()
    st.subheader("Data I/O")
    uploaded = st.file_uploader("Load profile (JSON)", type=["json"], help="Must contain keys matching the form fields.")

# ----------
# Form inputs
# ----------
with st.form("applicant_form"):
    st.subheader("Applicant Profile")
    col1, col2 = st.columns(2)
    with col1:
        credit_amount = st.number_input("Credit Amount ($)", min_value=0, step=1000, value=20000)
        annual_income = st.number_input("Annual Income ($)", min_value=0, step=1000, value=60000)
        dti = st.number_input("Debt-to-Income (%)", min_value=0.0, max_value=100.0, step=0.1, value=30.0)
        credit_hist = st.number_input("Credit History (years)", min_value=0.0, max_value=60.0, step=0.5, value=5.0)
        age = st.number_input("Age", min_value=18, max_value=90, step=1, value=28)
        children = st.number_input("Number of Children", min_value=0, max_value=10, step=1, value=0)
    with col2:
        days_reg = st.number_input("Days Since Registration", min_value=0, max_value=100000, step=10, value=1200)
        car_age = st.number_input("Car Ownership (years)", min_value=0.0, max_value=40.0, step=0.5, value=2.0)
        live_apart = st.selectbox("Apartment Living (flag)", ["No", "Yes"], index=0)
        flag_doc8 = st.selectbox("Document Flag 8 (flag)", ["No", "Yes"], index=0)
        gender = st.selectbox("Gender", ["Female", "Male"])  # demo 0/1
        contract_type = st.selectbox("Contract Type", ["Revolving", "Cash loans"])  # demo 0/1

    col3, col4 = st.columns(2)
    with col3:
        owns_car = st.selectbox("Owns Car", ["No", "Yes"], index=1)
        owns_realty = st.selectbox("Owns Real Estate", ["No", "Yes"], index=0)
        live_not_work = st.selectbox("Live City ≠ Work City", ["No", "Yes"], index=0)
    with col4:
        education = st.selectbox("Education Level", ["Other", "Secondary", "Higher", "Postgrad"], index=2)
        marital = st.selectbox("Marital Status", ["Other", "Single", "Married", "Divorced"], index=2)
        employment = st.selectbox("Employment Type", ["Other", "Salaried", "Self-Employed", "Contract"], index=1)

    submitted = st.form_submit_button("Assess Credit Risk", use_container_width=True)

# Load profile from JSON if provided
if uploaded is not None:
    try:
        payload = json.load(uploaded)
        credit_amount = payload.get("credit_amount", credit_amount)
        annual_income = payload.get("annual_income", annual_income)
        dti = payload.get("debt_to_income", dti)
        credit_hist = payload.get("credit_history_years", credit_hist)
        age = payload.get("age", age)
        children = payload.get("children_count", children)
        days_reg = payload.get("days_registration", days_reg)
        car_age = payload.get("own_car_age", car_age)
        live_apart = "Yes" if payload.get("living_apartments_mode", 0) == 1 else "No"
        flag_doc8 = "Yes" if payload.get("flag_document_8", 0) == 1 else "No"
        gender = "Male" if payload.get("gender_male", 0) == 1 else "Female"
        contract_type = "Cash loans" if payload.get("contract_cash_loans", 0) == 1 else "Revolving"
        owns_car = "Yes" if payload.get("owns_car", 0) == 1 else "No"
        owns_realty = "Yes" if payload.get("owns_realty", 0) == 1 else "No"
        live_not_work = "Yes" if payload.get("live_city_not_work_city", 0) == 1 else "No"
        education = ["Other", "Secondary", "Higher", "Postgrad"][payload.get("education_level", 2)]
        marital = ["Other", "Single", "Married", "Divorced"][payload.get("marital_status", 2)]
        employment = ["Other", "Salaried", "Self-Employed", "Contract"][payload.get("employment_type", 1)]
        st.success("Loaded profile from JSON.")
    except Exception as e:
        st.warning(f"Could not load JSON: {e}")

# Encode features for the model input
x_demo = {
    "credit_amount": float(credit_amount),
    "annual_income": float(annual_income),
    "debt_to_income": float(dti),
    "credit_history_years": float(credit_hist),
    "age": float(age),
    "children_count": int(children),
    "days_registration": int(days_reg),
    "own_car_age": float(car_age),
    "living_apartments_mode": 1 if live_apart == "Yes" else 0,
    "flag_document_8": 1 if flag_doc8 == "Yes" else 0,
    "gender_male": 1 if gender == "Male" else 0,
    "contract_cash_loans": 1 if contract_type == "Cash loans" else 0,
    "owns_car": 1 if owns_car == "Yes" else 0,
    "owns_realty": 1 if owns_realty == "Yes" else 0,
    "live_city_not_work_city": 1 if live_not_work == "Yes" else 0,
    "education_level": ["Other", "Secondary", "Higher", "Postgrad"].index(education),
    "marital_status": ["Other", "Single", "Married", "Divorced"].index(marital),
    "employment_type": ["Other", "Salaried", "Self-Employed", "Contract"].index(employment),
}

X = np.array([x_demo[f] for f in FEATURES], dtype=float).reshape(1, -1)

model, explainer = load_model_and_explainer()
proba = model.predict_proba(X)[0, 1]

dec = decision_from_prob(proba, threshold)

if submitted:
    status = "APPROVED" if dec.approved else "REVIEW / DECLINE"
    color = "#16a34a" if dec.approved else "#dc2626"

    st.markdown(
        f"""
        <div style='padding:16px;border-radius:12px;background:transparent;border:2px solid {color};'>
            <h3 style='margin:0;color:{color};font-size:24px;font-weight:bold;'>{status}</h3>
            <p style='margin:6px 0;color:#ffffff;font-size:16px;'>
                <b>Default Probability:</b> {dec.default_prob:.1%}<br/>
                <b>Risk Level:</b> {dec.risk_band}<br/>
                <b>Threshold:</b> {dec.threshold:.1%}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Explainability
    st.subheader("Top Risk Factors")
    impacts = top_feature_impacts(X.flatten(), FEATURES, model, explainer, k=6)
    imp_df = pd.DataFrame([
        {"Feature": friendly_label(n), "Impact": v} for n, v in impacts
    ])
    st.dataframe(imp_df, width='stretch', hide_index=True)

    # Simple impact bar chart (positive = increases risk)
    try:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(6, 3.5))
        order = np.argsort(np.abs(imp_df["Impact"].values))[::-1]
        plt.barh(imp_df["Feature"].values[order], imp_df["Impact"].values[order])
        plt.xlabel("Impact on Risk (±)")
        plt.tight_layout()
        st.pyplot(fig)
    except Exception:
        pass

    # Plain-language tips
    st.subheader("Recommendations (What could improve approval?)")
    tips = recs_from_impacts(impacts)
    if tips:
        for t in tips:
            st.markdown(f"- {t}")
    else:
        st.markdown("- Maintain strong repayment history and stable income.")

    # Downloadable payload
    st.subheader("Reproducibility")
    st.caption("Download the exact JSON used for this assessment.")
    st.download_button(
        "Download JSON payload",
        data=json.dumps(x_demo, indent=2),
        file_name="safelend_profile.json",
        mime="application/json",
    )

st.divider()
st.caption("© 2025 SafeLend Demo • Educational use only • No financial advice or real lending decisions.")
