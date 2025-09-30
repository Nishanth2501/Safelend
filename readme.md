# SafeLend Credit Risk Assessment

This project is a simple and clear demonstration of how machine learning can be used to simulate a credit risk assessment system. It is built with Streamlit and shows the end-to-end flow: taking in user details, making a prediction, explaining the decision, and giving friendly recommendations.

This is only a demo project created for learning purposes. It does not make real financial or lending decisions.

---

## What the app does

* Lets a user fill in details like income, loan amount, debt ratio, age, and other factors.
* Predicts the chance (probability) that the user will default on a loan.
* Decides whether the loan would be approved or declined based on a threshold that you can adjust.
* Explains the top features that increased or decreased the risk.
* Provides plain language suggestions, such as lowering debt-to-income or showing proof of stable income.
* Shows a simple bar chart to visualize feature impacts.
* Allows downloading the exact data (JSON) used in the assessment.

---

## How it works

The app comes with a mock model that gives realistic results. If you want, you can replace it with your own trained model. If you also provide a SHAP explainer, the app will show more advanced explanations.

---

## Requirements

* Python 3.9 or later
* Streamlit
* NumPy
* Pandas
* Matplotlib
* SHAP (optional, for explainability)
* Joblib (if you want to load your own model)

Install dependencies with:

```bash
pip install -r requirements.txt
```

---

## How to run

1. Clone this repository:

```bash
git clone https://github.com/your-username/SafeLend-Credit-Risk.git
cd SafeLend-Credit-Risk
```

2. Run the app:

```bash
streamlit run safelend_app.py
```

3. Open the provided local URL in your browser to use the app.

---

## Project structure

```
SafeLend-Credit-Risk/
├── safelend_app.py        # Main app file
├── models/
│   ├── model.pkl          # (Optional) your own trained model
│   └── explainer.pkl      # (Optional) SHAP explainer
├── docs/
│   └── screenshots/       # Screenshots for the README
├── requirements.txt
└── README.md
```

---

## Example output

* Default Probability: 20%
* Risk Band: Low
* Decision: Approved
* Top factors: debt-to-income ratio increased risk, longer credit history reduced risk.

---

## Ideas to extend

* Add a backend API with FastAPI for a more production-like setup.
* Store assessments in a database.
* Deploy on Render, Heroku, or AWS.
* Train and plug in a stronger model like XGBoost or Random Forest.
* Add more user-friendly charts or visual explanations.

---

This project is meant to show how data science can be applied in a realistic way, and how to communicate model decisions clearly to end users.
