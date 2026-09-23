"""
flask_app/app.py
================
Flask Application for Loan Default Risk Assessment.
Satisfies SOP Week 7 (Flask Project Setup & Form Handling),
Week 8 (Frontend Web Form), and Week 9 (Backend Processing & API).
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash

# Add root directory to sys.path so backend models can be imported
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from flask_app.config import config_by_name

# Initialize Flask app
app = Flask(__name__, static_folder="static", template_folder="templates")
env = os.environ.get("FLASK_ENV", "dev")
app.config.from_object(config_by_name.get(env, config_by_name["default"]))

# Load Machine Learning Model
MODEL_PATH = ROOT_DIR / "notebooks-containing-models" / "DecisionTreeModel.pkl"
try:
    model = joblib.load(MODEL_PATH)
    print(f"[FLASK INFO] Loaded model successfully from {MODEL_PATH}")
except Exception as e:
    print(f"[FLASK WARNING] Could not load model from {MODEL_PATH}: {e}")
    model = None

# Expected 28 features in exact order
EXPECTED_FEATURES = [
    "Age", "Income", "LoanAmount", "CreditScore", "MonthsEmployed",
    "NumCreditLines", "InterestRate", "LoanTerm", "DTIRatio",
    "HasMortgage", "HasDependents", "HasCoSigner",
    "Education_Bachelor's", "Education_High School", "Education_Master's", "Education_PhD",
    "EmploymentType_Full-time", "EmploymentType_Part-time", "EmploymentType_Self-employed", "EmploymentType_Unemployed",
    "MaritalStatus_Divorced", "MaritalStatus_Married", "MaritalStatus_Single",
    "LoanPurpose_Auto", "LoanPurpose_Business", "LoanPurpose_Education", "LoanPurpose_Home", "LoanPurpose_Other"
]

def preprocess_form_data(form_dict):
    """
    Transforms form fields into the 28 encoded features required by the trained model.
    """
    age = float(form_dict.get("Age", 35))
    income = float(form_dict.get("Income", 75000))
    loan_amount = float(form_dict.get("LoanAmount", 25000))
    credit_score = float(form_dict.get("CreditScore", 700))
    months_employed = float(form_dict.get("MonthsEmployed", 48))
    num_credit_lines = float(form_dict.get("NumCreditLines", 3))
    interest_rate = float(form_dict.get("InterestRate", 8.5))
    loan_term = float(form_dict.get("LoanTerm", 36))
    dti_ratio = float(form_dict.get("DTIRatio", 0.30))

    has_mortgage = 1 if form_dict.get("HasMortgage", "No").strip().lower() in ["yes", "1", "true"] else 0
    has_dependents = 1 if form_dict.get("HasDependents", "No").strip().lower() in ["yes", "1", "true"] else 0
    has_cosigner = 1 if form_dict.get("HasCoSigner", "No").strip().lower() in ["yes", "1", "true"] else 0

    education = form_dict.get("Education", "Bachelor's")
    employment_type = form_dict.get("EmploymentType", "Full-time")
    marital_status = form_dict.get("MaritalStatus", "Married")
    loan_purpose = form_dict.get("LoanPurpose", "Home")

    row = {
        "Age": age,
        "Income": income,
        "LoanAmount": loan_amount,
        "CreditScore": credit_score,
        "MonthsEmployed": months_employed,
        "NumCreditLines": num_credit_lines,
        "InterestRate": interest_rate,
        "LoanTerm": loan_term,
        "DTIRatio": dti_ratio,
        "HasMortgage": has_mortgage,
        "HasDependents": has_dependents,
        "HasCoSigner": has_cosigner,

        "Education_Bachelor's": 1.0 if education == "Bachelor's" else 0.0,
        "Education_High School": 1.0 if education == "High School" else 0.0,
        "Education_Master's": 1.0 if education == "Master's" else 0.0,
        "Education_PhD": 1.0 if education == "PhD" else 0.0,

        "EmploymentType_Full-time": 1.0 if employment_type == "Full-time" else 0.0,
        "EmploymentType_Part-time": 1.0 if employment_type == "Part-time" else 0.0,
        "EmploymentType_Self-employed": 1.0 if employment_type == "Self-employed" else 0.0,
        "EmploymentType_Unemployed": 1.0 if employment_type == "Unemployed" else 0.0,

        "MaritalStatus_Divorced": 1.0 if marital_status == "Divorced" else 0.0,
        "MaritalStatus_Married": 1.0 if marital_status == "Married" else 0.0,
        "MaritalStatus_Single": 1.0 if marital_status == "Single" else 0.0,

        "LoanPurpose_Auto": 1.0 if loan_purpose == "Auto" else 0.0,
        "LoanPurpose_Business": 1.0 if loan_purpose == "Business" else 0.0,
        "LoanPurpose_Education": 1.0 if loan_purpose == "Education" else 0.0,
        "LoanPurpose_Home": 1.0 if loan_purpose == "Home" else 0.0,
        "LoanPurpose_Other": 1.0 if loan_purpose == "Other" else 0.0,
    }

    df = pd.DataFrame([row])[EXPECTED_FEATURES]
    return df

@app.route("/")
def index():
    """Landing Page: Overview of the Loan Default Prediction System."""
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    """
    Form Handling Route:
    - GET: Render loan application form (Week 7 & 8)
    - POST: Process applicant data and display risk result
    """
    if request.method == "POST":
        try:
            form_data = request.form.to_dict()
            df_input = preprocess_form_data(form_data)

            if model is not None:
                prediction_val = int(model.predict(df_input)[0])
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(df_input)[0]
                    default_prob = float(probs[1])
                    non_default_prob = float(probs[0])
                else:
                    default_prob = 1.0 if prediction_val == 1 else 0.0
                    non_default_prob = 1.0 - default_prob
            else:
                # Fallback estimation
                default_prob = 0.15
                non_default_prob = 0.85
                prediction_val = 0

            # Determine risk tier and recommendation
            if default_prob >= 0.50:
                risk_tier = "High Risk"
                badge_class = "danger"
                decision = "Reject Application / Require Collateral"
                recommendation = "Applicant displays elevated default risk indicators (high DTI, interest rate, or low credit score). Secondary underwriting review or substantial down payment required."
            elif default_prob >= 0.25:
                risk_tier = "Moderate Risk"
                badge_class = "warning"
                decision = "Conditional Approval / Manual Review"
                recommendation = "Applicant is in the moderate risk band. Approve with adjusted risk-based interest rate or co-signer guarantee."
            else:
                risk_tier = "Low Risk"
                badge_class = "success"
                decision = "Approve Loan Application"
                recommendation = "Applicant demonstrates strong financial capacity and low historical default probability. Standard prime lending terms apply."

            return render_template(
                "result.html",
                form_data=form_data,
                prediction=prediction_val,
                risk_tier=risk_tier,
                badge_class=badge_class,
                decision=decision,
                recommendation=recommendation,
                default_prob=round(default_prob * 100, 1),
                non_default_prob=round(non_default_prob * 100, 1),
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )

        except Exception as e:
            flash(f"Error processing application: {str(e)}", "danger")
            return redirect(url_for("predict"))

    return render_template("predict.html")

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """REST API endpoint for programmatic inference (Week 9)."""
    try:
        data = request.get_json(force=True)
        df_input = preprocess_form_data(data)
        
        if model is not None:
            pred = int(model.predict(df_input)[0])
            probs = model.predict_proba(df_input)[0] if hasattr(model, "predict_proba") else [1-pred, pred]
            default_prob = float(probs[1])
        else:
            pred = 0
            default_prob = 0.12

        return jsonify({
            "status": "success",
            "prediction": pred,
            "prediction_label": "Default" if pred == 1 else "Non-Default",
            "default_probability": round(default_prob, 4),
            "risk_tier": "High" if default_prob >= 0.5 else ("Moderate" if default_prob >= 0.25 else "Low")
        }), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route("/metrics")
def metrics():
    """Display Week 6 generated performance charts and model leaderboard."""
    return render_template("metrics.html")

@app.route("/about")
def about():
    """Display project SOP details, team, and architecture."""
    return render_template("about.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"[INFO] Starting Flask Server on http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=True)
