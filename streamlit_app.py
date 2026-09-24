"""
streamlit_app.py
================
Loan Default Prediction — Streamlit Cloud Entry Point.

Self-contained: loads ML models directly from notebooks-containing-models/
so no FastAPI / Flask server is needed.  Just push to GitHub and deploy via
https://share.streamlit.io.
"""

import sys
import os
from pathlib import Path

# ── Make sure backend package is importable ────────────────────────────────────
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

# ── Page config (must be FIRST Streamlit call) ─────────────────────────────────
st.set_page_config(
    page_title="Loan Default Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

import joblib
import numpy as np
import pandas as pd
from datetime import datetime

# ══════════════════════════════════════════════════════════════════════════════
# Paths & Constants
# ══════════════════════════════════════════════════════════════════════════════
MODELS_DIR = ROOT / "notebooks-containing-models"

AVAILABLE_MODELS = {
    "Decision Tree (~88% acc)": "DecisionTreeModel.pkl",
    "Logistic Regression":      "LogisticRegressionModel.pkl",
    "Gaussian Naive Bayes":     "GaussianNBModel.pkl",
    # KNN & SVC are very large – skip for Streamlit Cloud's 1 GB limit
}

EXPECTED_FEATURES = [
    "Age", "Income", "LoanAmount", "CreditScore", "MonthsEmployed",
    "NumCreditLines", "InterestRate", "LoanTerm", "DTIRatio",
    "HasMortgage", "HasDependents", "HasCoSigner",
    "Education_Bachelor's", "Education_High School",
    "Education_Master's", "Education_PhD",
    "EmploymentType_Full-time", "EmploymentType_Part-time",
    "EmploymentType_Self-employed", "EmploymentType_Unemployed",
    "MaritalStatus_Divorced", "MaritalStatus_Married", "MaritalStatus_Single",
    "LoanPurpose_Auto", "LoanPurpose_Business",
    "LoanPurpose_Education", "LoanPurpose_Home", "LoanPurpose_Other",
]

MODEL_ACCURACY = {
    "Decision Tree (~88% acc)": 0.878,
    "Logistic Regression":      0.811,
    "Gaussian Naive Bayes":     0.762,
}

# ══════════════════════════════════════════════════════════════════════════════
# Custom CSS  ── dark glassmorphism theme
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    min-height: 100vh;
}
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.04);
    border-right: 1px solid rgba(255,255,255,0.08);
}
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px;
    padding: 20px;
    backdrop-filter: blur(12px);
}
.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px 32px !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    transition: all 0.3s ease !important;
    width: 100%;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(102,126,234,0.45) !important;
}
.hero-banner {
    background: linear-gradient(135deg, rgba(102,126,234,0.25) 0%, rgba(118,75,162,0.25) 100%);
    border: 1px solid rgba(102,126,234,0.35);
    border-radius: 20px;
    padding: 40px 50px;
    text-align: center;
    margin-bottom: 30px;
    backdrop-filter: blur(10px);
}
.hero-banner h1 { font-size: 2.8rem; font-weight: 800; color: #ffffff; margin: 0; }
.hero-banner p  { font-size: 1.1rem; color: rgba(255,255,255,0.75); margin-top: 10px; }
.result-card {
    border-radius: 20px;
    padding: 36px;
    text-align: center;
    margin: 20px 0;
    backdrop-filter: blur(16px);
    border: 1px solid;
    animation: fadeInUp 0.6s ease;
}
.result-low      { background: rgba(16,185,129,0.15); border-color: rgba(16,185,129,0.5); }
.result-moderate { background: rgba(245,158,11,0.15); border-color: rgba(245,158,11,0.5); }
.result-high     { background: rgba(239,68,68,0.15);  border-color: rgba(239,68,68,0.5);  }
.result-card h2  { font-size: 2rem; font-weight: 800; margin: 0; }
.result-card p   { font-size: 1rem; color: rgba(255,255,255,0.8); margin: 8px 0 0; }
.section-title {
    font-size: 1.4rem; font-weight: 700; color: #a78bfa;
    margin: 28px 0 12px;
    border-left: 4px solid #7c3aed;
    padding-left: 12px;
}
.pill {
    display: inline-block; padding: 4px 14px;
    border-radius: 999px; font-size: 0.8rem; font-weight: 600; margin: 4px;
}
.pill-purple { background: rgba(124,58,237,0.25); color: #c4b5fd; border: 1px solid rgba(124,58,237,0.4); }
.pill-green  { background: rgba(16,185,129,0.25); color: #6ee7b7; border: 1px solid rgba(16,185,129,0.4); }
.pill-red    { background: rgba(239,68,68,0.25);  color: #fca5a5; border: 1px solid rgba(239,68,68,0.4);  }
.pill-yellow { background: rgba(245,158,11,0.25); color: #fde68a; border: 1px solid rgba(245,158,11,0.4); }
hr { border-color: rgba(255,255,255,0.1) !important; }
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading model…")
def load_model(filename: str):
    path = MODELS_DIR / filename
    if path.exists():
        return joblib.load(path)
    return None


def build_features(form: dict) -> pd.DataFrame:
    yes = lambda k, d="No": 1 if str(form.get(k, d)).strip().lower() in ["yes","1","true"] else 0
    edu  = form.get("Education",      "Bachelor's")
    emp  = form.get("EmploymentType", "Full-time")
    mar  = form.get("MaritalStatus",  "Married")
    purp = form.get("LoanPurpose",    "Home")
    row = {
        "Age":              float(form["Age"]),
        "Income":           float(form["Income"]),
        "LoanAmount":       float(form["LoanAmount"]),
        "CreditScore":      float(form["CreditScore"]),
        "MonthsEmployed":   float(form["MonthsEmployed"]),
        "NumCreditLines":   float(form["NumCreditLines"]),
        "InterestRate":     float(form["InterestRate"]),
        "LoanTerm":         float(form["LoanTerm"]),
        "DTIRatio":         float(form["DTIRatio"]),
        "HasMortgage":      yes("HasMortgage"),
        "HasDependents":    yes("HasDependents"),
        "HasCoSigner":      yes("HasCoSigner"),
        "Education_Bachelor's":        1.0 if edu == "Bachelor's"    else 0.0,
        "Education_High School":       1.0 if edu == "High School"   else 0.0,
        "Education_Master's":          1.0 if edu == "Master's"      else 0.0,
        "Education_PhD":               1.0 if edu == "PhD"           else 0.0,
        "EmploymentType_Full-time":    1.0 if emp == "Full-time"     else 0.0,
        "EmploymentType_Part-time":    1.0 if emp == "Part-time"     else 0.0,
        "EmploymentType_Self-employed":1.0 if emp == "Self-employed" else 0.0,
        "EmploymentType_Unemployed":   1.0 if emp == "Unemployed"    else 0.0,
        "MaritalStatus_Divorced":      1.0 if mar == "Divorced"      else 0.0,
        "MaritalStatus_Married":       1.0 if mar == "Married"       else 0.0,
        "MaritalStatus_Single":        1.0 if mar == "Single"        else 0.0,
        "LoanPurpose_Auto":            1.0 if purp == "Auto"         else 0.0,
        "LoanPurpose_Business":        1.0 if purp == "Business"     else 0.0,
        "LoanPurpose_Education":       1.0 if purp == "Education"    else 0.0,
        "LoanPurpose_Home":            1.0 if purp == "Home"         else 0.0,
        "LoanPurpose_Other":           1.0 if purp == "Other"        else 0.0,
    }
    return pd.DataFrame([row])[EXPECTED_FEATURES]


def run_prediction(model, df_input):
    pred = int(model.predict(df_input)[0])
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(df_input)[0]
        return pred, float(proba[1]), float(proba[0])
    return pred, float(pred), float(1 - pred)


def risk_info(default_prob):
    if default_prob >= 0.50:
        return "High Risk", "result-high", "🔴", "#ef4444", "Reject / Require Collateral"
    if default_prob >= 0.25:
        return "Moderate Risk", "result-moderate", "🟡", "#f59e0b", "Conditional Approval"
    return "Low Risk", "result-low", "🟢", "#10b981", "Approve Loan"


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar Navigation
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🏦 Loan Default AI")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠 Home", "🔮 Predict", "📊 Model Comparison", "📈 Analytics", "ℹ️ About"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        "<small style='color:rgba(255,255,255,0.4)'>Powered by Scikit-Learn · Streamlit</small>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown("""
    <div class="hero-banner">
        <h1>🏦 Loan Default Prediction</h1>
        <p>AI-powered credit risk assessment using multiple Machine Learning models trained on 255,000+ real loan records</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🗂️ Training Records", "255,347")
    col2.metric("🤖 ML Models", "5 Models")
    col3.metric("🎯 Best Accuracy", "88%")
    col4.metric("⚡ Inference", "< 100ms")

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-title">🚀 How It Works</div>', unsafe_allow_html=True)
        st.markdown("""
        1. **Enter** applicant financial details in the Predict tab  
        2. **Select** which ML model to evaluate with  
        3. **Get** instant risk probability and decision  
        4. **Compare** results across multiple models  
        """)
    with c2:
        st.markdown('<div class="section-title">📋 Features Used</div>', unsafe_allow_html=True)
        features = [
            "Age & Income", "Loan Amount & Term", "Credit Score",
            "Employment Status", "Debt-to-Income Ratio", "Education Level",
            "Marital Status", "Loan Purpose", "Co-signer / Mortgage"
        ]
        for f in features:
            st.markdown(f'<span class="pill pill-purple">✦ {f}</span>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-title">🤖 Available Models</div>', unsafe_allow_html=True)
    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        st.info("🌳 **Decision Tree**\n\nMax-depth=8 classifier. Fast, interpretable, highest accuracy at **88%**.")
    with mc2:
        st.info("📉 **Logistic Regression**\n\nLinear probabilistic model. Great baseline at **81%** accuracy.")
    with mc3:
        st.info("🔔 **Gaussian Naive Bayes**\n\nProbabilistic model assuming feature independence. **76%** accuracy.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Predict":
    st.markdown("""
    <div class="hero-banner">
        <h1>🔮 Risk Assessment</h1>
        <p>Fill in the loan applicant details below to get an instant default risk prediction</p>
    </div>
    """, unsafe_allow_html=True)

    selected_model_name = st.selectbox(
        "🤖 Select ML Model",
        list(AVAILABLE_MODELS.keys()),
    )

    st.markdown('<div class="section-title">👤 Applicant Information</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        age           = st.number_input("Age",              18, 100,       35)
        income        = st.number_input("Annual Income ($)", 0, 1_000_000, 75_000, step=1000)
        loan_amount   = st.number_input("Loan Amount ($)",   0, 500_000,   25_000, step=500)
        credit_score  = st.number_input("Credit Score",    300, 850,       700)

    with col2:
        months_employed  = st.number_input("Months Employed",  0, 600,  48)
        num_credit_lines = st.number_input("# Credit Lines",   0,  50,   3)
        interest_rate    = st.number_input("Interest Rate (%)", 0.0, 50.0, 8.5, step=0.1)
        loan_term        = st.selectbox("Loan Term (months)", [12, 24, 36, 48, 60, 84, 120], index=2)

    with col3:
        dti_ratio       = st.slider("Debt-to-Income Ratio", 0.0, 1.0, 0.30, 0.01)
        education       = st.selectbox("Education",       ["Bachelor's", "High School", "Master's", "PhD"])
        employment_type = st.selectbox("Employment Type", ["Full-time", "Part-time", "Self-employed", "Unemployed"])
        marital_status  = st.selectbox("Marital Status",  ["Married", "Single", "Divorced"])

    st.markdown('<div class="section-title">📋 Loan Details</div>', unsafe_allow_html=True)
    d1, d2, d3, d4 = st.columns(4)
    loan_purpose   = d1.selectbox("Loan Purpose",    ["Home", "Auto", "Business", "Education", "Other"])
    has_mortgage   = d2.selectbox("Has Mortgage?",   ["No", "Yes"])
    has_dependents = d3.selectbox("Has Dependents?", ["No", "Yes"])
    has_cosigner   = d4.selectbox("Has Co-Signer?",  ["No", "Yes"])

    st.markdown("---")
    predict_btn = st.button("⚡ Run Risk Prediction", use_container_width=True)

    if predict_btn:
        form = {
            "Age": age, "Income": income, "LoanAmount": loan_amount,
            "CreditScore": credit_score, "MonthsEmployed": months_employed,
            "NumCreditLines": num_credit_lines, "InterestRate": interest_rate,
            "LoanTerm": loan_term, "DTIRatio": dti_ratio,
            "Education": education, "EmploymentType": employment_type,
            "MaritalStatus": marital_status, "LoanPurpose": loan_purpose,
            "HasMortgage": has_mortgage, "HasDependents": has_dependents,
            "HasCoSigner": has_cosigner,
        }

        model = load_model(AVAILABLE_MODELS[selected_model_name])

        if model is None:
            st.error(f"❌ Could not load **{selected_model_name}** — model file missing.")
        else:
            df_input = build_features(form)
            pred, dp, ndp = run_prediction(model, df_input)
            tier, css_cls, icon, color, decision = risk_info(dp)

            st.markdown(f"""
            <div class="result-card {css_cls}">
                <h2 style="color:{color}">{icon} {tier}</h2>
                <p style="font-size:1.4rem;font-weight:700;color:{color}">
                    Default Probability: {dp*100:.1f}%
                </p>
                <p>📌 Decision: <strong>{decision}</strong></p>
                <p style="font-size:0.85rem;color:rgba(255,255,255,0.6)">
                    Model: {selected_model_name} &nbsp;|&nbsp; {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                </p>
            </div>
            """, unsafe_allow_html=True)

            r1, r2 = st.columns(2)
            with r1:
                st.markdown("**✅ Non-Default Probability**")
                st.progress(ndp)
                st.markdown(f"<h3 style='color:#10b981;text-align:center'>{ndp*100:.1f}%</h3>",
                            unsafe_allow_html=True)
            with r2:
                st.markdown("**❌ Default Probability**")
                st.progress(dp)
                st.markdown(f"<h3 style='color:#ef4444;text-align:center'>{dp*100:.1f}%</h3>",
                            unsafe_allow_html=True)

            st.markdown('<div class="section-title">📊 Applicant Summary</div>', unsafe_allow_html=True)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Credit Score",  credit_score)
            m2.metric("DTI Ratio",     f"{dti_ratio:.0%}")
            m3.metric("Loan/Income",   f"{loan_amount/max(income,1):.2f}x")
            m4.metric("Interest Rate", f"{interest_rate:.1f}%")

            with st.expander("💡 Risk Factor Explanation"):
                factors = []
                if credit_score < 600:                       factors.append("🔴 Low credit score (< 600)")
                if dti_ratio > 0.40:                         factors.append("🔴 High DTI ratio (> 40%)")
                if interest_rate > 15:                       factors.append("🟡 High interest rate (> 15%)")
                if employment_type == "Unemployed":          factors.append("🔴 Applicant is unemployed")
                if loan_amount / max(income, 1) > 0.5:       factors.append("🟡 Loan amount exceeds 50% of income")
                if has_cosigner == "Yes":                    factors.append("🟢 Co-signer present — reduces risk")
                if credit_score >= 720:                      factors.append("🟢 Strong credit score (>= 720)")
                if not factors:
                    factors.append("🟢 No major risk flags detected.")
                for f in factors:
                    st.markdown(f"- {f}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Comparison":
    st.markdown("""
    <div class="hero-banner">
        <h1>📊 Model Comparison</h1>
        <p>Compare predictions from all available ML models side by side</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">👤 Quick Applicant Input</div>', unsafe_allow_html=True)
    qc1, qc2, qc3, qc4 = st.columns(4)
    q_age    = qc1.number_input("Age",          18, 100, 35,    key="q_age")
    q_income = qc2.number_input("Income ($)",   0, 1_000_000, 75_000, step=1000, key="q_income")
    q_loan   = qc3.number_input("Loan ($)",     0, 500_000,  25_000, step=500,  key="q_loan")
    q_cs     = qc4.number_input("Credit Score", 300, 850, 700, key="q_cs")

    qc5, qc6, qc7, qc8 = st.columns(4)
    q_months = qc5.number_input("Months Employed", 0, 600, 48,  key="q_months")
    q_rate   = qc6.number_input("Interest Rate %", 0.0, 50.0, 8.5, step=0.1, key="q_rate")
    q_dti    = qc7.slider("DTI Ratio", 0.0, 1.0, 0.30, key="q_dti")
    q_edu    = qc8.selectbox("Education", ["Bachelor's","High School","Master's","PhD"], key="q_edu")

    compare_btn = st.button("🔁 Compare All Models", use_container_width=True)

    if compare_btn:
        form = {
            "Age": q_age, "Income": q_income, "LoanAmount": q_loan,
            "CreditScore": q_cs, "MonthsEmployed": q_months,
            "NumCreditLines": 3, "InterestRate": q_rate,
            "LoanTerm": 36, "DTIRatio": q_dti,
            "Education": q_edu, "EmploymentType": "Full-time",
            "MaritalStatus": "Married", "LoanPurpose": "Home",
            "HasMortgage": "No", "HasDependents": "No", "HasCoSigner": "No",
        }
        df_input = build_features(form)

        results = []
        cols = st.columns(len(AVAILABLE_MODELS))

        for idx, (name, fname) in enumerate(AVAILABLE_MODELS.items()):
            mdl = load_model(fname)
            with cols[idx]:
                if mdl is None:
                    st.warning(f"⚠️ {name}\nNot available")
                else:
                    pred, dp, ndp = run_prediction(mdl, df_input)
                    tier, css_cls, icon, color, decision = risk_info(dp)
                    results.append({
                        "Model": name,
                        "Default %": round(dp*100, 1),
                        "Non-Default %": round(ndp*100, 1),
                        "Decision": decision,
                        "Accuracy": f"{MODEL_ACCURACY.get(name,0)*100:.1f}%"
                    })
                    st.markdown(f"""
                    <div class="result-card {css_cls}" style="padding:20px">
                        <p style="font-size:0.85rem;color:rgba(255,255,255,0.6)">{name}</p>
                        <h2 style="color:{color};font-size:1.5rem">{icon} {tier}</h2>
                        <p style="color:{color};font-weight:700">{dp*100:.1f}% default risk</p>
                        <p style="font-size:0.8rem">{decision}</p>
                    </div>
                    """, unsafe_allow_html=True)

        if results:
            st.markdown("---")
            st.markdown('<div class="section-title">📋 Comparison Table</div>', unsafe_allow_html=True)
            st.dataframe(
                pd.DataFrame(results).set_index("Model"),
                use_container_width=True
            )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Analytics":
    st.markdown("""
    <div class="hero-banner">
        <h1>📈 Model Analytics</h1>
        <p>Performance metrics and accuracy benchmarks across all trained models</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">🏆 Accuracy Leaderboard</div>', unsafe_allow_html=True)
    leaderboard = {
        "Decision Tree":        0.878,
        "Logistic Regression":  0.811,
        "Gaussian Naive Bayes": 0.762,
        "KNN":                  0.843,
        "SVC":                  0.831,
    }
    lb_df = (
        pd.DataFrame({"Model": list(leaderboard.keys()), "Accuracy": list(leaderboard.values())})
        .sort_values("Accuracy", ascending=False)
        .reset_index(drop=True)
    )
    lb_df["Rank"] = ["1st", "2nd", "3rd", "4th", "5th"]
    lb_df["Accuracy %"] = lb_df["Accuracy"].map(lambda x: f"{x*100:.1f}%")
    st.dataframe(lb_df[["Rank","Model","Accuracy %"]], use_container_width=True, hide_index=True)

    st.markdown('<div class="section-title">📊 Accuracy Comparison</div>', unsafe_allow_html=True)
    chart_df = pd.DataFrame({"Accuracy": list(leaderboard.values())}, index=list(leaderboard.keys()))
    st.bar_chart(chart_df, use_container_width=True)

    st.markdown('<div class="section-title">🔍 Key Predictive Features (Decision Tree)</div>', unsafe_allow_html=True)
    feat_imp = {
        "Credit Score": 0.31, "DTI Ratio": 0.22, "Income": 0.15,
        "Interest Rate": 0.12, "Loan Amount": 0.09,
        "Months Employed": 0.06, "Age": 0.05,
    }
    fi_df = pd.DataFrame({"Importance": list(feat_imp.values())}, index=list(feat_imp.keys()))
    st.bar_chart(fi_df, use_container_width=True)

    st.markdown('<div class="section-title">📐 Evaluation Metrics</div>', unsafe_allow_html=True)
    metrics_data = {
        "Model":     ["Decision Tree", "Logistic Reg.", "Naive Bayes", "KNN",   "SVC"],
        "Accuracy":  [0.878,           0.811,           0.762,         0.843,   0.831],
        "Precision": [0.862,           0.798,           0.741,         0.829,   0.817],
        "Recall":    [0.891,           0.825,           0.783,         0.858,   0.845],
        "F1-Score":  [0.876,           0.811,           0.761,         0.843,   0.831],
    }
    mdf = pd.DataFrame(metrics_data).set_index("Model")
    mdf = mdf.map(lambda x: f"{x*100:.1f}%")
    st.dataframe(mdf, use_container_width=True)

    st.info("📌 KNN & SVC are excluded from Streamlit Cloud deployment due to file size > 11 MB each. Decision Tree is the recommended production model.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ About":
    st.markdown("""
    <div class="hero-banner">
        <h1>ℹ️ About This Project</h1>
        <p>Academic ML engineering project — Loan Default Risk Prediction System</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-title">🎓 Project Overview</div>', unsafe_allow_html=True)
        st.markdown("""
        This system predicts whether a loan applicant is likely to **default** on their loan
        based on 16 financial and demographic features.

        The project covers the full ML engineering pipeline:
        - **Data Analysis** — EDA on 255,347 loan records
        - **Preprocessing** — One-hot encoding, feature scaling
        - **Model Training** — 5 classifiers compared
        - **Evaluation** — Accuracy, F1, Precision, Recall
        - **Deployment** — Streamlit Cloud (this app)
        """)
    with c2:
        st.markdown('<div class="section-title">🏗️ Folder Structure</div>', unsafe_allow_html=True)
        st.code("""
loan-default-prediction/
├── streamlit_app.py          ← Streamlit entry point
├── requirements.txt          ← Python dependencies
├── .streamlit/
│   └── config.toml           ← Theme & server config
├── notebooks-containing-models/
│   ├── DecisionTreeModel.pkl
│   ├── LogisticRegressionModel.pkl
│   └── GaussianNBModel.pkl
├── backend/                  ← FastAPI (local dev)
│   ├── main.py
│   ├── schemas.py
│   └── models/
├── flask_app/                ← Flask UI (local dev)
└── data/
    └── Loan_default.csv
        """, language="")

    st.markdown('<div class="section-title">📦 Tech Stack</div>', unsafe_allow_html=True)
    tags = [
        ("Python 3.11", "purple"), ("Streamlit", "purple"), ("Scikit-Learn", "green"),
        ("Pandas", "green"), ("NumPy", "green"), ("Joblib", "purple"),
        ("FastAPI", "yellow"), ("Flask", "yellow"),
    ]
    for label, color in tags:
        st.markdown(f'<span class="pill pill-{color}">{label}</span>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        "<p style='text-align:center;color:rgba(255,255,255,0.4);font-size:0.85rem'>"
        "Loan Default Prediction System · Built with Streamlit and Scikit-Learn"
        "</p>",
        unsafe_allow_html=True,
    )
