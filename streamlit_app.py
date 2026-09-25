"""
streamlit_app.py  —  CrediPulse AI
====================================
Exact Streamlit replica of the Flask app (run_flask.py).
Pages: Overview · Risk Evaluator · Model Analytics · Architecture
"""

import sys
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

st.set_page_config(
    page_title="CrediPulse AI — Loan Default Prediction",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

import joblib
import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
MODELS_DIR = ROOT / "notebooks-containing-models"
VIZ_DIR    = ROOT / "flask_app" / "static" / "visualizations"

AVAILABLE_MODELS = {
    "Decision Tree (Depth 8) — Default":  "DecisionTreeModel.pkl",
    "Logistic Regression (L2)":           "LogisticRegressionModel.pkl",
    "Gaussian Naive Bayes":               "GaussianNBModel.pkl",
}

EXPECTED_FEATURES = [
    "Age","Income","LoanAmount","CreditScore","MonthsEmployed",
    "NumCreditLines","InterestRate","LoanTerm","DTIRatio",
    "HasMortgage","HasDependents","HasCoSigner",
    "Education_Bachelor's","Education_High School",
    "Education_Master's","Education_PhD",
    "EmploymentType_Full-time","EmploymentType_Part-time",
    "EmploymentType_Self-employed","EmploymentType_Unemployed",
    "MaritalStatus_Divorced","MaritalStatus_Married","MaritalStatus_Single",
    "LoanPurpose_Auto","LoanPurpose_Business",
    "LoanPurpose_Education","LoanPurpose_Home","LoanPurpose_Other",
]

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

/* ── App background ── */
.stApp {
    background: #0a0a14;
    color: #e2e8f0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0d0d1f !important;
    border-right: 1px solid rgba(255,255,255,0.07);
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

/* ── Top brand bar ── */
.brand-bar {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 18px 0 10px 0;
    border-bottom: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 20px;
}
.brand-bar .icon { font-size: 1.6rem; }
.brand-bar .name {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.25rem; font-weight: 700; color: #fff;
}
.brand-bar .name span { color: #818cf8; }

/* ── Hero ── */
.hero-badge {
    display: inline-block;
    background: rgba(129,140,248,0.15);
    border: 1px solid rgba(129,140,248,0.4);
    color: #a5b4fc; font-size: 0.75rem; font-weight: 700;
    letter-spacing: 0.08em; text-transform: uppercase;
    padding: 5px 14px; border-radius: 999px; margin-bottom: 18px;
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.8rem; font-weight: 800; line-height: 1.15;
    color: #fff; margin-bottom: 16px;
}
.hero-title .grad {
    background: linear-gradient(135deg, #818cf8, #c084fc);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero-sub {
    font-size: 1.05rem; color: #94a3b8;
    max-width: 680px; line-height: 1.7; margin-bottom: 30px;
}

/* ── Stat cards ── */
.stats-grid {
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px;
    margin: 30px 0;
}
.stat-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px; padding: 24px; text-align: center;
    backdrop-filter: blur(10px);
}
.stat-number {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.2rem; font-weight: 700;
    background: linear-gradient(135deg, #818cf8, #c084fc);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.stat-label { font-size: 0.8rem; color: #64748b; margin-top: 4px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }

/* ── Feature cards grid ── */
.cards-grid {
    display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-top: 20px;
}
.feature-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px; padding: 28px;
    transition: border-color 0.3s, transform 0.3s;
}
.feature-card:hover { border-color: rgba(129,140,248,0.4); transform: translateY(-3px); }
.card-icon { font-size: 2rem; margin-bottom: 12px; }
.feature-card h3 { font-size: 1rem; font-weight: 700; color: #e2e8f0; margin-bottom: 8px; }
.feature-card p  { font-size: 0.875rem; color: #64748b; line-height: 1.6; }

/* ── Section titles ── */
.section-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.8rem; font-weight: 700; color: #fff;
    margin: 40px 0 8px;
}
.section-sub { font-size: 0.95rem; color: #64748b; margin-bottom: 20px; }

/* ── Form sections ── */
.form-section-title {
    font-size: 0.8rem; font-weight: 700; letter-spacing: 0.1em;
    text-transform: uppercase; color: #818cf8;
    border-left: 3px solid #818cf8; padding-left: 10px;
    margin: 24px 0 12px;
}

/* ── Preset buttons ── */
.preset-row { display: flex; gap: 10px; margin-bottom: 20px; flex-wrap: wrap; }
.preset-badge {
    display: inline-block; padding: 6px 18px; border-radius: 999px;
    font-size: 0.85rem; font-weight: 600; cursor: pointer;
}
.preset-low  { background: rgba(16,185,129,0.15); border: 1px solid rgba(16,185,129,0.4); color: #6ee7b7; }
.preset-med  { background: rgba(245,158,11,0.15); border: 1px solid rgba(245,158,11,0.4); color: #fde68a; }
.preset-high { background: rgba(239,68,68,0.15);  border: 1px solid rgba(239,68,68,0.4);  color: #fca5a5; }

/* ── CTA buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: #fff !important; border: none !important;
    border-radius: 12px !important; font-weight: 700 !important;
    font-size: 1rem !important;
    transition: all 0.3s ease !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(99,102,241,0.4) !important;
}

/* ── Result card ── */
.result-card {
    border-radius: 20px; padding: 36px;
    border: 1px solid; backdrop-filter: blur(16px);
    margin: 16px 0; animation: fadeUp 0.5s ease;
}
.card-success  { background: rgba(16,185,129,0.10); border-color: rgba(16,185,129,0.35); }
.card-warning  { background: rgba(245,158,11,0.10); border-color: rgba(245,158,11,0.35); }
.card-danger   { background: rgba(239,68,68,0.10);  border-color: rgba(239,68,68,0.35);  }
.result-label  { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em; font-weight: 700; margin-bottom: 8px; }
.result-title  { font-family: 'Space Grotesk', sans-serif; font-size: 1.8rem; font-weight: 800; color: #fff; }
.result-meta   { font-size: 0.85rem; color: #64748b; margin-top: 6px; }

/* ── Summary grid ── */
.summary-grid {
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-top: 20px;
}
.summary-item {
    background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px; padding: 14px;
}
.item-label { font-size: 0.72rem; color: #64748b; text-transform: uppercase; font-weight: 600; letter-spacing: 0.05em; }
.item-value { font-size: 0.95rem; font-weight: 700; color: #e2e8f0; margin-top: 4px; }

/* ── Metrics table ── */
.metrics-table {
    width: 100%; border-collapse: collapse; font-size: 0.9rem;
}
.metrics-table th {
    background: rgba(129,140,248,0.1); color: #a5b4fc;
    padding: 12px 16px; text-align: left; font-weight: 700;
    text-transform: uppercase; font-size: 0.75rem; letter-spacing: 0.06em;
    border-bottom: 2px solid rgba(129,140,248,0.2);
}
.metrics-table td {
    padding: 12px 16px; color: #e2e8f0;
    border-bottom: 1px solid rgba(255,255,255,0.05);
}
.metrics-table tr:hover td { background: rgba(255,255,255,0.03); }
.row-highlight td { background: rgba(129,140,248,0.07) !important; }

/* ── Badges ── */
.badge {
    display: inline-block; padding: 3px 10px; border-radius: 999px;
    font-size: 0.72rem; font-weight: 700;
}
.badge-success { background: rgba(16,185,129,0.2);  color: #6ee7b7; }
.badge-info    { background: rgba(56,189,248,0.2);  color: #7dd3fc; }
.badge-warning { background: rgba(245,158,11,0.2);  color: #fde68a; }
.badge-accent  { background: rgba(129,140,248,0.15); color: #a5b4fc; }

/* ── Viz images ── */
.viz-img { width: 100%; border-radius: 12px; border: 1px solid rgba(255,255,255,0.07); }
.gallery-card {
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px; padding: 20px; margin-bottom: 20px;
}
.gallery-card h3 { font-size: 1rem; font-weight: 700; color: #e2e8f0; margin-bottom: 4px; }
.gallery-card p  { font-size: 0.85rem; color: #64748b; margin-bottom: 12px; }

/* ── Roadmap ── */
.roadmap-item {
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px; padding: 24px; margin-bottom: 14px;
    border-left: 3px solid #6366f1;
}
.step-num {
    font-size: 0.7rem; font-weight: 800; letter-spacing: 0.12em;
    text-transform: uppercase; color: #818cf8; margin-bottom: 6px;
}
.roadmap-item h3 { font-size: 1rem; font-weight: 700; color: #e2e8f0; margin-bottom: 6px; }
.roadmap-item p  { font-size: 0.875rem; color: #64748b; line-height: 1.6; }

/* ── Recommendation box ── */
.rec-box {
    border-radius: 12px; padding: 20px; margin: 18px 0;
    border: 1px solid;
}
.rec-success { background: rgba(16,185,129,0.08); border-color: rgba(16,185,129,0.3); }
.rec-warning { background: rgba(245,158,11,0.08); border-color: rgba(245,158,11,0.3); }
.rec-danger  { background: rgba(239,68,68,0.08);  border-color: rgba(239,68,68,0.3);  }
.rec-box h4  { font-size: 0.9rem; font-weight: 700; color: #e2e8f0; margin-bottom: 6px; }
.rec-box p   { font-size: 0.875rem; color: #94a3b8; line-height: 1.6; }

/* ── Footer ── */
.footer-bar {
    border-top: 1px solid rgba(255,255,255,0.07);
    margin-top: 60px; padding-top: 24px;
    display: flex; justify-content: space-between; align-items: center;
}
.footer-bar p { font-size: 0.85rem; color: #475569; }

/* ── Metric containers ── */
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 14px !important;
}

/* ── Inputs ── */
.stSelectbox > div > div,
.stNumberInput > div > div > input {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    color: #e2e8f0 !important; border-radius: 10px !important;
}

/* ── Progress ── */
.stProgress > div > div > div { border-radius: 999px; }

hr { border-color: rgba(255,255,255,0.08) !important; }

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}
</style>
""", unsafe_allow_html=True)

# ── Helpers ────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_model(filename: str):
    p = MODELS_DIR / filename
    return joblib.load(p) if p.exists() else None


def build_features(f: dict) -> pd.DataFrame:
    yes = lambda k: 1 if str(f.get(k, "No")).strip().lower() in ["yes","1","true"] else 0
    edu  = f.get("Education",      "Bachelor's")
    emp  = f.get("EmploymentType", "Full-time")
    mar  = f.get("MaritalStatus",  "Married")
    purp = f.get("LoanPurpose",    "Home")
    row = {
        "Age":              float(f["Age"]),
        "Income":           float(f["Income"]),
        "LoanAmount":       float(f["LoanAmount"]),
        "CreditScore":      float(f["CreditScore"]),
        "MonthsEmployed":   float(f["MonthsEmployed"]),
        "NumCreditLines":   float(f["NumCreditLines"]),
        "InterestRate":     float(f["InterestRate"]),
        "LoanTerm":         float(f["LoanTerm"]),
        "DTIRatio":         float(f["DTIRatio"]),
        "HasMortgage":  yes("HasMortgage"),
        "HasDependents":yes("HasDependents"),
        "HasCoSigner":  yes("HasCoSigner"),
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


def run_pred(model, df):
    pred = int(model.predict(df)[0])
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(df)[0]
        return pred, float(p[1]), float(p[0])
    return pred, float(pred), float(1 - pred)


def risk_meta(dp):
    if dp >= 0.50:
        return "High Risk",     "danger",  "Reject Application / Require Collateral",  \
               "Applicant displays elevated default risk indicators (high DTI, interest rate, or low credit score). Secondary underwriting review or substantial down payment required.", \
               "#ef4444"
    if dp >= 0.25:
        return "Moderate Risk", "warning", "Conditional Approval / Manual Review", \
               "Applicant is in the moderate risk band. Approve with adjusted risk-based interest rate or co-signer guarantee.", \
               "#f59e0b"
    return     "Low Risk",      "success", "Approve Loan Application", \
               "Applicant demonstrates strong financial capacity and low historical default probability. Standard prime lending terms apply.", \
               "#10b981"


# ── Sidebar Navigation ─────────────────────────────────────────────────────────
if "nav_page" not in st.session_state:
    st.session_state["nav_page"] = "Overview"

with st.sidebar:
    st.markdown("""
    <div class="brand-bar">
        <span class="icon">🛡️</span>
        <span class="name">CrediPulse <span>AI</span></span>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["Overview", "Risk Evaluator", "Model Analytics"],
        key="nav_page",
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        "<small style='color:#475569'>Enterprise Credit Risk Underwriting Intelligence<br>"
        "ML Platform for Real-Time Credit Assessment</small>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW  (matches Flask index.html)
# ══════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.markdown('<div class="hero-badge">Enterprise Credit Risk Underwriting Engine</div>', unsafe_allow_html=True)
    st.markdown("""
    <h1 class="hero-title">Automated Loan Default
        <span class="grad">Risk Prediction</span>
    </h1>
    <p class="hero-sub">
        Production-ready machine learning underwriting engine trained on 255,000+ real-world loan
        applications. Evaluates credit risk in milliseconds with tree-based algorithms and custom
        pure NumPy models.
    </p>
    """, unsafe_allow_html=True)

    # CTA buttons
    bc1, bc2, bc3 = st.columns([2, 2, 6])
    with bc1:
        if st.button("⚡ Launch Risk Evaluator →", use_container_width=True):
            st.session_state["nav_page"] = "Risk Evaluator"
            st.rerun()
    with bc2:
        if st.button("📊 Explore Analytics", use_container_width=True):
            st.session_state["nav_page"] = "Model Analytics"
            st.rerun()

    # Stats grid
    st.markdown("""
    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-number">255k+</div>
            <div class="stat-label">Loan Applications</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">88.6%</div>
            <div class="stat-label">Test Set Accuracy</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">0.736</div>
            <div class="stat-label">Ensemble ROC-AUC</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">&lt; 15ms</div>
            <div class="stat-label">Real-time Inference</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <h2 class="section-title">End-to-End System Architecture</h2>
    <p class="section-sub">Modular, scalable machine learning pipeline from exploratory analysis to live scoring</p>

    <div class="cards-grid">
        <div class="feature-card">
            <div class="card-icon">📊</div>
            <h3>Data Engineering & Preprocessing</h3>
            <p>Outlier treatment via IQR, missing value verification, categorical dummy encoding, and feature scaling comparisons.</p>
        </div>
        <div class="feature-card">
            <div class="card-icon">🧠</div>
            <h3>Dual-Engine Model Architecture</h3>
            <p>Trained Scikit-Learn Decision Trees alongside a pure Python + NumPy Decision Tree implemented from first principles.</p>
        </div>
        <div class="feature-card">
            <div class="card-icon">⚖️</div>
            <h3>Diagnostics & Ensemble Learning</h3>
            <p>Bias-variance tradeoff sweeps, 5-Fold Stratified Cross-Validation, Random Forest, and Gradient Boosting tuning.</p>
        </div>
        <div class="feature-card">
            <div class="card-icon">📈</div>
            <h3>Performance & Visual Analytics</h3>
            <p>High-resolution ROC curves, Precision-Recall curves, confusion matrices, and feature importance rankings.</p>
        </div>
        <div class="feature-card">
            <div class="card-icon">⚡</div>
            <h3>Interactive Web Application</h3>
            <p>Modern application with interactive 16-parameter input form, presets, and real-time risk gauges.</p>
        </div>
        <div class="feature-card">
            <div class="card-icon">🚀</div>
            <h3>Production API & Scalability</h3>
            <p>REST API endpoints, Docker containerization, cloud deployment configs, and comprehensive system validation.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="footer-bar">
        <p><strong>CrediPulse AI</strong> — Credit Risk Underwriting Intelligence</p>
        <p>Enterprise Machine Learning Platform for Real-Time Credit Risk Assessment</p>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — RISK EVALUATOR  (matches Flask predict.html + result.html)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Risk Evaluator":
    st.markdown('<div class="hero-badge">Real-Time Risk Scoring Engine</div>', unsafe_allow_html=True)
    st.markdown("## Loan Applicant Risk Evaluation")
    st.markdown('<p style="color:#64748b">Enter borrower credit attributes or select a preset to evaluate default probability.</p>', unsafe_allow_html=True)

    # ── Model selector ─────────────────────────────────────────────────────────
    model_choice = st.selectbox("🤖 Scoring Model", list(AVAILABLE_MODELS.keys()))

    # ── Preset selector ────────────────────────────────────────────────────────
    st.markdown('<div class="form-section-title">Quick Fill Test Profiles</div>', unsafe_allow_html=True)
    PRESETS = {
        "🟢 Low Risk Borrower":      dict(Age=42, Income=120000, LoanAmount=18000, CreditScore=780,
                                          MonthsEmployed=96, NumCreditLines=5, InterestRate=5.2,
                                          LoanTerm=36, DTIRatio=0.18, Education="Master's",
                                          EmploymentType="Full-time", MaritalStatus="Married",
                                          LoanPurpose="Home", HasMortgage="Yes",
                                          HasDependents="No", HasCoSigner="Yes"),
        "🟡 Moderate Risk Borrower": dict(Age=34, Income=65000, LoanAmount=28000, CreditScore=640,
                                          MonthsEmployed=36, NumCreditLines=3, InterestRate=11.5,
                                          LoanTerm=48, DTIRatio=0.35, Education="Bachelor's",
                                          EmploymentType="Part-time", MaritalStatus="Single",
                                          LoanPurpose="Auto", HasMortgage="No",
                                          HasDependents="Yes", HasCoSigner="No"),
        "🔴 High Risk Borrower":     dict(Age=27, Income=32000, LoanAmount=45000, CreditScore=520,
                                          MonthsEmployed=8,  NumCreditLines=2, InterestRate=22.0,
                                          LoanTerm=60, DTIRatio=0.62, Education="High School",
                                          EmploymentType="Unemployed", MaritalStatus="Divorced",
                                          LoanPurpose="Other", HasMortgage="No",
                                          HasDependents="Yes", HasCoSigner="No"),
    }

    pc1, pc2, pc3, pc4 = st.columns([2, 2, 2, 4])
    preset_choice = None
    with pc1:
        if st.button("🟢 Low Risk Borrower",      use_container_width=True): preset_choice = "🟢 Low Risk Borrower"
    with pc2:
        if st.button("🟡 Moderate Risk Borrower", use_container_width=True): preset_choice = "🟡 Moderate Risk Borrower"
    with pc3:
        if st.button("🔴 High Risk Borrower",     use_container_width=True): preset_choice = "🔴 High Risk Borrower"

    # Store preset in session
    if preset_choice:
        st.session_state["preset"] = PRESETS[preset_choice]

    p = st.session_state.get("preset", PRESETS["🟢 Low Risk Borrower"])

    # ── Section 1: Demographics ────────────────────────────────────────────────
    st.markdown('<div class="form-section-title">👤 1. Applicant Demographics</div>', unsafe_allow_html=True)
    d1, d2, d3, d4 = st.columns(4)
    age            = d1.number_input("Age (Years)",       18, 75,  p["Age"])
    education      = d2.selectbox("Education Level",      ["High School","Bachelor's","Master's","PhD"],
                                   index=["High School","Bachelor's","Master's","PhD"].index(p["Education"]))
    marital_status = d3.selectbox("Marital Status",       ["Single","Married","Divorced"],
                                   index=["Single","Married","Divorced"].index(p["MaritalStatus"]))
    has_dependents = d4.selectbox("Has Dependents?",      ["No","Yes"],
                                   index=["No","Yes"].index(p["HasDependents"]))

    # ── Section 2: Employment & Financial ──────────────────────────────────────
    st.markdown('<div class="form-section-title">💼 2. Employment & Financial Capacity</div>', unsafe_allow_html=True)
    e1, e2, e3, e4 = st.columns(4)
    employment_type  = e1.selectbox("Employment Type",        ["Full-time","Part-time","Self-employed","Unemployed"],
                                    index=["Full-time","Part-time","Self-employed","Unemployed"].index(p["EmploymentType"]))
    months_employed  = e2.number_input("Months Employed",      0, 180,  p["MonthsEmployed"])
    income           = e3.number_input("Annual Gross Income ($)", 5000, 250000, p["Income"], step=1000)
    dti_ratio        = e4.number_input("DTI Ratio",            0.05, 0.95, p["DTIRatio"], step=0.01,
                                       help="Total monthly debt ÷ gross income")

    # ── Section 3: Credit Profile ──────────────────────────────────────────────
    st.markdown('<div class="form-section-title">💳 3. Credit Profile</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    credit_score     = c1.number_input("Credit Score (FICO)", 300, 850, p["CreditScore"])
    num_credit_lines = c2.number_input("Open Credit Lines",     1,  15,  p["NumCreditLines"])
    has_mortgage     = c3.selectbox("Has Existing Mortgage?", ["Yes","No"],
                                    index=["Yes","No"].index(p["HasMortgage"]))
    has_cosigner     = c4.selectbox("Has Co-Signer?",         ["Yes","No"],
                                    index=["Yes","No"].index(p["HasCoSigner"]))

    # ── Section 4: Loan Terms ──────────────────────────────────────────────────
    st.markdown('<div class="form-section-title">📑 4. Requested Loan Terms</div>', unsafe_allow_html=True)
    l1, l2, l3, l4 = st.columns(4)
    loan_amount   = l1.number_input("Loan Amount ($)",     1000, 150000, p["LoanAmount"], step=500)
    interest_rate = l2.number_input("Interest Rate (%)",    2.0,   35.0, float(p["InterestRate"]), step=0.1)
    term_map      = {12:"12 Months (1 Year)",24:"24 Months (2 Years)",
                     36:"36 Months (3 Years)",48:"48 Months (4 Years)",60:"60 Months (5 Years)"}
    loan_term_sel = l3.selectbox("Loan Term", list(term_map.values()),
                                  index=list(term_map.keys()).index(p["LoanTerm"]))
    loan_term     = [k for k,v in term_map.items() if v == loan_term_sel][0]
    purpose_map   = {"Home":"Home Improvement / Purchase","Auto":"Auto Financing",
                     "Business":"Small Business","Education":"Education / Tuition","Other":"Other / Personal"}
    loan_purp_sel = l4.selectbox("Loan Purpose", list(purpose_map.values()),
                                  index=list(purpose_map.values()).index(purpose_map[p["LoanPurpose"]]))
    loan_purpose  = [k for k,v in purpose_map.items() if v == loan_purp_sel][0]

    st.markdown("---")
    sa, sb = st.columns([3, 1])
    evaluate_btn = sa.button("⚡ Evaluate Loan Default Risk", use_container_width=True)
    reset_btn    = sb.button("↺ Reset Fields",               use_container_width=True)

    if reset_btn:
        if "preset" in st.session_state:
            del st.session_state["preset"]
        st.rerun()

    # ── RESULT ─────────────────────────────────────────────────────────────────
    if evaluate_btn:
        form = dict(Age=age, Income=income, LoanAmount=loan_amount,
                    CreditScore=credit_score, MonthsEmployed=months_employed,
                    NumCreditLines=num_credit_lines, InterestRate=interest_rate,
                    LoanTerm=loan_term, DTIRatio=dti_ratio, Education=education,
                    EmploymentType=employment_type, MaritalStatus=marital_status,
                    LoanPurpose=loan_purpose, HasMortgage=has_mortgage,
                    HasDependents=has_dependents, HasCoSigner=has_cosigner)

        model = load_model(AVAILABLE_MODELS[model_choice])
        if model is None:
            st.error("❌ Model file not found. Ensure .pkl files are in notebooks-containing-models/")
        else:
            df_in = build_features(form)
            pred, dp, ndp = run_pred(model, df_in)
            tier, badge_class, decision, recommendation, color = risk_meta(dp)
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            st.markdown(f"""
            <div class="result-card card-{badge_class}">
                <div class="result-label" style="color:{color}">{tier}</div>
                <div class="result-title">Underwriting Decision: {decision}</div>
                <div class="result-meta">Assessed on {ts} · Engine: {model_choice}</div>
            </div>
            """, unsafe_allow_html=True)

            # Probability gauge
            st.markdown("**Non-Default Probability vs Default Risk Probability**")
            g1, g2 = st.columns(2)
            with g1:
                st.markdown(f"Non-Default: **{ndp*100:.1f}%**")
                st.progress(ndp)
            with g2:
                st.markdown(f"Default Risk: **{dp*100:.1f}%**")
                st.progress(dp)

            # Recommendation box
            st.markdown(f"""
            <div class="rec-box rec-{badge_class}">
                <h4>📋 Underwriting Recommendation</h4>
                <p>{recommendation}</p>
            </div>
            """, unsafe_allow_html=True)

            # Submitted parameters summary
            st.markdown("**Submitted Applicant Parameters**")
            st.markdown(f"""
            <div class="summary-grid">
                <div class="summary-item">
                    <div class="item-label">Age / Education</div>
                    <div class="item-value">{age} yrs · {education}</div>
                </div>
                <div class="summary-item">
                    <div class="item-label">Annual Income</div>
                    <div class="item-value">${income:,.0f}</div>
                </div>
                <div class="summary-item">
                    <div class="item-label">Loan Amount</div>
                    <div class="item-value">${loan_amount:,.0f}</div>
                </div>
                <div class="summary-item">
                    <div class="item-label">Credit Score</div>
                    <div class="item-value">{credit_score}</div>
                </div>
                <div class="summary-item">
                    <div class="item-label">Interest Rate</div>
                    <div class="item-value">{interest_rate}%</div>
                </div>
                <div class="summary-item">
                    <div class="item-label">DTI Ratio</div>
                    <div class="item-value">{dti_ratio}</div>
                </div>
                <div class="summary-item">
                    <div class="item-label">Employment Type</div>
                    <div class="item-value">{employment_type} ({months_employed} mos)</div>
                </div>
                <div class="summary-item">
                    <div class="item-label">Co-Signer / Mortgage</div>
                    <div class="item-value">Co-Signer: {has_cosigner} · Mortgage: {has_mortgage}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL ANALYTICS  (matches Flask metrics.html)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Model Analytics":
    st.markdown('<div class="hero-badge">Model Evaluation & Visual Analytics Suite</div>', unsafe_allow_html=True)
    st.markdown("## Machine Learning Performance Dashboard")
    st.markdown('<p style="color:#64748b">Empirical validation graphs and comparison metrics generated across 255k loan applications.</p>', unsafe_allow_html=True)

    # ── Algorithm Leaderboard ──────────────────────────────────────────────────
    st.markdown("""
    <table class="metrics-table">
      <thead>
        <tr>
          <th>Algorithm Architecture</th>
          <th>Test Accuracy</th>
          <th>ROC-AUC</th>
          <th>Precision (Default)</th>
          <th>F1-Score</th>
          <th>Implementation Mode</th>
        </tr>
      </thead>
      <tbody>
        <tr class="row-highlight">
          <td><strong>Gradient Boosting Classifier</strong></td>
          <td>88.64%</td><td>0.7356</td><td>59.80%</td><td>0.1183</td>
          <td><span class="badge badge-success">Top Ensemble</span></td>
        </tr>
        <tr>
          <td><strong>Random Forest Classifier</strong></td>
          <td>88.42%</td><td>0.7320</td><td>100.00%</td><td>0.0064</td>
          <td><span class="badge badge-info">Scikit-Learn</span></td>
        </tr>
        <tr>
          <td><strong>Logistic Regression (L2)</strong></td>
          <td>88.58%</td><td>0.7303</td><td>64.15%</td><td>0.0692</td>
          <td><span class="badge badge-info">Scikit-Learn</span></td>
        </tr>
        <tr>
          <td><strong>Gaussian Naive Bayes</strong></td>
          <td>88.55%</td><td>0.7255</td><td>60.66%</td><td>0.0747</td>
          <td><span class="badge badge-info">Scikit-Learn</span></td>
        </tr>
        <tr>
          <td><strong>Decision Tree Classifier (Depth 7)</strong></td>
          <td>88.02%</td><td>0.6923</td><td>37.39%</td><td>0.0824</td>
          <td><span class="badge badge-info">Scikit-Learn</span></td>
        </tr>
        <tr>
          <td><strong>Scratch Decision Tree (Pure NumPy)</strong></td>
          <td>88.85%</td><td>N/A</td><td>N/A</td><td>N/A</td>
          <td><span class="badge badge-warning">Custom (No Library)</span></td>
        </tr>
      </tbody>
    </table>
    <br>
    """, unsafe_allow_html=True)

    # ── Visualizations Gallery ──────────────────────────────────────────────────
    VIZ = [
        ("roc_curve_comparison.png",      "1. Receiver Operating Characteristic (ROC)",
         "Gradient Boosting achieves peak separation with AUC = 0.736."),
        ("precision_recall_curves.png",   "2. Precision-Recall Curves",
         "Evaluating true positive trade-offs under 11.6% class imbalance."),
        ("confusion_matrix_heatmaps.png", "3. Confusion Matrix Heatmaps",
         "Decision Tree vs Random Forest classification breakdown."),
        ("feature_importance_ranking.png","4. Feature Importance Ranking",
         "Age, Income, and InterestRate dominate default predictive power."),
        ("overfitting_learning_curves.png","5. Bias-Variance & Overfitting Curve",
         "Depth sweep identifies sweet spot at depth 7, preventing memorization."),
        ("model_calibration_curves.png",  "6. Model Calibration (Reliability)",
         "Reliability curves confirm well-calibrated posterior probabilities."),
        ("cross_validation_boxplots.png", "7. 5-Fold Stratified Cross-Validation",
         "Distribution of ROC-AUC scores across 5 random test folds."),
    ]

    col_a, col_b = st.columns(2)
    for i, (fname, title, desc) in enumerate(VIZ):
        img_path = VIZ_DIR / fname
        with (col_a if i % 2 == 0 else col_b):
            st.markdown(f"""
            <div class="gallery-card">
                <h3>{title}</h3>
                <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)
            if img_path.exists():
                st.image(str(img_path), use_container_width=True)
            else:
                st.info(f"📊 Visualization not found: {fname}")





