/**
 * Predict.jsx
 * ===========
 * Interactive Loan Default Prediction & Credit Risk Assessment Dashboard.
 * Platform: CrediPulse AI
 */

import React, { useState, useEffect } from "react";

// The FastAPI backend base URL
const API_BASE_URL = "http://localhost:8000";

// Initial default state for all 16 loan application fields
const INITIAL_FORM_DATA = {
  Age: 35,
  Income: 75000,
  LoanAmount: 25000,
  CreditScore: 720,
  MonthsEmployed: 48,
  NumCreditLines: 3,
  InterestRate: 7.5,
  LoanTerm: 36,
  DTIRatio: 0.28,
  Education: "Bachelor's",
  EmploymentType: "Full-time",
  MaritalStatus: "Married",
  HasMortgage: "Yes",
  HasDependents: "No",
  LoanPurpose: "Home",
  HasCoSigner: "Yes",
};

// Preset Profiles to allow 1-click testing of different risk scenarios
const PRESET_PROFILES = {
  lowRisk: {
    Age: 48,
    Income: 98000,
    LoanAmount: 30000,
    CreditScore: 780,
    MonthsEmployed: 96,
    NumCreditLines: 4,
    InterestRate: 4.8,
    LoanTerm: 36,
    DTIRatio: 0.18,
    Education: "Master's",
    EmploymentType: "Full-time",
    MaritalStatus: "Married",
    HasMortgage: "Yes",
    HasDependents: "Yes",
    LoanPurpose: "Home",
    HasCoSigner: "Yes",
  },
  moderateRisk: {
    Age: 32,
    Income: 45000,
    LoanAmount: 35000,
    CreditScore: 610,
    MonthsEmployed: 20,
    NumCreditLines: 5,
    InterestRate: 14.5,
    LoanTerm: 48,
    DTIRatio: 0.45,
    Education: "Bachelor's",
    EmploymentType: "Full-time",
    MaritalStatus: "Single",
    HasMortgage: "No",
    HasDependents: "No",
    LoanPurpose: "Auto",
    HasCoSigner: "No",
  },
  highRisk: {
    Age: 21,
    Income: 16000,
    LoanAmount: 95000,
    CreditScore: 410,
    MonthsEmployed: 3,
    NumCreditLines: 8,
    InterestRate: 22.8,
    LoanTerm: 60,
    DTIRatio: 0.78,
    Education: "High School",
    EmploymentType: "Unemployed",
    MaritalStatus: "Single",
    HasMortgage: "No",
    HasDependents: "Yes",
    LoanPurpose: "Other",
    HasCoSigner: "No",
  },
};

function PredictPage() {
  // ---------------------------------------------------------------------------
  // REACT STATE HOOKS
  // ---------------------------------------------------------------------------
  const [formData, setFormData] = useState(INITIAL_FORM_DATA);
  const [models, setModels] = useState([
    { id: "decision_tree", name: "Decision Tree Classifier", description: "Standard Decision Tree" }
  ]);
  const [selectedModel, setSelectedModel] = useState("decision_tree");

  // UI state
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [copied, setCopied] = useState(false);

  // ---------------------------------------------------------------------------
  // FETCH REGISTERED MODELS ON MOUNT
  // ---------------------------------------------------------------------------
  useEffect(() => {
    async function fetchModels() {
      try {
        const response = await fetch(`${API_BASE_URL}/api/models`);
        if (response.ok) {
          const data = await response.json();
          if (data.models && data.models.length > 0) {
            setModels(data.models);
            setSelectedModel(data.default_model_id || data.models[0].id);
          }
        }
      } catch (err) {
        console.warn("Could not fetch models from backend, using default:", err);
      }
    }
    fetchModels();
  }, []);

  // ---------------------------------------------------------------------------
  // FORM HANDLERS
  // ---------------------------------------------------------------------------
  const handleChange = (e) => {
    const { name, value, type } = e.target;
    let parsedValue = value;
    if (type === "number") {
      parsedValue = value === "" ? "" : parseFloat(value);
    }
    setFormData((prev) => ({
      ...prev,
      [name]: parsedValue,
    }));
  };

  const handleApplyPreset = (profileKey) => {
    if (PRESET_PROFILES[profileKey]) {
      setFormData({ ...PRESET_PROFILES[profileKey] });
      setResult(null);
      setError(null);
    }
  };

  const handleReset = () => {
    setFormData({ ...INITIAL_FORM_DATA });
    setResult(null);
    setError(null);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const payload = {
        ...formData,
        model_id: selectedModel,
      };

      const response = await fetch(`${API_BASE_URL}/api/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(
          errData.detail || `Server returned HTTP ${response.status} ${response.statusText}`
        );
      }

      const data = await response.json();
      setResult(data);

      setTimeout(() => {
        const el = document.getElementById("prediction-result");
        if (el) {
          el.scrollIntoView({ behavior: "smooth" });
        }
      }, 100);
    } catch (err) {
      console.error("Prediction submission failed:", err);
      setError(
        err.message.includes("Failed to fetch")
          ? "Cannot connect to the backend server. Please verify that FastAPI is running on http://localhost:8000."
          : err.message
      );
    } finally {
      setLoading(false);
    }
  };

  // Copy report summary to clipboard
  const handleCopyReport = () => {
    if (!result) return;
    const reportText = `CrediPulse AI - Credit Risk Evaluation Report
---------------------------------------------
Model Used: ${result.model_name}
Outcome: ${result.is_default ? "DEFAULT PREDICTED (High Risk)" : "SAFE / NO DEFAULT"}
Risk Tier: ${result.risk_level}
Default Probability: ${(result.default_probability * 100).toFixed(1)}%
Safe Repayment Probability: ${(result.non_default_probability * 100).toFixed(1)}%
Summary: ${result.summary}
Applicant: Age ${formData.Age}, Income $${formData.Income}, Loan Amount $${formData.LoanAmount}, Credit Score ${formData.CreditScore}, DTI ${formData.DTIRatio}`;

    navigator.clipboard.writeText(reportText).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2500);
    });
  };

  const handlePrintReport = () => {
    window.print();
  };

  return (
    <div className="predict-container pb-4">
      {/* 1. PAGE HEADER */}
      <div className="page-header text-center">
        <h1 className="page-title">
          Credit Risk &amp; <span className="gradient-text">Default Evaluator</span>
        </h1>
        <p className="page-subtitle">
          Input applicant parameters to calculate default likelihood, risk tier classification, and automated underwriting guidance.
        </p>
      </div>

      {/* 2. PRESET PROFILES TOOLBAR */}
      <div className="preset-bar">
        <div className="preset-title">
          <span>⚡ Fast Test Profiles:</span>
        </div>
        <div className="preset-btn-group">
          <button
            type="button"
            className="btn-preset btn-preset-success"
            onClick={() => handleApplyPreset("lowRisk")}
            title="Load Tier 1 prime credit profile"
          >
            <span>🟢 Tier 1: Prime Profile</span>
          </button>
          <button
            type="button"
            className="btn-preset btn-preset-warning"
            onClick={() => handleApplyPreset("moderateRisk")}
            title="Load Tier 2 moderate risk profile"
          >
            <span>🟡 Tier 2: Moderate Risk</span>
          </button>
          <button
            type="button"
            className="btn-preset btn-preset-danger"
            onClick={() => handleApplyPreset("highRisk")}
            title="Load Tier 3 subprime high risk profile"
          >
            <span>🔴 Tier 3: High Risk Profile</span>
          </button>
          <button
            type="button"
            className="btn-preset"
            onClick={handleReset}
            title="Reset form fields to default"
          >
            <span>🔄 Reset Form</span>
          </button>
        </div>
      </div>

      {/* 3. ERROR BANNER */}
      {error && (
        <div className="alert alert-danger d-flex align-items-center justify-content-between my-3" role="alert">
          <div>
            <strong>Backend / Validation Error:</strong> {error}
          </div>
          <button
            type="button"
            className="btn-close"
            aria-label="Close"
            onClick={() => setError(null)}
          ></button>
        </div>
      )}

      {/* 4. MAIN PREDICTION FORM */}
      <form onSubmit={handleSubmit} className="text-start">
        {/* SECTION 1: PERSONAL & DEMOGRAPHIC */}
        <div className="form-section-card">
          <div className="form-section-header">
            <div className="section-icon-badge">👤</div>
            <h3 className="form-section-title">1. Personal &amp; Demographic Profile</h3>
          </div>

          <div className="row g-3">
            <div className="col-md-3">
              <label className="form-label">
                Age
                <span className="field-hint">(18 - 100)</span>
              </label>
              <input
                type="number"
                className="form-control"
                name="Age"
                min="18"
                max="100"
                required
                value={formData.Age}
                onChange={handleChange}
              />
            </div>

            <div className="col-md-3">
              <label className="form-label">Education</label>
              <select
                className="form-select"
                name="Education"
                value={formData.Education}
                onChange={handleChange}
              >
                <option value="High School">High School</option>
                <option value="Bachelor's">Bachelor's</option>
                <option value="Master's">Master's</option>
                <option value="PhD">PhD</option>
              </select>
            </div>

            <div className="col-md-3">
              <label className="form-label">Marital Status</label>
              <select
                className="form-select"
                name="MaritalStatus"
                value={formData.MaritalStatus}
                onChange={handleChange}
              >
                <option value="Single">Single</option>
                <option value="Married">Married</option>
                <option value="Divorced">Divorced</option>
              </select>
            </div>

            <div className="col-md-3">
              <label className="form-label">Has Dependents</label>
              <select
                className="form-select"
                name="HasDependents"
                value={formData.HasDependents}
                onChange={handleChange}
              >
                <option value="No">No</option>
                <option value="Yes">Yes</option>
              </select>
            </div>
          </div>
        </div>

        {/* SECTION 2: EMPLOYMENT & INCOME */}
        <div className="form-section-card">
          <div className="form-section-header">
            <div className="section-icon-badge">💼</div>
            <h3 className="form-section-title">2. Employment &amp; Income Stream</h3>
          </div>

          <div className="row g-3">
            <div className="col-md-4">
              <label className="form-label">
                Annual Income
                <span className="field-hint">($ USD)</span>
              </label>
              <input
                type="number"
                className="form-control"
                name="Income"
                min="1000"
                step="1000"
                required
                value={formData.Income}
                onChange={handleChange}
              />
            </div>

            <div className="col-md-4">
              <label className="form-label">Employment Type</label>
              <select
                className="form-select"
                name="EmploymentType"
                value={formData.EmploymentType}
                onChange={handleChange}
              >
                <option value="Full-time">Full-time</option>
                <option value="Part-time">Part-time</option>
                <option value="Self-employed">Self-employed</option>
                <option value="Unemployed">Unemployed</option>
              </select>
            </div>

            <div className="col-md-4">
              <label className="form-label">
                Months Employed
                <span className="field-hint">(Tenure)</span>
              </label>
              <input
                type="number"
                className="form-control"
                name="MonthsEmployed"
                min="0"
                max="600"
                required
                value={formData.MonthsEmployed}
                onChange={handleChange}
              />
            </div>
          </div>
        </div>

        {/* SECTION 3: CREDIT HEALTH & LIABILITIES */}
        <div className="form-section-card">
          <div className="form-section-header">
            <div className="section-icon-badge">📊</div>
            <h3 className="form-section-title">3. Credit Health &amp; Liabilities</h3>
          </div>

          <div className="row g-3">
            <div className="col-md-3">
              <label className="form-label">
                Credit Score
                <span className="field-hint">(300 - 850)</span>
              </label>
              <input
                type="number"
                className="form-control"
                name="CreditScore"
                min="300"
                max="850"
                required
                value={formData.CreditScore}
                onChange={handleChange}
              />
            </div>

            <div className="col-md-3">
              <label className="form-label">
                Debt-to-Income (DTI)
                <span className="field-hint">(0.0 - 1.0)</span>
              </label>
              <input
                type="number"
                step="0.01"
                className="form-control"
                name="DTIRatio"
                min="0.0"
                max="1.0"
                required
                value={formData.DTIRatio}
                onChange={handleChange}
              />
            </div>

            <div className="col-md-2">
              <label className="form-label">
                Credit Lines
                <span className="field-hint">(Open)</span>
              </label>
              <input
                type="number"
                className="form-control"
                name="NumCreditLines"
                min="0"
                max="50"
                required
                value={formData.NumCreditLines}
                onChange={handleChange}
              />
            </div>

            <div className="col-md-2">
              <label className="form-label">Mortgage?</label>
              <select
                className="form-select"
                name="HasMortgage"
                value={formData.HasMortgage}
                onChange={handleChange}
              >
                <option value="No">No</option>
                <option value="Yes">Yes</option>
              </select>
            </div>

            <div className="col-md-2">
              <label className="form-label">Co-Signer?</label>
              <select
                className="form-select"
                name="HasCoSigner"
                value={formData.HasCoSigner}
                onChange={handleChange}
              >
                <option value="No">No</option>
                <option value="Yes">Yes</option>
              </select>
            </div>
          </div>
        </div>

        {/* SECTION 4: LOAN SPECIFICATIONS */}
        <div className="form-section-card">
          <div className="form-section-header">
            <div className="section-icon-badge">📋</div>
            <h3 className="form-section-title">4. Requested Loan Specifications</h3>
          </div>

          <div className="row g-3">
            <div className="col-md-3">
              <label className="form-label">
                Loan Amount
                <span className="field-hint">($ USD)</span>
              </label>
              <input
                type="number"
                className="form-control"
                name="LoanAmount"
                min="500"
                step="500"
                required
                value={formData.LoanAmount}
                onChange={handleChange}
              />
            </div>

            <div className="col-md-3">
              <label className="form-label">
                Interest Rate
                <span className="field-hint">(% APR)</span>
              </label>
              <input
                type="number"
                step="0.01"
                className="form-control"
                name="InterestRate"
                min="0.1"
                max="40"
                required
                value={formData.InterestRate}
                onChange={handleChange}
              />
            </div>

            <div className="col-md-3">
              <label className="form-label">
                Loan Term
                <span className="field-hint">(Months)</span>
              </label>
              <input
                type="number"
                className="form-control"
                name="LoanTerm"
                min="6"
                max="120"
                required
                value={formData.LoanTerm}
                onChange={handleChange}
              />
            </div>

            <div className="col-md-3">
              <label className="form-label">Loan Purpose</label>
              <select
                className="form-select"
                name="LoanPurpose"
                value={formData.LoanPurpose}
                onChange={handleChange}
              >
                <option value="Auto">Auto</option>
                <option value="Business">Business</option>
                <option value="Education">Education</option>
                <option value="Home">Home</option>
                <option value="Other">Other</option>
              </select>
            </div>
          </div>
        </div>

        {/* SUBMIT & MODEL SELECTION BAR */}
        <div className="submit-card">
          <div className="d-flex align-items-center gap-3">
            <label className="form-label mb-0 fw-bold">Active Engine:</label>
            <select
              className="form-select w-auto"
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
            >
              {models.map((m) => (
                <option key={m.id} value={m.id}>
                  🌲 {m.name}
                </option>
              ))}
            </select>
          </div>

          <button
            type="submit"
            className="btn btn-predict"
            disabled={loading}
          >
            {loading ? (
              <>
                <span className="spinner-small"></span>
                <span>Calculating Credit Risk...</span>
              </>
            ) : (
              <>
                <span>⚡ Calculate Default Risk</span>
              </>
            )}
          </button>
        </div>
      </form>

      {/* 5. PREDICTION RESULTS DASHBOARD */}
      {result && (
        <div
          id="prediction-result"
          className={`result-card ${
            result.risk_level === "Low"
              ? "low-risk"
              : result.risk_level === "Moderate"
              ? "moderate-risk"
              : "high-risk"
          }`}
        >
          {/* Result Header */}
          <div className="result-header">
            <div>
              <span className="text-muted small text-uppercase fw-bold">
                CrediPulse AI Underwriting Report
              </span>
              <h3 className="mb-0 mt-1">
                {result.is_default ? "⚠️ Default Likely (High Credit Risk)" : "✅ Safe Application (Low / Moderate Risk)"}
              </h3>
            </div>

            <div
              className={`result-badge ${
                result.risk_level === "Low"
                  ? "badge-approved"
                  : result.risk_level === "Moderate"
                  ? "badge-moderate"
                  : "badge-rejected"
              }`}
            >
              <span>
                {result.risk_level === "Low"
                  ? "🟢 Tier 1: Low Risk Approved"
                  : result.risk_level === "Moderate"
                  ? "🟡 Tier 2: Moderate Risk"
                  : "🔴 Tier 3: High Risk Warning"}
              </span>
            </div>
          </div>

          {/* Probability Progress Meter */}
          <div className="probability-section">
            <div className="probability-labels">
              <span className="text-success">
                🛡️ Safe Repayment Probability: {(result.non_default_probability * 100).toFixed(1)}%
              </span>
              <span className="text-danger">
                ⚠️ Default Risk Probability: {(result.default_probability * 100).toFixed(1)}%
              </span>
            </div>

            <div className="progress-track">
              <div
                className="progress-fill-safe"
                style={{ width: `${result.non_default_probability * 100}%` }}
                title={`Safe Probability: ${(result.non_default_probability * 100).toFixed(1)}%`}
              ></div>
              <div
                className="progress-fill-risk"
                style={{ width: `${result.default_probability * 100}%` }}
                title={`Default Risk: ${(result.default_probability * 100).toFixed(1)}%`}
              ></div>
            </div>
          </div>

          {/* Key Metrics Breakdown */}
          <div className="result-metrics-grid">
            <div className="metric-box">
              <div className="metric-label">Prediction Status</div>
              <div className="metric-value">
                {result.prediction === 0 ? "Standard Pass" : "Default Flag"}
              </div>
            </div>

            <div className="metric-box">
              <div className="metric-label">Default Risk Prob.</div>
              <div className="metric-value text-danger">
                {(result.default_probability * 100).toFixed(1)}%
              </div>
            </div>

            <div className="metric-box">
              <div className="metric-label">Safe Prob.</div>
              <div className="metric-value text-success">
                {(result.non_default_probability * 100).toFixed(1)}%
              </div>
            </div>

            <div className="metric-box">
              <div className="metric-label">Model Engine</div>
              <div className="metric-value" style={{ fontSize: "1.05rem" }}>
                {result.model_name}
              </div>
            </div>
          </div>

          {/* AI Decision Summary Box */}
          <div className="result-summary-box text-start">
            <strong className="d-block mb-1">🤖 Model Analytical Insight:</strong>
            <p className="mb-0 text-muted">{result.summary}</p>
          </div>

          {/* Actionable Underwriting Guidance Checklist */}
          <div className="result-recommendations-box text-start">
            <div className="recommendations-title">
              <span>📋 Underwriting Decision Recommendations:</span>
            </div>
            <ul className="recommendation-list">
              {result.risk_level === "Low" ? (
                <>
                  <li className="recommendation-item">
                    <span>✅</span>
                    <span><strong>Direct Approval:</strong> Applicant meets all creditworthiness thresholds with low DTI and healthy repayment history.</span>
                  </li>
                  <li className="recommendation-item">
                    <span>✅</span>
                    <span><strong>Prime Pricing:</strong> Eligible for preferential interest rate tiers and maximum term flexibility.</span>
                  </li>
                  <li className="recommendation-item">
                    <span>✅</span>
                    <span><strong>Verification:</strong> Standard automated income and employment verification is sufficient.</span>
                  </li>
                </>
              ) : result.risk_level === "Moderate" ? (
                <>
                  <li className="recommendation-item">
                    <span>⚠️</span>
                    <span><strong>Conditional Approval:</strong> Require verified pay stubs and 3-month bank statement audit.</span>
                  </li>
                  <li className="recommendation-item">
                    <span>⚠️</span>
                    <span><strong>Risk Mitigation:</strong> Recommend reducing loan term by 12 months or adding a creditworthy Co-Signer.</span>
                  </li>
                  <li className="recommendation-item">
                    <span>⚠️</span>
                    <span><strong>Pricing Adjustment:</strong> Apply standard risk premium interest rate buffer (+1.5% to 2.5% APR).</span>
                  </li>
                </>
              ) : (
                <>
                  <li className="recommendation-item">
                    <span>⛔</span>
                    <span><strong>High Risk Warning:</strong> Elevated probability of default based on low credit score, high DTI, or unstable tenure.</span>
                  </li>
                  <li className="recommendation-item">
                    <span>⛔</span>
                    <span><strong>Mandatory Guarantor:</strong> Application requires a co-signer with credit score &gt; 720 and debt ratio &lt; 0.30.</span>
                  </li>
                  <li className="recommendation-item">
                    <span>⛔</span>
                    <span><strong>Alternative Option:</strong> Offer a reduced secured loan amount or collateralized credit facility.</span>
                  </li>
                </>
              )}
            </ul>
          </div>

          {/* Action Utilities (Copy & Print) */}
          <div className="result-actions-bar">
            <button
              type="button"
              className="btn-action-tool"
              onClick={handleCopyReport}
            >
              <span>{copied ? "✅ Copied to Clipboard!" : "📋 Copy Assessment Report"}</span>
            </button>
            <button
              type="button"
              className="btn-action-tool"
              onClick={handlePrintReport}
            >
              <span>🖨️ Print / Save Summary</span>
            </button>
          </div>
        </div>`
        `
      )}
    </div>
  );
}

export default PredictPage;