/**
 * HomePage.jsx
 * ============
 * Landing page for CrediPulse AI Platform.
 * Displays key underwriting metrics, technical architecture highlights,
 * and key credit default risk factors before guiding users to the evaluator.
 */

import React from "react";
import { Link } from "react-router-dom";

function HomePage() {
  return (
    <div className="home-container">
      {/* 1. HERO SECTION */}
      <section className="hero-section">
        <div className="hero-badge">
          <span>🌲 Decision Tree Classifier Engine &bull; Max Depth = 8</span>
        </div>
        <h1 className="hero-title">
          Intelligent <span className="gradient-text">Loan Default Risk</span> Analytics
        </h1>
        <p className="hero-subtitle">
          Evaluate borrower creditworthiness and predict loan default probabilities in real-time
          leveraging machine learning trained on 255,000+ historical financial records.
        </p>

        {/* Live Metrics Strip */}
        <div className="stats-banner">
          <div className="stat-item">
            <span className="stat-icon">📈</span>
            <div className="text-start">
              <div className="stat-value">255,000+</div>
              <div className="stat-label">Training Records</div>
            </div>
          </div>
          <div className="stat-divider"></div>
          <div className="stat-item">
            <span className="stat-icon">🎯</span>
            <div className="text-start">
              <div className="stat-value">~88.4%</div>
              <div className="stat-label">Model Accuracy</div>
            </div>
          </div>
          <div className="stat-divider"></div>
          <div className="stat-item">
            <span className="stat-icon">⚡</span>
            <div className="text-start">
              <div className="stat-value">&lt; 10ms</div>
              <div className="stat-label">Inference Latency</div>
            </div>
          </div>
        </div>

        {/* Call to Actions */}
        <div className="d-flex justify-content-center gap-3 flex-wrap">
          <Link to="/predict" className="btn-cta-primary text-decoration-none">
            <span>🚀 Launch Risk Evaluator</span>
          </Link>
          <a
            href="http://localhost:8000/docs"
            target="_blank"
            rel="noreferrer"
            className="btn-cta-secondary text-decoration-none"
          >
            <span>⚡ View Swagger API</span>
          </a>
        </div>
      </section>

      {/* 2. CORE CAPABILITIES (4 CARDS) */}
      <section className="row g-4 my-3">
        <div className="col-md-6 col-lg-3">
          <div className="feature-card">
            <div className="feature-icon-wrapper">📊</div>
            <h4 className="feature-title">Predictive Scoring</h4>
            <p className="feature-desc">
              Scikit-Learn Decision Tree model optimized to detect subtle borrower risk flags with transparent tree boundaries.
            </p>
          </div>
        </div>

        <div className="col-md-6 col-lg-3">
          <div className="feature-card">
            <div className="feature-icon-wrapper">⚡</div>
            <h4 className="feature-title">Real-Time FastAPI</h4>
            <p className="feature-desc">
              High-throughput asynchronous Python backend serving predictions with strict Pydantic input schema validation.
            </p>
          </div>
        </div>

        <div className="col-md-6 col-lg-3">
          <div className="feature-card">
            <div className="feature-icon-wrapper">🛡️</div>
            <h4 className="feature-title">28 Feature Encoding</h4>
            <p className="feature-desc">
              Dynamic pipeline encoding raw categorical fields (education, marital status, loan intent) into numerical space.
            </p>
          </div>
        </div>

        <div className="col-md-6 col-lg-3">
          <div className="feature-card">
            <div className="feature-icon-wrapper">💡</div>
            <h4 className="feature-title">Decision Intelligence</h4>
            <p className="feature-desc">
              Instant probability breakdown and risk tiers (Low, Moderate, High) with actionable underwriting recommendations.
            </p>
          </div>
        </div>
      </section>

      {/* 3. KEY UNDERWRITING RISK FACTORS */}
      <section className="methodology-box my-4">
        <div className="text-center mb-4">
          <h3 className="fw-bold mb-2">Key Risk Factors Evaluated by the Model</h3>
          <p className="text-muted small">
            Our ML model assesses 16 core attributes across 4 distinct financial dimensions:
          </p>
        </div>

        <div className="row g-3">
          <div className="col-md-6">
            <div className="factor-pill">
              <span className="factor-num">1</span>
              <div>
                <strong className="d-block text-dark">Debt-to-Income (DTI) Ratio</strong>
                <span className="text-muted small">
                  Compares monthly debt obligations to gross income. Values above 0.40 significantly elevate default probability.
                </span>
              </div>
            </div>
          </div>

          <div className="col-md-6">
            <div className="factor-pill">
              <span className="factor-num">2</span>
              <div>
                <strong className="d-block text-dark">Credit Score &amp; Credit Lines</strong>
                <span className="text-muted small">
                  FICO score profile and active borrowing lines indicate historical repayment discipline and credit utilization.
                </span>
              </div>
            </div>
          </div>

          <div className="col-md-6">
            <div className="factor-pill">
              <span className="factor-num">3</span>
              <div>
                <strong className="d-block text-dark">Employment Tenure &amp; Type</strong>
                <span className="text-muted small">
                  Months continuously employed and employment stability (Full-time, Self-employed, Unemployed).
                </span>
              </div>
            </div>
          </div>

          <div className="col-md-6">
            <div className="factor-pill">
              <span className="factor-num">4</span>
              <div>
                <strong className="d-block text-dark">Loan Term &amp; Co-Signer Guarantee</strong>
                <span className="text-muted small">
                  Duration of loan payback, interest rate APR, and presence of a secondary guarantor or co-signer.
                </span>
              </div>
            </div>
          </div>
        </div>

        <div className="text-center mt-4 pt-2">
          <Link to="/predict" className="btn-cta-primary text-decoration-none">
            Test a Borrower Application &rarr;
          </Link>
        </div>
      </section>
    </div>
  );
}

export default HomePage;