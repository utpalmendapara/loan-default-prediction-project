/**
 * Navbar.jsx
 * ==========
 * Top navigation bar component for the CrediPulse AI Platform.
 */

import React from 'react';
import { NavLink, Link } from 'react-router-dom';

function Navbar() {
  return (
    <nav className="navbar navbar-expand-lg custom-navbar fixed-top">
      <div className="container">
        {/* Brand Logo & Title */}
        <Link className="navbar-brand" to="/">
          <span className="brand-icon">🛡️</span>
          <span>CrediPulse AI</span>
          <span className="badge-ai ms-1">v2.4 Pro</span>
        </Link>

        {/* Live Model Status Indicator */}
        <div className="d-none d-md-flex align-items-center ms-3">
          <span className="system-status-chip">
            <span className="status-dot"></span>
            <span>Decision Tree Active</span>
          </span>
        </div>

        {/* Mobile Hamburger Toggle Button */}
        <button
          className="navbar-toggler border-0 shadow-none"
          type="button"
          data-bs-toggle="collapse"
          data-bs-target="#navbarNav"
          aria-controls="navbarNav"
          aria-expanded="false"
          aria-label="Toggle navigation"
        >
          <span className="navbar-toggler-icon"></span>
        </button>

        {/* Navigation Links */}
        <div className="collapse navbar-collapse" id="navbarNav">
          <ul className="navbar-nav ms-auto align-items-center gap-2">
            <li className="nav-item">
              <NavLink 
                to="/" 
                className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
                end
              >
                🏠 Dashboard
              </NavLink>
            </li>
            <li className="nav-item">
              <NavLink 
                to="/predict" 
                className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
              >
                📊 Risk Evaluator
              </NavLink>
            </li>
            <li className="nav-item ms-lg-2">
              <a 
                href="http://localhost:8000/docs" 
                target="_blank" 
                rel="noreferrer" 
                className="btn-nav-api text-decoration-none d-inline-flex align-items-center gap-1"
                title="Open FastAPI Swagger Interactive Docs"
              >
                <span>⚡ FastAPI Docs</span>
              </a>
            </li>
          </ul>
        </div>
      </div>
    </nav>
  );
}

export default Navbar;
