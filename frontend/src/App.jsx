/**
 * App.jsx
 * =======
 * Main Application Component for CrediPulse AI.
 */

import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import HomePage from './pages/HomePage';
import PredictPage from './pages/Predict';
import NotFound from './pages/NotFound';
import './App.css';

function App() {
  return (
    <Router>
      {/* Top persistent navigation bar */}
      <Navbar />

      {/* Main page content area */}
      <main className="main-wrapper">
        <div className="container">
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/predict" element={<PredictPage />} />
            <Route path="*" element={<NotFound />} />
          </Routes>
        </div>
      </main>

      {/* Modern Footer */}
      <footer className="py-4 border-top bg-white mt-auto">
        <div className="container">
          <div className="d-flex flex-column flex-md-row justify-content-between align-items-center gap-2">
            <div>
              <span className="fw-bold text-dark">🛡️ CrediPulse AI</span>
              <span className="text-muted small ms-2">&bull; Credit Risk Analytics &amp; Underwriting Intelligence</span>
            </div>
            <div className="text-muted small">
              <span>Powered by Decision Tree Classifier &bull; FastAPI Asynchronous Service</span>
            </div>
          </div>
        </div>
      </footer>
    </Router>
  );
}

export default App;
