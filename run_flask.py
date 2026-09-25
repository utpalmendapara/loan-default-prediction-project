"""
run_flask.py
============
Runner script for the Loan Default Prediction Flask Application.
Usage:
    python run_flask.py
"""

import os
import sys
from pathlib import Path

# Ensure root directory is in path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from flask_app.app import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n=======================================================")
    print(f"[*] CrediPulse AI Flask Application Starting...")
    print(f"[>] Local URL: http://127.0.0.1:{port}")
    print(f"[>] Form Predictor: http://127.0.0.1:{port}/predict")
    print(f"[>] SOP Metrics: http://127.0.0.1:{port}/metrics")
    print(f"=======================================================\n")
    app.run(host="0.0.0.0", port=port, debug=False)
