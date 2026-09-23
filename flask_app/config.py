"""
config.py
=========
Configuration settings for the Loan Default Prediction Flask Application.
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "ml-project-sop-flask-secret-2026")
    DATA_PATH = os.environ.get("DATA_PATH", str(BASE_DIR / "data" / "Loan_default.csv"))
    MODEL_DIR = os.environ.get("MODEL_DIR", str(BASE_DIR / "notebooks-containing-models"))
    REPORTS_DIR = os.environ.get("REPORTS_DIR", str(BASE_DIR / "reports"))
    DEBUG = False
    TESTING = False

class DevelopmentConfig(Config):
    DEBUG = True

class ProductionConfig(Config):
    DEBUG = False

config_by_name = {
    "dev": DevelopmentConfig,
    "prod": ProductionConfig,
    "default": DevelopmentConfig
}
