"""
models/decision_tree.py
=======================
Decision Tree model predictor implementation.

This module:
1. Loads the pre-trained DecisionTreeModel.pkl file created in the notebooks.
2. Preprocesses the 16 raw input features into the exact 28 encoded numerical features
   that the Decision Tree classifier was trained on.
3. Performs prediction and probability calculation (predict_proba).
4. Generates a risk analysis and human-readable explanation of the result.
"""

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List
import joblib
import pandas as pd
import numpy as np

from backend.models.base import BaseModelPredictor
from backend.schemas import LoanPredictionInput, PredictionResponse


class DecisionTreePredictor(BaseModelPredictor):
    """
    Predictor wrapper around the pre-trained Scikit-Learn DecisionTreeClassifier.
    """

    # The exact 28 feature names in the exact order expected by the trained model:
    EXPECTED_FEATURES: List[str] = [
        "Age",
        "Income",
        "LoanAmount",
        "CreditScore",
        "MonthsEmployed",
        "NumCreditLines",
        "InterestRate",
        "LoanTerm",
        "DTIRatio",
        "HasMortgage",
        "HasDependents",
        "HasCoSigner",
        "Education_Bachelor's",
        "Education_High School",
        "Education_Master's",
        "Education_PhD",
        "EmploymentType_Full-time",
        "EmploymentType_Part-time",
        "EmploymentType_Self-employed",
        "EmploymentType_Unemployed",
        "MaritalStatus_Divorced",
        "MaritalStatus_Married",
        "MaritalStatus_Single",
        "LoanPurpose_Auto",
        "LoanPurpose_Business",
        "LoanPurpose_Education",
        "LoanPurpose_Home",
        "LoanPurpose_Other"
    ]

    def __init__(self, pkl_path: str = None):
        super().__init__(
            model_id="decision_tree",
            name="Decision Tree Classifier",
            description="Decision Tree model trained with max_depth=8 on 255k+ loan applications (~88% test accuracy)."
        )

        # Locate the pickle file path relative to workspace or backend
        if pkl_path is None:
            # Default location: notebooks-containing-models/DecisionTreeModel.pkl
            # Find the root project folder relative to this file
            current_dir = Path(__file__).resolve().parent
            project_root = current_dir.parent.parent
            self.pkl_path = project_root / "notebooks-containing-models" / "DecisionTreeModel.pkl"
        else:
            self.pkl_path = Path(pkl_path)

        self.load_model()

    def load_model(self) -> None:
        """
        Loads the pickled model from disk using joblib.
        """
        if not self.pkl_path.exists():
            raise FileNotFoundError(
                f"Model file not found at: {self.pkl_path}. Please ensure DecisionTreeModel.pkl is present."
            )

        print(f"[INFO] Loading Decision Tree model from {self.pkl_path}...")
        self.model = joblib.load(self.pkl_path)
        print(f"[INFO] Decision Tree model loaded successfully.")

    def preprocess(self, data: LoanPredictionInput) -> pd.DataFrame:
        """
        Transforms raw user input into a single-row DataFrame with the exact 28 one-hot & label encoded columns.

        How it works:
        1. Numerical columns (Age, Income, LoanAmount, etc.) are used directly as floats/ints.
        2. Binary categorical columns ('HasMortgage', 'HasDependents', 'HasCoSigner') were encoded
           via Scikit-Learn LabelEncoder in the notebook: 'No' -> 0, 'Yes' -> 1.
        3. Multi-class categorical columns ('Education', 'EmploymentType', 'MaritalStatus', 'LoanPurpose')
           were one-hot encoded in the notebook into separate 0.0 or 1.0 indicator columns.
        """
        # 1. Binary features (Label Encoding: 'Yes' = 1, 'No' = 0)
        has_mortgage_val = 1 if data.HasMortgage.strip().lower() in ["yes", "1", "true"] else 0
        has_dependents_val = 1 if data.HasDependents.strip().lower() in ["yes", "1", "true"] else 0
        has_cosigner_val = 1 if data.HasCoSigner.strip().lower() in ["yes", "1", "true"] else 0

        # 2. Build feature dictionary with exact names
        features = {
            "Age": float(data.Age),
            "Income": float(data.Income),
            "LoanAmount": float(data.LoanAmount),
            "CreditScore": float(data.CreditScore),
            "MonthsEmployed": float(data.MonthsEmployed),
            "NumCreditLines": float(data.NumCreditLines),
            "InterestRate": float(data.InterestRate),
            "LoanTerm": float(data.LoanTerm),
            "DTIRatio": float(data.DTIRatio),
            "HasMortgage": has_mortgage_val,
            "HasDependents": has_dependents_val,
            "HasCoSigner": has_cosigner_val,

            # One-hot encoded Education
            "Education_Bachelor's": 1.0 if data.Education == "Bachelor's" else 0.0,
            "Education_High School": 1.0 if data.Education == "High School" else 0.0,
            "Education_Master's": 1.0 if data.Education == "Master's" else 0.0,
            "Education_PhD": 1.0 if data.Education == "PhD" else 0.0,

            # One-hot encoded EmploymentType
            "EmploymentType_Full-time": 1.0 if data.EmploymentType == "Full-time" else 0.0,
            "EmploymentType_Part-time": 1.0 if data.EmploymentType == "Part-time" else 0.0,
            "EmploymentType_Self-employed": 1.0 if data.EmploymentType == "Self-employed" else 0.0,
            "EmploymentType_Unemployed": 1.0 if data.EmploymentType == "Unemployed" else 0.0,

            # One-hot encoded MaritalStatus
            "MaritalStatus_Divorced": 1.0 if data.MaritalStatus == "Divorced" else 0.0,
            "MaritalStatus_Married": 1.0 if data.MaritalStatus == "Married" else 0.0,
            "MaritalStatus_Single": 1.0 if data.MaritalStatus == "Single" else 0.0,

            # One-hot encoded LoanPurpose
            "LoanPurpose_Auto": 1.0 if data.LoanPurpose == "Auto" else 0.0,
            "LoanPurpose_Business": 1.0 if data.LoanPurpose == "Business" else 0.0,
            "LoanPurpose_Education": 1.0 if data.LoanPurpose == "Education" else 0.0,
            "LoanPurpose_Home": 1.0 if data.LoanPurpose == "Home" else 0.0,
            "LoanPurpose_Other": 1.0 if data.LoanPurpose == "Other" else 0.0,
        }

        # Convert dictionary to DataFrame and ensure columns match EXPECTED_FEATURES in order
        df_input = pd.DataFrame([features])[self.EXPECTED_FEATURES]
        return df_input

    def predict(self, data: LoanPredictionInput) -> PredictionResponse:
        """
        Executes prediction on the incoming application data.
        """
        if self.model is None:
            self.load_model()

        # Step 1: Preprocess features into tabular DataFrame
        df_input = self.preprocess(data)

        # Step 2: Run inference
        # predict() returns an array with class label: [0] or [1]
        prediction_arr = self.model.predict(df_input)
        raw_prediction = int(prediction_arr[0])

        # predict_proba() returns probabilities for [Class 0 (No Default), Class 1 (Default)]
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(df_input)[0]
            non_default_prob = float(probabilities[0])
            default_prob = float(probabilities[1])
        else:
            # Fallback if probabilities are not available
            default_prob = 1.0 if raw_prediction == 1 else 0.0
            non_default_prob = 1.0 - default_prob

        # Step 3: Compute risk tier and descriptive summary
        if default_prob >= 0.5:
            risk_level = "High"
            prediction_label = "Default (High Risk)"
            is_default = True
            summary = (
                f"Applicant exhibits a high estimated default probability of {default_prob * 100:.1f}%. "
                f"Key risk indicators may include debt-to-income ratio ({data.DTIRatio * 100:.0f}%), "
                f"credit score ({data.CreditScore}), or employment status ({data.EmploymentType})."
            )
        elif default_prob >= 0.25:
            risk_level = "Moderate"
            prediction_label = "No Default (Moderate Risk)"
            is_default = False
            summary = (
                f"Applicant is predicted not to default with a moderate risk score ({default_prob * 100:.1f}% default probability). "
                f"Further manual underwriting review or collateral verification is recommended."
            )
        else:
            risk_level = "Low"
            prediction_label = "No Default (Low Risk)"
            is_default = False
            summary = (
                f"Applicant has a strong profile with an estimated default probability of only {default_prob * 100:.1f}%. "
                f"Low risk profile with favorable income-to-loan ratios."
            )

        # Step 4: Assemble and return structured response
        return PredictionResponse(
            model_id=self.model_id,
            model_name=self.name,
            prediction=raw_prediction,
            prediction_label=prediction_label,
            is_default=is_default,
            default_probability=round(default_prob, 4),
            non_default_probability=round(non_default_prob, 4),
            risk_level=risk_level,
            summary=summary,
            timestamp=datetime.now(timezone.utc).isoformat()
        )

    def get_info(self) -> Dict[str, Any]:
        """
        Returns model metadata.
        """
        return {
            "id": self.model_id,
            "name": self.name,
            "description": self.description,
            "is_default": True,
            "version": "1.0",
            "accuracy_score": 0.88
        }
