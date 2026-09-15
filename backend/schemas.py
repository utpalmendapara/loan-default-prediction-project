"""
schemas.py
==========
This file defines the data contracts (Pydantic models) used by FastAPI.
Pydantic ensures that:
1. All incoming request data is validated against the specified types and constraints.
2. Any invalid data generates clear, automatic error responses.
3. Outgoing responses follow a clean, standardized structure.
4. Auto-generated interactive API documentation (Swagger UI at /docs) stays accurate and informative.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class LoanPredictionInput(BaseModel):
    """
    LoanPredictionInput represents the 16 features submitted by the frontend form
    to evaluate whether a loan applicant is likely to default.
    """
    Age: int = Field(
        ...,
        ge=18,
        le=100,
        description="Applicant's age in years (18 to 100)",
        examples=[35]
    )
    Income: float = Field(
        ...,
        ge=0,
        description="Annual income of the applicant in USD",
        examples=[75000.0]
    )
    LoanAmount: float = Field(
        ...,
        ge=0,
        description="Requested loan amount in USD",
        examples=[25000.0]
    )
    CreditScore: int = Field(
        ...,
        ge=300,
        le=850,
        description="Credit score of the applicant (FICO range 300 - 850)",
        examples=[720]
    )
    MonthsEmployed: int = Field(
        ...,
        ge=0,
        description="Total months the applicant has been employed",
        examples=[48]
    )
    NumCreditLines: int = Field(
        ...,
        ge=0,
        description="Total number of existing open credit lines",
        examples=[3]
    )
    InterestRate: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Annual interest rate percentage (e.g. 5.5 for 5.5%)",
        examples=[8.5]
    )
    LoanTerm: int = Field(
        ...,
        ge=1,
        description="Loan repayment duration in months (e.g. 12, 24, 36, 48, 60)",
        examples=[36]
    )
    DTIRatio: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Debt-to-Income ratio as a decimal between 0.0 and 1.0 (e.g. 0.35)",
        examples=[0.32]
    )
    Education: str = Field(
        ...,
        description="Applicant's highest education level: 'High School', \"Bachelor's\", \"Master's\", 'PhD'",
        examples=["Bachelor's"]
    )
    EmploymentType: str = Field(
        ...,
        description="Employment status: 'Full-time', 'Part-time', 'Self-employed', 'Unemployed'",
        examples=["Full-time"]
    )
    MaritalStatus: str = Field(
        ...,
        description="Marital status: 'Single', 'Married', 'Divorced'",
        examples=["Married"]
    )
    HasMortgage: str = Field(
        ...,
        description="Whether the applicant currently has a mortgage: 'Yes' or 'No'",
        examples=["No"]
    )
    HasDependents: str = Field(
        ...,
        description="Whether the applicant has dependents: 'Yes' or 'No'",
        examples=["No"]
    )
    LoanPurpose: str = Field(
        ...,
        description="Purpose for taking the loan: 'Auto', 'Business', 'Education', 'Home', 'Other'",
        examples=["Home"]
    )
    HasCoSigner: str = Field(
        ...,
        description="Whether the loan has a co-signer: 'Yes' or 'No'",
        examples=["Yes"]
    )
    # Optional field allowing frontend/client to select a specific model in the registry
    model_id: Optional[str] = Field(
        default="decision_tree",
        description="ID of the model to use for prediction (defaults to 'decision_tree')",
        examples=["decision_tree"]
    )


class PredictionResponse(BaseModel):
    """
    Standardized response returned to the frontend after running inference.
    """
    model_id: str = Field(..., description="ID of the model that generated this prediction")
    model_name: str = Field(..., description="Human-readable name of the ML model")
    prediction: int = Field(..., description="Binary prediction: 0 for No Default, 1 for Default")
    prediction_label: str = Field(..., description="Readable label: 'No Default (Low Risk)' or 'Default (High Risk)'")
    is_default: bool = Field(..., description="True if predicted to default, False otherwise")
    default_probability: float = Field(..., description="Calculated probability of defaulting (0.0 to 1.0)")
    non_default_probability: float = Field(..., description="Calculated probability of non-defaulting (0.0 to 1.0)")
    risk_level: str = Field(..., description="Categorical risk assessment: 'Low', 'Moderate', or 'High'")
    summary: str = Field(..., description="Human-friendly summary explaining the result")
    timestamp: str = Field(..., description="ISO 8601 timestamp when prediction was generated")


class ModelInfo(BaseModel):
    """
    Describes an available machine learning model in the registry.
    """
    id: str
    name: str
    description: str
    is_default: bool
    version: str
    accuracy_score: Optional[float] = None


class ModelListResponse(BaseModel):
    """
    Response returned by GET /api/models listing all available models.
    """
    models: List[ModelInfo]
    default_model_id: str


class HealthResponse(BaseModel):
    """
    Response returned by GET /api/health for system status monitoring.
    """
    status: str
    message: str
    active_models: List[str]
