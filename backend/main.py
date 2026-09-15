"""
main.py
=======
FastAPI Application Entry Point.

This file sets up:
1. The FastAPI application instance with metadata and docs config.
2. CORS (Cross-Origin Resource Sharing) middleware to allow browser requests
   from the React frontend (running on http://localhost:5173 or other ports).
3. API Endpoints:
   - GET  /api/health   -> Check backend health status
   - GET  /api/models   -> List available models (scalable multi-model support)
   - POST /api/predict  -> Run inference using the selected (or default) model
4. Uvicorn server launcher when executed directly with `python main.py`.
"""

import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.schemas import (
    LoanPredictionInput,
    PredictionResponse,
    ModelListResponse,
    HealthResponse
)
from backend.models.registry import model_registry

# ==============================================================================
# 1. Initialize FastAPI Application
# ==============================================================================
app = FastAPI(
    title="Loan Default Prediction API",
    description=(
        "Production-ready FastAPI backend for evaluating loan default risk "
        "using Machine Learning models (Decision Tree and future models)."
    ),
    version="1.0.0",
    docs_url="/docs",      # Interactive Swagger UI docs available at /docs
    redoc_url="/redoc"     # ReDoc documentation available at /redoc
)

# ==============================================================================
# 2. CORS Middleware Configuration
# ==============================================================================
# Why CORS? When your React frontend (e.g. running on http://localhost:5173) makes
# requests to your FastAPI backend (http://localhost:8000), web browsers block
# cross-origin requests by default for security. Adding CORSMiddleware allows
# the frontend to communicate with the backend smoothly.
# ==============================================================================
origins = [
    "http://localhost:5173",  # Default Vite dev server port
    "http://127.0.0.1:5173",
    "http://localhost:3000",  # Standard React port
    "http://127.0.0.1:3000",
    "*"                       # Allow all origins during local development
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],      # Allow all HTTP methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],      # Allow all HTTP headers
)


# ==============================================================================
# 3. API Routes
# ==============================================================================

@app.get(
    "/api/health",
    response_model=HealthResponse,
    summary="Health Check",
    tags=["System"]
)
def health_check() -> HealthResponse:
    """
    Returns the server health status and list of registered ML models.
    """
    models = [m.id for m in model_registry.list_models()]
    return HealthResponse(
        status="healthy",
        message="Loan Default Prediction API is running smoothly.",
        active_models=models
    )


@app.get(
    "/api/models",
    response_model=ModelListResponse,
    summary="List Available Models",
    tags=["Models"]
)
def get_models() -> ModelListResponse:
    """
    Returns a list of all registered machine learning models.
    This enables the frontend to dynamically populate a model selector dropdown.
    """
    models = model_registry.list_models()
    return ModelListResponse(
        models=models,
        default_model_id=model_registry.default_model_id
    )


@app.post(
    "/api/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict Loan Default Risk",
    tags=["Prediction"]
)
def predict_loan_default(input_data: LoanPredictionInput) -> PredictionResponse:
    """
    Evaluates a loan applicant's details and predicts default risk.

    - **input_data**: Contains 16 applicant attributes (Age, Income, CreditScore, LoanAmount, etc.)
    - **model_id**: Optional model identifier (defaults to 'decision_tree').
    """
    try:
        # Retrieve the requested model from the registry
        model_id = input_data.model_id or model_registry.default_model_id
        predictor = model_registry.get_model(model_id)

        # Run inference through the model predictor
        result = predictor.predict(input_data)
        return result

    except KeyError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred during prediction: {str(e)}"
        )


# Root route for quick verification in browser
@app.get("/", tags=["System"])
def root():
    return {
        "message": "Welcome to the Loan Default Prediction API",
        "docs": "/docs",
        "models_endpoint": "/api/models",
        "predict_endpoint": "/api/predict"
    }


# ==============================================================================
# 4. Entrypoint for running with `python main.py`
# ==============================================================================
if __name__ == "__main__":
    print("[INFO] Starting FastAPI server on http://localhost:8000...")
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
