"""
models/base.py
==============
Defines the abstract base class (BaseModelPredictor) that all ML models must inherit from.
This enforces a consistent interface across different algorithms (Decision Tree, Logistic
Regression, Random Forest, KNN, etc.), making the system completely scalable.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
from backend.schemas import LoanPredictionInput, PredictionResponse


class BaseModelPredictor(ABC):
    """
    Abstract Base Class for all machine learning model predictors.
    Any new model added to the system in the future should subclass this and implement:
    1. `load_model()`: How to load model weights/artifacts.
    2. `predict()`: How to preprocess input, run inference, and format the output.
    3. `get_info()`: Metadata about the model (name, description, accuracy, etc.).
    """

    def __init__(self, model_id: str, name: str, description: str):
        self.model_id = model_id
        self.name = name
        self.description = description
        self.model = None

    @abstractmethod
    def load_model(self) -> None:
        """
        Loads the trained model artifact from disk (e.g. .pkl file).
        """
        pass

    @abstractmethod
    def predict(self, data: LoanPredictionInput) -> PredictionResponse:
        """
        Executes preprocessing and model inference for a single loan application.
        Returns a standardized PredictionResponse.
        """
        pass

    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """
        Returns metadata describing the model.
        """
        pass
