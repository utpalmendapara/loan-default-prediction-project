"""
models/registry.py
==================
The ModelRegistry is a scalable factory / manager for machine learning models.

Why use a registry?
-------------------
1. Decoupling: The API endpoints don't need to know how each model works or where it's stored.
2. Extensibility: When you add future models (e.g. Logistic Regression, KNN, SVC, Random Forest),
   you simply create a new predictor class and call `registry.register(MyNewPredictor())`.
3. Multi-model support: Frontend users or API clients can compare predictions across different models
   by passing `model_id` in their request.
"""

from typing import Dict, List, Optional
from backend.models.base import BaseModelPredictor
from backend.models.decision_tree import DecisionTreePredictor
from backend.models.sklearn_pickle import SklearnPicklePredictor
from backend.schemas import ModelInfo


class ModelRegistry:
    """
    Registry that manages model instances.
    """

    def __init__(self):
        self._models: Dict[str, BaseModelPredictor] = {}
        self._default_model_id: str = "decision_tree"

    def register(self, predictor: BaseModelPredictor, set_as_default: bool = False) -> None:
        """
        Registers a new model predictor in the registry.
        """
        print(f"[REGISTRY] Registering model: '{predictor.model_id}' ({predictor.name})")
        self._models[predictor.model_id] = predictor
        if set_as_default:
            self._default_model_id = predictor.model_id

    def get_model(self, model_id: Optional[str] = None) -> BaseModelPredictor:
        """
        Retrieves a model by its ID. If model_id is None or empty, returns the default model.
        Raises KeyError if the requested model ID is not registered.
        """
        target_id = model_id or self._default_model_id
        if target_id not in self._models:
            available = list(self._models.keys())
            raise KeyError(
                f"Model '{target_id}' is not registered. Available models: {available}"
            )
        return self._models[target_id]

    def list_models(self) -> List[ModelInfo]:
        """
        Returns a list of all registered models as ModelInfo schema objects.
        """
        model_list = []
        for model_id, predictor in self._models.items():
            info = predictor.get_info()
            model_list.append(
                ModelInfo(
                    id=info["id"],
                    name=info["name"],
                    description=info["description"],
                    is_default=(model_id == self._default_model_id),
                    version=info.get("version", "1.0"),
                    accuracy_score=info.get("accuracy_score")
                )
            )
        return model_list

    @property
    def default_model_id(self) -> str:
        return self._default_model_id


# ==============================================================================
# Global Registry Initialization
# ==============================================================================
# We initialize the registry and register every compatible saved model on startup.
# ==============================================================================

model_registry = ModelRegistry()

try:
    decision_tree_predictor = DecisionTreePredictor()
    model_registry.register(decision_tree_predictor, set_as_default=True)
except Exception as e:
    print(f"[ERROR] Failed to initialize DecisionTreePredictor: {e}")


ADDITIONAL_MODELS = [
    {
        "model_id": "gaussian_nb",
        "name": "Gaussian Naive Bayes",
        "description": "Gaussian Naive Bayes classifier trained on the loan-default dataset.",
        "filename": "GaussianNBModel.pkl",
    },
    {
        "model_id": "knn",
        "name": "K-Nearest Neighbors",
        "description": "K-Nearest Neighbors classifier trained on the loan-default dataset.",
        "filename": "KNNModel.pkl",
    },
    {
        "model_id": "logistic_regression",
        "name": "Logistic Regression",
        "description": "Logistic Regression classifier trained on the loan-default dataset.",
        "filename": "LogisticRegressionModel.pkl",
    },
    {
        "model_id": "svc",
        "name": "Support Vector Classifier",
        "description": "Support Vector Classifier trained on the loan-default dataset.",
        "filename": "SVCModel.pkl",
    },
]

for config in ADDITIONAL_MODELS:
    try:
        model_registry.register(SklearnPicklePredictor(**config))
    except Exception as e:
        print(f"[ERROR] Failed to initialize {config['name']}: {e}")
