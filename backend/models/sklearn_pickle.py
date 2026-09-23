"""Reusable predictor for the project's scikit-learn pickle models.

Each saved model was trained on the same 28 encoded loan features.  Reusing the
DecisionTreePredictor's preprocessing therefore keeps every model compatible
with the single prediction form.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import joblib

from backend.models.base import BaseModelPredictor
from backend.models.decision_tree import DecisionTreePredictor


class SklearnPicklePredictor(DecisionTreePredictor):
    """Load a compatible scikit-learn pickle and use the common form transform."""

    def __init__(
        self,
        model_id: str,
        name: str,
        description: str,
        filename: str,
        accuracy_score: Optional[float] = None,
    ) -> None:
        # Do not call DecisionTreePredictor.__init__: it always loads the tree.
        BaseModelPredictor.__init__(self, model_id, name, description)
        self.accuracy_score = accuracy_score
        project_root = Path(__file__).resolve().parent.parent.parent
        self.pkl_path = project_root / "notebooks-containing-models" / filename
        self.model = None

    def load_model(self) -> None:
        if self.model is not None:
            return
        if not self.pkl_path.exists():
            raise FileNotFoundError(f"Model file not found at: {self.pkl_path}")
        print(f"[INFO] Loading {self.name} from {self.pkl_path}...")
        self.model = joblib.load(self.pkl_path)

    def get_info(self) -> Dict[str, Any]:
        return {
            "id": self.model_id,
            "name": self.name,
            "description": self.description,
            "version": "1.0",
            "accuracy_score": self.accuracy_score,
        }
