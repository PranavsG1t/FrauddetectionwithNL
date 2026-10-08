"""Model loading + scoring. Knows nothing about HTTP.

Imports lightgbm/shap (via joblib) but NOT torch or the Gemini client - see
the load-order rules in the blueprint's Day 6 notes.
"""
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data.preprocess import FEATURE_COLUMNS

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def default_model_dir() -> Path:
    p = Path(os.environ.get("MODEL_DIR", "outputs/models"))
    return p if p.is_absolute() else PROJECT_ROOT / p


class ModelService:
    def __init__(self, model, explainer):
        self.model = model
        self.explainer = explainer

    @classmethod
    def load(cls, model_dir: Path) -> "ModelService":
        model = joblib.load(model_dir / "lightgbm.joblib")
        explainer = joblib.load(model_dir / "shap_explainer.joblib")
        return cls(model, explainer)

    def score(self, features: dict) -> tuple[float, dict]:
        """Returns (risk_score 0-1, {feature: shap_value} sorted by |impact|)."""
        # Column order must match training exactly.
        X = pd.DataFrame([features], columns=FEATURE_COLUMNS)

        if hasattr(self.model, "predict_proba"):
            risk_score = float(self.model.predict_proba(X)[0, 1])
        else:  # raw lightgbm Booster returns P(class 1) directly
            risk_score = float(self.model.predict(X)[0])

        shap_row = self._positive_class_shap(self.explainer.shap_values(X))
        shap_dict = {name: float(v) for name, v in zip(FEATURE_COLUMNS, shap_row)}
        shap_dict = dict(sorted(shap_dict.items(), key=lambda kv: abs(kv[1]), reverse=True))
        return risk_score, shap_dict

    @staticmethod
    def _positive_class_shap(values) -> np.ndarray:
        """shap's return shape differs by version/model:
        list of two arrays | (n, f, 2) array | (n, f) array. Normalise to
        the fraud-class row for the single input row."""
        if isinstance(values, list):
            values = values[1] if len(values) == 2 else values[0]
        values = np.asarray(values)
        if values.ndim == 3:
            values = values[..., 1]
        return values[0]