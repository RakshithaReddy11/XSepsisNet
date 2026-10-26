"""Prediction, calibration, and SHAP service for XSepsisNet."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.explainability.xai_service import XAIService


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CALIBRATOR_FILE = (
    PROJECT_ROOT / "models" / "evaluation" / "final_bilstm_platt_calibrator.json"
)
CLASSIFICATION_THRESHOLD = 0.50


class PredictionService:
    """Load the model artifacts once and expose consistent prediction outputs."""

    def __init__(self) -> None:
        self.xai_service = XAIService()
        with CALIBRATOR_FILE.open("r", encoding="utf-8") as file:
            calibrator = json.load(file)
        self.calibration_coefficient = float(calibrator["coefficient"])
        self.calibration_intercept = float(calibrator["intercept"])
        if not np.isfinite(
            [self.calibration_coefficient, self.calibration_intercept]
        ).all():
            raise ValueError("Calibration artifact contains non-finite parameters.")

    def _prepare_sequence(self, sequence: Any) -> np.ndarray:
        array = np.asarray(sequence, dtype=np.float32)
        expected = (
            self.xai_service.sequence_length,
            self.xai_service.input_size,
        )
        if array.shape != expected:
            raise ValueError(
                f"Expected preprocessed sequence shape {expected}; "
                f"received {array.shape}."
            )
        if not np.isfinite(array).all():
            raise ValueError("Preprocessed sequence contains NaN or infinite values.")
        return array

    def _raw_probability(self, sequence: np.ndarray) -> float:
        x = torch.as_tensor(sequence[None, ...], dtype=torch.float32)
        with torch.no_grad():
            logits = self.xai_service.model(x)
            return float(torch.sigmoid(logits).reshape(-1)[0].item())

    def _calibrated_probability(self, raw_probability: float) -> float:
        calibrated_logit = (
            self.calibration_coefficient * raw_probability
            + self.calibration_intercept
        )
        return float(
            torch.sigmoid(torch.tensor(calibrated_logit, dtype=torch.float64)).item()
        )

    def prediction(self, sequence: Any) -> dict[str, Any]:
        """Return outputs using raw probability at the fixed 0.50 threshold.

        The calibrated probability is returned separately for risk reporting and
        does not currently drive the classification decision.
        """
        array = self._prepare_sequence(sequence)
        raw_probability = self._raw_probability(array)
        calibrated_probability = self._calibrated_probability(raw_probability)
        classification_probability = raw_probability
        predicted_class = int(
            classification_probability >= CLASSIFICATION_THRESHOLD
        )
        return {
            "model": "final_bilstm",
            "sequence_length": self.xai_service.sequence_length,
            "clinical_features_per_hour": len(self.xai_service.clinical_features),
            "input_features_per_hour": self.xai_service.input_size,
            "raw_probability": raw_probability,
            "calibrated_probability": calibrated_probability,
            "classification_probability": classification_probability,
            "classification_threshold": CLASSIFICATION_THRESHOLD,
            "predicted_class": predicted_class,
            "risk_level": "high" if predicted_class else "low",
        }

    def predict_probability(self, sequence: Any) -> float:
        """Return the raw probability for compatibility with existing callers."""
        return float(self.prediction(sequence)["raw_probability"])

    def explain(self, sequence: Any) -> dict[str, Any]:
        """Explain the exact model-ready sequence used for prediction."""
        array = self._prepare_sequence(sequence)
        result = self.xai_service.explain(array)
        result.update(self.prediction(array))
        return result


service = PredictionService()

__all__ = ["CLASSIFICATION_THRESHOLD", "PredictionService", "service"]
