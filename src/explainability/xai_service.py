
"""Reusable XAI service for XSepsisNet.

Accepts an already-preprocessed six-hour sequence and returns
raw model probability with SHAP feature/hour contributions.
Does not train or modify the model.
"""

import numpy as np
import torch
import shap

from src.explainability import shap_explainer as core


class XAIService:
    def __init__(self, background_size=20):
        self.config = core.load_config()
        self.clinical_features, self.feature_names = (
            core.load_feature_names()
        )

        self.sequence_length = int(self.config["sequence_length"])
        self.input_size = int(self.config["input_size"])

        self.model = core.load_model(self.config)
        self.model.eval()

        with np.load(core.TEST_FILE) as data:
            X = data["X"]

        expected = (self.sequence_length, self.input_size)
        if X.ndim != 3 or tuple(X.shape[1:]) != expected:
            raise ValueError(
                f"Test data must have shape (samples, {expected[0]}, "
                f"{expected[1]}), received {X.shape}"
            )
        if len(X) == 0:
            raise ValueError("SHAP background dataset is empty.")

        self.background = torch.as_tensor(
            X[:min(background_size, len(X))],
            dtype=torch.float32,
        )

    def explain(self, sequence):
        """Explain one preprocessed sequence with shape (6, 78)."""
        sequence = np.asarray(sequence, dtype=np.float32)
        expected = (self.sequence_length, self.input_size)

        if sequence.shape != expected:
            raise ValueError(
                f"Expected sequence shape {expected}; "
                f"received {sequence.shape}"
            )
        if not np.isfinite(sequence).all():
            raise ValueError("Input contains NaN or infinite values.")

        x = torch.as_tensor(sequence[None, ...], dtype=torch.float32)

        with torch.no_grad():
            raw_probability = float(
                torch.sigmoid(self.model(x)).reshape(-1)[0].item()
            )

        wrapped_model = core.ModelWrapper(self.model)
        wrapped_model.eval()

        try:
            explainer = shap.DeepExplainer(
                wrapped_model, self.background
            )
            result = explainer.shap_values(x)
        except Exception as deep_error:
            try:
                explainer = shap.GradientExplainer(
                    wrapped_model, self.background
                )
                result = explainer.shap_values(x)
            except Exception as gradient_error:
                raise RuntimeError(
                    "Both SHAP explainers failed. "
                    f"DeepExplainer: {deep_error}; "
                    f"GradientExplainer: {gradient_error}"
                ) from gradient_error

        if isinstance(result, list):
            result = result[0]

        values = np.asarray(result)

        # Expected shape: (1, hours, features).
        if values.ndim == 4:
            if values.shape[-1] != 1:
                raise ValueError(
                    f"Unexpected SHAP output shape: {values.shape}"
                )
            values = values[..., 0]

        expected_shap = (1, self.sequence_length, self.input_size)
        if values.shape != expected_shap:
            raise ValueError(
                f"Expected SHAP shape {expected_shap}; "
                f"received {values.shape}"
            )

        values = values[0]

        # Explain clinical features across all six hours.
        clinical = []
        for index, name in enumerate(self.clinical_features):
            contributions = values[:, index]
            clinical.append({
                "feature": name,
                "total_shap": float(contributions.sum()),
                "absolute_shap": float(
                    np.abs(contributions).sum()
                ),
                "most_influential_hour": int(
                    np.argmax(np.abs(contributions)) + 1
                ),
            })

        clinical.sort(
            key=lambda item: item["absolute_shap"],
            reverse=True,
        )

        # Preserve all 78 model features in the hourly output.
        hourly = []
        for hour in range(self.sequence_length):
            for index, name in enumerate(self.feature_names):
                hourly.append({
                    "hour": hour + 1,
                    "feature": name,
                    "shap_value": float(values[hour, index]),
                })

        return {
            "model": "final_bilstm",
            "sequence_length": self.sequence_length,
            "input_features_per_hour": self.input_size,
            "raw_probability": raw_probability,
            "explanation_output": "model_logit",
            "clinical_contributions": clinical,
            "hourly_contributions": hourly,
            "warning": (
                "Research output, not a clinical diagnosis. "
                "Input must use the exact training preprocessing and feature order."
            ),
        }
