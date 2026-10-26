"""Regression tests for inference, calibration, and classification semantics."""

from __future__ import annotations

import json
import unittest
from unittest.mock import patch

import numpy as np
import torch
from fastapi.testclient import TestClient

from backend.main import app
from backend.services.preprocessing_service import (
    FEATURE_COLUMNS,
    SCALER_MEAN,
    SCALER_SCALE,
    preprocess_hours,
)
from backend.services.prediction_service import CLASSIFICATION_THRESHOLD, service
from src.evaluation.manual_sepsis_predictor import FinalBiLSTM


CALIBRATION_COEFFICIENT = 3.7239330966455126
CALIBRATION_INTERCEPT = -3.592676182818465


def _known_hours() -> list[dict[str, float | None]]:
    return [
        {
            feature: (
                None
                if index % 7 == 0
                else float(index + hour)
            )
            for index, feature in enumerate(FEATURE_COLUMNS)
        }
        for hour in range(6)
    ]


class PredictionIntegrationTests(unittest.TestCase):
    def test_backend_preprocessing_and_raw_inference_match_reference(self) -> None:
        hours = _known_hours()
        backend_input = preprocess_hours(hours)

        raw = np.asarray(
            [
                [
                    np.nan if value is None else value
                    for value in (hour[feature] for feature in FEATURE_COLUMNS)
                ]
                for hour in hours
            ],
            dtype=np.float64,
        )
        missing_mask = np.isnan(raw).astype(np.float32)
        imputed = np.where(np.isnan(raw), SCALER_MEAN, raw)
        reference_input = np.concatenate(
            [
                (imputed - SCALER_MEAN) / SCALER_SCALE,
                missing_mask,
            ],
            axis=1,
        ).astype(np.float32)
        np.testing.assert_array_equal(backend_input, reference_input)

        config_path = r"models\bilstm\final_bilstm_config.json"
        checkpoint_path = r"models\bilstm\final_bilstm_best.pt"
        with open(config_path, encoding="utf-8") as file:
            config = json.load(file)
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        reference_model = FinalBiLSTM(
            input_size=config["input_size"],
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"],
            bidirectional=config["bidirectional"],
            dropout=config["dropout"],
        )
        reference_model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        reference_model.eval()

        with torch.no_grad():
            reference_probability = float(
                torch.sigmoid(
                    reference_model(torch.as_tensor(reference_input).unsqueeze(0))
                ).item()
            )

        backend_probability = service.predict_probability(backend_input)
        self.assertAlmostEqual(
            backend_probability,
            reference_probability,
            places=6,
        )

    def test_calibration_matches_persisted_formula_for_multiple_values(self) -> None:
        for raw_probability in (0.0, 0.1, 0.5, 0.813443005, 1.0):
            expected = 1.0 / (
                1.0
                + np.exp(
                    -(
                        CALIBRATION_COEFFICIENT * raw_probability
                        + CALIBRATION_INTERCEPT
                    )
                )
            )
            self.assertAlmostEqual(
                service._calibrated_probability(raw_probability),
                expected,
                places=12,
            )

    def test_prediction_returns_distinct_probabilities_and_applies_calibration_once(
        self,
    ) -> None:
        sequence = np.zeros((6, 78), dtype=np.float32)
        raw_probability = 0.25
        result = None
        with patch.object(
            service,
            "_raw_probability",
            return_value=raw_probability,
        ), patch.object(
            service,
            "_calibrated_probability",
            wraps=service._calibrated_probability,
        ) as calibrate:
            result = service.prediction(sequence)

        expected = service._calibrated_probability(raw_probability)
        self.assertEqual(result["raw_probability"], raw_probability)
        self.assertAlmostEqual(result["calibrated_probability"], expected, places=12)
        self.assertNotEqual(result["raw_probability"], result["calibrated_probability"])
        self.assertEqual(calibrate.call_count, 1)

    def test_classification_uses_raw_probability_at_point_five(self) -> None:
        sequence = np.zeros((6, 78), dtype=np.float32)
        for raw_probability, expected_class in (
            (CLASSIFICATION_THRESHOLD - 1e-8, 0),
            (CLASSIFICATION_THRESHOLD, 1),
            (CLASSIFICATION_THRESHOLD + 1e-8, 1),
        ):
            with patch.object(
                service,
                "_raw_probability",
                return_value=raw_probability,
            ):
                result = service.prediction(sequence)
            self.assertEqual(result["classification_probability"], raw_probability)
            self.assertEqual(result["classification_threshold"], 0.5)
            self.assertEqual(result["predicted_class"], expected_class)

    def test_invalid_input_returns_structured_error(self) -> None:
        client = TestClient(app)
        response = client.post("/predict", json={"hours": [{} for _ in range(5)]})
        self.assertEqual(response.status_code, 422)
        self.assertIn("detail", response.json())


if __name__ == "__main__":
    unittest.main()
