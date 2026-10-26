"""Focused tests for the FastAPI foundation and clinical preprocessing."""

from __future__ import annotations

import unittest

import numpy as np

from backend.main import app
from backend.services.preprocessing_service import (
    FEATURE_COLUMNS,
    SCALER_MEAN,
    SCALER_SCALE,
    preprocess_hours,
)


class FastAPIFoundationTests(unittest.TestCase):
    def test_app_generates_openapi_schema(self) -> None:
        schema = app.openapi()
        self.assertEqual(schema["info"]["title"], "XSepsisNet")
        paths = schema["paths"]
        self.assertIn("/health", paths)
        self.assertIn("/predict", paths)
        self.assertIn("/explainability", paths)


class PreprocessingTests(unittest.TestCase):
    def test_output_shape_order_and_missingness(self) -> None:
        hours = [
            {"heart_rate": 100.0, "wbc": 8.0}
            for _ in range(6)
        ]
        result = preprocess_hours(hours)

        self.assertEqual(result.shape, (6, 78))
        self.assertEqual(result.dtype, np.float32)
        self.assertAlmostEqual(
            float(result[0, 0]),
            (100.0 - SCALER_MEAN[0]) / SCALER_SCALE[0],
            places=6,
        )
        self.assertAlmostEqual(
            float(result[0, 38]),
            (8.0 - SCALER_MEAN[38]) / SCALER_SCALE[38],
            places=6,
        )
        self.assertTrue(np.all(result[:, 39:39 + 39] >= 0))
        self.assertEqual(float(result[0, 39]), 0.0)
        self.assertEqual(float(result[0, 40]), 1.0)
        self.assertEqual(float(result[0, 77]), 0.0)

    def test_scaling_matches_reference_formula(self) -> None:
        hours = [{feature: float(index) for index, feature in enumerate(FEATURE_COLUMNS)} for _ in range(6)]
        result = preprocess_hours(hours)
        expected_clinical = (
            np.arange(39, dtype=np.float64) - SCALER_MEAN
        ) / SCALER_SCALE
        np.testing.assert_allclose(
            result[:, :39],
            np.broadcast_to(expected_clinical.astype(np.float32), (6, 39)),
        )
        np.testing.assert_array_equal(result[:, 39:], np.zeros((6, 39), dtype=np.float32))

    def test_invalid_hour_count_and_feature_name(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly 6"):
            preprocess_hours([{}] * 5)
        with self.assertRaisesRegex(ValueError, "unknown feature"):
            preprocess_hours([{"not_a_feature": 1.0}] * 6)

    def test_non_finite_values_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            preprocess_hours([{"heart_rate": float("nan")}] * 6)
        with self.assertRaisesRegex(ValueError, "finite"):
            preprocess_hours([{"heart_rate": float("inf")}] * 6)


if __name__ == "__main__":
    unittest.main()
