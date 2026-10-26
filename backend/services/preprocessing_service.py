"""Preprocess six hourly clinical records for the final BiLSTM."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCALER_FILE = PROJECT_ROOT / "models" / "scalers" / "final_bilstm_scaler.json"
SEQUENCE_LENGTH = 6
MODEL_FEATURES_PER_HOUR = 78


def _load_scaler() -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    with SCALER_FILE.open("r", encoding="utf-8") as file:
        scaler = json.load(file)

    feature_columns = tuple(scaler["feature_columns"])
    means = np.asarray(scaler["mean"], dtype=np.float64)
    scales = np.asarray(scaler["scale"], dtype=np.float64)

    if len(feature_columns) != 39 or means.shape != (39,) or scales.shape != (39,):
        raise ValueError("Final scaler must contain 39 feature names, means, and scales.")
    if not np.isfinite(means).all() or not np.isfinite(scales).all():
        raise ValueError("Final scaler contains non-finite values.")
    if np.any(scales == 0):
        raise ValueError("Final scaler contains a zero scale.")

    return feature_columns, means, scales


FEATURE_COLUMNS, SCALER_MEAN, SCALER_SCALE = _load_scaler()
FEATURE_INDEX = {name: index for index, name in enumerate(FEATURE_COLUMNS)}


def _validate_hour(hour: Mapping[str, Any], hour_index: int) -> np.ndarray:
    if not isinstance(hour, Mapping):
        raise ValueError(f"Hour {hour_index + 1} must be an object of clinical features.")

    unknown = sorted(set(hour) - set(FEATURE_COLUMNS))
    if unknown:
        raise ValueError(
            f"Hour {hour_index + 1} contains unknown feature names: {unknown}"
        )

    values = np.full(len(FEATURE_COLUMNS), np.nan, dtype=np.float64)
    for feature, value in hour.items():
        if value is None or value == "":
            continue
        if isinstance(value, bool):
            raise ValueError(
                f"Hour {hour_index + 1} feature {feature!r} must be numeric or null."
            )
        try:
            numeric = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Hour {hour_index + 1} feature {feature!r} must be numeric or null."
            ) from error
        if not np.isfinite(numeric):
            raise ValueError(
                f"Hour {hour_index + 1} feature {feature!r} must be finite."
            )
        values[FEATURE_INDEX[feature]] = numeric

    return values


def preprocess_hours(hours: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """Convert six 39-feature records into the model's six-by-78 input."""
    if not isinstance(hours, Sequence) or isinstance(hours, (str, bytes)):
        raise ValueError("hours must be a sequence of six feature objects.")
    if len(hours) != SEQUENCE_LENGTH:
        raise ValueError(f"Expected exactly {SEQUENCE_LENGTH} hourly records; received {len(hours)}.")

    raw_data = np.vstack(
        [_validate_hour(hour, index) for index, hour in enumerate(hours)]
    )
    missing_mask = np.isnan(raw_data).astype(np.float32)
    imputed_data = np.where(np.isnan(raw_data), SCALER_MEAN, raw_data)
    scaled_data = (imputed_data - SCALER_MEAN) / SCALER_SCALE
    model_input = np.concatenate([scaled_data, missing_mask], axis=1).astype(np.float32)

    if model_input.shape != (SEQUENCE_LENGTH, MODEL_FEATURES_PER_HOUR):
        raise ValueError(f"Unexpected model input shape: {model_input.shape}")
    if not np.isfinite(model_input).all():
        raise ValueError("Preprocessed model input contains NaN or infinite values.")

    return model_input


def _as_model_sequence(sequence: Any) -> np.ndarray:
    array = np.asarray(sequence, dtype=np.float32)
    if array.shape != (SEQUENCE_LENGTH, MODEL_FEATURES_PER_HOUR):
        raise ValueError(
            "Preprocessed sequences must have shape "
            f"({SEQUENCE_LENGTH}, {MODEL_FEATURES_PER_HOUR}); received {array.shape}."
        )
    if not np.isfinite(array).all():
        raise ValueError("Preprocessed sequence contains NaN or infinite values.")
    return array


def build_sequence(payload: Mapping[str, Any]) -> np.ndarray:
    """Build model input from raw six-hour records or accept a prepared sequence."""
    if not isinstance(payload, Mapping):
        raise ValueError("Request payload must be an object.")

    if payload.get("hours") is not None:
        return preprocess_hours(payload["hours"])
    if payload.get("sequence") is not None:
        return _as_model_sequence(payload["sequence"])
    if payload.get("features") is not None:
        return _as_model_sequence(payload["features"])
    raise ValueError("Request must include six raw hourly records in 'hours'.")


__all__ = [
    "FEATURE_COLUMNS",
    "MODEL_FEATURES_PER_HOUR",
    "SCALER_MEAN",
    "SCALER_SCALE",
    "SEQUENCE_LENGTH",
    "build_sequence",
    "preprocess_hours",
]
