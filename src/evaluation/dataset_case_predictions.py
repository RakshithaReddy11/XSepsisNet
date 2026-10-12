
import sys
import json
import csv
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "evaluation"))

from manual_sepsis_predictor import (
    FinalBiLSTM,
    FEATURE_COLUMNS,
    DISPLAY_NAMES,
    load_json,
    MODEL_FILE,
    CONFIG_FILE,
    CALIBRATOR_FILE,
)

DATA_FILE = ROOT / "data" / "processed" / "final_bilstm_test.npz"
OUT = ROOT / "outputs" / "patient_xai"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# DATASET
# ============================================================

def load_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    with np.load(DATA_FILE, allow_pickle=False) as d:
        X = d["X"].astype(np.float32)
        y = d["y"].astype(np.int64)

    if X.ndim != 3 or len(X) != len(y):
        raise ValueError(f"Invalid shapes: X={X.shape}, y={y.shape}")

    if X.shape[2] != 78:
        raise ValueError(
            f"Expected 78 model inputs, received {X.shape[2]}"
        )

    if not np.isfinite(X).all():
        raise ValueError("Dataset contains NaN or infinite values.")

    if not np.isin(y, [0, 1]).all():
        raise ValueError("Expected binary labels 0 and 1.")

    if len(FEATURE_COLUMNS) != 39:
        raise ValueError(
            f"Expected 39 original feature names, got "
            f"{len(FEATURE_COLUMNS)}. Check the training feature list."
        )

    print("\n" + "=" * 72)
    print("X-SepsisNet: TWO-PATIENT PREDICTION AND XAI")
    print("=" * 72)
    print(f"Test samples: {len(X)}")
    print(f"Sequence length: {X.shape[1]} hours")
    print(f"Original clinical features: {len(FEATURE_COLUMNS)}")
    print(f"Model input features: {X.shape[2]}")
    print(f"Positive dataset labels: {int((y == 1).sum())}")
    print(f"Negative dataset labels: {int((y == 0).sum())}")

    return X, y


# ============================================================
# MODEL
# ============================================================

def load_model():
    for p in (MODEL_FILE, CONFIG_FILE, CALIBRATOR_FILE):
        if not Path(p).exists():
            raise FileNotFoundError(f"Missing project file: {p}")

    config = load_json(CONFIG_FILE)

    model = FinalBiLSTM(
        input_size=int(config["input_size"]),
        hidden_size=int(config["hidden_size"]),
        num_layers=int(config["num_layers"]),
        bidirectional=bool(config["bidirectional"]),
        dropout=float(config["dropout"]),
    )

    checkpoint = torch.load(
        MODEL_FILE,
        map_location=DEVICE,
        weights_only=False,
    )

    if "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    else:
        state = checkpoint

    model.load_state_dict(state, strict=True)
    model.to(DEVICE)
    model.eval()

    calibrator = load_json(CALIBRATOR_FILE)
    coefficient = float(calibrator["coefficient"])
    intercept = float(calibrator["intercept"])

    threshold = float(config.get("classification_threshold", 0.50))

    print("\nTrained BiLSTM loaded.")
    print(f"Device: {DEVICE}")
    print(f"Classification threshold: {threshold:.2f}")

    return model, coefficient, intercept, threshold, config


# ============================================================
# PREDICTION
# ============================================================

def predict_raw(model, X):
    result = []

    with torch.no_grad():
        for start in range(0, len(X), 128):
            batch = torch.as_tensor(
                X[start:start + 128],
                dtype=torch.float32,
                device=DEVICE,
            )

            logits = model(batch)

            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            elif isinstance(logits, dict):
                logits = logits["logits"]

            result.extend(
                torch.sigmoid(logits.reshape(-1))
                .cpu().numpy().tolist()
            )

    return np.asarray(result, dtype=np.float64)


def calibrate(raw, coefficient, intercept):
    # Preserve the formula used by the existing project predictor.
    z = coefficient * raw + intercept
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))


# ============================================================
# SELECT ONE POSITIVE-LABEL AND ONE NEGATIVE-LABEL PATIENT
# ============================================================

def select_patients(y, probabilities):
    positive_ids = np.flatnonzero(y == 1)
    negative_ids = np.flatnonzero(y == 0)

    if not len(positive_ids) or not len(negative_ids):
        raise ValueError("Both dataset label classes are required.")

    # Choose the highest-scoring positive-labelled sample.
    pos_idx = int(
        positive_ids[np.argmax(probabilities[positive_ids])]
    )

    # Choose the lowest-scoring negative-labelled sample.
    neg_idx = int(
        negative_ids[np.argmin(probabilities[negative_ids])]
    )

    return pos_idx, neg_idx


# ============================================================
# DISPLAY THE 39 ORIGINAL FEATURE VALUES
# ============================================================

def feature_names():
    if isinstance(DISPLAY_NAMES, dict):
        return [
            str(DISPLAY_NAMES.get(name, name))
            for name in FEATURE_COLUMNS
        ]

    return [str(name) for name in FEATURE_COLUMNS]


def display_patient(title, idx, X, y, probability, threshold):
    names = feature_names()

    # The model input contains 78 columns. This code assumes
    # the first 39 are clinical values and the remaining 39
    # are missingness indicators. Verify against preprocessing.
    clinical_values = X[idx, -1, :39]
    missing_indicators = X[idx, -1, 39:]

    predicted_class = int(probability >= threshold)

    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
    print(f"Dataset sample index: {idx}")
    print(f"Dataset reference label: {int(y[idx])}")
    print(f"Projected probability: {probability * 100:.2f}%")
    print(f"Threshold: {threshold * 100:.2f}%")
    print(
        "Model prediction: "
        + ("POSITIVE (1)" if predicted_class else "NEGATIVE (0)")
    )
    print(
        "Prediction matches dataset label: "
        + ("YES" if predicted_class == int(y[idx]) else "NO")
    )

    print("\n39 CLINICAL INPUT FEATURES — FINAL TIME STEP")
    print("-" * 72)

    for i, name in enumerate(names):
        value = float(clinical_values[i])
        is_missing = bool(missing_indicators[i] > 0.5)

        suffix = " [missing indicator set]" if is_missing else ""
        print(f"{i + 1:02d}. {name:<40} {value:>12.5f}{suffix}")

    print("-" * 72)
    print(
        "Values above are taken from the processed model input. "
        "They may be scaled, not original clinical units."
    )

    return predicted_class


# ============================================================
# SHAP MODEL OUTPUT WRAPPER
# ============================================================

class SHAPModel(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, x):
        result = self.base_model(x)

        if isinstance(result, (tuple, list)):
            result = result[0]
        elif isinstance(result, dict):
            result = result["logits"]

        return result.reshape(-1, 1)


# ============================================================
# EXPLAIN ONE PATIENT
# ============================================================

def explain_patient(model, X, idx, title):
    import shap

    rng = np.random.default_rng(42)
    bg_ids = rng.choice(
        len(X),
        size=min(20, len(X)),
        replace=False,
    )

    background = torch.as_tensor(
        X[bg_ids],
        dtype=torch.float32,
        device=DEVICE,
    )

    sample = torch.as_tensor(
        X[idx:idx + 1],
        dtype=torch.float32,
        device=DEVICE,
    )

    wrapped = SHAPModel(model).to(DEVICE)
    wrapped.eval()

    explainer = shap.GradientExplainer(wrapped, background)
    values = explainer.shap_values(sample)

    if isinstance(values, list):
        values = values[0]

    values = np.squeeze(np.asarray(values))

    if values.shape != X[idx].shape:
        raise ValueError(
            f"Unexpected SHAP shape {values.shape}; "
            f"expected {X[idx].shape}"
        )

    names39 = feature_names()

    # Aggregate absolute contributions over six time steps.
    # The model has 39 value inputs + 39 missingness inputs.
    importance78 = np.mean(np.abs(values), axis=0)
    importance39 = (
        importance78[:39] + importance78[39:]
    )

    order = np.argsort(importance39)[::-1]

    print("\n" + "=" * 72)
    print(f"XAI / SHAP EXPLANATION: {title}")
    print("=" * 72)

    for rank, i in enumerate(order[:10], start=1):
        print(
            f"{rank:02d}. {names39[i]:<40} "
            f"Importance: {importance39[i]:.6f}"
        )

    OUT.mkdir(parents=True, exist_ok=True)

    # One feature ranking for each patient.
    csv_path = OUT / f"{title.lower()}_shap_features.csv"

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.writer(f)
        writer.writerow([
            "clinical_feature",
            "value_input_mean_abs_shap",
            "missingness_input_mean_abs_shap",
            "combined_importance",
        ])

        for i in order:
            writer.writerow([
                names39[i],
                float(importance78[i]),
                float(importance78[i + 39]),
                float(importance39[i]),
            ])

    # Plot top 15 original clinical features.
    top = order[:15][::-1]

    plt.figure(figsize=(10, 6))
    plt.barh(
        [names39[i] for i in top],
        importance39[top],
    )
    plt.xlabel("Combined mean absolute SHAP magnitude")
    plt.ylabel("Clinical feature")
    plt.title(f"{title}: SHAP Explanation")
    plt.tight_layout()

    plot_path = OUT / f"{title.lower()}_shap.png"
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()

    # Save full time-step contributions for auditability.
    hourly_path = OUT / f"{title.lower()}_shap_hourly.csv"

    with open(
        hourly_path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.writer(f)
        writer.writerow([
            "hour",
            "input_feature",
            "shap_value",
            "processed_input_value",
        ])

        for hour in range(values.shape[0]):
            for j in range(values.shape[1]):
                writer.writerow([
                    hour + 1,
                    (
                        names39[j]
                        if j < 39
                        else names39[j - 39] + " - missingness indicator"
                    ),
                    float(values[hour, j]),
                    float(X[idx, hour, j]),
                ])

    print(f"\nSHAP plot: {plot_path}")
    print(f"Feature CSV: {csv_path}")
    print(f"Hourly CSV: {hourly_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    OUT.mkdir(parents=True, exist_ok=True)

    X, y = load_data()
    model, coefficient, intercept, threshold, config = load_model()

    if X.shape[1] != int(config["sequence_length"]):
        raise ValueError("Dataset sequence length does not match config.")

    if X.shape[2] != int(config["input_size"]):
        raise ValueError("Dataset feature count does not match config.")

    raw = predict_raw(model, X)
    probabilities = calibrate(raw, coefficient, intercept)

    print("\nTEST SET SUMMARY")
    print("-" * 72)
    print(f"Raw probability range: {raw.min():.6f} - {raw.max():.6f}")
    print(
        f"Calibrated probability range: "
        f"{probabilities.min():.6f} - {probabilities.max():.6f}"
    )
    print(
        f"Predicted positives at threshold {threshold:.2f}: "
        f"{int((probabilities >= threshold).sum())}"
    )

    pos_idx, neg_idx = select_patients(y, probabilities)

    # Show both patients' 39 clinical feature values and scores.
    display_patient(
        "POSITIVE-LABEL PATIENT",
        pos_idx, X, y, probabilities[pos_idx], threshold,
    )

    display_patient(
        "NEGATIVE-LABEL PATIENT",
        neg_idx, X, y, probabilities[neg_idx], threshold,
    )

    # Explain the exact same two sequences used for prediction.
    explain_patient(model, X, pos_idx, "Positive_Patient")
    explain_patient(model, X, neg_idx, "Negative_Patient")

    print("\n" + "=" * 72)
    print("BOTH PATIENTS PROCESSED THROUGH PREDICTION AND XAI")
    print(f"Results folder: {OUT}")
    print("=" * 72)


if __name__ == "__main__":
    main()
