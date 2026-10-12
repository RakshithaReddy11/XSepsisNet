from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
    average_precision_score,
)
from sklearn.calibration import calibration_curve


# ============================================================
# XSepsisNet - Validation-Fitted Probability Calibration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

VALIDATION_FILE = (
    PROJECT_ROOT
    / "models"
    / "evaluation"
    / "final_bilstm_validation_predictions.npz"
)

TEST_FILE = (
    PROJECT_ROOT
    / "models"
    / "bilstm"
    / "final_bilstm_test_predictions.npz"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "models"
    / "evaluation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CALIBRATOR_FILE = (
    OUTPUT_DIR
    / "final_bilstm_platt_calibrator.json"
)

COMPARISON_FILE = (
    OUTPUT_DIR
    / "final_bilstm_calibration_comparison.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "final_bilstm_calibrated_evaluation_summary.json"
)

PLOT_FILE = (
    OUTPUT_DIR
    / "final_bilstm_calibration_before_after.png"
)

CALIBRATED_TEST_FILE = (
    OUTPUT_DIR
    / "final_bilstm_test_calibrated_predictions.npz"
)


# ============================================================
# Helper: Expected Calibration Error
# ============================================================

def calculate_ece(
    y_true,
    probabilities,
    n_bins=10
):
    """
    Calculate Expected Calibration Error using
    equal-width probability bins.
    """

    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)

    bin_edges = np.linspace(
        0.0,
        1.0,
        n_bins + 1
    )

    total = len(y_true)
    ece = 0.0

    rows = []

    for i in range(n_bins):

        lower = bin_edges[i]
        upper = bin_edges[i + 1]

        if i == n_bins - 1:

            mask = (
                (probabilities >= lower)
                &
                (probabilities <= upper)
            )

        else:

            mask = (
                (probabilities >= lower)
                &
                (probabilities < upper)
            )

        count = int(mask.sum())

        if count == 0:
            continue

        mean_probability = float(
            probabilities[mask].mean()
        )

        observed_rate = float(
            y_true[mask].mean()
        )

        difference = abs(
            mean_probability
            - observed_rate
        )

        ece += (
            count / total
        ) * difference

        rows.append(
            {
                "bin": i + 1,
                "lower": lower,
                "upper": upper,
                "count": count,
                "mean_predicted_probability":
                    mean_probability,
                "observed_positive_fraction":
                    observed_rate,
                "absolute_difference":
                    difference,
            }
        )

    return float(ece), pd.DataFrame(rows)


# ============================================================
# Helper: Model metrics
# ============================================================

def calculate_metrics(
    y_true,
    probabilities
):

    brier = brier_score_loss(
        y_true,
        probabilities
    )

    logloss = log_loss(
        y_true,
        probabilities,
        labels=[0, 1]
    )

    auroc = roc_auc_score(
        y_true,
        probabilities
    )

    auprc = average_precision_score(
        y_true,
        probabilities
    )

    ece, ece_table = calculate_ece(
        y_true,
        probabilities,
        n_bins=10
    )

    return {
        "brier_score": float(brier),
        "log_loss": float(logloss),
        "auroc": float(auroc),
        "auprc": float(auprc),
        "ece": float(ece),
    }, ece_table


# ============================================================
# Main
# ============================================================

print("=" * 70)
print("XSepsisNet - Validation-Fitted Platt Calibration")
print("=" * 70)


# ============================================================
# 1. Check input files
# ============================================================

print("\n[1] Checking input files...")

for file_path in [
    VALIDATION_FILE,
    TEST_FILE,
]:

    if not file_path.exists():

        raise FileNotFoundError(
            f"Required file not found:\n{file_path}"
        )

    print(
        "OK:",
        file_path.relative_to(PROJECT_ROOT)
    )


# ============================================================
# 2. Load validation predictions
# ============================================================

print("\n[2] Loading validation predictions...")

validation = np.load(
    VALIDATION_FILE
)

y_val = validation["y_true"].astype(int)

p_val_raw = validation[
    "y_probability"
].astype(float)

print(
    f"Validation samples: {len(y_val)}"
)

print(
    f"Validation positive: {np.sum(y_val == 1)}"
)

print(
    f"Validation negative: {np.sum(y_val == 0)}"
)


# ============================================================
# 3. Load untouched test predictions
# ============================================================

print("\n[3] Loading untouched test predictions...")

test = np.load(
    TEST_FILE
)

y_test = test["y_true"].astype(int)

p_test_raw = test[
    "y_probability"
].astype(float)

print(
    f"Test samples: {len(y_test)}"
)

print(
    f"Test positive: {np.sum(y_test == 1)}"
)

print(
    f"Test negative: {np.sum(y_test == 0)}"
)


# ============================================================
# 4. Validate data
# ============================================================

print("\n[4] Validating probability data...")

for name, y, p in [
    ("validation", y_val, p_val_raw),
    ("test", y_test, p_test_raw),
]:

    if len(y) != len(p):

        raise ValueError(
            f"{name}: y and probability lengths differ."
        )

    if np.isnan(p).any():

        raise ValueError(
            f"{name}: NaN probabilities found."
        )

    if np.isinf(p).any():

        raise ValueError(
            f"{name}: infinite probabilities found."
        )

    if np.any(
        (p < 0)
        |
        (p > 1)
    ):

        raise ValueError(
            f"{name}: probabilities outside [0,1]."
        )

print("Probability validation passed.")


# ============================================================
# 5. Evaluate raw validation probabilities
# ============================================================

print("\n[5] Evaluating raw validation probabilities...")

raw_val_metrics, _ = calculate_metrics(
    y_val,
    p_val_raw
)

print(
    f"Validation raw Brier : "
    f"{raw_val_metrics['brier_score']:.6f}"
)

print(
    f"Validation raw LogLoss: "
    f"{raw_val_metrics['log_loss']:.6f}"
)

print(
    f"Validation raw AUROC : "
    f"{raw_val_metrics['auroc']:.6f}"
)

print(
    f"Validation raw AUPRC  : "
    f"{raw_val_metrics['auprc']:.6f}"
)

print(
    f"Validation raw ECE    : "
    f"{raw_val_metrics['ece']:.6f}"
)


# ============================================================
# 6. Fit Platt scaling ONLY on validation
# ============================================================

print("\n[6] Fitting Platt calibration on VALIDATION only...")

# LogisticRegression expects a feature matrix.
# The single feature is the original model probability.

calibrator = LogisticRegression(
    solver="lbfgs",
    random_state=42
)

calibrator.fit(
    p_val_raw.reshape(-1, 1),
    y_val
)

coef = float(
    calibrator.coef_[0][0]
)

intercept = float(
    calibrator.intercept_[0]
)

print(
    f"Calibration coefficient: {coef:.8f}"
)

print(
    f"Calibration intercept  : {intercept:.8f}"
)


# ============================================================
# 7. Apply calibrator
# ============================================================

print("\n[7] Applying calibration...")

p_val_calibrated = calibrator.predict_proba(
    p_val_raw.reshape(-1, 1)
)[:, 1]

p_test_calibrated = calibrator.predict_proba(
    p_test_raw.reshape(-1, 1)
)[:, 1]


# ============================================================
# 8. Evaluate validation calibrated probabilities
# ============================================================

print(
    "\n[8] Evaluating calibrated validation probabilities..."
)

cal_val_metrics, _ = calculate_metrics(
    y_val,
    p_val_calibrated
)

print(
    f"Validation calibrated Brier : "
    f"{cal_val_metrics['brier_score']:.6f}"
)

print(
    f"Validation calibrated LogLoss: "
    f"{cal_val_metrics['log_loss']:.6f}"
)

print(
    f"Validation calibrated ECE    : "
    f"{cal_val_metrics['ece']:.6f}"
)


# ============================================================
# 9. Final evaluation on untouched test
# ============================================================

print(
    "\n[9] Evaluating RAW vs CALIBRATED on TEST..."
)

raw_test_metrics, raw_test_ece_table = (
    calculate_metrics(
        y_test,
        p_test_raw
    )
)

cal_test_metrics, cal_test_ece_table = (
    calculate_metrics(
        y_test,
        p_test_calibrated
    )
)


# ============================================================
# 10. Print test comparison
# ============================================================

print("\n" + "-" * 70)
print("TEST SET COMPARISON")
print("-" * 70)

print(
    f"{'Metric':<15}"
    f"{'Raw':>15}"
    f"{'Calibrated':>18}"
)

print("-" * 48)

for metric, label in [
    ("brier_score", "Brier Score"),
    ("log_loss", "Log Loss"),
    ("auroc", "AUROC"),
    ("auprc", "AUPRC"),
    ("ece", "ECE"),
]:

    print(
        f"{label:<15}"
        f"{raw_test_metrics[metric]:>15.6f}"
        f"{cal_test_metrics[metric]:>18.6f}"
    )


# ============================================================
# 11. Save calibrated predictions
# ============================================================

print(
    "\n[10] Saving calibrated test predictions..."
)

np.savez_compressed(
    CALIBRATED_TEST_FILE,
    y_true=y_test.astype(np.float32),
    raw_probability=p_test_raw.astype(np.float32),
    calibrated_probability=(
        p_test_calibrated.astype(np.float32)
    )
)

print(
    "Saved:",
    CALIBRATED_TEST_FILE.relative_to(
        PROJECT_ROOT
    )
)


# ============================================================
# 12. Save calibration parameters
# ============================================================

print(
    "\n[11] Saving calibration parameters..."
)

calibrator_summary = {
    "method": "Platt scaling",
    "fit_dataset": "validation",
    "fit_samples": int(len(y_val)),
    "validation_positive": int(
        np.sum(y_val == 1)
    ),
    "validation_negative": int(
        np.sum(y_val == 0)
    ),
    "coefficient": coef,
    "intercept": intercept,
    "formula": (
        "calibrated_probability = "
        "sigmoid(coefficient * raw_probability "
        "+ intercept)"
    ),
    "test_data_used_for_fitting": False,
}

with open(
    CALIBRATOR_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        calibrator_summary,
        f,
        indent=2
    )

print(
    "Saved:",
    CALIBRATOR_FILE.relative_to(
        PROJECT_ROOT
    )
)


# ============================================================
# 13. Save comparison table
# ============================================================

comparison_rows = []

for metric in [
    "brier_score",
    "log_loss",
    "auroc",
    "auprc",
    "ece",
]:

    comparison_rows.append(
        {
            "metric": metric,
            "raw_test": raw_test_metrics[metric],
            "calibrated_test":
                cal_test_metrics[metric],
            "change":
                cal_test_metrics[metric]
                - raw_test_metrics[metric],
        }
    )

comparison_df = pd.DataFrame(
    comparison_rows
)

comparison_df.to_csv(
    COMPARISON_FILE,
    index=False
)

print(
    "Saved:",
    COMPARISON_FILE.relative_to(
        PROJECT_ROOT
    )
)


# ============================================================
# 14. Save detailed summary
# ============================================================

summary = {
    "model": "XSepsisNet Final BiLSTM",
    "calibration_method": "Platt scaling",
    "calibration_fit_dataset": "validation",
    "validation_samples": int(len(y_val)),
    "test_samples": int(len(y_test)),
    "test_calibration_fitting": False,
    "raw_test_metrics": raw_test_metrics,
    "calibrated_test_metrics": cal_test_metrics,
    "calibration_parameters": {
        "coefficient": coef,
        "intercept": intercept,
    },
    "interpretation": (
        "The BiLSTM itself was not retrained. "
        "Platt calibration was fitted using validation "
        "predictions only and then applied to the "
        "untouched test predictions."
    ),
}

with open(
    SUMMARY_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=2
    )

print(
    "Saved:",
    SUMMARY_FILE.relative_to(
        PROJECT_ROOT
    )
)


# ============================================================
# 15. Calibration curves
# ============================================================

print(
    "\n[12] Generating before/after calibration plot..."
)

raw_fraction, raw_mean = calibration_curve(
    y_test,
    p_test_raw,
    n_bins=10,
    strategy="uniform"
)

cal_fraction, cal_mean = calibration_curve(
    y_test,
    p_test_calibrated,
    n_bins=10,
    strategy="uniform"
)

plt.figure(figsize=(8, 6))

plt.plot(
    raw_mean,
    raw_fraction,
    marker="o",
    linewidth=2,
    label="Raw BiLSTM"
)

plt.plot(
    cal_mean,
    cal_fraction,
    marker="o",
    linewidth=2,
    label="Platt Calibrated"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    linewidth=1,
    label="Perfect calibration"
)

plt.xlabel(
    "Mean Predicted Probability"
)

plt.ylabel(
    "Observed Positive Fraction"
)

plt.title(
    "XSepsisNet - Before vs After Calibration"
)

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    PLOT_FILE,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Saved:",
    PLOT_FILE.relative_to(
        PROJECT_ROOT
    )
)


# ============================================================
# Final
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION-FITTED CALIBRATION COMPLETE")
print("=" * 70)

print(
    "\nBiLSTM retraining        : NO"
)

print(
    "Calibration fit on test  : NO"
)

print(
    "Calibration fit on valid : YES"
)

print(
    "\nRaw Test Brier Score     : "
    f"{raw_test_metrics['brier_score']:.6f}"
)

print(
    "Calibrated Test Brier    : "
    f"{cal_test_metrics['brier_score']:.6f}"
)

print(
    "\nRaw Test ECE             : "
    f"{raw_test_metrics['ece']:.6f}"
)

print(
    "Calibrated Test ECE      : "
    f"{cal_test_metrics['ece']:.6f}"
)

print(
    "\nAll calibration artifacts "
    "saved under models\\evaluation."
)