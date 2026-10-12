from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss


# ============================================================
# XSepsisNet - BiLSTM Probability Calibration Analysis
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PREDICTION_FILE = (
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

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. Load predictions
# ============================================================

print("=" * 70)
print("XSepsisNet - BiLSTM Calibration Analysis")
print("=" * 70)

print("\n[1] Loading test predictions...")

if not PREDICTION_FILE.exists():
    raise FileNotFoundError(
        f"Prediction file not found:\n{PREDICTION_FILE}"
    )

data = np.load(PREDICTION_FILE)

y_true = data["y_true"].astype(int)
y_probability = data["y_probability"].astype(float)


# ============================================================
# 2. Validate
# ============================================================

print("\n[2] Validating probabilities...")

if len(y_true) != len(y_probability):
    raise ValueError(
        "y_true and y_probability lengths do not match."
    )

if np.isnan(y_probability).any():
    raise ValueError(
        "NaN values found in predicted probabilities."
    )

if np.isinf(y_probability).any():
    raise ValueError(
        "Infinite values found in predicted probabilities."
    )

if np.any(
    (y_probability < 0) |
    (y_probability > 1)
):
    raise ValueError(
        "Predicted probabilities must be between 0 and 1."
    )

print(f"Samples: {len(y_true)}")
print(f"Positive: {np.sum(y_true == 1)}")
print(f"Negative: {np.sum(y_true == 0)}")


# ============================================================
# 3. Brier score
# ============================================================

print("\n[3] Calculating Brier score...")

brier = brier_score_loss(
    y_true,
    y_probability
)

print(f"Brier Score: {brier:.6f}")


# ============================================================
# 4. Calibration curve
# ============================================================

print("\n[4] Calculating calibration curve...")

fraction_positive, mean_predicted = calibration_curve(
    y_true,
    y_probability,
    n_bins=10,
    strategy="uniform"
)


# ============================================================
# 5. Create calibration table
# ============================================================

print("\n[5] Creating calibration table...")

calibration_rows = []

for i in range(len(mean_predicted)):

    calibration_rows.append(
        {
            "bin": i + 1,
            "mean_predicted_probability": float(
                mean_predicted[i]
            ),
            "observed_positive_fraction": float(
                fraction_positive[i]
            ),
            "absolute_difference": float(
                abs(
                    mean_predicted[i]
                    - fraction_positive[i]
                )
            ),
        }
    )

calibration_df = pd.DataFrame(
    calibration_rows
)

calibration_file = (
    OUTPUT_DIR
    / "final_bilstm_calibration_table.csv"
)

calibration_df.to_csv(
    calibration_file,
    index=False
)

print(
    calibration_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# ============================================================
# 6. Calibration plot
# ============================================================

print("\n[6] Generating calibration plot...")

plt.figure(figsize=(8, 6))

plt.plot(
    mean_predicted,
    fraction_positive,
    marker="o",
    linewidth=2,
    label="XSepsisNet BiLSTM"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    linewidth=1,
    label="Perfect calibration"
)

plt.xlabel("Mean Predicted Probability")
plt.ylabel("Observed Positive Fraction")

plt.title(
    "XSepsisNet - Calibration Curve"
)

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

calibration_plot = (
    OUTPUT_DIR
    / "final_bilstm_calibration_curve.png"
)

plt.savefig(
    calibration_plot,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 7. Calibration summary
# ============================================================

print("\n[7] Saving calibration summary...")

mean_absolute_calibration_error = float(
    calibration_df["absolute_difference"].mean()
)

summary = {
    "model": "XSepsisNet Final BiLSTM",
    "test_samples": int(len(y_true)),
    "brier_score": float(brier),
    "mean_absolute_calibration_error": (
        mean_absolute_calibration_error
    ),
    "number_of_bins": int(len(calibration_df)),
    "note": (
        "This is an assessment of the raw model "
        "probabilities on the test set. No calibration "
        "correction was fitted on the test set."
    ),
    "files": {
        "calibration_table": str(
            calibration_file.relative_to(PROJECT_ROOT)
        ),
        "calibration_curve": str(
            calibration_plot.relative_to(PROJECT_ROOT)
        ),
    },
}

summary_file = (
    OUTPUT_DIR
    / "final_bilstm_calibration_summary.json"
)

with open(
    summary_file,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        summary,
        f,
        indent=2
    )


# ============================================================
# 8. Final output
# ============================================================

print("\n" + "=" * 70)
print("CALIBRATION ANALYSIS COMPLETE")
print("=" * 70)

print(f"\nBrier Score: {brier:.6f}")

print(
    "Mean Absolute Calibration Error: "
    f"{mean_absolute_calibration_error:.6f}"
)

print("\nGenerated files:")

print(
    f"1. {calibration_file.relative_to(PROJECT_ROOT)}"
)

print(
    f"2. {calibration_plot.relative_to(PROJECT_ROOT)}"
)

print(
    f"3. {summary_file.relative_to(PROJECT_ROOT)}"
)

print(
    "\nNo model retraining or test-set calibration "
    "fitting was performed."
)