from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)


# ============================================================
# XSepsisNet - Final BiLSTM Evaluation
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
# 1. Load test predictions
# ============================================================

print("=" * 70)
print("XSepsisNet - Final BiLSTM Evaluation")
print("=" * 70)

print("\n[1] Loading test predictions...")

if not PREDICTION_FILE.exists():
    raise FileNotFoundError(
        f"Prediction file not found:\n{PREDICTION_FILE}"
    )

data = np.load(PREDICTION_FILE)

required_arrays = {"y_true", "y_probability"}

if not required_arrays.issubset(set(data.files)):
    raise ValueError(
        f"Prediction file must contain {required_arrays}. "
        f"Found: {data.files}"
    )

y_true = data["y_true"].astype(int)
y_probability = data["y_probability"].astype(float)


# ============================================================
# 2. Basic validation
# ============================================================

print("\n[2] Validating prediction arrays...")

if len(y_true) != len(y_probability):
    raise ValueError(
        "y_true and y_probability have different lengths."
    )

if len(y_true) == 0:
    raise ValueError("Test prediction arrays are empty.")

if np.isnan(y_probability).any():
    raise ValueError("NaN values found in y_probability.")

if np.isinf(y_probability).any():
    raise ValueError("Infinite values found in y_probability.")

if np.any((y_probability < 0) | (y_probability > 1)):
    raise ValueError(
        "y_probability contains values outside [0, 1]."
    )

unique_labels = np.unique(y_true)

if not set(unique_labels).issubset({0, 1}):
    raise ValueError(
        f"Unexpected labels found: {unique_labels}"
    )

n_samples = len(y_true)
n_positive = int(np.sum(y_true == 1))
n_negative = int(np.sum(y_true == 0))

prevalence = n_positive / n_samples

print(f"Test samples : {n_samples}")
print(f"Positive     : {n_positive}")
print(f"Negative     : {n_negative}")
print(f"Prevalence   : {prevalence:.6f} ({prevalence * 100:.2f}%)")


# ============================================================
# 3. Overall ranking metrics
# ============================================================

print("\n[3] Calculating overall ranking metrics...")

auroc = roc_auc_score(y_true, y_probability)
auprc = average_precision_score(y_true, y_probability)

print(f"AUROC : {auroc:.6f}")
print(f"AUPRC : {auprc:.6f}")


# ============================================================
# 4. Threshold analysis
# ============================================================

print("\n[4] Running threshold analysis...")

thresholds = np.round(
    np.arange(0.10, 0.91, 0.05),
    2
)

threshold_rows = []

for threshold in thresholds:

    y_pred = (y_probability >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    npv = (
        tn / (tn + fn)
        if (tn + fn) > 0
        else 0.0
    )

    predicted_positive_rate = (
        (tp + fp) / n_samples
    )

    threshold_rows.append(
        {
            "threshold": float(threshold),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "precision": float(precision),
            "recall_sensitivity": float(recall),
            "specificity": float(specificity),
            "f1": float(f1),
            "npv": float(npv),
            "predicted_positive_rate": float(
                predicted_positive_rate
            ),
        }
    )

threshold_df = pd.DataFrame(threshold_rows)

threshold_file = (
    OUTPUT_DIR
    / "final_bilstm_threshold_analysis.csv"
)

threshold_df.to_csv(
    threshold_file,
    index=False
)

print("\nThreshold analysis:")
print(
    threshold_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# 5. Baseline threshold = 0.50
# ============================================================

print("\n[5] Calculating baseline threshold metrics (0.50)...")

baseline_threshold = 0.50

y_pred_050 = (
    y_probability >= baseline_threshold
).astype(int)

tn, fp, fn, tp = confusion_matrix(
    y_true,
    y_pred_050,
    labels=[0, 1]
).ravel()

baseline_precision = precision_score(
    y_true,
    y_pred_050,
    zero_division=0
)

baseline_recall = recall_score(
    y_true,
    y_pred_050,
    zero_division=0
)

baseline_f1 = f1_score(
    y_true,
    y_pred_050,
    zero_division=0
)

baseline_specificity = (
    tn / (tn + fp)
    if (tn + fp) > 0
    else 0.0
)

print(f"Threshold    : {baseline_threshold:.2f}")
print(f"TN           : {tn}")
print(f"FP           : {fp}")
print(f"FN           : {fn}")
print(f"TP           : {tp}")
print(f"Precision    : {baseline_precision:.6f}")
print(f"Recall       : {baseline_recall:.6f}")
print(f"Specificity  : {baseline_specificity:.6f}")
print(f"F1           : {baseline_f1:.6f}")


# ============================================================
# 6. ROC Curve
# ============================================================

print("\n[6] Generating ROC curve...")

fpr, tpr, roc_thresholds = roc_curve(
    y_true,
    y_probability
)

plt.figure(figsize=(8, 6))

plt.plot(
    fpr,
    tpr,
    linewidth=2,
    label=f"BiLSTM (AUROC = {auroc:.4f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    linewidth=1
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("XSepsisNet - ROC Curve")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

roc_file = OUTPUT_DIR / "final_bilstm_roc_curve.png"

plt.savefig(
    roc_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 7. Precision-Recall Curve
# ============================================================

print("[7] Generating Precision-Recall curve...")

precision_curve, recall_curve, pr_thresholds = (
    precision_recall_curve(
        y_true,
        y_probability
    )
)

plt.figure(figsize=(8, 6))

plt.plot(
    recall_curve,
    precision_curve,
    linewidth=2,
    label=f"BiLSTM (AUPRC = {auprc:.4f})"
)

plt.axhline(
    prevalence,
    linestyle="--",
    linewidth=1,
    label=f"Prevalence = {prevalence:.4f}"
)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("XSepsisNet - Precision-Recall Curve")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

pr_file = (
    OUTPUT_DIR
    / "final_bilstm_precision_recall_curve.png"
)

plt.savefig(
    pr_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 8. Confusion Matrix at 0.50
# ============================================================

print("[8] Generating confusion matrix...")

cm = np.array(
    [
        [tn, fp],
        [fn, tp]
    ]
)

plt.figure(figsize=(7, 6))

plt.imshow(cm)

plt.title(
    "XSepsisNet - Confusion Matrix (Threshold = 0.50)"
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.xticks(
    [0, 1],
    ["Negative", "Positive"]
)

plt.yticks(
    [0, 1],
    ["Negative", "Positive"]
)

for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center",
            fontsize=14
        )

plt.colorbar()

plt.tight_layout()

cm_file = (
    OUTPUT_DIR
    / "final_bilstm_confusion_matrix_050.png"
)

plt.savefig(
    cm_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 9. Save summary JSON
# ============================================================

print("[9] Saving evaluation summary...")

summary = {
    "model": "XSepsisNet Final BiLSTM",
    "test_samples": int(n_samples),
    "positive_samples": int(n_positive),
    "negative_samples": int(n_negative),
    "positive_prevalence": float(prevalence),

    "auroc": float(auroc),
    "auprc": float(auprc),

    "baseline_threshold": baseline_threshold,

    "baseline_confusion_matrix": {
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    },

    "baseline_metrics": {
        "precision": float(baseline_precision),
        "recall_sensitivity": float(baseline_recall),
        "specificity": float(baseline_specificity),
        "f1": float(baseline_f1),
    },

    "files": {
        "threshold_analysis": str(
            threshold_file.relative_to(PROJECT_ROOT)
        ),
        "roc_curve": str(
            roc_file.relative_to(PROJECT_ROOT)
        ),
        "precision_recall_curve": str(
            pr_file.relative_to(PROJECT_ROOT)
        ),
        "confusion_matrix": str(
            cm_file.relative_to(PROJECT_ROOT)
        ),
    },
}

summary_file = (
    OUTPUT_DIR
    / "final_bilstm_evaluation_summary.json"
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
# 10. Final output
# ============================================================

print("\n" + "=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)

print("\nOverall metrics:")
print(f"AUROC : {auroc:.6f}")
print(f"AUPRC : {auprc:.6f}")

print("\nBaseline threshold = 0.50:")
print(f"Precision   : {baseline_precision:.6f}")
print(f"Recall      : {baseline_recall:.6f}")
print(f"Specificity : {baseline_specificity:.6f}")
print(f"F1          : {baseline_f1:.6f}")

print("\nGenerated files:")

print(
    f"1. {threshold_file.relative_to(PROJECT_ROOT)}"
)

print(
    f"2. {roc_file.relative_to(PROJECT_ROOT)}"
)

print(
    f"3. {pr_file.relative_to(PROJECT_ROOT)}"
)

print(
    f"4. {cm_file.relative_to(PROJECT_ROOT)}"
)

print(
    f"5. {summary_file.relative_to(PROJECT_ROOT)}"
)

print("\nNo model retraining was performed.")