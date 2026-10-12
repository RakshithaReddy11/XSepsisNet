from pathlib import Path
import json
import csv

import numpy as np
import torch
import torch.nn as nn
import shap
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# XSepsisNet - Manual Prediction + SHAP Explanation
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_FILE = (
    PROJECT_ROOT / "models" / "bilstm"
    / "final_bilstm_best.pt"
)

CONFIG_FILE = (
    PROJECT_ROOT / "models" / "bilstm"
    / "final_bilstm_config.json"
)

SCALER_FILE = (
    PROJECT_ROOT / "models" / "scalers"
    / "final_bilstm_scaler.json"
)

CALIBRATOR_FILE = (
    PROJECT_ROOT / "models" / "evaluation"
    / "final_bilstm_platt_calibrator.json"
)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_DIR = (
    PROJECT_ROOT / "models" / "explainability"
)


# ============================================================
# Exact 39 clinical features used by the model
# ============================================================

FEATURE_COLUMNS = [
    "heart_rate",
    "systolic_blood_pressure",
    "diastolic_blood_pressure",
    "mean_arterial_pressure",
    "respiratory_rate",
    "vital_oxygen_saturation",
    "temperature",
    "fio2",
    "etco2",
    "central_venous_pressure",
    "pulmonary_artery_pressure_systolic",
    "pulmonary_artery_pressure_diastolic",
    "pulmonary_artery_pressure_mean",
    "albumin",
    "alkaline_phosphatase",
    "alt",
    "anion_gap",
    "ast",
    "base_excess",
    "bicarbonate",
    "bilirubin",
    "bun",
    "calcium",
    "chloride",
    "creatinine",
    "glucose",
    "hematocrit",
    "hemoglobin",
    "lactate",
    "magnesium",
    "lab_oxygen_saturation",
    "pco2",
    "ph",
    "phosphate",
    "platelets",
    "po2",
    "potassium",
    "sodium",
    "wbc",
]


DISPLAY_NAMES = {
    "heart_rate": "Heart Rate",
    "systolic_blood_pressure": "Systolic BP",
    "diastolic_blood_pressure": "Diastolic BP",
    "mean_arterial_pressure": "Mean Arterial Pressure",
    "respiratory_rate": "Respiratory Rate",
    "vital_oxygen_saturation": "Vital Oxygen Saturation",
    "temperature": "Temperature",
    "fio2": "FiO2",
    "etco2": "EtCO2",
    "central_venous_pressure": "Central Venous Pressure",
    "pulmonary_artery_pressure_systolic": "Pulmonary Artery Pressure - Systolic",
    "pulmonary_artery_pressure_diastolic": "Pulmonary Artery Pressure - Diastolic",
    "pulmonary_artery_pressure_mean": "Pulmonary Artery Pressure - Mean",
    "albumin": "Albumin",
    "alkaline_phosphatase": "Alkaline Phosphatase",
    "alt": "ALT",
    "anion_gap": "Anion Gap",
    "ast": "AST",
    "base_excess": "Base Excess",
    "bicarbonate": "Bicarbonate",
    "bilirubin": "Bilirubin",
    "bun": "BUN",
    "calcium": "Calcium",
    "chloride": "Chloride",
    "creatinine": "Creatinine",
    "glucose": "Glucose",
    "hematocrit": "Hematocrit",
    "hemoglobin": "Hemoglobin",
    "lactate": "Lactate",
    "magnesium": "Magnesium",
    "lab_oxygen_saturation": "Lab Oxygen Saturation",
    "pco2": "pCO2",
    "ph": "pH",
    "phosphate": "Phosphate",
    "platelets": "Platelets",
    "po2": "pO2",
    "potassium": "Potassium",
    "sodium": "Sodium",
    "wbc": "WBC",
}


# ============================================================
# BiLSTM architecture
# ============================================================

class FinalBiLSTM(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        bidirectional,
        dropout,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        output_size = (
            hidden_size * 2
            if bidirectional
            else hidden_size
        )

        self.classifier = nn.Linear(
            output_size,
            1,
        )

    def forward(self, x):
        output, _ = self.lstm(x)
        last_output = output[:, -1, :]
        logits = self.classifier(last_output)
        return logits.squeeze(1)


# ============================================================
# Calibrated probability model for SHAP
# ============================================================

class CalibratedProbabilityModel(nn.Module):

    def __init__(self, base_model, coefficient, intercept):
        super().__init__()
        self.base_model = base_model
        self.coefficient = float(coefficient)
        self.intercept = float(intercept)

    def forward(self, x):
        raw_probability = torch.sigmoid(self.base_model(x))

        calibrated_logit = (
            self.coefficient * raw_probability
            + self.intercept
        )

        # SHAP's PyTorch gradient explainer expects a batch dimension and
        # an explicit single-output dimension.
        return torch.sigmoid(calibrated_logit).unsqueeze(1)


# ============================================================
# Utility functions
# ============================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def read_value(feature_name):
    display_name = DISPLAY_NAMES.get(
        feature_name,
        feature_name,
    )

    while True:
        user_input = input(
            f"  {display_name}: "
        ).strip()

        if user_input == "":
            return np.nan

        try:
            value = float(user_input)

            if not np.isfinite(value):
                print("  Enter a finite number or leave blank.")
                continue

            return value

        except ValueError:
            print("  Invalid value. Enter a number or leave blank.")


def sigmoid(value):
    value = float(np.clip(value, -500, 500))
    return 1.0 / (1.0 + np.exp(-value))


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 72)
    print("             X-SepsisNet Patient Prediction")
    print("=" * 72)

    print("\nExisting trained BiLSTM will be used.")
    print("No retraining will be performed.")
    print("Enter clinical measurements for six consecutive hours.")
    print("Leave a value blank if it is unavailable.")

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    required_files = [
        MODEL_FILE,
        CONFIG_FILE,
        SCALER_FILE,
        CALIBRATOR_FILE,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found:\n{path}"
            )

    # --------------------------------------------------------
    # Load configuration
    # --------------------------------------------------------

    config = load_json(CONFIG_FILE)

    input_size = int(config["input_size"])
    sequence_length = int(config["sequence_length"])
    hidden_size = int(config["hidden_size"])
    num_layers = int(config["num_layers"])
    bidirectional = bool(config["bidirectional"])
    dropout = float(config["dropout"])

    if input_size != 78 or sequence_length != 6:
        raise ValueError(
            f"Unexpected model configuration: "
            f"sequence_length={sequence_length}, "
            f"input_size={input_size}. Expected 6 and 78."
        )

    # --------------------------------------------------------
    # Load scaler
    # --------------------------------------------------------

    scaler = load_json(SCALER_FILE)

    scaler_features = scaler["feature_columns"]
    scaler_mean = np.asarray(
        scaler["mean"],
        dtype=np.float64,
    )
    scaler_scale = np.asarray(
        scaler["scale"],
        dtype=np.float64,
    )

    if scaler_features != FEATURE_COLUMNS:
        raise ValueError(
            "Feature order differs from the training scaler."
        )

    if len(scaler_mean) != 39 or len(scaler_scale) != 39:
        raise ValueError(
            "Scaler must contain 39 means and 39 scale values."
        )

    if np.any(scaler_scale == 0):
        raise ValueError(
            "Scaler contains a zero scale value."
        )

    # --------------------------------------------------------
    # Load calibration parameters
    # --------------------------------------------------------

    calibrator = load_json(CALIBRATOR_FILE)

    coefficient = float(calibrator["coefficient"])
    intercept = float(calibrator["intercept"])

    # --------------------------------------------------------
    # Load trained model
    # --------------------------------------------------------

    print("\nLoading trained BiLSTM...")

    device = torch.device("cpu")

    model = FinalBiLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bidirectional=bidirectional,
        dropout=dropout,
    )

    checkpoint = torch.load(
        MODEL_FILE,
        map_location=device,
        weights_only=False,
    )

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        raise ValueError("Unsupported checkpoint format.")

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    print("Trained BiLSTM loaded successfully.")

    # --------------------------------------------------------
    # Collect 6 hours x 39 features
    # --------------------------------------------------------

    raw_data = np.full(
        (sequence_length, len(FEATURE_COLUMNS)),
        np.nan,
        dtype=np.float64,
    )

    for hour in range(sequence_length):

        print("\n" + "-" * 72)
        print(f"HOUR {hour + 1} / {sequence_length}")
        print("-" * 72)

        for index, feature_name in enumerate(FEATURE_COLUMNS):
            raw_data[hour, index] = read_value(feature_name)

    # --------------------------------------------------------
    # Missingness indicators and imputation
    # --------------------------------------------------------

    missing_mask = np.isnan(raw_data).astype(np.float32)
    imputed_data = raw_data.copy()

    for index in range(len(FEATURE_COLUMNS)):
        missing = np.isnan(imputed_data[:, index])
        imputed_data[missing, index] = scaler_mean[index]

    # --------------------------------------------------------
    # Standardization and 78-feature input
    # --------------------------------------------------------

    scaled_data = (
        imputed_data - scaler_mean
    ) / scaler_scale

    model_input = np.concatenate(
        [scaled_data, missing_mask],
        axis=1,
    ).astype(np.float32)

    if model_input.shape != (6, 78):
        raise ValueError(
            f"Unexpected input shape: {model_input.shape}"
        )

    X = torch.tensor(
        model_input,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    print("\nRunning prediction...")

    with torch.no_grad():
        logits = model(X)
        raw_probability = float(torch.sigmoid(logits).item())

    calibrated_probability = sigmoid(
        coefficient * raw_probability + intercept
    )

    # --------------------------------------------------------
    # Display prediction
    # --------------------------------------------------------

    threshold = float(
        config.get("classification_threshold", 0.50)
    )

    raw_classification_threshold = 0.50
    predicted_class = int(
        raw_probability >= raw_classification_threshold
    )

    print("\n" + "=" * 72)
    print("              X-SepsisNet Prediction Result")
    print("=" * 72)

    print("Input sequence       : 6 hours")
    print("Clinical features    : 39 per hour")
    print("Model input shape    : 1 x 6 x 78")
    print(f"Raw model probability: {raw_probability:.6f}")
    print(
        f"Calibrated probability: "
        f"{calibrated_probability:.6f}"
    )
    print(
        f"Predicted probability: "
        f"{calibrated_probability * 100:.2f}%"
    )
    print(f"Classification threshold: {threshold:.2f}")

    print(
        "Predicted class      : "
        + (
            "POSITIVE (project target)"
            if predicted_class == 1
            else "NEGATIVE (project target)"
        )
    )

    if "classification_threshold" not in config:
        print(
            "Note: 0.50 is a default technical threshold, "
            "not confirmed as a clinically validated threshold."
        )

    print(
        "\nThis is a model prediction, not a clinical diagnosis."
    )

    # --------------------------------------------------------
    # Find background dataset for SHAP
    # --------------------------------------------------------

    print("\nGenerating SHAP explanation for this input...")

    candidate_files = [
        PROCESSED_DIR / "final_bilstm_train.npz",
        PROCESSED_DIR / "final_bilstm_training.npz",
        PROCESSED_DIR / "final_bilstm_val.npz",
        PROCESSED_DIR / "final_bilstm_test.npz",
    ]

    background_file = next(
        (path for path in candidate_files if path.exists()),
        None,
    )

    if background_file is None:
        raise FileNotFoundError(
            "No suitable processed NPZ found for SHAP background "
            "inside data/processed."
        )

    background_data = np.load(background_file)

    if "X" not in background_data.files:
        raise ValueError(
            f"X array is missing from {background_file.name}"
        )

    background_array = background_data["X"]

    if (
        background_array.ndim != 3
        or background_array.shape[1:] != (6, 78)
    ):
        raise ValueError(
            f"Unexpected background shape: {background_array.shape}"
        )

    background_count = min(50, len(background_array))

    if background_count == 0:
        raise ValueError("SHAP background dataset is empty.")

    background_tensor = torch.tensor(
        background_array[:background_count],
        dtype=torch.float32,
        device=device,
    )

    calibrated_model = CalibratedProbabilityModel(
        model,
        coefficient,
        intercept,
    ).to(device)

    calibrated_model.eval()

    explainer = shap.GradientExplainer(
        calibrated_model,
        background_tensor,
    )

    shap_values = explainer.shap_values(X)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap_values = np.asarray(shap_values)

    if shap_values.ndim == 4 and shap_values.shape[-1] == 1:
        shap_values = shap_values[..., 0]

    if shap_values.ndim == 3 and shap_values.shape[0] == 1:
        shap_values = shap_values[0]

    if shap_values.shape != (6, 78):
        raise ValueError(
            f"Unexpected SHAP output shape: {shap_values.shape}"
        )

    # --------------------------------------------------------
    # Rank SHAP feature contributions
    # --------------------------------------------------------

    feature_names = (
        FEATURE_COLUMNS
        + [f"{name}_missing" for name in FEATURE_COLUMNS]
    )

    display_names = (
        [DISPLAY_NAMES[name] for name in FEATURE_COLUMNS]
        + [
            f"{DISPLAY_NAMES[name]} (missing)"
            for name in FEATURE_COLUMNS
        ]
    )

    aggregated = shap_values.sum(axis=0)
    ranked_indices = np.argsort(np.abs(aggregated))[::-1]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save CSV
    csv_path = OUTPUT_DIR / "manual_patient_shap_contributions.csv"

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.writer(file)

        writer.writerow([
            "feature",
            "shap_contribution",
            "direction",
            "absolute_contribution",
        ])

        for index in ranked_indices:
            value = float(aggregated[index])

            if value > 0:
                direction = "increases model output"
            elif value < 0:
                direction = "decreases model output"
            else:
                direction = "no net contribution"

            writer.writerow([
                feature_names[index],
                f"{value:.8f}",
                direction,
                f"{abs(value):.8f}",
            ])

    print("\nTOP SHAP CONTRIBUTIONS")
    print("-" * 72)

    for index in ranked_indices[:10]:
        value = float(aggregated[index])

        if value > 0:
            direction = "pushes model output higher"
        elif value < 0:
            direction = "pushes model output lower"
        else:
            direction = "has no net direction"

        print(
            f"{display_names[index]}: {direction} "
            f"(SHAP={value:+.6f})"
        )

    # --------------------------------------------------------
    # Save SHAP plot for screenshot
    # --------------------------------------------------------

    top_indices = ranked_indices[:12][::-1]

    plot_labels = [
        display_names[index]
        for index in top_indices
    ]

    plot_values = [
        float(aggregated[index])
        for index in top_indices
    ]

    plt.figure(figsize=(11, 7))
    plt.barh(plot_labels, plot_values)
    plt.axvline(0, linewidth=1)
    plt.xlabel("SHAP contribution to calibrated model output")
    plt.title(
        "X-SepsisNet - SHAP Explanation for Entered Patient Data"
    )
    plt.tight_layout()

    plot_path = OUTPUT_DIR / "manual_patient_shap.png"

    plt.savefig(
        plot_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print("\nSHAP explanation completed.")
    print(f"SHAP plot: {plot_path}")
    print(f"SHAP CSV : {csv_path}")
    print(
        "\nSHAP explains model output; it is not a clinical "
        "diagnosis or treatment recommendation."
    )

    print("=" * 72)


if __name__ == "__main__":
    main()