"""
XSepsisNet - SHAP Explainability
================================

Explains the FINAL 6-hour BiLSTM prediction.

Final model:
    6 hours
    78 features/hour
    39 clinical features
    39 missingness features
    2-layer Bidirectional LSTM
    hidden size = 64

Outputs:
    models/explainability/
        shap_feature_contributions.csv
        shap_hourly_contributions.csv
        shap_summary.json
        shap_explanation.png

Important:
    This script does NOT retrain the model.
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import shap
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "bilstm"
    / "final_bilstm_best.pt"
)

CONFIG_FILE = (
    PROJECT_ROOT
    / "models"
    / "bilstm"
    / "final_bilstm_config.json"
)

SCALER_FILE = (
    PROJECT_ROOT
    / "models"
    / "scalers"
    / "final_bilstm_scaler.json"
)

TEST_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "final_bilstm_test.npz"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "models"
    / "explainability"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# MODEL DEFINITION
# ============================================================

class BiLSTMClassifier(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        dropout,
        bidirectional=True
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        output_size = (
            hidden_size * 2
            if bidirectional
            else hidden_size
        )

        self.classifier = nn.Linear(
            output_size,
            1
        )

    def forward(self, x):

        output, _ = self.lstm(x)

        # Use the final time step.
        final_output = output[:, -1, :]

        logits = self.classifier(
            final_output
        )

        return logits


# ============================================================
# LOAD CONFIGURATION
# ============================================================

def load_config():

    with open(
        CONFIG_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


# ============================================================
# LOAD FEATURE NAMES
# ============================================================

def load_feature_names():

    with open(
        SCALER_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        scaler = json.load(f)

    clinical_features = scaler["feature_columns"]

    if len(clinical_features) != 39:
        raise ValueError(
            f"Expected 39 clinical features, "
            f"found {len(clinical_features)}"
        )

    # The final model contains:
    #
    # 39 clinical features
    # +
    # 39 missingness features
    #
    # = 78 features/hour

    missing_features = [
        f"{feature}_missing"
        for feature in clinical_features
    ]

    feature_names = (
        clinical_features
        + missing_features
    )

    if len(feature_names) != 78:
        raise ValueError(
            f"Expected 78 model features, "
            f"found {len(feature_names)}"
        )

    return clinical_features, feature_names


# ============================================================
# LOAD FINAL MODEL
# ============================================================

def load_model(config):

    checkpoint = torch.load(
        MODEL_FILE,
        map_location="cpu",
        weights_only=False
    )

    model = BiLSTMClassifier(
        input_size=checkpoint["input_size"],
        hidden_size=checkpoint["hidden_size"],
        num_layers=checkpoint["num_layers"],
        dropout=checkpoint["dropout"],
        bidirectional=config["bidirectional"],
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model


# ============================================================
# MODEL PREDICTION FUNCTION
# ============================================================

def model_predict(model, x):

    if isinstance(x, np.ndarray):

        x = torch.tensor(
            x,
            dtype=torch.float32
        )

    with torch.no_grad():

        logits = model(x)

        probabilities = torch.sigmoid(
            logits
        )

    return probabilities.numpy().reshape(-1)


# ============================================================
# SHAP WRAPPER
# ============================================================

class ModelWrapper(nn.Module):

    def __init__(self, model):

        super().__init__()

        self.model = model

    def forward(self, x):

        return self.model(x)


# ============================================================
# CREATE FLAT FEATURE NAMES
# ============================================================

def create_flat_feature_names(
    feature_names,
    sequence_length
):

    names = []

    for hour in range(sequence_length):

        for feature in feature_names:

            names.append(
                f"Hour_{hour + 1}_{feature}"
            )

    return names


# ============================================================
# AGGREGATE SHAP VALUES
# ============================================================

def aggregate_shap_values(
    shap_values,
    feature_names,
    sequence_length
):

    # Expected:
    #
    # samples × hours × features
    #

    if shap_values.ndim == 4:

        shap_values = shap_values[0]

    if shap_values.ndim == 2:

        shap_values = shap_values.reshape(
            -1,
            sequence_length,
            len(feature_names)
        )

    return shap_values


# ============================================================
# MAIN XAI PIPELINE
# ============================================================

def main():

    print("=" * 72)
    print("XSepsisNet - SHAP Explainability")
    print("=" * 72)

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    print("\nChecking required files...")

    required_files = [
        MODEL_FILE,
        CONFIG_FILE,
        SCALER_FILE,
        TEST_FILE,
    ]

    for file_path in required_files:

        if not file_path.exists():

            raise FileNotFoundError(
                f"Required file not found:\n{file_path}"
            )

        print(f"  OK: {file_path.relative_to(PROJECT_ROOT)}")

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    config = load_config()

    sequence_length = config["sequence_length"]
    input_size = config["input_size"]
    hidden_size = config["hidden_size"]
    num_layers = config["num_layers"]
    dropout = config["dropout"]

    print("\nModel configuration:")
    print(f"  Sequence length : {sequence_length}")
    print(f"  Input size      : {input_size}")
    print(f"  Hidden size     : {hidden_size}")
    print(f"  Layers          : {num_layers}")
    print(f"  Bidirectional   : {config['bidirectional']}")
    print(f"  Dropout         : {dropout}")

    # --------------------------------------------------------
    # Feature names
    # --------------------------------------------------------

    clinical_features, feature_names = (
        load_feature_names()
    )

    print(
        f"\nClinical features : {len(clinical_features)}"
    )

    print(
        f"Model features    : {len(feature_names)}"
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("\nLoading final BiLSTM...")

    model = load_model(config)

    print("Final BiLSTM loaded successfully.")

    # --------------------------------------------------------
    # Load test data
    # --------------------------------------------------------

    print("\nLoading test data...")

    data = np.load(
        TEST_FILE
    )

    X_test = data["X"]
    y_test = data["y"]

    print(
        f"Test X shape: {X_test.shape}"
    )

    print(
        f"Test y shape: {y_test.shape}"
    )

    if X_test.ndim != 3:

        raise ValueError(
            "Expected test X to have shape "
            "(samples, 6, 78)."
        )

    if X_test.shape[1] != sequence_length:

        raise ValueError(
            f"Expected {sequence_length} hours, "
            f"found {X_test.shape[1]}"
        )

    if X_test.shape[2] != input_size:

        raise ValueError(
            f"Expected {input_size} features/hour, "
            f"found {X_test.shape[2]}"
        )

    # --------------------------------------------------------
    # Select representative samples
    # --------------------------------------------------------

    # We use a small background set because SHAP on an
    # LSTM with 78 features × 6 hours can be computationally
    # expensive on CPU.

    background_size = min(
        50,
        len(X_test)
    )

    explain_size = min(
        10,
        len(X_test)
    )

    background = X_test[
        :background_size
    ]

    explain_data = X_test[
        :explain_size
    ]

    explain_labels = y_test[
        :explain_size
    ]

    print(
        f"\nSHAP background samples: "
        f"{background.shape[0]}"
    )

    print(
        f"Samples to explain: "
        f"{explain_data.shape[0]}"
    )

    # --------------------------------------------------------
    # SHAP DeepExplainer
    # --------------------------------------------------------

    print("\nCreating SHAP DeepExplainer...")

    background_tensor = torch.tensor(
        background,
        dtype=torch.float32
    )

    explain_tensor = torch.tensor(
        explain_data,
        dtype=torch.float32
    )

    wrapped_model = ModelWrapper(
        model
    )

    wrapped_model.eval()

    try:

        explainer = shap.DeepExplainer(
            wrapped_model,
            background_tensor
        )

        print(
            "SHAP DeepExplainer created."
        )

        print(
            "Calculating SHAP values..."
        )

        shap_result = explainer.shap_values(
            explain_tensor
        )

    except Exception as exc:

        print(
            "\nDeepExplainer failed."
        )

        print(
            f"Reason: {exc}"
        )

        print(
            "\nTrying SHAP GradientExplainer..."
        )

        explainer = shap.GradientExplainer(
            wrapped_model,
            background_tensor
        )

        shap_result = explainer.shap_values(
            explain_tensor
        )

    # --------------------------------------------------------
    # Normalize SHAP output
    # --------------------------------------------------------

    if isinstance(
        shap_result,
        list
    ):

        shap_values = shap_result[0]

    else:

        shap_values = shap_result

    shap_values = np.asarray(
        shap_values
    )

    print(
        "\nRaw SHAP shape:",
        shap_values.shape
    )

    # SHAP can return:
    #
    # samples × hours × features
    #
    # or:
    #
    # samples × hours × features × outputs
    #

    if shap_values.ndim == 4:

        shap_values = shap_values[:, :, :, 0]

    if shap_values.ndim != 3:

        raise ValueError(
            "Unexpected SHAP shape: "
            f"{shap_values.shape}"
        )

    # --------------------------------------------------------
    # Verify shape
    # --------------------------------------------------------

    expected_shape = (
        explain_size,
        sequence_length,
        input_size
    )

    if shap_values.shape != expected_shape:

        raise ValueError(
            f"Unexpected SHAP dimensions.\n"
            f"Expected: {expected_shape}\n"
            f"Found:    {shap_values.shape}"
        )

    print(
        "Verified SHAP shape:",
        shap_values.shape
    )

    # --------------------------------------------------------
    # Model predictions
    # --------------------------------------------------------

    probabilities = model_predict(
        model,
        explain_data
    )

    # --------------------------------------------------------
    # Create detailed contribution table
    # --------------------------------------------------------

    detailed_rows = []

    for sample_idx in range(
        explain_size
    ):

        for hour_idx in range(
            sequence_length
        ):

            for feature_idx in range(
                input_size
            ):

                detailed_rows.append(
                    {
                        "sample_index":
                            sample_idx,

                        "hour":
                            hour_idx + 1,

                        "feature":
                            feature_names[
                                feature_idx
                            ],

                        "shap_value":
                            float(
                                shap_values[
                                    sample_idx,
                                    hour_idx,
                                    feature_idx
                                ]
                            ),

                        "absolute_shap":
                            float(
                                abs(
                                    shap_values[
                                        sample_idx,
                                        hour_idx,
                                        feature_idx
                                    ]
                                )
                            ),

                        "model_probability":
                            float(
                                probabilities[
                                    sample_idx
                                ]
                            ),

                        "true_label":
                            int(
                                explain_labels[
                                    sample_idx
                                ]
                            ),
                    }
                )

    detailed_df = pd.DataFrame(
        detailed_rows
    )

    detailed_file = (
        OUTPUT_DIR
        / "shap_hourly_contributions.csv"
    )

    detailed_df.to_csv(
        detailed_file,
        index=False
    )

    print(
        f"\nSaved:\n"
        f"  {detailed_file.relative_to(PROJECT_ROOT)}"
    )

    # --------------------------------------------------------
    # Aggregate clinical features
    # --------------------------------------------------------

    clinical_rows = []

    for sample_idx in range(
        explain_size
    ):

        for feature_idx, feature in enumerate(
            clinical_features
        ):

            # Clinical feature contribution
            # across all 6 hours.

            values = shap_values[
                sample_idx,
                :,
                feature_idx
            ]

            total_contribution = float(
                np.sum(values)
            )

            absolute_contribution = float(
                np.sum(np.abs(values))
            )

            max_hour_index = int(
                np.argmax(np.abs(values))
            )

            clinical_rows.append(
                {
                    "sample_index":
                        sample_idx,

                    "feature":
                        feature,

                    "total_shap":
                        total_contribution,

                    "absolute_shap":
                        absolute_contribution,

                    "most_influential_hour":
                        max_hour_index + 1,

                    "model_probability":
                        float(
                            probabilities[
                                sample_idx
                            ]
                        ),

                    "true_label":
                        int(
                            explain_labels[
                                sample_idx
                            ]
                        ),
                }
            )

    clinical_df = pd.DataFrame(
        clinical_rows
    )

    clinical_file = (
        OUTPUT_DIR
        / "shap_feature_contributions.csv"
    )

    clinical_df.to_csv(
        clinical_file,
        index=False
    )

    print(
        f"  {clinical_file.relative_to(PROJECT_ROOT)}"
    )

    # --------------------------------------------------------
    # Overall feature importance
    # --------------------------------------------------------

    overall_importance = (
        clinical_df
        .groupby("feature")[
            "absolute_shap"
        ]
        .mean()
        .sort_values(
            ascending=False
        )
    )

    print(
        "\nTop clinical features by SHAP importance:"
    )

    for feature, value in (
        overall_importance.head(10).items()
    ):

        print(
            f"  {feature:<40} "
            f"{value:.6f}"
        )

    # --------------------------------------------------------
    # Explain first sample
    # --------------------------------------------------------

    first_sample_df = (
        clinical_df[
            clinical_df["sample_index"] == 0
        ]
        .sort_values(
            "absolute_shap",
            ascending=False
        )
        .head(10)
    )

    print(
        "\nTop contributors for sample 0:"
    )

    for _, row in first_sample_df.iterrows():

        direction = (
            "INCREASES risk"
            if row["total_shap"] > 0
            else "DECREASES risk"
        )

        print(
            f"  {row['feature']:<35} "
            f"{row['total_shap']:+.6f} "
            f"({direction})"
        )

    # --------------------------------------------------------
    # Summary JSON
    # --------------------------------------------------------

    summary = {

        "model": "final_bilstm",

        "sequence_length":
            sequence_length,

        "input_features_per_hour":
            input_size,

        "clinical_features":
            len(clinical_features),

        "missingness_features":
            len(clinical_features),

        "shap_version":
            shap.__version__,

        "background_samples":
            background_size,

        "explained_samples":
            explain_size,

        "test_samples":
            int(len(X_test)),

        "mean_test_probability":
            float(
                np.mean(
                    model_predict(
                        model,
                        X_test
                    )
                )
            ),

        "top_features": [
            {
                "feature": feature,
                "mean_absolute_shap":
                    float(value)
            }
            for feature, value
            in overall_importance.head(10).items()
        ],
    }

    summary_file = (
        OUTPUT_DIR
        / "shap_summary.json"
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

    print(
        f"\nSaved:\n"
        f"  {summary_file.relative_to(PROJECT_ROOT)}"
    )

    # --------------------------------------------------------
    # Simple importance plot
    # --------------------------------------------------------

    plot_data = (
        overall_importance
        .head(15)
        .sort_values()
    )

    plt.figure(
        figsize=(10, 7)
    )

    plt.barh(
        plot_data.index,
        plot_data.values
    )

    plt.xlabel(
        "Mean Absolute SHAP Value"
    )

    plt.ylabel(
        "Clinical Feature"
    )

    plt.title(
        "XSepsisNet - Top SHAP Feature Importance"
    )

    plt.tight_layout()

    plot_file = (
        OUTPUT_DIR
        / "shap_explanation.png"
    )

    plt.savefig(
        plot_file,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"  {plot_file.relative_to(PROJECT_ROOT)}"
    )

    # --------------------------------------------------------
    # Final status
    # --------------------------------------------------------

    print("\n" + "=" * 72)
    print("SHAP EXPLAINABILITY COMPLETE")
    print("=" * 72)

    print(
        "\nThe trained BiLSTM was NOT modified or retrained."
    )

    print(
        "SHAP explanations were generated for the final "
        "6-hour model."
    )

    print(
        "\nOutput directory:"
    )

    print(
        f"  {OUTPUT_DIR}"
    )


if __name__ == "__main__":

    main()