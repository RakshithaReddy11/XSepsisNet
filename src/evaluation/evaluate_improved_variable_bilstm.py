# ============================================================
# XSepsisNet
# STAGE 2 - IMPROVED VARIABLE-LENGTH BiLSTM EVALUATION
#
# TEST EVALUATION ONLY
# ============================================================

import os
import json
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    brier_score_loss,
    log_loss,
)

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = r"C:\Users\raksh\Downloads\XSepsisNet"

TEST_NPZ = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "final_3to12_test.npz",
)

WINDOW_INDEX = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "final_3to12_window_index.csv",
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "bilstm",
    "improved_variable_bilstm_best.pt",
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "models",
    "evaluation",
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


# ============================================================
# SETTINGS
# ============================================================

DEVICE = torch.device("cpu")

INPUT_SIZE = 78
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.45
BIDIRECTIONAL = True

BATCH_SIZE = 512
THRESHOLD = 0.50


# ============================================================
# MODEL
# EXACT SAME ARCHITECTURE USED DURING TRAINING
# ============================================================

class ImprovedVariableBiLSTM(nn.Module):

    def __init__(
        self,
        input_size=78,
        hidden_size=64,
        num_layers=2,
        dropout=0.45,
        bidirectional=True,
    ):
        super().__init__()

        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=(
                dropout
                if num_layers > 1
                else 0.0
            ),
            bidirectional=bidirectional,
        )

        direction_factor = (
            2 if bidirectional else 1
        )

        self.dropout = nn.Dropout(
            dropout
        )

        self.classifier = nn.Sequential(
            nn.Linear(
                hidden_size * direction_factor,
                hidden_size,
            ),
            nn.LayerNorm(
                hidden_size
            ),
            nn.ReLU(),
            nn.Dropout(
                dropout
            ),
            nn.Linear(
                hidden_size,
                1,
            ),
        )

    def forward(
        self,
        x,
        lengths,
    ):

        lengths_cpu = (
            lengths.detach().cpu()
        )

        packed = (
            nn.utils.rnn.pack_padded_sequence(
                x,
                lengths_cpu,
                batch_first=True,
                enforce_sorted=False,
            )
        )

        _, (
            h_n,
            _,
        ) = self.lstm(
            packed
        )

        if self.bidirectional:

            forward_hidden = h_n[-2]

            backward_hidden = h_n[-1]

            hidden = torch.cat(
                [
                    forward_hidden,
                    backward_hidden,
                ],
                dim=1,
            )

        else:

            hidden = h_n[-1]

        hidden = self.dropout(
            hidden
        )

        logits = self.classifier(
            hidden
        )

        return logits.squeeze(1)


# ============================================================
# LOAD TEST DATA
# ============================================================

def load_test_data():

    print(
        "\n[2/8] Loading untouched test data..."
    )

    data = np.load(
        TEST_NPZ,
        allow_pickle=True,
    )

    print("\nNPZ contents:")

    for key in data.files:

        value = data[key]

        print(
            f"  {key}: "
            f"shape={value.shape}, "
            f"dtype={value.dtype}"
        )

    if "X" not in data.files:
        raise KeyError(
            "X not found in test NPZ."
        )

    if "lengths" not in data.files:
        raise KeyError(
            "lengths not found in test NPZ."
        )

    if "y" not in data.files:
        raise KeyError(
            "y not found in test NPZ."
        )

    X = np.asarray(
        data["X"],
        dtype=np.float32,
    )

    lengths = np.asarray(
        data["lengths"],
        dtype=np.int64,
    )

    y = np.asarray(
        data["y"]
    ).astype(int)

    return X, lengths, y


# ============================================================
# VERIFY TEST DATA
# ============================================================

def verify_test_data(
    X,
    lengths,
    y,
):

    print(
        "\n[3/8] Verifying test data..."
    )

    if X.ndim != 3:
        raise ValueError(
            f"Invalid X shape: {X.shape}"
        )

    if X.shape[1] != 12:
        raise ValueError(
            f"Expected padded length 12, "
            f"got {X.shape[1]}"
        )

    if X.shape[2] != INPUT_SIZE:
        raise ValueError(
            f"Expected {INPUT_SIZE} features, "
            f"got {X.shape[2]}"
        )

    if len(X) != len(lengths):
        raise ValueError(
            "X and lengths mismatch."
        )

    if len(X) != len(y):
        raise ValueError(
            "X and y mismatch."
        )

    if np.any(lengths < 3):
        raise ValueError(
            "History length below 3 found."
        )

    if np.any(lengths > 12):
        raise ValueError(
            "History length above 12 found."
        )

    print(
        "Feature dimension : PASSED"
    )

    print(
        "Sample alignment  : PASSED"
    )

    print(
        "History range     : PASSED"
    )

    print(
        f"History range     : "
        f"{lengths.min()} - "
        f"{lengths.max()} hours"
    )

    print(
        f"Test samples      : "
        f"{len(y)}"
    )

    print(
        f"Positive samples  : "
        f"{np.sum(y == 1)}"
    )

    print(
        f"Negative samples  : "
        f"{np.sum(y == 0)}"
    )


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print(
        "\n[4/8] Loading improved best model..."
    )

    model = ImprovedVariableBiLSTM(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        bidirectional=BIDIRECTIONAL,
    ).to(DEVICE)

    # IMPORTANT:
    # This checkpoint was created by our own training script.
    # PyTorch 2.6+ defaults to weights_only=True.
    # Our checkpoint contains metadata with NumPy scalar values,
    # so weights_only=False is required for this trusted local file.
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    print(
        f"Checkpoint type: "
        f"{type(checkpoint)}"
    )

    if isinstance(
        checkpoint,
        dict,
    ):

        print(
            "\nCheckpoint information:"
        )

        for key in checkpoint.keys():

            if key != "model_state_dict":

                print(
                    f"  {key}: "
                    f"{checkpoint[key]}"
                )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):

        state_dict = (
            checkpoint[
                "model_state_dict"
            ]
        )

    else:

        state_dict = checkpoint

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model.eval()

    print(
        "\nModel architecture: PASSED"
    )

    print(
        "Checkpoint loading : PASSED"
    )

    return model


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

def generate_predictions(
    model,
    X,
    lengths,
):

    print(
        "\n[5/8] Generating test predictions..."
    )

    probabilities = []

    model.eval()

    with torch.no_grad():

        for start in range(
            0,
            len(X),
            BATCH_SIZE,
        ):

            end = min(
                start + BATCH_SIZE,
                len(X),
            )

            xb = torch.tensor(
                X[start:end],
                dtype=torch.float32,
                device=DEVICE,
            )

            lb = torch.tensor(
                lengths[start:end],
                dtype=torch.long,
                device=DEVICE,
            )

            logits = model(
                xb,
                lb,
            )

            probs = torch.sigmoid(
                logits
            )

            probabilities.extend(
                probs.cpu().numpy()
            )

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    print(
        f"Prediction count : "
        f"{len(probabilities)}"
    )

    print(
        f"Probability min  : "
        f"{probabilities.min():.6f}"
    )

    print(
        f"Probability max  : "
        f"{probabilities.max():.6f}"
    )

    print(
        f"Probability mean : "
        f"{probabilities.mean():.6f}"
    )

    return probabilities


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    probabilities,
    threshold=0.50,
):

    y_true = np.asarray(
        y_true
    ).astype(int)

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = (
        confusion_matrix(
            y_true,
            predictions,
            labels=[0, 1],
        ).ravel()
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    clipped = np.clip(
        probabilities,
        1e-7,
        1 - 1e-7,
    )

    return {

        "samples": int(
            len(y_true)
        ),

        "positive_samples": int(
            np.sum(y_true == 1)
        ),

        "negative_samples": int(
            np.sum(y_true == 0)
        ),

        "positive_rate": float(
            np.mean(y_true)
        ),

        "AUROC": float(
            roc_auc_score(
                y_true,
                probabilities,
            )
        ),

        "AUPRC": float(
            average_precision_score(
                y_true,
                probabilities,
            )
        ),

        "Precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),

        "Recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),

        "F1": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),

        "Accuracy": float(
            accuracy_score(
                y_true,
                predictions,
            )
        ),

        "Specificity": float(
            specificity
        ),

        "Brier": float(
            brier_score_loss(
                y_true,
                probabilities,
            )
        ),

        "LogLoss": float(
            log_loss(
                y_true,
                clipped,
                labels=[0, 1],
            )
        ),

        "Threshold": float(
            threshold
        ),

        "TN": int(tn),

        "FP": int(fp),

        "FN": int(fn),

        "TP": int(tp),
    }


# ============================================================
# HISTORY LENGTH EVALUATION
# ============================================================

def evaluate_by_history(
    y_true,
    probabilities,
    lengths,
):

    print(
        "\n[7/8] Evaluating by history length..."
    )

    rows = []

    for hours in range(
        3,
        13,
    ):

        mask = (
            lengths == hours
        )

        if not np.any(mask):
            continue

        metrics = calculate_metrics(
            y_true[mask],
            probabilities[mask],
            THRESHOLD,
        )

        metrics[
            "history_hours"
        ] = hours

        rows.append(
            metrics
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

def threshold_analysis(
    y_true,
    probabilities,
):

    rows = []

    for threshold in np.arange(
        0.10,
        0.91,
        0.05,
    ):

        metrics = calculate_metrics(
            y_true,
            probabilities,
            float(threshold),
        )

        rows.append(
            metrics
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# WINDOW INDEX VERIFICATION
# ============================================================

def verify_window_index(
    y_true,
    lengths,
):

    print(
        "\nVerifying window index..."
    )

    window_df = pd.read_csv(
        WINDOW_INDEX
    )

    if "split" not in window_df.columns:

        print(
            "WARNING: split column missing."
        )

        return

    test_df = (
        window_df[
            window_df["split"]
            .astype(str)
            .str.lower()
            == "test"
        ]
        .reset_index(drop=True)
    )

    if len(test_df) != len(y_true):

        raise ValueError(
            "Window index and test NPZ "
            "sample counts do not match."
        )

    if "label" in test_df.columns:

        index_labels = (
            test_df["label"]
            .astype(int)
            .to_numpy()
        )

        if np.array_equal(
            index_labels,
            y_true,
        ):

            print(
                "Label alignment: PASSED"
            )

        else:

            raise ValueError(
                "Label alignment FAILED."
            )

    if "history_hours" in test_df.columns:

        index_lengths = (
            test_df[
                "history_hours"
            ]
            .astype(int)
            .to_numpy()
        )

        if np.array_equal(
            index_lengths,
            lengths,
        ):

            print(
                "History-length alignment: PASSED"
            )

        else:

            raise ValueError(
                "History-length alignment FAILED."
            )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)

    print(
        "XSepsisNet"
    )

    print(
        "STAGE 2 - IMPROVED VARIABLE-LENGTH BiLSTM"
    )

    print(
        "3-12 HOUR TEST EVALUATION"
    )

    print("=" * 80)

    print(
        f"\nProject root : "
        f"{PROJECT_ROOT}"
    )

    print(
        f"Device       : "
        f"{DEVICE}"
    )

    # --------------------------------------------------------
    # 1. File checks
    # --------------------------------------------------------

    print(
        "\n[1/8] Checking required files..."
    )

    for path in [
        TEST_NPZ,
        WINDOW_INDEX,
        MODEL_PATH,
    ]:

        if not os.path.exists(path):

            raise FileNotFoundError(
                f"\nFile not found:\n{path}"
            )

        print(
            f"FOUND: {path}"
        )

    # --------------------------------------------------------
    # 2. Load test data
    # --------------------------------------------------------

    X, lengths, y_true = (
        load_test_data()
    )

    # --------------------------------------------------------
    # 3. Verify
    # --------------------------------------------------------

    verify_test_data(
        X,
        lengths,
        y_true,
    )

    # --------------------------------------------------------
    # 4. Load model
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # 5. Predict
    # --------------------------------------------------------

    probabilities = (
        generate_predictions(
            model,
            X,
            lengths,
        )
    )

    # --------------------------------------------------------
    # 6. Overall metrics
    # --------------------------------------------------------

    print(
        "\n[6/8] Calculating overall metrics..."
    )

    overall = calculate_metrics(
        y_true,
        probabilities,
        THRESHOLD,
    )

    print(
        "\n" + "=" * 80
    )

    print(
        "IMPROVED VARIABLE-LENGTH"
    )

    print(
        "OVERALL TEST RESULTS"
    )

    print(
        "=" * 80
    )

    for key, value in overall.items():

        if isinstance(
            value,
            float,
        ):

            print(
                f"{key:<20}: "
                f"{value:.6f}"
            )

        else:

            print(
                f"{key:<20}: "
                f"{value}"
            )

    # --------------------------------------------------------
    # 7. By history
    # --------------------------------------------------------

    history_df = (
        evaluate_by_history(
            y_true,
            probabilities,
            lengths,
        )
    )

    print(
        "\n" + "=" * 80
    )

    print(
        "RESULTS BY HISTORY LENGTH"
    )

    print(
        "=" * 80
    )

    columns = [
        "history_hours",
        "samples",
        "positive_rate",
        "AUROC",
        "AUPRC",
        "Precision",
        "Recall",
        "F1",
        "Specificity",
        "Brier",
    ]

    print(
        history_df[
            columns
        ].to_string(
            index=False,
            float_format=lambda x:
            f"{x:.4f}",
        )
    )

    # --------------------------------------------------------
    # Verify alignment
    # --------------------------------------------------------

    verify_window_index(
        y_true,
        lengths,
    )

    # --------------------------------------------------------
    # Threshold analysis
    # --------------------------------------------------------

    print(
        "\nCreating threshold analysis..."
    )

    threshold_df = (
        threshold_analysis(
            y_true,
            probabilities,
        )
    )

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    prediction_path = os.path.join(
        OUTPUT_DIR,
        "improved_variable_test_predictions.npz",
    )

    np.savez_compressed(
        prediction_path,
        y_true=y_true.astype(
            np.float32
        ),
        y_probability=probabilities.astype(
            np.float32
        ),
        history_hours=lengths.astype(
            np.int16
        ),
    )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    summary_path = os.path.join(
        OUTPUT_DIR,
        "improved_variable_evaluation_summary.json",
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            overall,
            f,
            indent=4,
        )

    # --------------------------------------------------------
    # Save history evaluation
    # --------------------------------------------------------

    history_path = os.path.join(
        OUTPUT_DIR,
        "improved_variable_evaluation_by_history.csv",
    )

    history_df.to_csv(
        history_path,
        index=False,
    )

    # --------------------------------------------------------
    # Save threshold analysis
    # --------------------------------------------------------

    threshold_path = os.path.join(
        OUTPUT_DIR,
        "improved_variable_threshold_analysis.csv",
    )

    threshold_df.to_csv(
        threshold_path,
        index=False,
    )

    # --------------------------------------------------------
    # Save confusion matrix
    # --------------------------------------------------------

    predictions = (
        probabilities >= THRESHOLD
    ).astype(int)

    cm = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    )

    cm_df = pd.DataFrame(
        cm,
        index=[
            "Actual Negative",
            "Actual Positive",
        ],
        columns=[
            "Predicted Negative",
            "Predicted Positive",
        ],
    )

    cm_path = os.path.join(
        OUTPUT_DIR,
        "improved_variable_confusion_matrix.csv",
    )

    cm_df.to_csv(
        cm_path
    )

    # --------------------------------------------------------
    # Save prediction table
    # --------------------------------------------------------

    prediction_table = pd.DataFrame(
        {
            "y_true": y_true,
            "probability": probabilities,
            "prediction": predictions,
            "history_hours": lengths,
        }
    )

    table_path = os.path.join(
        OUTPUT_DIR,
        "improved_variable_test_prediction_table.csv",
    )

    prediction_table.to_csv(
        table_path,
        index=False,
    )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print(
        "\n" + "=" * 80
    )

    print(
        "STAGE 2 EVALUATION COMPLETED"
    )

    print(
        "=" * 80
    )

    print(
        "\nSaved files:"
    )

    print(
        prediction_path
    )

    print(
        summary_path
    )

    print(
        history_path
    )

    print(
        threshold_path
    )

    print(
        cm_path
    )

    print(
        table_path
    )

    print(
        "\nNo training was performed."
    )

    print(
        "No calibration was performed."
    )

    print(
        "No XAI was performed."
    )

    print(
        "No FastAPI/frontend integration "
        "was performed."
    )

    print(
        "\nNext:"
    )

    print(
        "Compare improved variable-length "
        "model with the previous models."
    )

    print(
        "=" * 80
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()