# ============================================================
# XSepsisNet - FINAL STAGE 2
# 3-12 HOUR BiLSTM EVALUATION
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
    "final_3to12_bilstm_best.pt",
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "models",
    "evaluation",
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

DEVICE = torch.device("cpu")

INPUT_SIZE = 78
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.30
BIDIRECTIONAL = True

THRESHOLD = 0.50
BATCH_SIZE = 512


# ============================================================
# MODEL
#
# IMPORTANT:
# The training checkpoint contains:
# classifier.0
# classifier.3
#
# Therefore the evaluation architecture MUST match it.
# ============================================================

class VariableBiLSTM(nn.Module):

    def __init__(
        self,
        input_size=78,
        hidden_size=64,
        num_layers=2,
        dropout=0.30,
        bidirectional=True,
    ):
        super().__init__()

        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        direction_factor = 2 if bidirectional else 1

        self.dropout = nn.Dropout(
            dropout
        )

        # IMPORTANT:
        # This matches the saved training checkpoint:
        #
        # classifier.0.weight
        # classifier.0.bias
        # classifier.3.weight
        # classifier.3.bias
        #
        self.classifier = nn.Sequential(
            nn.Linear(
                hidden_size * direction_factor,
                hidden_size,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(
                hidden_size,
                1,
            ),
        )

    def forward(
        self,
        x,
        lengths=None,
    ):

        if lengths is not None:

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

            packed_output, (
                h_n,
                c_n,
            ) = self.lstm(packed)

        else:

            output, (
                h_n,
                c_n,
            ) = self.lstm(x)

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
        "\n[2/7] Loading untouched test data..."
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

    if "y" not in data.files:
        raise KeyError(
            "y not found in test NPZ."
        )

    X = np.asarray(
        data["X"],
        dtype=np.float32,
    )

    y = np.asarray(
        data["y"]
    ).astype(int)

    if "lengths" in data.files:

        lengths = np.asarray(
            data["lengths"]
        ).astype(np.int64)

    else:

        raise KeyError(
            "lengths not found in test NPZ."
        )

    print(
        f"\nX shape      : {X.shape}"
    )

    print(
        f"lengths shape: {lengths.shape}"
    )

    print(
        f"y shape      : {y.shape}"
    )

    return X, lengths, y


# ============================================================
# VERIFY DATA
# ============================================================

def verify_data(
    X,
    lengths,
    y,
):

    print(
        "\n[3/7] Verifying test data..."
    )

    if X.ndim != 3:

        raise ValueError(
            f"Expected X to be 3D. "
            f"Got {X.shape}"
        )

    if X.shape[1] != 12:

        raise ValueError(
            f"Expected padded sequence "
            f"length 12. Got {X.shape}"
        )

    if X.shape[2] != INPUT_SIZE:

        raise ValueError(
            f"Expected {INPUT_SIZE} "
            f"features. Got {X.shape[2]}"
        )

    if len(X) != len(lengths):

        raise ValueError(
            "X and lengths count mismatch."
        )

    if len(X) != len(y):

        raise ValueError(
            "X and y count mismatch."
        )

    if np.any(lengths < 3):

        raise ValueError(
            "Found history length below 3."
        )

    if np.any(lengths > 12):

        raise ValueError(
            "Found history length above 12."
        )

    print(
        "Feature dimension: PASSED"
    )

    print(
        "Sample alignment : PASSED"
    )

    print(
        "History range    : PASSED"
    )

    print(
        f"History lengths  : "
        f"{lengths.min()} - "
        f"{lengths.max()} hours"
    )

    print(
        f"Positive samples : "
        f"{np.sum(y == 1)}"
    )

    print(
        f"Negative samples : "
        f"{np.sum(y == 0)}"
    )


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print(
        "\n[4/7] Loading best trained model..."
    )

    model = VariableBiLSTM(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        bidirectional=BIDIRECTIONAL,
    ).to(DEVICE)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
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
            "Checkpoint keys:"
        )

        for key in checkpoint.keys():

            print(
                f"  {key}"
            )

    # --------------------------------------------------------
    # Extract state dictionary
    # --------------------------------------------------------

    if isinstance(
        checkpoint,
        dict
    ) and "model_state_dict" in checkpoint:

        state_dict = (
            checkpoint[
                "model_state_dict"
            ]
        )

    elif isinstance(
        checkpoint,
        dict
    ) and "state_dict" in checkpoint:

        state_dict = (
            checkpoint[
                "state_dict"
            ]
        )

    else:

        state_dict = checkpoint

    # --------------------------------------------------------
    # Load exact architecture
    # --------------------------------------------------------

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model.eval()

    print(
        "Model architecture: PASSED"
    )

    print(
        "Checkpoint loading : PASSED"
    )

    return model


# ============================================================
# PREDICTIONS
# ============================================================

def generate_predictions(
    model,
    X,
    lengths,
):

    print(
        "\n[5/7] Generating TEST predictions..."
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
    y_probability,
    threshold=0.50,
):

    y_true = np.asarray(
        y_true
    ).astype(int)

    y_probability = np.asarray(
        y_probability,
        dtype=np.float64,
    )

    y_pred = (
        y_probability >= threshold
    ).astype(int)

    tn, fp, fn, tp = (
        confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1],
        ).ravel()
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    clipped_probability = np.clip(
        y_probability,
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
                y_probability,
            )
        ),

        "AUPRC": float(
            average_precision_score(
                y_true,
                y_probability,
            )
        ),

        "Precision": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),

        "Recall": float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),

        "F1": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),

        "Accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),

        "Specificity": float(
            specificity
        ),

        "Brier": float(
            brier_score_loss(
                y_true,
                y_probability,
            )
        ),

        "LogLoss": float(
            log_loss(
                y_true,
                clipped_probability,
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
# THRESHOLD ANALYSIS
# ============================================================

def threshold_analysis(
    y_true,
    y_probability,
):

    rows = []

    thresholds = np.arange(
        0.10,
        0.91,
        0.05,
    )

    for threshold in thresholds:

        metrics = calculate_metrics(
            y_true,
            y_probability,
            threshold=float(
                threshold
            ),
        )

        rows.append(
            metrics
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# HISTORY-LENGTH ANALYSIS
# ============================================================

def evaluate_by_history(
    y_true,
    y_probability,
    lengths,
):

    print(
        "\n[6/7] Evaluating by history length..."
    )

    rows = []

    for history_hours in range(
        3,
        13,
    ):

        mask = (
            lengths
            == history_hours
        )

        count = int(
            np.sum(mask)
        )

        if count == 0:
            continue

        y_h = y_true[mask]

        p_h = y_probability[mask]

        metrics = calculate_metrics(
            y_h,
            p_h,
            THRESHOLD,
        )

        metrics[
            "history_hours"
        ] = history_hours

        rows.append(
            metrics
        )

    result = pd.DataFrame(
        rows
    )

    return result


# ============================================================
# VERIFY WINDOW INDEX
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
            "WARNING: 'split' column "
            "not found."
        )

        return

    test_windows = (
        window_df[
            window_df["split"]
            .astype(str)
            .str.lower()
            == "test"
        ]
        .reset_index(drop=True)
    )

    if len(test_windows) != len(y_true):

        raise ValueError(
            "\nWindow-index count mismatch.\n"
            f"NPZ samples  : {len(y_true)}\n"
            f"Test windows : {len(test_windows)}"
        )

    if "label" in test_windows.columns:

        index_labels = (
            test_windows["label"]
            .astype(int)
            .to_numpy()
        )

        if not np.array_equal(
            index_labels,
            y_true,
        ):

            raise ValueError(
                "Label alignment mismatch."
            )

        print(
            "Label alignment: PASSED"
        )

    if "history_hours" in test_windows.columns:

        index_lengths = (
            test_windows[
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

            print(
                "WARNING: history-length "
                "ordering differs."
            )

    else:

        print(
            "History-hours column not found "
            "in window index."
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print(
        "XSepsisNet - FINAL STAGE 2"
    )
    print(
        "3-12 HOUR BiLSTM EVALUATION"
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
        "\n[1/7] Checking required files..."
    )

    required_files = [
        TEST_NPZ,
        WINDOW_INDEX,
        MODEL_PATH,
    ]

    for path in required_files:

        if not os.path.exists(path):

            raise FileNotFoundError(
                f"\nRequired file not found:\n"
                f"{path}"
            )

        print(
            f"FOUND: {path}"
        )

    # --------------------------------------------------------
    # 2. Load data
    # --------------------------------------------------------

    X, lengths, y_true = (
        load_test_data()
    )

    # --------------------------------------------------------
    # 3. Verify
    # --------------------------------------------------------

    verify_data(
        X,
        lengths,
        y_true,
    )

    # --------------------------------------------------------
    # 4. Model
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # 5. Prediction
    # --------------------------------------------------------

    y_probability = (
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
        "\n[6/7] Calculating overall "
        "evaluation metrics..."
    )

    overall = calculate_metrics(
        y_true,
        y_probability,
        THRESHOLD,
    )

    print(
        "\n" + "=" * 80
    )

    print(
        "OVERALL 3-12 HOUR TEST RESULTS"
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
    # History analysis
    # --------------------------------------------------------

    history_results = (
        evaluate_by_history(
            y_true,
            y_probability,
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
        history_results[
            columns
        ].to_string(
            index=False,
            float_format=lambda x:
            f"{x:.4f}",
        )
    )

    # --------------------------------------------------------
    # Window verification
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

    threshold_results = (
        threshold_analysis(
            y_true,
            y_probability,
        )
    )

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    prediction_path = os.path.join(
        OUTPUT_DIR,
        "final_3to12_test_predictions.npz",
    )

    np.savez_compressed(
        prediction_path,
        y_true=y_true.astype(
            np.float32
        ),
        y_probability=y_probability.astype(
            np.float32
        ),
        history_hours=lengths.astype(
            np.int16
        ),
    )

    # --------------------------------------------------------
    # Save overall metrics
    # --------------------------------------------------------

    metrics_path = os.path.join(
        OUTPUT_DIR,
        "final_3to12_evaluation_summary.json",
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            overall,
            f,
            indent=4,
        )

    # --------------------------------------------------------
    # Save history metrics
    # --------------------------------------------------------

    history_path = os.path.join(
        OUTPUT_DIR,
        "final_3to12_evaluation_by_history.csv",
    )

    history_results.to_csv(
        history_path,
        index=False,
    )

    # --------------------------------------------------------
    # Save threshold analysis
    # --------------------------------------------------------

    threshold_path = os.path.join(
        OUTPUT_DIR,
        "final_3to12_threshold_analysis.csv",
    )

    threshold_results.to_csv(
        threshold_path,
        index=False,
    )

    # --------------------------------------------------------
    # Save confusion matrix
    # --------------------------------------------------------

    predictions = (
        y_probability
        >= THRESHOLD
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
        "final_3to12_confusion_matrix.csv",
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
            "probability": y_probability,
            "prediction": predictions,
            "history_hours": lengths,
        }
    )

    prediction_table_path = os.path.join(
        OUTPUT_DIR,
        "final_3to12_test_prediction_table.csv",
    )

    prediction_table.to_csv(
        prediction_table_path,
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
        f"1. {prediction_path}"
    )

    print(
        f"2. {metrics_path}"
    )

    print(
        f"3. {history_path}"
    )

    print(
        f"4. {threshold_path}"
    )

    print(
        f"5. {cm_path}"
    )

    print(
        f"6. {prediction_table_path}"
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
        "\nNEXT:"
    )

    print(
        "We will review the 3-12 hour "
        "results against the existing "
        "6-hour baseline."
    )

    print(
        "=" * 80
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()