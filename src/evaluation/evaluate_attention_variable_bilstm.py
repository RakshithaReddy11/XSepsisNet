from pathlib import Path
import json
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
    log_loss
)


# ============================================================
# XSepsisNet
# STAGE 2 - V2 ATTENTION BiLSTM
# 3-12 HOUR TEST EVALUATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "final_3to12_test.npz"
)

WINDOW_INDEX_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "final_3to12_window_index.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "bilstm"
    / "attention_variable_bilstm_best.pt"
)

EVALUATION_DIR = (
    PROJECT_ROOT
    / "models"
    / "evaluation"
)

EVALUATION_DIR.mkdir(
    parents=True,
    exist_ok=True
)

PREDICTIONS_PATH = (
    EVALUATION_DIR
    / "attention_variable_test_predictions.npz"
)

SUMMARY_PATH = (
    EVALUATION_DIR
    / "attention_variable_evaluation_summary.json"
)

HISTORY_RESULTS_PATH = (
    EVALUATION_DIR
    / "attention_variable_evaluation_by_history.csv"
)

THRESHOLD_PATH = (
    EVALUATION_DIR
    / "attention_variable_threshold_analysis.csv"
)

CONFUSION_PATH = (
    EVALUATION_DIR
    / "attention_variable_confusion_matrix.csv"
)

PREDICTION_TABLE_PATH = (
    EVALUATION_DIR
    / "attention_variable_test_prediction_table.csv"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

INPUT_SIZE = 78
HIDDEN_SIZE = 96
NUM_LAYERS = 2

HISTORY_MIN = 3
HISTORY_MAX = 12

DEFAULT_THRESHOLD = 0.50


# ============================================================
# ATTENTION BiLSTM
# EXACT ARCHITECTURE USED DURING TRAINING
# ============================================================

class AttentionBiLSTM(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        dropout,
        classifier_dropout
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=(
                dropout
                if num_layers > 1
                else 0.0
            )
        )

        representation_size = hidden_size * 2

        self.attention = nn.Sequential(
            nn.Linear(
                representation_size,
                hidden_size
            ),
            nn.Tanh(),
            nn.Linear(
                hidden_size,
                1
            )
        )

        self.classifier = nn.Sequential(
            nn.Linear(
                representation_size,
                64
            ),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(
                classifier_dropout
            ),
            nn.Linear(
                64,
                1
            )
        )

    def forward(
        self,
        x,
        lengths
    ):

        max_length = x.size(1)

        packed = (
            nn.utils.rnn.pack_padded_sequence(
                x,
                lengths.cpu(),
                batch_first=True,
                enforce_sorted=False
            )
        )

        packed_output, _ = self.lstm(
            packed
        )

        lstm_output, _ = (
            nn.utils.rnn.pad_packed_sequence(
                packed_output,
                batch_first=True,
                total_length=max_length
            )
        )

        time_indices = torch.arange(
            max_length,
            device=x.device
        ).unsqueeze(0)

        valid_mask = (
            time_indices
            < lengths.unsqueeze(1)
        )

        attention_scores = (
            self.attention(
                lstm_output
            ).squeeze(-1)
        )

        attention_scores = (
            attention_scores.masked_fill(
                ~valid_mask,
                -1e9
            )
        )

        attention_weights = torch.softmax(
            attention_scores,
            dim=1
        )

        context = torch.sum(
            lstm_output
            * attention_weights.unsqueeze(-1),
            dim=1
        )

        logits = (
            self.classifier(
                context
            ).squeeze(-1)
        )

        return logits


# ============================================================
# CHECK FILE
# ============================================================

def check_file(path):

    if not path.exists():

        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )

    print(
        f"FOUND: {path}"
    )


# ============================================================
# LOAD TEST DATA
# ============================================================

def load_test_data():

    data = np.load(
        TEST_PATH
    )

    required_keys = {
        "X",
        "lengths",
        "y"
    }

    missing = (
        required_keys
        -
        set(data.files)
    )

    if missing:

        raise ValueError(
            f"Missing NPZ keys: {missing}"
        )

    X = data["X"].astype(
        np.float32
    )

    lengths = data["lengths"].astype(
        np.int64
    )

    y = data["y"].astype(
        np.float32
    )

    return X, lengths, y


# ============================================================
# VERIFY TEST DATA
# ============================================================

def verify_test_data(
    X,
    lengths,
    y
):

    print()
    print("=" * 80)
    print("TEST DATA VERIFICATION")
    print("=" * 80)

    print(
        f"X shape       : {X.shape}"
    )

    print(
        f"Lengths shape : {lengths.shape}"
    )

    print(
        f"Y shape       : {y.shape}"
    )

    if X.ndim != 3:

        raise ValueError(
            "Test X must be 3-dimensional."
        )

    if X.shape[2] != INPUT_SIZE:

        raise ValueError(
            f"Expected {INPUT_SIZE} features, "
            f"got {X.shape[2]}"
        )

    if len(X) != len(lengths):

        raise ValueError(
            "X and lengths are not aligned."
        )

    if len(X) != len(y):

        raise ValueError(
            "X and y are not aligned."
        )

    if np.any(
        lengths < HISTORY_MIN
    ):

        raise ValueError(
            "Found history below 3 hours."
        )

    if np.any(
        lengths > HISTORY_MAX
    ):

        raise ValueError(
            "Found history above 12 hours."
        )

    positive = int(
        np.sum(y == 1)
    )

    negative = int(
        np.sum(y == 0)
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
        f"{positive}"
    )

    print(
        f"Negative samples  : "
        f"{negative}"
    )


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    # weights_only=False is intentional because this is our
    # own trusted checkpoint and it contains metadata.

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    print(
        f"Checkpoint type: "
        f"{type(checkpoint)}"
    )

    print()
    print(
        "Checkpoint information:"
    )

    for key in [
        "input_size",
        "hidden_size",
        "num_layers",
        "dropout",
        "classifier_dropout",
        "history_min_hours",
        "history_max_hours",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "positive_weight",
        "seed",
        "device",
        "best_epoch",
        "best_validation_auprc",
        "trainable_parameters"
    ]:

        if key in checkpoint:

            print(
                f"  {key}: "
                f"{checkpoint[key]}"
            )

    required_architecture = {
        "input_size": INPUT_SIZE,
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS
    }

    for key, expected in (
        required_architecture.items()
    ):

        actual = checkpoint.get(
            key
        )

        if actual != expected:

            raise ValueError(
                f"Architecture mismatch: "
                f"{key} expected {expected}, "
                f"got {actual}"
            )

    dropout = float(
        checkpoint.get(
            "dropout",
            0.35
        )
    )

    classifier_dropout = float(
        checkpoint.get(
            "classifier_dropout",
            0.35
        )
    )

    model = AttentionBiLSTM(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=dropout,
        classifier_dropout=classifier_dropout
    ).to(
        DEVICE
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    print()
    print(
        "Model architecture: PASSED"
    )

    print(
        "Checkpoint loading : PASSED"
    )

    return model, checkpoint


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

@torch.no_grad()
def generate_predictions(
    model,
    X,
    lengths
):

    model.eval()

    probabilities = []

    batch_size = 256

    for start in range(
        0,
        len(X),
        batch_size
    ):

        end = min(
            start + batch_size,
            len(X)
        )

        X_batch = torch.tensor(
            X[start:end],
            dtype=torch.float32,
            device=DEVICE
        )

        lengths_batch = torch.tensor(
            lengths[start:end],
            dtype=torch.long,
            device=DEVICE
        )

        logits = model(
            X_batch,
            lengths_batch
        )

        probs = torch.sigmoid(
            logits
        )

        probabilities.extend(
            probs.cpu().numpy()
        )

    return np.asarray(
        probabilities,
        dtype=np.float32
    )


# ============================================================
# SAFE LOG LOSS
# ============================================================

def calculate_logloss(
    y_true,
    probabilities
):

    probabilities = np.clip(
        probabilities,
        1e-7,
        1.0 - 1e-7
    )

    return float(
        log_loss(
            y_true,
            probabilities
        )
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    probabilities,
    threshold=0.50
):

    predictions = (
        probabilities
        >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true.astype(int),
        predictions,
        labels=[0, 1]
    ).ravel()

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0
    )

    accuracy = accuracy_score(
        y_true,
        predictions
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    auroc = roc_auc_score(
        y_true,
        probabilities
    )

    auprc = average_precision_score(
        y_true,
        probabilities
    )

    brier = brier_score_loss(
        y_true,
        probabilities
    )

    logloss = calculate_logloss(
        y_true,
        probabilities
    )

    return {
        "samples": int(len(y_true)),
        "positive_samples": int(
            np.sum(y_true == 1)
        ),
        "negative_samples": int(
            np.sum(y_true == 0)
        ),
        "positive_rate": float(
            np.mean(y_true)
        ),
        "AUROC": float(auroc),
        "AUPRC": float(auprc),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1": float(f1),
        "Accuracy": float(accuracy),
        "Specificity": float(specificity),
        "Brier": float(brier),
        "LogLoss": float(logloss),
        "Threshold": float(threshold),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp)
    }


# ============================================================
# EVALUATE BY HISTORY LENGTH
# ============================================================

def evaluate_by_history(
    y_true,
    probabilities,
    lengths
):

    rows = []

    for history_hours in range(
        HISTORY_MIN,
        HISTORY_MAX + 1
    ):

        mask = (
            lengths
            == history_hours
        )

        if not np.any(mask):

            continue

        y_subset = (
            y_true[mask]
        )

        p_subset = (
            probabilities[mask]
        )

        metrics = calculate_metrics(
            y_subset,
            p_subset,
            DEFAULT_THRESHOLD
        )

        rows.append({

            "history_hours":
                history_hours,

            "samples":
                metrics["samples"],

            "positive_rate":
                metrics["positive_rate"],

            "AUROC":
                metrics["AUROC"],

            "AUPRC":
                metrics["AUPRC"],

            "Precision":
                metrics["Precision"],

            "Recall":
                metrics["Recall"],

            "F1":
                metrics["F1"],

            "Specificity":
                metrics["Specificity"],

            "Brier":
                metrics["Brier"]
        })

    return pd.DataFrame(
        rows
    )


# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

def create_threshold_analysis(
    y_true,
    probabilities
):

    thresholds = np.arange(
        0.10,
        0.91,
        0.05
    )

    rows = []

    for threshold in thresholds:

        metrics = calculate_metrics(
            y_true,
            probabilities,
            float(threshold)
        )

        rows.append({

            "threshold":
                float(threshold),

            "precision":
                metrics["Precision"],

            "recall":
                metrics["Recall"],

            "f1":
                metrics["F1"],

            "specificity":
                metrics["Specificity"],

            "accuracy":
                metrics["Accuracy"],

            "TP":
                metrics["TP"],

            "FP":
                metrics["FP"],

            "TN":
                metrics["TN"],

            "FN":
                metrics["FN"]
        })

    return pd.DataFrame(
        rows
    )


# ============================================================
# WINDOW INDEX VERIFICATION
# ============================================================

def verify_window_index(
    lengths,
    y
):

    if not WINDOW_INDEX_PATH.exists():

        print()
        print(
            "Window index not found."
        )

        print(
            "Skipping window-index verification."
        )

        return

    print()
    print(
        "Verifying window index..."
    )

    df = pd.read_csv(
        WINDOW_INDEX_PATH
    )

    # --------------------------------------------------------
    # Try to identify history column
    # --------------------------------------------------------

    history_column = None

    for column in [
        "history_hours",
        "history_length",
        "sequence_length",
        "length"
    ]:

        if column in df.columns:

            history_column = column

            break

    # --------------------------------------------------------
    # Try to identify label column
    # --------------------------------------------------------

    label_column = None

    for column in [
        "label",
        "y",
        "target",
        "sepsis_label"
    ]:

        if column in df.columns:

            label_column = column

            break

    if (
        history_column is not None
        and len(df) >= len(lengths)
    ):

        index_lengths = (
            df[history_column]
            .to_numpy()
            [-len(lengths):]
        )

        if np.array_equal(
            index_lengths.astype(int),
            lengths.astype(int)
        ):

            print(
                "History-length alignment: PASSED"
            )

        else:

            print(
                "History-length alignment: "
                "WARNING - direct comparison "
                "did not match."
            )

    else:

        print(
            "History-length alignment: "
            "SKIPPED - compatible column "
            "not found."
        )

    if (
        label_column is not None
        and len(df) >= len(y)
    ):

        index_labels = (
            df[label_column]
            .to_numpy()
            [-len(y):]
        )

        if np.array_equal(
            index_labels.astype(int),
            y.astype(int)
        ):

            print(
                "Label alignment: PASSED"
            )

        else:

            print(
                "Label alignment: "
                "WARNING - direct comparison "
                "did not match."
            )

    else:

        print(
            "Label alignment: "
            "SKIPPED - compatible column "
            "not found."
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
        "STAGE 2 - V2 ATTENTION BiLSTM"
    )

    print(
        "3-12 HOUR TEST EVALUATION"
    )

    print("=" * 80)

    print()

    print(
        f"Project root : "
        f"{PROJECT_ROOT}"
    )

    print(
        f"Device       : "
        f"{DEVICE}"
    )

    # ========================================================
    # 1. FILE CHECKS
    # ========================================================

    print()
    print(
        "[1/8] Checking required files..."
    )

    check_file(
        TEST_PATH
    )

    check_file(
        WINDOW_INDEX_PATH
    )

    check_file(
        MODEL_PATH
    )

    # ========================================================
    # 2. LOAD TEST DATA
    # ========================================================

    print()
    print(
        "[2/8] Loading untouched test data..."
    )

    X, lengths, y = (
        load_test_data()
    )

    print()
    print(
        "NPZ contents:"
    )

    print(
        f"  X       : "
        f"shape={X.shape}, "
        f"dtype={X.dtype}"
    )

    print(
        f"  lengths : "
        f"shape={lengths.shape}, "
        f"dtype={lengths.dtype}"
    )

    print(
        f"  y       : "
        f"shape={y.shape}, "
        f"dtype={y.dtype}"
    )

    # ========================================================
    # 3. VERIFY TEST DATA
    # ========================================================

    print()
    print(
        "[3/8] Verifying test data..."
    )

    verify_test_data(
        X,
        lengths,
        y
    )

    # ========================================================
    # 4. LOAD MODEL
    # ========================================================

    print()
    print(
        "[4/8] Loading V2 Attention-BiLSTM..."
    )

    model, checkpoint = (
        load_model()
    )

    # ========================================================
    # 5. PREDICTIONS
    # ========================================================

    print()
    print(
        "[5/8] Generating test predictions..."
    )

    probabilities = (
        generate_predictions(
            model,
            X,
            lengths
        )
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

    # ========================================================
    # 6. OVERALL METRICS
    # ========================================================

    print()
    print(
        "[6/8] Calculating overall metrics..."
    )

    overall = calculate_metrics(
        y,
        probabilities,
        DEFAULT_THRESHOLD
    )

    print()
    print("=" * 80)
    print(
        "V2 ATTENTION BiLSTM"
    )
    print(
        "OVERALL TEST RESULTS"
    )
    print("=" * 80)

    print(
        f"samples             : "
        f"{overall['samples']}"
    )

    print(
        f"positive_samples    : "
        f"{overall['positive_samples']}"
    )

    print(
        f"negative_samples    : "
        f"{overall['negative_samples']}"
    )

    print(
        f"positive_rate       : "
        f"{overall['positive_rate']:.6f}"
    )

    print(
        f"AUROC               : "
        f"{overall['AUROC']:.6f}"
    )

    print(
        f"AUPRC               : "
        f"{overall['AUPRC']:.6f}"
    )

    print(
        f"Precision           : "
        f"{overall['Precision']:.6f}"
    )

    print(
        f"Recall              : "
        f"{overall['Recall']:.6f}"
    )

    print(
        f"F1                  : "
        f"{overall['F1']:.6f}"
    )

    print(
        f"Accuracy            : "
        f"{overall['Accuracy']:.6f}"
    )

    print(
        f"Specificity         : "
        f"{overall['Specificity']:.6f}"
    )

    print(
        f"Brier               : "
        f"{overall['Brier']:.6f}"
    )

    print(
        f"LogLoss             : "
        f"{overall['LogLoss']:.6f}"
    )

    print(
        f"Threshold           : "
        f"{overall['Threshold']:.6f}"
    )

    print(
        f"TN                  : "
        f"{overall['TN']}"
    )

    print(
        f"FP                  : "
        f"{overall['FP']}"
    )

    print(
        f"FN                  : "
        f"{overall['FN']}"
    )

    print(
        f"TP                  : "
        f"{overall['TP']}"
    )

    # ========================================================
    # 7. BY HISTORY LENGTH
    # ========================================================

    print()
    print(
        "[7/8] Evaluating by history length..."
    )

    history_df = evaluate_by_history(
        y,
        probabilities,
        lengths
    )

    print()
    print("=" * 80)
    print(
        "RESULTS BY HISTORY LENGTH"
    )
    print("=" * 80)

    print(
        history_df.to_string(
            index=False,
            float_format=lambda value:
                f"{value:.4f}"
        )
    )

    # ========================================================
    # WINDOW INDEX
    # ========================================================

    verify_window_index(
        lengths,
        y
    )

    # ========================================================
    # THRESHOLD ANALYSIS
    # ========================================================

    print()
    print(
        "Creating threshold analysis..."
    )

    threshold_df = (
        create_threshold_analysis(
            y,
            probabilities
        )
    )

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    confusion_df = pd.DataFrame({

        "metric": [
            "TN",
            "FP",
            "FN",
            "TP"
        ],

        "value": [
            overall["TN"],
            overall["FP"],
            overall["FN"],
            overall["TP"]
        ]
    })

    # ========================================================
    # PREDICTION TABLE
    # ========================================================

    prediction_df = pd.DataFrame({

        "sample_index":
            np.arange(
                len(y)
            ),

        "history_hours":
            lengths,

        "y_true":
            y.astype(int),

        "probability":
            probabilities,

        "prediction":
            (
                probabilities
                >= DEFAULT_THRESHOLD
            ).astype(int)
    })

    # ========================================================
    # SAVE PREDICTIONS
    # ========================================================

    np.savez(
        PREDICTIONS_PATH,
        y_true=y.astype(
            np.float32
        ),
        y_probability=probabilities.astype(
            np.float32
        ),
        history_hours=lengths.astype(
            np.int64
        )
    )

    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    summary = {

        "model_name":
            "AttentionBiLSTM_V2",

        "model_path":
            str(MODEL_PATH),

        "test_path":
            str(TEST_PATH),

        "samples":
            overall["samples"],

        "positive_samples":
            overall["positive_samples"],

        "negative_samples":
            overall["negative_samples"],

        "positive_rate":
            overall["positive_rate"],

        "AUROC":
            overall["AUROC"],

        "AUPRC":
            overall["AUPRC"],

        "Precision":
            overall["Precision"],

        "Recall":
            overall["Recall"],

        "F1":
            overall["F1"],

        "Accuracy":
            overall["Accuracy"],

        "Specificity":
            overall["Specificity"],

        "Brier":
            overall["Brier"],

        "LogLoss":
            overall["LogLoss"],

        "threshold":
            DEFAULT_THRESHOLD,

        "TN":
            overall["TN"],

        "FP":
            overall["FP"],

        "FN":
            overall["FN"],

        "TP":
            overall["TP"],

        "best_epoch":
            checkpoint.get(
                "best_epoch"
            ),

        "best_validation_auprc":
            checkpoint.get(
                "best_validation_auprc"
            ),

        "test_data_used_for_training":
            False,

        "calibration_performed":
            False,

        "xai_performed":
            False,

        "fastapi_integration":
            False
    }

    with open(
        SUMMARY_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=2
        )

    history_df.to_csv(
        HISTORY_RESULTS_PATH,
        index=False
    )

    threshold_df.to_csv(
        THRESHOLD_PATH,
        index=False
    )

    confusion_df.to_csv(
        CONFUSION_PATH,
        index=False
    )

    prediction_df.to_csv(
        PREDICTION_TABLE_PATH,
        index=False
    )

    # ========================================================
    # FINISHED
    # ========================================================

    print()
    print("=" * 80)
    print(
        "STAGE 2 V2 EVALUATION COMPLETED"
    )
    print("=" * 80)

    print()

    print(
        "Saved files:"
    )

    print(
        PREDICTIONS_PATH
    )

    print(
        SUMMARY_PATH
    )

    print(
        HISTORY_RESULTS_PATH
    )

    print(
        THRESHOLD_PATH
    )

    print(
        CONFUSION_PATH
    )

    print(
        PREDICTION_TABLE_PATH
    )

    print()

    print(
        "No training was performed."
    )

    print(
        "No calibration was performed."
    )

    print(
        "No XAI was performed."
    )

    print(
        "No FastAPI/frontend integration was performed."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()