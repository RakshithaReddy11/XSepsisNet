from pathlib import Path
import json
import random

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    raise SystemExit(
        "\nPyTorch is not installed.\n"
        "Install it using:\n"
        "pip install torch\n"
    )


# ============================================================
# XSepsisNet - BiLSTM TRAINING
# ============================================================
#
# Input:
#   data/processed/temporal_bilstm_hourly.csv
#   data/processed/train_window_index.csv
#   data/processed/validation_window_index.csv
#
# History:
#   6 hours
#
# Features:
#   39 clinical features
#   +
#   39 missingness indicators
#   =
#   78 input features
#
# Model:
#   2-layer Bidirectional LSTM
#
# Target:
#   Current temporal proxy label
#
# IMPORTANT:
# The current target is a temporal proxy based on:
#   sepsis-candidate ICU stay
#   +
#   first microbiology temporal evidence
#
# First microbiology is NOT exact sepsis onset.
# Therefore this model must not be described as a
# confirmed Sepsis-3 onset predictor.
# ============================================================


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed"

MODEL_DIR = PROJECT_ROOT / "models" / "bilstm"

SCALER_DIR = PROJECT_ROOT / "models" / "scalers"

HOURLY_FILE = (
    DATA_DIR /
    "temporal_bilstm_hourly.csv"
)

TRAIN_INDEX_FILE = (
    DATA_DIR /
    "train_window_index.csv"
)

VALIDATION_INDEX_FILE = (
    DATA_DIR /
    "validation_window_index.csv"
)

MODEL_FILE = (
    MODEL_DIR /
    "bilstm_best.pt"
)

PREPROCESS_FILE = (
    SCALER_DIR /
    "bilstm_preprocessing.json"
)


# ============================================================
# CONFIGURATION
# ============================================================

HISTORY_HOURS = 6

BATCH_SIZE = 256

HIDDEN_SIZE = 64

NUM_LAYERS = 2

DROPOUT = 0.30

LEARNING_RATE = 0.001

WEIGHT_DECAY = 1e-5

EPOCHS = 10

# Number of negative samples per positive sample.
NEGATIVE_TO_POSITIVE = 5

RANDOM_SEED = 42

NUM_WORKERS = 0


# ============================================================
# CLINICAL FEATURES
# ============================================================

FEATURES = [

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


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


# ============================================================
# FILE CHECK
# ============================================================

def require_file(path):

    if not path.exists():

        raise FileNotFoundError(
            f"\nRequired file not found:\n{path}"
        )


# ============================================================
# LOAD HOURLY DATA
# ============================================================

def load_hourly():

    print(
        "\n" +
        "=" * 70
    )

    print(
        "STEP 1: LOAD HOURLY CLINICAL DATA"
    )

    print(
        "=" * 70
    )

    require_file(
        HOURLY_FILE
    )

    required_columns = [
        "stay_id",
        "icu_hour"
    ] + FEATURES

    df = pd.read_csv(
        HOURLY_FILE,
        usecols=required_columns,
        low_memory=False
    )

    df["stay_id"] = pd.to_numeric(
        df["stay_id"],
        errors="coerce"
    )

    df["icu_hour"] = pd.to_numeric(
        df["icu_hour"],
        errors="coerce"
    )

    df = df.dropna(
        subset=[
            "stay_id",
            "icu_hour"
        ]
    ).copy()

    df["stay_id"] = (
        df["stay_id"]
        .astype(np.int64)
    )

    df["icu_hour"] = (
        df["icu_hour"]
        .astype(np.int64)
    )

    df = df.sort_values(
        [
            "stay_id",
            "icu_hour"
        ]
    ).reset_index(
        drop=True
    )

    duplicates = df.duplicated(
        subset=[
            "stay_id",
            "icu_hour"
        ]
    ).sum()

    if duplicates > 0:

        raise RuntimeError(
            f"Duplicate stay-hour rows found: "
            f"{duplicates:,}"
        )

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"ICU stays: "
        f"{df['stay_id'].nunique():,}"
    )

    print(
        f"Clinical features: "
        f"{len(FEATURES)}"
    )

    return df


# ============================================================
# LOAD WINDOW INDEX
# ============================================================

def load_window_indices():

    print(
        "\n" +
        "=" * 70
    )

    print(
        "STEP 2: LOAD TRAIN / VALIDATION WINDOWS"
    )

    print(
        "=" * 70
    )

    require_file(
        TRAIN_INDEX_FILE
    )

    require_file(
        VALIDATION_INDEX_FILE
    )

    train = pd.read_csv(
        TRAIN_INDEX_FILE,
        low_memory=False
    )

    validation = pd.read_csv(
        VALIDATION_INDEX_FILE,
        low_memory=False
    )

    required = [
        "stay_id",
        "prediction_hour",
        "history_start_hour",
        "history_end_hour",
        "label"
    ]

    for column in required:

        if column not in train.columns:

            raise ValueError(
                f"Missing train column: {column}"
            )

        if column not in validation.columns:

            raise ValueError(
                f"Missing validation column: {column}"
            )

    print(
        f"Train windows: "
        f"{len(train):,}"
    )

    print(
        f"Validation windows: "
        f"{len(validation):,}"
    )

    print(
        f"Train positive: "
        f"{int(train['label'].sum()):,}"
    )

    print(
        f"Validation positive: "
        f"{int(validation['label'].sum()):,}"
    )

    return train, validation


# ============================================================
# TRAINING-ONLY PREPROCESSING
# ============================================================

def fit_preprocessing(
    hourly,
    train_index
):

    print(
        "\n" +
        "=" * 70
    )

    print(
        "STEP 3: FIT TRAINING-ONLY PREPROCESSING"
    )

    print(
        "=" * 70
    )

    train_stays = set(
        train_index[
            "stay_id"
        ].unique()
    )

    train_rows = hourly[
        hourly[
            "stay_id"
        ].isin(
            train_stays
        )
    ].copy()

    print(
        f"Training stays: "
        f"{len(train_stays):,}"
    )

    print(
        f"Training hourly rows: "
        f"{len(train_rows):,}"
    )

    medians = {}

    means = {}

    stds = {}

    for feature in FEATURES:

        values = pd.to_numeric(
            train_rows[feature],
            errors="coerce"
        )

        median = values.median()

        if pd.isna(median):

            median = 0.0

        filled = values.fillna(
            median
        )

        mean = filled.mean()

        std = filled.std()

        if (
            pd.isna(std)
            or
            std < 1e-8
        ):

            std = 1.0

        medians[feature] = float(
            median
        )

        means[feature] = float(
            mean
        )

        stds[feature] = float(
            std
        )

    print(
        "Training-only median/mean/std calculated."
    )

    return (
        medians,
        means,
        stds
    )


# ============================================================
# PREPARE NUMPY FEATURE MATRIX
# ============================================================

def prepare_matrix(
    hourly,
    medians,
    means,
    stds
):

    print(
        "\n" +
        "=" * 70
    )

    print(
        "STEP 4: CREATE FEATURE + MISSINGNESS MATRIX"
    )

    print(
        "=" * 70
    )

    n_rows = len(hourly)

    n_features = len(FEATURES)

    values = np.zeros(
        (
            n_rows,
            n_features
        ),
        dtype=np.float32
    )

    missing = np.zeros(
        (
            n_rows,
            n_features
        ),
        dtype=np.float32
    )

    for j, feature in enumerate(
        FEATURES
    ):

        column = pd.to_numeric(
            hourly[feature],
            errors="coerce"
        ).to_numpy(
            dtype=np.float32
        )

        missing[:, j] = (
            np.isnan(column)
            .astype(np.float32)
        )

        column = np.nan_to_num(
            column,
            nan=medians[feature]
        )

        column = (
            column
            -
            means[feature]
        ) / stds[feature]

        values[:, j] = column

    matrix = np.concatenate(
        [
            values,
            missing
        ],
        axis=1
    )

    print(
        f"Clinical features: "
        f"{n_features}"
    )

    print(
        f"Missingness features: "
        f"{n_features}"
    )

    print(
        f"Final model input size: "
        f"{matrix.shape[1]}"
    )

    print(
        f"Matrix shape: "
        f"{matrix.shape}"
    )

    print(
        f"Matrix memory: "
        f"{matrix.nbytes / (1024 ** 3):.2f} GB"
    )

    return matrix


# ============================================================
# ATTACH ROW POSITIONS
# ============================================================

def attach_row_positions(
    hourly,
    index
):

    print(
        "\nMapping windows to hourly rows..."
    )

    lookup = hourly[
        [
            "stay_id",
            "icu_hour"
        ]
    ].copy()

    lookup["row_position"] = np.arange(
        len(lookup),
        dtype=np.int64
    )

    lookup = lookup.rename(
        columns={
            "icu_hour":
                "prediction_hour"
        }
    )

    result = index.merge(
        lookup,
        on=[
            "stay_id",
            "prediction_hour"
        ],
        how="inner",
        validate="one_to_one"
    )

    result["row_start"] = (
        result["row_position"]
        -
        HISTORY_HOURS
        +
        1
    )

    result["row_end"] = (
        result["row_position"]
    )

    result = result[
        result["row_start"] >= 0
    ].copy()

    # --------------------------------------------------------
    # Verify both endpoints belong to same ICU stay.
    # --------------------------------------------------------

    stay_array = (
        hourly[
            "stay_id"
        ].to_numpy()
    )

    starts = (
        result[
            "row_start"
        ].to_numpy(
            dtype=np.int64
        )
    )

    ends = (
        result[
            "row_end"
        ].to_numpy(
            dtype=np.int64
        )
    )

    valid = (
        stay_array[starts]
        ==
        stay_array[ends]
    )

    result = result[
        valid
    ].copy()

    print(
        f"Windows mapped: "
        f"{len(result):,}"
    )

    return result


# ============================================================
# BALANCE TRAINING WINDOWS
# ============================================================

def select_training_windows(
    train_index
):

    print(
        "\n" +
        "=" * 70
    )

    print(
        "STEP 5: BALANCE TRAINING WINDOWS"
    )

    print(
        "=" * 70
    )

    positives = train_index[
        train_index[
            "label"
        ] == 1
    ].copy()

    negatives = train_index[
        train_index[
            "label"
        ] == 0
    ].copy()

    print(
        f"Original positives: "
        f"{len(positives):,}"
    )

    print(
        f"Original negatives: "
        f"{len(negatives):,}"
    )

    target_negative_count = min(
        len(negatives),
        len(positives)
        *
        NEGATIVE_TO_POSITIVE
    )

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    if (
        target_negative_count
        <
        len(negatives)
    ):

        selected = rng.choice(
            len(negatives),
            size=target_negative_count,
            replace=False
        )

        negatives = negatives.iloc[
            selected
        ].copy()

    selected_train = pd.concat(
        [
            positives,
            negatives
        ],
        ignore_index=True
    )

    selected_train = (
        selected_train
        .sample(
            frac=1.0,
            random_state=RANDOM_SEED
        )
        .reset_index(
            drop=True
        )
    )

    print(
        f"Selected positives: "
        f"{len(positives):,}"
    )

    print(
        f"Selected negatives: "
        f"{len(negatives):,}"
    )

    print(
        f"Selected total: "
        f"{len(selected_train):,}"
    )

    return selected_train


# ============================================================
# PYTORCH DATASET
# ============================================================

class TemporalDataset(
    Dataset
):

    def __init__(
        self,
        matrix,
        index
    ):

        self.matrix = matrix

        self.row_start = (
            index[
                "row_start"
            ]
            .to_numpy(
                dtype=np.int64
            )
        )

        self.row_end = (
            index[
                "row_end"
            ]
            .to_numpy(
                dtype=np.int64
            )
        )

        self.labels = (
            index[
                "label"
            ]
            .to_numpy(
                dtype=np.float32
            )
        )

    def __len__(self):

        return len(
            self.labels
        )

    def __getitem__(
        self,
        idx
    ):

        start = self.row_start[
            idx
        ]

        end = self.row_end[
            idx
        ]

        sequence = self.matrix[
            start:
            end + 1
        ]

        label = self.labels[
            idx
        ]

        return (
            torch.from_numpy(
                sequence
            ),
            torch.tensor(
                label,
                dtype=torch.float32
            )
        )


# ============================================================
# BiLSTM MODEL
# ============================================================

class BiLSTMModel(
    nn.Module
):

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        dropout
    ):

        super().__init__()

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

        self.dropout = nn.Dropout(
            dropout
        )

        self.classifier = nn.Linear(
            hidden_size * 2,
            1
        )

    def forward(
        self,
        x
    ):

        output, _ = self.lstm(
            x
        )

        # Last time step
        last_output = output[
            :, -1, :
        ]

        last_output = self.dropout(
            last_output
        )

        logits = self.classifier(
            last_output
        ).squeeze(
            -1
        )

        return logits


# ============================================================
# VALIDATION
# ============================================================

def evaluate_loss(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    total_loss = 0.0

    total_samples = 0

    with torch.no_grad():

        for x, y in loader:

            x = x.to(
                device
            )

            y = y.to(
                device
            )

            logits = model(
                x
            )

            loss = criterion(
                logits,
                y
            )

            batch_size = (
                y.size(0)
            )

            total_loss += (
                loss.item()
                *
                batch_size
            )

            total_samples += (
                batch_size
            )

    return (
        total_loss
        /
        max(
            total_samples,
            1
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "XSepsisNet - BiLSTM TRAINING"
    )

    print(
        "=" * 70
    )

    set_seed(
        RANDOM_SEED
    )

    # --------------------------------------------------------
    # Create output directories
    # --------------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    SCALER_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    if torch.cuda.is_available():

        device = torch.device(
            "cuda"
        )

    else:

        device = torch.device(
            "cpu"
        )

    print(
        f"\nDevice: {device}"
    )

    if device.type == "cuda":

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    hourly = load_hourly()

    train_index, validation_index = (
        load_window_indices()
    )

    # --------------------------------------------------------
    # Training-only preprocessing
    # --------------------------------------------------------

    (
        medians,
        means,
        stds
    ) = fit_preprocessing(
        hourly,
        train_index
    )

    # --------------------------------------------------------
    # Feature matrix
    # --------------------------------------------------------

    matrix = prepare_matrix(
        hourly,
        medians,
        means,
        stds
    )

    # --------------------------------------------------------
    # Map windows to rows
    # --------------------------------------------------------

    train_index = attach_row_positions(
        hourly,
        train_index
    )

    validation_index = attach_row_positions(
        hourly,
        validation_index
    )

    # --------------------------------------------------------
    # Balance TRAIN only
    # --------------------------------------------------------

    selected_train = (
        select_training_windows(
            train_index
        )
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    train_dataset = TemporalDataset(
        matrix,
        selected_train
    )

    validation_dataset = TemporalDataset(
        matrix,
        validation_index
    )

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(
            device.type == "cuda"
        )
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(
            device.type == "cuda"
        )
    )

    print(
        f"\nTraining batches: "
        f"{len(train_loader):,}"
    )

    print(
        f"Validation batches: "
        f"{len(validation_loader):,}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    input_size = (
        len(FEATURES)
        *
        2
    )

    model = BiLSTMModel(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    )

    model = model.to(
        device
    )

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"\nModel input size: "
        f"{input_size}"
    )

    print(
        f"Hidden size: "
        f"{HIDDEN_SIZE}"
    )

    print(
        f"LSTM layers: "
        f"{NUM_LAYERS}"
    )

    print(
        f"Trainable parameters: "
        f"{parameter_count:,}"
    )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------
    #
    # Training windows are already sampled at 1:5.
    # We use normal BCE here to avoid excessively weighting
    # the positive class twice.
    # --------------------------------------------------------

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    best_validation_loss = float(
        "inf"
    )

    print(
        "\n" +
        "=" * 70
    )

    print(
        "STEP 6: TRAIN BiLSTM"
    )

    print(
        "=" * 70
    )

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        model.train()

        running_loss = 0.0

        sample_count = 0

        for batch_number, (
            x,
            y
        ) in enumerate(
            train_loader,
            start=1
        ):

            x = x.to(
                device,
                non_blocking=True
            )

            y = y.to(
                device,
                non_blocking=True
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(
                x
            )

            loss = criterion(
                logits,
                y
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            batch_size = (
                y.size(0)
            )

            running_loss += (
                loss.item()
                *
                batch_size
            )

            sample_count += (
                batch_size
            )

        train_loss = (
            running_loss
            /
            max(
                sample_count,
                1
            )
        )

        validation_loss = evaluate_loss(
            model,
            validation_loader,
            criterion,
            device
        )

        print(
            f"Epoch "
            f"{epoch:02d}/{EPOCHS} | "
            f"Train Loss: "
            f"{train_loss:.6f} | "
            f"Validation Loss: "
            f"{validation_loss:.6f}"
        )

        # ----------------------------------------------------
        # Save best validation model
        # ----------------------------------------------------

        if (
            validation_loss
            <
            best_validation_loss
        ):

            best_validation_loss = (
                validation_loss
            )

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "input_size":
                        input_size,

                    "hidden_size":
                        HIDDEN_SIZE,

                    "num_layers":
                        NUM_LAYERS,

                    "dropout":
                        DROPOUT,

                    "history_hours":
                        HISTORY_HOURS,

                    "features":
                        FEATURES,

                    "best_validation_loss":
                        best_validation_loss,

                    "label_type":
                        "temporal_proxy"
                },
                MODEL_FILE
            )

            print(
                "  BEST MODEL SAVED"
            )

    # --------------------------------------------------------
    # Save preprocessing
    # --------------------------------------------------------

    preprocessing = {

        "features":
            FEATURES,

        "medians":
            medians,

        "means":
            means,

        "stds":
            stds,

        "history_hours":
            HISTORY_HOURS,

        "input_size":
            input_size,

        "missingness_features":
            len(FEATURES),

        "label_type":
            "temporal_proxy",

        "note":
            (
                "Preprocessing statistics were calculated "
                "using training stays only."
            )
    }

    with open(
        PREPROCESS_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            preprocessing,
            file,
            indent=2
        )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print(
        "\n" +
        "=" * 70
    )

    print(
        "SUCCESS - BiLSTM TRAINING COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"\nBest model:"
    )

    print(
        MODEL_FILE
    )

    print(
        f"\nPreprocessing:"
    )

    print(
        PREPROCESS_FILE
    )

    print(
        f"\nBest validation loss:"
        f" {best_validation_loss:.6f}"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()