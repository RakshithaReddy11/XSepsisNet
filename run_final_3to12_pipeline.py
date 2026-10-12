from pathlib import Path
import json
import random
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")


# ============================================================
# XSEPSISNET
# STAGE 1 ONLY
#
# Existing processed data
#        ↓
# 3–12 hour preprocessing
#        ↓
# Patient-level split
#        ↓
# Train-only scaler
#        ↓
# Variable-length BiLSTM
#        ↓
# Training
#        ↓
# Best model checkpoint
#
# NO EVALUATION
# NO CALIBRATION
# NO XAI
# NO FASTAPI
# NO FRONTEND
# ============================================================


# ------------------------------------------------------------
# PROJECT PATHS
# ------------------------------------------------------------

ROOT = Path(__file__).resolve().parent

DATA_PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
BILSTM_DIR = MODELS / "bilstm"
SCALER_DIR = MODELS / "scalers"

DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
BILSTM_DIR.mkdir(parents=True, exist_ok=True)
SCALER_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------

SEED = 42

MIN_HISTORY_HOURS = 3
MAX_HISTORY_HOURS = 12

HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.40

BATCH_SIZE = 256
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4

MAX_EPOCHS = 20
EARLY_STOPPING_PATIENCE = 5
GRADIENT_CLIP = 1.0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ------------------------------------------------------------
# REPRODUCIBILITY
# ------------------------------------------------------------

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ------------------------------------------------------------
# PRINT HEADER
# ------------------------------------------------------------

print()
print("=" * 80)
print("XSepsisNet - FINAL STAGE 1")
print("3-12 HOUR PREPROCESSING + BiLSTM TRAINING")
print("=" * 80)
print()

print("Project root :", ROOT)
print("Device       :", DEVICE)
print()


# ============================================================
# 1. LOAD EXISTING PROCESSED DATA
# ============================================================

print("[1/8] Loading existing processed clinical data...")

hourly_path = DATA_PROCESSED / "icu_hourly_combined.csv"

labels_with_patient_path = (
    DATA_PROCESSED /
    "temporal_sepsis_labels_with_patient.csv"
)

labels_basic_path = (
    DATA_PROCESSED /
    "temporal_sepsis_labels.csv"
)


if not hourly_path.exists():
    raise FileNotFoundError(
        "\nMissing required file:\n"
        f"{hourly_path}\n\n"
        "The previously completed MIMIC preprocessing "
        "must be present."
    )


if labels_with_patient_path.exists():
    labels_path = labels_with_patient_path
elif labels_basic_path.exists():
    labels_path = labels_basic_path
else:
    raise FileNotFoundError(
        "\nCould not find temporal sepsis labels.\n"
        "Expected one of:\n"
        f"{labels_with_patient_path}\n"
        f"{labels_basic_path}"
    )


print("Hourly data :", hourly_path)
print("Labels      :", labels_path)
print()


hourly = pd.read_csv(hourly_path)
labels = pd.read_csv(labels_path)


print("Hourly dataset shape :", hourly.shape)
print("Labels shape         :", labels.shape)
print()


# ============================================================
# 2. DETECT REQUIRED COLUMNS
# ============================================================

print("[2/8] Detecting required columns...")


def detect_column(df, candidates, required=True):

    exact = {
        str(c).lower(): c
        for c in df.columns
    }

    for candidate in candidates:

        key = candidate.lower()

        if key in exact:
            return exact[key]

    for candidate in candidates:

        candidate = candidate.lower()

        for actual_lower, actual in exact.items():

            if candidate in actual_lower:
                return actual

    if required:

        raise ValueError(
            "\nCould not find required column.\n"
            f"Tried: {candidates}\n\n"
            f"Available columns:\n{list(df.columns)}"
        )

    return None


stay_col = detect_column(
    hourly,
    [
        "stay_id",
        "icustay_id"
    ]
)


hour_col = detect_column(
    hourly,
    [
        "hour",
        "hour_index",
        "relative_hour",
        "icu_hour",
        "chart_hour"
    ]
)


label_stay_col = detect_column(
    labels,
    [
        "stay_id",
        "icustay_id"
    ]
)


label_hour_col = detect_column(
    labels,
    [
        "prediction_hour",
        "hour",
        "relative_hour",
        "icu_hour"
    ]
)


label_col = detect_column(
    labels,
    [
        "label",
        "sepsis_label",
        "temporal_sepsis_label",
        "target"
    ]
)


subject_col = detect_column(
    labels,
    [
        "subject_id"
    ],
    required=False
)


print("Stay column       :", stay_col)
print("Hour column       :", hour_col)
print("Label stay column :", label_stay_col)
print("Label hour column :", label_hour_col)
print("Label column      :", label_col)
print("Subject column    :", subject_col)
print()


# ============================================================
# 3. CLEAN / AUDIT EXISTING HOURLY DATA
# ============================================================

print("[3/8] Preparing existing hourly clinical data...")


hourly[stay_col] = pd.to_numeric(
    hourly[stay_col],
    errors="coerce"
)

hourly[hour_col] = pd.to_numeric(
    hourly[hour_col],
    errors="coerce"
)

labels[label_stay_col] = pd.to_numeric(
    labels[label_stay_col],
    errors="coerce"
)

labels[label_hour_col] = pd.to_numeric(
    labels[label_hour_col],
    errors="coerce"
)

labels[label_col] = pd.to_numeric(
    labels[label_col],
    errors="coerce"
)


hourly = hourly.dropna(
    subset=[
        stay_col,
        hour_col
    ]
).copy()


labels = labels.dropna(
    subset=[
        label_stay_col,
        label_hour_col,
        label_col
    ]
).copy()


hourly[stay_col] = hourly[stay_col].astype(np.int64)
hourly[hour_col] = hourly[hour_col].astype(np.int64)

labels[label_stay_col] = labels[
    label_stay_col
].astype(np.int64)

labels[label_hour_col] = labels[
    label_hour_col
].astype(np.int64)

labels[label_col] = labels[
    label_col
].astype(np.int64)


labels = labels[
    labels[label_col].isin([0, 1])
].copy()


# ------------------------------------------------------------
# Identify clinical numeric features
# ------------------------------------------------------------

excluded_columns = {
    stay_col,
    hour_col
}


clinical_features = []


for column in hourly.columns:

    if column in excluded_columns:
        continue

    if pd.api.types.is_numeric_dtype(
        hourly[column]
    ):
        clinical_features.append(column)


if len(clinical_features) == 0:

    raise RuntimeError(
        "No numeric clinical features were detected."
    )


# Convert clinical columns to numeric
for column in clinical_features:

    hourly[column] = pd.to_numeric(
        hourly[column],
        errors="coerce"
    )


# ------------------------------------------------------------
# Remove duplicate stay-hour rows if any
# ------------------------------------------------------------

duplicate_count = hourly.duplicated(
    subset=[
        stay_col,
        hour_col
    ]
).sum()


if duplicate_count > 0:

    print(
        "Duplicate stay-hour rows found:",
        duplicate_count
    )

    hourly = (
        hourly
        .sort_values(
            [
                stay_col,
                hour_col
            ]
        )
        .drop_duplicates(
            subset=[
                stay_col,
                hour_col
            ],
            keep="first"
        )
    )

else:

    print(
        "Duplicate stay-hour rows: 0"
    )


hourly = hourly.sort_values(
    [
        stay_col,
        hour_col
    ]
).reset_index(drop=True)


print()
print("Clinical feature count :", len(clinical_features))
print(
    "Unique ICU stays       :",
    hourly[stay_col].nunique()
)
print(
    "Unique prediction stays:",
    labels[label_stay_col].nunique()
)
print(
    "Label rows             :",
    len(labels)
)
print(
    "Positive labels        :",
    int(labels[label_col].sum())
)
print(
    "Negative labels        :",
    int((labels[label_col] == 0).sum())
)
print(
    "Positive rate          :",
    f"{labels[label_col].mean() * 100:.2f}%"
)
print()


# ============================================================
# 4. BUILD 3–12 HOUR WINDOWS
# ============================================================

print("[4/8] Building final 3–12 hour windows...")
print(
    "This uses the EXISTING processed hourly data."
)
print(
    "Raw MIMIC preprocessing is NOT being repeated."
)
print()


# ------------------------------------------------------------
# Create hourly lookup
# ------------------------------------------------------------

hour_lookup = {}


hourly_subset = hourly[
    [
        stay_col,
        hour_col
    ] + clinical_features
]


for row in hourly_subset.itertuples(
    index=False,
    name=None
):

    stay = int(row[0])
    hour = int(row[1])

    values = np.asarray(
        row[2:],
        dtype=np.float32
    )

    hour_lookup[
        (stay, hour)
    ] = values


print(
    "Hourly lookup entries:",
    len(hour_lookup)
)


# ------------------------------------------------------------
# Remove duplicate label points
# ------------------------------------------------------------

labels = (
    labels
    .sort_values(
        [
            label_stay_col,
            label_hour_col
        ]
    )
    .drop_duplicates(
        subset=[
            label_stay_col,
            label_hour_col
        ],
        keep="last"
    )
)


# ------------------------------------------------------------
# Label lookup
# ------------------------------------------------------------

label_lookup = {}

for _, row in labels.iterrows():

    stay = int(
        row[label_stay_col]
    )

    prediction_hour = int(
        row[label_hour_col]
    )

    target = int(
        row[label_col]
    )

    label_lookup[
        (stay, prediction_hour)
    ] = target


# ------------------------------------------------------------
# Patient lookup
# ------------------------------------------------------------

patient_lookup = {}


if subject_col is not None:

    for _, row in labels.iterrows():

        stay = int(
            row[label_stay_col]
        )

        prediction_hour = int(
            row[label_hour_col]
        )

        patient = int(
            row[subject_col]
        )

        patient_lookup[
            (stay, prediction_hour)
        ] = patient


# ------------------------------------------------------------
# Build windows
# ------------------------------------------------------------

records = []


history_counts = {
    h: 0
    for h in range(
        MIN_HISTORY_HOURS,
        MAX_HISTORY_HOURS + 1
    )
}


positive_counts = {
    h: 0
    for h in range(
        MIN_HISTORY_HOURS,
        MAX_HISTORY_HOURS + 1
    )
}


for (
    stay,
    prediction_hour
), target in label_lookup.items():

    for history_length in range(
        MIN_HISTORY_HOURS,
        MAX_HISTORY_HOURS + 1
    ):

        start_hour = (
            prediction_hour
            - history_length
            + 1
        )

        required_hours = range(
            start_hour,
            prediction_hour + 1
        )


        # Every hour must exist.
        complete_history = all(
            (
                stay,
                h
            ) in hour_lookup
            for h in required_hours
        )


        if not complete_history:
            continue


        if (
            stay,
            prediction_hour
        ) not in hour_lookup:
            continue


        patient = patient_lookup.get(
            (
                stay,
                prediction_hour
            ),
            -1
        )


        records.append(
            {
                "stay_id": stay,
                "prediction_hour": prediction_hour,
                "history_hours": history_length,
                "start_hour": start_hour,
                "end_hour": prediction_hour,
                "label": target,
                "subject_id": patient
            }
        )


        history_counts[
            history_length
        ] += 1


        if target == 1:

            positive_counts[
                history_length
            ] += 1


window_index = pd.DataFrame(
    records
)


if len(window_index) == 0:

    raise RuntimeError(
        "No valid 3–12 hour windows were generated."
    )


# ------------------------------------------------------------
# Group ID
# ------------------------------------------------------------

if (
    subject_col is not None
    and
    (window_index["subject_id"] >= 0).any()
):

    window_index["group_id"] = (
        window_index["subject_id"]
    )

else:

    print(
        "WARNING: subject_id unavailable."
    )

    print(
        "Using stay_id for leakage control."
    )

    window_index["group_id"] = (
        window_index["stay_id"]
    )


window_index = window_index.sort_values(
    [
        "group_id",
        "stay_id",
        "prediction_hour",
        "history_hours"
    ]
).reset_index(drop=True)


window_index_path = (
    DATA_PROCESSED /
    "final_3to12_window_index.csv"
)


window_index.to_csv(
    window_index_path,
    index=False
)


print()
print(
    "Total 3–12h windows:",
    len(window_index)
)


for history_length in range(
    MIN_HISTORY_HOURS,
    MAX_HISTORY_HOURS + 1
):

    print(
        f"{history_length:2d}h : "
        f"{history_counts[history_length]:,} windows | "
        f"{positive_counts[history_length]:,} positive"
    )


print()


# ============================================================
# 5. PATIENT-LEVEL TRAIN / VALIDATION / TEST SPLIT
# ============================================================

print("[5/8] Creating patient-level split...")


groups = (
    window_index["group_id"]
    .drop_duplicates()
    .to_numpy()
)


rng = np.random.default_rng(
    SEED
)

rng.shuffle(groups)


number_of_groups = len(groups)


train_count = int(
    number_of_groups * 0.70
)

validation_count = int(
    number_of_groups * 0.15
)


train_groups = set(
    groups[
        :train_count
    ]
)


validation_groups = set(
    groups[
        train_count:
        train_count + validation_count
    ]
)


test_groups = set(
    groups[
        train_count + validation_count:
    ]
)


window_index["split"] = "test"


window_index.loc[
    window_index["group_id"].isin(
        train_groups
    ),
    "split"
] = "train"


window_index.loc[
    window_index["group_id"].isin(
        validation_groups
    ),
    "split"
] = "validation"


# ------------------------------------------------------------
# Leakage checks
# ------------------------------------------------------------

train_set = set(
    window_index.loc[
        window_index["split"] == "train",
        "group_id"
    ]
)

validation_set = set(
    window_index.loc[
        window_index["split"] == "validation",
        "group_id"
    ]
)

test_set = set(
    window_index.loc[
        window_index["split"] == "test",
        "group_id"
    ]
)


if train_set & validation_set:

    raise RuntimeError(
        "DATA LEAKAGE: train and validation overlap."
    )


if train_set & test_set:

    raise RuntimeError(
        "DATA LEAKAGE: train and test overlap."
    )


if validation_set & test_set:

    raise RuntimeError(
        "DATA LEAKAGE: validation and test overlap."
    )


# Save split
window_index.to_csv(
    window_index_path,
    index=False
)


print(
    "Train groups      :",
    len(train_set)
)

print(
    "Validation groups :",
    len(validation_set)
)

print(
    "Test groups       :",
    len(test_set)
)

print()


for split_name in [
    "train",
    "validation",
    "test"
]:

    subset = window_index[
        window_index["split"] == split_name
    ]

    print(
        f"{split_name:12s} | "
        f"windows={len(subset):,} | "
        f"positive={int(subset['label'].sum()):,} | "
        f"rate={subset['label'].mean() * 100:.2f}%"
    )


print()
print(
    "Patient-level leakage check: PASSED"
)
print()


# ============================================================
# 6. TRAIN-ONLY SCALER
# ============================================================

print("[6/8] Fitting scaler using TRAIN data ONLY...")


train_windows = window_index[
    window_index["split"] == "train"
]


train_hour_values = []


for row in train_windows.itertuples(
    index=False
):

    stay = int(row.stay_id)
    start_hour = int(row.start_hour)
    end_hour = int(row.end_hour)


    for hour in range(
        start_hour,
        end_hour + 1
    ):

        values = hour_lookup[
            (
                stay,
                hour
            )
        ]

        train_hour_values.append(
            values
        )


train_hour_values = np.vstack(
    train_hour_values
)


print(
    "Training hourly observations used for scaler:",
    train_hour_values.shape
)


scaler = StandardScaler()


scaler.fit(
    train_hour_values
)


scaler_path = (
    SCALER_DIR /
    "final_3to12_bilstm_scaler.json"
)


with open(
    scaler_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        {
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
            "features": clinical_features,
            "fit_on": "TRAIN ONLY",
            "history_range": "3-12 hours"
        },
        f,
        indent=2
    )


print(
    "TRAIN-only scaler saved:"
)
print(
    scaler_path
)
print()


# ============================================================
# 7. CREATE FINAL BiLSTM ARRAYS
# ============================================================

print("[7/8] Creating 3–12 hour BiLSTM training arrays...")


MAX_SEQUENCE_LENGTH = MAX_HISTORY_HOURS

CLINICAL_FEATURE_COUNT = len(
    clinical_features
)

MODEL_INPUT_SIZE = (
    CLINICAL_FEATURE_COUNT * 2
)


def create_arrays(
    split_name
):

    subset = window_index[
        window_index["split"] == split_name
    ].reset_index(drop=True)


    X_clinical = np.zeros(
        (
            len(subset),
            MAX_SEQUENCE_LENGTH,
            CLINICAL_FEATURE_COUNT
        ),
        dtype=np.float32
    )


    X_missing = np.ones(
        (
            len(subset),
            MAX_SEQUENCE_LENGTH,
            CLINICAL_FEATURE_COUNT
        ),
        dtype=np.float32
    )


    lengths = np.zeros(
        len(subset),
        dtype=np.int64
    )


    y = subset[
        "label"
    ].to_numpy(
        dtype=np.float32
    )


    for index, row in enumerate(
        subset.itertuples(
            index=False
        )
    ):

        stay = int(
            row.stay_id
        )

        start_hour = int(
            row.start_hour
        )

        end_hour = int(
            row.end_hour
        )

        history_length = int(
            row.history_hours
        )


        raw_sequence = np.vstack(
            [
                hour_lookup[
                    (
                        stay,
                        hour
                    )
                ]
                for hour in range(
                    start_hour,
                    end_hour + 1
                )
            ]
        ).astype(
            np.float32
        )


        missing_mask = np.isnan(
            raw_sequence
        )


        # Missing values are replaced by
        # TRAINING mean before scaling.
        filled = np.where(
            missing_mask,
            scaler.mean_,
            raw_sequence
        )


        scaled = scaler.transform(
            filled
        ).astype(
            np.float32
        )


        missing_indicator = (
            missing_mask.astype(
                np.float32
            )
        )


        X_clinical[
            index,
            :history_length,
            :
        ] = scaled


        X_missing[
            index,
            :history_length,
            :
        ] = missing_indicator


        lengths[index] = (
            history_length
        )


    # 39 clinical features
    # +
    # 39 missingness features
    #
    # Total = 78 if clinical count = 39

    X = np.concatenate(
        [
            X_clinical,
            X_missing
        ],
        axis=2
    )


    return (
        X,
        lengths,
        y
    )


train_X, train_lengths, train_y = (
    create_arrays("train")
)

validation_X, validation_lengths, validation_y = (
    create_arrays("validation")
)

test_X, test_lengths, test_y = (
    create_arrays("test")
)


print()
print(
    "Train X:",
    train_X.shape
)

print(
    "Validation X:",
    validation_X.shape
)

print(
    "Test X:",
    test_X.shape
)

print(
    "Model input size:",
    MODEL_INPUT_SIZE
)


# ------------------------------------------------------------
# Save arrays
# ------------------------------------------------------------

np.savez_compressed(
    DATA_PROCESSED /
    "final_3to12_train.npz",
    X=train_X,
    lengths=train_lengths,
    y=train_y
)


np.savez_compressed(
    DATA_PROCESSED /
    "final_3to12_validation.npz",
    X=validation_X,
    lengths=validation_lengths,
    y=validation_y
)


np.savez_compressed(
    DATA_PROCESSED /
    "final_3to12_test.npz",
    X=test_X,
    lengths=test_lengths,
    y=test_y
)


# ------------------------------------------------------------
# Save metadata
# ------------------------------------------------------------

metadata = {

    "project":
        "XSepsisNet",

    "history_min_hours":
        MIN_HISTORY_HOURS,

    "history_max_hours":
        MAX_HISTORY_HOURS,

    "clinical_feature_count":
        CLINICAL_FEATURE_COUNT,

    "model_input_size":
        MODEL_INPUT_SIZE,

    "features":
        clinical_features,

    "train_windows":
        len(train_y),

    "validation_windows":
        len(validation_y),

    "test_windows":
        len(test_y),

    "train_positive":
        int(train_y.sum()),

    "validation_positive":
        int(validation_y.sum()),

    "test_positive":
        int(test_y.sum()),

    "scaler":
        "StandardScaler fitted on TRAIN ONLY",

    "patient_level_split":
        True,

    "seed":
        SEED
}


with open(
    DATA_PROCESSED /
    "final_3to12_training_metadata.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metadata,
        f,
        indent=2
    )


print()
print(
    "Training arrays created successfully."
)
print()


# ============================================================
# 8. BiLSTM TRAINING
# ============================================================

print("[8/8] Starting final 3–12 hour BiLSTM training...")
print()


class SepsisBiLSTM(
    nn.Module
):

    def __init__(
        self,
        input_size,
        hidden_size=64,
        num_layers=2,
        dropout=0.40
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


        self.classifier = nn.Sequential(

            nn.Linear(
                hidden_size * 2,
                64
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout
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

        packed = (
            nn.utils.rnn
            .pack_padded_sequence(
                x,
                lengths.cpu(),
                batch_first=True,
                enforce_sorted=False
            )
        )


        packed_output, _ = (
            self.lstm(
                packed
            )
        )


        output, _ = (
            nn.utils.rnn
            .pad_packed_sequence(
                packed_output,
                batch_first=True,
                total_length=x.size(1)
            )
        )


        batch_indices = torch.arange(
            output.size(0),
            device=output.device
        )


        last_indices = (
            lengths.to(
                output.device
            ) - 1
        )


        last_output = output[
            batch_indices,
            last_indices
        ]


        logits = self.classifier(
            self.dropout(
                last_output
            )
        )


        return logits.squeeze(1)


# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------

class SepsisDataset(
    Dataset
):

    def __init__(
        self,
        X,
        lengths,
        y
    ):

        self.X = torch.tensor(
            X,
            dtype=torch.float32
        )

        self.lengths = torch.tensor(
            lengths,
            dtype=torch.long
        )

        self.y = torch.tensor(
            y,
            dtype=torch.float32
        )


    def __len__(self):

        return len(self.y)


    def __getitem__(
        self,
        index
    ):

        return (
            self.X[index],
            self.lengths[index],
            self.y[index]
        )


train_dataset = SepsisDataset(
    train_X,
    train_lengths,
    train_y
)


validation_dataset = SepsisDataset(
    validation_X,
    validation_lengths,
    validation_y
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)


validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ------------------------------------------------------------
# Model
# ------------------------------------------------------------

model = SepsisBiLSTM(
    input_size=MODEL_INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT
).to(DEVICE)


trainable_parameters = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)


print(
    "BiLSTM trainable parameters:",
    f"{trainable_parameters:,}"
)


# ------------------------------------------------------------
# Class imbalance
# ------------------------------------------------------------

positive_count = float(
    train_y.sum()
)

negative_count = float(
    len(train_y) - positive_count
)


pos_weight_value = (
    negative_count /
    max(
        positive_count,
        1.0
    )
)


print(
    "Training positive samples:",
    int(positive_count)
)

print(
    "Training negative samples:",
    int(negative_count)
)

print(
    "BCE positive weight:",
    f"{pos_weight_value:.6f}"
)


criterion = (
    nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            pos_weight_value,
            dtype=torch.float32,
            device=DEVICE
        )
    )
)


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


scheduler = (
    torch.optim.lr_scheduler
    .ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2
    )
)


# ------------------------------------------------------------
# Training function
# ------------------------------------------------------------

def train_one_epoch():

    model.train()

    losses = []


    for (
        X,
        lengths,
        y
    ) in train_loader:

        X = X.to(
            DEVICE
        )

        lengths = lengths.to(
            DEVICE
        )

        y = y.to(
            DEVICE
        )


        optimizer.zero_grad()


        logits = model(
            X,
            lengths
        )


        loss = criterion(
            logits,
            y
        )


        loss.backward()


        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRADIENT_CLIP
        )


        optimizer.step()


        losses.append(
            float(
                loss.item()
            )
        )


    return float(
        np.mean(losses)
    )


# ------------------------------------------------------------
# Validation LOSS ONLY
#
# IMPORTANT:
# This Stage-1 script does NOT perform model evaluation.
# Validation loss is used ONLY for early stopping/model
# checkpoint selection during training.
# Full evaluation happens in Stage 2.
# ------------------------------------------------------------

@torch.no_grad()
def validation_loss():

    model.eval()

    losses = []


    for (
        X,
        lengths,
        y
    ) in validation_loader:

        X = X.to(
            DEVICE
        )

        lengths = lengths.to(
            DEVICE
        )

        y = y.to(
            DEVICE
        )


        logits = model(
            X,
            lengths
        )


        loss = criterion(
            logits,
            y
        )


        losses.append(
            float(
                loss.item()
            )
        )


    return float(
        np.mean(losses)
    )


# ------------------------------------------------------------
# TRAIN
# ------------------------------------------------------------

best_validation_loss = float(
    "inf"
)

best_epoch = 0

bad_epochs = 0

training_history = []


for epoch in range(
    1,
    MAX_EPOCHS + 1
):

    train_loss = (
        train_one_epoch()
    )


    val_loss = (
        validation_loss()
    )


    scheduler.step(
        val_loss
    )


    current_lr = (
        optimizer
        .param_groups[0]["lr"]
    )


    training_history.append(
        {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": val_loss,
            "learning_rate": current_lr
        }
    )


    print(
        f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
        f"Train Loss: {train_loss:.6f} | "
        f"Validation Loss: {val_loss:.6f} | "
        f"LR: {current_lr:.2e}"
    )


    # --------------------------------------------------------
    # BEST CHECKPOINT
    # --------------------------------------------------------

    if val_loss < best_validation_loss:

        best_validation_loss = val_loss

        best_epoch = epoch

        bad_epochs = 0


        checkpoint = {

            "model_state_dict":
                model.state_dict(),

            "input_size":
                MODEL_INPUT_SIZE,

            "clinical_feature_count":
                CLINICAL_FEATURE_COUNT,

            "features":
                clinical_features,

            "hidden_size":
                HIDDEN_SIZE,

            "num_layers":
                NUM_LAYERS,

            "dropout":
                DROPOUT,

            "history_min_hours":
                MIN_HISTORY_HOURS,

            "history_max_hours":
                MAX_HISTORY_HOURS,

            "batch_size":
                BATCH_SIZE,

            "learning_rate":
                LEARNING_RATE,

            "weight_decay":
                WEIGHT_DECAY,

            "positive_weight":
                pos_weight_value,

            "seed":
                SEED,

            "device":
                str(DEVICE),

            "best_epoch":
                best_epoch,

            "best_validation_loss":
                best_validation_loss,

            "trainable_parameters":
                trainable_parameters
        }


        torch.save(
            checkpoint,
            BILSTM_DIR /
            "final_3to12_bilstm_best.pt"
        )


        print(
            "  >>> NEW BEST MODEL SAVED"
        )


    else:

        bad_epochs += 1


        if (
            bad_epochs
            >=
            EARLY_STOPPING_PATIENCE
        ):

            print()
            print(
                "Early stopping triggered."
            )

            break


# ------------------------------------------------------------
# Save training history
# ------------------------------------------------------------

history_path = (
    BILSTM_DIR /
    "final_3to12_training_history.csv"
)


pd.DataFrame(
    training_history
).to_csv(
    history_path,
    index=False
)


# ------------------------------------------------------------
# Save model configuration
# ------------------------------------------------------------

config = {

    "project":
        "XSepsisNet",

    "stage":
        "Preprocessing + Model Training ONLY",

    "history_range":
        "3-12 hours",

    "forecast_horizon":
        "24 hours",

    "clinical_features":
        clinical_features,

    "clinical_feature_count":
        CLINICAL_FEATURE_COUNT,

    "model_input_size":
        MODEL_INPUT_SIZE,

    "architecture":
        "Bidirectional LSTM",

    "hidden_size":
        HIDDEN_SIZE,

    "num_layers":
        NUM_LAYERS,

    "bidirectional":
        True,

    "dropout":
        DROPOUT,

    "batch_size":
        BATCH_SIZE,

    "learning_rate":
        LEARNING_RATE,

    "weight_decay":
        WEIGHT_DECAY,

    "gradient_clip":
        GRADIENT_CLIP,

    "maximum_epochs":
        MAX_EPOCHS,

    "early_stopping_patience":
        EARLY_STOPPING_PATIENCE,

    "best_epoch":
        best_epoch,

    "best_validation_loss":
        best_validation_loss,

    "trainable_parameters":
        trainable_parameters,

    "scaler":
        "TRAIN ONLY",

    "split":
        "Patient level",

    "seed":
        SEED,

    "device":
        str(DEVICE)
}


with open(
    BILSTM_DIR /
    "final_3to12_bilstm_config.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        config,
        f,
        indent=2
    )


# ============================================================
# FINAL STAGE-1 REPORT
# ============================================================

print()
print("=" * 80)
print("STAGE 1 COMPLETED")
print("=" * 80)

print()
print(
    "3–12 hour preprocessing : DONE"
)

print(
    "Patient-level split     : DONE"
)

print(
    "TRAIN-only scaler       : DONE"
)

print(
    "BiLSTM training         : DONE"
)

print(
    "Best model saved        : DONE"
)

print()
print(
    "Best epoch:",
    best_epoch
)

print(
    "Best validation loss:",
    f"{best_validation_loss:.6f}"
)

print()
print(
    "BEST MODEL:"
)

print(
    BILSTM_DIR /
    "final_3to12_bilstm_best.pt"
)

print()
print(
    "TRAINING HISTORY:"
)

print(
    history_path
)

print()
print(
    "IMPORTANT:"
)

print(
    "No test-set evaluation was performed in this script."
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

print()
print("=" * 80)
print(
    "NEXT STAGE: EVALUATION"
)
print("=" * 80)