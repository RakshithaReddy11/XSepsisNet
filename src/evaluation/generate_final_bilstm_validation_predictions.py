from pathlib import Path
import json
import numpy as np
import torch
import torch.nn as nn


# ============================================================
# XSepsisNet
# Generate Validation Predictions from Final BiLSTM
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

VALIDATION_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "final_bilstm_validation.npz"
)

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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "models"
    / "evaluation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "final_bilstm_validation_predictions.npz"
)


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
        dropout
    ):
        super().__init__()

        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=(
                dropout
                if num_layers > 1
                else 0.0
            )
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

        # Use the final time step
        last_output = output[:, -1, :]

        logits = self.classifier(
            last_output
        )

        return logits.squeeze(1)


# ============================================================
# Main
# ============================================================

print("=" * 70)
print("XSepsisNet - Generate Validation Predictions")
print("=" * 70)


# ============================================================
# 1. Check files
# ============================================================

print("\n[1] Checking required files...")

for file_path in [
    VALIDATION_FILE,
    MODEL_FILE,
    CONFIG_FILE
]:

    if not file_path.exists():

        raise FileNotFoundError(
            f"Required file not found:\n{file_path}"
        )

    print(
        f"OK: {file_path.relative_to(PROJECT_ROOT)}"
    )


# ============================================================
# 2. Load configuration
# ============================================================

print("\n[2] Loading model configuration...")

with open(
    CONFIG_FILE,
    "r",
    encoding="utf-8"
) as f:

    config = json.load(f)

input_size = int(
    config["input_size"]
)

sequence_length = int(
    config["sequence_length"]
)

hidden_size = int(
    config["hidden_size"]
)

num_layers = int(
    config["num_layers"]
)

bidirectional = bool(
    config["bidirectional"]
)

dropout = float(
    config["dropout"]
)

print(
    f"Input size       : {input_size}"
)

print(
    f"Sequence length  : {sequence_length}"
)

print(
    f"Hidden size      : {hidden_size}"
)

print(
    f"Number of layers : {num_layers}"
)

print(
    f"Bidirectional    : {bidirectional}"
)

print(
    f"Dropout          : {dropout}"
)


# ============================================================
# 3. Load validation data
# ============================================================

print("\n[3] Loading validation tensors...")

validation = np.load(
    VALIDATION_FILE
)

print(
    "Arrays:",
    validation.files
)

X_val = validation["X"]
y_val = validation["y"]

print(
    f"X validation shape: {X_val.shape}"
)

print(
    f"y validation shape: {y_val.shape}"
)


# ============================================================
# 4. Validate tensor dimensions
# ============================================================

print("\n[4] Validating tensor dimensions...")

if X_val.ndim != 3:

    raise ValueError(
        f"Expected X to have 3 dimensions, "
        f"got {X_val.ndim}"
    )

if X_val.shape[1] != sequence_length:

    raise ValueError(
        f"Expected sequence length "
        f"{sequence_length}, "
        f"got {X_val.shape[1]}"
    )

if X_val.shape[2] != input_size:

    raise ValueError(
        f"Expected input size "
        f"{input_size}, "
        f"got {X_val.shape[2]}"
    )

if len(X_val) != len(y_val):

    raise ValueError(
        "X and y sample counts do not match."
    )

if np.isnan(X_val).any():

    raise ValueError(
        "NaN values found in validation X."
    )

if np.isinf(X_val).any():

    raise ValueError(
        "Infinite values found in validation X."
    )

print("Tensor validation passed.")


# ============================================================
# 5. Create model
# ============================================================

print("\n[5] Reconstructing final BiLSTM...")

device = torch.device("cpu")

model = FinalBiLSTM(
    input_size=input_size,
    hidden_size=hidden_size,
    num_layers=num_layers,
    bidirectional=bidirectional,
    dropout=dropout
)

model = model.to(device)


# ============================================================
# 6. Load trained checkpoint
# ============================================================

print("\n[6] Loading trained checkpoint...")

checkpoint = torch.load(
    MODEL_FILE,
    map_location=device,
    weights_only=False
)


# Handle different checkpoint formats
if isinstance(checkpoint, dict):

    if "model_state_dict" in checkpoint:

        state_dict = checkpoint[
            "model_state_dict"
        ]

    elif "state_dict" in checkpoint:

        state_dict = checkpoint[
            "state_dict"
        ]

    else:

        # Assume checkpoint itself is state_dict
        state_dict = checkpoint

else:

    raise ValueError(
        "Unsupported checkpoint format."
    )


model.load_state_dict(
    state_dict,
    strict=True
)

model.eval()

print(
    "Checkpoint loaded successfully."
)


# ============================================================
# 7. Generate validation predictions
# ============================================================

print("\n[7] Generating validation predictions...")

X_tensor = torch.tensor(
    X_val,
    dtype=torch.float32
)

batch_size = 128

all_probabilities = []

with torch.no_grad():

    for start in range(
        0,
        len(X_tensor),
        batch_size
    ):

        end = min(
            start + batch_size,
            len(X_tensor)
        )

        batch = X_tensor[
            start:end
        ].to(device)

        logits = model(batch)

        probabilities = torch.sigmoid(
            logits
        )

        all_probabilities.append(
            probabilities.cpu().numpy()
        )


y_probability = np.concatenate(
    all_probabilities
).astype(np.float32)

y_true = y_val.astype(np.float32)


# ============================================================
# 8. Validate predictions
# ============================================================

print("\n[8] Validating generated predictions...")

if len(y_probability) != len(y_true):

    raise ValueError(
        "Prediction count does not match "
        "validation label count."
    )

if np.isnan(y_probability).any():

    raise ValueError(
        "NaN values found in predictions."
    )

if np.isinf(y_probability).any():

    raise ValueError(
        "Infinite values found in predictions."
    )

if np.any(
    (y_probability < 0)
    |
    (y_probability > 1)
):

    raise ValueError(
        "Predicted probabilities outside [0, 1]."
    )


# ============================================================
# 9. Print summary
# ============================================================

positive_count = int(
    np.sum(y_true == 1)
)

negative_count = int(
    np.sum(y_true == 0)
)

print(
    f"\nValidation samples : {len(y_true)}"
)

print(
    f"Validation positive: {positive_count}"
)

print(
    f"Validation negative: {negative_count}"
)

print(
    "Validation prevalence: "
    f"{positive_count / len(y_true):.6f}"
)

print(
    "Probability minimum: "
    f"{y_probability.min():.6f}"
)

print(
    "Probability maximum: "
    f"{y_probability.max():.6f}"
)

print(
    "Probability mean: "
    f"{y_probability.mean():.6f}"
)


# ============================================================
# 10. Save predictions
# ============================================================

print("\n[9] Saving validation predictions...")

np.savez_compressed(
    OUTPUT_FILE,
    y_true=y_true,
    y_probability=y_probability
)

print(
    f"Saved:\n"
    f"{OUTPUT_FILE.relative_to(PROJECT_ROOT)}"
)


# ============================================================
# 11. Final validation
# ============================================================

saved = np.load(
    OUTPUT_FILE
)

saved_y_true = saved["y_true"]

saved_probability = saved[
    "y_probability"
]

if not np.array_equal(
    saved_y_true,
    y_true
):

    raise RuntimeError(
        "Saved y_true does not match."
    )

if not np.allclose(
    saved_probability,
    y_probability
):

    raise RuntimeError(
        "Saved probabilities do not match."
    )


print("\n" + "=" * 70)
print("VALIDATION PREDICTIONS COMPLETE")
print("=" * 70)

print(
    "\nNo training was performed."
)

print(
    "The existing final BiLSTM checkpoint "
    "was used only for inference."
)

print(
    "\nNext stage:"
)

print(
    "Fit calibration using VALIDATION predictions "
    "and evaluate the calibrated probabilities "
    "on the untouched TEST set."
)