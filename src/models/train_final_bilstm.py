from pathlib import Path
import json
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models" / "bilstm"

TRAIN_FILE = DATA_DIR / "final_bilstm_train.npz"
VAL_FILE = DATA_DIR / "final_bilstm_validation.npz"
TEST_FILE = DATA_DIR / "final_bilstm_test.npz"

BEST_MODEL_FILE = MODEL_DIR / "final_bilstm_best.pt"
HISTORY_FILE = MODEL_DIR / "final_bilstm_training_history.json"
CONFIG_FILE = MODEL_DIR / "final_bilstm_config.json"
TEST_PREDICTIONS_FILE = MODEL_DIR / "final_bilstm_test_predictions.npz"


# ============================================================
# SETTINGS
# ============================================================

RANDOM_SEED = 42

INPUT_SIZE = 78
SEQUENCE_LENGTH = 6

HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.30

BATCH_SIZE = 128

EPOCHS = 20

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-5

PATIENCE = 5

NUM_WORKERS = 0


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


# ============================================================
# MODEL
# ============================================================

class BiLSTMClassifier(nn.Module):

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
            ),
        )

        self.dropout = nn.Dropout(
            dropout
        )

        self.classifier = nn.Linear(
            hidden_size * 2,
            1
        )

    def forward(self, x):

        output, _ = self.lstm(x)

        # Last time step
        last_output = output[:, -1, :]

        last_output = self.dropout(
            last_output
        )

        logits = self.classifier(
            last_output
        )

        return logits.squeeze(1)


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset(path):

    if not path.is_file():

        raise FileNotFoundError(
            f"Dataset not found:\n{path}"
        )

    data = np.load(
        path
    )

    X = data["X"].astype(
        np.float32
    )

    y = data["y"].astype(
        np.float32
    )

    return X, y


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    probabilities,
    threshold=0.5
):

    predictions = (
        probabilities >= threshold
    ).astype(
        np.int32
    )

    metrics = {}

    # --------------------------------------------------------
    # AUROC
    # --------------------------------------------------------

    if len(np.unique(y_true)) == 2:

        metrics["auroc"] = float(
            roc_auc_score(
                y_true,
                probabilities
            )
        )

        metrics["auprc"] = float(
            average_precision_score(
                y_true,
                probabilities
            )
        )

    else:

        metrics["auroc"] = None

        metrics["auprc"] = None

    # --------------------------------------------------------
    # Classification metrics
    # --------------------------------------------------------

    metrics["precision"] = float(
        precision_score(
            y_true,
            predictions,
            zero_division=0
        )
    )

    metrics["recall"] = float(
        recall_score(
            y_true,
            predictions,
            zero_division=0
        )
    )

    metrics["f1"] = float(
        f1_score(
            y_true,
            predictions,
            zero_division=0
        )
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    ).ravel()

    metrics["true_negative"] = int(tn)

    metrics["false_positive"] = int(fp)

    metrics["false_negative"] = int(fn)

    metrics["true_positive"] = int(tp)

    metrics["specificity"] = float(
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    return metrics


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device
):

    model.train()

    running_loss = 0.0

    total_samples = 0

    for X_batch, y_batch in loader:

        X_batch = X_batch.to(
            device
        )

        y_batch = y_batch.to(
            device
        )

        optimizer.zero_grad()

        logits = model(
            X_batch
        )

        loss = criterion(
            logits,
            y_batch
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        batch_size = (
            X_batch.size(0)
        )

        running_loss += (
            loss.item()
            * batch_size
        )

        total_samples += batch_size

    return (
        running_loss
        / total_samples
    )


# ============================================================
# EVALUATE
# ============================================================

def evaluate(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    running_loss = 0.0

    total_samples = 0

    all_probabilities = []

    all_labels = []

    with torch.no_grad():

        for X_batch, y_batch in loader:

            X_batch = X_batch.to(
                device
            )

            y_batch = y_batch.to(
                device
            )

            logits = model(
                X_batch
            )

            loss = criterion(
                logits,
                y_batch
            )

            probabilities = torch.sigmoid(
                logits
            )

            batch_size = (
                X_batch.size(0)
            )

            running_loss += (
                loss.item()
                * batch_size
            )

            total_samples += batch_size

            all_probabilities.append(
                probabilities.cpu().numpy()
            )

            all_labels.append(
                y_batch.cpu().numpy()
            )

    average_loss = (
        running_loss
        / total_samples
    )

    probabilities = np.concatenate(
        all_probabilities
    )

    labels = np.concatenate(
        all_labels
    )

    metrics = calculate_metrics(
        labels,
        probabilities,
        threshold=0.5
    )

    metrics["loss"] = float(
        average_loss
    )

    return metrics, labels, probabilities


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("FINAL BILSTM TRAINING")
    print("=" * 80)

    set_seed(
        RANDOM_SEED
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"\nDevice: {device}"
    )

    if device.type == "cuda":

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    else:

        print(
            "GPU not available. "
            "Training will use CPU."
        )

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    print(
        "\nLoading prepared datasets..."
    )

    X_train, y_train = load_dataset(
        TRAIN_FILE
    )

    X_val, y_val = load_dataset(
        VAL_FILE
    )

    X_test, y_test = load_dataset(
        TEST_FILE
    )

    print(
        f"Train: "
        f"X={X_train.shape}, "
        f"y={y_train.shape}"
    )

    print(
        f"Validation: "
        f"X={X_val.shape}, "
        f"y={y_val.shape}"
    )

    print(
        f"Test: "
        f"X={X_test.shape}, "
        f"y={y_test.shape}"
    )

    # --------------------------------------------------------
    # Validate shapes
    # --------------------------------------------------------

    expected_train_shape = (
        len(X_train),
        SEQUENCE_LENGTH,
        INPUT_SIZE
    )

    expected_val_shape = (
        len(X_val),
        SEQUENCE_LENGTH,
        INPUT_SIZE
    )

    expected_test_shape = (
        len(X_test),
        SEQUENCE_LENGTH,
        INPUT_SIZE
    )

    if X_train.shape != expected_train_shape:

        raise ValueError(
            f"Unexpected train shape: "
            f"{X_train.shape}"
        )

    if X_val.shape != expected_val_shape:

        raise ValueError(
            f"Unexpected validation shape: "
            f"{X_val.shape}"
        )

    if X_test.shape != expected_test_shape:

        raise ValueError(
            f"Unexpected test shape: "
            f"{X_test.shape}"
        )

    # --------------------------------------------------------
    # Class distribution
    # --------------------------------------------------------

    train_positive = int(
        y_train.sum()
    )

    train_negative = (
        len(y_train)
        - train_positive
    )

    print(
        "\nTRAIN CLASS DISTRIBUTION"
    )

    print(
        f"Positive: {train_positive:,}"
    )

    print(
        f"Negative: {train_negative:,}"
    )

    print(
        f"Positive rate: "
        f"{train_positive / len(y_train):.4%}"
    )

    # --------------------------------------------------------
    # Tensor datasets
    # --------------------------------------------------------

    train_dataset = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train)
    )

    validation_dataset = TensorDataset(
        torch.from_numpy(X_val),
        torch.from_numpy(y_val)
    )

    test_dataset = TensorDataset(
        torch.from_numpy(X_test),
        torch.from_numpy(y_test)
    )

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

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(
            device.type == "cuda"
        )
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print(
        "\nCreating BiLSTM model..."
    )

    model = BiLSTMClassifier(
        input_size=INPUT_SIZE,
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
        if p.requires_grad
    )

    print(
        f"Trainable parameters: "
        f"{parameter_count:,}"
    )

    # --------------------------------------------------------
    # Positive class weighting
    # --------------------------------------------------------

    positive_weight = (
        train_negative
        / train_positive
    )

    print(
        f"\nPositive class weight: "
        f"{positive_weight:.4f}"
    )

    pos_weight = torch.tensor(
        [positive_weight],
        dtype=torch.float32,
        device=device
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    history = []

    best_validation_auprc = -np.inf

    best_epoch = 0

    epochs_without_improvement = 0

    print("\n" + "=" * 80)
    print("TRAINING")
    print("=" * 80)

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device
        )

        validation_metrics, _, _ = evaluate(
            model,
            validation_loader,
            criterion,
            device
        )

        validation_auprc = (
            validation_metrics["auprc"]
        )

        scheduler.step(
            validation_auprc
        )

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        epoch_record = {

            "epoch":
                epoch,

            "train_loss":
                float(train_loss),

            "validation_loss":
                validation_metrics["loss"],

            "validation_auroc":
                validation_metrics["auroc"],

            "validation_auprc":
                validation_metrics["auprc"],

            "validation_precision":
                validation_metrics["precision"],

            "validation_recall":
                validation_metrics["recall"],

            "validation_f1":
                validation_metrics["f1"],

            "validation_specificity":
                validation_metrics["specificity"],

            "learning_rate":
                float(current_lr),
        }

        history.append(
            epoch_record
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.5f} | "
            f"Val Loss: "
            f"{validation_metrics['loss']:.5f} | "
            f"Val AUROC: "
            f"{validation_metrics['auroc']:.4f} | "
            f"Val AUPRC: "
            f"{validation_metrics['auprc']:.4f} | "
            f"Val F1: "
            f"{validation_metrics['f1']:.4f}"
        )

        # ----------------------------------------------------
        # Save best model by validation AUPRC
        # ----------------------------------------------------

        if (
            validation_auprc
            > best_validation_auprc
        ):

            best_validation_auprc = (
                validation_auprc
            )

            best_epoch = epoch

            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "input_size":
                        INPUT_SIZE,

                    "sequence_length":
                        SEQUENCE_LENGTH,

                    "hidden_size":
                        HIDDEN_SIZE,

                    "num_layers":
                        NUM_LAYERS,

                    "dropout":
                        DROPOUT,

                    "best_epoch":
                        best_epoch,

                    "best_validation_auprc":
                        float(
                            best_validation_auprc
                        ),

                    "random_seed":
                        RANDOM_SEED,
                },
                BEST_MODEL_FILE
            )

            print(
                "  -> Best model saved."
            )

        else:

            epochs_without_improvement += 1

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if (
            epochs_without_improvement
            >= PATIENCE
        ):

            print(
                f"\nEarly stopping after "
                f"{epoch} epochs."
            )

            break

    # --------------------------------------------------------
    # Save history
    # --------------------------------------------------------

    with open(
        HISTORY_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            history,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # Load best model
    # --------------------------------------------------------

    print(
        "\nLoading best validation model..."
    )

    checkpoint = torch.load(
        BEST_MODEL_FILE,
        map_location=device
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    # --------------------------------------------------------
    # Final test evaluation
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("FINAL TEST EVALUATION")
    print("=" * 80)

    test_metrics, test_labels, test_probabilities = evaluate(
        model,
        test_loader,
        criterion,
        device
    )

    print(
        f"\nTest Loss       : "
        f"{test_metrics['loss']:.6f}"
    )

    print(
        f"Test AUROC      : "
        f"{test_metrics['auroc']:.6f}"
    )

    print(
        f"Test AUPRC      : "
        f"{test_metrics['auprc']:.6f}"
    )

    print(
        f"Test Precision   : "
        f"{test_metrics['precision']:.6f}"
    )

    print(
        f"Test Recall      : "
        f"{test_metrics['recall']:.6f}"
    )

    print(
        f"Test F1          : "
        f"{test_metrics['f1']:.6f}"
    )

    print(
        f"Test Specificity : "
        f"{test_metrics['specificity']:.6f}"
    )

    print(
        "\nConfusion matrix:"
    )

    print(
        f"TN={test_metrics['true_negative']} "
        f"FP={test_metrics['false_positive']} "
        f"FN={test_metrics['false_negative']} "
        f"TP={test_metrics['true_positive']}"
    )

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    np.savez_compressed(
        TEST_PREDICTIONS_FILE,
        y_true=test_labels,
        y_probability=test_probabilities
    )

    # --------------------------------------------------------
    # Save configuration
    # --------------------------------------------------------

    config = {

        "random_seed":
            RANDOM_SEED,

        "input_size":
            INPUT_SIZE,

        "sequence_length":
            SEQUENCE_LENGTH,

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

        "epochs_requested":
            EPOCHS,

        "learning_rate":
            LEARNING_RATE,

        "weight_decay":
            WEIGHT_DECAY,

        "patience":
            PATIENCE,

        "positive_class_weight":
            float(
                positive_weight
            ),

        "best_epoch":
            int(best_epoch),

        "best_validation_auprc":
            float(
                best_validation_auprc
            ),

        "test_metrics":
            test_metrics,

    }

    with open(
        CONFIG_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            config,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("FINAL BILSTM TRAINING COMPLETE")
    print("=" * 80)

    print(
        f"\nBest epoch: {best_epoch}"
    )

    print(
        f"Best validation AUPRC: "
        f"{best_validation_auprc:.6f}"
    )

    print(
        f"\nBest model:"
        f"\n{BEST_MODEL_FILE}"
    )

    print(
        f"\nTraining history:"
        f"\n{HISTORY_FILE}"
    )

    print(
        f"\nTest predictions:"
        f"\n{TEST_PREDICTIONS_FILE}"
    )

    print(
        f"\nConfiguration:"
        f"\n{CONFIG_FILE}"
    )

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()