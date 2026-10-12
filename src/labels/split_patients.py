from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "temporal_sepsis_labels_with_patient.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"


TRAIN_FILE = OUTPUT_DIR / "patient_train_labels.csv"
VAL_FILE = OUTPUT_DIR / "patient_validation_labels.csv"
TEST_FILE = OUTPUT_DIR / "patient_test_labels.csv"

SUMMARY_FILE = OUTPUT_DIR / "patient_split_summary.json"


# ============================================================
# SETTINGS
# ============================================================

RANDOM_SEED = 42

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("PATIENT-LEVEL TRAIN / VALIDATION / TEST SPLIT")
    print("=" * 80)

    # --------------------------------------------------------
    # 1. Check input
    # --------------------------------------------------------

    if not INPUT_FILE.is_file():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    print(f"\nInput file:")
    print(INPUT_FILE)

    # --------------------------------------------------------
    # 2. Load labels
    # --------------------------------------------------------

    print("\nLoading final temporal labels...")

    df = pd.read_csv(INPUT_FILE)

    print(f"Total rows    : {len(df):,}")
    print(f"Total columns : {len(df.columns)}")

    required_columns = {
        "subject_id",
        "stay_id",
        "icu_hour",
        "label"
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    # --------------------------------------------------------
    # 3. Basic validation
    # --------------------------------------------------------

    print("\nValidating input...")

    if df["subject_id"].isna().any():
        raise ValueError(
            "Missing subject_id values detected."
        )

    if df["stay_id"].isna().any():
        raise ValueError(
            "Missing stay_id values detected."
        )

    if df["label"].isna().any():
        raise ValueError(
            "Missing label values detected."
        )

    duplicate_rows = df.duplicated(
        subset=["stay_id", "icu_hour"]
    ).sum()

    print(
        f"Duplicate stay_id + icu_hour rows: "
        f"{duplicate_rows:,}"
    )

    if duplicate_rows > 0:
        raise ValueError(
            "Duplicate prediction-time rows detected."
        )

    # --------------------------------------------------------
    # 4. Get unique patients
    # --------------------------------------------------------

    patients = (
        df["subject_id"]
        .drop_duplicates()
        .to_numpy()
    )

    print(f"\nUnique patients: {len(patients):,}")

    # --------------------------------------------------------
    # 5. First split
    #
    # 70% train
    # 30% temporary
    # --------------------------------------------------------

    train_patients, temp_patients = train_test_split(
        patients,
        test_size=(VALIDATION_RATIO + TEST_RATIO),
        random_state=RANDOM_SEED,
        shuffle=True
    )

    # --------------------------------------------------------
    # 6. Second split
    #
    # Temporary 30%:
    # 15% validation
    # 15% test
    #
    # Therefore:
    # validation = 50% of temporary
    # test       = 50% of temporary
    # --------------------------------------------------------

    validation_patients, test_patients = train_test_split(
        temp_patients,
        test_size=0.5,
        random_state=RANDOM_SEED,
        shuffle=True
    )

    # Convert to sets for fast lookup
    train_set = set(train_patients)
    validation_set = set(validation_patients)
    test_set = set(test_patients)

    # --------------------------------------------------------
    # 7. Check patient overlap
    # --------------------------------------------------------

    print("\nChecking patient overlap...")

    train_val_overlap = train_set & validation_set
    train_test_overlap = train_set & test_set
    val_test_overlap = validation_set & test_set

    print(
        f"Train ∩ Validation: {len(train_val_overlap)}"
    )
    print(
        f"Train ∩ Test      : {len(train_test_overlap)}"
    )
    print(
        f"Validation ∩ Test : {len(val_test_overlap)}"
    )

    if train_val_overlap:
        raise ValueError(
            "Patient leakage detected between Train and Validation."
        )

    if train_test_overlap:
        raise ValueError(
            "Patient leakage detected between Train and Test."
        )

    if val_test_overlap:
        raise ValueError(
            "Patient leakage detected between Validation and Test."
        )

    # --------------------------------------------------------
    # 8. Assign each row to a split
    # --------------------------------------------------------

    print("\nAssigning prediction rows...")

    def assign_split(subject_id):

        if subject_id in train_set:
            return "train"

        if subject_id in validation_set:
            return "validation"

        if subject_id in test_set:
            return "test"

        raise ValueError(
            f"Patient {subject_id} was not assigned to any split."
        )

    df["split"] = df["subject_id"].map(assign_split)

    # --------------------------------------------------------
    # 9. Split dataframe
    # --------------------------------------------------------

    train_df = df[df["split"] == "train"].copy()
    validation_df = df[df["split"] == "validation"].copy()
    test_df = df[df["split"] == "test"].copy()

    # Remove helper column before saving
    train_df.drop(columns=["split"], inplace=True)
    validation_df.drop(columns=["split"], inplace=True)
    test_df.drop(columns=["split"], inplace=True)

    # --------------------------------------------------------
    # 10. Statistics
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("SPLIT RESULTS")
    print("=" * 80)

    print("\nPATIENT COUNTS")
    print(f"Train      : {train_df['subject_id'].nunique():,}")
    print(f"Validation : {validation_df['subject_id'].nunique():,}")
    print(f"Test       : {test_df['subject_id'].nunique():,}")

    print("\nROW COUNTS")
    print(f"Train      : {len(train_df):,}")
    print(f"Validation : {len(validation_df):,}")
    print(f"Test       : {len(test_df):,}")

    print("\nSTAY COUNTS")
    print(f"Train      : {train_df['stay_id'].nunique():,}")
    print(f"Validation : {validation_df['stay_id'].nunique():,}")
    print(f"Test       : {test_df['stay_id'].nunique():,}")

    # --------------------------------------------------------
    # 11. Label distributions
    # --------------------------------------------------------

    print("\nLABEL DISTRIBUTION")

    for name, part in [
        ("TRAIN", train_df),
        ("VALIDATION", validation_df),
        ("TEST", test_df)
    ]:

        positives = int((part["label"] == 1).sum())
        negatives = int((part["label"] == 0).sum())
        total = len(part)

        positive_rate = (
            positives / total
            if total > 0
            else 0
        )

        print(
            f"{name:<12} "
            f"total={total:,} "
            f"positive={positives:,} "
            f"negative={negatives:,} "
            f"positive_rate={positive_rate:.4%}"
        )

    # --------------------------------------------------------
    # 12. Validate all rows are assigned exactly once
    # --------------------------------------------------------

    total_split_rows = (
        len(train_df)
        + len(validation_df)
        + len(test_df)
    )

    if total_split_rows != len(df):
        raise ValueError(
            "Split row counts do not add up to original dataset."
        )

    # --------------------------------------------------------
    # 13. Validate patient counts
    # --------------------------------------------------------

    total_split_patients = (
        train_df["subject_id"].nunique()
        + validation_df["subject_id"].nunique()
        + test_df["subject_id"].nunique()
    )

    if total_split_patients != df["subject_id"].nunique():
        raise ValueError(
            "Split patient counts do not add up to original patient count."
        )

    # --------------------------------------------------------
    # 14. Save datasets
    # --------------------------------------------------------

    print("\nSaving split files...")

    train_df.to_csv(
        TRAIN_FILE,
        index=False
    )

    validation_df.to_csv(
        VAL_FILE,
        index=False
    )

    test_df.to_csv(
        TEST_FILE,
        index=False
    )

    # --------------------------------------------------------
    # 15. Save summary
    # --------------------------------------------------------

    summary = {
        "random_seed": RANDOM_SEED,
        "ratios": {
            "train": TRAIN_RATIO,
            "validation": VALIDATION_RATIO,
            "test": TEST_RATIO
        },
        "total": {
            "rows": int(len(df)),
            "patients": int(df["subject_id"].nunique()),
            "stays": int(df["stay_id"].nunique())
        },
        "train": {
            "rows": int(len(train_df)),
            "patients": int(train_df["subject_id"].nunique()),
            "stays": int(train_df["stay_id"].nunique()),
            "positive": int((train_df["label"] == 1).sum()),
            "negative": int((train_df["label"] == 0).sum())
        },
        "validation": {
            "rows": int(len(validation_df)),
            "patients": int(validation_df["subject_id"].nunique()),
            "stays": int(validation_df["stay_id"].nunique()),
            "positive": int((validation_df["label"] == 1).sum()),
            "negative": int((validation_df["label"] == 0).sum())
        },
        "test": {
            "rows": int(len(test_df)),
            "patients": int(test_df["subject_id"].nunique()),
            "stays": int(test_df["stay_id"].nunique()),
            "positive": int((test_df["label"] == 1).sum()),
            "negative": int((test_df["label"] == 0).sum())
        },
        "patient_overlap": {
            "train_validation": len(train_val_overlap),
            "train_test": len(train_test_overlap),
            "validation_test": len(val_test_overlap)
        }
    }

    with open(
        SUMMARY_FILE,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            summary,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # 16. Final success message
    # --------------------------------------------------------

    print("\nFiles created:")

    print(f"Train:")
    print(TRAIN_FILE)

    print(f"\nValidation:")
    print(VAL_FILE)

    print(f"\nTest:")
    print(TEST_FILE)

    print(f"\nSummary:")
    print(SUMMARY_FILE)

    print("\n" + "=" * 80)
    print("SUCCESS")
    print("=" * 80)
    print("Patient-level split completed successfully.")
    print("No patient appears in more than one split.")
    print("=" * 80)


if __name__ == "__main__":
    main()