from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

LABELS_FILE = PROJECT_ROOT / "data" / "processed" / "temporal_sepsis_labels.csv"
COHORT_FILE = PROJECT_ROOT / "data" / "processed" / "icu_cohort.csv"
OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "temporal_sepsis_labels_with_patient.csv"
)


def main():
    print("=" * 80)
    print("ATTACH SUBJECT_ID TO FINAL TEMPORAL SEPSIS LABELS")
    print("=" * 80)

    # ------------------------------------------------------------------
    # 1. Check input files
    # ------------------------------------------------------------------
    if not LABELS_FILE.is_file():
        raise FileNotFoundError(
            f"Labels file not found:\n{LABELS_FILE}"
        )

    if not COHORT_FILE.is_file():
        raise FileNotFoundError(
            f"ICU cohort file not found:\n{COHORT_FILE}"
        )

    print(f"\nLabels file : {LABELS_FILE}")
    print(f"Cohort file : {COHORT_FILE}")

    # ------------------------------------------------------------------
    # 2. Load final temporal labels
    # ------------------------------------------------------------------
    print("\nLoading temporal labels...")

    labels = pd.read_csv(LABELS_FILE)

    print(f"Label rows   : {len(labels):,}")
    print(f"Label columns: {len(labels.columns)}")

    if "stay_id" not in labels.columns:
        raise ValueError(
            "The temporal labels file does not contain 'stay_id'."
        )

    # ------------------------------------------------------------------
    # 3. Load ICU cohort mapping
    # ------------------------------------------------------------------
    print("\nLoading ICU cohort...")

    cohort = pd.read_csv(
        COHORT_FILE,
        usecols=["stay_id", "subject_id"]
    )

    print(f"Cohort rows: {len(cohort):,}")

    # ------------------------------------------------------------------
    # 4. Validate cohort mapping
    # ------------------------------------------------------------------
    print("\nValidating stay_id -> subject_id mapping...")

    missing_stay_id = cohort["stay_id"].isna().sum()
    missing_subject_id = cohort["subject_id"].isna().sum()

    print(f"Missing stay_id    : {missing_stay_id:,}")
    print(f"Missing subject_id : {missing_subject_id:,}")

    if missing_stay_id > 0:
        raise ValueError("Cohort contains missing stay_id values.")

    if missing_subject_id > 0:
        raise ValueError("Cohort contains missing subject_id values.")

    duplicate_stays = cohort["stay_id"].duplicated().sum()

    print(f"Duplicate stay_id mappings: {duplicate_stays:,}")

    if duplicate_stays > 0:
        raise ValueError(
            "A stay_id maps to multiple cohort rows. "
            "Patient mapping cannot safely continue."
        )

    # ------------------------------------------------------------------
    # 5. Attach subject_id
    # ------------------------------------------------------------------
    print("\nMapping subject_id to labels...")

    labels_with_patient = labels.merge(
        cohort,
        on="stay_id",
        how="left",
        validate="many_to_one"
    )

    # ------------------------------------------------------------------
    # 6. Validate mapping completeness
    # ------------------------------------------------------------------
    missing_patient = labels_with_patient["subject_id"].isna().sum()

    print(f"Labels without subject_id: {missing_patient:,}")

    if missing_patient > 0:
        missing_stays = (
            labels_with_patient.loc[
                labels_with_patient["subject_id"].isna(),
                "stay_id"
            ]
            .drop_duplicates()
            .tolist()
        )

        print("\nExample unmapped stay_ids:")
        print(missing_stays[:20])

        raise ValueError(
            "Some temporal labels could not be mapped to subject_id."
        )

    # ------------------------------------------------------------------
    # 7. Check that every stay has exactly one subject_id
    # ------------------------------------------------------------------
    mapping_check = (
        labels_with_patient[["stay_id", "subject_id"]]
        .drop_duplicates()
        .groupby("stay_id")["subject_id"]
        .nunique()
    )

    invalid_stays = (mapping_check != 1).sum()

    print(f"Stays with invalid patient mapping: {invalid_stays:,}")

    if invalid_stays > 0:
        raise ValueError(
            "At least one stay_id maps to more than one subject_id."
        )

    # ------------------------------------------------------------------
    # 8. Basic label validation
    # ------------------------------------------------------------------
    print("\nValidating label rows...")

    duplicate_label_rows = labels_with_patient.duplicated(
        subset=["stay_id", "icu_hour"]
    ).sum()

    print(
        f"Duplicate stay_id + icu_hour rows: "
        f"{duplicate_label_rows:,}"
    )

    if duplicate_label_rows > 0:
        raise ValueError(
            "Duplicate prediction-time rows detected."
        )

    # ------------------------------------------------------------------
    # 9. Patient statistics
    # ------------------------------------------------------------------
    unique_stays = labels_with_patient["stay_id"].nunique()
    unique_patients = labels_with_patient["subject_id"].nunique()

    print("\nFinal mapping statistics:")
    print(f"Unique ICU stays    : {unique_stays:,}")
    print(f"Unique patients     : {unique_patients:,}")
    print(f"Total label rows    : {len(labels_with_patient):,}")

    if "label" in labels_with_patient.columns:
        print("\nLabel distribution:")
        print(
            labels_with_patient["label"]
            .value_counts(dropna=False)
            .sort_index()
        )

    # ------------------------------------------------------------------
    # 10. Save
    # ------------------------------------------------------------------
    print("\nSaving output...")

    labels_with_patient.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(f"\nSaved successfully:")
    print(OUTPUT_FILE)

    # ------------------------------------------------------------------
    # 11. Final verification by reloading
    # ------------------------------------------------------------------
    print("\nReloading output for final verification...")

    check = pd.read_csv(
        OUTPUT_FILE,
        usecols=["stay_id", "icu_hour", "subject_id"]
    )

    print(f"Reloaded rows: {len(check):,}")

    if len(check) != len(labels):
        raise ValueError(
            "Output row count does not match input label row count."
        )

    if check["subject_id"].isna().any():
        raise ValueError(
            "Output still contains missing subject_id values."
        )

    print("\n" + "=" * 80)
    print("SUCCESS")
    print("=" * 80)
    print("Every final temporal label has a valid subject_id.")
    print("The dataset is ready for patient-level splitting.")
    print("=" * 80)


if __name__ == "__main__":
    main()