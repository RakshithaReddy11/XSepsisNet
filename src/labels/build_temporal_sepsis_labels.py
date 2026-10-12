from pathlib import Path
import pandas as pd
import numpy as np
import json


# ============================================================
# XSepsisNet - TEMPORAL SEPSIS-3 LABEL CONSTRUCTION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

SOFA_FILE = PROCESSED_DIR / "final_sofa_hourly.csv"
MICRO_FILE = PROCESSED_DIR / "microbiology_icu.csv"

OUTPUT_FILE = PROCESSED_DIR / "temporal_sepsis_labels.csv"
SUMMARY_FILE = PROCESSED_DIR / "temporal_sepsis_label_summary.json"

HORIZON_HOURS = 6
MIN_COMPONENTS = 4

# EXACT columns present in final_sofa_hourly.csv
SOFA_COMPONENTS = [
    "respiratory_sofa",
    "coagulation_sofa",
    "liver_sofa",
    "cardiovascular_sofa",
    "cns_sofa",
    "renal_creatinine_sofa",
]


def print_header(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def main():

    print_header(
        "XSepsisNet - TEMPORAL SEPSIS-3 LABEL CONSTRUCTION"
    )

    # ========================================================
    # STEP 1: CHECK FILES
    # ========================================================

    print_header("STEP 1: CHECK INPUT FILES")

    if not SOFA_FILE.exists():
        raise FileNotFoundError(
            f"SOFA file not found:\n{SOFA_FILE}"
        )

    if not MICRO_FILE.exists():
        raise FileNotFoundError(
            f"Microbiology file not found:\n{MICRO_FILE}"
        )

    print(f"SOFA file: {SOFA_FILE}")
    print(f"Microbiology file: {MICRO_FILE}")

    # ========================================================
    # STEP 2: LOAD FINAL SOFA
    # ========================================================

    print_header("STEP 2: LOAD FINAL SOFA")

    required_columns = [
        "stay_id",
        "icu_hour",
        *SOFA_COMPONENTS,
    ]

    sofa = pd.read_csv(
        SOFA_FILE,
        usecols=required_columns
    )

    print(f"Rows: {len(sofa):,}")
    print(f"ICU stays: {sofa['stay_id'].nunique():,}")

    print()
    print("Detected SOFA components:")

    for col in SOFA_COMPONENTS:
        print(f"  {col}")

    # Numeric conversion
    sofa["stay_id"] = pd.to_numeric(
        sofa["stay_id"],
        errors="coerce"
    )

    sofa["icu_hour"] = pd.to_numeric(
        sofa["icu_hour"],
        errors="coerce"
    )

    for col in SOFA_COMPONENTS:
        sofa[col] = pd.to_numeric(
            sofa[col],
            errors="coerce"
        )

    sofa = sofa.dropna(
        subset=["stay_id", "icu_hour"]
    )

    sofa["stay_id"] = sofa["stay_id"].astype(int)
    sofa["icu_hour"] = sofa["icu_hour"].astype(int)

    sofa = (
        sofa
        .sort_values(["stay_id", "icu_hour"])
        .drop_duplicates(
            subset=["stay_id", "icu_hour"],
            keep="first"
        )
        .reset_index(drop=True)
    )

    # ========================================================
    # STEP 3: LOAD MICROBIOLOGY
    # ========================================================

    print_header("STEP 3: LOAD MICROBIOLOGY")

    micro = pd.read_csv(
        MICRO_FILE,
        usecols=["stay_id", "icu_hour"]
    )

    print(f"Microbiology rows: {len(micro):,}")
    print(
        f"Microbiology ICU stays: "
        f"{micro['stay_id'].nunique():,}"
    )

    micro["stay_id"] = pd.to_numeric(
        micro["stay_id"],
        errors="coerce"
    )

    micro["icu_hour"] = pd.to_numeric(
        micro["icu_hour"],
        errors="coerce"
    )

    micro = micro.dropna(
        subset=["stay_id", "icu_hour"]
    )

    micro["stay_id"] = micro["stay_id"].astype(int)
    micro["icu_hour"] = micro["icu_hour"].astype(int)

    micro_hours = (
        micro[
            ["stay_id", "icu_hour"]
        ]
        .drop_duplicates()
        .sort_values(
            ["stay_id", "icu_hour"]
        )
    )

    print(
        f"Unique microbiology stay-hours: "
        f"{len(micro_hours):,}"
    )

    # ========================================================
    # STEP 4: INDEX MICROBIOLOGY BY ICU STAY
    # ========================================================

    print_header("STEP 4: INDEX INFECTION EVIDENCE")

    micro_by_stay = {}

    for stay_id, group in micro_hours.groupby(
        "stay_id",
        sort=False
    ):
        micro_by_stay[int(stay_id)] = np.sort(
            group["icu_hour"]
            .astype(int)
            .unique()
        )

    print(
        f"Microbiology stays indexed: "
        f"{len(micro_by_stay):,}"
    )

    # ========================================================
    # STEP 5: PREPARE SOFA COMPONENT AVAILABILITY
    # ========================================================

    print_header("STEP 5: PREPARE SOFA COMPONENTS")

    sofa["component_count"] = (
        sofa[SOFA_COMPONENTS]
        .notna()
        .sum(axis=1)
    )

    sofa["complete_sofa_available"] = (
        sofa["component_count"] == 6
    )

    # Use existing total_sofa only when complete.
    # We deliberately do NOT treat missing components as zero.
    sofa["strict_total_sofa"] = np.where(
        sofa["complete_sofa_available"],
        sofa[SOFA_COMPONENTS].sum(axis=1),
        np.nan
    )

    print(
        "Complete six-component SOFA rows: "
        f"{sofa['complete_sofa_available'].sum():,}"
    )

    # ========================================================
    # STEP 6: CREATE TEMPORAL LABELS
    # ========================================================

    print_header("STEP 6: CREATE TEMPORAL LABELS")

    records = []

    total_stays = sofa["stay_id"].nunique()

    print(
        f"Processing ICU stays: {total_stays:,}"
    )

    processed = 0

    for stay_id, group in sofa.groupby(
        "stay_id",
        sort=False
    ):

        group = (
            group
            .sort_values("icu_hour")
            .reset_index(drop=True)
        )

        hours = group["icu_hour"].to_numpy()

        hour_to_index = {
            int(hour): index
            for index, hour in enumerate(hours)
        }

        infection_hours = micro_by_stay.get(
            int(stay_id),
            np.array([], dtype=int)
        )

        if len(infection_hours) == 0:
            processed += 1
            continue

        for current_hour in hours:

            current_hour = int(current_hour)

            future_start = current_hour + 1
            future_end = (
                current_hour + HORIZON_HOURS
            )

            # ------------------------------------------------
            # Future infection evidence
            # ------------------------------------------------

            future_micro = infection_hours[
                (
                    infection_hours
                    >= future_start
                )
                &
                (
                    infection_hours
                    <= future_end
                )
            ]

            if len(future_micro) == 0:
                continue

            current_index = hour_to_index[
                current_hour
            ]

            current_row = group.iloc[
                current_index
            ]

            # ------------------------------------------------
            # Search future SOFA observations
            # ------------------------------------------------

            future_indices = []

            for future_hour in range(
                future_start,
                future_end + 1
            ):
                if future_hour in hour_to_index:
                    future_indices.append(
                        hour_to_index[future_hour]
                    )

            if not future_indices:
                continue

            best_supported_delta = np.nan
            best_supported_future_hour = np.nan
            best_common_components = 0

            best_strict_delta = np.nan
            best_strict_future_hour = np.nan

            # ------------------------------------------------
            # Compare current SOFA with future SOFA
            # ------------------------------------------------

            for future_index in future_indices:

                future_row = group.iloc[
                    future_index
                ]

                # Components available at BOTH times.
                common_components = []

                for component in SOFA_COMPONENTS:

                    current_value = (
                        current_row[component]
                    )

                    future_value = (
                        future_row[component]
                    )

                    if (
                        pd.notna(current_value)
                        and
                        pd.notna(future_value)
                    ):
                        common_components.append(
                            component
                        )

                common_count = len(
                    common_components
                )

                # ------------------------------------------------
                # SUPPORTED TEMPORAL SOFA
                # ------------------------------------------------

                if common_count >= MIN_COMPONENTS:

                    current_sum = sum(
                        float(
                            current_row[c]
                        )
                        for c in common_components
                    )

                    future_sum = sum(
                        float(
                            future_row[c]
                        )
                        for c in common_components
                    )

                    delta = (
                        future_sum
                        - current_sum
                    )

                    if (
                        pd.isna(
                            best_supported_delta
                        )
                        or
                        delta > best_supported_delta
                    ):

                        best_supported_delta = (
                            float(delta)
                        )

                        best_supported_future_hour = (
                            int(
                                future_row[
                                    "icu_hour"
                                ]
                            )
                        )

                        best_common_components = (
                            common_count
                        )

                # ------------------------------------------------
                # STRICT COMPLETE SOFA
                # ------------------------------------------------

                if (
                    bool(
                        current_row[
                            "complete_sofa_available"
                        ]
                    )
                    and
                    bool(
                        future_row[
                            "complete_sofa_available"
                        ]
                    )
                ):

                    strict_delta = (
                        float(
                            future_row[
                                "strict_total_sofa"
                            ]
                        )
                        -
                        float(
                            current_row[
                                "strict_total_sofa"
                            ]
                        )
                    )

                    if (
                        pd.isna(
                            best_strict_delta
                        )
                        or
                        strict_delta
                        > best_strict_delta
                    ):

                        best_strict_delta = (
                            strict_delta
                        )

                        best_strict_future_hour = (
                            int(
                                future_row[
                                    "icu_hour"
                                ]
                            )
                        )

            # ------------------------------------------------
            # FINAL LABELS
            # ------------------------------------------------

            supported_positive = int(
                (
                    not pd.isna(
                        best_supported_delta
                    )
                )
                and
                (
                    best_supported_delta >= 2
                )
            )

            strict_positive = int(
                (
                    not pd.isna(
                        best_strict_delta
                    )
                )
                and
                (
                    best_strict_delta >= 2
                )
            )

            records.append({

                "stay_id": int(stay_id),

                "prediction_hour": current_hour,

                "horizon_hours": HORIZON_HOURS,

                "future_infection_evidence": 1,

                "first_future_micro_hour": int(
                    future_micro.min()
                ),

                "supported_sofa_delta": (
                    best_supported_delta
                ),

                "supported_sofa_future_hour": (
                    best_supported_future_hour
                ),

                "supported_common_components": (
                    best_common_components
                ),

                "temporal_sepsis_supported": (
                    supported_positive
                ),

                "strict_sofa_delta": (
                    best_strict_delta
                ),

                "strict_sofa_future_hour": (
                    best_strict_future_hour
                ),

                "temporal_sepsis_strict": (
                    strict_positive
                ),
            })

        processed += 1

        if processed % 5000 == 0:
            print(
                f"  Processed stays: "
                f"{processed:,} / {total_stays:,}"
            )

    labels = pd.DataFrame(records)

    # ========================================================
    # STEP 7: VALIDATE
    # ========================================================

    print_header("STEP 7: VALIDATE LABEL DATASET")

    if labels.empty:
        raise RuntimeError(
            "No temporal label rows were generated."
        )

    labels = (
        labels
        .sort_values(
            ["stay_id", "prediction_hour"]
        )
        .reset_index(drop=True)
    )

    duplicate_count = labels.duplicated(
        subset=[
            "stay_id",
            "prediction_hour"
        ]
    ).sum()

    if duplicate_count != 0:
        raise RuntimeError(
            f"Duplicate prediction positions: "
            f"{duplicate_count}"
        )

    supported_positive = int(
        labels[
            "temporal_sepsis_supported"
        ].sum()
    )

    strict_positive = int(
        labels[
            "temporal_sepsis_strict"
        ].sum()
    )

    supported_known = int(
        labels[
            "supported_sofa_delta"
        ].notna().sum()
    )

    strict_known = int(
        labels[
            "strict_sofa_delta"
        ].notna().sum()
    )

    total_rows = len(labels)

    supported_negative = (
        total_rows
        - supported_positive
    )

    strict_negative = (
        total_rows
        - strict_positive
    )

    print(
        f"Label rows: {total_rows:,}"
    )

    print(
        f"ICU stays represented: "
        f"{labels['stay_id'].nunique():,}"
    )

    print()
    print("--------------------------------------")
    print("SUPPORTED TEMPORAL LABEL")
    print("--------------------------------------")

    print(
        f"Positive: {supported_positive:,}"
    )

    print(
        f"Negative: {supported_negative:,}"
    )

    print(
        f"Positive rate: "
        f"{supported_positive / total_rows * 100:.4f}%"
    )

    print()
    print("--------------------------------------")
    print("STRICT TEMPORAL LABEL")
    print("--------------------------------------")

    print(
        f"Positive: {strict_positive:,}"
    )

    print(
        f"Negative: {strict_negative:,}"
    )

    print(
        f"Positive rate: "
        f"{strict_positive / total_rows * 100:.4f}%"
    )

    print()
    print(
        "Supported SOFA comparisons available: "
        f"{supported_known:,}"
    )

    print(
        "Strict complete-SOFA comparisons available: "
        f"{strict_known:,}"
    )

    # ========================================================
    # STEP 8: SAVE
    # ========================================================

    print_header("STEP 8: SAVE TEMPORAL LABELS")

    labels.to_csv(
        OUTPUT_FILE,
        index=False
    )

    summary = {

        "horizon_hours": HORIZON_HOURS,

        "minimum_common_sofa_components": (
            MIN_COMPONENTS
        ),

        "label_rows": int(total_rows),

        "icu_stays": int(
            labels["stay_id"].nunique()
        ),

        "supported_positive": (
            supported_positive
        ),

        "supported_negative": (
            supported_negative
        ),

        "strict_positive": (
            strict_positive
        ),

        "strict_negative": (
            strict_negative
        ),

        "supported_sofa_comparisons": (
            supported_known
        ),

        "strict_sofa_comparisons": (
            strict_known
        ),

        "supported_positive_rate": float(
            supported_positive / total_rows
        ),

        "strict_positive_rate": float(
            strict_positive / total_rows
        ),

        "interpretation": (
            "The strict temporal label requires complete "
            "six-component SOFA at the prediction hour and "
            "future hour, plus a SOFA increase of at least 2 "
            "during the six-hour horizon with future "
            "microbiology evidence. The supported label "
            "requires at least four SOFA components observed "
            "at both times and a deterioration of at least 2. "
            "The supported label is a project-specific temporal "
            "SOFA deterioration proxy and should not be "
            "described as confirmed Sepsis-3 onset."
        )
    }

    with open(
        SUMMARY_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=2
        )

    # ========================================================
    # FINAL
    # ========================================================

    print_header(
        "SUCCESS - TEMPORAL LABEL CONSTRUCTION COMPLETE"
    )

    print()
    print("Label file:")
    print(OUTPUT_FILE)

    print()
    print("Summary file:")
    print(SUMMARY_FILE)

    print()
    print("IMPORTANT:")
    print(
        "  Strict label = conservative complete-SOFA label."
    )

    print(
        "  Supported label = temporal SOFA deterioration "
        "proxy using >=4 common components."
    )

    print()
    print(
        "Next step: analyze the label counts before "
        "rebuilding the final BiLSTM dataset."
    )


if __name__ == "__main__":
    main()