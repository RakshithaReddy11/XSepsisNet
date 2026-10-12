from pathlib import Path
import json
import numpy as np
import pandas as pd


# ============================================================
# XSepsisNet
# FINAL TEMPORAL SEPSIS LABEL CONSTRUCTION
#
# Final operational label:
#
#   Prediction hour
#          |
#          +--> future microbiology within 24h
#          |
#          +--> future SOFA observations within 24h
#          |
#          +--> at least 3 common SOFA components
#          |
#          +--> maximum future SOFA deterioration >= 2
#          |
#          +--> label = 1
#
# Otherwise:
#          label = 0
#
# IMPORTANT:
# This is a project-specific temporal proxy label.
# It is NOT being claimed as exact Sepsis-3 onset.
#
# Future information is used ONLY for target construction.
# It must NOT be included in the BiLSTM input features.
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOFA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "final_sofa_hourly.csv"
)

MICRO_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "microbiology_icu.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "temporal_sepsis_labels.csv"
)

SUMMARY_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "temporal_sepsis_label_summary.json"
)


HORIZON = 24

MIN_COMMON_COMPONENTS = 3

SOFA_DELTA_THRESHOLD = 2

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


# ============================================================
# LOAD SOFA
# ============================================================

def load_sofa():

    print_header("STEP 1: LOAD FINAL SOFA")

    columns = [
        "stay_id",
        "icu_hour",
        *SOFA_COMPONENTS,
    ]

    df = pd.read_csv(
        SOFA_FILE,
        usecols=columns
    )

    df["stay_id"] = pd.to_numeric(
        df["stay_id"],
        errors="coerce"
    )

    df["icu_hour"] = pd.to_numeric(
        df["icu_hour"],
        errors="coerce"
    )

    for col in SOFA_COMPONENTS:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df = df.dropna(
        subset=[
            "stay_id",
            "icu_hour",
        ]
    ).copy()

    df["stay_id"] = df["stay_id"].astype(
        np.int64
    )

    df["icu_hour"] = df["icu_hour"].astype(
        np.int64
    )

    df = df.sort_values(
        [
            "stay_id",
            "icu_hour",
        ]
    ).reset_index(
        drop=True
    )

    print(
        f"SOFA rows: {len(df):,}"
    )

    print(
        f"ICU stays: "
        f"{df['stay_id'].nunique():,}"
    )

    return df


# ============================================================
# LOAD MICROBIOLOGY
# ============================================================

def load_microbiology():

    print_header(
        "STEP 2: LOAD MICROBIOLOGY"
    )

    df = pd.read_csv(
        MICRO_FILE,
        usecols=[
            "stay_id",
            "icu_hour",
        ]
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
            "icu_hour",
        ]
    ).copy()

    df["stay_id"] = df["stay_id"].astype(
        np.int64
    )

    df["icu_hour"] = df["icu_hour"].astype(
        np.int64
    )

    df = (
        df[
            [
                "stay_id",
                "icu_hour",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "stay_id",
                "icu_hour",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    print(
        f"Unique microbiology stay-hours: "
        f"{len(df):,}"
    )

    print(
        f"Microbiology stays: "
        f"{df['stay_id'].nunique():,}"
    )

    return df


# ============================================================
# MICROBIOLOGY INDEX
# ============================================================

def build_micro_index(micro):

    print_header(
        "STEP 3: BUILD MICROBIOLOGY INDEX"
    )

    micro_index = {}

    for stay_id, group in micro.groupby(
        "stay_id",
        sort=False
    ):

        micro_index[int(stay_id)] = (
            group["icu_hour"]
            .to_numpy(
                dtype=np.int32
            )
        )

    print(
        f"Indexed stays: "
        f"{len(micro_index):,}"
    )

    return micro_index


# ============================================================
# PREPARE SOFA STAYS
# ============================================================

def prepare_stay_data(sofa):

    print_header(
        "STEP 4: PREPARE SOFA BY ICU STAY"
    )

    stay_data = {}

    for stay_id, group in sofa.groupby(
        "stay_id",
        sort=False
    ):

        group = group.sort_values(
            "icu_hour"
        )

        hours = group[
            "icu_hour"
        ].to_numpy(
            dtype=np.int32
        )

        values = group[
            SOFA_COMPONENTS
        ].to_numpy(
            dtype=np.float32
        )

        stay_data[int(stay_id)] = {
            "hours": hours,
            "values": values,
        }

    print(
        f"Prepared stays: "
        f"{len(stay_data):,}"
    )

    return stay_data


# ============================================================
# FUTURE MICROBIOLOGY CHECK
# ============================================================

def has_future_microbiology(
    micro_hours,
    prediction_hour
):

    if micro_hours is None:
        return False

    lower = prediction_hour + 1

    upper = (
        prediction_hour
        + HORIZON
    )

    left = np.searchsorted(
        micro_hours,
        lower,
        side="left"
    )

    right = np.searchsorted(
        micro_hours,
        upper,
        side="right"
    )

    return right > left


# ============================================================
# FIND MICROBIOLOGY TIMING
# ============================================================

def first_future_micro_hour(
    micro_hours,
    prediction_hour
):

    if micro_hours is None:
        return np.nan

    lower = prediction_hour + 1

    upper = (
        prediction_hour
        + HORIZON
    )

    left = np.searchsorted(
        micro_hours,
        lower,
        side="left"
    )

    right = np.searchsorted(
        micro_hours,
        upper,
        side="right"
    )

    if right <= left:
        return np.nan

    return int(
        micro_hours[left]
    )


# ============================================================
# BUILD LABELS
# ============================================================

def build_labels(
    stay_data,
    micro_index
):

    print_header(
        "STEP 5: BUILD FINAL TEMPORAL LABELS"
    )

    output_rows = []

    total_prediction_hours = 0

    micro_eligible = 0

    sofa_eligible = 0

    positive_count = 0

    negative_count = 0

    positive_stays = set()

    negative_stays = set()

    for stay_id, data in stay_data.items():

        hours = data["hours"]

        values = data["values"]

        micro_hours = micro_index.get(
            stay_id
        )

        for pred_idx, prediction_hour in enumerate(
            hours
        ):

            prediction_hour = int(
                prediction_hour
            )

            total_prediction_hours += 1

            # ------------------------------------------------
            # Infection evidence in next 24 hours.
            # ------------------------------------------------

            if not has_future_microbiology(
                micro_hours,
                prediction_hour
            ):
                continue

            micro_eligible += 1

            micro_hour = first_future_micro_hour(
                micro_hours,
                prediction_hour
            )

            # ------------------------------------------------
            # Find all observed future SOFA hours.
            # ------------------------------------------------

            start_idx = np.searchsorted(
                hours,
                prediction_hour + 1,
                side="left"
            )

            end_idx = np.searchsorted(
                hours,
                prediction_hour + HORIZON,
                side="right"
            )

            if start_idx >= end_idx:
                continue

            current_values = values[
                pred_idx
            ]

            max_delta = None

            max_delta_hour = None

            max_common_components = None

            # ------------------------------------------------
            # Calculate maximum future SOFA deterioration.
            # ------------------------------------------------

            for future_idx in range(
                start_idx,
                end_idx
            ):

                future_values = values[
                    future_idx
                ]

                valid = (
                    ~np.isnan(
                        current_values
                    )
                    &
                    ~np.isnan(
                        future_values
                    )
                )

                common_components = int(
                    valid.sum()
                )

                if (
                    common_components
                    < MIN_COMMON_COMPONENTS
                ):
                    continue

                current_sofa = float(
                    current_values[
                        valid
                    ].sum()
                )

                future_sofa = float(
                    future_values[
                        valid
                    ].sum()
                )

                delta = (
                    future_sofa
                    - current_sofa
                )

                if (
                    max_delta is None
                    or delta > max_delta
                ):

                    max_delta = delta

                    max_delta_hour = int(
                        hours[future_idx]
                    )

                    max_common_components = (
                        common_components
                    )

            # ------------------------------------------------
            # No valid SOFA comparison.
            # ------------------------------------------------

            if max_delta is None:
                continue

            sofa_eligible += 1

            # ------------------------------------------------
            # Final label.
            # ------------------------------------------------

            if max_delta >= SOFA_DELTA_THRESHOLD:

                label = 1

                positive_count += 1

                positive_stays.add(
                    stay_id
                )

            else:

                label = 0

                negative_count += 1

                negative_stays.add(
                    stay_id
                )

            output_rows.append(
                {
                    "stay_id": stay_id,
                    "icu_hour": prediction_hour,

                    "horizon_hours": HORIZON,

                    "min_common_components": (
                        MIN_COMMON_COMPONENTS
                    ),

                    "future_microbiology": 1,

                    "first_future_micro_hour": (
                        micro_hour
                    ),

                    "max_future_sofa_delta": (
                        max_delta
                    ),

                    "max_delta_future_hour": (
                        max_delta_hour
                    ),

                    "max_delta_common_components": (
                        max_common_components
                    ),

                    "sofa_delta_threshold": (
                        SOFA_DELTA_THRESHOLD
                    ),

                    "label": label,
                }
            )

    labels = pd.DataFrame(
        output_rows
    )

    print(
        f"Total prediction hours: "
        f"{total_prediction_hours:,}"
    )

    print(
        f"Microbiology-eligible hours: "
        f"{micro_eligible:,}"
    )

    print(
        f"SOFA-eligible labeled hours: "
        f"{sofa_eligible:,}"
    )

    print(
        f"Positive labels: "
        f"{positive_count:,}"
    )

    print(
        f"Negative labels: "
        f"{negative_count:,}"
    )

    if len(labels) > 0:

        positive_rate = (
            positive_count
            / len(labels)
            * 100
        )

    else:

        positive_rate = 0.0

    print(
        f"Positive rate: "
        f"{positive_rate:.4f}%"
    )

    print(
        f"Positive stays: "
        f"{len(positive_stays):,}"
    )

    print(
        f"Negative stays: "
        f"{len(negative_stays):,}"
    )

    return labels


# ============================================================
# VALIDATE LABELS
# ============================================================

def validate_labels(labels):

    print_header(
        "STEP 6: VALIDATE FINAL LABELS"
    )

    if labels.empty:

        raise RuntimeError(
            "No labels were generated."
        )

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    required_columns = [
        "stay_id",
        "icu_hour",
        "horizon_hours",
        "min_common_components",
        "future_microbiology",
        "first_future_micro_hour",
        "max_future_sofa_delta",
        "max_delta_future_hour",
        "max_delta_common_components",
        "sofa_delta_threshold",
        "label",
    ]

    missing = [
        col
        for col in required_columns
        if col not in labels.columns
    ]

    if missing:

        raise RuntimeError(
            f"Missing columns: {missing}"
        )

    # --------------------------------------------------------
    # Duplicate prediction positions
    # --------------------------------------------------------

    duplicate_count = labels.duplicated(
        subset=[
            "stay_id",
            "icu_hour",
        ]
    ).sum()

    print(
        f"Duplicate stay-hour labels: "
        f"{duplicate_count:,}"
    )

    if duplicate_count != 0:

        raise RuntimeError(
            "Duplicate stay-hour labels detected."
        )

    # --------------------------------------------------------
    # Check label values
    # --------------------------------------------------------

    invalid_labels = (
        ~labels["label"].isin([0, 1])
    ).sum()

    print(
        f"Invalid labels: "
        f"{invalid_labels:,}"
    )

    if invalid_labels != 0:

        raise RuntimeError(
            "Invalid label values detected."
        )

    # --------------------------------------------------------
    # Positive labels must have delta >= 2.
    # --------------------------------------------------------

    invalid_positive = (
        (
            labels["label"] == 1
        )
        &
        (
            labels[
                "max_future_sofa_delta"
            ] < SOFA_DELTA_THRESHOLD
        )
    ).sum()

    print(
        f"Invalid positive labels: "
        f"{invalid_positive:,}"
    )

    if invalid_positive != 0:

        raise RuntimeError(
            "Positive label validation failed."
        )

    # --------------------------------------------------------
    # Negative labels must have delta < 2.
    # --------------------------------------------------------

    invalid_negative = (
        (
            labels["label"] == 0
        )
        &
        (
            labels[
                "max_future_sofa_delta"
            ] >= SOFA_DELTA_THRESHOLD
        )
    ).sum()

    print(
        f"Invalid negative labels: "
        f"{invalid_negative:,}"
    )

    if invalid_negative != 0:

        raise RuntimeError(
            "Negative label validation failed."
        )

    # --------------------------------------------------------
    # All labels must have future microbiology.
    # --------------------------------------------------------

    invalid_micro = (
        labels[
            "future_microbiology"
        ] != 1
    ).sum()

    print(
        f"Invalid microbiology flags: "
        f"{invalid_micro:,}"
    )

    if invalid_micro != 0:

        raise RuntimeError(
            "Microbiology validation failed."
        )

    # --------------------------------------------------------
    # Future hour must actually be future.
    # --------------------------------------------------------

    invalid_future_hour = (
        labels[
            "max_delta_future_hour"
        ]
        <=
        labels[
            "icu_hour"
        ]
    ).sum()

    print(
        f"Invalid future SOFA hours: "
        f"{invalid_future_hour:,}"
    )

    if invalid_future_hour != 0:

        raise RuntimeError(
            "Future SOFA timing validation failed."
        )

    # --------------------------------------------------------
    # Horizon check.
    # --------------------------------------------------------

    outside_horizon = (
        (
            labels[
                "max_delta_future_hour"
            ]
            -
            labels[
                "icu_hour"
            ]
        )
        > HORIZON
    ).sum()

    print(
        f"SOFA observations outside horizon: "
        f"{outside_horizon:,}"
    )

    if outside_horizon != 0:

        raise RuntimeError(
            "Horizon validation failed."
        )

    # --------------------------------------------------------
    # Component count.
    # --------------------------------------------------------

    invalid_components = (
        labels[
            "max_delta_common_components"
        ]
        < MIN_COMMON_COMPONENTS
    ).sum()

    print(
        f"Rows below minimum component count: "
        f"{invalid_components:,}"
    )

    if invalid_components != 0:

        raise RuntimeError(
            "SOFA component validation failed."
        )

    print(
        "All validation checks passed."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print_header(
        "XSepsisNet - FINAL TEMPORAL SEPSIS LABEL"
    )

    print(
        "Final operational definition:"
    )

    print(
        "24-hour horizon + >=3 common SOFA "
        "components + future microbiology + "
        "maximum SOFA deterioration >=2"
    )

    sofa = load_sofa()

    micro = load_microbiology()

    micro_index = build_micro_index(
        micro
    )

    stay_data = prepare_stay_data(
        sofa
    )

    labels = build_labels(
        stay_data,
        micro_index
    )

    validate_labels(
        labels
    )

    # --------------------------------------------------------
    # Save final labels
    # --------------------------------------------------------

    print_header(
        "STEP 7: SAVE FINAL LABEL DATASET"
    )

    labels = labels.sort_values(
        [
            "stay_id",
            "icu_hour",
        ]
    ).reset_index(
        drop=True
    )

    labels.to_csv(
        OUTPUT_FILE,
        index=False
    )

    positive_count = int(
        (
            labels["label"] == 1
        ).sum()
    )

    negative_count = int(
        (
            labels["label"] == 0
        ).sum()
    )

    total_count = len(
        labels
    )

    positive_rate = (
        positive_count
        / total_count
        * 100
    )

    summary = {
        "label_definition": {
            "horizon_hours": HORIZON,
            "minimum_common_components": (
                MIN_COMMON_COMPONENTS
            ),
            "sofa_delta_threshold": (
                SOFA_DELTA_THRESHOLD
            ),
            "infection_evidence": (
                "Future microbiology evidence "
                "strictly after prediction hour "
                "and within 24 hours."
            ),
            "sofa_rule": (
                "Maximum future SOFA deterioration "
                "using at least 3 components observed "
                "at both prediction and future hour."
            ),
        },
        "dataset": {
            "rows": total_count,
            "positive": positive_count,
            "negative": negative_count,
            "positive_rate_percent": (
                positive_rate
            ),
            "icu_stays": int(
                labels["stay_id"].nunique()
            ),
        },
        "important_note": (
            "This is a project-specific temporal "
            "proxy label. It is not treated as "
            "exact Sepsis-3 onset."
        ),
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

    print(
        f"Saved labels:\n{OUTPUT_FILE}"
    )

    print(
        f"Saved summary:\n{SUMMARY_FILE}"
    )

    print_header(
        "FINAL LABEL SUMMARY"
    )

    print(
        f"Rows: {total_count:,}"
    )

    print(
        f"Positive: {positive_count:,}"
    )

    print(
        f"Negative: {negative_count:,}"
    )

    print(
        f"Positive rate: "
        f"{positive_rate:.4f}%"
    )

    print(
        f"ICU stays: "
        f"{labels['stay_id'].nunique():,}"
    )

    print_header(
        "FINAL TEMPORAL LABEL CONSTRUCTION COMPLETE"
    )


if __name__ == "__main__":
    main()