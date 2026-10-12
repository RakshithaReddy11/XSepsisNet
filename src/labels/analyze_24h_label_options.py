from pathlib import Path
import json
import pandas as pd


# ============================================================
# XSepsisNet
# Analyze ONLY 24-hour temporal label options
#
# We already completed:
#   6h  >=3, >=4, >=5, >=6
#   12h >=3, >=4, >=5, >=6
#   24h >=3
#
# This script calculates only:
#   24h >=4
#   24h >=5
#   24h >=6
#
# Label logic:
#   Future microbiology evidence within 24h
#   AND SOFA deterioration >= 2
#   AND at least N common SOFA components between
#       prediction hour and future hour.
#
# IMPORTANT:
# This is a temporal proxy label for project development.
# It is NOT being claimed as exact Sepsis-3 onset.
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

OUTPUT_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "temporal_24h_label_options.csv"
)

OUTPUT_JSON = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "temporal_24h_label_options_summary.json"
)


HORIZON = 24

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


def load_sofa():
    print_header("STEP 1: LOAD SOFA")

    required = [
        "stay_id",
        "icu_hour",
        *SOFA_COMPONENTS,
    ]

    df = pd.read_csv(
        SOFA_FILE,
        usecols=required
    )

    df["stay_id"] = pd.to_numeric(
        df["stay_id"],
        errors="coerce"
    ).astype("Int64")

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
        subset=["stay_id", "icu_hour"]
    ).copy()

    df["stay_id"] = df["stay_id"].astype("int64")
    df["icu_hour"] = df["icu_hour"].astype("int64")

    df = df.sort_values(
        ["stay_id", "icu_hour"]
    ).reset_index(drop=True)

    print(f"Rows: {len(df):,}")
    print(f"ICU stays: {df['stay_id'].nunique():,}")

    return df


def load_microbiology():
    print_header("STEP 2: LOAD MICROBIOLOGY")

    required = [
        "stay_id",
        "icu_hour",
    ]

    df = pd.read_csv(
        MICRO_FILE,
        usecols=required
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
        subset=["stay_id", "icu_hour"]
    ).copy()

    df["stay_id"] = df["stay_id"].astype("int64")
    df["icu_hour"] = df["icu_hour"].astype("int64")

    # We only need whether microbiology evidence exists
    # at a particular stay/hour.
    micro_hours = (
        df[["stay_id", "icu_hour"]]
        .drop_duplicates()
        .sort_values(["stay_id", "icu_hour"])
        .reset_index(drop=True)
    )

    print(
        f"Unique microbiology stay-hours: "
        f"{len(micro_hours):,}"
    )

    print(
        f"Microbiology stays indexed: "
        f"{micro_hours['stay_id'].nunique():,}"
    )

    return micro_hours


def build_micro_index(micro):
    """
    For each ICU stay, store sorted microbiology hours.
    """

    result = {}

    for stay_id, group in micro.groupby(
        "stay_id",
        sort=False
    ):
        result[int(stay_id)] = (
            group["icu_hour"]
            .drop_duplicates()
            .to_numpy(dtype="int32")
        )

    return result


def get_future_microbiology_hours(
    micro_hours,
    prediction_hour,
    horizon
):
    """
    Return whether microbiology evidence exists
    strictly after prediction_hour and within horizon.

    Future interval:
        prediction_hour + 1
        ...
        prediction_hour + horizon
    """

    if micro_hours is None:
        return False

    if len(micro_hours) == 0:
        return False

    lower = prediction_hour + 1
    upper = prediction_hour + horizon

    # Binary-search instead of repeatedly filtering a DataFrame.
    import numpy as np

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


def calculate_option(
    sofa,
    micro_index,
    min_components
):
    """
    Calculate one 24-hour label option.
    """

    positives = 0
    negatives = 0
    comparisons = 0

    # Process stay by stay.
    for stay_id, group in sofa.groupby(
        "stay_id",
        sort=False
    ):
        stay_id = int(stay_id)

        micro_hours = micro_index.get(
            stay_id
        )

        if micro_hours is None:
            continue

        # Convert relevant columns to NumPy arrays.
        hours = group["icu_hour"].to_numpy(
            dtype="int32"
        )

        component_values = (
            group[SOFA_COMPONENTS]
            .to_numpy(
                dtype="float32"
            )
        )

        hour_to_position = {
            int(hour): idx
            for idx, hour in enumerate(hours)
        }

        for pred_idx, prediction_hour in enumerate(hours):

            # We need future microbiology evidence.
            if not get_future_microbiology_hours(
                micro_hours,
                int(prediction_hour),
                HORIZON
            ):
                continue

            future_hour = None

            # The temporal target looks for SOFA deterioration
            # at a future observed SOFA hour within 24h.
            #
            # Search the observed hours after prediction hour.
            start_future = prediction_hour + 1
            end_future = prediction_hour + HORIZON

            for candidate_hour in range(
                start_future,
                end_future + 1
            ):
                if candidate_hour in hour_to_position:
                    candidate_idx = hour_to_position[
                        candidate_hour
                    ]

                    # Check common SOFA components.
                    pred_values = component_values[
                        pred_idx
                    ]

                    future_values = component_values[
                        candidate_idx
                    ]

                    valid = (
                        ~pd.isna(pred_values)
                        &
                        ~pd.isna(future_values)
                    )

                    common = int(
                        valid.sum()
                    )

                    if common < min_components:
                        continue

                    comparisons += 1

                    pred_sofa = (
                        pred_values[valid]
                        .sum()
                    )

                    future_sofa = (
                        future_values[valid]
                        .sum()
                    )

                    delta = (
                        future_sofa
                        - pred_sofa
                    )

                    if delta >= 2:
                        positives += 1

                    else:
                        negatives += 1

                    # One future comparison per prediction
                    # hour is enough for this option.
                    break

    total = positives + negatives

    positive_rate = (
        positives / total * 100
        if total > 0
        else 0.0
    )

    return {
        "horizon_hours": HORIZON,
        "minimum_common_components": min_components,
        "positive": positives,
        "negative": negatives,
        "total": total,
        "positive_rate_percent": positive_rate,
        "comparisons": comparisons,
    }


def main():

    print_header(
        "XSepsisNet - 24-HOUR LABEL OPTIONS"
    )

    print(
        "This script evaluates ONLY 24-hour "
        "options >=4, >=5 and >=6."
    )

    sofa = load_sofa()
    micro = load_microbiology()

    print_header(
        "STEP 3: BUILD FAST MICROBIOLOGY INDEX"
    )

    micro_index = build_micro_index(
        micro
    )

    print(
        f"Indexed microbiology stays: "
        f"{len(micro_index):,}"
    )

    results = []

    print_header(
        "STEP 4: ANALYZE 24-HOUR OPTIONS"
    )

    for min_components in [4, 5, 6]:

        print(
            f"\nAnalyzing horizon = 24 hours"
        )

        print(
            f"  minimum common components = "
            f"{min_components}"
        )

        result = calculate_option(
            sofa,
            micro_index,
            min_components
        )

        results.append(result)

        print(
            f"  positive={result['positive']:,}, "
            f"negative={result['negative']:,}, "
            f"positive_rate="
            f"{result['positive_rate_percent']:.4f}%"
        )

        print(
            f"  comparisons="
            f"{result['comparisons']:,}"
        )

    print_header(
        "STEP 5: SAVE RESULTS"
    )

    result_df = pd.DataFrame(
        results
    )

    result_df.to_csv(
        OUTPUT_CSV,
        index=False
    )

    summary = {
        "horizon_hours": HORIZON,
        "options": results,
        "note": (
            "Temporal proxy label based on "
            "future microbiology evidence and "
            "SOFA deterioration. This is not "
            "treated as exact sepsis onset."
        ),
    }

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            summary,
            f,
            indent=2
        )

    print(
        f"Saved CSV: {OUTPUT_CSV}"
    )

    print(
        f"Saved JSON: {OUTPUT_JSON}"
    )

    print_header(
        "24-HOUR ANALYSIS COMPLETE"
    )

    print(
        result_df.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()