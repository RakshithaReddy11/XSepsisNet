from pathlib import Path
import json
import numpy as np
import pandas as pd


# ============================================================
# XSepsisNet
# TEMPORAL SEPSIS LABEL FEASIBILITY ANALYSIS
#
# Label logic:
#
# Current prediction hour
#        |
#        +--> future microbiology within horizon
#        |
#        +--> ALL valid future SOFA observations
#        |
#        +--> maximum SOFA deterioration
#        |
#        +--> delta SOFA >= 2 => positive
#
# Horizons:
#   6h, 12h, 24h
#
# Common SOFA components:
#   >=3, >=4, >=5, >=6
#
# IMPORTANT:
# This is a project-specific temporal proxy.
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
    / "temporal_label_options_max_delta.csv"
)

OUTPUT_JSON = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "temporal_label_options_max_delta_summary.json"
)

HORIZONS = [6, 12, 24]

MIN_COMPONENT_OPTIONS = [3, 4, 5, 6]

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

    print_header("STEP 1: LOAD SOFA")

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
            "icu_hour"
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
            "icu_hour"
        ]
    ).reset_index(
        drop=True
    )

    print(
        f"Rows: {len(df):,}"
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

    columns = [
        "stay_id",
        "icu_hour",
    ]

    df = pd.read_csv(
        MICRO_FILE,
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

    df = df.dropna(
        subset=[
            "stay_id",
            "icu_hour"
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
                "icu_hour"
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "stay_id",
                "icu_hour"
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
# BUILD MICROBIOLOGY INDEX
# ============================================================

def build_micro_index(micro):

    print_header(
        "STEP 3: INDEX INFECTION EVIDENCE"
    )

    micro_index = {}

    for stay_id, group in micro.groupby(
        "stay_id",
        sort=False
    ):

        hours = (
            group["icu_hour"]
            .to_numpy(
                dtype=np.int32
            )
        )

        micro_index[int(stay_id)] = hours

    print(
        f"Microbiology stays indexed: "
        f"{len(micro_index):,}"
    )

    return micro_index


# ============================================================
# PREPARE SOFA STAYS
# ============================================================

def prepare_stay_data(sofa):

    print_header(
        "STEP 4: PREPARE TEMPORAL SOFA DATA"
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
        f"SOFA stays indexed: "
        f"{len(stay_data):,}"
    )

    return stay_data


# ============================================================
# FUTURE MICROBIOLOGY
# ============================================================

def has_future_microbiology(
    micro_hours,
    prediction_hour,
    horizon
):

    if micro_hours is None:
        return False

    lower = prediction_hour + 1
    upper = prediction_hour + horizon

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
# ANALYZE ONE OPTION
# ============================================================

def analyze_option(
    stay_data,
    micro_index,
    horizon,
    min_components
):

    positive = 0
    negative = 0
    comparisons = 0

    # --------------------------------------------------------
    # Process one ICU stay at a time.
    # --------------------------------------------------------

    for stay_id, data in stay_data.items():

        micro_hours = micro_index.get(
            stay_id
        )

        if micro_hours is None:
            continue

        hours = data["hours"]
        values = data["values"]

        if len(hours) < 2:
            continue

        # ----------------------------------------------------
        # Process every observed prediction hour.
        # ----------------------------------------------------

        for pred_idx, prediction_hour in enumerate(
            hours
        ):

            prediction_hour = int(
                prediction_hour
            )

            # ------------------------------------------------
            # Require future microbiology evidence.
            # ------------------------------------------------

            if not has_future_microbiology(
                micro_hours,
                prediction_hour,
                horizon
            ):
                continue

            # ------------------------------------------------
            # Find ALL observed SOFA hours inside:
            #
            # prediction_hour + 1
            # through
            # prediction_hour + horizon
            # ------------------------------------------------

            start_idx = np.searchsorted(
                hours,
                prediction_hour + 1,
                side="left"
            )

            end_idx = np.searchsorted(
                hours,
                prediction_hour + horizon,
                side="right"
            )

            if start_idx >= end_idx:
                continue

            current_values = values[
                pred_idx
            ]

            # ------------------------------------------------
            # Check every future SOFA observation.
            # ------------------------------------------------

            max_delta = None

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
                    < min_components
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

            # ------------------------------------------------
            # No valid SOFA comparison.
            # ------------------------------------------------

            if max_delta is None:
                continue

            comparisons += 1

            # ------------------------------------------------
            # Temporal deterioration criterion.
            # ------------------------------------------------

            if max_delta >= 2:
                positive += 1

            else:
                negative += 1

    total = (
        positive
        + negative
    )

    if total > 0:

        positive_rate = (
            positive
            / total
            * 100.0
        )

    else:

        positive_rate = 0.0

    return {
        "horizon_hours": horizon,
        "minimum_common_components": (
            min_components
        ),
        "positive": positive,
        "negative": negative,
        "total": total,
        "positive_rate_percent": (
            positive_rate
        ),
        "comparisons": comparisons,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print_header(
        "XSepsisNet - MAX FUTURE SOFA DELTA ANALYSIS"
    )

    print(
        "Label rule:"
    )

    print(
        "Future microbiology + maximum "
        "SOFA deterioration >= 2"
    )

    sofa = load_sofa()

    micro = load_microbiology()

    micro_index = build_micro_index(
        micro
    )

    stay_data = prepare_stay_data(
        sofa
    )

    print_header(
        "STEP 5: ANALYZE LABEL OPTIONS"
    )

    results = []

    for horizon in HORIZONS:

        print()
        print(
            f"Analyzing horizon = "
            f"{horizon} hours"
        )

        for min_components in (
            MIN_COMPONENT_OPTIONS
        ):

            result = analyze_option(
                stay_data,
                micro_index,
                horizon,
                min_components
            )

            results.append(
                result
            )

            print(
                f"  components >= "
                f"{min_components}: "
                f"positive="
                f"{result['positive']:,}, "
                f"negative="
                f"{result['negative']:,}, "
                f"positive_rate="
                f"{result['positive_rate_percent']:.4f}%"
            )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    print_header(
        "STEP 6: SAVE RESULTS"
    )

    result_df = pd.DataFrame(
        results
    )

    result_df.to_csv(
        OUTPUT_CSV,
        index=False
    )

    summary = {
        "horizons_hours": HORIZONS,
        "minimum_common_components": (
            MIN_COMPONENT_OPTIONS
        ),
        "sofa_components": SOFA_COMPONENTS,
        "label_logic": {
            "infection_evidence": (
                "Future microbiology evidence "
                "strictly after prediction hour "
                "and within selected horizon."
            ),
            "future_sofa": (
                "All observed SOFA hours within "
                "the selected future horizon."
            ),
            "sofa_deterioration": (
                "Maximum future SOFA minus "
                "current SOFA."
            ),
            "positive_condition": (
                "Maximum SOFA deterioration >= 2."
            ),
        },
        "note": (
            "This is a project-specific temporal "
            "proxy label and is not treated as "
            "exact Sepsis-3 onset."
        ),
        "results": results,
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
        f"Saved CSV:\n{OUTPUT_CSV}"
    )

    print(
        f"Saved JSON:\n{OUTPUT_JSON}"
    )

    print_header(
        "FINAL LABEL OPTION TABLE"
    )

    display_df = result_df.copy()

    display_df[
        "positive_rate_percent"
    ] = display_df[
        "positive_rate_percent"
    ].round(4)

    print(
        display_df.to_string(
            index=False
        )
    )

    print_header(
        "ANALYSIS COMPLETE"
    )


if __name__ == "__main__":
    main()