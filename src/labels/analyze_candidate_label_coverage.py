from pathlib import Path
import json
import numpy as np
import pandas as pd


# ============================================================
# XSepsisNet
# CANDIDATE TEMPORAL LABEL COVERAGE ANALYSIS
#
# Candidates:
#   6h  >=3
#   6h  >=4
#   12h >=3
#   12h >=4
#   24h >=3
#   24h >=4
#
# Label:
#   Future microbiology evidence within horizon
#   +
#   maximum future SOFA deterioration >= 2
#
# We calculate labels for every hourly prediction position
# where enough SOFA information exists.
#
# This is a coverage analysis only.
# It does NOT create the final BiLSTM dataset.
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
    / "candidate_label_coverage.csv"
)

OUTPUT_JSON = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "candidate_label_coverage_summary.json"
)


CANDIDATES = [
    (6, 3),
    (6, 4),
    (12, 3),
    (12, 4),
    (24, 3),
    (24, 4),
]


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
# BUILD MICRO INDEX
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
# PREPARE STAY DATA
# ============================================================

def prepare_stay_data(sofa):

    print_header(
        "STEP 4: PREPARE HOURLY SOFA DATA"
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
# MICROBIOLOGY CHECK
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
# ANALYZE ONE CANDIDATE
# ============================================================

def analyze_candidate(
    stay_data,
    micro_index,
    horizon,
    min_components
):

    total_prediction_hours = 0

    micro_eligible_hours = 0

    sofa_eligible_hours = 0

    positive_hours = 0

    negative_hours = 0

    positive_stays = set()

    negative_stays = set()

    sofa_eligible_stays = set()

    micro_eligible_stays = set()

    # --------------------------------------------------------
    # Process ICU stays
    # --------------------------------------------------------

    for stay_id, data in stay_data.items():

        hours = data["hours"]

        values = data["values"]

        micro_hours = micro_index.get(
            stay_id
        )

        # Every observed hourly row is a potential
        # prediction position.
        total_prediction_hours += len(
            hours
        )

        if micro_hours is None:
            continue

        # ----------------------------------------------------
        # Prediction hours with future infection evidence
        # ----------------------------------------------------

        for pred_idx, prediction_hour in enumerate(
            hours
        ):

            prediction_hour = int(
                prediction_hour
            )

            if not has_future_microbiology(
                micro_hours,
                prediction_hour,
                horizon
            ):
                continue

            micro_eligible_hours += 1

            micro_eligible_stays.add(
                stay_id
            )

            # ------------------------------------------------
            # Find ALL future SOFA observations.
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

            max_delta = None

            # ------------------------------------------------
            # Find maximum SOFA deterioration.
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

                common = int(
                    valid.sum()
                )

                if common < min_components:
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
            # No adequate SOFA comparison.
            # ------------------------------------------------

            if max_delta is None:
                continue

            sofa_eligible_hours += 1

            sofa_eligible_stays.add(
                stay_id
            )

            # ------------------------------------------------
            # Final candidate label.
            # ------------------------------------------------

            if max_delta >= 2:

                positive_hours += 1

                positive_stays.add(
                    stay_id
                )

            else:

                negative_hours += 1

                negative_stays.add(
                    stay_id
                )

    total_labeled = (
        positive_hours
        + negative_hours
    )

    if total_labeled > 0:

        positive_rate = (
            positive_hours
            / total_labeled
            * 100.0
        )

    else:

        positive_rate = 0.0

    # --------------------------------------------------------
    # Positive/negative stay overlap.
    # --------------------------------------------------------

    overlap_stays = (
        positive_stays
        &
        negative_stays
    )

    return {
        "horizon_hours": horizon,
        "minimum_common_components": min_components,

        "total_prediction_hours": (
            total_prediction_hours
        ),

        "microbiology_eligible_hours": (
            micro_eligible_hours
        ),

        "sofa_eligible_hours": (
            sofa_eligible_hours
        ),

        "positive_hours": (
            positive_hours
        ),

        "negative_hours": (
            negative_hours
        ),

        "positive_rate_percent": (
            positive_rate
        ),

        "microbiology_eligible_stays": (
            len(micro_eligible_stays)
        ),

        "sofa_eligible_stays": (
            len(sofa_eligible_stays)
        ),

        "positive_stays": (
            len(positive_stays)
        ),

        "negative_stays": (
            len(negative_stays)
        ),

        "positive_negative_overlap_stays": (
            len(overlap_stays)
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print_header(
        "XSepsisNet - CANDIDATE LABEL COVERAGE"
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
        "STEP 5: ANALYZE CANDIDATE LABELS"
    )

    results = []

    for horizon, min_components in CANDIDATES:

        print()
        print(
            f"Analyzing: "
            f"{horizon}h / "
            f">={min_components} components"
        )

        result = analyze_candidate(
            stay_data,
            micro_index,
            horizon,
            min_components
        )

        results.append(
            result
        )

        print(
            f"  total prediction hours: "
            f"{result['total_prediction_hours']:,}"
        )

        print(
            f"  microbiology eligible: "
            f"{result['microbiology_eligible_hours']:,}"
        )

        print(
            f"  SOFA eligible: "
            f"{result['sofa_eligible_hours']:,}"
        )

        print(
            f"  positive: "
            f"{result['positive_hours']:,}"
        )

        print(
            f"  negative: "
            f"{result['negative_hours']:,}"
        )

        print(
            f"  positive rate: "
            f"{result['positive_rate_percent']:.4f}%"
        )

        print(
            f"  positive stays: "
            f"{result['positive_stays']:,}"
        )

    # ========================================================
    # SAVE
    # ========================================================

    print_header(
        "STEP 6: SAVE COVERAGE RESULTS"
    )

    result_df = pd.DataFrame(
        results
    )

    result_df.to_csv(
        OUTPUT_CSV,
        index=False
    )

    summary = {
        "candidates": [
            {
                "horizon_hours": h,
                "minimum_common_components": c
            }
            for h, c in CANDIDATES
        ],
        "label_rule": (
            "Future microbiology evidence "
            "within horizon AND maximum "
            "future SOFA deterioration >= 2."
        ),
        "note": (
            "Candidate coverage analysis only. "
            "Final temporal label has not yet "
            "been selected."
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
        "FINAL COVERAGE TABLE"
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
        "COVERAGE ANALYSIS COMPLETE"
    )


if __name__ == "__main__":
    main()