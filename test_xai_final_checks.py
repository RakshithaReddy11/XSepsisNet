
"""
XSepsisNet: Combined XAI Evaluation
Checks:
1. XAI output contract and validity
2. Perturbation-based faithfulness (exploratory)
3. Explanation stability
4. Saves a JSON evaluation report

This does not test live backend/frontend integration.
"""

import json
from pathlib import Path

import numpy as np
import torch

from src.explainability.xai_service import XAIService


DATA_FILE = Path("data/processed/final_bilstm_test.npz")
REPORT_FILE = Path("models/explainability/xai_final_checks.json")

N_FAITHFULNESS_SAMPLES = 10
N_STABILITY_SAMPLES = 5
N_RANDOM_TRIALS = 10
TOP_K = 5
NOISE_SCALE = 0.02
RANDOM_SEED = 42


def predict_probability(model, sequence):
    """Return raw sigmoid probability for a (6, 78) input."""
    x = torch.as_tensor(
        sequence[np.newaxis, ...],
        dtype=torch.float32,
    )

    model.eval()
    with torch.no_grad():
        logits = model(x).reshape(-1)[0]
        return float(torch.sigmoid(logits).item())


def validate_explanation(result, service):
    """Validate the expected XAI service output contract."""
    required_keys = {
        "model",
        "sequence_length",
        "input_features_per_hour",
        "raw_probability",
        "clinical_contributions",
        "hourly_contributions",
        "warning",
    }

    missing_keys = required_keys - set(result.keys())
    if missing_keys:
        raise ValueError(
            f"Missing output keys: {sorted(missing_keys)}"
        )

    probability = float(result["raw_probability"])
    if not np.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Raw probability is invalid.")

    if result["sequence_length"] != service.sequence_length:
        raise ValueError("Unexpected sequence length in output.")

    if result["input_features_per_hour"] != service.input_size:
        raise ValueError("Unexpected input feature count in output.")

    clinical = result["clinical_contributions"]
    hourly = result["hourly_contributions"]

    if len(clinical) != len(service.clinical_features):
        raise ValueError("Unexpected clinical contribution count.")

    expected_hourly = service.sequence_length * service.input_size
    if len(hourly) != expected_hourly:
        raise ValueError(
            f"Expected {expected_hourly} hourly contributions; "
            f"received {len(hourly)}."
        )

    for item in clinical:
        for key in ("feature", "total_shap", "absolute_shap",
                    "most_influential_hour"):
            if key not in item:
                raise ValueError(
                    f"Clinical contribution missing key: {key}"
                )

        if not np.isfinite(item["total_shap"]):
            raise ValueError("Non-finite clinical SHAP value.")

        if not np.isfinite(item["absolute_shap"]):
            raise ValueError("Non-finite absolute SHAP value.")

    for item in hourly:
        if not np.isfinite(item["shap_value"]):
            raise ValueError("Non-finite hourly SHAP value.")

        if not 1 <= item["hour"] <= service.sequence_length:
            raise ValueError("Invalid hour in SHAP output.")

    return True


def feature_scores(result, service):
    """Aggregate absolute SHAP values across hours per model input."""
    scores = np.zeros(service.input_size, dtype=np.float64)

    name_to_index = {
        name: index
        for index, name in enumerate(service.feature_names)
    }

    for item in result["hourly_contributions"]:
        name = item["feature"]
        if name not in name_to_index:
            raise ValueError(
                f"Unknown feature in SHAP output: {name}"
            )

        scores[name_to_index[name]] += abs(item["shap_value"])

    return scores


def rank_values(values):
    """Convert values into ranks without requiring SciPy."""
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)

    # Average ranks for tied values.
    unique_values, inverse, counts = np.unique(
        values, return_inverse=True, return_counts=True
    )
    for index, count in enumerate(counts):
        if count > 1:
            positions = np.where(inverse == index)[0]
            ranks[positions] = ranks[positions].mean()

    return ranks


def spearman_correlation(first, second):
    """Spearman rank correlation, implemented with NumPy."""
    rank_first = rank_values(first)
    rank_second = rank_values(second)

    if np.std(rank_first) == 0 or np.std(rank_second) == 0:
        return float("nan")

    return float(np.corrcoef(rank_first, rank_second)[0, 1])


def top_k_jaccard(first_scores, second_scores, k):
    """Compare overlap of the top-k ranked model input features."""
    k = min(k, len(first_scores))
    first = set(np.argsort(first_scores)[-k:])
    second = set(np.argsort(second_scores)[-k:])

    union = first | second
    if not union:
        return 1.0

    return len(first & second) / len(union)


def main():
    rng = np.random.default_rng(RANDOM_SEED)
    report = {
        "project": "XSepsisNet",
        "evaluation_type": "exploratory XAI checks",
        "checks": {},
        "limitations": [
            "Small evaluation sample; results are preliminary.",
            "Perturbations may create unrealistic clinical combinations.",
            "Stability test uses small noise on the first clinical input features.",
            "No live backend/frontend integration is tested.",
            "This is not clinical validation.",
        ],
    }

    print("=" * 60)
    print("XSEPSISNET - COMBINED XAI EVALUATION")
    print("=" * 60)

    service = XAIService()

    with np.load(DATA_FILE) as data:
        X = data["X"]

    expected_shape = (service.sequence_length, service.input_size)

    if X.ndim != 3 or tuple(X.shape[1:]) != expected_shape:
        raise ValueError(
            f"Expected data shape (samples, {expected_shape}); "
            f"received {X.shape}"
        )

    if len(X) == 0:
        raise ValueError("Test dataset is empty.")

    background = service.background.detach().cpu().numpy()
    background_mean = background.mean(axis=0)

    if background.shape[1:] != expected_shape:
        raise ValueError("Background dimensions do not match input.")

    # ---------------------------------------------------------
    # CHECK 1: Service output contract and valid SHAP values
    # ---------------------------------------------------------
    print("\n[1/3] Checking XAI output contract...")

    baseline_count = min(
        max(N_FAITHFULNESS_SAMPLES, N_STABILITY_SAMPLES),
        len(X),
    )

    baseline_results = []
    baseline_scores = []

    for index in range(baseline_count):
        sequence = np.asarray(X[index], dtype=np.float32).copy()
        result = service.explain(sequence)
        validate_explanation(result, service)

        baseline_results.append(result)
        baseline_scores.append(feature_scores(result, service))

    report["checks"]["output_contract"] = {
        "passed": True,
        "samples_checked": baseline_count,
        "clinical_features_per_explanation": len(
            service.clinical_features
        ),
        "hourly_contributions_per_explanation": (
            service.sequence_length * service.input_size
        ),
        "probabilities_finite_and_in_range": True,
        "shap_values_finite": True,
    }

    print(f"PASS: {baseline_count} explanations validated.")

    # ---------------------------------------------------------
    # CHECK 2: Top-SHAP vs random feature perturbations
    # ---------------------------------------------------------
    print("\n[2/3] Running perturbation faithfulness check...")

    top_changes = []
    random_changes = []
    n_faithfulness = min(N_FAITHFULNESS_SAMPLES, len(X))

    for index in range(n_faithfulness):
        sequence = np.asarray(X[index], dtype=np.float32).copy()
        scores = baseline_scores[index]

        top_indices = np.argsort(scores)[::-1][
            :min(TOP_K, service.input_size)
        ]

        original_probability = predict_probability(
            service.model, sequence
        )

        changed = sequence.copy()
        changed[:, top_indices] = background_mean[:, top_indices]

        changed_probability = predict_probability(
            service.model, changed
        )

        top_changes.append(
            abs(original_probability - changed_probability)
        )

        trial_changes = []

        for _ in range(N_RANDOM_TRIALS):
            random_indices = rng.choice(
                service.input_size,
                size=len(top_indices),
                replace=False,
            )

            random_input = sequence.copy()
            random_input[:, random_indices] = (
                background_mean[:, random_indices]
            )

            random_probability = predict_probability(
                service.model, random_input
            )

            trial_changes.append(
                abs(original_probability - random_probability)
            )

        random_changes.append(float(np.mean(trial_changes)))

    mean_top = float(np.mean(top_changes))
    mean_random = float(np.mean(random_changes))

    report["checks"]["perturbation_faithfulness"] = {
        "samples": n_faithfulness,
        "top_k_features": TOP_K,
        "random_trials_per_sample": N_RANDOM_TRIALS,
        "mean_top_shap_probability_change": mean_top,
        "mean_random_probability_change": mean_random,
        "top_shap_change_larger": mean_top > mean_random,
        "relative_ratio": (
            mean_top / mean_random if mean_random > 0 else None
        ),
        "interpretation": (
            "Top-SHAP perturbations changed the output more on average."
            if mean_top > mean_random
            else "Top-SHAP perturbations did not change the output "
                 "more on average."
        ),
    }

    print(f"Mean top-SHAP change: {mean_top:.6f}")
    print(f"Mean random change:   {mean_random:.6f}")
    print(f"Top-SHAP larger:      {mean_top > mean_random}")

    # ---------------------------------------------------------
    # CHECK 3: Stability under small clinical-input changes
    # ---------------------------------------------------------
    print("\n[3/3] Running explanation stability check...")

    stability_spearman = []
    stability_jaccard = []
    n_stability = min(N_STABILITY_SAMPLES, len(X))

    n_clinical = len(service.clinical_features)

    if n_clinical > service.input_size:
        raise ValueError(
            "Clinical feature count exceeds model input size."
        )

    # Estimate noise scale from clinical input variation in
    # the SHAP background. Missingness indicators are left alone.
    clinical_std = np.std(background[:, :, :n_clinical], axis=0)
    noise_std = NOISE_SCALE * clinical_std

    for index in range(n_stability):
        sequence = np.asarray(X[index], dtype=np.float32).copy()
        original_scores = baseline_scores[index]

        noisy_sequence = sequence.copy()
        noise = rng.normal(
            loc=0.0,
            scale=noise_std,
            size=(service.sequence_length, n_clinical),
        ).astype(np.float32)

        noisy_sequence[:, :n_clinical] += noise

        noisy_result = service.explain(noisy_sequence)
        validate_explanation(noisy_result, service)
        noisy_scores = feature_scores(noisy_result, service)

        correlation = spearman_correlation(
            original_scores, noisy_scores
        )
        overlap = top_k_jaccard(
            original_scores, noisy_scores, TOP_K
        )

        stability_spearman.append(correlation)
        stability_jaccard.append(overlap)

        print(
            f"Sample {index + 1}: "
            f"rank correlation={correlation:.4f}, "
            f"top-{TOP_K} overlap={overlap:.4f}"
        )

    valid_correlations = [
        value for value in stability_spearman
        if np.isfinite(value)
    ]

    report["checks"]["explanation_stability"] = {
        "samples": n_stability,
        "clinical_input_noise_scale": NOISE_SCALE,
        "mean_spearman_rank_correlation": (
            float(np.mean(valid_correlations))
            if valid_correlations else None
        ),
        "mean_top_k_jaccard_overlap": float(
            np.mean(stability_jaccard)
        ),
        "note": (
            "Higher correlation and overlap indicate more stable "
            "rankings under the tested small input perturbations."
        ),
    }

    # ---------------------------------------------------------
    # Save report
    # ---------------------------------------------------------
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with REPORT_FILE.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, allow_nan=False)

    print("\n" + "=" * 60)
    print("EVALUATION FINISHED")
    print("=" * 60)
    print(f"JSON report saved to: {REPORT_FILE}")
    print("Output contract: PASS")
    print(
        "Perturbation check:",
        report["checks"]["perturbation_faithfulness"]["interpretation"],
    )

    correlation = report["checks"]["explanation_stability"][
        "mean_spearman_rank_correlation"
    ]
    overlap = report["checks"]["explanation_stability"][
        "mean_top_k_jaccard_overlap"
    ]

    print(f"Mean stability rank correlation: {correlation}")
    print(f"Mean top-{TOP_K} overlap: {overlap}")
    print(
        "\nThese are exploratory research checks, not an XAI "
        "accuracy score or clinical validation."
    )
    print(
        "Live backend/frontend integration must be tested separately."
    )


if __name__ == "__main__":
    main()