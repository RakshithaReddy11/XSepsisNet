
"""Exploratory SHAP faithfulness check for XSepsisNet."""

import numpy as np
import torch

from src.explainability.xai_service import XAIService


DATA_FILE = "data/processed/final_bilstm_test.npz"
N_SAMPLES = 10
N_RANDOM_TRIALS = 20
TOP_K = 5
RANDOM_SEED = 42


def predict_probability(model, sequence):
    """Predict raw sigmoid probability for one (6, 78) sequence."""
    x = torch.as_tensor(
        sequence[np.newaxis, ...],
        dtype=torch.float32,
    )

    model.eval()
    with torch.no_grad():
        logits = model(x).reshape(-1)[0]
        return float(torch.sigmoid(logits).item())


def main():
    rng = np.random.default_rng(RANDOM_SEED)
    service = XAIService()

    with np.load(DATA_FILE) as data:
        X = data["X"]

    expected_shape = (
        service.sequence_length,
        service.input_size,
    )

    if X.ndim != 3 or X.shape[1:] != expected_shape:
        raise ValueError(
            f"Expected X shape (samples, {expected_shape}), "
            f"got {X.shape}"
        )

    if len(X) == 0:
        raise ValueError("Test dataset is empty.")

    if service.background.ndim != 3:
        raise ValueError("SHAP background must be a 3D tensor.")

    if tuple(service.background.shape[1:]) != expected_shape:
        raise ValueError(
            "SHAP background dimensions do not match model input."
        )

    n_samples = min(N_SAMPLES, len(X))
    background_mean = (
        service.background.detach().cpu().numpy().mean(axis=0)
    )

    top_changes = []
    random_changes = []

    print("XSepsisNet SHAP reliability check")
    print(f"Samples: {n_samples}")
    print(f"Top features: {TOP_K}")
    print(f"Random trials per sample: {N_RANDOM_TRIALS}")
    print("-" * 55)

    for sample_index in range(n_samples):
        sequence = np.asarray(
            X[sample_index],
            dtype=np.float32,
        ).copy()

        explanation = service.explain(sequence)

        # Sum absolute SHAP contributions across six hours.
        feature_scores = np.zeros(
            service.input_size,
            dtype=np.float64,
        )

        for item in explanation["hourly_contributions"]:
            if item["hour"] < 1 or item["hour"] > service.sequence_length:
                raise ValueError("Unexpected SHAP hour index.")

            feature_index = service.feature_names.index(
                item["feature"]
            )
            feature_scores[feature_index] += abs(
                item["shap_value"]
            )

        top_indices = np.argsort(feature_scores)[::-1][
            :min(TOP_K, service.input_size)
        ]

        original_probability = predict_probability(
            service.model, sequence
        )

        # Replace selected input features at all six hours
        # with their corresponding background means.
        top_perturbed = sequence.copy()
        top_perturbed[:, top_indices] = background_mean[
            :, top_indices
        ]

        top_probability = predict_probability(
            service.model, top_perturbed
        )
        top_change = abs(
            original_probability - top_probability
        )
        top_changes.append(top_change)

        trial_changes = []

        for _ in range(N_RANDOM_TRIALS):
            random_indices = rng.choice(
                service.input_size,
                size=len(top_indices),
                replace=False,
            )

            random_perturbed = sequence.copy()
            random_perturbed[:, random_indices] = (
                background_mean[:, random_indices]
            )

            random_probability = predict_probability(
                service.model, random_perturbed
            )

            trial_changes.append(
                abs(original_probability - random_probability)
            )

        mean_random_change = float(np.mean(trial_changes))
        random_changes.append(mean_random_change)

        print(
            f"Sample {sample_index + 1}: "
            f"original={original_probability:.4f}, "
            f"top-SHAP change={top_change:.4f}, "
            f"random mean change={mean_random_change:.4f}"
        )

    mean_top = float(np.mean(top_changes))
    mean_random = float(np.mean(random_changes))

    print("-" * 55)
    print(f"Mean top-SHAP change: {mean_top:.6f}")
    print(f"Mean random change:   {mean_random:.6f}")

    if mean_top > mean_random:
        print("Result: top-SHAP perturbations caused a larger")
        print("average output change in this exploratory test.")
    else:
        print("Result: top-SHAP perturbations did not cause a")
        print("larger average output change in this test.")

    print(
        "Note: This is an exploratory faithfulness check, "
        "not an XAI accuracy score or clinical validation."
    )


if __name__ == "__main__":
    main()