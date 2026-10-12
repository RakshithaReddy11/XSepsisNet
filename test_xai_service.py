
import numpy as np
from src.explainability.xai_service import XAIService

service = XAIService()

with np.load("data/processed/final_bilstm_test.npz") as data:
    sample = data["X"][0].copy()

result = service.explain(sample)

print("Raw model probability:", result["raw_probability"])
print("Top 5 clinical contributions:")
for item in result["clinical_contributions"][:5]:
    print(item)

assert len(result["clinical_contributions"]) == 39
assert len(result["hourly_contributions"]) == 468

print("XAI smoke test passed.")
