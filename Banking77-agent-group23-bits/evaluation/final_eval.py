"""Final evaluation: run the final system's classifier (main.classify_intent) once on the
held-out subset. No approach was chosen on these rows.

Run from the project root: uv run -m evaluation.final_eval"""

import pandas as pd

from main import classify_intent

data = pd.read_csv("evaluation/heldout_subset.csv")
data["predicted"] = [classify_intent(t).get("intent", "") for t in data["text"]]
data["correct"] = data["predicted"] == data["category"]
data.to_csv("evaluation/results/results_heldout.csv", index=False)

print(f"held-out accuracy (final system): {data['correct'].mean():.2f}")
print(f"no valid intent: {(data['predicted'] == '').sum()}")
