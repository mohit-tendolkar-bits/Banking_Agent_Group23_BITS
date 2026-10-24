"""Create the two fixed evaluation subsets from the BANKING77 test split.

eval_subset.csv    development set: 100 rows used by every experiment (E1 to E4, retrieval)
                   to compare approaches and choose the final system.
heldout_subset.csv held-out test set: 100 different rows, used once, for the final system only.

Run from the project root: uv run -m evaluation.make_eval_subset
"""

import pandas as pd

test = pd.read_csv("data/test.csv")
subset = test.sample(n=100, random_state=0).sort_index()
subset.to_csv("evaluation/eval_subset.csv", index=False)
heldout = test.drop(subset.index).sample(n=100, random_state=1).sort_index()
heldout.to_csv("evaluation/heldout_subset.csv", index=False)

print(f"test split: {len(test)} rows, {test['category'].nunique()} intents")
print(f"development subset: {len(subset)} rows, {subset['category'].nunique()} intents")
print(f"held-out subset: {len(heldout)} rows, {heldout['category'].nunique()} intents, overlap with development: {len(set(subset.index) & set(heldout.index))}")
