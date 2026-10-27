"""Create reproducible, intent-covered development and held-out subsets.

Each 100-row subset includes every BANKING77 intent. The two subsets do not
overlap; the development set is for experiments and the held-out set is for
the final evaluation only.

Run from the project root: uv run -m evaluation.make_eval_subset
"""

import pandas as pd

test = pd.read_csv("data/test.csv")
intent_count = test["category"].nunique()
subset_size = 100
if intent_count > subset_size:
    raise ValueError(
        f"Cannot cover {intent_count} intents in a {subset_size}-row subset."
    )

# Reserve a distinct example for every intent in each split before filling
# the remaining rows from examples not already selected.
required = test.groupby("category", group_keys=False).sample(n=2, random_state=0)
required = required.sort_values("category")
development_core = required.groupby("category", group_keys=False).head(1)
heldout_core = required.groupby("category", group_keys=False).tail(1)
reserved_indices = development_core.index.union(heldout_core.index)

extra_count = subset_size - intent_count
remaining = test.drop(index=reserved_indices)
extras = remaining.sample(n=extra_count * 2, random_state=1)
subset = pd.concat([development_core, extras.iloc[:extra_count]]).sort_index()
heldout = pd.concat([heldout_core, extras.iloc[extra_count:]]).sort_index()

if len(subset) != subset_size or subset["category"].nunique() != intent_count:
    raise RuntimeError("Development subset does not meet its size/intent coverage requirements.")
if len(heldout) != subset_size or heldout["category"].nunique() != intent_count:
    raise RuntimeError("Held-out subset does not meet its size/intent coverage requirements.")
if not set(subset.index).isdisjoint(heldout.index):
    raise RuntimeError("Development and held-out subsets overlap.")

subset.to_csv("evaluation/eval_subset.csv", index=False)
heldout.to_csv("evaluation/heldout_subset.csv", index=False)

print(f"test split: {len(test)} rows, {test['category'].nunique()} intents")
print(f"development subset: {len(subset)} rows, {subset['category'].nunique()} intents")
print(
    f"held-out subset: {len(heldout)} rows, {heldout['category'].nunique()} intents, "
    f"overlap with development: {len(set(subset.index) & set(heldout.index))}"
)
