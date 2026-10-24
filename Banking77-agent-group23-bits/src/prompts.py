"""E2: compare four prompt styles on the same 100 rows, greedy decoding."""

import pandas as pd

from src.baseline import TASK, TRAIN, parse_intent
from src.llm import generate

ROLE = "You are a customer support assistant for an online bank."
CONSTRAINT = "Reply with the exact intent name from the list and nothing else. Always choose exactly one intent."
EXAMPLES = TRAIN.sample(n=5, random_state=0)
SHOTS = "".join(f"Message: {r.text}\nIntent: {r.category}\n\n" for r in EXAMPLES.itertuples())

STYLES = ["zero_shot", "role", "constraint", "few_shot", "few_shot_constraint"]


def build_prompt(message: str, style: str) -> list[dict]:
    query = f"Message: {message}\nIntent:"
    if style == "zero_shot":
        return [{"role": "user", "content": f"{TASK}\n\n{query}"}]
    if style == "role":
        return [{"role": "system", "content": ROLE}, {"role": "user", "content": f"{TASK}\n\n{query}"}]
    if style == "constraint":
        return [{"role": "user", "content": f"{TASK}\n{CONSTRAINT}\n\n{query}"}]
    if style == "few_shot":
        return [{"role": "user", "content": f"{TASK}\n\n{SHOTS}{query}"}]
    if style == "few_shot_constraint":
        return [{"role": "user", "content": f"{TASK}\n{CONSTRAINT}\n\n{SHOTS}{query}"}]
    raise ValueError(style)


if __name__ == "__main__":
    data = pd.read_csv("evaluation/eval_subset.csv")
    table = []
    for style in STYLES:
        result = data.copy()
        result["output"] = [generate(build_prompt(t, style), max_new_tokens=32) for t in data["text"]]
        result["predicted"] = [parse_intent(o) for o in result["output"]]
        result["correct"] = result["predicted"] == result["category"]
        result.to_csv(f"evaluation/results/results_{style}.csv", index=False)
        table.append(
            {
                "prompt": style,
                "accuracy": result["correct"].mean(),
                "no_intent": (result["predicted"] == "").sum(),
                "preamble": result["output"].str.match(r"^\s*The").sum(),
            }
        )
        print(f"\n== {style}: accuracy {table[-1]['accuracy']:.2f}")
        for _, row in result.head(3).iterrows():
            print(f"  {row['text']!r} -> {row['output']!r} (expected {row['category']})")

    print()
    print(pd.DataFrame(table).to_string(index=False))
