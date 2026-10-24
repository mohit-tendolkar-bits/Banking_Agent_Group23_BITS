"""E3: compare decoding settings with the best prompt (few_shot_constraint)."""

import pandas as pd
import torch

from src.baseline import parse_intent
from src.llm import generate
from src.prompts import build_prompt

STYLE = "few_shot_constraint"
SETTINGS = {
    "greedy": {"do_sample": False},
    "temperature_0.7": {"do_sample": True, "temperature": 0.7},
    "temperature_1.5": {"do_sample": True, "temperature": 1.5},
    "top_k_50": {"do_sample": True, "temperature": 1.0, "top_k": 50},
    "top_p_0.9": {"do_sample": True, "temperature": 1.0, "top_k": 0, "top_p": 0.9},
}

if __name__ == "__main__":
    data = pd.read_csv("evaluation/eval_subset.csv")
    table = []
    for name, decoding in SETTINGS.items():
        torch.manual_seed(0)
        result = data.copy()
        result["output"] = [generate(build_prompt(t, STYLE), max_new_tokens=32, **decoding) for t in data["text"]]
        result["predicted"] = [parse_intent(o) for o in result["output"]]
        result["correct"] = result["predicted"] == result["category"]
        result.to_csv(f"evaluation/results/results_decoding_{name}.csv", index=False)
        table.append(
            {
                "decoding": name,
                "accuracy": result["correct"].mean(),
                "no_intent": (result["predicted"] == "").sum(),
                "mean_output_chars": result["output"].str.len().mean().round(1),
            }
        )
        print(f"\n== {name}: accuracy {table[-1]['accuracy']:.2f}")
        for _, row in result.head(3).iterrows():
            print(f"  {row['text']!r} -> {row['output']!r} (expected {row['category']})")

    print()
    print(pd.DataFrame(table).to_string(index=False))
