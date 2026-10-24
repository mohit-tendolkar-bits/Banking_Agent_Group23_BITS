"""E4: structured output. The model answers in JSON; each answer is validated against a schema."""

import json

import pandas as pd
from jsonschema import ValidationError, validate

from src.baseline import INTENTS, TASK
from src.llm import generate
from src.prompts import EXAMPLES

SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": INTENTS},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["intent", "confidence"],
    "additionalProperties": False,
}

FORMAT = (
    'Reply with one JSON object only, no other text: {"intent": <one intent name from the list>, '
    '"confidence": <number between 0 and 1>}.'
)
SHOTS = "".join(
    f"Message: {r.text}\n{json.dumps({'intent': r.category, 'confidence': 0.9})}\n\n" for r in EXAMPLES.itertuples()
)


def build_prompt(message: str) -> list[dict]:
    return [{"role": "user", "content": f"{TASK}\n{FORMAT}\n\n{SHOTS}Message: {message}\n"}]


def parse_json(output: str) -> tuple[dict | None, str]:
    """Return (object, error). error is '' when the output is valid against SCHEMA."""
    text = output.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        return None, f"not JSON: {e.msg}"
    try:
        validate(obj, SCHEMA)
    except ValidationError as e:
        return obj, f"schema: {e.json_path} = {e.instance!r} is invalid"
    return obj, ""


if __name__ == "__main__":
    data = pd.read_csv("evaluation/eval_subset.csv")
    data["output"] = [generate(build_prompt(t), max_new_tokens=48) for t in data["text"]]
    parsed = [parse_json(o) for o in data["output"]]
    data["error"] = [err for _, err in parsed]
    data["predicted"] = [obj.get("intent") if isinstance(obj, dict) else None for obj, _ in parsed]
    data["confidence"] = [obj.get("confidence") if isinstance(obj, dict) else None for obj, _ in parsed]
    data["correct"] = data["predicted"] == data["category"]
    data.to_csv("evaluation/results/results_structured.csv", index=False)

    valid = data["error"] == ""
    print(f"valid JSON:        {data['error'].str.startswith('not JSON').eq(False).sum()}/100")
    print(f"valid vs schema:   {valid.sum()}/100")
    print(f"accuracy (all):    {data['correct'].mean():.2f}")
    print(f"accuracy (valid):  {data.loc[valid, 'correct'].mean():.2f}")
    print(f"mean confidence when right: {data.loc[valid & data['correct'], 'confidence'].mean():.2f}")
    print(f"mean confidence when wrong: {data.loc[valid & ~data['correct'], 'confidence'].mean():.2f}")
    print("\nvalidation errors:")
    print(data.loc[~valid, "error"].value_counts().to_string())
    print("\nexamples:")
    for _, row in data.head(3).iterrows():
        print(f"  {row['text']!r} -> {row['output']!r} (expected {row['category']})")
