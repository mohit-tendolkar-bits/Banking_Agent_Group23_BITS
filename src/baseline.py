"""E1: baseline intent classification with one simple prompt, greedy decoding."""

import logging

import pandas as pd

from src.llm import generate

log = logging.getLogger("baseline")

TRAIN = pd.read_csv("data/train.csv")
INTENTS = sorted(TRAIN["category"].unique())
TASK = "Classify the customer message into one of these intents:\n" + ", ".join(INTENTS)
log.info("loaded %d intent names from the BANKING77 train split", len(INTENTS))


def build_prompt(message: str) -> list[dict]:
    """Simplest prompt: the task, the 77 names, the message. No role, no examples, no rules."""
    return [{"role": "user", "content": f"{TASK}\n\nMessage: {message}\nIntent:"}]


def parse_intent(output: str) -> str:
    """Return the longest intent name found in the output, or '' if none."""
    text = output.lower().replace(" ", "_")
    found = [i for i in INTENTS if i.lower() in text]
    intent = max(found, key=len) if found else ""
    log.info("parse_intent: found %s in output -> %r", found or "nothing", intent)
    return intent


if __name__ == "__main__":
    data = pd.read_csv("evaluation/eval_subset.csv")
    log.info("E1 baseline: %d test messages, greedy decoding, max 32 new tokens", len(data))

    outputs, predicted = [], []
    for i, (text, label) in enumerate(zip(data["text"], data["category"]), start=1):
        log.info("--- row %d/%d: %r (expected %s)", i, len(data), text, label)
        outputs.append(generate(build_prompt(text), max_new_tokens=32))
        predicted.append(parse_intent(outputs[-1]))

    data["output"] = outputs
    data["predicted"] = predicted
    data["correct"] = data["predicted"] == data["category"]
    data.to_csv("evaluation/results/results_baseline.csv", index=False)
    log.info("saved evaluation/results/results_baseline.csv with columns %s", list(data.columns))

    print(f"accuracy: {data['correct'].mean():.2f}")
    print(f"no intent found: {(data['predicted'] == '').sum()}")
    for _, row in data.head(3).iterrows():
        print(f"\nmessage:  {row['text']}\noutput:   {row['output']!r}\nexpected: {row['category']}")
