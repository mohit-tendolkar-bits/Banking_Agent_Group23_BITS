"""Hugging Face causal LM used for all prompt and decoding experiments."""

import logging
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")
logging.getLogger("httpx2").setLevel(logging.WARNING)
log = logging.getLogger("llm")

MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-1.5B-Instruct")  # set MODEL=Qwen/Qwen2.5-3B-Instruct to compare
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

log.info("loading %s on %s", MODEL, DEVICE)
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype="auto").to(DEVICE)
log.info("model ready, dtype %s, %.1fB parameters", model.dtype, model.num_parameters() / 1e9)


def generate(messages: list[dict], max_new_tokens: int = 64, tools: list[dict] | None = None, **decoding) -> str:
    """Run one chat turn. `decoding` is passed to `model.generate`, for example
    do_sample=True, temperature=0.7, top_k=50, top_p=0.9. Default is greedy."""
    text = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    decoding = {"do_sample": False, **decoding}
    n_prompt = inputs["input_ids"].shape[1]
    log.info("LLM call: %d prompt tokens, decoding=%s", n_prompt, decoding)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, **decoding)
    text = tokenizer.decode(output[0, n_prompt:], skip_special_tokens=True)
    log.info("LLM output (%d new tokens): %r", output.shape[1] - n_prompt, text)
    return text


if __name__ == "__main__":
    print(generate([{"role": "user", "content": "I lost my card. What should I do?"}]))
