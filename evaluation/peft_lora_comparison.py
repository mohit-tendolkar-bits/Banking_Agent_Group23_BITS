"""Fine-tune Qwen with LoRA and compare it with prompting and retrieval.

Run from the project root on a CUDA-capable GPU:
    uv run -m evaluation.peft_lora_comparison
"""

import argparse
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
SEED = 17
MAX_NEW_TOKENS = 16


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(name: str) -> torch.device:
    if name != "auto":
        device = torch.device(name)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available.")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_system_prompt(labels: list[str]) -> str:
    label_list = "\n".join(labels)
    return (
        "Classify the customer message into exactly one BANKING77 intent. "
        "Reply with only the exact intent name from this list:\n"
        f"{label_list}"
    )


def prompt_token_ids(tokenizer, text: str, labels: list[str], max_length: int) -> list[int]:
    messages = [
        {"role": "system", "content": make_system_prompt(labels)},
        {"role": "user", "content": f"Customer message: {text}"},
    ]
    encoded_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    token_ids = encoded_prompt["input_ids"]
    return list(token_ids[-max_length:])


class IntentFineTuningDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, tokenizer, labels: list[str], max_length: int):
        self.samples: list[tuple[list[int], list[int]]] = []
        eos_id = tokenizer.eos_token_id
        if eos_id is None:
            raise ValueError("The tokenizer must define an EOS token for supervised fine-tuning.")

        for row in frame.itertuples(index=False):
            prompt_ids = prompt_token_ids(tokenizer, row.text, labels, max_length)
            answer_ids = tokenizer(row.category, add_special_tokens=False)["input_ids"] + [eos_id]
            if len(answer_ids) >= max_length:
                raise ValueError(f"Intent label exceeds max sequence length: {row.category}")
            prompt_ids = prompt_ids[: max_length - len(answer_ids)]
            input_ids = prompt_ids + answer_ids
            target_ids = [-100] * len(prompt_ids) + answer_ids
            self.samples.append((input_ids, target_ids))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[list[int], list[int]]:
        return self.samples[index]


def collate_training_batch(batch, pad_token_id: int) -> dict[str, torch.Tensor]:
    max_length = max(len(input_ids) for input_ids, _ in batch)
    input_ids = []
    labels = []
    attention_mask = []
    for token_ids, target_ids in batch:
        padding = max_length - len(token_ids)
        input_ids.append(token_ids + [pad_token_id] * padding)
        labels.append(target_ids + [-100] * padding)
        attention_mask.append([1] * len(token_ids) + [0] * padding)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


def predict_intents(
    model,
    tokenizer,
    frame: pd.DataFrame,
    labels: list[str],
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> tuple[list[str], list[str]]:
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    prompts = [
        prompt_token_ids(tokenizer, text, labels, max_length)
        for text in frame["text"].astype(str)
    ]
    predictions: list[str] = []
    outputs: list[str] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(prompts), batch_size):
            prompt_batch = prompts[start : start + batch_size]
            padded = tokenizer.pad(
                {"input_ids": prompt_batch},
                padding=True,
                return_tensors="pt",
            ).to(device)
            generated = model.generate(
                **padded,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            generated = generated[:, padded["input_ids"].shape[1] :]
            batch_outputs = tokenizer.batch_decode(generated, skip_special_tokens=True)
            outputs.extend(batch_outputs)
            predictions.extend([extract_intent(output, labels) for output in batch_outputs])
    return predictions, outputs


def extract_intent(output: str, labels: list[str]) -> str:
    normalized = output.strip().lower().replace(" ", "_")
    for label in labels:
        if normalized == label.lower():
            return label
    matches = [
        label
        for label in labels
        if re.search(rf"(?<!\w){re.escape(label.lower())}(?!\w)", output.lower())
    ]
    return max(matches, key=len) if matches else ""


def train_model(
    model,
    train_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    accumulation_steps: int,
    device: torch.device,
) -> None:
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=learning_rate)
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        batch_count = len(train_loader)
        for batch_index, batch in enumerate(train_loader):
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = model(**batch).loss
            (loss / accumulation_steps).backward()
            is_group_end = (batch_index + 1) % accumulation_steps == 0
            is_epoch_end = batch_index + 1 == batch_count
            if is_group_end or is_epoch_end:
                remainder = (batch_index + 1) % accumulation_steps
                if is_epoch_end and remainder:
                    scale = accumulation_steps / remainder
                    for parameter in trainable_parameters:
                        if parameter.grad is not None:
                            parameter.grad.mul_(scale)
                clip_grad_norm_(trainable_parameters, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            total_loss += loss.detach().item()
        print(f"LoRA epoch {epoch + 1}/{epochs}: average loss {total_loss / batch_count:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--max-train-samples", type=int, help="Limit training rows for a quick smoke test.")
    parser.add_argument("--max-eval-samples", type=int, help="Limit each evaluation split for a quick smoke test.")
    parser.add_argument("--train-file", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--development-file", type=Path, default=Path("evaluation/eval_subset.csv"))
    parser.add_argument("--heldout-file", type=Path, default=Path("evaluation/heldout_subset.csv"))
    parser.add_argument("--adapter-dir", type=Path, default=Path("artifacts/banking77-qwen-lora"))
    parser.add_argument("--results-file", type=Path, default=Path("evaluation/results/results_peft_lora.csv"))
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.gradient_accumulation_steps < 1:
        parser.error("epochs, batch-size, and gradient-accumulation-steps must be positive.")
    if args.max_length < 64:
        parser.error("max-length must be at least 64 to fit the intent list and message.")

    set_seed(args.seed)
    device = choose_device(args.device)
    if device.type == "cpu":
        print("Warning: Qwen LoRA fine-tuning on CPU is memory-intensive and very slow; a CUDA GPU is recommended.")

    train_frame = pd.read_csv(args.train_file)
    development_frame = pd.read_csv(args.development_file)
    heldout_frame = pd.read_csv(args.heldout_file)
    labels = sorted(train_frame["category"].unique())
    for split_name, frame in (("development", development_frame), ("held-out", heldout_frame)):
        unseen = set(frame["category"]) - set(labels)
        if unseen:
            raise ValueError(f"{split_name} split contains labels absent from training: {sorted(unseen)}")
    if args.max_train_samples:
        train_frame = train_frame.iloc[: args.max_train_samples].copy()
    if args.max_eval_samples:
        development_frame = development_frame.iloc[: args.max_eval_samples].copy()
        heldout_frame = heldout_frame.iloc[: args.max_eval_samples].copy()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    training_dataset = IntentFineTuningDataset(train_frame, tokenizer, labels, args.max_length)
    train_loader = DataLoader(
        training_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_training_batch(batch, tokenizer.pad_token_id),
    )

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype="auto")
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config).to(device)
    model.print_trainable_parameters()
    train_model(
        model,
        train_loader,
        args.epochs,
        args.learning_rate,
        args.gradient_accumulation_steps,
        device,
    )

    args.adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.adapter_dir)
    tokenizer.save_pretrained(args.adapter_dir)
    model.config.use_cache = True

    result_rows = []
    prediction_rows = []
    for split_name, frame in (("development", development_frame), ("heldout", heldout_frame)):
        predictions, outputs = predict_intents(
            model,
            tokenizer,
            frame,
            labels,
            args.max_length,
            args.eval_batch_size,
            device,
        )
        gold = frame["category"].tolist()
        correct = [prediction == label for prediction, label in zip(predictions, gold)]
        result_rows.append(
            {
                "method": "Qwen LoRA PEFT",
                "base_model": args.model,
                "split": split_name,
                "examples": len(frame),
                "accuracy": sum(correct) / len(correct),
                "invalid_predictions": sum(not prediction for prediction in predictions),
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "learning_rate": args.learning_rate,
                "lora_rank": 8,
                "max_length": args.max_length,
                "seed": args.seed,
            }
        )
        prediction_rows.extend(
            {
                "split": split_name,
                "text": text,
                "category": label,
                "output": output,
                "predicted": prediction,
                "correct": is_correct,
            }
            for text, label, output, prediction, is_correct in zip(
                frame["text"], gold, outputs, predictions, correct
            )
        )

    results_file = args.results_file
    results_file.parent.mkdir(parents=True, exist_ok=True)
    predictions_file = results_file.with_name(f"{results_file.stem}_predictions.csv")
    pd.DataFrame(result_rows).to_csv(results_file, index=False)
    pd.DataFrame(prediction_rows).to_csv(predictions_file, index=False)
    print("\nPEFT evaluation:")
    print(pd.DataFrame(result_rows).to_string(index=False))
    print(f"Saved metrics to {results_file}")
    print(f"Saved predictions to {predictions_file}")
    print(f"Saved LoRA adapter to {args.adapter_dir}")


if __name__ == "__main__":
    main()
