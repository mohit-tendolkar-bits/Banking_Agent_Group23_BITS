# Assignment Coverage and Remaining Work

This tracker reflects the current repository, including the experiments,
technical report, and presentation documents. It supersedes the initial gap
analysis written for the repository's first version.

## Requested topics

| Topic | Coverage | Evidence and remaining limitation |
|---|---|---|
| GAN | Included and trained | The family-comparison script trains a conditional TF-IDF GAN with an auxiliary intent head and evaluates intent accuracy. It generates feature vectors, not customer-facing text; results are in `evaluation/results/generative_family_comparison.csv`. |
| VAE | Included and trained | The script trains a VAE to reconstruct TF-IDF features with a supervised intent head. It is not a conditional text-generation VAE; its intent predictions come from the latent classifier. |
| Fine-tuning / PEFT | Implemented; full run pending | `evaluation/peft_lora_comparison.py` fine-tunes Qwen with LoRA and evaluates the development and held-out sets. A complete run requires a suitable accelerator; this environment is CPU-only with about 1.1 GB free memory, so no PEFT metric is claimed yet. |
| Model comparison | Included: causal LM vs. VAE vs. GAN | Section 3 compares train/test intent accuracy for these three families on the same BANKING77 split. |
| Adaptation decision | Covered and implemented | Prompting, retrieval, and PEFT are compared in the report. Retrieval with five similar labelled examples remains the runtime assistant strategy; the separate PEFT experiment is available to measure the trade-off. |
| Hallucination | Covered with observed cases and controls | The report documents invented intent labels, unsupported claims, and invented tool arguments. Schema validation, code-controlled tool order, confirmation, and human fallback are described as controls. |

## Model-family comparison limitations

- The causal GRU is trained with next-token prediction and an auxiliary
  supervised intent head. It is an autoregressive language model with a
  classification head; intent accuracy is measured from that head.
- The VAE and GAN operate on TF-IDF features, whereas the causal LM operates
  on token sequences. Their prediction methods also differ, so this is an
  educational family comparison on a shared intent task, not a fully
  controlled architecture-only benchmark.
- The GAN is a feature-space generator/discriminator and does not generate
  text. A low score describes this compact implementation's performance,
  not the GAN family as a whole.
- Accuracy values in the report, README, and result CSV must be refreshed
  together whenever the benchmark is rerun with changed code or settings.

## Other project evidence

| Area | Status | Evidence |
|---|---|---|
| Problem and dataset | Covered | `docs/FINAL_REPORT.md`, Sections 1–2 |
| Prompting and decoding | Implemented and evaluated | `src/prompts.py`, `src/decoding.py`, experiment result CSVs |
| Structured output | Implemented and evaluated | `src/structured.py`; report includes schema pass/fail results |
| Tool calling and final assistant | Implemented | `main.py`, `ui.py`, and `notebooks/banking_agent.ipynb` |
| Evaluation and failure analysis | Covered | `docs/FINAL_REPORT.md`, `docs/FAILURES.md`, and saved result files |
| Responsible use | Covered | `docs/FINAL_REPORT.md`, Section 13 |

## Remaining work

1. Run the full PEFT experiment on a suitable CUDA GPU and add its measured
   development/held-out scores to the adaptation comparison.
2. Regenerate `docs/SLIDES.pptx` from the updated Markdown before presenting.
3. Fill in the project-specific group/member details and contribution
   statement before submission.
