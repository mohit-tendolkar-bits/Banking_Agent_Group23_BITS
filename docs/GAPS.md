> **Historical.** This is the gap analysis of the *first* version of the code (about 3 of 20 marks). All gaps listed here have since been closed. For the current status see `docs/REQUIREMENTS_TRACEABILITY.md`.

# Gap Analysis: Code vs Requirements

Source: `PCAM_ZG516_Group_Major_Project_Assignment_1_CS1_CS5.docx`
Dataset: BANKING77 (option D1)
Code: `main.py` (96 lines)

## What the code has

- Loads the BANKING77 train split from GitHub.
- TF-IDF + Logistic Regression intent classifier (77 intents).
- Two tools with JSON schemas: `classify_intent` and `handle_intent`.
- Agent loop: `qwen2.5:3b` through Ollama calls the tools and writes the final reply.
- CLI loop for interactive use.

## Requirement status

| # | Requirement | Status | Gap |
|---|---|---|---|
| 1 | Problem definition | Not done | No target user, scenario, input/output, or reason for an LLM. |
| 2 | Dataset understanding | Partial | Data is loaded. No description, field analysis, sample examples, preprocessing record, or evaluation subset. |
| 3 | Generative-model analysis | Not done | No comparison of three model families. No justification for a causal LM. |
| 4 | LLM architecture | Not done | No explanation of tokenizer, embeddings, attention, LM head, autoregressive generation. Ollama hides the tokenizer step. |
| 5 | Baseline system | Partial | Model runs through Ollama, not Hugging Face `transformers`. No simple baseline prompt. No 3 baseline outputs. |
| 6 | Prompt engineering | Not done | One system prompt only. No zero-shot, few-shot, role-based, constraint-based variants. No comparison table. |
| 7 | Decoding study | Not done | No decoding options set. No greedy vs temperature vs top-k/top-p comparison. |
| 8 | Adaptation decision | Not done | No prompting vs RAG vs PEFT analysis. |
| 9 | Structured output | Partial | Tools return JSON dicts. No schema definition. No validation step. No sample validated outputs. |
| 10 | Tool / function calling | Done | Two tools with input schemas. Agent loop invokes them. `handle_intent` is a stub with the same output for all 77 intents. |
| 11 | Programmatic prompting | Partial | `DESCRIPTION` is an f-string. No parameterized template function for different inputs or prompt variants. |
| 12 | Evaluation | Not done | No criteria, no metrics, no failure analysis. |
| 13 | Responsible use | Not done | No risk analysis or controls. |

## Required experiments

| Exp | Minimum requirement | Status |
|---|---|---|
| E1 Baseline | One simple prompt, 3+ outputs | Not done |
| E2 Prompt comparison | 4+ prompt variants, comparison table | Not done |
| E3 Decoding | Greedy + temperature + top-k/top-p | Not done |
| E4 Structured output | One schema + validation, JSON examples | Not done |
| E5 Tool/function | One function, schema + invocation | Done |
| E6 Failure analysis | 3+ failure cases, analysis table | Not done |
| E7 Final system | Integrated workflow, notebook/demo | Partial (CLI only, no notebook) |

## Deliverables

| Deliverable | Status |
|---|---|
| Technical report (10–15 pages) | Not done |
| Colab notebook or executable implementation | Partial (`main.py` runs, no notebook) |
| Dataset description and preprocessing record | Not done |
| Prompt experiment log | Not done |
| Decoding comparison table | Not done |
| Structured-output schema and validated samples | Not done |
| Tool/function-calling specification and demo | Partial (schemas exist, no recorded demo) |
| Evaluation and failure-analysis table | Not done |
| 8–10 minute presentation | Not done |
| Individual contribution statement | Not done |

## Design gaps

1. **The LLM does not identify the intent.** A TF-IDF classifier does this inside `classify_intent`. The requirement is "intent understanding" with an LLM. Prompt experiments (Req 6) have nothing to measure until the LLM classifies the intent.
2. **Ollama, not Hugging Face.** Req 5 asks for a Hugging Face causal LM. The tokenizer, logits, and decoding parameters are not accessible through Ollama. Reqs 4, 6, 7 depend on this access.
3. **`handle_intent` is a stub.** The requirement accepts a simulation, but the examples (transaction status, card block, branch lookup) have realistic inputs and outputs. The current tool returns one shape for all intents.
4. **No evaluation subset.** All experiments need a fixed test subset from BANKING77. None exists.

## Rubric coverage estimate

| Component | Marks | Evidence in code |
|---|---|---|
| Problem + dataset analysis | 2 | None |
| LLM/generative-model understanding | 3 | None |
| Prompt engineering | 3 | None |
| Adaptation strategy | 2 | None |
| Structured output + tool calling | 3 | Partial (tool calling) |
| Implementation | 3 | Partial (runs, not reproducible notebook) |
| Evaluation + failure analysis | 2 | None |
| Report + demonstration + contribution | 2 | None |

Estimated evidence: about 3 of 20 marks.
