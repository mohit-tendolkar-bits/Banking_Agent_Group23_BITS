# Requirements Traceability

Maps every item of the assignment (`PCAM_ZG516_Group_Major_Project_Assignment_1_CS1_CS5.docx`) to the code,
result files and report section that cover it. Use it as a checklist before submitting.

Report section numbers refer to `docs/REPORT.md` (the 10 to 15 page submission report).
`docs/FINAL_REPORT.md` is the longer reference version.

## Section 4 of the assignment: major requirements

| # | Requirement | Code | Evidence (results / notebook) | Report |
|---|---|---|---|---|
| 1 | Problem definition | n/a | n/a | 2 |
| 2 | Dataset understanding | `evaluation/make_eval_subset.py`, `data/*.csv` | notebook section 1; `eval_subset.csv`, `heldout_subset.csv` | 3 |
| 3 | Generative-model analysis (3+ families) | **`src/generative_families.py`** | `results_generative_families.csv`, `generative_families_samples.txt`; notebook section 1b | 4.1, 4.2 |
| 4 | LLM architecture | `src/llm.py` | notebook section 2 (token IDs, embedding shape, top-5 next-token probabilities) | 4.3 |
| 5 | Baseline (HF causal LM, simple prompt) | `src/baseline.py` | `results_baseline.csv` (0.41) | 6 |
| 6 | Prompt engineering (zero-shot, few-shot, role, constraint) | `src/prompts.py` | `results_zero_shot / role / constraint / few_shot / few_shot_constraint.csv` | 7 |
| 7 | Decoding study (greedy, temperature, top-k, top-p) | `src/decoding.py` | `results_decoding_*.csv` (5 files) | 8 |
| 8 | Adaptation decision (prompting vs RAG vs PEFT) | `src/retrieval.py` | `results_retrieval.csv` (0.84), `results_tfidf_logreg.csv` | 9 |
| 9 | Structured output (schema + validation) | `src/structured.py`, `main.py` (`INTENT_SCHEMA`) | `results_structured.csv` (86/100 pass schema) | 10.1 |
| 10 | Tool / function calling | `main.py` (`classify_intent`, `handle_intent`, `TOOLS`) | notebook section 9 (live invocation) | 10.2 |
| 11 | Programmatic prompting | `src/prompts.py`, `src/retrieval.py`, `src/structured.py` | notebook section 4 | 7.1 |
| 12 | Evaluation (4+ criteria) + failure analysis | `evaluation/final_eval.py`, `evaluation/reference_classifier.py` | `results_heldout.csv` (0.77), `docs/FAILURES.md` | 11, 12 |
| 13 | Responsible use | `main.py` (enum validation, code-driven tool order) | n/a | 12.4 |

## Section 6 of the assignment: required experiments

| Exp | Minimum requirement | Where |
|---|---|---|
| E1 Baseline | One simple prompt, 3+ outputs | `src/baseline.py`; report 6 (three baseline outputs) |
| E2 Prompt comparison | 4+ variants, comparison table | `src/prompts.py`; report 7 (5 variants) |
| E3 Decoding | Greedy + temperature + top-k/top-p | `src/decoding.py`; report 8 (5 settings) |
| E4 Structured output | One schema + validation, JSON examples | `src/structured.py`; report 10.1 |
| E5 Tool / function | 1+ function, schema + invocation | `main.py`; report 10.2 (2 tools) |
| E6 Failure analysis | 3+ failure cases, table | `docs/FAILURES.md` (11 cases); report 12 |
| E7 Final system | Integrated workflow, notebook / demo | `main.py`, `notebooks/banking_agent.ipynb` |

## Section 9 of the assignment: deliverables

| Deliverable | Status | File |
|---|---|---|
| Technical report, about 10 to 15 pages | Ready. Fill in the title page. | `docs/REPORT.md`, `docs/Banking_Agent_Report.docx` |
| Working Colab notebook or equivalent | Ready | `notebooks/banking_agent.ipynb`, `main.py` |
| Dataset description and preprocessing record | In report | report 3 |
| Prompt experiment log (baseline and improved) | In report and results | report 6, 7; `results_*.csv` |
| Decoding comparison table | In report | report 8 |
| Structured-output schema and validated samples | In report | report 10.1 |
| Tool / function-calling specification and demo | In report and notebook | report 10.2; notebook section 9 |
| Evaluation and failure-analysis table | In report | report 11, 12; `docs/FAILURES.md` |
| 8 to 10 minute demonstration | Script ready; record or present it | `docs/DEMO_SCRIPT.md` |
| Individual contribution statement | **Group must complete and sign** | report Appendix B |

## Things only the group can do

1. Fill in group number, member names and IDs on the report title page.
2. Fill in and have every member confirm the contribution statement (Appendix B).
3. Push the repository to GitHub if you will present from Colab. The notebook's first cell clones it.
4. Rehearse the demo so that every member can explain the whole workflow (assignment section 14).
