# Banking Intent Agent

A customer support assistant for an online bank. A local Hugging Face LLM
(`Qwen/Qwen2.5-1.5B-Instruct`) reads a customer message, finds the intent
(one of the 77 BANKING77 intents), runs a tool for that intent, and writes
a short reply.

The project has two parts:

1. **Experiments** (E1 to E7) that compare prompts, decoding settings,
   structured output, and retrieval. Each experiment is one script and
   writes one or more `results_*.csv` files to `evaluation/results/`.
2. **The final assistant** (`main.py`) that uses the best settings from the
   experiments.

## Requirements

- Python 3.13 or later
- [uv](https://docs.astral.sh/uv/) (installs Python and all packages)
- About 4 GB of free disk space for the model
- 8 GB RAM. The scripts run on Apple GPU (`mps`), NVIDIA GPU (`cuda`), or CPU.

## Setup

```bash
git clone https://github.com/2025paml556-bhoomika/banking-agent.git
cd banking-agent
uv sync
```

`uv sync` creates `.venv` and installs the packages from `uv.lock`.
The model downloads automatically the first time you run a script
(about 3 GB, cached by Hugging Face in `~/.cache/huggingface`).

## Quick start

Run the final assistant:

```bash
uv run main.py
```

Type a message, for example `I lost my card, please block it`.
The assistant prints each tool call and then the reply. Press Ctrl-D to exit.

Run one test generation to check that the model loads:

```bash
uv run -m src.llm
```

## Run the experiments

Run the scripts in this order from the project root. Each one reads
`evaluation/eval_subset.csv` (the same 100 test messages) and writes its
results as CSV to `evaluation/results/`.
Times are for an M3 Mac with 8 GB RAM.

| Step | Command | Writes | Time |
|---|---|---|---|
| Subsets | `uv run -m evaluation.make_eval_subset` | `eval_subset.csv`, `heldout_subset.csv` (already in the repo) | seconds |
| Generative families (Req 3) | `uv run -m src.generative_families` | `results_generative_families.csv`, `generative_families_samples.txt` | 1 min, CPU only |
| E1 baseline | `uv run -m src.baseline` | `results_baseline.csv` | 2 min |
| E2 prompts | `uv run -m src.prompts` | `results_<style>.csv` (5 files) | 8 min |
| E3 decoding | `uv run -m src.decoding` | `results_decoding_<setting>.csv` (5 files) | 8 min |
| E4 structured output | `uv run -m src.structured` | `results_structured.csv` | 3 min |
| Adaptation (retrieval) | `uv run -m src.retrieval` | `results_retrieval.csv` | 2 min |
| Reference classifier | `uv run -m evaluation.reference_classifier` | `results_tfidf_logreg.csv` | seconds |
| Final, held-out | `uv run -m evaluation.final_eval` | `results_heldout.csv` | 2 min |

The experiments use `evaluation/eval_subset.csv` (development set) to compare
approaches. `evaluation/final_eval.py` runs the chosen system once on
`evaluation/heldout_subset.csv`, 100 different test rows that no experiment used.

You do not have to rerun the experiments to read the results. All
`results_*.csv` files are in `evaluation/results/`.

To compare a larger model, set the `MODEL` variable before any command:

```bash
MODEL=Qwen/Qwen2.5-3B-Instruct uv run -m src.baseline
```

The 3B model needs about 6 GB RAM and does not fit on an 8 GB Mac.
Use Google Colab for it.

## Notebooks

`notebooks/banking_agent.ipynb` runs every command in the table above in
order, shows the result tables, and runs the live demo. "Run all" takes
about 21 minutes on an M3 Mac. The outputs are saved in the file, so you
can read the results without running it. To run it in about one minute,
skip the `%run -m <module>` cells; the result cells read the saved CSV
files.

The notebook imports the code from `main.py` and the `src/` modules. The first cell
installs the dependencies when it runs on Colab and changes the working
directory to the project root. To run a different model on Colab, set
`os.environ["MODEL"]` in that cell; the notebook clones the GitHub
repository, so push your changes first.

To open a notebook locally, open it in VS Code or Cursor and select the
`.venv` Python as the kernel. The `ipykernel` package is already installed
by `uv sync`.

## Files

| File | Purpose |
|---|---|
| `main.py` | E7. The final assistant with `classify_intent` and `handle_intent` tools. |
| `src/generative_families.py` | Req 3. Trains a toy autoregressive LM, VAE and GAN on the train messages and compares them. NumPy only; no model download. |
| `src/llm.py` | Loads the model. `generate(messages, **decoding)` is the one function all scripts use. |
| `src/baseline.py` | E1. Loads the 77 intent names, the simplest prompt, and `parse_intent`. |
| `src/prompts.py` | E2. Five prompt styles: zero-shot, role, constraint, few-shot, few-shot + constraint. |
| `src/decoding.py` | E3. Greedy, temperature 0.7, temperature 1.5, top-k 50, top-p 0.9. |
| `src/structured.py` | E4. JSON output validated with `jsonschema`. |
| `src/retrieval.py` | Adaptation. Five examples chosen by TF-IDF similarity. |
| `data/train.csv`, `data/test.csv` | BANKING77 train and test splits (local copies from the PolyAI GitHub repository). |
| `evaluation/make_eval_subset.py` | Creates `eval_subset.csv` and `heldout_subset.csv` from `data/test.csv`. |
| `evaluation/reference_classifier.py` | TF-IDF + logistic regression, trained on the train split, scored on the development subset. The classical bar to beat. |
| `evaluation/final_eval.py` | Runs the final system once on the held-out subset. |
| `evaluation/eval_subset.csv` | Development set: the 100 test messages used by every experiment. |
| `evaluation/heldout_subset.csv` | Held-out set: 100 other test messages, used only by `final_eval.py`. |
| `evaluation/results/results_*.csv` | Output of each experiment. |
| `docs/PROGRESS.md` | Step-by-step log: what we did, why, and what we found. |
| `docs/FAILURES.md` | E6. Failure types and worked failure cases. |
| `docs/GAPS.md` | Gap analysis of the first version against the assignment. |
| `docs/REPORT.md` | The technical report (submission version). Also provided as `docs/Banking_Agent_Report.docx`. |
| `docs/FINAL_REPORT.md` | Longer reference version of the report. |
| `docs/REQUIREMENTS_TRACEABILITY.md` | Every assignment requirement mapped to code, results and report section. |
| `docs/DEMO_SCRIPT.md` | Timed script for the 8 to 10 minute demonstration. |

## Results in short

| System | Accuracy on 100 development messages |
|---|---|
| E1 baseline, zero-shot | 0.41 |
| E2 best prompt (few-shot + constraint) | 0.43 |
| E3 best decoding (greedy) | 0.43 |
| E4 JSON output, schema-valid rows | 0.50 |
| TF-IDF + Logistic Regression (classic baseline) | 0.79 |
| Retrieval few-shot (used in `main.py`) | 0.84 |

| Final system | Accuracy on 100 held-out messages |
|---|---|
| **`main.py` classifier, run once** | **0.77** |

Read `docs/PROGRESS.md` for the reasons behind each number.

## Common problems

- **`uv: command not found`**: install uv with
  `curl -LsSf https://astral.sh/uv/install.sh | sh` and open a new terminal.
- **The first run is slow**: the model downloads once. Later runs load it
  from the cache.
- **Out of memory**: close other applications or use the default 1.5B model.
  Do not set `MODEL` to a larger model on an 8 GB machine.
- **No internet**: the BANKING77 CSVs are in `data/`, so the scripts do not
  download them. The model needs internet for the first download only.
