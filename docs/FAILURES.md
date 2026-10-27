# E6: Evaluation and Failure Analysis

All numbers are on the fixed 100-row subset (`eval_subset.csv`), model
`Qwen/Qwen2.5-1.5B-Instruct`, greedy decoding unless stated.

## Evaluation criteria

| Criterion | How we measure it | Why it matters |
|---|---|---|
| Accuracy | Predicted intent == gold label | The agent executes the predicted intent; a wrong one is a wrong action |
| Valid output | Output contains exactly one real intent name (E4: passes the JSON schema) | An invalid label cannot be executed; the agent must ask again |
| Format discipline | No preamble text before the label | Shorter output, cheaper, easier to parse |
| Reproducibility | Same output on repeat runs | Needed for debugging and for the report |

## Summary of all systems

| System | Accuracy | No valid intent | Preamble | Deterministic |
|---|---|---|---|---|
| E1 zero-shot | 0.41 | 4 | 57 | yes |
| E2 role | 0.40 | 4 | 30 | yes |
| E2 constraint | 0.40 | 2 | 0 | yes |
| E2 few-shot (random) | 0.43 | 12 | 1 | yes |
| E2 few-shot + constraint | 0.43 | 8 | 0 | yes |
| E3 same prompt, temperature 0.7 | 0.44 | 10 | 0 | no |
| E3 same prompt, temperature 1.5 | 0.40 | 7 | 0 | no |
| E3 same prompt, top-k 50 | 0.44 | 7 | 0 | no |
| E3 same prompt, top-p 0.9 | 0.38 | 11 | 0 | no |
| E4 JSON + schema | 0.43 (0.50 on valid rows) | 14 rejected by schema | 0 | yes |
| Retrieval few-shot + constraint | **0.84** | 3 | 0 | yes |
| Reference: TF-IDF + LogReg | 0.79 | 0 | n/a | yes |
| Reference: 5-NN vote, no LLM | 0.77 | 0 | n/a | yes |

Across the four main systems (zero-shot, few-shot+constraint, JSON,
retrieval): 24 rows were right in all four, 12 were wrong in all four.

## Failure types

| Type | What happens | Seen in | Fixed by |
|---|---|---|---|
| A. Near-synonym label | Model picks a neighbouring intent that means almost the same | All systems. ~35 rows in E1, 11 in retrieval | Retrieval (most), not all |
| B. Favourite label | Short generic names absorb unrelated messages (`pending_transfer`, `top_up_failed`, `get_physical_card`) | E1, E2 | Retrieval |
| C. Invented label | A plausible snake_case name that is not in the list | E2 few-shot (12), E4 (14) | Schema `enum` rejects it; retrieval reduces it to 3 |
| D. Refusal | "None of the provided intents match" | E1 (2), retrieval (3) | Constraint reduces it; not removed |
| E. Preamble | "The intent for the message ... is:" before the label | E1 (57), role (30) | Constraint or few-shot |
| F. Retrieval misleads | The 5 nearest examples share surface words but not meaning, and the model follows them | Retrieval only (2 rows right in E1, wrong here) | Better retriever (embeddings), more examples |
| G. Random decoding | Sampling changes a right answer to a wrong one | E3 (5 to 13 flips per setting) | Greedy |

## Failure cases

| # | Input | Expected | Output | System | Type | Likely cause | Possible fix |
|---|---|---|---|---|---|---|---|
| 1 | How do I track my card? | `card_arrival` | `card_delivery_estimate` | E1 | A | Both labels are about a card in transit; the model has no definition of either | Retrieval fixed this row (nearest examples were `card_arrival`) |
| 2 | Why is it asking me to verify my identity | `why_verify_identity` | `verify_my_identity` | E1, E2 | A | Dataset separates "how do I verify" from "why must I verify"; the names alone do not show this | Retrieval fixed this row; a one-line description per intent would also help |
| 3 | Why would I have a pending payment? | `pending_card_payment` | `pending_transfer` | E1 | B | `pending_transfer` is a short, frequent-looking label; the model reaches for it on any "pending" message | Retrieval fixed this row |
| 4 | How can I top up with this cheque I got? | `top_up_by_cash_or_cheque` | `Top_up_by_cheque_or_cash_deposit` (E1), `top_up_by_cheque` (E2 few-shot), `top_up_by_cheque` (E4, rejected) | E1, E2, E4 | C | The model generates text; it composes a name from the message words instead of copying from the list | Schema `enum` catches it (E4); retrieval fixed the row |
| 5 | How many miles away is the ATM? | `atm_support` | "The given message ... does not match any of the provided intents" | E1 | D | Message is unusual; the model prefers honesty to a forced choice | Constraint "always choose exactly one" fixed this row |
| 6 | What do i have to do to change my name? | `edit_personal_details` | "None of the provided intents match this message." | Retrieval | D, F | The 5 nearest examples all start "What do I have to do to ..." but are about pin, card, transfer. None is about personal details. The model saw no fit and refused. E1 got this row right. | Retriever that matches meaning, not words (sentence embeddings) |
| 7 | How long should my new card take to arrive? | `card_arrival` | `card_delivery_estimate` | Retrieval | F | 3 of 5 nearest examples are `card_delivery_estimate`; the model followed the majority. E1 got this row right. Gold label is arguably debatable here. | More examples (k=10), or show both labels' descriptions |
| 8 | Why am I unable to take money from my card but the statement says the transaction is still pending? | `pending_cash_withdrawal` | `pending_transfer` (all systems) | All | A, F | Long message with two topics ("unable to take money", "still pending"). Nearest examples are all `declined_cash_withdrawal` or `failed_transfer`; none is `pending_cash_withdrawal`. | Larger model, or retrieval on the second half of the message |
| 9 | How do I exchange currency? | `exchange_via_app` | `supported_cards_and_currencies` (all systems) | All | A | The gold label assumes app context that is not in the message; `supported_cards_and_currencies` is a reasonable reading. The 5-NN vote was right, the LLM overrode it. | Accept as a dataset ambiguity; report it |
| 10 | What foreign exchange rate will you use? | `exchange_rate` | `supported_cards_and_currencies` | E2 constraint | G-like | Same prompt + few-shot got this right; the constraint-only prompt lost it. Small prompt changes flip borderline rows. | Few-shot or retrieval |
| 11 | when does the rate get determined | `exchange_rate` | `unable_to_verify_identity` | E3 temperature 1.5 | G | Greedy gave `exchange_rate`. High temperature sampled an unrelated label. | Greedy decoding for classification |

## What the analysis says

1. The dominant failure is near-synonym confusion (type A). Instructions
   cannot fix it because the problem is missing knowledge of label
   boundaries, not missing instructions. Relevant examples fix most of
   it (0.43 to 0.84).
2. Format failures (C, D, E) are cheap to fix. The constraint line removed
   all preamble; the JSON schema turns invented labels into a hard error
   the agent can handle ("please rephrase").
3. Retrieval has its own failure mode (F). TF-IDF matches words, not
   meaning. "What do I have to do to change my name" retrieves five
   "What do I have to do to ..." rows about other topics. A sentence
   embedding retriever would fix this class.
4. Sampling (G) adds errors and removes none. Use greedy for the intent
   step. Sampling is appropriate only for the free-text reply to the
   customer.
5. Some gold labels are debatable (cases 7 and 9). A small number of the
   remaining 16 errors are dataset ambiguity, not model failure.

## Reproduce

```bash
uv run -m src.baseline      # E1
uv run -m src.prompts       # E2
uv run -m src.decoding      # E3
uv run -m src.structured    # E4
uv run -m src.retrieval     # adaptation
```

Each script writes `results_*.csv` with one row per test message:
`text, category, output, predicted, correct`.
