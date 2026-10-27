---
marp: true
paginate: true
style: |
  section { font-size: 25px; padding: 40px 50px; }
  h1 { font-size: 38px; margin-bottom: 14px; }
  table { font-size: 21px; margin: 8px 0; }
  th, td { padding: 4px 10px; }
  ul { margin: 6px 0; }
  li { margin: 2px 0; }
  pre { font-size: 17px; margin: 6px 0; }
---

# Banking Intent Assistant on BANKING77

PCAM ZG516 Applied Generative AI, Group Major Project, Assignment 1

Group _[number]_: _[name, ID]_ · _[name, ID]_ · _[name, ID]_ · _[name, ID]_

Code: https://github.com/2025paml556-bhoomika/banking-agent

<!-- 0:30. One sentence: a customer writes one free-text message, the assistant finds the intent, calls the right tool, and replies. -->

---

# 1. Problem and dataset

- Customer of an online bank writes one message in free text: "I lost my card, please block it"
- The bank must route it to one of **77 intents**, trigger the action, and answer
- Dataset D1, BANKING77: 13,083 messages, 77 labels, fixed train/test split
- Train (10,003 rows): source of examples only. No training
- Two fixed 100-row test samples: **development** (every experiment) and **held-out** (final system, once)
- Why an LLM: paraphrase and typos in, fluent grounded reply out, no training needed, one model for intent + tool call + reply

<!-- 1:00. Point out that the held-out set was never touched until the end. -->

---

# 2. Model and architecture

- `Qwen/Qwen2.5-1.5B-Instruct`, Hugging Face `transformers`, bfloat16, Apple M3 with 8 GB, MPS
- Assignment comparison: causal GRU language model vs. TF-IDF VAE vs. conditional TF-IDF auxiliary-classifier GAN, trained on the same BANKING77 training split
- The causal model learns next-token prediction and has a supervised intent head; the VAE reconstructs TF-IDF features with a latent intent head; the GAN's discriminator intent head classifies TF-IDF feature vectors
- Scores are intent accuracy on the full train and official test splits; input representations and prediction methods differ, so interpret this as an educational family comparison rather than a controlled architecture-only benchmark
- Four stages shown on a real message, "How do I track my card?"
  - Tokenization: 7 tokens
  - Embeddings: shape 1 × 7 × 1536
  - LM head: top-5 next-token probabilities
  - Greedy generation
- Four run-time levers, one experiment each: **prompt, decoding, output schema, tools**

<!-- 1:00. The notebook section 2 prints these four stages live. -->

---

# 3. E1 Baseline and E2 Prompts

| Prompt | Accuracy | No valid intent | Preamble |
|---|---|---|---|
| `zero_shot` (baseline) | 0.41 | 4 | 57 |
| `role` | 0.40 | 4 | 30 |
| `constraint` | 0.40 | 2 | 0 |
| `few_shot` (5 random) | 0.43 | 12 | 1 |
| `few_shot_constraint` | 0.43 | 8 | 0 |

- Classical reference on the same rows: TF-IDF + logistic regression **0.79**
- Instructions fix **format** (preamble 57 → 0), not **knowledge** (accuracy flat at 0.40 to 0.43)
- Failures: near-synonym labels (35), favourite labels (10), invented names (3), refusals (2)

<!-- 1:00. The one message to quote: "track my card" → card_delivery_estimate instead of card_arrival. The names alone do not say where the boundary is. -->

---

# 4. E3 Decoding

| Setting | Accuracy | No valid intent | Flips vs greedy |
|---|---|---|---|
| `greedy` | 0.43 | 8 | - |
| `temperature_0.7` | 0.44 | 10 | 5 |
| `temperature_1.5` | 0.40 | 7 | 13 |
| `top_k_50` | 0.44 | 7 | 7 |
| `top_p_0.9` | 0.38 | 11 | 11 |

- A one-label answer has one right token sequence; sampling can only move away from the best guess
- Flips went both ways; at the strongest settings more right→wrong than wrong→right
- Decision: **greedy for the intent, sampling only for the free-text reply**

<!-- 0:45. -->

---

# 5. E4 Structured output

```json
{"type": "object", "required": ["intent", "confidence"],
 "properties": {"intent": {"type": "string", "enum": ["<77 names>"]},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1}}}
```

| Check | Count |
|---|---|
| Parses as JSON | 100 / 100 |
| Passes the schema | 86 / 100 |
| Accuracy, all rows / schema-valid rows | 0.43 / 0.50 |

- All 14 rejections were **invented intent names**: `get_virtual_card`, `top_up_by_card`, `refund_not_showing_up` (lowercase r)
- The `enum` turns "is this a real intent?" into a hard check. `confidence` was not informative (model copied 0.9)

<!-- 0:45. Hallucination in the classification setting: the model generates a value that looks right instead of selecting one. -->

---

# 6. Adaptation decision: prompting, RAG, and PEFT

| Adaptation | Mechanism | Evidence / status |
|---|---|---|
| Prompting | Instructions and examples in context | Five prompts moved accuracy from 0.40 to 0.43 |
| RAG | Retrieve five similar labelled train rows for each message | 0.84 development accuracy; 0.77 held-out assistant accuracy |
| PEFT (LoRA) | Train low-rank Qwen adapter weights on BANKING77 | Implemented in `evaluation/peft_lora_comparison.py`; full GPU run and scores pending |

- RAG remains the runtime assistant strategy; PEFT is a separately evaluated adaptation option and does not change `main.py`
- Run `uv run -m evaluation.peft_lora_comparison` on a suitable CUDA GPU; it evaluates on the same development and held-out subsets
- Do not claim PEFT accuracy until the full-data run has produced results

<!-- 1:15. Explain the three approaches; identify PEFT results as pending until run. -->

---

# 7. Final system: tools and the agent loop

`handle(message, pending)` in `main.py`, one request end to end:

1. `classify_intent(message)`: retrieve 5 nearest train rows → prompt → greedy → parse → validate against the 77-name `enum`
2. Action intent (17, irreversible)? Ask first: "I understood: lost or stolen card. Shall I go ahead?" Act only on "yes"
3. `handle_intent(intent)`: simulated backend → `done` (reference `REF-nnnn`) / `information` (34) / `unsupported` (26, how to reach a human)
4. Reply: LLM, sampled (temperature 0.7), written from the tool transcript only
5. Error path: no valid intent → ask the customer to rephrase, no tool call

- Both tools have JSON schemas and are passed to the chat template as `tools` (sequence diagram in the report, Section 10)
- **The code fixes the tool order and checks the yes/no.** The 1.5B model driving the loop itself gave 0 of 8 correct sequences: skipped tools, invented `BLOCK_CARD_LOST`, claimed actions not taken
- "Do not claim any action that the tool results do not show"

<!-- 1:00. Live demo here if time: `uv run ui.py`, type "I lost my card, please block it", then "yes". -->

---

# 8. Generative-family comparison and evaluation

| Model | Family | Train accuracy | Test accuracy |
|---|---|---:|---:|
| GRU causal LM + intent head | Autoregressive / causal LM | 0.3112 | 0.2864 |
| TF-IDF VAE + intent head | Variational autoencoder | 0.8588 | 0.7653 |
| Conditional TF-IDF auxiliary-classifier GAN | GAN | 0.0190 | 0.0136 |

- These are the three models compared for the assignment. Qwen is the separate runtime assistant model, not part of this family comparison.
- The causal LM is trained with next-token prediction plus a supervised intent head; the VAE and GAN use supervised intent heads over TF-IDF features.
- LoRA PEFT is implemented as a standalone experiment; full training metrics need a suitable CUDA GPU. Retrieval remains the runtime assistant strategy.
- Failure analysis: invented labels, misleading claims, and unsupported tool arguments are documented with schema and code-controlled safeguards.

<!-- The metrics are the verified six-epoch, seed-17 run in the generated CSV. -->

---

# 9. Responsible use and limits

- Wrong intent → wrong action: schema enum, "rephrase" path, and a confirmation step before irreversible actions in production
- Hallucinated claims: control in code, language in the model; reply generated from the tool transcript
- Prompt injection: the message enters only as data; tool order and arguments are set by code
- Privacy: model runs locally; PII masking before retrieval and logs in production
- Limits: small models; single turn; `handle_intent` is a simulation; TF-IDF retrieval matches words, not meaning

**Reproduce:** `notebooks/banking_agent.ipynb` runs the main assistant experiments. Run `uv run -m evaluation.generative_family_comparison` separately to train and compare the three family models.

<!-- 0:45. Close with the three principles: control in code, validate at the boundary, fail towards a human. -->

---

# Contributions

| Member | Work |
|---|---|
| _[name, ID]_ | _[...]_ |
| _[name, ID]_ | _[...]_ |
| _[name, ID]_ | _[...]_ |
| _[name, ID]_ | _[...]_ |

Questions

<!-- Total: about 9 minutes. -->
