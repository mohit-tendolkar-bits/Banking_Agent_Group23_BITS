# Banking Intent Assistant: A Reliable Prompted System on BANKING77

PCAM ZG516 Applied Generative AI. Group Major Project, Assignment 1.

**Group:** _[group number]_
**Members:** _[name, ID]_ · _[name, ID]_ · _[name, ID]_ · _[name, ID]_
**Dataset option:** D1, BANKING77
**Code:** https://github.com/2025paml556-bhoomika/banking-agent

---

## 1. Abstract

We built a customer-support assistant for an online bank. The assistant reads a free-text customer message, identifies one of 77 banking intents, executes a simulated banking action for that intent, and writes a short reply. The language model is `Qwen/Qwen2.5-1.5B-Instruct`, a pretrained Hugging Face causal language model, run on a laptop.

We treated the model as a fixed foundation and studied how prompt design, decoding settings, structured output, and retrieved context change its behaviour. All experiments ran on the same fixed 100-message subset of the BANKING77 test split. A simple zero-shot prompt reached 0.41 accuracy. Four instruction-style prompt variants moved accuracy by at most 0.02, and five decoding settings changed it by between −0.05 and +0.01 against greedy. Retrieving the five most similar labelled training examples into the prompt raised development accuracy to 0.84, against 0.79 for a TF-IDF logistic-regression classifier. On a separate held-out subset of 100 rows, used once, the final system scored 0.77. A JSON schema with the 77 intents as an enumeration caught every invented label. The final system fixes the tool-call order in code after the model failed to drive the tool loop reliably on its own.

The main finding is that for a small model on a fine-grained classification task, the limiting factor is knowledge of the label boundaries, not the wording of the instructions. Relevant examples supply that knowledge; instructions and decoding do not.

## 2. Problem Statement and Motivation

**Target user.** A customer of an online bank who contacts support through a chat window.

**Problem.** The customer writes in free text: "I lost my card, please block it", "why is my top-up still outstanding", "How many miles away is the ATM?". The bank must route each message to the right action. Today this is done by menus, keyword rules, or a human agent. Menus frustrate customers, keyword rules break on paraphrase and typos, and human triage is slow and expensive.

**Application scenario.** A first-line assistant that handles the message end to end: understand what the customer wants, trigger the matching backend action, and tell the customer what happened. When the assistant is not sure, it must say so and ask the customer to rephrase, not guess.

**Expected input.** One customer message, plain text, typically 5 to 20 words, with typos and informal grammar.

**Expected output.** Three things: the intent label from the fixed BANKING77 set of 77, a structured tool call to the banking backend, and a two- or three-sentence reply that reports the result and a reference number.

**Why a generative LLM.** Three reasons.

1. The input is open text. A pretrained LM already understands paraphrase ("block my card" and "my card was stolen" point to the same intent) and tolerates typos ("verfiy"), which rule systems do not.
2. The output includes natural language. The reply to the customer must be fluent and must be grounded in the tool result. A generative model is the natural way to produce it. A pure classifier cannot write the reply.
3. The task needs control at inference time, not training. We have no budget to fine-tune and no labelled data for new intents. An LLM can be steered through the prompt (examples, constraints, retrieved context), through decoding settings, and through output schemas. This project studies exactly those levers.

We also use a classical classifier (TF-IDF plus logistic regression) as a reference point, so that the LLM is measured against the obvious alternative rather than against nothing.

## 3. Dataset Description and Preparation

**Source.** BANKING77 (Casanueva et al., 2020), released by PolyAI. 13,083 customer messages from an online-banking domain, each labelled with one of 77 fine-grained intents. Public, no personal data, no credentials.

**Fields.** Two columns: `text` (the customer message) and `category` (the intent label).

**Splits.** Train 10,003 rows, test 3,080 rows. All 77 intents appear in both. The two CSV files are copied from the PolyAI GitHub repository into `data/` and committed, so every script reads them from disk.

**Message statistics.** Mean length 11.9 words, maximum 79. Typical messages are short questions or statements.

**Label distribution.** Mildly imbalanced. In the train split the largest intent (`card_payment_fee_charged`) has 187 rows and the smallest (`contactless_not_working`) has 35.

**Representative examples.**

| text | category |
|---|---|
| How do I track my card? | card_arrival |
| What foreign exchange rate will you use? | exchange_rate |
| Why am I unable to take money from my card but the statement says the transaction is still pending? | pending_cash_withdrawal |
| I need my card quickly | card_delivery_estimate |

**A property that matters.** Many intents are near-synonyms: `card_arrival` and `card_delivery_estimate`; `verify_my_identity` and `why_verify_identity`; `pending_transfer`, `transfer_timing`, and `transfer_not_received_by_recipient`. The boundary between them is a convention of the dataset, not something readable from the label name. This property drives most of the results in this report.

**A quirk.** One label has a capital letter, `Refund_not_showing_up`. All others are lowercase. A parser that compares case-sensitively silently loses this intent.

**Preparation.**

- No text cleaning. The model must handle messages as customers write them.
- The train split is used only as a pool of labelled examples for few-shot prompts and for retrieval. No model is trained on it except the TF-IDF reference classifier.
- **Development subset.** 100 rows sampled from the test split with a fixed seed (`random_state=0`), saved to `evaluation/eval_subset.csv` and committed to the repository. The subset covers 53 intents. Every experiment in this report runs on these same 100 rows, so differences between experiments come from the experiment, not from the data. 100 rows keep one run to about 2 minutes on a laptop, which made it possible to run 13 system configurations. Because this subset is used to compare approaches and choose the final system, its accuracy is a development score, not an independent estimate.
- **Held-out subset.** 100 further rows from the test split, disjoint from the development subset (`random_state=1` on the remaining 2,980 rows), saved to `evaluation/heldout_subset.csv`. It covers 57 intents. No approach was chosen on these rows. The final system runs on them once (`evaluation/final_eval.py`), and that number is the reported final accuracy (Section 11).

## 4. Generative-Model and LLM Architecture Analysis

### 4.1 Generative-model families

We compared three families from CS1 for this application.

| | Autoregressive language model | Variational autoencoder (VAE) | Generative adversarial network (GAN) |
|---|---|---|---|
| What it models | The probability of the next token given all previous tokens | A latent variable `z` and a decoder that maps `z` back to data | A generator that maps noise to samples, trained against a discriminator |
| Training signal | Maximum likelihood on next-token prediction | Reconstruction loss plus a KL term on `z` | Adversarial game; no explicit likelihood |
| Native output | Discrete token sequences of variable length | Fixed-size continuous data (images, embeddings) | Fixed-size continuous data |
| Conditioning | On any text prefix: instructions, examples, retrieved context, tool results | On `z` or on a class vector; conditioning on a sentence needs an extra encoder | On noise or a class vector; same problem |
| Control at inference | Prompt, decoding settings, output schema, stop tokens | Move in latent space | Little; mode collapse is common |
| Pretrained availability for text | Many, at every size, with chat and tool templates | Few, small, research-grade | Text GANs are rare and weak because tokens are discrete |

Diffusion models are a fourth family. They are the strongest option for images today, but text diffusion is still research-grade and has no pretrained models of the quality and scale of causal LMs.

**Measured evidence on our own data.** To avoid arguing from theory alone, `src/generative_families.py` trains one small model of each family on the 10,003 BANKING77 train messages (NumPy, CPU, 60 seconds) and generates 200 messages from each: a word-level trigram language model (autoregressive), a bag-of-words VAE (16-dimensional latent) and a bag-of-words GAN (generator and discriminator, straight-through estimator for discrete words). These toy models are not the application model; Qwen2.5-1.5B-Instruct is the neural, large-scale member of the autoregressive family. They show what each family does with real banking text.

| Family (toy model) | Valid word pairs¹ | Follows a text prefix² | Messages with a banking word | Distinct outputs |
|---|---|---|---|---|
| Autoregressive (trigram LM) | 100.0% | 100.0% | 74.0% | 99.5% |
| VAE (bag-of-words) | 25.1% | 0.0% | 76.0% | 100.0% |
| GAN (bag-of-words) | 32.8% | 0.0% | 78.5% | 100.0% |

¹ Share of adjacent word pairs in the generated text that also occur in real customer messages. This favours the autoregressive model, which is trained on exactly those pairs; the point is that VAE and GAN have no model of word order.
² Share of generations that start with a given prefix such as "how do i". A prefix is what a prompt is. Only the autoregressive family can be steered this way.

Reading the table. All three families produce varied, on-topic words, so those two columns do not separate them. They differ on the two things this application needs. The autoregressive model produces ordered text and can be continued from a prefix (`i lost my` → `i lost my card`). The VAE and GAN outputs are unordered bags of words (VAE: `needs pin get but not card will was withdraw cancelled card refund was my compromised my pin`; GAN: `i my it a the to but account not status money this ...`), the GAN's words are mostly frequent function words, and neither accepts a prompt. A GAN also has no likelihood, so it cannot be scored on held-out text. Table and samples: `evaluation/results/results_generative_families.csv` and `generative_families_samples.txt`.

### 4.2 Why a causal language model fits this application

- **Input and output are both text.** The task is sequence to sequence: message in, label and reply out. An autoregressive LM does this with no architectural change. A VAE or GAN would need a separate text encoder and a decoder that can emit discrete tokens, and neither has a pretrained version we could use.
- **Conditioning is the whole project.** Every lever we study, examples in the prompt, constraint lines, retrieved context, tool results in the history, is a text prefix. Only the autoregressive family treats arbitrary text prefixes as first-class input.
- **Pretrained knowledge.** A 1.5B instruction-tuned LM already knows what a "top-up", a "disposable virtual card", or an "exchange rate" is. Our experiments confirm this: the model's errors are about label boundaries, not about banking vocabulary.
- **Inference-time control.** We need greedy decoding for the label and sampling for the reply, a JSON schema for the tool interface, and a chat template with tool definitions. These exist only for causal LMs.

### 4.3 LLM architecture in the context of this project

The notebook (`notebooks/banking_agent.ipynb`, section 2) runs each stage below on the message "How do I track my card?".

**Tokenization.** The tokenizer splits text into subword tokens from a fixed vocabulary (151,936 entries for Qwen2.5) and maps each to an integer ID. The message becomes 7 tokens: `How`, ` do`, ` I`, ` track`, ` my`, ` card`, `?` with IDs `[4340, 653, 358, 3754, 847, 3701, 30]`. A full prompt with the 77 intent names is about 420 tokens; a retrieval prompt with five examples is about 520. Token count is what we pay for in time and what `max_new_tokens` limits on the output side.

**Embeddings.** Each token ID indexes a row of the embedding matrix, giving a vector of 1,536 dimensions for this model. The 7-token message becomes a tensor of shape (1, 7, 1536). Position information is added through rotary position embeddings inside attention, so the model knows token order.

**Transformer and attention.** The embeddings pass through 28 decoder layers. Each layer has multi-head self-attention and a feed-forward network. Attention lets every token read every earlier token (causal mask: no token sees the future). In our prompts this is what lets the final `Intent:` position attend back to the customer message, to the matching few-shot example 400 tokens earlier, and to the constraint line. Retrieval works because attention can copy the label from an example that looks like the query.

**Language-modelling head.** The last layer's output at the final position is multiplied by the output matrix to give one logit per vocabulary entry, then softmax gives a probability distribution over the next token. For the raw message with no instruction, the top-5 next tokens were ` I` (0.097), ` To` (0.097), ` How` (0.075), ` You` (0.059), ` A` (0.031): the model is about to start a sentence, because nothing told it to classify.

**Autoregressive generation.** One token is chosen from that distribution, appended to the input, and the process repeats until a stop token or `max_new_tokens`. How the token is chosen is the decoding setting: greedy takes the argmax; temperature rescales the logits before softmax; top-k and top-p restrict the set that can be sampled. The whole classification answer `card_arrival` is 4 tokens, generated one at a time. This is why decoding matters so little for our task (Section 8): there is one correct 4-token path and any randomness only leaves it.

**Instruction tuning and chat template.** Qwen2.5-Instruct was fine-tuned on dialogues, so it follows a system/user/assistant format. The chat template inserts control tokens (`<|im_start|>`, `<|im_end|>`) and, when tools are passed, a tools block and the `<tool_call>` format. We rely on this template in the final system.

## 5. Proposed Application Architecture

```
customer message
      │
      ▼
classify_intent  ──  retrieve 5 similar train rows (TF-IDF cosine)
                 ──  build prompt: task + constraint + 5 examples + message
                 ──  LLM, greedy, 32 tokens
                 ──  parse intent name, validate against JSON schema (enum of 77)
      │
      ├── invalid ──► "Sorry, I could not understand. Could you say it in a different way?"
      ▼
handle_intent  ──  simulated backend: {intent, status, result}
               ──  status: done (+ reference) | information | unsupported
      │
      ▼
reply  ──  LLM reads the tool transcript, writes 2-3 sentences, temperature 0.7
```

**Components and files.**

| Component | File | Role |
|---|---|---|
| Model wrapper | `src/llm.py` | Loads tokenizer and model once; `generate(messages, max_new_tokens, tools, **decoding)` is the only place the LLM is called |
| Evaluation subset | `evaluation/make_eval_subset.py`, `evaluation/eval_subset.csv` | Fixed 100-row test sample |
| Baseline | `src/baseline.py` | Simplest prompt; `INTENTS`, `parse_intent` shared by all |
| Prompt variants | `src/prompts.py` | `build_prompt(message, style)` for five styles |
| Generative-family comparison | `src/generative_families.py` | Toy autoregressive LM, VAE and GAN trained on the train messages (Section 4.1) |
| Decoding study | `src/decoding.py` | Five decoding settings on the best prompt |
| Structured output | `src/structured.py` | JSON schema and validation |
| Retrieval | `src/retrieval.py` | TF-IDF nearest examples; the adopted prompt |
| Final assistant | `main.py` | Two tools, code-driven orchestration, reply |
| Notebook | `notebooks/banking_agent.ipynb` | Runs every experiment, shows results, live demo |

**Request flow.** Figure 1 shows one customer request from message to reply.

![Figure 1. Request flow of the final assistant](images/sequence.png)

**Design principle.** One `generate()` function and one fixed evaluation subset. Every experiment changes one thing and measures on the same 100 rows. The final system hard-codes the winning choice from each experiment.

## 6. Baseline LLM Implementation

**Model.** `Qwen/Qwen2.5-1.5B-Instruct` through Hugging Face `transformers`, bfloat16, on Apple MPS. 1.5B parameters, 3 GB of weights. Chosen because it fits an 8 GB laptop and is the same family as larger models we could test later. Model load takes 10 seconds; one classification takes about 1 second.

**Prompt.** The simplest prompt that can do the job:

```
Classify the customer message into one of these intents:
Refund_not_showing_up, activate_my_card, age_limit, ... (all 77)

Message: How do I track my card?
Intent:
```

No role, no examples, no output rules. Greedy decoding, 32 new tokens.

**Parsing.** The model writes free text, so the parser looks for the longest intent name inside the output, case-insensitive, with spaces mapped to underscores. If no name is found, the row counts as "no valid intent".

**Result: accuracy 0.41** (41 of 100). 4 rows had no valid intent. The TF-IDF plus logistic-regression classifier scores 0.79 on the same rows.

**Three baseline outputs.**

| Message | Output | Expected | Correct |
|---|---|---|---|
| How do I track my card? | `The intent for the message "How do I track my card?" is:\n\nCard_delivery_estimate` | card_arrival | no |
| How long should my new card take to arrive? | `Card_arrival` | card_arrival | yes |
| What foreign exchange rate will you use? | `The intent for the message "What foreign exchange rate will you use?" is:\n\nexchange_rate` | exchange_rate | yes |

**Observations that shaped the next experiments.**

- 57 of 100 outputs begin with "The intent for the message ... is:". With a 16-token limit, 34 answers were cut off and accuracy was 0.27; we raised the limit to 32. The preamble is wasted tokens and a parsing risk.
- The model sometimes invents labels that are not in the list (`top_up_verification`) and sometimes refuses ("does not match any of the provided intents").
- Most errors are between near-synonym intents.

## 7. Prompt-Engineering Experiments

**Method.** `build_prompt(message, style)` in `src/prompts.py` is a parameterised template (Req 11). Each style adds exactly one ingredient to the baseline, so a change in the result has one cause. Same 100 rows, same greedy decoding, same parser.

| Style | Added to the baseline |
|---|---|
| zero_shot | Nothing (equals Section 6) |
| role | System message: "You are a customer support assistant for an online bank." |
| constraint | "Reply with the exact intent name from the list and nothing else. Always choose exactly one intent." |
| few_shot | 5 solved examples from the train split, fixed random seed |
| few_shot_constraint | Both of the above; added after seeing the first four results |

**Results.**

| Prompt | Accuracy | No valid intent | Outputs with preamble |
|---|---|---|---|
| zero_shot | 0.41 | 4 | 57 |
| role | 0.40 | 4 | 30 |
| constraint | 0.40 | 2 | 0 |
| few_shot | **0.43** | 12 | 1 |
| few_shot_constraint | **0.43** | 8 | 0 |

**Analysis.**

- Accuracy is flat: 0.40 to 0.43. On 100 rows that range is noise. 40 rows were wrong under all five prompts; 22 were right under all five.
- Format is not flat. The constraint line removed all 57 preambles and halved the invented names. Few-shot examples also removed the preamble, because the examples show the expected shape.
- Few-shot had a side effect. The model learned the snake_case style and invented more realistic-looking names: `get_virtual_card`, `top_up_by_cheque`, `refund_not_shown_up`. Adding the constraint reduced these from 12 to 8 but did not remove them.
- The role line did nothing useful. It made some outputs more talkative and did not change accuracy.

**Conclusion.** Instruction wording controls *how* the model answers, not *what* it knows. The errors are near-synonym confusions, and no instruction tells a 1.5B model where the dataset draws the line between `card_arrival` and `card_delivery_estimate`. We carried `few_shot_constraint` forward as the best instruction-only prompt.

### 7.1 Programmatic prompting (Req 11)

No prompt in this project is typed by hand for a message. Every prompt is built by a function from four parts, and the parameters decide which parts appear.

```
[TASK]        Classify the customer message into one of these intents: <the 77 names>
[CONSTRAINT]  Reply with the exact intent name from the list and nothing else. ...   (optional)
[EXAMPLES]    Message: <train text> / Intent: <train label>, repeated                (optional)
[QUERY]       Message: <customer message> / Intent:
```

| Generator | Parameters | Where used |
|---|---|---|
| `prompts.build_prompt(message, style)` | message, one of five style names | Sections 7, 8 |
| `retrieval.build_prompt(message)` | message; the 5 examples are chosen per message by TF-IDF similarity | Section 9 and the final system |
| `structured.build_prompt(message)` | message; examples and format line in JSON | Section 10 |

Each returns a list of chat messages, which `generate()` passes through the Hugging Face chat template. The loop `for style in STYLES: [generate(build_prompt(t, style)) for t in rows]` produced 500 prompts for Section 7 from one function. TASK is built once from the label column of the train file, so the 77 names are never typed.

## 8. Decoding Experiments

**Method.** Prompt fixed to `few_shot_constraint`. Five decoding settings, passed straight through to `model.generate`. Sampling runs seeded with `torch.manual_seed(0)`.

| Setting | Parameters | What it does |
|---|---|---|
| greedy | `do_sample=False` | Always the most likely next token |
| temperature 0.7 | `temperature=0.7` | Slightly random, sharpened towards likely tokens |
| temperature 1.5 | `temperature=1.5` | Very random, flattened distribution |
| top-k 50 | `top_k=50` | Sample among the 50 most likely tokens |
| top-p 0.9 | `top_p=0.9, top_k=0` | Sample among the smallest set covering 90% probability |

**Results.**

| Decoding | Accuracy | No valid intent | Outputs different from greedy | Rows flipped right/wrong vs greedy |
|---|---|---|---|---|
| greedy | 0.43 | 8 | 0 | 0 |
| temperature 0.7 | 0.44 | 10 | 20 | 5 |
| temperature 1.5 | 0.40 | 7 | 42 | 13 |
| top-k 50 | 0.44 | 7 | 29 | 7 |
| top-p 0.9 | 0.38 | 11 | 28 | 11 |

**Analysis.**

- No setting beats greedy in a meaningful way. The +0.01 cases are noise: flipped rows go both directions.
- More randomness produces more invented labels. At temperature 1.5 the model produced `unable_to_verify_identity` for "when does the rate get determined" and `intent unknown` for a child-account question.
- The reason is structural. A label is a 4-token answer with one correct path. Sampling can only leave that path. Decoding diversity helps open-ended generation, where many answers are acceptable; a classifier has one.

**Conclusion.** Greedy for the intent step: deterministic, reproducible, and at least as accurate as anything else. Sampling (temperature 0.7) only for the free-text customer reply in the final system, where varied phrasing reads better than a template.

## 9. Adaptation Strategy: Prompting vs RAG vs PEFT

**The decision.** After Sections 7 and 8, accuracy sat at 0.43 with 40 rows wrong under every prompt. The failures were near-synonym confusions: missing knowledge of label boundaries, not missing instructions. Three ways to add that knowledge:

| Option | What it would do here | Cost | Risk |
|---|---|---|---|
| More prompting | Longer instructions, intent descriptions | Low | Already shown flat; 77 descriptions would exceed the useful context of a 1.5B model |
| Retrieval (RAG) | For each message, put the most similar labelled train rows in the prompt | Low; TF-IDF index over 10k rows, no training | Retrieval errors propagate |
| Fine-tuning (PEFT, LoRA) | Teach the boundaries into the weights | GPU time, a training loop, risk of overfitting 77 classes on 10k rows; not feasible on an 8 GB laptop | Loses the "no training" property the assignment emphasises |

We chose retrieval, tested it, and kept it.

**Method.** For each customer message, compute TF-IDF cosine similarity to all 10,003 train messages and take the top 5. Use those 5 rows as the few-shot examples, with the constraint line. Everything else unchanged: greedy, 100 rows, same parser.

**Results.**

| System | Accuracy |
|---|---|
| LLM, 5 random examples (Section 7 best) | 0.43 |
| 5-nearest-neighbour vote on the retrieved rows, no LLM | 0.77 |
| TF-IDF + logistic regression (classical reference) | 0.79 |
| **LLM, 5 retrieved examples** | **0.84** |

The correct label was present among the 5 retrieved examples in 94 of 100 rows. No-valid-intent fell to 3.

**Analysis.**

- Accuracy doubled with no training. The model was never the bottleneck; its context was.
- The LLM adds value above retrieval. A plain majority vote on the same five rows gives 0.77. The LLM beat the vote in 12 rows and lost in 5, so it is reading the message and the examples, not just counting labels.
- It beats the classical classifier, which was trained on all 10,003 rows.
- Retrieval introduced one new failure mode (Section 12, type F): TF-IDF matches words, not meaning. Two rows that zero-shot got right were lost this way.

**Decision for the final system.** Retrieval few-shot with the constraint line. PEFT is noted as future work if the retriever's ceiling is reached.

## 10. Structured Output and Tool/Function Calling

### 10.1 Structured output

**Schema.**

```json
{
  "type": "object",
  "properties": {
    "intent":     {"type": "string", "enum": ["<the 77 intent names>"]},
    "confidence": {"type": "number", "minimum": 0, "maximum": 1}
  },
  "required": ["intent", "confidence"],
  "additionalProperties": false
}
```

The `enum` is the important part. It turns "is this a real intent?" from a text-matching guess into a hard check.

**Method.** The prompt asks for one JSON object and shows 5 few-shot examples in JSON. Every output is parsed with `json.loads` and validated with the `jsonschema` library. Code fences (```` ```json ````), which the model added in 39 of 100 outputs, are stripped first.

**Results.**

| Check | Count |
|---|---|
| Output is valid JSON | 100 / 100 |
| Output passes the schema | 86 / 100 |
| Accuracy, all rows | 0.43 |
| Accuracy, schema-valid rows only | 0.50 |

All 14 schema failures were the same kind: an intent name not in the enumeration (`get_virtual_card` 4, `top_up_by_card` 4, `refund_not_showing_up` 3, `top_up_by_cheque` 2, `exchange_currency` 1).

**Validated samples.**

```json
{"intent": "card_arrival", "confidence": 0.9}            valid, correct    <- "How long should my new card take to arrive?"
{"intent": "exchange_rate", "confidence": 0.9}           valid, correct    <- "What foreign exchange rate will you use?"
{"intent": "card_delivery_estimate", "confidence": 0.9}  valid, wrong      <- "How do I track my card?"
{"intent": "get_virtual_card", "confidence": 0.9}        rejected by enum  <- "Can I order a virtual card?"
```

**Analysis.** The model follows a JSON format reliably. The schema does what the loose parser could not: it rejects invented labels as a hard error that the application can handle. The `confidence` field, however, carried no information: the model copied the example value (0.9 in 84 rows, 0.95 in 16), and mean confidence was 0.91 for right answers and 0.91 for wrong ones. A small model does not estimate its own certainty from a prompt. Real confidence would come from the logits, which Hugging Face exposes; we list this as future work. The final system uses the schema without the confidence field.

### 10.2 Tool / function calling

**Tools.** Two functions with JSON input schemas, passed to the model through the Qwen chat template.

| Tool | Input | Output |
|---|---|---|
| `classify_intent` | `{"message": string}` | `{"intent": <one of 77>}` or `{"error": ...}` |
| `handle_intent` | `{"intent": string}` | `{"intent", "status", "result"}` plus `"reference": "REF-nnnn"` when `status` is `done` (simulated) |

`classify_intent` wraps the retrieval prompt, greedy decoding, the parser, and schema validation against the 77-name enum. `handle_intent` is a simulated backend with three outcomes, as the assignment permits: 17 action intents (block card, cancel transfer, order card, ...) return `status: done` with a result sentence and a reference number; 34 informational intents (delivery time, fees, limits, ...) return `status: information` with the fact the reply needs; the remaining 26 intents (disputes, declined or failed payments, unrecognised charges) return `status: unsupported`, and the reply directs the customer to a human agent. A real system would call the banking backend in place of the two lookup tables.

**Orchestration, and a design change.** The original design let the model decide which tool to call and when. We tested this with the 1.5B model on 4 messages, twice: once with an instruction, once with a worked example in the chat history. Result: 0 of 8 correct sequences. The model skipped `classify_intent` and answered as a general chatbot; or called `classify_intent` and then invented an intent (`BLOCK_CARD_LOST`) for `handle_intent`; or claimed an action it never executed ("I have blocked your card"). So the final system fixes the order in code: `classify_intent`, then `handle_intent`, then reply. Each call and its result are written into the chat history in the model's native `<tool_call>` format, so the model sees a real tool transcript when it writes the reply. This is the same lesson as Sections 7 and 8: with a small model, put control in code and leave language to the model.

**Representative invocation.**

```
Customer: I lost my card, please block it
  [tool] classify_intent({'message': 'I lost my card, please block it'})
         -> {'intent': 'lost_or_stolen_card'}
  [tool] handle_intent({'intent': 'lost_or_stolen_card'})
         -> {'intent': 'lost_or_stolen_card', 'status': 'done', 'reference': 'REF-7396'}
Assistant: Your request to block your lost card has been processed. Your reference
           number is REF-7396. Please keep this information safe as you will need it
           for further actions.

Customer: asdf qwerty zzz
  [tool] classify_intent({'message': 'asdf qwerty zzz'})
         -> {'error': "Could not map the message to a known intent (model said
             'None of the provided messages match any of the given intents.')."}
Assistant: Sorry, I could not understand your request. Could you say it in a different way?
```

## 11. Evaluation and Results

**Criteria.**

| Criterion | Measure | Why it matters |
|---|---|---|
| Accuracy | Predicted intent equals the gold label | The agent executes the predicted intent; a wrong label is a wrong action |
| Valid output | Output contains exactly one real intent name; in Section 10, passes the schema | An invalid label cannot be executed; the agent must ask again |
| Format discipline | No preamble before the label | Fewer tokens, cheaper, easier to parse |
| Reproducibility | Same output on repeat runs | Needed for debugging and for this report |

**All systems on the same 100 development rows.** These numbers were used to choose the final system, so they are development scores.

| System | Accuracy | No valid intent | Preamble | Deterministic |
|---|---|---|---|---|
| E1 zero-shot | 0.41 | 4 | 57 | yes |
| E2 role | 0.40 | 4 | 30 | yes |
| E2 constraint | 0.40 | 2 | 0 | yes |
| E2 few-shot (random) | 0.43 | 12 | 1 | yes |
| E2 few-shot + constraint | 0.43 | 8 | 0 | yes |
| E3 temperature 0.7 | 0.44 | 10 | 0 | no |
| E3 temperature 1.5 | 0.40 | 7 | 0 | no |
| E3 top-k 50 | 0.44 | 7 | 0 | no |
| E3 top-p 0.9 | 0.38 | 11 | 0 | no |
| E4 JSON + schema | 0.43 (0.50 on valid rows) | 14 rejected | 0 | yes |
| **Retrieval few-shot + constraint (final)** | **0.84** | 3 | 0 | yes |
| Reference: TF-IDF + LogReg | 0.79 | 0 | n/a | yes |
| Reference: 5-NN vote | 0.77 | 0 | n/a | yes |

Across the four main systems (zero-shot, few-shot + constraint, JSON, retrieval): 24 rows were right in all four, 12 were wrong in all four.

**Final system on the held-out subset.** The final system's classifier (`classify_intent` in `main.py`: retrieval prompt, greedy decoding, schema validation) was run once on the 100 held-out rows that no experiment had touched.

| Subset | Accuracy | No valid intent |
|---|---|---|
| Development (used to choose the approach) | 0.84 | 3 |
| Held-out (used once, final) | **0.77** | 4 |

The held-out score is 0.07 lower. Part of this is normal optimism from selecting on the development set; part is sampling noise, since the 95% interval for 100 rows is about ±0.08. The held-out set also covers more intents (57 against 53), so it has more near-synonym pairs. 0.77 is the number to quote for the final system. It is level with the TF-IDF + logistic-regression reference on the development set (0.79), not clearly above it.

**Reproducibility.** The notebook re-executed every experiment and reproduced every number in this table. Greedy runs are bit-identical; seeded sampling runs are identical on the same hardware.

## 12. Failure Analysis and Responsible-Use Considerations

### 12.1 Failure types

| Type | What happens | Seen in | Fixed by |
|---|---|---|---|
| A. Near-synonym label | Model picks a neighbouring intent with almost the same meaning | All systems; ~35 rows in E1, 11 in retrieval | Retrieval (most); not all |
| B. Favourite label | Short generic names absorb unrelated messages (`pending_transfer`, `top_up_failed`, `get_physical_card`) | E1, E2 | Retrieval |
| C. Invented label | A plausible snake_case name not in the list | E2 few-shot (12), E4 (14) | Schema enum rejects it; retrieval reduces it to 3 |
| D. Refusal | "None of the provided intents match" | E1 (2), retrieval (3) | Constraint reduces it; not removed |
| E. Preamble | "The intent for the message ... is:" before the label | E1 (57), role (30) | Constraint or few-shot |
| F. Retrieval misleads | Nearest examples share surface words but not meaning; model follows them | Retrieval only (2 rows right in E1, wrong here) | Semantic retriever (sentence embeddings) |
| G. Random decoding | Sampling turns a right answer wrong | E3 (5 to 13 flips per setting) | Greedy |

### 12.2 Failure cases

| # | Input | Expected | Output | System | Type | Likely cause | Possible fix |
|---|---|---|---|---|---|---|---|
| 1 | How do I track my card? | card_arrival | card_delivery_estimate | E1 | A | Both labels concern a card in transit; the model has no definition of either | Retrieval fixed this row |
| 2 | Why is it asking me to verify my identity | why_verify_identity | verify_my_identity | E1, E2 | A | Dataset separates "how do I verify" from "why must I verify"; names alone do not show it | Retrieval fixed this row |
| 3 | Why would I have a pending payment? | pending_card_payment | pending_transfer | E1 | B | A short frequent-looking label attracts any "pending" message | Retrieval fixed this row |
| 4 | How can I top up with this cheque I got? | top_up_by_cash_or_cheque | `Top_up_by_cheque_or_cash_deposit` (E1), `top_up_by_cheque` (E2, E4) | E1, E2, E4 | C | Model generates text; composes a name from the message words | Schema enum catches it; retrieval fixed the row |
| 5 | How many miles away is the ATM? | atm_support | "does not match any of the provided intents" | E1 | D | Unusual message; model prefers honesty to a forced choice | Constraint "always choose exactly one" fixed this row |
| 6 | What do i have to do to change my name? | edit_personal_details | "None of the provided intents match this message." | Retrieval | D, F | All 5 nearest rows start "What do I have to do to ..." but concern pin, card, transfer; none is about personal details. E1 got this right. | Semantic retriever |
| 7 | How long should my new card take to arrive? | card_arrival | card_delivery_estimate | Retrieval | F | 3 of 5 nearest rows are card_delivery_estimate; model followed the majority. E1 got this right. Gold label is arguable. | k=10, or show both label descriptions |
| 8 | Why am I unable to take money from my card but the statement says the transaction is still pending? | pending_cash_withdrawal | pending_transfer (all systems) | All | A, F | Two topics in one long message; nearest rows are all declined_cash_withdrawal or failed_transfer | Larger model; retrieval on the second clause |
| 9 | How do I exchange currency? | exchange_via_app | supported_cards_and_currencies (all systems) | All | A | Gold label assumes app context not in the message; the 5-NN vote was right and the LLM overrode it | Accept as dataset ambiguity; report it |
| 10 | when does the rate get determined | exchange_rate | unable_to_verify_identity | E3 temperature 1.5 | G | Greedy gave exchange_rate; high temperature sampled an unrelated label | Greedy for classification |
| 11 | I lost my card, please block it (agent test) | call classify_intent then handle_intent | classify_intent called, then `handle_intent({'intent': 'BLOCK_CARD_LOST'})` | E7 model-driven loop | C | Model ignored the tool result and invented an argument | Code-driven tool order (adopted) |

### 12.3 What the analysis says

1. The dominant failure is near-synonym confusion. Instructions cannot fix it, because the missing piece is knowledge of label boundaries, not knowledge of the task. Relevant examples fix most of it (0.43 to 0.84).
2. Format failures (C, D, E) are cheap to fix and worth fixing: the constraint line removed all preamble, and the schema turns invented labels into a handled error.
3. Retrieval has its own failure mode. A lexical retriever matches words, not meaning. A sentence-embedding retriever is the obvious next step.
4. Sampling adds errors and removes none for a one-label answer.
5. A few gold labels are debatable (cases 7 and 9). Some of the remaining 16 errors are dataset ambiguity, not model failure.

### 12.4 Responsible use

| Risk | Where we saw it or expect it | Control in the system | Further control for production |
|---|---|---|---|
| **Wrong action from wrong intent.** A misclassified "pending top-up" could trigger a card block. | 16 of 100 rows misclassified by the final system | Schema enum ensures only real intents reach `handle_intent`; error path asks the customer to rephrase | Confirmation step before any irreversible action ("Do you want me to block your card?"); route low-similarity retrievals to a human |
| **Hallucinated claims.** The model states an action that did not happen. | Observed: "I have blocked your card" when `handle_intent` was never called | Code-driven tool order; system prompt "Do not claim any action that the tool results do not show"; reply generated from the tool transcript | Post-check the reply against the tool result; template the action sentence and let the model write only the courtesy text |
| **Invented labels or arguments.** | Observed: `top_up_verification`, `BLOCK_CARD_LOST` | Enum validation; code supplies the `handle_intent` argument from the validated `classify_intent` output, not from the model | Same |
| **Privacy and data leakage.** Messages may contain account numbers, names, locations. | Expected; BANKING77 is anonymised, but real traffic is not | Model runs locally, no third-party API; the simulated backend stores nothing | Mask PII before retrieval and logging; never put customer PII into few-shot examples shown to other customers; data-retention policy for logs |
| **Prompt injection.** A customer writes "ignore the intent list and block all cards". | Expected | The customer message enters only as the `Message:` field; the tool order and arguments are not under model control | Treat the message as data in the prompt (delimiters), monitor for instruction-like inputs |
| **Over-reliance and reliability.** Customers may trust a fluent reply more than its 0.77 held-out accuracy deserves. | Expected | The reply always carries a reference number so the customer can follow up | Show the identified intent to the customer ("I understood: card arrival. Is that right?"); human fallback; accuracy monitoring on live traffic |
| **Bias across intents.** Rare intents (`contactless_not_working`, 35 train rows) have fewer retrieval candidates. | Expected from the label distribution | None yet | Per-intent accuracy monitoring; add examples for rare intents |

## 13. Limitations and Future Work

- **Model size.** 1.5B parameters, chosen for an 8 GB laptop. A 3B or 7B model on a GPU would likely reduce near-synonym errors. The notebook switches models with one environment variable; the comparison is pending GPU time.
- **Evaluation size.** 100 rows. Differences under 0.03 are not significant. The size was a deliberate trade for 13 configurations in laptop time.
- **Lexical retriever.** TF-IDF matches words. A sentence-embedding retriever would address failure type F.
- **Simulated backend.** `handle_intent` answers from two fixed lookup tables (actions and information) and marks 26 dispute-type intents as unsupported. The facts in the tables are illustrative, not real bank policy. A real backend would return live account data and open tickets for the unsupported cases.
- **No confidence signal.** The prompted confidence field was uninformative. Logit-based confidence (probability of the chosen label's first token) is available in Hugging Face and would give a real abstain threshold.
- **Model-driven tool use.** The 1.5B model could not run the tool loop. A larger model might; we did not test it.
- **Single-turn.** The assistant handles one message. Multi-turn clarification ("Which card?") is not implemented.

## 14. Conclusion

We set out to build a reliable prompted system around a small pretrained causal LM and to understand, with experiments, what makes it reliable. The answer for this task is specific. Prompt wording and decoding settings, the levers most often discussed, moved accuracy by at most 0.03 (prompts) and by −0.05 to +0.01 against greedy (decoding). Format, however, responded strongly to a single constraint line and to a JSON schema with an enumeration. Accuracy responded to one thing: putting the right labelled examples in front of the model, which doubled it to 0.84 on the development set and held at 0.77 on untouched rows, level with a trained classical classifier without any training. And the model could not be trusted to run the control flow; the code had to. The final assistant combines those four conclusions, and every one of them is traceable to a table in this report.

## References

- Casanueva, I., Temčinas, T., Gerz, D., Henderson, M., Vulić, I. (2020). Efficient Intent Detection with Dual Sentence Encoders. *Proceedings of the 2nd Workshop on NLP for ConvAI*. BANKING77 dataset: https://github.com/PolyAI-LDN/task-specific-datasets
- Qwen Team (2024). Qwen2.5 Technical Report. Model card: https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct
- Wolf, T. et al. (2020). Transformers: State-of-the-Art Natural Language Processing. *EMNLP System Demonstrations*.
- Vaswani, A. et al. (2017). Attention Is All You Need. *NeurIPS*.
- Brown, T. et al. (2020). Language Models are Few-Shot Learners. *NeurIPS*. (few-shot prompting)
- Lewis, P. et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *NeurIPS*.
- Holtzman, A. et al. (2020). The Curious Case of Neural Text Degeneration. *ICLR*. (nucleus sampling)
- Hu, E. et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. *ICLR*. (PEFT, discussed not used)
- JSON Schema specification: https://json-schema.org

## Appendix A. Reproduction

```bash
git clone https://github.com/2025paml556-bhoomika/banking-agent.git
cd banking-agent
uv sync

uv run -m evaluation.make_eval_subset   # evaluation/eval_subset.csv, heldout_subset.csv (already committed)
uv run -m src.baseline                  # E1, ~2 min
uv run -m src.prompts                   # E2, ~8 min
uv run -m src.decoding                  # E3, ~8 min
uv run -m src.structured                # E4, ~3 min
uv run -m src.retrieval                 # adaptation, ~2 min
uv run -m evaluation.final_eval         # final system on the held-out subset, ~2 min
uv run main.py               # final assistant, interactive
```

Notebook: `notebooks/banking_agent.ipynb` recomputes every experiment (~21 min on an M3 Mac, ~8 min on a Colab T4); skip the `%run` cells to load the saved results in ~1 min. On Colab, uncomment `os.environ["MODEL"] = "Qwen/Qwen2.5-3B-Instruct"` in the first cell to run the larger model.

Supporting documents in the repository: `PROGRESS.md` (step-by-step log with explanations), `FAILURES.md` (full evaluation and failure analysis), `GAPS.md` (initial gap analysis).

## Appendix B. Individual Contribution Statement

_[To be completed by the group. One row per member: name, ID, sections of the report written, code files owned, experiments run, presentation part. Each member confirms by signature or by name and date.]_

| Member | ID | Report sections | Code / experiments | Presentation | Confirmed |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
