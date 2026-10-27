# Presentation script

Slides: `SLIDES.md` / `SLIDES.pptx`. Target: 9 minutes, 4 speakers. Times are per slide.

| Member | Slides | Time |
|---|---|---|
| 1 | 1, 2, 3: title, problem and dataset, model and architecture | 2:15 |
| 2 | 4, 5: baseline and prompts, decoding | 2:00 |
| 3 | 6, 7: structured output, adaptation decision | 2:15 |
| 4 | 8, 9, 10, 11: final system, evaluation, responsible use, close | 2:40 |

---

## Member 1

### Slide 1, Title (0:20)

Good morning. We are group _[number]_. Our project is a banking intent assistant built on the BANKING77 dataset.

A customer writes one message in free text. The assistant finds the intent, calls the right tool, and replies. We built it with one small language model, with no training. Everything we show is in the GitHub repository on the slide.

### Slide 2, Problem and dataset (1:00)

The user is a customer of an online bank, writing in a support chat. A message like "I lost my card, please block it". The bank must route that message to one of 77 intents, trigger the matching action, and answer the customer.

Today this is done with menus, keyword rules, or a human. Menus frustrate, rules break on paraphrase and typos, humans are slow.

The dataset is BANKING77: 13,083 messages, 77 labels, with a fixed train and test split. We used the train split, about 10,000 rows, only as a source of examples. We never trained on it.

For evaluation we took two fixed samples of 100 rows from the test split. The development set: every experiment runs on it, and it chose the final system. And a held-out set, 100 different rows, that no experiment touched. The final system runs on it once, at the end. We will come back to that number.

Why a language model and not a classifier? Three reasons. Free text with paraphrase and typos comes in. A fluent reply that is grounded in the tool result goes out. And one model does all three jobs: intent, tool call, and reply.

### Slide 3, Model and architecture (0:55)

The model is Qwen 2.5, 1.5 billion parameters, instruction tuned. It runs on a laptop with 8 gigabytes of memory through Hugging Face transformers.

The assignment's model comparison is between three generative families trained on BANKING77: a causal GRU language model, a TF-IDF VAE, and a conditional TF-IDF GAN. The causal model learns next-token prediction and also has a supervised intent head. The VAE reconstructs TF-IDF features and classifies from its latent representation. The GAN uses the discriminator's auxiliary intent head. The default run's test accuracies are 0.2864 for the causal LM, 0.7653 for the VAE, and 0.0136 for the GAN; the corresponding training accuracies are 0.3112, 0.8588, and 0.0190.

These are not identical model designs: one uses token sequences while the other two use TF-IDF vectors, and their prediction heads differ. So we treat the results as an educational generative-family comparison on one shared classification task, not as a perfectly controlled comparison. Qwen is the separate runtime assistant, not a competing row in this comparison.

Those run-time levers are the structure of the project. Four levers, four experiments: the prompt, the decoding settings, the output schema, and tool calling. Member 2 starts with the prompt.

---

## Member 2

### Slide 4, E1 Baseline and E2 Prompts (1:10)

The baseline is the simplest prompt: the task line, the 77 intent names, and the message. Greedy decoding. Accuracy was 0.41.

For comparison, a classical TF-IDF plus logistic regression classifier, trained on the full train file, scores 0.79 on the same 100 rows. That is the bar the LLM must beat.

In experiment 2 we built a parameterised prompt function and tested five styles. A system role. A constraint line: "reply with the exact intent name and nothing else". Five random solved examples. And examples plus constraint.

Look at the two columns on the right. The constraint line removed all preamble text, 57 outputs down to 0, and cut the invented names. But accuracy did not move: 0.40 to 0.43 across all five.

The lesson: instructions fix format, not knowledge. The failures were near-synonym labels. "Track my card" went to card delivery estimate instead of card arrival. The model has only the two names. Nothing tells it where the dataset draws the line between them. No instruction can supply that.

### Slide 5, E3 Decoding (0:50)

Experiment 3 keeps the best prompt and changes the decoding: greedy, temperature 0.7, temperature 1.5, top-k 50, top-p 0.9. The sampling runs are seeded, so they are repeatable.

No setting beat greedy in a meaningful way, 0.38 to 0.44. The last column shows how many answers each setting changed against greedy, 5 to 13 rows. The flips went both ways, and at the two strongest settings more went right to wrong than wrong to right.

The reason is simple. A one-label answer has one correct token sequence. Randomness can only move away from the model's best guess.

So the decision: greedy for the intent, and sampling only for the free-text reply, where variety is welcome. Member 3 takes the output schema.

---

## Member 3

### Slide 6, E4 Structured output (0:55)

In experiment 4 the model answers in JSON: an intent and a confidence. Every output is validated with jsonschema. The key part of the schema is on the slide: the intent field is an enum of the 77 names.

All 100 outputs parsed as JSON. 86 passed the schema. All 14 rejections were invented intent names. "get virtual card" instead of "getting virtual card". "top up by card" instead of "topping up by card". The model builds a name that looks right instead of selecting a real one. This is hallucination in a classification setting.

The enum turns "is this a real intent?" into a hard check. Accuracy on the 86 valid rows was 0.50, against 0.43 overall. The confidence field was not useful: the model copied the 0.9 from the example for almost every row.

### Slide 7, Adaptation decision (1:20)

This is the key adaptation decision. After experiments 2 and 3 we were at 0.43, and the errors were missing knowledge. We compare three options: more prompting, retrieval, and parameter-efficient fine-tuning.

Prompting was already shown flat. Retrieval needs only a TF-IDF index over the train file, built in seconds. For each message, the five most similar train rows become the few-shot examples. We also implemented a separate PEFT experiment: LoRA trains adapter weights on the labeled training data and evaluates on the same development and held-out subsets. Training the 1.5-billion-parameter base model needs an accelerator with enough memory; our current environment has CPU-only PyTorch and insufficient free memory, so we have not generated PEFT accuracy and do not claim a result.

The table. Random examples: 0.43. The retrieved examples alone, a 5-nearest-neighbour vote with no LLM: 0.77. The classical classifier: 0.79. The LLM reading the five retrieved examples: 0.84.

Two more facts. The correct label was among the five retrieved rows in 94 of 100 cases. And against the vote, the LLM was right where the vote was wrong 12 times, and wrong where the vote was right 5 times. So the examples give the floor, and the model adds on top.

RAG remains the runtime assistant's strategy because it improved accuracy without training the large model and fits the laptop workflow. PEFT is available for a measured comparison on a suitable GPU; it is not wired into the runtime assistant. Member 4 shows the final system.

---

## Member 4

### Slide 8, Final system (0:50)

`main.py` puts the conclusions together. One function, `handle`, takes a message and runs five steps.

One: `classify_intent`. Retrieve five examples, build the prompt, greedy decoding, parse, and validate against the 77-name enum. Two: if the intent is one of the 17 actions, such as blocking a card, the assistant does not act yet. It says what it understood and asks for a yes, because the classifier is right about 8 times in 10 and a wrong block is irreversible. Three: `handle_intent`. A simulated backend that returns "done" with a reference number for actions, "information" for 34 question intents, or "unsupported", with how to reach a human, for 26 intents. Four: the reply, sampled, written from the tool transcript only. Five: if there is no valid intent, the assistant asks the customer to rephrase and calls no tool.

One design point. The code fixes the tool order and checks the yes or no, not the model. We tested letting the model drive the tool loop: 4 messages, two setups, 0 of 8 correct sequences. It skipped tools, invented an argument, "BLOCK CARD LOST", and claimed it had blocked a card when it had not. With a small model, control belongs in code.

### Slide 9, Evaluation and failure analysis (1:00)

The main assistant's held-out intent accuracy is 0.77; its development accuracy was 0.84. The assignment's separate generative-family experiment trains a causal LM, VAE, and GAN, then compares their train/test intent accuracy. The three family results and their limitations are in Section 3 and the generated CSV.

The failure analysis has seven types and eleven worked cases in the report. The main ones are on the slide. Near-synonym labels, which retrieval mostly fixed; 10 held-out rows are still wrong for both models, and they are all pairs like card arrival against card delivery estimate. Invented labels, which the schema catches. Retrieval that misleads, when the nearest rows share words but not meaning; that needs a retriever that matches meaning, and is future work. And random decoding, fixed by greedy.

### Slide 10, Responsible use and limits (0:35)

This assistant can trigger actions on a bank account from free text. Three principles guided the controls. Put control in code and leave language to the model: the model never chooses which tool runs. Validate at the boundary: every value that crosses from model text to code passes a schema. Fail towards a human: when unsure, say so and point to an agent.

Limits: small models, single turn, a simulated backend, and word-level retrieval.

The notebook runs the assistant experiments; the family comparison is a separate command: `uv run -m evaluation.generative_family_comparison`.

### Slide 11, Close (0:15)

The contribution table is on the slide. Thank you. We are happy to take questions.

---

## Handover lines

- 1 → 2: "Four levers, four experiments. Member 2 starts with the prompt."
- 2 → 3: "Greedy for the intent, sampling for the reply. Member 3 takes the output schema."
- 3 → 4: "The adaptation decision is retrieval. Member 4 shows the final system."

## If asked

- **Was PEFT implemented?** Yes. `evaluation/peft_lora_comparison.py` fine-tunes Qwen with LoRA and reports development/held-out metrics. The full run has not been measured in this CPU-only environment; run it on a suitable CUDA GPU before quoting scores.
- **Why is there no PEFT accuracy in the report yet?** This environment has CPU-only PyTorch and about 1.1 GB of free memory; loading and fine-tuning the 1.5B base model is not feasible here. The script is ready to run on a suitable GPU, and its result files should be added before claiming a measured three-way adaptation comparison.
- **Why did held-out drop from 0.84 to 0.77?** The development set also chose the approach, so 0.84 is optimistic by construction. 0.77 is the honest number. Both are 100 rows, so one row is one point.
- **Are the VAE/GAN numbers directly comparable to the main assistant's Qwen score?** No. They use separately trained compact models and a full official test split; the main assistant's scores are from fixed 100-row subsets. The family comparison is only among the causal LM, VAE, and GAN.
- **Is the confidence field useful?** No. The model copied the example value. Retrieval similarity would be a better signal for a human fallback.
