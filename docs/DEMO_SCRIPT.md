# Demonstration Script (8 to 10 minutes)

One speaker per block. Every member should be able to answer questions on any block.
Use `notebooks/banking_agent.ipynb`; saved outputs mean you can show results without re-running anything.

| Time | Speaker | What to show | Key message |
|---|---|---|---|
| 0:00 to 1:00 | 1 | Problem: customer message in, intent + action + reply out. Notebook section 0 | Why an LLM: open text in, language out, control at inference time |
| 1:00 to 2:00 | 1 | Dataset: BANKING77, 77 intents, 10,003 train / 3,080 test; the two evaluation subsets. Section 1 | Near-synonym labels are the core difficulty |
| 2:00 to 3:30 | 2 | **Generative families.** Section 1b table: autoregressive vs VAE vs GAN, plus the sample outputs | Only the autoregressive family accepts a text prefix and produces ordered text, so a causal LM fits |
| 3:30 to 4:30 | 2 | Architecture: token IDs, embedding shape, top-5 next-token probabilities. Section 2 | Tokenizer, embeddings, attention, LM head, one token at a time |
| 4:30 to 6:00 | 3 | Baseline 0.41, prompt table, decoding table. Sections 3 to 5 | Prompt wording and decoding barely move accuracy; the constraint line fixes format |
| 6:00 to 7:00 | 3 | Retrieval few-shot 0.84 dev, 0.77 held-out, against TF-IDF + LogReg 0.79. Section 7 | Adaptation decision: prompting vs RAG vs PEFT. Missing label knowledge needs examples |
| 7:00 to 8:30 | 4 | **Live demo.** Run the interactive cell. Try: `I lost my card, please block it`, `how long does a transfer take`, a nonsense message. Show the `[tool]` lines | Schema validation, two tools, code-driven order, reference number |
| 8:30 to 9:30 | 4 | Failure analysis and responsible use: invented labels, hallucinated action, prompt injection. Section 8 and report 12 | What can go wrong and the controls |
| 9:30 to 10:00 | all | Conclusion and what we would do next (embeddings retriever, LoRA) | |

## Questions to rehearse

- Why not train a VAE or GAN for this? (Section 1b: no prefix control, no word order, no pretrained instruction-tuned model.)
- Why is greedy used for the label and sampling for the reply? (Decoding table: sampling only loses accuracy; reply wording benefits from variety.)
- Why is the held-out score (0.77) lower than the development score (0.84)? (Selection on the development set plus noise; the interval for 100 rows is about plus or minus 0.08.)
- Why does code, not the model, decide the tool order? (0 of 8 correct sequences when the 1.5B model decided.)
- Why not fine-tune? (Not approved, needs a GPU, retrieval already beat the classical baseline. LoRA is the next step.)
