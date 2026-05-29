# MaxGPT

**Four conversational language models built from scratch in PyTorch, scaling from 23M to 235M parameters.** No pretrained weights, no `transformers` library. I wrote the tokenizer, the transformer, the training loop, the fine-tuning pipeline, and the evaluation harness myself.

This is my CS final project. The goal was not to beat ChatGPT (a 235M model never could), it was to understand how large language models actually work by building every piece, and to demonstrate what changes as you scale a small model 10x and then fine-tune it into a chatbot.

---

## The models at a glance

| Model | Params | Architecture | Context | Training | What it demonstrates |
|---|---|---|---|---|---|
| **MaxGPT-1** | 23M | 6 layers x 384 dim, 6 heads | 256 | ~30 min | Coherent English emerges |
| **MaxGPT-2** | 110M | 12 layers x 768 dim, 12 heads | 512 | ~10 hrs | Learns the USER/ASSISTANT chat format |
| **MaxGPT-3** | 235M | 16 layers x 1024 dim, 16 heads | 1024 | ~6-7 days | A real chatbot (7-dataset, ~5B-token mix) |
| **MaxGPT-3.5** | 235M | same as MaxGPT-3, then SFT | 1024 | +~16 hrs | Chat-tuned on assistant responses only |

Same architecture, scaled about 10x from the first model to the last. Total training was roughly **28 exaFLOPs**, all on a single RTX 5070. For perspective, GPT-3 used about **10,000x more** compute.

---

## Why build from scratch?

Most ML projects call a pretrained API. That hides everything interesting. Building from scratch forced me to understand every architectural decision and every line of code, and it makes a point that is easy to forget: a language model is not magic, it is a stack of matrix multiplications trained to predict the next token.

Language generation is also a problem that *only* machine learning can solve. You cannot hand-write if-then rules for conversation. The only thing that works is a network that learns statistical patterns from a large amount of text. That is what makes this an AI project rather than a normal programming project.

---

## How it works

### The tokenizer (custom byte-level BPE)

Text has to become numbers before a model can read it. I implemented byte-pair encoding (BPE) from scratch with a 16,000-token vocabulary, trained on a sample of the data.

My first version was correct but slow: it rescanned the entire token list on every merge, which meant 20+ hours just for data prep. I rebuilt it with two ideas:

- A **doubly-linked list** of token positions, so merging two tokens is O(1).
- A **min-heap** of `(merge-rank, position)`, so finding the next merge is O(log n).

The per-merge cost dropped from O(all tokens) to O(occurrences of that pair).

| | Speedup |
|---|---|
| Tokenizer training | **31x** |
| Encoding | **465x** (1115s to 2.4s per chunk) |

Without this, preparing MaxGPT-3's data would have taken weeks instead of hours.

### The transformer

A decoder-only (GPT-style) transformer, written by hand. The core is multi-head causal self-attention:

```python
qkv = self.qkv_proj(x)                 # (B, T, 3C)
q, k, v = qkv.split(C, dim=-1)
q = q.view(B, T, nh, hd).transpose(1, 2)   # (B, heads, T, head_dim)
scores = (q @ k.transpose(-2, -1)) / sqrt(hd)
scores = scores.masked_fill(causal_mask == 0, -inf)   # no peeking ahead
attn = softmax(scores, dim=-1)
out = attn @ v                          # weighted sum of values
```

Every component is built up from primitives: multi-head attention, a feed-forward network with GELU and 4x expansion, pre-norm LayerNorm, residual connections, and learned positional embeddings. A config flag switches between this exact manual math and an optimized fused-attention path, which was useful for both learning and speed.

### Data

MaxGPT-1 and MaxGPT-2 trained on a small, focused mix (TinyStories + OpenAssistant) so the only thing changing between them was model size, giving a clean scaling comparison. MaxGPT-3 used a seven-dataset mix totaling roughly 5B unique tokens:

TinyStories, OASST1 + OASST2, UltraChat, Cosmopedia v2, OpenOrca, WildChat-1M, and Anthropic HH-RLHF. The mix balances narrative fluency, real human conversations, synthetic instruction data, and educational content. All datasets are credited in [CITATIONS.md](CITATIONS.md).

### Training

- bf16 mixed precision via `torch.autocast`
- AdamW with beta2 = 0.95 (an LLM-specific choice, not the PyTorch default)
- Cosine learning-rate schedule with linear warmup
- Gradient clipping at norm 1.0
- `torch.compile` for fused kernels, fused AdamW, pinned-memory transfers
- Multiprocessing data encoding with streaming to disk

MaxGPT-3 ran for 2.44M steps (about 20B token-passes, roughly 4x the Chinchilla-optimal amount, following the LLaMA "train past optimal" recipe) and finished at a loss around 2.0.

---

## What scaling looks like

The clearest way to see the whole project is one prompt through all four models: *"can you give me a recipe for brownies?"*

- **MaxGPT-1 (23M):** "my Kitchen Solyn, Iraq laid the Rocky Processory Picky..." Word salad.
- **MaxGPT-2 (110M):** grammatical and recipe-shaped, but suggests brownies with brown rice and tofu, and bleeds the training format.
- **MaxGPT-3 (235M):** a genuinely real-looking recipe with ingredients and numbered steps, but it wanders off into French etouffee instead of brownies.
- **MaxGPT-3.5 (SFT):** clean assistant formatting and it stays on brownies (though the ingredients are still absurd).

None produce a correct recipe, because 235M parameters is still tiny. But you can watch coherence emerge with scale, and watch fine-tuning change behavior, in a single example.

---

## Supervised fine-tuning (MaxGPT-3.5)

MaxGPT-3.5 is MaxGPT-3 fine-tuned on chat-only data for about one epoch (300K steps, learning rate 5e-5). The key detail is **loss masking**: the loss is computed only on the assistant's tokens, so the model learns to *reply* rather than to imitate the user. It finished at roughly 1.87 train / 1.95 validation loss, with the two tracking each other closely (a sign of healthy generalization, not memorization).

---

## Evaluation (the honest part)

Single examples are misleading at temperature 0.8, so I evaluated properly: **blind, multi-sample scoring** (one greedy plus five sampled generations per prompt) across five checkpoints, scored by independent raters against a fixed rubric. I also wrote a separate multi-turn test of twelve scripted conversations where the final question can only be answered using an earlier turn.

The result is a genuine trade-off, not a clean win:

**Single-turn quality:** the base model actually scored slightly higher on raw response quality (about 2.5 vs 2.3 on a 1-5 scale). Fine-tuning did not make every answer better.

**Where fine-tuning clearly won:**

| Skill | Base MaxGPT-3 | MaxGPT-3.5 (SFT) |
|---|---|---|
| Cross-turn reference resolution ("feed *him*" to the puppy) | 25% | **92%** |
| Multi-turn context use (overall) | 25% | **44%** |
| Factual structure (photosynthesis, correct of 6) | 0 | **4** |
| Assistant self-awareness ("as an AI, I don't have...") | never | yes |

**Where both failed:** exact factual recall (remembering a name stated earlier), corrections, and creative adaptation. These need either more capacity or more targeted data, and they sit at the limit of a 235M model.

The honest summary: **SFT specialized the model toward conversational behavior, with big wins in reference resolution and assistant framing, at the cost of some verbosity and creative range.** The most important methodological lesson was that my own first-pass eyeball estimate overstated the effect by roughly 2x, and rigorous blind scoring corrected it. Reporting that honestly mattered more than claiming a bigger win.

---

## Challenges I had to solve

These were the real engineering, and the most useful things I learned.

- **Flash Attention was missing on my GPU.** The Windows PyTorch wheel for my Blackwell card shipped without Flash-Attention kernels, and auto-selection silently fell back to a slow path. I wrote a diagnostic that tested every attention backend, forced the memory-efficient one, and eventually moved training to WSL2 (Linux) to get real Flash Attention 2.
- **A silent data-split bug.** When I concatenated datasets and then sliced 90/10 for train/validation, one whole dataset ended up entirely in validation, so the model trained on only one source and produced gibberish on chat prompts. The fix was to split each dataset separately and then combine.
- **100% of tokens masked during fine-tuning.** Byte-pair merges had absorbed my role-marker tokens (the newline and colon around `ASSISTANT:` got merged into neighboring tokens), so my token-level mask matched nothing. The fix was to search for markers at the byte level after decoding, which is robust to how BPE happens to tokenize.
- **The 12 GB VRAM ceiling.** Loading a checkpoint plus its optimizer state overflowed the GPU. The fix was to load to CPU first and move only the weights. I also added a repetition penalty to break the low-entropy loops small models fall into.

---

## The web app

A Streamlit app ([webui/](webui/)) ties it together:

- **Chat mode** with full multi-turn memory.
- **Compare mode** that races multiple models on the same prompt side by side.
- **Token-by-token streaming**, the live "typing" effect.
- A custom import-collision loader so all four models (each with its own `config`, `model`, and `tokenizer` modules sharing the same class names) can run in one app.

---

## What I would do in version 2

- A larger vocabulary (32K-50K) so each token carries more meaning.
- Modern architecture pieces: RMSNorm, SwiGLU, and RoPE (I deliberately skipped these to keep a clean scale-only comparison between models).
- Cleaner fine-tuning data plus replaying some pretraining text, so the model keeps its general strengths instead of trading them away.
- Resume-from-checkpoint training, so a crash does not cost days.
- A proper evaluation harness with perplexity and benchmark scores, not just sample prompts.
- Quantization (int8/int4) so the model could run on a phone.

---

## Repo structure

```
maxgpt-1/ maxgpt-2/ maxgpt-3/   each model: config.py, tokenizer.py, model.py,
                                prepare_data.py, train.py, sample.py
maxgpt-3/finetune_data.py       SFT data loader with byte-level loss masking
maxgpt-3/finetune.py            supervised fine-tuning loop
maxgpt-3/eval_compare.py        single-turn multi-sample evaluation
maxgpt-3/eval_multiturn.py      multi-turn context-tracking evaluation
webui/app.py                    Streamlit demo (chat + compare)
CITATIONS.md                    datasets, papers, libraries, AI-use disclosure
```

To run the demo: `cd webui && streamlit run app.py` (needs the trained checkpoints, which are gitignored due to size).

---

## Citations and AI use

Full citations are in [CITATIONS.md](CITATIONS.md). In short: the architecture comes from the Transformer and BPE papers, the "small model on focused data" idea from TinyStories, and the scaling recipe from Chinchilla and LLaMA. My main tutorial reference was Andrej Karpathy's "Let's build GPT" videos.

I used Claude (Anthropic) as a collaborator for architecture discussions, pseudo-code, code review, debugging, and help with the evaluation and write-up. Following the assignment's rules, I wrote and understand every line of the model, tokenizer, and training code, and I can explain the reasoning behind each decision.

---

*Built by Max, 2026.*
