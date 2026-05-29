# Citations — MaxGPT

Per the assignment rules, everything below is cited because it is **not** my own
original work. I wrote and understand every line of the model, tokenizer, and
training code that I am submitting.

## Papers / research
- **Vaswani et al. (2017)** — *Attention Is All You Need* — the transformer architecture.
- **Sennrich et al. (2016)** — *Neural Machine Translation of Rare Words with Subword Units* — byte-pair encoding (BPE).
- **Eldan & Li (2023)** — *TinyStories: How Small Can Language Models Be and Still Speak Coherent English?* — motivated the "small model on focused data" approach.
- **Hoffmann et al. (2022)** — *Training Compute-Optimal Large Language Models* (Chinchilla) — the ~20 tokens/parameter rule.
- **Touvron et al. (2023)** — *LLaMA: Open and Efficient Foundation Language Models* — the "train past Chinchilla" recipe used for MaxGPT-3.
- **Dao et al. (2022)** — *FlashAttention* — the fused attention kernel approach.

## Tutorials / references
- **Andrej Karpathy — "Let's build GPT: from scratch, in code, spelled out"** (YouTube) — primary architecture reference.
- **Andrej Karpathy — "Let's build the GPT Tokenizer"** (YouTube) — BPE reference.
- **Andrej Karpathy — nanoGPT** (GitHub) — architectural conventions.

## Datasets (via Hugging Face)
- TinyStories — `roneneldan/TinyStories`
- OASST1 / OASST2 — `OpenAssistant/oasst1`, `OpenAssistant/oasst2`
- UltraChat — `stingning/ultrachat`
- Cosmopedia v2 — `HuggingFaceTB/smollm-corpus` (cosmopedia-v2 split)
- OpenOrca — `Open-Orca/OpenOrca`
- WildChat-1M — `allenai/WildChat-1M`
- HH-RLHF — `Anthropic/hh-rlhf`

## Libraries
- **PyTorch** — model, autograd, mixed precision, `sdpa_kernel`, `torch.compile`
- **Hugging Face `datasets`** — dataset loading + streaming
- **NumPy** — token buffers, memmap batch sampling
- **tqdm** — progress bars
- **Streamlit** — the web UI

## AI assistance (Claude, Anthropic)
I used Claude as a collaborator for: architecture discussions, pseudo-code
generation, code review, debugging help, conceptual explanations, and assistance
preparing the model evaluation and the presentation. A second Claude Code session
helped build the Streamlit web UI in parallel with training.

How I followed the assignment's AI rules:
- **Cited** — this section documents what I used and how.
- **Modified / made my own** — I wrote the actual Python myself from pseudo-code and
  design discussions; the tokenizer optimizations, the data pipeline, the SFT
  loss-masking, and the evaluation harness were implemented and debugged by me.
- **Understand it** — I can explain every line of the model, tokenizer, and training
  loop, and the reasoning behind each architectural decision.
