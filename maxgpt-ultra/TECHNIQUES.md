# MaxGPT-Ultra — Techniques, Tricks & Rationale

The deep-dive companion to `PLAN.md`. Every optimization, trick, and quality lever we
use, with *why* it helps and *how* we apply it. Maintained **incrementally** — each
entry gets filled in (with depth, and a one-line citation) the moment we implement it,
so by the end this is a complete, accurate account of the build. Goal: if someone asks
"what would you do differently next time," the answer is "nothing, given the hardware."

Legend: ☐ planned · ◐ implemented (stub) · ☑ implemented + written up.

## 1. Architecture
- ☐ **RoPE** (rotary position embeddings) — why, theta, how applied
- ☐ **RMSNorm** (vs LayerNorm) — why cheaper/stabler
- ☐ **SwiGLU** MLP — why it beats GELU/ReLU MLPs; hidden-dim sizing (8/3·d)
- ☐ **GQA** (grouped-query attention) — KV-head sharing, the memory/speed win
- ☐ **QK-norm** — why it stabilizes high-LR training
- ☐ **Tied embeddings** — param savings, when it helps small models
- ☐ **Pre-norm + no biases** — stability and why biases are dropped
- ☐ **z-loss** — taming the softmax/logit scale

## 2. Tokenizer
- ☐ **~49k byte-level BPE** — vocab-size tradeoff (compression vs embedding params)
- ☐ **Digit-splitting** — why it helps arithmetic/math
- ☐ **Byte fallback** — no OOV, ever

## 3. Data
- ☐ **FineWeb-Edu backbone** — what the edu filter buys us
- ☐ **Cosmopedia v2 / synthetic textbooks** — distilled quality for free
- ☐ **Code + math mixing** — reasoning structure transfer
- ☐ **Dedup + quality filtering** — why duplicates hurt
- ☐ **Curriculum / final-phase annealing** — best data saved for the decay
- ☐ **Memmapped binary shards** — fast, RAM-light streaming
- ☐ **Data position tracking** — correct resume without replaying tokens

## 4. Training & optimization
- ☐ **WSD schedule** (warmup-stable-decay) — why over cosine for our case
- ☐ **AdamW config** (betas, weight decay, grad clip)
- ☐ **Muon optimizer** (optional) — the convergence-speed win
- ☐ **Large effective batch via grad-accum** — stability of a big token batch
- ☐ **bf16 mixed precision** — why bf16 over fp16 on Blackwell
- ☐ **torch.compile** — fusion/graph speedups
- ☐ **FlashAttention / SDPA** — memory-linear attention

## 5. Fitting 1B into 12GB (memory engineering)
- ☐ **Gradient checkpointing** — recompute vs store, the time/memory trade
- ☐ **8-bit AdamW** (bitsandbytes) — optimizer-state compression
- ☐ **Micro-batch + accumulation tuning** — saturating the card
- ☐ **Activation/dtype bookkeeping** — what lives where in VRAM

## 6. Post-training
- ☐ **SFT** — chat template, loss masking on prompts
- ☐ **DPO** — preference optimization, the tractable RLHF
- ☐ **Reasoning / CoT data** — teaching "thinking"
- ☐ **(stretch) GRPO RL** on checkable math

## 7. Inference
- ☐ **RAG** (retriever + web tool) — why it's the small-model superpower
- ☐ **Tool / function calling** — taught via SFT
- ☐ **Quantization** (GGUF/llama.cpp) — fast local serving
- ☐ **Sampling** (temp/top-p/repetition) + **self-consistency / best-of-N**

## 8. Evaluation
- ☐ **Perplexity + benchmark harness** (HellaSwag/ARC/MMLU-slice)
- ☐ **Fixed-prompt generation tracking** — watching the same prompt improve

## 9. Engineering
- ☐ **Atomic checkpointing + resume** — exact-continuation guarantee
- ☐ **Pause/play subprocess supervisor** — freeing VRAM on demand
- ☐ **The mission-control dashboard** — metrics, MFU, live charts
