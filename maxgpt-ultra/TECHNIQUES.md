# MaxGPT-Ultra — Techniques, Tricks & Rationale

The deep-dive companion to `PLAN.md`. Every optimization, trick, and quality lever we
use, with *why* it helps and *how* we apply it. Maintained **incrementally** — each
entry gets filled in (with depth, and a one-line citation) the moment we implement it,
so by the end this is a complete, accurate account of the build. Goal: if someone asks
"what would you do differently next time," the answer is "nothing, given the hardware."

Legend: ☐ planned · ◐ implemented (stub) · ☑ implemented + written up.

## 1. Architecture  ☑ (in `model/model.py`, verified by `scripts/smoke_test.py`)

**RoPE (rotary position embeddings).** Instead of adding learned position vectors, we
rotate each query and key vector by an angle proportional to its position (precomputed
cos/sin tables, applied as `x*cos + rotate_half(x)*sin`, base theta 10000). The payoff:
the attention score between positions i and j depends only on their relative offset
(i minus j), so the model learns relative distance directly, extrapolates to longer
context better, and spends zero parameters on positions. De-facto standard (Llama,
Qwen, Mistral, SmolLM2). Tables are non-persistent buffers, built once, in fp32 for
angle precision.

**RMSNorm.** Normalize each token vector by its root-mean-square (no mean subtraction),
scale by a learned per-feature gain, no bias. The mean-subtraction and bias that
LayerNorm adds contribute little in transformers, so dropping them is cheaper, has
fewer parameters, and is a touch more stable. Computed in fp32 then cast back so bf16
training stays well-behaved.

**SwiGLU feed-forward.** `FFN(x) = down(silu(gate(x)) * up(x))`. The multiplicative
gate lets the network modulate its own activations, which beats a plain GELU MLP at
equal parameters (Shazeer 2020; used by Llama/PaLM). Hidden width is about 8/3 of
d_model so the three projections total roughly the same parameters as a standard 4x
GELU MLP: the gating benefit at no extra cost.

**GQA (grouped-query attention).** Many query heads, few key/value heads, each KV head
shared by a group of query heads (12 to 4 in the prototype, 16 to 4 in the 1B).
Attention's memory bandwidth and KV-cache size scale with the number of KV heads, not
query heads, so sharing them shrinks the cache and speeds up training and especially
inference at negligible quality cost. Big deal on a 12GB card. `repeat_kv()` expands KV
heads to match query heads before SDPA.

**QK-norm.** RMSNorm applied to each head's query and key (over head_dim) before
attention. Without it, Q dot K can grow large during training and spike the softmax,
forcing a lower learning rate or risking divergence over a long run. Normalizing Q and
K bounds the logit scale so we can train faster and more safely. Cheap insurance.

**Tied embeddings.** The output head shares the input embedding matrix. The two learn
closely related token-to-vector maps, so tying saves a full vocab x d_model matrix
(about 100M params, ~9% of the 1B) and tends to help quality at small scale by giving
the shared matrix more gradient signal.

**Pre-norm, no biases.** Each sublayer is `x = x + sublayer(norm(x))`, and no Linear or
norm carries a bias. Pre-norm preserves a clean residual highway from input to output,
which is what lets deep transformers train stably without fragile warmup tricks
(post-norm is far twitchier). Biases add parameters that do not help in large
transformers, so we drop them.

**Deep-net residual init.** All weights start at N(0, 0.02); then the output projection
of each sublayer (`o_proj`, `down_proj`) is rescaled by `1/sqrt(2 * n_layers)`. Because
every layer adds its output to the residual stream, this keeps the stream's variance
roughly constant with depth instead of letting it grow, stabilizing early training.
Matters more the deeper the model.

**z-loss.** A small penalty `weight * mean(logsumexp(logits)^2)` added to the
cross-entropy. It stops the logits' logsumexp from drifting large, keeping the softmax
numerically safe in bf16 and mildly regularizing overconfidence (PaLM and others use
it). Off by default, enabled at 1e-4 via config.

**Attention kernel.** We call `F.scaled_dot_product_attention(..., is_causal=True)`,
which dispatches to FlashAttention on the GPU (memory-linear in sequence length) and a
correct math fallback on CPU, so the same code is fast on the 5070 and runnable for
tests on the Mac.

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
- ☑ **FlashAttention / SDPA** (used by the model now; see the Attention kernel note in §1)

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
