# MaxGPT-Ultra: Techniques, Tricks, and Rationale

The deep-dive companion to `PLAN.md`. Every optimization, trick, and quality lever we
use. Format for each entry: **what it is** (in plain terms, plus why it is generally
used), then **why we use it here**. Maintained incrementally, so each entry is written
the moment we implement it and by the end this is a complete, accurate account of the
build. Goal: if someone asks "what would you do differently next time," the answer is
"nothing, given the hardware."

Legend: ☐ planned · ◐ implemented (stub) · ☑ implemented + written up.

## 1. Architecture  ☑ (in `model/model.py`, verified by `scripts/smoke_test.py`)

**RoPE (rotary position embeddings).**
*What it is:* The mechanism that tells the model where each token sits. A transformer's
attention is order-blind on its own (it sees an unordered set of tokens), so position
has to be injected somehow. RoPE does it by rotating each query and key vector within
2D subspaces by an angle proportional to the token's position, using fixed sine/cosine
frequencies (base theta 10000). Nothing is learned and nothing is added to the input;
the rotation is applied to Q and K on the fly (`x*cos + rotate_half(x)*sin`).
*Why:* The dot product of a rotated query at position i and a rotated key at position j
ends up depending only on the relative distance (i minus j), so the model learns "how
far apart" tokens are instead of memorizing absolute slots. That generalizes better,
extrapolates to longer sequences, costs zero parameters, and works cleanly with
KV-caching at inference, which is why almost every modern open model (Llama, Qwen,
Mistral, SmolLM2) uses it. We compute the tables in fp32 for angle precision.

**RMSNorm (root-mean-square normalization).**
*What it is:* A normalization layer used inside every block. It divides a token's vector
by its root-mean-square magnitude so the vector has a consistent overall scale, then
multiplies by a learned per-feature gain. Unlike the older LayerNorm, it does not
subtract the mean and has no bias.
*Why:* Normalization keeps activations at a stable scale as they flow through a deep
network, which is what stops training from exploding or stalling; every transformer has
some form of it. RMSNorm is the modern default because LayerNorm's mean-subtraction and
bias contribute little in practice, so removing them is cheaper, uses fewer parameters,
and is slightly more stable. We compute it in fp32 then cast back so bf16 stays safe.

**SwiGLU feed-forward.**
*What it is:* The feed-forward sublayer transforms each token independently through a
small network and is where much of the model's stored knowledge lives. SwiGLU is a
gated version: `down( silu(gate(x)) * up(x) )` — the input goes through two parallel
projections, one is passed through a SiLU activation and used to gate (multiply) the
other, then the result is projected back down.
*Why:* Gated feed-forwards consistently beat a plain GELU/ReLU MLP at equal parameters
because the multiplicative gate lets the network control how much of each feature passes
through (Shazeer 2020; adopted by Llama and PaLM). We size the hidden width at about
8/3 of d_model so the three matrices total roughly the same parameters as a standard 4x
GELU MLP, giving the quality boost at no extra cost.

**GQA (grouped-query attention).**
*What it is:* Standard multi-head attention gives every head its own query, key, and
value. GQA keeps many query heads but has several share one key/value head (here 16
query heads share 4 KV heads). It's the middle ground between full multi-head attention
and a single shared KV head.
*Why:* During generation the model caches the keys and values of every past token, and
attention is bottlenecked by reading that cache; both the cache size and that bandwidth
scale with the number of KV heads, not query heads. Cutting KV heads 16 to 4 shrinks the
cache about 4x and speeds up generation with almost no quality loss, exactly the trade
you want on a 12GB card and for snappy local chat. Standard in Llama 2/3, Qwen, Mistral.
`repeat_kv()` expands KV heads back to query-head count before the attention call.

**QK-norm.**
*What it is:* An extra RMSNorm applied to each attention head's query and key vectors
(over the head dimension) right before they are used to compute attention scores.
*Why:* Attention scores are dot products of queries and keys, so if those vectors grow
large during training the scores blow up, the softmax saturates, and training
destabilizes or must be slowed down. Normalizing Q and K caps that scale, letting us
train at a higher, faster learning rate without divergence. Cheap stabilizer that
matters more the longer the run, so it earns its place in a multi-month pretrain.

**Tied embeddings.**
*What it is:* A language model has two big token tables: the input embedding (token id to
vector) and the output head (final vector to a score per token). Tying makes them
literally the same weight matrix, used in both directions.
*Why:* The two tables learn closely related maps, so sharing them removes a whole
vocab x d_model matrix (about 100M params, ~9% of the 1B) and frees that capacity for
the transformer body; it also tends to improve quality at small scale because the shared
matrix gets gradient signal from both ends. Standard for small models.

**Pre-norm residuals, with no biases.**
*What it is:* Each block keeps a running "residual stream" and adds its sublayers'
outputs back onto it: `x = x + sublayer(norm(x))`. "Pre-norm" means we normalize the
input going into each sublayer rather than its output. Separately, no linear layer or
norm uses an additive bias.
*Why:* The residual stream is a clean, uninterrupted path from input to output; keeping
it un-normalized (normalizing only the branch into each sublayer) lets gradients flow
straight through, which is what allows very deep transformers to train stably without
the fragile learning-rate warmup that post-norm needs. Biases are dropped because at
this scale they add parameters that empirically don't help and slightly slow compute.

**Deep-net residual initialization.**
*What it is:* How the weights are randomly set before training. All start from a normal
distribution with std 0.02; then the output projection of each sublayer (o_proj,
down_proj) is additionally scaled down by 1/sqrt(2 * n_layers).
*Why:* Because every layer adds its output onto the shared residual stream, the stream's
magnitude tends to grow with the number of layers, making the first training steps
unstable in deep models. Scaling each layer's contribution down by the depth factor
keeps the stream's variance roughly constant regardless of depth, so even a deep model
starts cleanly. This is the GPT-2 recipe; it matters more the deeper the network.

**z-loss.**
*What it is:* A small extra term added to the training loss:
`weight * mean(logsumexp(logits)^2)`. Logsumexp is the normalizing denominator of the
softmax, so this measures and penalizes how large the output logits get overall.
*Why:* It keeps logits from drifting to very large magnitudes, which both keeps the
softmax numerically safe in low-precision bf16 and gently discourages overconfidence (a
mild regularizer). Near-free stability insurance used by PaLM and others; we keep it on
at a tiny weight (1e-4) for the long run.

**Attention kernel (SDPA / FlashAttention).**
*What it is:* The core attention computation, softmax of scaled (query dot key) times
value. We call PyTorch's `scaled_dot_product_attention`, which auto-selects the fastest
valid implementation for the hardware.
*Why:* A naive version builds the full T x T score matrix in memory, slow and
memory-hungry for long sequences. FlashAttention computes the identical result in tiles
without ever materializing that matrix, so it's much faster and uses memory that grows
only linearly with sequence length. On the 5070 PyTorch dispatches to it automatically;
on the Mac it falls back to a correct (slower) math version, so the same code runs
everywhere.

**Model shape: d_model, layers, heads, context.**
*What it is:* The handful of numbers that set the model's size and capacity.
- **d_model = 2048**: the width of the residual stream, i.e. the length of the vector
  that represents each token throughout the network. Embeddings, attention, and the
  MLP's input/output are all this wide.
- **n_layers = 22 ("22L")**: how many transformer blocks are stacked, i.e. depth.
- **16/4 heads**: attention runs as 16 parallel query heads, each working on a 128-dim
  slice of the vector (head_dim = 2048/16 = 128); with GQA those 16 share 4 KV heads.
- **context (seq_len) = 2048**: the most tokens the model can attend over at once
  (~1,500 words, a few pages). The prototype uses 1024.
*Why these values:* Total parameters are dominated by roughly `12 * n_layers * d_model^2`,
so width and depth are the two dials that set size; d=2048 with L=22 lands at the
measured 1.09B. The width/depth split has a broad, forgiving sweet spot (final loss is
insensitive over a wide range near it), and 2048/22 sits squarely in the standard zone
for ~1B models (essentially SmolLM2-1.7B's shape trimmed to 1B). head_dim 128 is the
size FlashAttention is most optimized for, so 16 heads at d=2048 is natural, and a 4x
GQA ratio (16 to 4) is the common, well-tested setting. We start context at 2048 to keep
attention cheap during the long pretrain; because positions come from RoPE, we can
extend it later (to 4k-8k) with a short bout of continued training instead of retraining
from scratch.

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
