# MaxGPT-Ultra: Techniques, Tricks, and Rationale

The deep-dive companion to `PLAN.md`. Every optimization, trick, and quality lever we
use. Format for each entry: **what it is** (in plain terms), **why it was developed**
(the problem it was invented to solve), then **why we use it here** (our choice over the
alternatives). Maintained incrementally, so each entry is written the moment we
implement it and by the end this is a complete, accurate account of the build. Goal: if
someone asks "what would you do differently next time," the answer is "nothing, given
the hardware."

Legend: ☐ planned · ◐ implemented (stub) · ☑ implemented + written up.
Sections below §1 are checklists for now; each gets the full three-part writeup as it lands.

## 1. Architecture  ☑ (in `model/model.py`, verified by `scripts/smoke_test.py`)

**RoPE (rotary position embeddings).**
*What it is:* The mechanism that tells the model where each token sits in the sequence.
Attention is otherwise order-blind (it sees an unordered set of tokens), so position has
to be injected. RoPE rotates each query and key vector within 2D subspaces by an angle
proportional to the token's position, using fixed sine/cosine frequencies (base theta
10000). Nothing is learned and nothing is added to the input; the rotation is applied to
Q and K on the fly.
*Why it was developed:* The original Transformer and GPT/BERT used absolute, learned
position embeddings, which cost parameters, don't extend past the trained length, and
only encode absolute slots. Relative-position schemes (Shaw 2018, T5's bias) fixed the
"relative" part but added compute or complexity. RoPE (Su et al., 2021) was created to
bake relative position directly into the attention dot product through rotation, with no
extra parameters and no separate attention bias.
*Why we use it here:* Over absolute embeddings, RoPE extrapolates to longer context and
costs zero parameters (and lets us extend context later with a short fine-tune). Over
ALiBi (another relative scheme), RoPE is generally stronger and is the de-facto standard
the whole open ecosystem is tuned around (Llama, Qwen, Mistral, SmolLM2), so recipes and
tooling assume it. It's also KV-cache friendly for fast inference.

**RMSNorm (root-mean-square normalization).**
*What it is:* A normalization layer inside every block. It divides a token's vector by
its root-mean-square magnitude so the vector has a consistent scale, then multiplies by
a learned per-feature gain. Unlike LayerNorm it doesn't subtract the mean and has no
bias.
*Why it was developed:* LayerNorm (Ba et al., 2016) stabilized transformer training by
re-centering and re-scaling activations. RMSNorm (Zhang & Sennrich, 2019) tested whether
the costly mean-subtraction was actually necessary, found it largely wasn't, and dropped
it, keeping only the RMS re-scaling for a cheaper, simpler norm with equal quality.
*Why we use it here:* Over LayerNorm, RMSNorm is faster and has fewer parameters at no
measurable quality cost, which compounds over a long run. Over no normalization, it's
the difference between stable and divergent training. It's the modern default (Llama,
Qwen). We compute it in fp32 then cast back so bf16 stays safe.

**SwiGLU feed-forward.**
*What it is:* The feed-forward sublayer transforms each token independently through a
small network and holds much of the model's stored knowledge. SwiGLU is a gated version:
`down( silu(gate(x)) * up(x) )` — two parallel projections, one passed through a SiLU
activation and used to gate (multiply) the other, then projected down.
*Why it was developed:* The classic FFN is a plain two-layer MLP with ReLU/GELU. Gated
Linear Units (Dauphin et al., 2017) showed a multiplicative gate helps; Shazeer (2020,
"GLU Variants Improve Transformer") put these gates inside transformer FFNs and found
the SiLU-gated variant (SwiGLU) consistently lowered loss.
*Why we use it here:* Over a GELU/ReLU MLP, SwiGLU gives better quality at the same
parameter count. Over other GLU variants (GEGLU and friends), differences are marginal
and SwiGLU is what Llama/PaLM standardized on, so it's the best-supported choice. We
size the hidden width at ~8/3 of d_model so the three matrices total the same parameters
as a standard 4x MLP: the gain is free.

**GQA (grouped-query attention).**
*What it is:* Standard multi-head attention gives every head its own query, key, and
value. GQA keeps many query heads but lets several share one key/value head (here 16
query heads share 4 KV heads). It sits between full multi-head attention and a single
shared KV head.
*Why it was developed:* Multi-head attention (Vaswani et al., 2017) keeps a per-head
key/value cache that dominates memory and bandwidth at inference. Multi-query attention
(Shazeer, 2019) collapsed that to one KV head for speed but lost quality and could
destabilize training. GQA (Ainslie et al., 2023) was developed as the middle ground: a
few KV heads recover almost all the quality while keeping most of the speed.
*Why we use it here:* Over full multi-head attention, GQA shrinks the KV cache ~4x and
speeds up generation, which matters enormously on a 12GB card and for snappy local chat.
Over multi-query attention, GQA is more stable and higher quality. It's the setting
Llama 2/3, Qwen, and Mistral all converged on.

**QK-norm.**
*What it is:* An extra RMSNorm applied to each attention head's query and key vectors
(over the head dimension) right before they compute attention scores.
*Why it was developed:* As models and learning rates scaled, attention logits
(query dot key) could grow very large mid-training, saturating the softmax and causing
loss spikes or divergence. Normalizing Q and K (Henry et al., 2020; used prominently in
ViT-22B and several recent LLMs) was developed to bound that scale and keep attention
well-behaved.
*Why we use it here:* Over doing nothing, it lets us train at a higher, faster learning
rate without risking a blow-up, valuable insurance across a months-long run. Over the
alternative of attention-logit soft-capping (Gemma's approach), QK-norm is simpler and
more widely used. It's cheap, so it earns its place.

**Tied embeddings.**
*What it is:* A language model has two big token tables: the input embedding (token id to
vector) and the output head (final vector to a score per token). Tying makes them
literally the same matrix, used in both directions.
*Why it was developed:* Press & Wolf (2017) and Inan et al. (2017) observed that the
input and output embeddings learn closely related representations, and that forcing them
to share weights both saves a large matrix and improves perplexity.
*Why we use it here:* Over untied embeddings, tying frees ~100M parameters (~9% of the
1B) for the transformer body and tends to improve quality at small scale (the shared
matrix gets gradient from both ends). Very large models sometimes untie for extra
capacity, but at our scale tying is the clear win.

**Pre-norm residuals, with no biases.**
*What it is:* Each block keeps a running "residual stream" and adds its sublayers'
outputs onto it: `x = x + sublayer(norm(x))`. "Pre-norm" means we normalize the input
into each sublayer rather than its output. Separately, no linear layer or norm carries
an additive bias.
*Why it was developed:* The original Transformer placed the norm after each sublayer
("post-norm"), which needs a delicate learning-rate warmup and gets unstable as depth
grows. Pre-norm (Baevski & Auli, 2018; analyzed by Xiong et al., 2020) moves the norm
inside the residual branch so a clean, un-normalized path runs end to end and gradients
flow straight through, making deep nets trainable. Dropping biases was popularized by
GPT/PaLM/Llama after biases were found to add parameters without helping.
*Why we use it here:* Over post-norm, pre-norm trains a deep model stably without fragile
warmup tricks. Over keeping biases, dropping them costs nothing in quality and slightly
simplifies and speeds compute. Both are standard in modern LLMs.

**Deep-net residual initialization.**
*What it is:* How weights are randomly set before training. All start from a normal
distribution with std 0.02; then the output projection of each sublayer (`o_proj`,
`down_proj`) is scaled down by `1/sqrt(2 * n_layers)`.
*Why it was developed:* Because every layer adds onto the shared residual stream, the
stream's variance grows with depth, making the first steps of deep-model training
unstable. GPT-2 (Radford et al., 2019) introduced scaling the residual-projection init
by the number of layers to counteract this (related ideas: Fixup, T-Fixup).
*Why we use it here:* Over a flat 0.02 init everywhere, the depth-aware scaling keeps the
residual variance roughly constant so a 22-layer model starts cleanly. Over more
elaborate schemes (Fixup et al.), the GPT-2 `1/sqrt(2N)` rule is simpler and the proven
default. Matters more the deeper we go.

**z-loss.**
*What it is:* A small extra training-loss term, `weight * mean(logsumexp(logits)^2)`.
Logsumexp is the softmax's normalizing denominator, so this measures and penalizes how
large the output logits get overall.
*Why it was developed:* In large-scale and low-precision training, the softmax
normalizer can drift to large values and cause numerical instability or loss spikes.
z-loss (used in Google's mesh-tensorflow / T5 stack and in PaLM) was added to gently pin
that normalizer near zero and stabilize training.
*Why we use it here:* Over nothing, it's near-free protection against bf16 loss spikes
across a long run, plus a mild anti-overconfidence regularizer. There isn't really a
competing technique for this specific problem, so we keep this standard safeguard on at
1e-4.

**Attention kernel (SDPA / FlashAttention).**
*What it is:* The core attention computation, softmax of scaled (query dot key) times
value. We call PyTorch's `scaled_dot_product_attention`, which auto-selects the fastest
valid backend for the hardware.
*Why it was developed:* A naive implementation materializes the full T x T attention
matrix in memory, making it quadratic in memory and bottlenecked on memory traffic.
FlashAttention (Dao et al., 2022) was developed as an IO-aware, tiled algorithm that
computes the exact same result without ever writing that matrix to memory: dramatically
faster and linear in memory.
*Why we use it here:* Over naive attention, it's much faster and uses far less memory,
letting us fit longer sequences on 12GB. Over approximate-attention methods (Performer,
Linformer), FlashAttention is exact (no quality loss). And we get it for free through
PyTorch's SDPA, which also provides a correct CPU fallback for testing on the Mac.

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
*Why these values:* (This is a sizing decision rather than an invented component, so it's
what-it-is plus why-these-numbers.) Total parameters are dominated by roughly
`12 * n_layers * d_model^2`, so width and depth are the two dials that set size; d=2048
with L=22 lands at the measured 1.09B. The width/depth split has a broad, forgiving
sweet spot (final loss is insensitive over a wide range near it), and 2048/22 sits
squarely in the standard zone for ~1B models (essentially SmolLM2-1.7B's shape trimmed
to 1B). head_dim 128 is the size FlashAttention is most optimized for, so 16 heads at
d=2048 is natural, and a 4x GQA ratio (16 to 4) is the common, well-tested setting. We
start context at 2048 to keep attention cheap during the long pretrain; because positions
come from RoPE, we can extend it later (to 4k-8k) with a short bout of continued training
instead of retraining from scratch.

## 2. Tokenizer
- ☐ **~49k byte-level BPE**: vocab-size tradeoff (compression vs embedding params)
- ☐ **Digit-splitting**: why it helps arithmetic/math
- ☐ **Byte fallback**: no out-of-vocab, ever

## 3. Data
- ☐ **FineWeb-Edu backbone**: what the edu filter buys us
- ☐ **Cosmopedia v2 / synthetic textbooks**: distilled quality for free
- ☐ **Code + math mixing**: reasoning-structure transfer
- ☐ **Dedup + quality filtering**: why duplicates hurt
- ☐ **Curriculum / final-phase annealing**: best data saved for the decay
- ☐ **Memmapped binary shards**: fast, RAM-light streaming
- ☐ **Data-position tracking**: correct resume without replaying tokens

## 4. Training & optimization
- ☐ **WSD schedule** (warmup-stable-decay): why over cosine for our case
- ☐ **AdamW config**: betas, weight decay, grad clip
- ☐ **Muon optimizer** (optional): the convergence-speed win
- ☐ **Large effective batch via grad-accum**: stability of a big token batch
- ☐ **bf16 mixed precision**: why bf16 over fp16 on Blackwell
- ☐ **torch.compile**: fusion / graph speedups
- ☑ **FlashAttention / SDPA**: used by the model now (see the Attention kernel note in §1)

## 5. Fitting 1B into 12GB (memory engineering)
- ☐ **Gradient checkpointing**: recompute vs store, the time/memory trade
- ☐ **8-bit AdamW** (bitsandbytes): optimizer-state compression
- ☐ **Micro-batch + accumulation tuning**: saturating the card
- ☐ **Activation / dtype bookkeeping**: what lives where in VRAM

## 6. Post-training
- ☐ **SFT**: chat template, loss masking on prompts
- ☐ **DPO**: preference optimization, the tractable RLHF
- ☐ **Reasoning / CoT data**: teaching "thinking"
- ☐ **(stretch) GRPO RL**: on checkable math

## 7. Inference
- ☐ **RAG** (retriever + web tool): why it's the small-model superpower
- ☐ **Tool / function calling**: taught via SFT
- ☐ **Quantization** (GGUF / llama.cpp): fast local serving
- ☐ **Sampling** (temp / top-p / repetition) + **self-consistency / best-of-N**

## 8. Evaluation
- ☐ **Perplexity + benchmark harness**: HellaSwag / ARC / MMLU-slice
- ☐ **Fixed-prompt generation tracking**: watching the same prompt improve

## 9. Engineering
- ☐ **Atomic checkpointing + resume**: exact-continuation guarantee
- ☐ **Pause/play subprocess supervisor**: freeing VRAM on demand
- ☐ **The mission-control dashboard**: metrics, MFU, live charts
