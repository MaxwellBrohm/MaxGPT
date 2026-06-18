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

## 2. Tokenizer  ☑ (module in `tokenizer/`, verified by `scripts/test_tokenizer.py`; the real 49k vocab gets trained on a corpus sample in the data milestone)

**Byte-level BPE (~49k vocab).**
*What it is:* BPE (byte-pair encoding) builds a vocabulary by repeatedly merging the
most frequent adjacent symbol pairs, starting from raw bytes. "Byte-level" means the
base alphabet is the 256 possible bytes, so any text is representable. We train to
49,152 tokens.
*Why it was developed:* Word-level vocabularies can't handle unseen words and balloon in
size; character-level is too granular (very long sequences). BPE (Sennrich et al., 2016;
popularized for LMs by GPT-2's byte-level variant, 2019) was the middle ground: a
fixed-size subword vocabulary that covers everything through merges.
*Why we use it here:* It's the universal LLM standard. We pick ~49k over 32k/64k because
a larger vocab compresses text better (shorter sequences, faster effective training,
better on code/math) while the bigger embedding table stays affordable at 1B with tied
embeddings; 49,152 is hardware-aligned and the proven SmolLM2 size. Trained on our own
corpus mix so the merges fit our data.

**Digit-splitting.**
*What it is:* A pre-tokenization rule that splits every run of digits into individual
digit tokens, so "1234" becomes "1","2","3","4" before BPE ever sees it.
*Why it was developed:* Default BPE merges common number strings ("2024", "100") into
single tokens, so the model sees "17" and "18" as unrelated atoms with no place-value
structure to learn arithmetic from. Splitting digits (Llama 3 and others) gives every
number a uniform, compositional representation.
*Why we use it here:* It measurably improves arithmetic and math, which we care about
(math is in the data, reasoning is a goal), at essentially no cost. Merged number tokens
are strictly worse for math.

**Byte fallback.**
*What it is:* Guaranteeing all 256 raw bytes are in the base vocabulary (we seed the
trainer with the full byte alphabet), so any input decomposes into known tokens and
there is never an out-of-vocabulary token.
*Why it was developed:* Older subword tokenizers used an explicit "UNK" token for
anything outside the vocab, silently destroying information (rare characters, other
scripts, symbols). Byte-level / byte-fallback schemes (GPT-2; SentencePiece's
byte_fallback) eliminated UNK entirely.
*Why we use it here:* It makes the tokenizer lossless and universal: emoji, any language,
code, and odd symbols all round-trip exactly (verified in the test), which matters for
arbitrary user input and attachments. No real downside, so it's standard.

**Special tokens reserved up front.**
*What it is:* A fixed set of non-text control tokens baked into the vocab from the start:
`<|endoftext|>` (BOS/EOS/separator), `<|pad|>`, ChatML markers `<|im_start|>`/`<|im_end|>`,
tool markers `<|tool_call|>`/`<|tool_response|>`, reasoning tags `<think>`/`</think>`,
plus 16 spare reserved slots.
*Why it was developed:* Models need in-band control signals for turn boundaries, document
ends, tool calls, and reasoning; chat formats like ChatML (OpenAI) standardized this.
Reserving them (and spares) in the original vocab is common practice because adding
tokens later forces resizing the embedding matrix and re-training.
*Why we use it here:* Baking in chat, tool, and reasoning tokens now means SFT, tool-use,
and "thinking" all work later without ever touching the vocab; the 16 spares cover what
we didn't foresee. Cheap insurance against a painful retrofit. (Verified atomic and
low-id in the smoke test.)

## 3. Data  ◐ (pipeline code in `data/`, verified by `scripts/test_data.py`; the data-content steps run at download time on the training box)

**Memmapped binary shards.**
*What it is:* We pre-tokenize all text once into flat files of uint16 token ids
(`shard_*.bin` + a `meta.json` index) and read them at train time via numpy memory-
mapping (the OS pages bytes in on demand rather than loading everything into RAM).
*Why it was developed:* Tokenizing on the fly wastes CPU every epoch and stalls the GPU,
and loading a multi-hundred-GB corpus into RAM is impossible. The pre-tokenized-shard +
memmap pattern (nanoGPT, llm.c, Megatron) feeds the GPU pre-digested tokens straight from
disk with near-zero overhead.
*Why we use it here:* It keeps the 5070 fed without a data bottleneck and uses almost no
RAM regardless of corpus size. uint16 (vs uint32) halves on-disk size since our vocab
fits in 16 bits; EOT ids separate documents in the stream.

**Sequence packing.**
*What it is:* Instead of one (padded) document per example, we read the token stream as
one continuous sequence and slice it into back-to-back `seq_len` windows, so documents
flow across window boundaries (separated by the EOT marker).
*Why it was developed:* Documents vary wildly in length; one-doc-per-example wastes a
large fraction of every batch on padding tokens that contribute nothing. Packing
(standard in GPT/Llama/T5 pretraining) fills every position with real tokens.
*Why we use it here:* Over a months-long run, having ~100% of each batch be real tokens
(instead of, say, 60% with padding) is a large, free efficiency gain. Our loader reads
contiguous windows so it tiles the corpus exactly.

**Data-position tracking (resumable loader).**
*What it is:* The loader's entire read state is a single integer (the global token
position) plus an epoch counter, saved/restored via `state_dict()`/`load_state_dict()`.
*Why it was developed:* Long runs crash and resume constantly; if the loader restarts
from the beginning each time, the model over-trains on early data and wastes compute.
Tracking the exact position is the standard fix.
*Why we use it here:* It's what makes your pause/play workflow correct: stop and resume
as often as you like and training continues over the exact same data with no replay.
Shards are written from an already-shuffled stream, so a simple sequential read is both
well-shuffled and trivially resumable.

**Mixture weights.**  ◐ (spec in `data/prepare.py: PRETRAIN_MIX`; weights to tune)
*What it is:* The target token-share of each source (web / textbooks / code / math /
wiki), realized by weighted streaming interleave.
*Why it matters:* The ratio is a real capability lever (too much code dulls general
fluency; too little dulls reasoning). We start from the proven SmolLM mix and tune.

Remaining (applied when we download + build the corpus on the 5070 box):
- ☐ **FineWeb-Edu backbone**: what the edu filter buys us
- ☐ **Cosmopedia v2 / synthetic textbooks**: distilled quality for free
- ☐ **Code + math mixing**: reasoning-structure transfer
- ☐ **Dedup + quality filtering**: FineWeb-Edu / SmolLM-corpus are already deduped; add light exact-dedup
- ☐ **Curriculum / final-phase annealing**: best data saved for the decay

## 4. Training & optimization  ◐ (loop in `train/`, verified by `scripts/test_train.py`)

**WSD schedule (warmup-stable-decay).**
*What it is:* The learning-rate curve over training: a short linear warmup to the peak
LR, a long stretch holding it constant, then a cosine decay to near-zero over the final
fraction of steps (`train/schedule.py`).
*Why it was developed:* The long-standing default is a single cosine curve, but it
requires committing to the total step count up front (the curve has to reach zero
exactly at the end). WSD (popularized by MiniCPM, 2024) decouples the middle from the
end, so the stable phase can run as long as you want and you decay only when you decide
to stop.
*Why we use it here:* Our run is open-ended and pause-heavy; WSD lets us extend or stop
without wasting the end-of-run loss drop, and keeps a pre-decay checkpoint to continue
from later. Same final quality as cosine, far more flexible.

**AdamW (decoupled weight decay).**
*What it is:* The optimizer. Adam adapts a per-parameter step size from running estimates
of the gradient's mean and variance; AdamW applies weight decay as a separate shrink
rather than folding it into the gradient. We use betas (0.9, 0.95), weight decay 0.1 on
2D+ tensors only (not norms), and gradient clipping at 1.0.
*Why it was developed:* Adam (Kingma & Ba, 2014) made transformer training robust to
gradient scaling where plain SGD struggled. AdamW (Loshchilov & Hutter, 2017) fixed a
subtle bug where L2 regularization interacts badly with Adam's adaptivity, by decoupling
weight decay.
*Why we use it here:* It's the proven LLM default; the (0.9, 0.95) betas + 0.1 decay are
the standard GPT/Llama recipe. Clipping at 1.0 stops a rare huge gradient from wrecking
the run; decaying only matmul/embedding weights (not norms) is standard best practice.

**Large effective batch via gradient accumulation.**
*What it is:* We run several micro-batches, sum their gradients, and only then take one
optimizer step, so the effective batch is `micro_batch * grad_accum * seq_len` tokens
(~0.5M for the 1B) even though only a micro-batch fits in 12GB at once.
*Why it was developed:* Large-batch training is more stable and supports a higher LR, but
big batches don't fit in memory; accumulation (used everywhere) decouples the statistical
batch size from what physically fits on the GPU.
*Why we use it here:* It's the only way to get a pretraining-scale token batch out of a
12GB card. We tune micro_batch to fill VRAM and grad_accum to reach the target batch.

**bf16 mixed precision.**
*What it is:* Run the forward/backward in bfloat16 (via autocast) while the optimizer
math stays fp32. bf16 keeps fp32's exponent range, just with fewer mantissa bits.
*Why it was developed:* fp32 is memory- and bandwidth-bound; fp16 is fast but its narrow
exponent range overflows easily and needs fragile loss-scaling. bf16 (Google Brain
float) keeps fp32's range, so it's fast *and* stable with no loss-scaling.
*Why we use it here:* The 5070 (Blackwell) has fast bf16 tensor cores, so it roughly
halves memory and speeds up matmuls with no stability hacks. Enabled on CUDA; falls back
to fp32 on CPU so the Mac tests still run.

- ☐ **Muon optimizer** (optional): a newer optimizer that orthogonalizes the update for faster convergence; a candidate to try
- ☐ **torch.compile**: fuse the graph for a speedup once the loop is stable on the GPU
- ☐ **Checkpoint averaging / model soup** (optional): average a few late checkpoints for a small, near-free final bump
- ☑ **FlashAttention / SDPA**: used by the model (see the Attention kernel note in §1)

## 5. Fitting 1B into 12GB (memory engineering)  ◐ (toggles wired in the trainer/model)

**Gradient checkpointing.**
*What it is:* During the forward pass, don't keep every layer's activations for the
backward pass; keep only a few and recompute the rest on the fly during backprop. Toggled
by `grad_checkpointing` (wraps each block in `torch.utils.checkpoint`).
*Why it was developed:* Activations, not weights, dominate memory for deep models, and
they grow with batch and sequence length. Checkpointing (Chen et al., 2016) trades a bit
of extra compute (~one extra forward) for a large drop in activation memory.
*Why we use it here:* It's a key lever for fitting the 1B's activations into 12GB; we pay
~30% more compute to make a model fit and a bigger batch possible. Off for the prototype
(plenty of room), on for the 1B.

**8-bit AdamW.**
*What it is:* Store Adam's two optimizer states (mean + variance) per parameter in 8-bit
with block-wise quantization instead of fp32, via bitsandbytes. Toggled by
`optimizer_8bit` (falls back to normal AdamW if unavailable / on CPU).
*Why it was developed:* Adam's states are 2x the model size in fp32 (8 bytes/param),
often the single biggest memory consumer; 8-bit optimizers (Dettmers et al., 2021) shrink
that ~4x with negligible quality loss.
*Why we use it here:* For a 1B model, fp32 Adam states alone are ~8GB and wouldn't fit
beside weights, grads, and activations on a 12GB card; 8-bit states (~2GB) make the 1B
trainable here. The prototype uses plain AdamW.

- ☐ **Micro-batch + accumulation tuning**: measure and saturate the card (set on the 5070)
- ☐ **Activation / dtype bookkeeping**: confirm what lives where in VRAM during a real run

## 6. Post-training  ◐ (SFT in `posttrain/`, verified by `scripts/test_sft.py`)

**SFT (supervised fine-tuning).**
*What it is:* Continue training the pretrained base on (prompt, ideal-reply) chat examples
formatted with the ChatML template, but compute the loss **only on the assistant's reply
tokens** (the system/user prompt is masked with -100). Lower LR, a few epochs.
*Why it was developed:* A pretrained model only continues text; it doesn't know it's an
assistant or when to stop. SFT / instruction tuning (InstructGPT, Alpaca lineage) teaches
the chat format and answering behavior by imitating good responses.
*Why we use it here:* it's the step that turns our text-completer into something you can
chat with. Assistant-only masking is essential: supervising the prompt would teach it to
parrot questions, whereas masking focuses the gradient on producing good replies. We reuse
the pretraining trainer unchanged (the -100 labels flow through the same loss), just with a
lower LR and short schedule.

**DPO (Direct Preference Optimization).**  ☑ (`posttrain/dpo.py`, verified by `scripts/test_dpo.py`)
*What it is:* After SFT, fine-tune on preference pairs (prompt, chosen, rejected). DPO
raises the policy's log-prob of the chosen reply and lowers the rejected one, measured
*relative to a frozen reference* (the SFT model) with a KL anchor (the beta term). No
reward model, no RL loop.
*Why it was developed:* RLHF (InstructGPT) aligns models to preferences but needs a
trained reward model plus PPO, which is complex and unstable. DPO (Rafailov et al., 2023)
showed the same preference objective can be optimized directly with a simple
classification-style loss, vastly simpler to run.
*Why we use it here:* it's the tractable, at-home way to make replies more helpful and
preferred after SFT, needing only a frozen copy of the model and a logsigmoid loss.
Memory note: two model copies are resident, so on 12GB use a small batch + gradient
checkpointing.

- ☐ **Reasoning / CoT data**: teaching "thinking" (include reasoning traces in SFT)
- ☐ **(stretch) GRPO RL**: on checkable math

## 7. Inference  ◐ (`rag/`, verified by `scripts/test_rag.py`)

**RAG (retrieval-augmented generation).**
*What it is:* At inference, retrieve text relevant to the question (from a knowledge base
via the TF-IDF retriever, from uploaded attachments, and/or a live web search), insert it
into the prompt as labeled context, and have the model answer from it.
*Why it was developed:* LLMs hallucinate facts they don't reliably store, and their
knowledge is frozen at training time. RAG (Lewis et al., 2020) grounds generation in
retrieved documents, improving factuality and adding fresh/private knowledge without
retraining.
*Why we use it here:* it's the single biggest usability lever for a small model. A 1B
can't memorize the world, but it can reason over text you hand it; retrieval turns its job
from "recall" (bad at) into "read and synthesize" (good at). Our retriever is
dependency-free TF-IDF; attachments and web feed the same pipeline.

**Attachment ingestion.**
*What it is:* Upload text-based files (txt/md/csv/code now; pdf/docx/pptx via parsers),
extract raw text, feed it as context (whole if small, chunked into the retriever if big).
*Why it was developed / why us:* "can I attach a file" was direct user feedback;
auto-extracting on upload (rather than relying on the model to call a tool) is simpler and
more reliable at this scale, and big files are chunked because the context window is small.

**Web search tool.**
*What it is:* Returns snippets for a query (DuckDuckGo, no API key), feeding the same
context pipeline; degrades to empty offline.
*Why us:* it's the model's "internet access" and the biggest knowledge boost: it can look
up anything current, then phrase an answer from what it found.

**Input hygiene (light).**
*What it is:* Untrusted text (attachments, web) is wrapped and labeled "data, not
instructions," with length caps; no heavy prompt-injection defenses.
*Why:* no secrets or side-effectful tools means heavy defenses are overkill, but
attachments/web bring in untrusted text, so the cheap label-and-cap habit is exactly the
surface worth bothering with.

- ☑ **Sampling** (temperature / top-k / top-p): in `model/generate.py` (see §8).
- ◐ **Tool / function calling**: the web tool + ChatML tool tokens exist; teaching the model to *emit* structured tool calls (SFT on tool-use traces) + an execute loop is a follow-up.
- (note) **Quantization / serving**: the ~1B serves comfortably in bf16 (~2GB) on the 5070, so quantization isn't needed for fast local inference. GGUF/llama.cpp would need a custom converter for our architecture, so we skip it; int8/4-bit via bitsandbytes is an optional future footprint reduction.
- ☐ **self-consistency / best-of-N**: sample several reasoning paths and vote (a later quality knob).

## 8. Evaluation  ◐ (`eval/`, verified by `scripts/test_eval.py`; benchmark loaders need PC data)

**Validation perplexity.**
*What it is:* Periodically run the model on held-out tokens it never trains on and report
the average next-token loss (and its exponential, perplexity).
*Why it was developed:* Training loss can fall just from memorizing; held-out loss
measures generalization. It's been the standard yardstick for language models for decades.
*Why we use it here:* It's the most reliable "is the run healthy and improving" signal
over months, it's cheap, and it's what the dashboard trends.

**Multiple-choice benchmarks (likelihood scoring).**
*What it is:* For tasks like HellaSwag and ARC, score each answer by the model's
length-normalized log-probability of that continuation given the context, and pick the
highest; report accuracy.
*Why it was developed:* Base (non-chat) models can't reliably be told to "answer A/B/C/D",
so the field scores choices by likelihood instead. It's how the standard harnesses
(EleutherAI lm-eval) evaluate base models.
*Why we use it here:* It gives a comparable, public number against known small models and
a capability signal beyond perplexity. Scoring is verified here; the dataset loaders run
on the PC.

**Fixed-prompt generation tracking.**
*What it is:* Generate from a fixed set of prompts at each eval, saving each output with
the step and generation speed, so we can scroll the same prompt across checkpoints.
*Why it was developed / why we use it:* numbers don't capture "does it sound good";
watching a fixed prompt's output improve over training is the most motivating and honest
qualitative signal, and it feeds the dashboard's sample archive.

**Generation (sampling).**
*What it is:* Autoregressive decoding with temperature + top-k + top-p (nucleus)
filtering (`model/generate.py`).
*Why it was developed:* greedy decoding is repetitive and bland; temperature/top-k (Fan
et al., 2018) and nucleus sampling (Holtzman et al., 2019) truncate the unreliable tail
of the distribution to trade coherence against diversity.
*Why we use it here:* it's needed for sample generations now and for chat/RAG later; one
function serves both.

## 9. Engineering  ◐ (checkpointing + divergence guard in `train/`, verified by `scripts/test_train.py`)

**Atomic checkpointing + resume.**
*What it is:* A checkpoint saves everything needed for exact continuation (weights,
optimizer state, step, data-loader position, configs, seed, RNG). Writes go to a temp
file then `os.replace` (atomic), we keep the last K plus a `latest.json` pointer, and
training resumes from latest automatically.
*Why it was developed:* Long runs crash, get killed, or lose power; saving only weights
loses optimizer momentum and the data position, and a non-atomic save can leave a
half-written, corrupt file if it dies mid-write. The temp-then-rename pattern and
full-state checkpoints are the standard fixes.
*Why we use it here:* It's the backbone of your pause/play workflow and power-outage
safety: a crash or pause costs at most the time since the last save, and resuming is
bit-for-bit a continuation (verified in the test).

**Divergence guard + auto-rollback.**
*What it is:* After each step we check the loss and gradient norm; if either is NaN/Inf,
we don't apply the step, log a divergence event, and roll back to the last good
checkpoint instead of continuing on corrupted weights.
*Why it was developed:* Large bf16 runs occasionally hit an instability (a bad batch, a
spike) that turns weights to NaN; without a guard the whole run silently dies and you
discover it hours or days later. Spike-detection + rollback is standard practice in big
training stacks.
*Why we use it here:* The run is unattended for long stretches, so a silent NaN could
waste days. The guard makes the run self-healing: it reverts and keeps going (verified in
the test). Pairs with the autosave and surfaces on the dashboard.

- ◐ **Run reproducibility**: we snapshot config + seed + RNG state in each checkpoint; still to add: stamp the git commit too

**Pause/play subprocess supervisor (stop-file).**  ☑ (`gui/server.py`, verified by `scripts/test_gui.py`)
*What it is:* The GUI runs training as a child process and controls it with a stop-file:
pause creates the file, the trainer sees it at the next step, checkpoints, and exits (so
the OS reclaims all VRAM); play deletes the file and relaunches `train.py`, which
auto-resumes from the latest checkpoint.
*Why this design:* the goal is to free the GPU on demand (to game) and resume later.
Killing mid-step loses progress; OS signals (SIGINT/SIGTERM) are awkward and inconsistent
on Windows (the training box). A stop-file is dead-simple, cross-platform, and lets the
trainer stop at a clean step boundary with a fresh checkpoint.
*Why we use it here:* it's exactly the "pause to game, unpause before bed" workflow, and
since exit frees VRAM and resume is exact, pausing costs only the few seconds to
checkpoint. Verified start/pause/play end-to-end.

**Mission-control dashboard.**  ◐ (`gui/`, core panels done; MFU / VRAM / sample-gen archive to add)
*What it is:* A local FastAPI + websocket web app that streams the live terminal
(xterm.js) and parsed metrics (uPlot charts for loss + LR, plus step / progress /
tokens-per-sec / grad-norm / ETA) from the run's `metrics.jsonl`, with play/pause/stop.
*Why build our own:* a bare terminal makes a long run opaque; a dashboard makes progress,
throughput, and stability visible at a glance and gives one-click control. We built it
in-repo (rather than wandb/TensorBoard) so it's local, zero-setup, controls the run, and
looks the way we want.
*Why we use it here:* it turns a months-long unattended run into something monitorable and
pausable at a glance, and it reads the JSONL the trainer already writes, so it adds no
training overhead. Still to add: MFU and VRAM/temp readouts, and the sample-generation
archive (watch a fixed prompt improve across checkpoints).
