# MaxGPT-Ultra — Build Plan

A genuinely usable, **from-scratch ~1.1B language model** trained at home on a single
RTX 5070 (12GB), then made actually useful with instruction tuning and retrieval.
Learning + portfolio capstone. Successor to the MaxGPT series.

## Honest expectations
- A 1B model from scratch on one 12GB GPU lands in the strong *small-model* class
  (SmolLM2 / TinyLlama tier): coherent, conversational, follows instructions,
  writes/summarizes, decent reasoning for its size. **Not** GPT-4.
- It can't memorize the world in 1B weights. **RAG** (retrieval + web at inference)
  closes the factual gap and is what makes it genuinely useful.
- Quality comes from **data + post-training**, not parameter count.

## Locked decisions
- **Model:** ~1.1B params, name **MaxGPT-Ultra**.
- **Prototype first:** the **shakedown** (~124M) is an *exact scaled-down replica* —
  same architecture, data, schedule, post-training, GUI. Debug the whole pipeline in
  days, then flip the config to 1B with confidence.
- **Schedule:** **WSD** (warmup → stable → decay). Aim high on tokens (time is no
  object); decay-and-ship at the target and keep the pre-decay checkpoint so the run
  can be *extended* later. (Infinite training is wasteful — it throws away the decay
  drop. A target + WSD is the sophisticated answer.)
- **Token budget:** ~100B target (ship earlier via the decay if desired).
- **Venue:** all training at home on the 5070 (no cloud, by choice).
- **Data cost:** ~$0 — free open data + free *distilled* open datasets; optional
  free top-ups via the Claude Max subscription.

## Architecture (modern decoder-only — Llama/Qwen/SmolLM2 lineage)
RoPE positions · RMSNorm · SwiGLU MLP · GQA (grouped-query attention) · QK-norm ·
tied embeddings · no biases · pre-norm · z-loss. Sizes in `configs/`.

## Data
**Bulk pretrain (free):** FineWeb-Edu (backbone) + Cosmopedia v2 (synthetic
textbooks) + code (StarCoder subset) + math (FineMath/OpenWebMath) + a little
Wikipedia/books. Tokenized to memmapped shards. **Tokenizer: ~49k byte-level BPE**
(49,152 vocab, digit-splitting + byte fallback), trained on the corpus mix. Bigger than
32k for better compression on code/math, at an embedding cost the 1B absorbs (it's
SmolLM2's vocab size). We reserve **special tokens up front** (ChatML-style chat roles,
tool-call markers, `<think>` tags, BOS/EOS/pad, plus spare slots) so chat, tools, and
reasoning work cleanly without resizing the vocab later. Mixture weights across sources
are a tuned quality lever, and we pack documents to fill sequences (no padding waste).
**Curriculum / annealing:** save the highest-quality data (textbook/instruction/
math/code) for the WSD **decay** phase — proven quality bump.
**SFT (free, distilled):** OpenHermes 2.5, Tülu 3, UltraChat, OASST2.
**Reasoning (free, distilled):** OpenThoughts / OpenR1 CoT traces.
**Preferences for DPO (free):** UltraFeedback.
**Optional custom top-ups:** Claude Max subscription via headless `claude` CLI /
Agent SDK (spends plan usage, not API $) for targeted gaps. Mind Anthropic usage
terms re: competing models; keep personal/non-commercial.

## Training recipe
bf16 · FlashAttention/SDPA · `torch.compile` · gradient checkpointing (1B) ·
8-bit AdamW (1B, to fit optimizer states in 12GB) · large effective batch via
grad-accum (~0.5M tokens) · grad clip 1.0 · WSD schedule · QK-norm + z-loss for
stable high-LR training · **Muon optimizer** optional for faster convergence.

## Quality levers (the "even better" list)
1. **Distillation / synthetic data** from a bigger teacher (mostly via free
   already-distilled open sets; optional subscription top-ups).
2. **Final-phase data annealing** (best data saved for the decay phase).
3. **QK-norm + z-loss** (stability → higher LR → faster, better).
4. **Muon optimizer** (more quality per hour).
5. **Auto-eval harness** at every checkpoint (don't fly blind over months).
6. **Test-time compute:** self-consistency / best-of-N for reasoning.

## Post-training (makes it an assistant)
1. **SFT** on instruction/chat + reasoning data, with a chat template.
2. **DPO** on preference data.
3. *(Stretch)* RL reasoning (GRPO on checkable math) later.

## Inference (makes it useful)
**RAG** (retriever + web-search tool) · tool/function calling (taught in SFT) ·
good sampling · GGUF/llama.cpp quantization · wired into the web UI + side-by-side
compare · optional self-consistency / best-of-N.

**Attachments (text-based files).** Let the user upload txt/md/docx/pptx/pdf/csv/code
and extract them to raw text, then inject as clearly-labeled context. Pure inference-layer
feature (no retraining). Key constraint: our context is small (2048, extendable), so big
files get chunked into the RAG retriever and only the relevant parts are pulled in per
question; small files inject whole. SFT on document-grounded QA makes the model actually
good at using them.

**Input hygiene (deliberately light).** No sensitive data and no side-effectful tools, so
we skip heavy prompt-injection defenses. But because attachments and web results bring in
untrusted text, we do the cheap, worthwhile basics: wrap that text in clear delimiters and
label it as data (not instructions), and cap lengths so it can't blow the context. That is
the whole security surface worth building for a personal model.

## Training GUI — the "mission control" dashboard
A dense, dark, monospace, hacker-aesthetic local web dashboard: looks cool to anyone,
fully legible to someone who knows ML. Compact panels, sparklines, data everywhere.
Stack: FastAPI + a websocket metrics stream; **xterm.js** for the real terminal panel;
**uPlot** for fast, compact live charts.

Panels:
- **Live terminal** — the raw training stdout, exactly as it scrolls in a shell (xterm.js).
- **Progress** — bar (tokens seen / target), step, % complete, **ETA** to target + to next checkpoint.
- **Loss** — train + val curve from step 0 → now (log-y, smoothed) + current value.
- **Learning rate** — LR curve showing the WSD phases (warmup / stable / decay), current LR, **current decay %**.
- **Throughput** — tokens/sec (live + sparkline), **MFU %** (model-FLOPs utilization), step time.
- **Hardware** — VRAM used/max, GPU util %, temp, power draw (via nvidia-smi).
- **Stability** — gradient norm (pre-clip), z-loss, spike flags.
- **Eval** — benchmark scores (HellaSwag / ARC / MMLU-slice) + val perplexity over time.
- **Sample generations** — outputs from a fixed test-prompt set, saved each eval with step,
  timestamp, and **gen speed (tok/s)**. An **archive** scrolls the SAME prompt across
  checkpoints (watch it get smarter) and compares any two side by side.
- **Checkpoints** — list with step, size, eval score; load / resume / mark-best.
- **Controls** — Pause / Play / Stop, autosave status + last-save time.

Mechanics:
- Supervisor runs training as a **subprocess**. **Pause** = checkpoint + exit the
  subprocess → frees *all* VRAM for games. **Play** = relaunch from checkpoint.
- **Autosave:** atomic write (temp file + rename) every ~10–15 min; rolling last-K +
  best-by-eval; autosave on graceful close; resume-from-latest on launch.
  A power outage costs at most one autosave interval.

## Checkpoint/resume guarantee
Resuming == continuous training, as long as we save **weights + optimizer state +
LR-schedule position + data position** (we will). bf16/cuDNN means not bit-identical
but quality-identical. Pause as often as you like — zero downside.

## Roadmap (build order)
0. **Repo scaffold + this plan.** ✅
1. **Model architecture** (modern decoder) + CPU smoke test. ✅
2. **Tokenizer + data pipeline**. Tokenizer module ✅ and shard/loader pipeline ✅
   (byte-level BPE + special tokens; packed, resumable memmap loader; all verified).
   One-command driver `scripts/prepare_data.py` ✅ (download mix -> train 49k tokenizer
   -> shard); runs on the 5070 box (`--smoke` dry-run verified on the Mac).
3. **Training loop:** WSD, bf16, checkpoint/resume, autosave, logging, eval hooks. ✅
   (built + CPU-tested: loss falls, exact resume, divergence guard + rollback; runs on GPU)
4. **GUI:** dashboard + pause/play + subprocess supervisor. ✅ core
   (FastAPI + websocket; xterm terminal + loss/LR charts + stats; stop-file pause that
   checkpoints & frees VRAM; backend-tested + renders. To add: MFU/VRAM, sample-gen archive)
5. **Shakedown run (124M)** end-to-end: pretrain → SFT → DPO → RAG → measure real
   throughput on the 5070.
6. **Scale config to 1B**, launch the MaxGPT-Ultra pretrain (the long haul).
7. **Post-train (SFT ✅ → DPO next) + RAG + quantize**, wire into the web UI.

## Workflow (dev vs train)
Develop on the **Mac** (write/test code, CPU smoke tests) → push to **GitHub** →
pull on the **Windows PC** → train on the **RTX 5070**. Code stays
cross-platform / CUDA.

## Time (rough, to be pinned down by the shakedown)
~5–8k tokens/sec for the 1B on a 5070 → roughly **3 months (50B tokens) to 6 months
(100B tokens)** of actual compute, longer in wall-clock with part-time/overnight use.
The shakedown will measure exact throughput so we can nail this down.
