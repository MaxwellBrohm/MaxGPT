# MaxGPT-Ultra

A from-scratch ~1.1B language model, trained at home on an RTX 5070, then
instruction-tuned and given retrieval so it's genuinely usable. Built on a modern
decoder architecture (RoPE · RMSNorm · SwiGLU · GQA), with a pause/play training GUI.

**Read [PLAN.md](PLAN.md) for the full design, decisions, and roadmap.**

Develop here on the Mac; train on the Windows PC with the 5070 (pull from GitHub).

## Layout
- `model/` — the transformer (architecture, attention, blocks)
- `tokenizer/` — 32k BPE tokenizer (train + load)
- `data/` — dataset download, filtering, tokenization to memmapped shards
- `train/` — training loop, WSD schedule, checkpoint/resume, autosave, logging
- `eval/` — perplexity + benchmark + chat vibe-test harness
- `posttrain/` — SFT and DPO
- `rag/` — retrieval + web-search tool for inference
- `gui/` — local pause/play training dashboard
- `configs/` — `shakedown.yaml` (124M prototype) and `ultra.yaml` (~1.1B)

## Status
- ✅ Model architecture (`model/`) + smoke test (`scripts/smoke_test.py`)
- ✅ Tokenizer (`tokenizer/`) + smoke test (`scripts/test_tokenizer.py`)
- ✅ Data pipeline (`data/`: shard writer + packed resumable loader) + test (`scripts/test_data.py`)
- ⏭ Next: download the corpus + train the real 49k tokenizer + shard it (on the 5070 box), then the training loop.

## Working across machines
This repo *is* the shared brain: `PLAN.md` (the full design) and `TECHNIQUES.md` (every
technique, with rationale) hold all the context. A Claude Code session is tied to one
machine, so the workflow is: develop on the Mac, `git push`, then run a **separate Claude
Code session on the Windows/5070 box** in the pulled repo for the GPU-heavy work. That
session reads PLAN.md + TECHNIQUES.md to pick up exactly where we are. Smoke tests run
anywhere (CPU); real training runs on the 5070.

To run the tests (from this folder, using the repo venv):
`../venv/bin/python scripts/smoke_test.py` · `scripts/test_tokenizer.py` · `scripts/test_data.py`
