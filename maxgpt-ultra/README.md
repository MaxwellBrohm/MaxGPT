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

## Status — core stack built + CPU-tested
- ✅ Model architecture (`model/`) — `scripts/smoke_test.py`
- ✅ Tokenizer (`tokenizer/`) — `scripts/test_tokenizer.py`
- ✅ Data pipeline (`data/`) + one-command `scripts/prepare_data.py` — `scripts/test_data.py`
- ✅ Training loop (`train/`: WSD, checkpoint/resume, divergence guard, autosave) — `scripts/test_train.py`
- ✅ Mission-control GUI (`gui/`: pause/play, live terminal + charts) — `scripts/test_gui.py`
- ✅ Eval harness (`eval/`: perplexity, MC benchmarks, sample generations) + `model/generate.py` — `scripts/test_eval.py`
- ✅ Post-training: SFT (`posttrain/sft_data.py`, `scripts/sft.py`) — `scripts/test_sft.py`; DPO (`posttrain/dpo.py`, `scripts/dpo.py`) — `scripts/test_dpo.py`
- ✅ Inference: RAG retriever + attachments + web tool + grounded chat (`rag/`) — `scripts/test_rag.py`
- ⏭ Remaining: execute the real pretraining (download → shakedown → 1B) on the 5070; then
  wire RAG into the web UI, a structured tool-call loop, reasoning data, and GUI MFU/VRAM.

### Run the real thing (on the 5070 box)
```
python scripts/prepare_data.py --config configs/shakedown.yaml      # download -> 49k tokenizer -> shards
python gui/server.py --config configs/shakedown.yaml --data data/shards --out runs/shakedown
# open http://127.0.0.1:8800 and hit play; validate at 124M, then switch to configs/ultra.yaml
```

## Working across machines
This repo *is* the shared brain: `PLAN.md` (the full design) and `TECHNIQUES.md` (every
technique, with rationale) hold all the context. A Claude Code session is tied to one
machine, so the workflow is: develop on the Mac, `git push`, then run a **separate Claude
Code session on the Windows/5070 box** in the pulled repo for the GPU-heavy work. That
session reads PLAN.md + TECHNIQUES.md to pick up exactly where we are. Smoke tests run
anywhere (CPU); real training runs on the 5070.

To run the tests (from this folder, using the repo venv):
`../venv/bin/python scripts/smoke_test.py` · `scripts/test_tokenizer.py` · `scripts/test_data.py`
