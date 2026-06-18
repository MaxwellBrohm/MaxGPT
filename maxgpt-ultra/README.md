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
Scaffolding. Next: the model architecture + a CPU smoke test.
