"""Evaluation harness.

Three things, so we can tell whether a months-long run is actually improving:
  - validation perplexity (held-out next-token loss): the core "is it learning" signal,
  - multiple-choice accuracy by length-normalized log-likelihood (how HellaSwag / ARC are
    scored for base models): does it pick the sensible continuation,
  - fixed-prompt sample generations: qualitative "watch it get smarter" outputs (with
    generation speed), which the dashboard archives across checkpoints.

`evaluate(...)` bundles whatever you pass into one metrics dict; the trainer logs it as an
`event:"eval"` row in metrics.jsonl. Benchmark loaders use `datasets` (PC / network).
"""
from __future__ import annotations

import math
import time

import torch

from model.generate import generate


@torch.no_grad()
def eval_perplexity(model, data, n_batches: int = 40, batch_size: int = 8, device: str = "cpu") -> dict:
    """Mean cross-entropy over the FIRST n_batches * batch_size windows of `data`: the same held-out
    slice on every call, so successive evals are comparable. It used to read on from wherever the
    previous eval had stopped, and slices of the held-out set differ by up to 0.4 nats, which made
    the Ultra run's val curve bounce between 8.7 and 13 ppl while the training loss moved smoothly
    (evals before step ~6,500 on 2026-09-24 are on moving slices; later ones on this fixed slice)."""
    model.eval()
    if hasattr(data, "pos"):                 # rewind the held-out stream: same windows every time
        data.pos, data.epoch, data._block_i = 0, 0, 0
    total, n, toks = 0.0, 0, 0
    for _ in range(n_batches):
        x, y = data.next_batch(batch_size, device)
        _, loss = model(x, y)            # plain cross-entropy (no z-loss for eval)
        total += float(loss)
        n += 1
        toks += int(x.numel())
    avg = total / max(1, n)
    return {"val_loss": avg, "val_ppl": math.exp(min(20.0, avg)), "val_tokens": toks}


@torch.no_grad()
def activation_stats(model, data, batch_size: int = 2, device: str = "cpu") -> dict:
    """fp16 headroom telemetry (docs/research_2026-09-22.md 2.7): on one held-out batch, the
    largest |value| per layer in the residual stream after each block and in each SwiGLU output.
    fp16 tops out at 65,504; the one published fp16 pretraining run on pre-Ampere hardware saw
    activations above 10,000 after 1T tokens, and sandwich/QK-norm plus a per-head gate (both
    here) measured peaks of ~100-200 at 7B. This is measured in fp32 (the eval forward), so it
    shows the true magnitude even where fp16 would already have overflowed."""
    raw = model
    for attr in ("module", "_orig_mod", "module"):
        raw = getattr(raw, attr, raw)
    blocks = list(raw.blocks)
    resid, mlp = [None] * len(blocks), [None] * len(blocks)
    hooks = []
    for i, b in enumerate(blocks):
        hooks.append(b.register_forward_hook(
            lambda m, inp, out, i=i: resid.__setitem__(i, float(out[0].detach().float().abs().max()))))
        hooks.append(b.mlp.register_forward_hook(
            lambda m, inp, out, i=i: mlp.__setitem__(i, float(out.detach().float().abs().max()))))
    try:
        model.eval()
        if hasattr(data, "pos"):
            data.pos, data.epoch, data._block_i = 0, 0, 0
        x, _ = data.next_batch(batch_size, device)
        model(x)
    finally:
        for h in hooks:
            h.remove()
    resid = [round(v, 2) if v is not None else None for v in resid]
    mlp = [round(v, 2) if v is not None else None for v in mlp]
    vals = [v for v in resid + mlp if v is not None]
    return {"act_max_resid": resid, "act_max_mlp": mlp, "act_max": max(vals) if vals else None}


@torch.no_grad()
def _choice_logprob(model, prompt_ids: list[int], choice_ids: list[int], device: str) -> float:
    """Length-normalized log-likelihood of `choice_ids` continuing `prompt_ids`."""
    ids = torch.tensor([prompt_ids + choice_ids], device=device)
    logits, _ = model(ids)
    logp = torch.log_softmax(logits[0].float(), dim=-1)
    start = len(prompt_ids)
    total = 0.0
    for i, tok in enumerate(choice_ids):
        total += float(logp[start - 1 + i, tok])     # position predicting choice token i
    return total / max(1, len(choice_ids))


@torch.no_grad()
def eval_multiple_choice(model, tokenizer, examples, device: str = "cpu") -> dict:
    """examples: [{'context': str, 'choices': [str, ...], 'answer': int}]."""
    model.eval()
    correct = 0
    for ex in examples:
        ctx = tokenizer.encode(ex["context"])
        scores = [_choice_logprob(model, ctx, tokenizer.encode(c), device) for c in ex["choices"]]
        pred = max(range(len(scores)), key=lambda i: scores[i])
        correct += int(pred == ex["answer"])
    return {"mc_acc": correct / max(1, len(examples)), "mc_n": len(examples)}


@torch.no_grad()
def run_sample_prompts(model, tokenizer, prompts, device: str = "cpu", max_new_tokens: int = 48,
                       temperature: float = 0.8, top_p: float = 0.95, chat: bool = False) -> list[dict]:
    model.eval()
    out = []
    for p in prompts:
        # chat=True wraps the prompt in ChatML so SFT/DPO samples show the assistant's reply
        text = (tokenizer.render_chat([{"role": "user", "content": p}], add_generation_prompt=True)
                if chat and hasattr(tokenizer, "render_chat") else p)
        ids = torch.tensor([tokenizer.encode(text)], device=device)
        t0 = time.time()
        gen = generate(model, ids, max_new_tokens=max_new_tokens, temperature=temperature,
                       top_p=top_p, eos_id=tokenizer.eos_id)
        dt = time.time() - t0
        new = gen[0, ids.shape[1]:].tolist()
        out.append({"prompt": p, "completion": tokenizer.decode(new, skip_special=True),
                    "new_tokens": len(new), "tok_per_s": len(new) / max(dt, 1e-6)})
    return out


def evaluate(model, tokenizer=None, val_data=None, mc_examples=None, sample_prompts=None,
             device: str = "cpu", **kw) -> dict:
    """Bundle the available evals into one metrics dict (whatever inputs are provided)."""
    m: dict = {}
    if val_data is not None:
        m.update(eval_perplexity(model, val_data, device=device,
                                 n_batches=kw.get("n_batches", 40), batch_size=kw.get("batch_size", 8)))
        if hasattr(getattr(model, "module", model), "blocks") or hasattr(model, "blocks"):
            m.update(activation_stats(model, val_data, device=device))   # fp16 headroom, per layer
    if mc_examples and tokenizer is not None:
        m.update(eval_multiple_choice(model, tokenizer, mc_examples, device=device))
    if sample_prompts and tokenizer is not None:
        m["samples"] = run_sample_prompts(model, tokenizer, sample_prompts, device=device,
                                          max_new_tokens=kw.get("max_new_tokens", 48),
                                          chat=kw.get("chat", False))
        lens = [len(tokenizer.encode(s["completion"])) for s in m["samples"]]
        m["sample_mean_tokens"] = sum(lens) / max(1, len(lens))    # a sharp rise under DPO = length exploitation
    return m


# --- benchmark loaders (run on the training box; need `datasets` + network) ---
def load_hellaswag(n: int = 200, split: str = "validation"):
    from datasets import load_dataset
    ds = load_dataset("Rowan/hellaswag", split=split)
    out = []
    for ex in ds.select(range(min(n, len(ds)))):
        out.append({"context": ex["ctx"], "choices": ex["endings"], "answer": int(ex["label"])})
    return out


def load_arc(n: int = 200, split: str = "validation", subset: str = "ARC-Easy"):
    from datasets import load_dataset
    ds = load_dataset("allenai/ai2_arc", subset, split=split)
    out = []
    for ex in ds.select(range(min(n, len(ds)))):
        choices = ex["choices"]["text"]
        labels = ex["choices"]["label"]
        try:
            answer = labels.index(ex["answerKey"])
        except ValueError:
            continue
        out.append({"context": ex["question"], "choices": choices, "answer": answer})
    return out
