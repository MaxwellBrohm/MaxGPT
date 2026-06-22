"""Autoregressive text generation.

Standard sampling loop with temperature, top-k, and top-p (nucleus) filtering. Used by
the eval harness (sample generations) and by inference / RAG. Pure model logic, so it
runs anywhere the model does.

Uses a KV cache: the prompt is encoded once (prefill), then each new token comes from a
single-token forward that reuses the cached keys/values. That makes generation O(n)
instead of O(n^2) (the old loop re-ran the whole sequence every step), which is roughly
10-50x faster for the lengths we generate.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _sample(logits: torch.Tensor, temperature: float, top_k: int | None, top_p: float | None) -> torch.Tensor:
    """logits: (B, vocab) -> (B, 1) next-token ids. temperature 0 => greedy argmax."""
    if temperature <= 0.0:
        return logits.argmax(dim=-1, keepdim=True)
    logits = logits / temperature
    if top_k:
        k = min(top_k, logits.size(-1))
        kth = torch.topk(logits, k, dim=-1).values[:, -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = F.softmax(sorted_logits, dim=-1)
        cum = probs.cumsum(dim=-1) - probs                   # cumulative prob *before* each token
        drop = cum > top_p
        sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def generate(model, ids: torch.Tensor, max_new_tokens: int = 64, temperature: float = 1.0,
             top_k: int | None = None, top_p: float | None = None,
             eos_id: int | None = None) -> torch.Tensor:
    """ids: (B, T) prompt token ids. Returns (B, T + n) with up to max_new_tokens appended.
    temperature 0 => greedy. Stops early once every sequence has emitted eos_id."""
    was_training = model.training
    model.eval()
    seq_len = model.cfg.seq_len
    ids = ids[:, -seq_len:]                                   # never start beyond the context window
    finished = torch.zeros(ids.size(0), dtype=torch.bool, device=ids.device)

    logits, past = model(ids, use_cache=True)                # prefill: encode the prompt, build the cache
    for _ in range(max_new_tokens):
        nxt = _sample(logits[:, -1, :], temperature, top_k, top_p)
        ids = torch.cat([ids, nxt], dim=1)
        if eos_id is not None:
            finished = finished | (nxt.squeeze(-1) == eos_id)
            if bool(finished.all()):
                break
        if ids.size(1) >= seq_len:                            # context window full (this cache doesn't slide)
            break
        logits, past = model(nxt, past=past, use_cache=True)  # decode: one new token, reuse the cache

    if was_training:
        model.train()
    return ids
