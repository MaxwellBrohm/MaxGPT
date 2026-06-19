"""Turn a weighted, mixed text stream into memmapped uint16 token shards.

- Documents are separated by the tokenizer's <|endoftext|> id.
- uint16 (vocab 49,152 fits in 16 bits) halves shard size vs uint32.
- The mix is sampled *proportionally* (by weight) as it streams, and we stop at a token
  budget, so a capped run keeps the full blend (web/textbook/code/math/wiki) rather than
  filling up on one source. Per-source token counts are recorded in meta.json.

Real run streams the sources below with HuggingFace `datasets` (on the 5070 box). The Mac
smoke test passes plain strings, which still work (source = None).
Verify the dataset ids/fields on HF before the real run; they drift over time.
"""
from __future__ import annotations

import json
import os
from typing import Iterable

import numpy as np

DTYPE = np.uint16

# Pretraining mix. Each `weight` is a *document* sampling probability, but the budget is counted
# in TOKENS, and document length varies a lot by source (code files and math pages are long;
# cosmopedia snippets are short). So these weights are tuned from a measured shakedown run so the
# realized TOKEN shares land near the intended blend: ~55% web-edu, 22% textbook, 10% code, 8%
# math, 5% wiki. (Raw doc-weights of 0.55/0.22/0.10/0.08/0.05 gave token shares of 46/13/23/14/4,
# because per document code over-counts ~2.3x and cosmopedia under-counts ~0.6x.)
# Code note: smollm-corpus "python-edu" stores blob_ids (text lives on S3), not inline text, so
# codeparrot-clean (inline `content`, no auth) supplies the code slice instead.
PRETRAIN_MIX = [
    {"path": "HuggingFaceTB/smollm-corpus", "name": "fineweb-edu-dedup", "text_field": "text",    "weight": 0.55},
    {"path": "HuggingFaceTB/smollm-corpus", "name": "cosmopedia-v2",     "text_field": "text",    "weight": 0.32},
    {"path": "codeparrot/codeparrot-clean",                              "text_field": "content", "weight": 0.037},
    {"path": "open-web-math/open-web-math",                              "text_field": "text",    "weight": 0.039},
    {"path": "wikimedia/wikipedia",          "name": "20231101.en",      "text_field": "text",    "weight": 0.048},
]


def stream_mixed(specs=PRETRAIN_MIX, seed: int = 0):
    """Weighted, streamed blend of the sources. Yields (text, source_name). Picks each
    document from a source with probability proportional to its weight, so the mix holds
    throughout (and therefore holds when we stop early at a token budget)."""
    import random
    from datasets import load_dataset

    rng = random.Random(seed)
    src = []
    for s in specs:
        name = s.get("name") or s["path"]
        try:
            d = load_dataset(s["path"], s.get("name"), split=s.get("split", "train"), streaming=True)
        except Exception as e:                      # one bad source must not kill the whole prep
            print(f"[data] WARNING: could not open {name}: {type(e).__name__}: {e}. Skipping it.")
            continue
        src.append({"it": iter(d), "w": float(s["weight"]),
                    "name": name, "field": s.get("text_field", "text"),
                    "alive": True})
    if not src:
        raise RuntimeError("no data sources could be opened (check network / dataset ids)")
    while any(s["alive"] for s in src):
        alive = [s for s in src if s["alive"]]
        s = rng.choices(alive, weights=[x["w"] for x in alive], k=1)[0]
        try:
            ex = next(s["it"])
        except StopIteration:
            s["alive"] = False
            continue
        text = ex.get(s["field"]) if isinstance(ex, dict) else None
        if text:
            yield text, s["name"]


def tokenize_to_shards(items: Iterable, tokenizer, out_dir: str, shard_size: int = 100_000_000,
                       eot_id: int | None = None, max_tokens: int | None = None) -> dict:
    """Encode `items` (each a str, or a (text, source) tuple), append an EOT after each doc,
    and write uint16 shards + a meta.json index. Stops at `max_tokens` if set."""
    os.makedirs(out_dir, exist_ok=True)
    if eot_id is None:
        eot_id = tokenizer.eos_id

    shards: list[dict] = []
    by_source: dict[str, int] = {}
    buf = np.empty(shard_size, dtype=DTYPE)
    fill = 0
    total = 0
    idx = 0

    def flush(n: int) -> None:
        nonlocal idx
        name = f"shard_{idx:05d}.bin"
        buf[:n].tofile(os.path.join(out_dir, name))
        shards.append({"name": name, "tokens": int(n)})
        idx += 1

    for item in items:
        text, source = item if isinstance(item, tuple) else (item, None)
        ids = tokenizer.encode(text)
        ids.append(eot_id)
        by_source[source] = by_source.get(source, 0) + len(ids)
        arr = np.asarray(ids, dtype=DTYPE)
        pos = 0
        while pos < len(arr):
            take = min(len(arr) - pos, shard_size - fill)
            buf[fill:fill + take] = arr[pos:pos + take]
            fill += take
            pos += take
            total += take
            if fill == shard_size:
                flush(shard_size)
                fill = 0
        if max_tokens and total >= max_tokens:
            break
    if fill > 0:
        flush(fill)

    meta = {"dtype": "uint16", "shard_size": shard_size, "total_tokens": int(total),
            "eot_id": int(eot_id), "shards": shards,
            "by_source": {str(k): int(v) for k, v in by_source.items()}}
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta
