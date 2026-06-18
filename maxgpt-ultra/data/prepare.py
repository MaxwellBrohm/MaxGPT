"""Turn a text stream into memmapped uint16 token shards for fast training.

- Documents are separated by the tokenizer's <|endoftext|> id, so sequence packing
  later can flow across document boundaries with a clean marker.
- uint16 is safe: our vocab (49,152) fits in 0..65535, and it halves shard size vs uint32.
- The real run streams the source mix below with HuggingFace `datasets` (on the 5070 box,
  where the data should live). The Mac smoke test calls `tokenize_to_shards` on a tiny
  local sample, so it needs no network and no `datasets` install.

Verify the exact dataset ids/fields on HF before the real run; they drift over time.
"""
from __future__ import annotations

import json
import os
from typing import Iterable

import numpy as np

DTYPE = np.uint16

# The pretraining mix. `weight` is the target token-share (a tuned quality lever, see
# TECHNIQUES). SmolLM-corpus bundles the proven FineWeb-Edu-dedup + Cosmopedia-v2 +
# Python-Edu subsets; we add web-math and a little Wikipedia.
PRETRAIN_MIX = [
    {"path": "HuggingFaceTB/smollm-corpus", "name": "fineweb-edu-dedup", "text_field": "text", "weight": 0.55},
    {"path": "HuggingFaceTB/smollm-corpus", "name": "cosmopedia-v2",     "text_field": "text", "weight": 0.22},
    {"path": "HuggingFaceTB/smollm-corpus", "name": "python-edu",        "text_field": "text", "weight": 0.10},
    {"path": "open-web-math/open-web-math",                              "text_field": "text", "weight": 0.08},
    {"path": "wikimedia/wikipedia",          "name": "20231101.en",      "text_field": "text", "weight": 0.05},
]


def stream_mixed(specs=PRETRAIN_MIX, seed: int = 0):
    """Weighted, streamed interleave of the source mix. Runs on the training box
    (needs `datasets` + network). Yields raw text strings."""
    from datasets import load_dataset, interleave_datasets

    dsets, probs = [], []
    for s in specs:
        d = load_dataset(s["path"], s.get("name"), split=s.get("split", "train"), streaming=True)
        field = s.get("text_field", "text")
        if field != "text":
            d = d.rename_column(field, "text")
        dsets.append(d)
        probs.append(float(s["weight"]))
    total = sum(probs)
    probs = [p / total for p in probs]
    mixed = interleave_datasets(dsets, probabilities=probs, seed=seed,
                                stopping_strategy="all_exhausted")
    for ex in mixed:
        text = ex.get("text")
        if text:
            yield text


def tokenize_to_shards(texts: Iterable[str], tokenizer, out_dir: str,
                       shard_size: int = 100_000_000, eot_id: int | None = None) -> dict:
    """Encode `texts`, append an EOT after each doc, and write fixed-size uint16 shards
    plus a meta.json index. Returns the meta dict."""
    os.makedirs(out_dir, exist_ok=True)
    if eot_id is None:
        eot_id = tokenizer.eos_id

    shards: list[dict] = []
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

    for text in texts:
        ids = tokenizer.encode(text)
        ids.append(eot_id)
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
    if fill > 0:
        flush(fill)

    meta = {
        "dtype": "uint16",
        "shard_size": shard_size,
        "total_tokens": int(total),
        "eot_id": int(eot_id),
        "shards": shards,
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta
