"""End-to-end smoke test for the data pipeline (no network).

sample text -> tiny tokenizer -> tokenize_to_shards (small shards) -> PackedShardDataset.
Verifies: shard/meta correctness, batch shapes, the next-token shift (y == x shifted),
EOT separators present, and exact resume from a saved position.

Run from maxgpt-ultra/:  ../venv/bin/python scripts/test_data.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # maxgpt-ultra/

import numpy as np
import torch

from tokenizer.tokenizer import train_tokenizer, UltraTokenizer
from data.prepare import tokenize_to_shards
from data.loader import PackedShardDataset

TOK = "/tmp/maxgpt_ultra_data_tok.json"
SHARDS = "/tmp/maxgpt_ultra_shards"

DOCS = [
    "The quick brown fox jumps over the lazy dog and then keeps on running for miles.",
    "In 2026 a high schooler trained a language model from scratch on a single GPU.",
    "def add(a, b):\n    return a + b\n\nprint(add(2, 3))  # 5",
    "Transformers attend to every token in parallel, which is why they train efficiently.",
    "Café, naïve, Zürich, 北京, 🚀 — byte-level tokenization handles all of it losslessly.",
] * 60


def main() -> None:
    print("=" * 72)
    print("MaxGPT-Ultra data-pipeline smoke test")
    print("=" * 72)

    print("\n[setup] tiny tokenizer + shards")
    train_tokenizer(iter(DOCS), vocab_size=1500, out_path=TOK)
    tok = UltraTokenizer(TOK)
    meta = tokenize_to_shards(DOCS, tok, SHARDS, shard_size=512)  # tiny shards -> several files
    print(f"  total_tokens={meta['total_tokens']}  shards={len(meta['shards'])}  eot_id={meta['eot_id']}")
    assert len(meta["shards"]) > 1, "expected multiple shards with shard_size=512"
    assert meta["total_tokens"] == sum(s["tokens"] for s in meta["shards"])

    print("\n[1] batch shapes + next-token shift")
    seq_len = 32
    ds = PackedShardDataset(SHARDS, seq_len)
    x, y = ds.next_batch(batch_size=4)
    assert x.shape == (4, seq_len) and y.shape == (4, seq_len), (x.shape, y.shape)
    assert x.dtype == torch.int64
    assert torch.equal(x[:, 1:], y[:, :-1]), "y must be x shifted by one token"
    print(f"  x={tuple(x.shape)} y={tuple(y.shape)} dtype={x.dtype}; shift relation holds ✓")

    print("\n[2] EOT separators are present in the stream")
    flat = ds._read(0, min(ds.total, 4000))
    assert (flat == meta["eot_id"]).any(), "no EOT markers found between documents"
    print(f"  found EOT (id {meta['eot_id']}) separating documents ✓")

    print("\n[3] exact resume from a saved position")
    ds.pos = 0
    ds.epoch = 0
    _ = ds.next_batch(3)               # advance a few batches
    _ = ds.next_batch(3)
    state = ds.state_dict()
    expract_x, expract_y = ds.next_batch(2)   # the batch we expect after resuming

    ds2 = PackedShardDataset(SHARDS, seq_len)
    ds2.load_state_dict(state)
    got_x, got_y = ds2.next_batch(2)
    assert torch.equal(expract_x, got_x) and torch.equal(expract_y, got_y), "resume mismatch"
    print(f"  resumed from pos={state['pos']} and reproduced the exact next batch ✓")

    print("\n[4] wraparound past end of data is seamless")
    ds3 = PackedShardDataset(SHARDS, seq_len)
    ds3.pos = ds3.total - seq_len // 2     # force a window that crosses the end
    xw, yw = ds3.next_batch(1)
    assert xw.shape == (1, seq_len) and torch.isfinite(xw.float()).all()
    assert ds3.epoch == 1, "epoch should have ticked over after wraparound"
    print(f"  window across the end read cleanly; epoch -> {ds3.epoch} ✓")

    print("\n" + "=" * 72)
    print("ALL CHECKS PASSED ✅")
    print("=" * 72)


if __name__ == "__main__":
    main()
