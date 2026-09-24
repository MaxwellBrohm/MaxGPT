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
    "Café, naïve, Zürich, 北京, 🚀 - byte-level tokenization handles all of it losslessly.",
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

    print("\n[5] shard build: crash in the middle of an encode chunk + resume is byte-identical")
    import glob, json as _json, shutil

    class FakeStream:                       # a resumable doc source, like the real MixedStream
        def __init__(self, docs, crash_after=None):
            self.docs, self.i, self.crash_after = docs, 0, crash_after

        def __iter__(self):
            while self.i < len(self.docs):
                if self.crash_after is not None and self.i >= self.crash_after:
                    raise RuntimeError("simulated crash")
                d = self.docs[self.i]
                self.i += 1
                yield (d, "fake")

        def state_dict(self):
            return {"i": self.i}

        def load_state_dict(self, st):
            self.i = int(st["i"])

    docs5 = [f"doc {i}: " + " ".join(DOCS[(i * 7 + k) % len(DOCS)] for k in range(1 + i % 3)) for i in range(300)]

    def build(out, stream):
        shutil.rmtree(out, ignore_errors=True)
        # shard_size small and batch_docs large, so shard boundaries fall INSIDE encode chunks
        return tokenize_to_shards(stream, tok, out, shard_size=700, batch_docs=32)

    def shard_bytes(out):
        return b"".join(open(f, "rb").read() for f in sorted(glob.glob(os.path.join(out, "shard_*.bin"))))

    ref = build(SHARDS + "_ref", FakeStream(docs5))
    out5 = SHARDS + "_crash"
    try:
        build(out5, FakeStream(docs5, crash_after=150))
        raise AssertionError("the simulated crash did not fire")
    except RuntimeError:
        pass
    prog = _json.load(open(os.path.join(out5, "progress.json")))
    assert len(prog["pending"]) > 1, "expected several pulled-but-unwritten docs pending mid-chunk"
    resumed = tokenize_to_shards(FakeStream(docs5), tok, out5, shard_size=700, batch_docs=32)
    assert resumed["total_tokens"] == ref["total_tokens"], (resumed["total_tokens"], ref["total_tokens"])
    assert shard_bytes(out5) == shard_bytes(SHARDS + "_ref"), "resumed build differs from the uninterrupted one"
    assert not os.path.exists(os.path.join(out5, "progress.json")), "finished build should drop progress.json"
    print(f"  {len(prog['pending'])} docs were pending at the crash; resumed build identical "
          f"({ref['total_tokens']} tokens, {len(ref['shards'])} shards) ✓")

    print("\n[6] many shards: lazy memmaps stay under the open-file cap and read the same bytes")
    import json as _json2
    many = SHARDS + "_many"
    shutil.rmtree(many, ignore_errors=True); os.makedirs(many)
    rows, expect = [], []
    for i in range(40):
        a = np.arange(i * 100, i * 100 + 50 + i, dtype=np.uint16)        # distinct, varying lengths
        a.tofile(os.path.join(many, f"shard_{i:05d}.bin")); rows.append({"name": f"shard_{i:05d}.bin", "tokens": int(len(a))}); expect.append(a)
    _json2.dump({"dtype": "uint16", "eot_id": 1, "shards": rows, "total_tokens": int(sum(len(a) for a in expect))}, open(os.path.join(many, "meta.json"), "w"))
    dm = PackedShardDataset(many, 64)
    flat = np.concatenate(expect)
    got = dm._read(0, len(flat))
    assert np.array_equal(got, flat), "lazy reads differ from the concatenated shards"
    assert len(dm._open) <= PackedShardDataset.MAX_OPEN, len(dm._open)
    # a resume position deep in the stream, spanning a shard boundary
    pos = int(dm.cum[17]) - 5
    assert np.array_equal(dm._read(pos, 20), flat[pos:pos + 20])
    print(f"  40 shards read exactly with {len(dm._open)} files open (cap {PackedShardDataset.MAX_OPEN}) ✓")

    print("\n[7] local jsonl source (chat rendered as ChatML) streams and resumes exactly")
    from data.prepare import MixedStream, LocalJsonlSource
    from tokenizer.tokenizer import IM_START, IM_END
    cj = SHARDS + "_chat.jsonl"
    with open(cj, "w", encoding="utf-8") as f:
        for i in range(30):
            f.write(_json.dumps({"messages": [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"a{i}"}]}) + "\n")
        f.write("\n")                                     # a blank line must be skipped, not crash
    spec = [{"local": cj, "name": "chat", "render": "chatml", "weight": 1.0}]
    full = [t for t, src in MixedStream(spec, seed=0)]
    assert len(full) == 30 and full[3] == f"{IM_START}user\nq3{IM_END}\n{IM_START}assistant\na3{IM_END}\n", full[3]
    ms = MixedStream(spec, seed=0)
    it = iter(ms)
    first = [next(it) for _ in range(12)]
    state = ms.state_dict()
    ms2 = MixedStream(spec, seed=0)
    ms2.load_state_dict(state)
    rest = [t for t, _ in ms2]
    assert [t for t, _ in first] + rest == full, "resume from a saved state changed the stream"
    print(f"  30 chats rendered; resume after 12 reproduces the remaining 18 exactly ✓")

    print("\n" + "=" * 72)
    print("ALL CHECKS PASSED ✅")
    print("=" * 72)


if __name__ == "__main__":
    main()
