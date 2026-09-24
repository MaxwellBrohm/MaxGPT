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

    print("\n[8] the mix holds in TOKENS, not documents (long vs short docs, unequal chars/token)")
    # source "long": 1200-2900-char docs at 2-4 chars/token (~300-1450 tokens each); source "short":
    # 20-79-char docs at 6-10 chars/token (2-13 tokens). Sizes and chars/token vary per document, as
    # real data does, so the mixer's in-flight estimates differ from the reported counts (which is
    # what makes the resume check in [9] sensitive to the order of report vs pull). Weights 0.6/0.4 by
    # tokens. Picking documents by weight would give long ~99.6% of the tokens; picking by
    # characters (no feedback) ~83%; by tokens 60%.
    lj, sj = SHARDS + "_long.jsonl", SHARDS + "_short.jsonl"
    with open(lj, "w") as f:
        for i in range(300):
            f.write(_json.dumps({"text": "a" * (1200 + (i * 37) % 1700)}) + "\n")
    with open(sj, "w") as f:
        for i in range(40000):
            f.write(_json.dumps({"text": "b" * (20 + (i * 7) % 60)}) + "\n")
    spec8 = [{"local": lj, "name": "long", "weight": 0.6}, {"local": sj, "name": "short", "weight": 0.4}]

    class FakeTok:                                    # chars/token depends on the doc: 2-4 for "a...", 6-10 for "b..."
        eos_id = 0

        def encode_batch(self, texts):
            return [[1] * (len(t) // ((2 + len(t) % 3) if t[:1] == "a" else (6 + len(t) % 5))) for t in texts]

    fake = FakeTok()
    ms8 = MixedStream(spec8, seed=0)
    seq, served = [], {"long": 0, "short": 0}
    for text, src in ms8:                             # immediate feedback, as the writer gives per chunk
        n = len(fake.encode_batch([text])[0]) + 1
        ms8.report(src, len(text), n)
        served[src] += n
        seq.append(src)
        if sum(served.values()) >= 120_000:
            break
    share = served["long"] / sum(served.values())
    assert abs(share - 0.6) < 0.02, f"long-doc token share {share:.3f}, wanted 0.60"
    assert seq[:1000].count("long") >= 5, "sources must interleave, not run one source dry first"
    print(f"  long-doc source got {100*share:.1f}% of tokens (target 60%) from {seq.count('long')} of "
          f"{len(seq)} docs; interleaved ✓")

    print("\n[9] MixedStream through the shard writer: shares hold, crash mid-build resumes byte-identical")

    class CrashingStream(MixedStream):                # the real mixer, dying after `crash_after` docs
        def __init__(self, *a, crash_after=None, **k):
            super().__init__(*a, **k)
            self.crash_after, self.n = crash_after, 0

        def __iter__(self):
            for item in super().__iter__():
                if self.crash_after is not None and self.n >= self.crash_after:
                    raise RuntimeError("simulated crash")
                self.n += 1
                yield item

    def build9(out, stream):
        return tokenize_to_shards(stream, fake, out, shard_size=5000, batch_docs=32, max_tokens=120_000)

    ref9 = SHARDS + "_mix_ref"; shutil.rmtree(ref9, ignore_errors=True)
    r9 = build9(ref9, MixedStream(spec8, seed=0))
    share9 = r9["by_source"]["long"] / r9["total_tokens"]
    assert abs(share9 - 0.6) < 0.02, f"writer-fed share {share9:.3f}, wanted 0.60 (is report() wired?)"
    out9 = SHARDS + "_mix_crash"; shutil.rmtree(out9, ignore_errors=True)
    try:
        build9(out9, CrashingStream(spec8, seed=0, crash_after=3000))
        raise AssertionError("the simulated crash did not fire")
    except RuntimeError:
        pass
    prog9 = _json.load(open(os.path.join(out9, "progress.json")))
    assert prog9["pending"] and prog9["stream"].get("rep_tokens"), "checkpoint must carry pending docs + token accounting"
    res9 = build9(out9, MixedStream(spec8, seed=0))
    assert res9["by_source"] == r9["by_source"], (res9["by_source"], r9["by_source"])
    assert shard_bytes(out9) == shard_bytes(ref9), "resumed mixed build differs from the uninterrupted one"
    print(f"  {100*share9:.1f}% long via the writer; crashed at doc 3000 with {len(prog9['pending'])} pending, "
          f"resumed build identical ({r9['total_tokens']} tokens, {len(r9['shards'])} shards) ✓")

    print("\n" + "=" * 72)
    print("ALL CHECKS PASSED ✅")
    print("=" * 72)


if __name__ == "__main__":
    main()
