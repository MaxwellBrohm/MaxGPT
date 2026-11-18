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

    print("\n[10] a small source repeats for its epochs, then runs dry; resume inside a pass is exact")
    aj, bj = SHARDS + "_epochA.jsonl", SHARDS + "_epochB.jsonl"
    with open(aj, "w") as f:
        for i in range(5):
            f.write(_json.dumps({"text": f"A{i} " + "a" * 40}) + "\n")
    with open(bj, "w") as f:
        for i in range(40):
            f.write(_json.dumps({"text": f"B{i:02d} " + "b" * 40}) + "\n")
    spec10 = [{"local": aj, "name": "A", "weight": 0.5, "epochs": 3}, {"local": bj, "name": "B", "weight": 0.5}]
    full = [(t, s) for t, s in MixedStream(spec10, seed=0)]
    a_docs = [t[:2] for t, s in full if s == "A"]
    assert a_docs == [f"A{i}" for i in range(5)] * 3, (len(a_docs), a_docs[:8])
    assert sum(1 for _, s in full if s == "B") == 40
    ms = MixedStream(spec10, seed=0)
    it = iter(ms)
    first = [next(it) for _ in range(6)]           # A is 3 docs into its FIRST pass here
    st = ms.state_dict()
    ms2 = MixedStream(spec10, seed=0)
    ms2.load_state_dict(st)
    rest = [(t, s) for t, s in ms2]
    assert first + rest == full, "resume changed the stream across the pass boundaries"
    assert sorted(ms2.exhausted()) == [("A", 3), ("B", 1)], ms2.exhausted()
    ms3 = MixedStream(spec10, seed=0)
    it3 = iter(ms3)
    first3 = [next(it3) for _ in range(14)]        # A is 2 docs into its SECOND pass here
    st3 = ms3.state_dict()
    assert [d for d in st3["sources"] if d["name"] == "A"][0]["epoch"] == 1, st3["sources"]
    ms4 = MixedStream(spec10, seed=0)
    ms4.load_state_dict(st3)
    assert first3 + [(t, s) for t, s in ms4] == full, "resume inside the second pass changed the stream"
    print(f"  A (5 docs, epochs=3) yielded 15 docs in order, B 40; resumed after 6 docs -> identical; "
          f"exhausted: {sorted(ms2.exhausted())} ✓")

    print("\n[11] anneal_check rejects a build whose small sources ran dry, accepts a good one")
    from scripts.anneal_check import check as anneal_check
    def fake_meta(d, chat, csn, total=6_000_000_000, shards=61):
        os.makedirs(d, exist_ok=True)
        rest = 1.0 - chat - csn
        bs = {"finemath-4plus": rest * 0.55, "stack-smol-xl-nonpython": rest * 0.28, "infiwebmath-4plus": rest * 0.17,
              "chat-sft": chat, "csn-java": csn * 0.5, "csn-php": csn * 0.5}
        _json.dump({"total_tokens": total, "shards": [{"name": f"s{i}", "tokens": 1} for i in range(shards)],
                    "by_source": {k: int(v * total) for k, v in bs.items()}}, open(os.path.join(d, "meta.json"), "w"))
    v2 = SHARDS + "_meta_v2"; fake_meta(v2, chat=0.021, csn=0.047)          # the real v2 numbers
    ok, line = anneal_check(v2, 5.9e9, 4.0, 6.0)
    assert not ok and "chat 2.1%" in line and "csn 4.7%" in line, line
    v3 = SHARDS + "_meta_v3"; fake_meta(v3, chat=0.063, csn=0.093)
    ok, line = anneal_check(v3, 5.9e9, 4.0, 6.0)
    assert ok and line.startswith("OK"), line
    short = SHARDS + "_meta_short"; fake_meta(short, chat=0.063, csn=0.093, total=4_000_000_000)
    assert not anneal_check(short, 5.9e9, 4.0, 6.0)[0], "a build under the token budget must be rejected"
    assert not anneal_check(SHARDS + "_no_such_dir", 5.9e9, 4.0, 6.0)[0]
    print("  v2-like shares rejected (chat 2.1%, csn 4.7%); v3-like accepted; short budget + missing meta rejected ✓")

    print("\n[12] files read in turn + score filter + local sources root: mixed languages, low scores dropped, exact resume")
    import data.prepare as DP
    root12 = SHARDS + "_srcroot"
    shutil.rmtree(root12, ignore_errors=True)
    d12 = os.path.join(root12, "fake", "code", "default"); os.makedirs(d12)
    layout = {"part-0.jsonl": [("A0", 3), ("A1", 2), ("A2", 4)],          # one "language" per file, like the real set
              "part-1.jsonl": [("B0", 2), ("B1", 3)],
              "part-2.jsonl": [("C0", 3), ("C1", 3), ("C2", 3), ("C3", 2)]}
    for fn, rows in layout.items():
        with open(os.path.join(d12, fn), "w") as f:
            for name, score in rows:
                f.write(_json.dumps({"text": name + " body", "int_score": score}) + "\n")
    spec12 = [{"path": "fake/code", "label": "code", "interleave_files": "*.jsonl", "min": {"int_score": 3}, "weight": 1.0}]
    ms12 = MixedStream(spec12, seed=0, sources_root=root12, strict=True)
    got12 = [t.split()[0] for t, _ in ms12]
    # raw turn order: A0 B0 C0 A1 B1 C1 A2 (B dry) C2 (A dry) C3; rows scored 2 (B0, A1, C3) are dropped
    assert got12 == ["A0", "C0", "B1", "C1", "A2", "C2"], got12
    assert ms12.positions() == {"code": 9}, ms12.positions()           # rows DRAWN, filtered ones included
    ms12b = MixedStream(spec12, seed=0, sources_root=root12, strict=True)
    it12 = iter(ms12b)
    head12 = [next(it12)[0].split()[0] for _ in range(3)]
    st12 = ms12b.state_dict()
    ms12c = MixedStream(spec12, seed=0, sources_root=root12, strict=True)
    ms12c.load_state_dict(st12)
    assert head12 + [t.split()[0] for t, _ in ms12c] == got12, "resume inside the interleaved stream changed it"
    real_open = DP.open_hf_source
    DP.open_hf_source = lambda spec, root=None: (_ for _ in ()).throw(ConnectionError("no network in this test"))
    try:
        bad_spec = [{"path": "fake/missing", "weight": 0.5}, {"local": cj, "name": "chat", "render": "chatml", "weight": 0.5}]
        try:
            list(MixedStream(bad_spec, seed=0, strict=True))
            raise AssertionError("strict mode must refuse to build without a source")
        except RuntimeError as e:
            assert "fake/missing" in str(e), e
        assert len(list(MixedStream(bad_spec, seed=0, strict=False))) == 30, "non-strict mode skips the source and carries on"
    finally:
        DP.open_hf_source = real_open
    print(f"  order {got12}; 9 rows drawn; resume after 3 identical; strict refuses a missing source ✓")

    print("\n[13] a later build continues each stream after the rows the first build drew (no document twice)")
    wj = SHARDS + "_wiki.jsonl"
    with open(wj, "w") as f:
        for i in range(12):
            f.write(_json.dumps({"text": f"W{i:02d} wiki body"}) + "\n")
    main13 = [{"local": wj, "name": "wiki", "weight": 0.5},
              {"path": "fake/code", "label": "code", "interleave_files": "*.jsonl", "weight": 0.5}]

    class Recording(MixedStream):                    # remembers every document it hands out
        def __init__(self, *a, **k):
            super().__init__(*a, **k); self.seen = []

        def __iter__(self):
            for item in super().__iter__():
                self.seen.append(item[0].split()[0])
                yield item

    out13 = SHARDS + "_cont_main"; shutil.rmtree(out13, ignore_errors=True)
    rec = Recording(main13, seed=0, sources_root=root12, strict=True)
    m13 = tokenize_to_shards(rec, fake, out13, shard_size=5000, batch_docs=2, max_tokens=14)
    assert 0 < len(rec.seen) < 21, f"the first build must stop partway ({len(rec.seen)} of 21 docs drawn)"
    assert set(m13["rows_by_source"]) == {"wiki", "code"} and sum(m13["rows_by_source"].values()) == len(rec.seen), m13["rows_by_source"]
    later = DP.apply_continuation([{"local": wj, "name": "wiki2", "after": "wiki", "weight": 0.5},
                                   {"path": "fake/code", "label": "code2", "interleave_files": "*.jsonl", "after": "code", "weight": 0.5}],
                                  m13["rows_by_source"])
    rest13 = [t.split()[0] for t, _ in MixedStream(later, seed=1, sources_root=root12, strict=True)]
    everything = {f"W{i:02d}" for i in range(12)} | {n for rows in layout.values() for n, _ in rows}
    assert not set(rest13) & set(rec.seen), f"repeated documents: {sorted(set(rest13) & set(rec.seen))}"
    assert set(rest13) | set(rec.seen) == everything and len(rest13) + len(rec.seen) == 21, "a document was lost or doubled"
    assert DP.apply_continuation([{"local": wj, "name": "x", "after": "nope", "weight": 1.0}], m13["rows_by_source"])[0].get("skip") is None
    # the row counts survive a crash and resume of the first build (they ride in progress.json)
    ref13, crash13 = SHARDS + "_cont_ref", SHARDS + "_cont_crash"
    shutil.rmtree(ref13, ignore_errors=True); shutil.rmtree(crash13, ignore_errors=True)
    mk13 = lambda cls=MixedStream, **k: cls(main13, seed=0, sources_root=root12, strict=True, **k)
    whole = tokenize_to_shards(mk13(), fake, ref13, shard_size=6, batch_docs=2)
    try:
        tokenize_to_shards(mk13(CrashingStream, crash_after=11), fake, crash13, shard_size=6, batch_docs=2)
        raise AssertionError("the simulated crash did not fire")
    except RuntimeError:
        pass
    assert os.path.exists(os.path.join(crash13, "progress.json")), "the crashed build should have checkpointed"
    again = tokenize_to_shards(mk13(), fake, crash13, shard_size=6, batch_docs=2)
    assert again["rows_by_source"] == whole["rows_by_source"] == {"wiki": 12, "code": 9}, (again["rows_by_source"], whole["rows_by_source"])
    print(f"  first build drew {len(rec.seen)} docs ({m13['rows_by_source']}); the continuation read the other {len(rest13)}, none twice ✓")

    print("\n[14] download planning: enough files for a source's share, everything for interleaved sources")
    from scripts.fetch_sources import choose_files, need_bytes
    files14 = [("a", 40), ("b", 40), ("c", 40), ("d", 40)]
    assert choose_files(files14, 1) == files14[:1] and choose_files(files14, 80) == files14[:2]
    assert choose_files(files14, 81) == files14[:3] and choose_files(files14, 10**9) == files14
    assert choose_files(files14, None) == files14 and choose_files([], 50) == []
    assert need_bytes({"weight": 0.25}, 1000, bytes_per_token=2.0, margin=1.5) == 750.0
    assert need_bytes({"weight": 0.25, "bytes_per_token": 4.0}, 1000, bytes_per_token=2.0, margin=1.5) == 1500.0
    assert need_bytes({"weight": 0.1, "interleave_files": "*.gz"}, 1000) is None and need_bytes({"weight": 0.1, "fetch": "all"}, 1000) is None
    for mix_name, mix in DP.MIXES.items():
        assert abs(sum(x["weight"] for x in mix) - 1.0) < 1e-9, mix_name
        labels = [DP.source_name(x) for x in mix]
        assert len(labels) == len(set(labels)), f"{mix_name}: two sources share a label {labels}"
    pre_labels = {DP.source_name(x) for x in DP.PRETRAIN_MIX}
    assert all(x["after"] in pre_labels for x in DP.ANNEAL_MIX if x.get("after")), "an anneal source continues a label the pretrain mix does not have"
    print("  file prefixes, per-source allowances, mix weights sum to 1, labels unique, anneal continues real labels ✓")

    print("\n" + "=" * 72)
    print("ALL CHECKS PASSED ✅")
    print("=" * 72)


if __name__ == "__main__":
    main()
