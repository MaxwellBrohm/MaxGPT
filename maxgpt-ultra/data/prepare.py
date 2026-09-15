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

import itertools
import json
import os
import random
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


def _rng_state_to_json(rng: random.Random):
    v, internal, gauss = rng.getstate()
    return [v, list(internal), gauss]


def _rng_state_from_json(j):
    return (j[0], tuple(j[1]), j[2])


class MixedStream:
    """Weighted blend of streaming sources, yielding (text, source_name). Picks each document
    from a source with probability proportional to its weight, so the mix holds throughout (and
    therefore holds when we stop early at a token budget). When a source runs dry it drops out and
    the rest carry on.

    Resumable: state_dict() captures the rng + each source's streaming position, load_state_dict()
    restores them, so an interrupted build resumes exactly (completed shards are skipped without
    re-downloading). It is iterable, so `for text, src in MixedStream(...)` works like the old
    generator did."""

    def __init__(self, specs=PRETRAIN_MIX, seed: int = 0):
        self.specs = specs
        self._rng = random.Random(seed)
        self._sources = None         # opened lazily on first iteration / state call
        self._pending_state = None   # a state handed to load_state_dict before the sources open

    def _open(self):
        if self._sources is not None:
            return
        from datasets import load_dataset
        srcs = []
        for s in self.specs:
            name = s.get("name") or s["path"]
            try:
                ds = load_dataset(s["path"], s.get("name"), split=s.get("split", "train"), streaming=True)
            except Exception as e:                  # one bad source must not kill the whole prep
                print(f"[data] WARNING: could not open {name}: {type(e).__name__}: {e}. Skipping it.")
                continue
            srcs.append({"name": name, "ds": ds, "w": float(s["weight"]),
                         "field": s.get("text_field", "text"), "alive": True})
        if not srcs:
            raise RuntimeError("no data sources could be opened (check network / dataset ids)")
        if self._pending_state is not None:
            self._apply_state(srcs, self._pending_state)
            self._pending_state = None
        for e in srcs:
            e["it"] = iter(e["ds"])
        self._sources = srcs

    def _apply_state(self, srcs, state):
        self._rng.setstate(_rng_state_from_json(state["rng"]))
        saved = {d["name"]: d for d in state["sources"]}
        for e in srcs:
            d = saved.get(e["name"])
            if not d:
                continue
            e["alive"] = d["alive"]
            if d.get("ds") is not None:
                try:
                    e["ds"].load_state_dict(d["ds"])
                except Exception as ex:
                    print(f"[data] WARNING: could not resume source {e['name']}: {ex}")

    def __iter__(self):
        self._open()
        src = self._sources
        fails = {e["name"]: 0 for e in src}            # consecutive stream errors per source
        while any(e["alive"] for e in src):
            alive = [e for e in src if e["alive"]]
            e = self._rng.choices(alive, weights=[x["w"] for x in alive], k=1)[0]
            try:
                ex = next(e["it"])
            except StopIteration:
                print(f"[data] source exhausted: {e['name']}", flush=True)
                e["alive"] = False
                continue
            except Exception as err:                   # a network / HF error must not kill the whole build
                fails[e["name"]] += 1
                print(f"[data] WARNING: {e['name']} stream error "
                      f"({type(err).__name__}: {err}); fail {fails[e['name']]}/10", flush=True)
                if fails[e["name"]] >= 10:
                    print(f"[data] dropping {e['name']} after repeated errors; continuing on the rest", flush=True)
                    e["alive"] = False
                continue
            fails[e["name"]] = 0
            text = ex.get(e["field"]) if isinstance(ex, dict) else None
            if text:
                yield text, e["name"]

    def state_dict(self) -> dict:
        self._open()
        return {"rng": _rng_state_to_json(self._rng),
                "sources": [{"name": e["name"], "alive": e["alive"],
                             "ds": (e["ds"].state_dict() if e["alive"] else None)}
                            for e in self._sources]}

    def load_state_dict(self, state: dict) -> None:
        if self._sources is None:
            self._pending_state = state              # applied when the sources open
        else:
            self._apply_state(self._sources, state)
            for e in self._sources:
                e["it"] = iter(e["ds"])


def stream_mixed(specs=PRETRAIN_MIX, seed: int = 0) -> MixedStream:
    """Weighted, streamed, resumable blend of the sources. Returns a MixedStream (iterable), so
    existing callers (`for text, src in stream_mixed(...)`) keep working unchanged."""
    return MixedStream(specs, seed)


def tokenize_to_shards(items, tokenizer, out_dir: str, shard_size: int = 100_000_000,
                       eot_id: int | None = None, max_tokens: int | None = None,
                       resume: bool = True, on_progress=None, batch_docs: int = 256) -> dict:
    """Encode `items` (each a str, or a (text, source) tuple), append an EOT after each doc, and
    write uint16 shards + a meta.json index. Stops at `max_tokens` if set.

    Resumable: if `items` supports state_dict()/load_state_dict() (a MixedStream), progress is
    checkpointed to progress.json at every shard boundary, so an interrupted multi-day build (crash,
    reboot, GUI restart) picks up where it stopped instead of re-downloading from zero. Shards are
    cut on document boundaries so each checkpoint is a clean point with nothing half-written. If
    meta.json already exists the build is finished and is returned as-is.

    Docs are pulled `batch_docs` at a time and encoded in one tokenizer call (multi-core); the
    per-doc shard logic is unchanged, and a checkpoint stores every pulled-but-unwritten doc of
    the current chunk as `pending`, so a resume never skips or repeats a document."""
    os.makedirs(out_dir, exist_ok=True)
    if eot_id is None:
        eot_id = tokenizer.eos_id
    meta_path = os.path.join(out_dir, "meta.json")
    prog_path = os.path.join(out_dir, "progress.json")

    if os.path.exists(meta_path):                       # already built -> reuse, don't redo
        with open(meta_path, encoding="utf-8") as f:
            return json.load(f)

    resumable = resume and hasattr(items, "state_dict") and hasattr(items, "load_state_dict")
    shards: list[dict] = []
    by_source: dict[str, int] = {}
    total, idx = 0, 0
    pending = None
    if resumable and os.path.exists(prog_path):
        try:
            with open(prog_path, encoding="utf-8") as f:
                prog = json.load(f)
            items.load_state_dict(prog["stream"])
            shards = prog["shards"]
            by_source = {k: int(v) for k, v in prog["by_source"].items()}
            total, idx, pending = int(prog["total"]), int(prog["idx"]), prog.get("pending")
            print(f"[data] resuming shard build from shard #{idx} ({total:,} tokens already written)")
        except Exception as e:
            print(f"[data] WARNING: could not resume ({type(e).__name__}: {e}); rebuilding from scratch")
            shards, by_source, total, idx, pending = [], {}, 0, 0, None

    buf = np.empty(shard_size, dtype=DTYPE)
    fill = 0

    def write_shard(arr) -> None:
        nonlocal idx
        name = f"shard_{idx:05d}.bin"
        np.asarray(arr, dtype=DTYPE).tofile(os.path.join(out_dir, name))
        shards.append({"name": name, "tokens": int(len(arr))})
        idx += 1

    def checkpoint(pending_items) -> None:
        """pending_items: the docs already pulled from the stream but not yet written (in order)."""
        if not resumable:
            return
        rec = {"idx": idx, "total": int(total), "shards": shards,
               "by_source": {str(k): int(v) for k, v in by_source.items()},
               "stream": items.state_dict(),
               "pending": [{"text": t, "source": src} for t, src in pending_items]}
        tmp = prog_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(rec, f)
        os.replace(tmp, prog_path)                      # atomic: a crash mid-write can't corrupt it

    src_iter = iter(items)
    if pending:                                         # re-feed the docs that were in flight at the crash
        if isinstance(pending, dict):                   # single-doc format of older progress files
            pending = [pending]
        src_iter = itertools.chain([(d["text"], d["source"]) for d in pending], src_iter)

    def norm(item):
        return item if isinstance(item, tuple) else (item, None)

    docs, done = 0, False
    while not done:
        chunk = [norm(it) for it in itertools.islice(src_iter, batch_docs)]
        if not chunk:
            break
        encoded = (tokenizer.encode_batch([t for t, _ in chunk]) if hasattr(tokenizer, "encode_batch")
                   else [tokenizer.encode(t) for t, _ in chunk])
        for j, ((text, source), ids) in enumerate(zip(chunk, encoded)):
            if on_progress:                             # report tokens-done (the callback throttles to ~5s)
                docs += 1
                if docs % 256 == 0:
                    on_progress(total)
            ids.append(eot_id)
            n = len(ids)
            if fill + n > shard_size and fill > 0:      # close the current shard on a doc boundary
                write_shard(buf[:fill])
                fill = 0
                checkpoint(chunk[j:])                   # shards done; this doc + the rest of the chunk pending
            if n > shard_size:                          # rare: one document exceeds a whole shard
                write_shard(np.asarray(ids, dtype=DTYPE))
                total += n
                by_source[source] = by_source.get(source, 0) + n
                checkpoint(chunk[j + 1:])
                continue
            buf[fill:fill + n] = np.asarray(ids, dtype=DTYPE)
            fill += n
            total += n
            by_source[source] = by_source.get(source, 0) + n
            if max_tokens and total >= max_tokens:
                done = True
                break
    if fill > 0:
        write_shard(buf[:fill])

    meta = {"dtype": "uint16", "shard_size": int(shard_size), "total_tokens": int(total),
            "eot_id": int(eot_id), "shards": shards,
            "by_source": {str(k): int(v) for k, v in by_source.items()}}
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    if os.path.exists(prog_path):                       # finished cleanly -> drop the checkpoint
        os.remove(prog_path)
    return meta
