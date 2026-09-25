"""Turn a weighted, mixed text stream into memmapped uint16 token shards.

- Documents are separated by the tokenizer's <|endoftext|> id.
- uint16 (vocab 49,152 fits in 16 bits) halves shard size vs uint32.
- The mix weights are TOKEN shares, held as it streams (each document is pulled from the source
  furthest behind its share; see MixedStream), and we stop at a token budget, so a capped run
  keeps the full blend (web/textbook/code/math/wiki) rather than filling up on one source.
  Per-source token counts are recorded in meta.json.

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

# Pretraining mix. Each `weight` is the source's share of TOKENS (MixedStream holds the blend in
# tokens as it streams): ~55% web-edu, 22% textbook, 10% code, 8% math, 5% wiki.
# History: the mixer used to pick *documents* by weight, and document length varies 10-100x by
# source (code files and math pages are long; cosmopedia snippets are short), so the 100B shards
# on the Lambda box were built with hand-calibrated doc-weights (0.55/0.32/0.037/0.039/0.048) and
# landed at 54.9/22.7/10.6/8.4/3.3 (data/shards/meta.json). The token-weighted mixer makes that
# calibration unnecessary; these weights now say what they mean. Existing shards are reused as-is.
# Code note: smollm-corpus "python-edu" stores blob_ids (text lives on S3), not inline text, so
# codeparrot-clean (inline `content`, no auth) supplies the code slice instead.
PRETRAIN_MIX = [
    {"path": "HuggingFaceTB/smollm-corpus", "name": "fineweb-edu-dedup", "text_field": "text",    "weight": 0.55},
    {"path": "HuggingFaceTB/smollm-corpus", "name": "cosmopedia-v2",     "text_field": "text",    "weight": 0.22},
    {"path": "codeparrot/codeparrot-clean",                              "text_field": "content", "weight": 0.10},
    {"path": "open-web-math/open-web-math",                              "text_field": "text",    "weight": 0.08},
    {"path": "wikimedia/wikipedia",          "name": "20231101.en",      "text_field": "text",    "weight": 0.05},
]

# Decay-phase annealing mix (the research sweep's biggest remaining lever, docs/research_2026-09-22.md
# section 2.3): high-quality math, code in languages the 100B did NOT contain (its code is Python
# only), and the chat data rendered in the ChatML template the SFT stage uses. Blended into the last
# 15% of pretraining at ~40% of each step (see train.anneal in the config); the other 60% keeps
# reading the original mix, so it is replay of unseen original data, not repetition.
# Code: the-stack-smol-xl (permissively licensed files across ~300 languages; Python excluded so it
# stays disjoint from the pretraining's Python-only code) plus CodeSearchNet functions in five
# non-Python languages. github-code-clean / Stack-Edu would have been better but one is a
# script-based dataset (unsupported by datasets >= 4) and the other ships no file contents.
ANNEAL_MIX = [
    {"path": "HuggingFaceTB/finemath",           "name": "finemath-4plus",    "text_field": "text", "weight": 0.40},
    {"path": "HuggingFaceTB/finemath",           "name": "infiwebmath-4plus", "text_field": "text", "weight": 0.12},
    {"path": "bigcode/the-stack-smol-xl",        "name": None, "label": "stack-smol-xl-nonpython",
     "text_field": "content", "exclude": {"lang": ["Python", "Jupyter Notebook"]},               "weight": 0.20},
    {"path": "code-search-net/code_search_net",  "name": "javascript", "label": "csn-javascript", "text_field": "whole_func_string", "weight": 0.04, "epochs": 2},
    {"path": "code-search-net/code_search_net",  "name": "java",       "label": "csn-java",       "text_field": "whole_func_string", "weight": 0.04, "epochs": 2},
    {"path": "code-search-net/code_search_net",  "name": "go",         "label": "csn-go",         "text_field": "whole_func_string", "weight": 0.03, "epochs": 2},
    {"path": "code-search-net/code_search_net",  "name": "php",        "label": "csn-php",        "text_field": "whole_func_string", "weight": 0.02, "epochs": 2},
    {"path": "code-search-net/code_search_net",  "name": "ruby",       "label": "csn-ruby",       "text_field": "whole_func_string", "weight": 0.02, "epochs": 2},
    {"local": "data/sft.jsonl",                   "name": "chat-sft",   "render": "chatml",        "weight": 0.13, "epochs": 3},
]
# "epochs": the small sources are far smaller than their share of a 6B-token build (the SFT chat is
# 127M tokens = 2.1%, all of CodeSearchNet ~280M = 4.7%), so with one pass they run dry early and the
# big three absorb the difference (v2 build: 52/26/16 math/code/webmath, 2% chat). Repeating them a
# few times costs little (repeated data is near-free up to ~4 epochs) and lifts chat to ~6% and the
# CodeSearchNet languages to ~9%; the remaining shortfall still goes to the big sources by ratio.


def source_name(spec: dict) -> str:
    """The name a source is tagged with in the stream and in meta.json's by_source: label, else the
    HF config name, else the dataset path, else the local file. One definition, used by the mixer
    and by the build script's missing-source check (which once used a different key and cried wolf)."""
    return spec.get("label") or spec.get("name") or spec.get("path") or spec.get("local")


def _row_ok(row, exclude) -> bool:
    """Optional per-source row filter: exclude={"lang": ["Python"]} drops rows whose `lang` is listed."""
    if not exclude or not isinstance(row, dict):
        return True
    return all(row.get(k) not in set(vals) for k, vals in exclude.items())


class LocalJsonlSource:
    """A local jsonl file as a streaming source with the same state_dict()/load_state_dict()
    contract as a HuggingFace IterableDataset (state = line index). `render="chatml"` turns
    {"messages": [...]} rows into the ChatML text the SFT stage trains on; otherwise the row's
    `text_field` is used."""

    def __init__(self, path: str, render: str | None = None, text_field: str = "text"):
        self.path, self.render, self.field = path, render, text_field
        self.i = 0                      # next line to yield
        self._resume = 0

    def _render(self, row: dict):
        if self.render == "chatml":
            from tokenizer.tokenizer import IM_START, IM_END
            msgs = row.get("messages") or []
            return "".join(f"{IM_START}{m['role']}\n{m['content']}{IM_END}\n" for m in msgs
                           if m.get("role") and m.get("content")) or None
        return row.get(self.field)

    def __iter__(self):
        with open(self.path, encoding="utf-8") as f:
            for n, line in enumerate(f):
                if n < self._resume:
                    continue
                self.i = n + 1
                line = line.strip()
                if not line:
                    continue
                try:
                    text = self._render(json.loads(line))
                except Exception:
                    continue
                if text:
                    yield {"text": text}
        self._resume = self.i = 0          # pass finished: the next pass (another epoch) starts at line 0

    def state_dict(self) -> dict:
        return {"line": int(self.i)}

    def load_state_dict(self, s: dict) -> None:
        self._resume = int(s.get("line", 0))
        self.i = self._resume


def _rng_state_to_json(rng: random.Random):
    v, internal, gauss = rng.getstate()
    return [v, list(internal), gauss]


def _rng_state_from_json(j):
    return (j[0], tuple(j[1]), j[2])


class MixedStream:
    """Weighted blend of streaming sources, yielding (text, source_name). The weights are TOKEN
    shares: each document is pulled from the source furthest behind its share of the tokens served
    so far, so the blend holds in tokens throughout (and therefore holds when we stop early at a
    token budget). Documents differ 10-100x in length between sources (a code file vs a chat
    turn), so picking documents by weight, as this used to, over-served the long-document sources
    by that factor (the first anneal build: 46% stack code, 2% chat, against targets of 20 / 13).
    Token counts come back from the shard writer through report(); until a source has been
    reported once its documents are estimated at `chars_per_token`, and the documents in flight
    between a pick and its report (one encode chunk) use the source's measured ratio. When a
    source runs dry it drops out and the rest carry on with their shares renormalized.

    Resumable: state_dict() captures the rng, the per-source token accounting and each source's
    streaming position; load_state_dict() restores them, so an interrupted build resumes exactly
    (completed shards are skipped without re-downloading). It is iterable, so
    `for text, src in MixedStream(...)` works like the old generator did."""

    def __init__(self, specs=PRETRAIN_MIX, seed: int = 0, chars_per_token: float = 4.0):
        self.specs = specs
        self._rng = random.Random(seed)
        self._cpt0 = float(chars_per_token)
        self._sources = None         # opened lazily on first iteration / state call
        self._pending_state = None   # a state handed to load_state_dict before the sources open
        self._chars: dict[str, int] = {}        # characters yielded per source
        self._rep_chars: dict[str, int] = {}    # characters whose token count has been reported
        self._rep_tokens: dict[str, int] = {}   # tokens reported for those characters

    def report(self, source: str, chars: int, tokens: int) -> None:
        """The shard writer's feedback: `chars` characters of `source` encoded to `tokens` tokens."""
        self._rep_chars[source] = self._rep_chars.get(source, 0) + int(chars)
        self._rep_tokens[source] = self._rep_tokens.get(source, 0) + int(tokens)

    def served(self, source: str) -> float:
        """Tokens served from `source` so far: the reported count, plus an estimate for the
        documents in flight (yielded, not yet reported) at the source's measured chars/token."""
        rc, rt = self._rep_chars.get(source, 0), self._rep_tokens.get(source, 0)
        cpt = max(rc / rt, 0.25) if rt > 0 else self._cpt0
        return rt + (self._chars.get(source, 0) - rc) / cpt

    def _pick(self, alive: list) -> dict:
        """The alive source furthest behind its (renormalized) token share; ties broken by the rng."""
        total = sum(self.served(e["name"]) for e in alive)
        wsum = sum(e["w"] for e in alive)
        best, best_d = [], None
        for e in alive:
            d = e["w"] / wsum * total - self.served(e["name"])      # tokens behind its share
            if best_d is None or d > best_d + 1e-9:
                best, best_d = [e], d
            elif abs(d - best_d) <= 1e-9:
                best.append(e)
        return best[0] if len(best) == 1 else self._rng.choice(best)

    def _open(self):
        if self._sources is not None:
            return
        srcs = []
        for s in self.specs:
            name = source_name(s)
            if s.get("local"):                      # a local jsonl (e.g. the SFT chat data) as a source
                ds = LocalJsonlSource(s["local"], render=s.get("render"), text_field=s.get("text_field", "text"))
                srcs.append({"name": name, "ds": ds, "w": float(s["weight"]), "field": "text", "alive": True,
                             "epochs": max(1, int(s.get("epochs", 1))), "epoch": 0})
                continue
            from datasets import load_dataset
            try:
                ds = load_dataset(s["path"], s.get("name"), split=s.get("split", "train"), streaming=True)
            except Exception as e:                  # one bad source must not kill the whole prep
                print(f"[data] WARNING: could not open {name}: {type(e).__name__}: {e}. Skipping it.")
                continue
            srcs.append({"name": name, "ds": ds, "w": float(s["weight"]),
                         "field": s.get("text_field", "text"), "alive": True, "exclude": s.get("exclude"),
                         "epochs": max(1, int(s.get("epochs", 1))), "epoch": 0})
        if not srcs:
            raise RuntimeError("no data sources could be opened (check network / dataset ids)")
        if self._pending_state is not None:
            self._apply_state(srcs, self._pending_state)
            self._pending_state = None
        for e in srcs:
            e["it"] = iter(e["ds"])
        self._sources = srcs

    def _apply_state(self, srcs, state):
        """The per-source part of a saved state (streaming positions, alive flags); needs open sources."""
        saved = {d["name"]: d for d in state["sources"]}
        for e in srcs:
            d = saved.get(e["name"])
            if not d:
                continue
            e["alive"] = d["alive"]
            e["epoch"] = int(d.get("epoch", 0))
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
            e = self._pick(alive)
            try:
                ex = next(e["it"])
            except StopIteration:
                e["epoch"] += 1
                if e["epoch"] < e["epochs"]:              # another pass over a small source
                    print(f"[data] source {e['name']}: pass {e['epoch'] + 1} of {e['epochs']}", flush=True)
                    e["it"] = iter(e["ds"])
                    continue
                print(f"[data] source exhausted: {e['name']} (after {e['epoch']} pass(es))", flush=True)
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
            if not _row_ok(ex, e.get("exclude")):
                continue
            text = ex.get(e["field"]) if isinstance(ex, dict) else None
            if text:
                self._chars[e["name"]] = self._chars.get(e["name"], 0) + len(text)
                yield text, e["name"]

    def state_dict(self) -> dict:
        self._open()
        return {"rng": _rng_state_to_json(self._rng),
                "chars": dict(self._chars), "rep_chars": dict(self._rep_chars), "rep_tokens": dict(self._rep_tokens),
                "sources": [{"name": e["name"], "alive": e["alive"], "epoch": e["epoch"],
                             "ds": (e["ds"].state_dict() if e["alive"] else None)}
                            for e in self._sources]}

    def exhausted(self) -> list[tuple[str, int]]:
        """(name, passes) of every source that ran dry, so a build can report which shares fell short."""
        self._open()
        return [(e["name"], e["epoch"]) for e in self._sources if not e["alive"]]

    def load_state_dict(self, state: dict) -> None:
        # The rng and the token accounting are restored right away: the shard writer replays the
        # documents that were in flight at the crash and report()s them BEFORE it asks the stream
        # for more (which is what opens the sources), and those reports must land on top of the
        # saved counts, not be overwritten by them. Only the source positions wait for the open.
        self._rng.setstate(_rng_state_from_json(state["rng"]))
        self._chars = {k: int(v) for k, v in state.get("chars", {}).items()}
        self._rep_chars = {k: int(v) for k, v in state.get("rep_chars", {}).items()}
        self._rep_tokens = {k: int(v) for k, v in state.get("rep_tokens", {}).items()}
        if self._sources is None:
            self._pending_state = state              # source positions applied when the sources open
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
    reporter = getattr(items, "report", None)          # a MixedStream wants the token counts back
    first = None
    if pending:                                         # re-feed the docs that were in flight at the crash
        if isinstance(pending, dict):                   # single-doc format of older progress files
            pending = [pending]
        # ... as a chunk of their own, so their token counts are reported before the stream is asked
        # for more: that is the order the uninterrupted build saw, so the resumed picks are identical.
        first = [(d["text"], d["source"]) for d in pending]

    def norm(item):
        return item if isinstance(item, tuple) else (item, None)

    docs, done = 0, False
    while not done:
        if first:
            chunk, first = first, None
        else:
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
                if reporter:
                    reporter(source, len(text), n)
                checkpoint(chunk[j + 1:])
                continue
            buf[fill:fill + n] = np.asarray(ids, dtype=DTYPE)
            fill += n
            total += n
            by_source[source] = by_source.get(source, 0) + n
            if reporter:
                reporter(source, len(text), n)
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
