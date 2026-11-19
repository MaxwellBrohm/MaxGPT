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

import fnmatch
import glob
import itertools
import json
import os
import random
from typing import Iterable

import numpy as np

DTYPE = np.uint16

# Pretraining mix, version 2: the restart on NAIRR H100s (docs/research_2026-11-18_restart.md, section 3).
# Each `weight` is the source's share of TOKENS (MixedStream holds the blend in tokens as it streams).
# What changed from the September mix (kept below as PRETRAIN_MIX_2026_09) and why:
#   - Web: FineWeb-Edu alone -> 60/40 FineWeb-Edu / DCLM-baseline. The two cover different skills
#     (FW-Edu wins MMLU and ARC, DCLM wins HellaSwag and CommonsenseQA) in SmolLM2's 1.7B ablation
#     and, with 3 seeds, in AI2's DataDecide 1B models trained on exactly 100B tokens.
#   - Cosmopedia v2: 22% -> 5%. Its own makers measured it well below plain web text at 1.2B
#     (10.33 vs 13.77 on a 12-task average) and used it at 4% in SmolLM2.
#   - Code: codeparrot (Python only) -> Common Pile's education-filtered Stack v2 (15 languages,
#     openly licensed, file contents included). Its files hold one language after another, so
#     they are read interleaved (`interleave_files`), and rows the education classifier scored
#     below 3 are dropped (`min`), the threshold SmolLM2's Stack-Edu used.
#   - Math: OpenWebMath -> FineMath 3+ and InfiWebMath 3+ (the filtered sets SmolLM2 moved to).
#   - Wikipedia: the 2023 dump -> FineWiki (cleaned); only its `text` column is read.
# Source spec keys: path / name (HF dataset id and config) or local (a jsonl file); text_field;
# weight (token share); label (name used in meta.json, defaults to name or path); columns (parquet
# columns to read); interleave_files (glob: read the repo's files in turn, one row each);
# min / exclude (row filters); epochs (passes over a small source); render ("chatml");
# after (label of a pretraining source whose stream this one continues, see apply_continuation);
# fetch ("all": scripts/fetch_sources.py downloads the whole source, not just its share).
PRETRAIN_MIX = [
    {"path": "HuggingFaceTB/smollm-corpus", "name": "fineweb-edu-dedup", "text_field": "text", "weight": 0.42},
    {"path": "mlfoundations/dclm-baseline-1.0-parquet", "label": "dclm-baseline", "text_field": "text", "weight": 0.28},
    {"path": "HuggingFaceTB/smollm-corpus", "name": "cosmopedia-v2", "text_field": "text", "weight": 0.05,
     "bytes_per_token": 6.0},      # its parquet carries prompt columns: about 4.4 compressed bytes per token
    {"path": "common-pile/stackv2_edu_filtered", "label": "stack-edu-code", "text_field": "text",
     "interleave_files": "*.json.gz", "min": {"int_score": 3}, "weight": 0.12},
    {"path": "HuggingFaceTB/finemath", "name": "finemath-3plus", "text_field": "text", "weight": 0.04},
    {"path": "HuggingFaceTB/finemath", "name": "infiwebmath-3plus", "text_field": "text", "weight": 0.04},
    {"path": "HuggingFaceFW/finewiki", "name": "en", "label": "finewiki-en", "text_field": "text",
     "columns": ["text"], "fetch": "all", "weight": 0.05},
]

# The same mix with NVIDIA's Nemotron-CC-v2 taking 30% of the tokens out of the web share. Best
# independent result found (LAION / Juelich, 1.7B on 1T tokens: average 0.66 vs 0.57 DCLM and 0.56
# FW-Edu), but the repository is GATED (manual approval, NVIDIA data agreement), so its config
# names and text field below are UNVERIFIED: run scripts/probe_sources.py --mix pretrain-nemotron
# once access exists, and fix them here before building.
PRETRAIN_MIX_NEMOTRON = [
    {"path": "nvidia/Nemotron-CC-v2", "name": "High-Quality", "label": "nemotron-hq", "text_field": "text", "weight": 0.16},
    {"path": "nvidia/Nemotron-CC-v2", "name": "High-Quality-Synthetic", "label": "nemotron-hq-synth", "text_field": "text", "weight": 0.10},
    {"path": "nvidia/Nemotron-CC-v2", "name": "Diverse-QA", "label": "nemotron-diverse-qa", "text_field": "text", "weight": 0.04},
    {"path": "HuggingFaceTB/smollm-corpus", "name": "fineweb-edu-dedup", "text_field": "text", "weight": 0.28},
    {"path": "mlfoundations/dclm-baseline-1.0-parquet", "label": "dclm-baseline", "text_field": "text", "weight": 0.14},
    {"path": "HuggingFaceTB/smollm-corpus", "name": "cosmopedia-v2", "text_field": "text", "weight": 0.03,
     "bytes_per_token": 6.0},
    {"path": "common-pile/stackv2_edu_filtered", "label": "stack-edu-code", "text_field": "text",
     "interleave_files": "*.json.gz", "min": {"int_score": 3}, "weight": 0.12},
    {"path": "HuggingFaceTB/finemath", "name": "finemath-3plus", "text_field": "text", "weight": 0.04},
    {"path": "HuggingFaceTB/finemath", "name": "infiwebmath-3plus", "text_field": "text", "weight": 0.04},
    {"path": "HuggingFaceFW/finewiki", "name": "en", "label": "finewiki-en", "text_field": "text",
     "columns": ["text"], "fetch": "all", "weight": 0.05},
]

# The mix of the September 2026 run (the 4x TITAN RTX attempt). Kept so the proxy A/B can measure
# the new mix against it. ~55% web-edu, 22% textbook, 10% Python code, 8% math, 5% wiki.
PRETRAIN_MIX_2026_09 = [
    {"path": "HuggingFaceTB/smollm-corpus", "name": "fineweb-edu-dedup", "text_field": "text",    "weight": 0.55},
    {"path": "HuggingFaceTB/smollm-corpus", "name": "cosmopedia-v2",     "text_field": "text",    "weight": 0.22},
    {"path": "codeparrot/codeparrot-clean",                              "text_field": "content", "weight": 0.10},
    {"path": "open-web-math/open-web-math",                              "text_field": "text",    "weight": 0.08},
    {"path": "wikimedia/wikipedia",          "name": "20231101.en",      "text_field": "text",    "weight": 0.05},
]

# Decay-phase annealing mix: blended into the last 20% of pretraining at ~40% of each step (see
# train.anneal in the config); the other 60% keeps reading the stable mix. Higher-quality math
# (the 4+ sets), more code, more Wikipedia, and chat rendered in the ChatML template the SFT stage
# uses. Code and Wikipedia CONTINUE the pretraining streams (`after`): the build skips the rows the
# pretraining build consumed, so the decay phase reads documents the model has not seen. The chat
# file is small next to its share, so it is repeated (`epochs`).
ANNEAL_MIX = [
    {"path": "HuggingFaceTB/finemath", "name": "finemath-4plus",    "text_field": "text", "weight": 0.27},
    {"path": "HuggingFaceTB/finemath", "name": "infiwebmath-4plus", "text_field": "text", "weight": 0.09},
    {"path": "common-pile/stackv2_edu_filtered", "label": "stack-edu-code", "text_field": "text",
     "interleave_files": "*.json.gz", "min": {"int_score": 3}, "after": "stack-edu-code", "weight": 0.38},
    {"path": "HuggingFaceFW/finewiki", "name": "en", "label": "finewiki-en", "text_field": "text",
     "columns": ["text"], "fetch": "all", "after": "finewiki-en", "weight": 0.14},
    {"local": "data/sft.jsonl", "name": "chat-sft", "render": "chatml", "weight": 0.12, "epochs": 4},
]

MIXES = {"pretrain": PRETRAIN_MIX, "pretrain-nemotron": PRETRAIN_MIX_NEMOTRON,
         "pretrain-2026-09": PRETRAIN_MIX_2026_09, "anneal": ANNEAL_MIX}


def apply_continuation(specs: list, rows_by_source: dict) -> list:
    """Copies of `specs` where every source with `after: <label>` starts after the rows the earlier
    build drew from that label (meta.json's rows_by_source). A source whose label is missing there
    is left to start from the beginning, and says so."""
    out = []
    for s in specs:
        s = dict(s)
        if s.get("after"):
            if s["after"] in rows_by_source:
                s["skip"] = int(rows_by_source[s["after"]])
            else:
                print(f"[data] WARNING: {source_name(s)} continues '{s['after']}' but the earlier build has no "
                      f"such source; it will start from the beginning (repeated documents)", flush=True)
        out.append(s)
    return out


def source_name(spec: dict) -> str:
    """The name a source is tagged with in the stream and in meta.json's by_source: label, else the
    HF config name, else the dataset path, else the local file. One definition, used by the mixer
    and by the build script's missing-source check (which once used a different key and cried wolf)."""
    return spec.get("label") or spec.get("name") or spec.get("path") or spec.get("local")


def _row_ok(row, exclude=None, minimum=None) -> bool:
    """Optional per-source row filters: exclude={"lang": ["Python"]} drops rows whose `lang` is
    listed; minimum={"int_score": 3} drops rows whose `int_score` is missing or below 3."""
    if not isinstance(row, dict):
        return True
    if exclude and not all(row.get(k) not in set(vals) for k, vals in exclude.items()):
        return False
    if minimum:
        for k, v in minimum.items():
            x = row.get(k)
            if x is None or x < v:
                return False
    return True


class LocalJsonlSource:
    """A local jsonl file as a streaming source with the same state_dict()/load_state_dict()
    contract as a HuggingFace IterableDataset (state = line index). `render="chatml"` turns
    {"messages": [...]} rows into the ChatML text the SFT stage trains on; otherwise the row's
    `text_field` is used."""

    def __init__(self, path: str, render: str | None = None, text_field: str = "text", start: int = 0):
        self.path, self.render, self.field = path, render, text_field
        self.i = int(start)             # next line to yield (start > 0: continue after an earlier build)
        self._resume = int(start)
        self.end = None                 # line count, known once a pass has reached the end of the file

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
        self.end = self.i                  # the whole file has been read at least once
        self._resume = self.i = 0          # pass finished: the next pass (another epoch) starts at line 0

    def state_dict(self) -> dict:
        return {"line": int(self.i)}

    def load_state_dict(self, s: dict) -> None:
        self._resume = int(s.get("line", 0))
        self.i = self._resume


class RoundRobinSource:
    """Several streams read one row at a time in turn, so a dataset stored one language (or one
    topic) per file comes out mixed instead of in long single-language runs. `parts` are streaming
    datasets (anything iterable with state_dict() / load_state_dict()). A part that runs dry drops
    out; the stream ends when all have (a single pass: it does not restart). The first `skip` rows
    are drawn and discarded (used to continue after the rows an earlier build consumed).
    Resumable: the state is whose turn it is, which parts are alive, and each part's position."""

    def __init__(self, parts: list, skip: int = 0):
        self.parts = list(parts)
        self.alive = [True] * len(self.parts)
        self.turn = 0
        self.to_skip = int(skip)

    def __iter__(self):
        its = [iter(p) if a else None for p, a in zip(self.parts, self.alive)]
        n = len(self.parts)
        while any(self.alive):
            i = self.turn % n
            self.turn = (i + 1) % n                 # advance first: a saved state names the NEXT part
            if not self.alive[i]:
                continue
            try:
                row = next(its[i])
            except StopIteration:
                self.alive[i] = False
                continue
            if self.to_skip > 0:
                self.to_skip -= 1
                continue
            yield row

    def state_dict(self) -> dict:
        return {"turn": int(self.turn), "alive": list(self.alive), "to_skip": int(self.to_skip),
                "parts": [p.state_dict() if a else None for p, a in zip(self.parts, self.alive)]}

    def load_state_dict(self, s: dict) -> None:
        self.turn, self.to_skip = int(s["turn"]), int(s.get("to_skip", 0))
        self.alive = [bool(a) for a in s["alive"]]
        for p, st in zip(self.parts, s["parts"]):
            if st is not None:
                p.load_state_dict(st)


_DATA_EXTS = ("parquet", "json.gz", "jsonl.gz", "jsonl", "json")


def local_source_dir(spec: dict, root: str) -> str:
    """Where scripts/fetch_sources.py puts a source's files: <root>/<dataset id>/<config or 'default'>."""
    return os.path.join(root, spec["path"], spec.get("name") or "default")


FETCH_RECORD = ".maxgpt_fetch.json"      # left in a source's folder by scripts/fetch_sources.py when its download finished


def read_fetch_record(spec: dict, root: str | None) -> dict | None:
    """The record of this source's finished download, or None if there is none."""
    if not root or not spec.get("path"):
        return None
    p = os.path.join(local_source_dir(spec, root), FETCH_RECORD)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def write_fetch_record(spec: dict, root: str, files: list, revision: str | None = None,
                       source_tokens: float | None = None) -> dict:
    """Record a finished download: the files (path inside the source's folder, bytes on disk) and
    the number of tokens of this source it was sized for (None: the whole source was taken).

    A download takes a prefix of the source's file list, so a later, smaller download of the same
    revision never shrinks the record; a different revision replaces it."""
    rec = {"repo": spec["path"], "config": spec.get("name"), "revision": revision,
           "source_tokens": None if source_tokens is None else float(source_tokens),
           "files": [[str(p), int(s)] for p, s in files]}
    old = read_fetch_record(spec, root)
    if old and old.get("revision") == revision:
        if len(old.get("files") or []) > len(rec["files"]):
            rec["files"] = old["files"]
        a, b = old.get("source_tokens", 0.0), rec["source_tokens"]
        rec["source_tokens"] = None if a is None or b is None else max(float(a), float(b))
    d = local_source_dir(spec, root)
    os.makedirs(d, exist_ok=True)
    tmp = os.path.join(d, FETCH_RECORD + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(rec, f)
    os.replace(tmp, os.path.join(d, FETCH_RECORD))
    return rec


def short_downloads(specs: list, root: str | None, total_tokens: float) -> list[str]:
    """Sources whose local download was sized for fewer tokens than a build of `total_tokens` takes
    from them, so that build would run them dry. Whole-source downloads and sources streamed from
    the hub are never listed."""
    out = []
    for s in specs:
        rec = read_fetch_record(s, root)
        if not rec or rec.get("source_tokens") is None:
            continue
        need = float(s["weight"]) * float(total_tokens)
        if float(rec["source_tokens"]) < 0.999 * need:
            out.append(f"{source_name(s)}: downloaded for {float(rec['source_tokens']):.3g} tokens, this build takes {need:.3g}")
    return out


def _local_files(spec: dict, root: str | None, strict: bool = False) -> list[str]:
    """This source's data files under the local sources root, or [] if it was not fetched there.

    With a download record, exactly the recorded files are read, after checking that each is present
    at its recorded size: a folder that lost files or holds half of one is an error, never a silently
    shorter source. Data files with no record (a download that did not finish, or files placed by
    hand) are an error when `strict` and are read as found otherwise."""
    if not root or not spec.get("path"):
        return []
    d = local_source_dir(spec, root)
    rec = read_fetch_record(spec, root)
    if rec is not None:
        bad = []
        for p, size in rec["files"]:
            fp = os.path.join(d, p)
            if not os.path.isfile(fp):
                bad.append(f"{p} (missing)")
            elif os.path.getsize(fp) != int(size):
                bad.append(f"{p} ({os.path.getsize(fp):,} bytes, {int(size):,} recorded)")
        if bad:
            raise FileNotFoundError(f"{len(bad)} of the {len(rec['files'])} downloaded files of {source_name(spec)} are missing "
                                    f"or changed under {d}, first: {bad[0]}. Run scripts/fetch_sources.py again.")
        return sorted(os.path.join(d, p) for p, _ in rec["files"])
    files = []
    for ext in _DATA_EXTS:
        files += glob.glob(os.path.join(d, "**", f"*.{ext}"), recursive=True)
    files = sorted(set(files))
    if files and strict:
        raise FileNotFoundError(f"{d} holds {len(files)} data file(s) but no download record ({FETCH_RECORD}): the download "
                                f"did not finish. Run scripts/fetch_sources.py again.")
    return files


def _builder_for(path: str) -> str:
    return "parquet" if path.endswith(".parquet") else "json"


def open_hf_source(spec: dict, root: str | None = None, strict: bool = False):
    """Open one HuggingFace source as a streaming dataset: from the local sources root when its
    files were fetched there (the cluster path: download first, tokenize from disk), else straight
    from the hub. Honors `columns`, `interleave_files` and `skip`. A local folder is checked against
    its download record first (see _local_files); `strict` also refuses a folder that has none."""
    from datasets import load_dataset
    kw = {"columns": list(spec["columns"])} if spec.get("columns") else {}
    files = _local_files(spec, root, strict=strict)
    pattern = spec.get("interleave_files")
    skip = int(spec.get("skip", 0) or 0)
    if pattern:
        if not files:                               # list the repo's files on the hub
            from huggingface_hub import HfApi
            names = sorted(f for f in HfApi().list_repo_files(spec["path"], repo_type="dataset")
                           if fnmatch.fnmatch(os.path.basename(f), pattern))
            files = [f"hf://datasets/{spec['path']}/{f}" for f in names]
        if not files:
            raise FileNotFoundError(f"no files matching {pattern!r} for {spec['path']}")
        parts = [load_dataset(_builder_for(f), data_files=f, split="train", streaming=True, **kw) for f in files]
        return RoundRobinSource(parts, skip=skip)
    if files:
        ds = load_dataset(_builder_for(files[0]), data_files=files, split="train", streaming=True, **kw)
    else:
        ds = load_dataset(spec["path"], spec.get("name"), split=spec.get("split", "train"), streaming=True, **kw)
    return ds.skip(skip) if skip else ds


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

    def __init__(self, specs=PRETRAIN_MIX, seed: int = 0, chars_per_token: float = 4.0,
                 sources_root: str | None = None, strict: bool = False):
        self.specs = specs
        self.sources_root = sources_root or os.environ.get("MAXGPT_SOURCES_ROOT") or None
        self.strict = bool(strict)      # True: a source that cannot be opened is an error, not a skipped source
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
                ds = LocalJsonlSource(s["local"], render=s.get("render"), text_field=s.get("text_field", "text"),
                                      start=int(s.get("skip", 0) or 0))
                srcs.append({"name": name, "ds": ds, "w": float(s["weight"]), "field": "text", "alive": True,
                             "exclude": s.get("exclude"), "min": s.get("min"), "rows": 0,
                             "epochs": max(1, int(s.get("epochs", 1))), "epoch": 0})
                continue
            try:
                ds = open_hf_source(s, self.sources_root, strict=self.strict)
            except Exception as e:
                if self.strict:                     # a real build must not quietly train without a source
                    raise RuntimeError(f"could not open data source {name}: {type(e).__name__}: {e}") from e
                print(f"[data] WARNING: could not open {name}: {type(e).__name__}: {e}. Skipping it.")
                continue
            srcs.append({"name": name, "ds": ds, "w": float(s["weight"]),
                         "field": s.get("text_field", "text"), "alive": True, "exclude": s.get("exclude"),
                         "min": s.get("min"), "rows": 0,
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
            e["rows"] = int(d.get("rows", 0))
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
            e["rows"] = e.get("rows", 0) + 1           # rows drawn, before any filter (see positions())
            if not _row_ok(ex, e.get("exclude"), e.get("min")):
                continue
            text = ex.get(e["field"]) if isinstance(ex, dict) else None
            if text:
                self._chars[e["name"]] = self._chars.get(e["name"], 0) + len(text)
                yield text, e["name"]

    def state_dict(self) -> dict:
        self._open()
        return {"rng": _rng_state_to_json(self._rng),
                "chars": dict(self._chars), "rep_chars": dict(self._rep_chars), "rep_tokens": dict(self._rep_tokens),
                "sources": [{"name": e["name"], "alive": e["alive"], "epoch": e["epoch"], "rows": e.get("rows", 0),
                             "ds": (e["ds"].state_dict() if e["alive"] else None)}
                            for e in self._sources]}

    def positions(self) -> dict:
        """Where each source's stream stands: rows drawn so far (for a local file, its line index).
        tokenize_to_shards writes this to meta.json as rows_by_source, so a later build (the anneal
        mix) can continue a source after the documents this build consumed (`after` in its spec).
        It can run up to one encode chunk ahead of what was written; the few skipped documents
        that costs are harmless."""
        self._open()
        def local_pos(ds):                               # a file read to its end counts as fully consumed
            return ds.end if ds.end is not None else ds.i
        return {e["name"]: int(local_pos(e["ds"]) if isinstance(e["ds"], LocalJsonlSource) else e.get("rows", 0))
                for e in self._sources}

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


def stream_mixed(specs=PRETRAIN_MIX, seed: int = 0, **kw) -> MixedStream:
    """Weighted, streamed, resumable blend of the sources. Returns a MixedStream (iterable), so
    existing callers (`for text, src in stream_mixed(...)`) keep working unchanged."""
    return MixedStream(specs, seed, **kw)


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
    if hasattr(items, "positions"):                     # lets a later build continue these streams
        meta["rows_by_source"] = items.positions()
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    if os.path.exists(prog_path):                       # finished cleanly -> drop the checkpoint
        os.remove(prog_path)
    return meta
