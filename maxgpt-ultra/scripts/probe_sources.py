"""Open every source of a mix and read a few rows: the 2-minute check before a multi-hour token build.

    python scripts/probe_sources.py --mix pretrain                      # straight from the hub
    python scripts/probe_sources.py --mix pretrain --sources-root $SCRATCH/maxgpt_sources
    python scripts/probe_sources.py --mix pretrain-nemotron             # after access to the gated set is granted

For each source it prints where it was read from (local files or the hub), the row fields, how many of the
rows drawn pass the source's filters and carry text, the median text length and the start of the first
text. It exits non-zero if any source cannot be opened or yields no usable text, so a wrong dataset id,
config or field name is caught here and not 11 hours into the build. A source read from local files is
also checked against the record its download left: a missing or cut-short file fails here.
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # maxgpt-ultra/

from data.prepare import MIXES, LocalJsonlSource, _local_files, _row_ok, open_hf_source, source_name


def probe(spec: dict, root: str | None, rows: int) -> dict:
    """Draw up to `rows` rows from one source. Returns counts and samples; raises if it cannot be opened."""
    if spec.get("local"):
        ds, where, field = LocalJsonlSource(spec["local"], render=spec.get("render"),
                                            text_field=spec.get("text_field", "text")), "local file", "text"
    else:
        local = _local_files(spec, root, strict=True)      # a local folder must match its download record
        ds = open_hf_source(spec, root, strict=True)
        where = f"local files ({len(local)})" if local else "hub"
        field = spec.get("text_field", "text")
    drawn, kept, lengths, first, fields = 0, 0, [], None, None
    for row in ds:
        drawn += 1
        fields = fields or (sorted(row.keys()) if isinstance(row, dict) else None)
        if _row_ok(row, spec.get("exclude"), spec.get("min")):
            text = row.get(field) if isinstance(row, dict) else None
            if text:
                kept += 1
                lengths.append(len(text))
                first = first or text
        if drawn >= rows:
            break
    return {"where": where, "drawn": drawn, "kept": kept, "fields": fields, "first": first,
            "median_chars": int(statistics.median(lengths)) if lengths else 0}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mix", choices=sorted(MIXES), default="pretrain")
    ap.add_argument("--sources-root", default=os.environ.get("MAXGPT_SOURCES_ROOT"))
    ap.add_argument("--rows", type=int, default=30)
    args = ap.parse_args()
    bad = []
    for spec in MIXES[args.mix]:
        name = source_name(spec)
        try:
            r = probe(spec, args.sources_root, args.rows)
        except Exception as e:
            print(f"[probe] {name}: FAILED to open: {type(e).__name__}: {str(e)[:300]}", flush=True)
            bad.append(name)
            continue
        ok = r["kept"] > 0
        print(f"[probe] {name:<22} {'ok  ' if ok else 'BAD '} from {r['where']}; {r['kept']}/{r['drawn']} rows usable, "
              f"median {r['median_chars']} chars; fields {r['fields']}", flush=True)
        if ok:
            print(f"          first text: {r['first'][:110]!r}", flush=True)
        else:
            bad.append(name)
    if bad:
        print(f"[probe] PROBLEM with: {bad}. Fix the spec in data/prepare.py, or finish the download, before building.")
        sys.exit(1)
    print("[probe] every source opened and yielded text")


if __name__ == "__main__":
    main()
