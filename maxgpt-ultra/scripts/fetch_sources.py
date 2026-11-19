"""Download the data files a mix needs into a local sources root, so the token build reads from disk.

    export HF_TOKEN=...                       # always: anonymous hub limits are per IP, and a cluster shares one
    python scripts/fetch_sources.py --mix pretrain --root $SCRATCH/maxgpt_sources --tokens 1.0e11 --dry-run
    python scripts/fetch_sources.py --mix pretrain --root $SCRATCH/maxgpt_sources --tokens 1.0e11
    python scripts/fetch_sources.py --mix anneal   --root $SCRATCH/maxgpt_sources --tokens 8.0e9
    MAXGPT_SOURCES_ROOT=$SCRATCH/maxgpt_sources python scripts/prepare_data.py --mix pretrain ...

Why download first: cluster compute nodes may have no internet, a multi-hour stream from the hub can be
throttled, and a local copy makes the build repeatable. Layout: <root>/<dataset id>/<config or 'default'>/...,
which is where data/prepare.py looks (local_source_dir).

How much of each source: its token share x the token budget x a generous bytes-per-token allowance, taken in
the repository's own file order (the order streaming reads them). Sources that are read interleaved across
all their files, or marked "fetch": "all", are downloaded whole. Local jsonl sources are skipped.

Safe to run again: files already there are kept, except one whose size differs from the hub's (cut short
or damaged), which is downloaded afresh. Only when every file is down does each source
get its download record (.maxgpt_fetch.json in its folder); the token build reads exactly the recorded files
and refuses a folder without a record or one that no longer matches it. Do not run two of these at once on
the same root, and not while a build is reading it.
"""
from __future__ import annotations

import argparse
import fnmatch
import os
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # maxgpt-ultra/

from data.prepare import MIXES, local_source_dir, source_name, write_fetch_record

# Compressed bytes allowed per training token. Parquet / gzip text is about 1.2 to 2 bytes per token for
# these sources; 3.5 with a 1.3 margin leaves room for row filters and metadata columns, so a source does
# not run dry mid-build (the build reports it if one does).
BYTES_PER_TOKEN = 3.5
MARGIN = 1.3


def need_bytes(spec: dict, total_tokens: float, bytes_per_token: float = BYTES_PER_TOKEN,
               margin: float = MARGIN) -> float | None:
    """Bytes of files to fetch for this source, or None for "all of them"."""
    if spec.get("interleave_files") or spec.get("fetch") == "all":
        return None
    return float(spec["weight"]) * float(total_tokens) * float(spec.get("bytes_per_token", bytes_per_token)) * margin


def choose_files(files: list[tuple[str, int]], need: float | None) -> list[tuple[str, int]]:
    """The shortest prefix of `files` (path, size), in the given order, whose sizes add up to at least
    `need` bytes. need=None takes everything; an empty list stays empty; at least one file is taken."""
    if need is None:
        return list(files)
    out, total = [], 0
    for path, size in files:
        out.append((path, size))
        total += int(size or 0)
        if total >= need:
            break
    return out


def remote_files(spec: dict) -> tuple[list[tuple[str, int]], str | None]:
    """((path in repo, size), ...) for this source in streaming order, and the revision they belong to."""
    from huggingface_hub import HfApi
    api, repo = HfApi(), spec["path"]
    if spec.get("interleave_files"):
        tree = api.list_repo_tree(repo, repo_type="dataset", recursive=True)
        files = sorted((t.path, int(getattr(t, "size", 0) or 0)) for t in tree
                       if hasattr(t, "size") and fnmatch.fnmatch(os.path.basename(t.path), spec["interleave_files"]))
        return files, None
    from datasets import load_dataset_builder
    kw = {"columns": list(spec["columns"])} if spec.get("columns") else {}
    builder = load_dataset_builder(spec["path"], spec.get("name"), **kw)
    urls = list(builder.config.data_files[spec.get("split", "train")])
    prefix = f"hf://datasets/{repo}@"
    paths, rev = [], None
    for u in urls:
        if not u.startswith(prefix):
            raise ValueError(f"unexpected data file url for {repo}: {u}")
        rev, p = u[len(prefix):].split("/", 1)
        paths.append(p)
    common = os.path.commonpath([os.path.dirname(p) for p in paths]) if paths else ""
    sizes = {t.path: int(getattr(t, "size", 0) or 0)
             for t in api.list_repo_tree(repo, path_in_repo=common or None, repo_type="dataset",
                                         recursive=True, revision=rev) if hasattr(t, "size")}
    return [(p, sizes.get(p, 0)) for p in paths], rev


def record_downloads(plans: list, root: str, total_tokens: float) -> None:
    """Once every file is on disk, leave each source's download record. data/prepare.py reads exactly the
    recorded files and refuses a folder that does not match its record, so a download that stopped
    halfway (no record yet) or lost files later can never pass for the whole source.

    plans: (spec, revision, [(path, size on the hub), ...], whole source taken?) per source."""
    for spec, rev, take, whole in plans:
        d = local_source_dir(spec, root)
        files = []
        for path, hub_size in take:
            size = os.path.getsize(os.path.join(d, path))
            if hub_size and size != int(hub_size):
                raise SystemExit(f"[fetch] {source_name(spec)}: {path} is {size:,} bytes on disk but {int(hub_size):,} "
                                 f"on the hub. Delete it and run this again.")
            files.append((path, size))
        write_fetch_record(spec, root, files, revision=rev,
                           source_tokens=None if whole else float(spec["weight"]) * float(total_tokens))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mix", choices=sorted(MIXES), default="pretrain")
    ap.add_argument("--root", required=True, help="local sources root (use scratch on a cluster)")
    ap.add_argument("--tokens", type=float, required=True, help="token budget of the build this is for")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true", help="print the plan, download nothing")
    args = ap.parse_args()

    plans, total_gb = [], 0.0
    for spec in MIXES[args.mix]:
        name = source_name(spec)
        if spec.get("local"):
            print(f"[fetch] {name}: local file {spec['local']} (nothing to download)")
            continue
        files, rev = remote_files(spec)
        if not files:
            raise SystemExit(f"[fetch] {name}: no data files found in {spec['path']}")
        take = choose_files(files, need_bytes(spec, args.tokens))
        gb = sum(sz for _, sz in take) / 1e9
        total_gb += gb
        why = "all files" if len(take) == len(files) else "share %.2f of %.3g tokens" % (spec["weight"], args.tokens)
        print(f"[fetch] {name:<22} {len(take):>6} of {len(files):>6} files  {gb:8.1f} GB  ({why})"
              f" -> {local_source_dir(spec, args.root)}", flush=True)
        plans.append((spec, rev, take, len(take) == len(files)))
    print(f"[fetch] total {total_gb:.1f} GB" + (" (dry run, nothing downloaded)" if args.dry_run else ""), flush=True)
    if args.dry_run:
        return
    if not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")):
        print("[fetch] WARNING: no HF_TOKEN set; anonymous downloads are rate-limited per IP", flush=True)

    from huggingface_hub import hf_hub_download

    def get(job):
        spec, rev, path, hub_size = job
        d = local_source_dir(spec, args.root)
        fp = os.path.join(d, path)
        if hub_size and os.path.isfile(fp) and os.path.getsize(fp) != int(hub_size):
            os.remove(fp)        # cut short or damaged: the hub library trusts its own notes and would keep it
        return hf_hub_download(spec["path"], path, repo_type="dataset", revision=rev, local_dir=d)

    jobs = [(spec, rev, p, sz) for spec, rev, take, _ in plans for p, sz in take]
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for _ in pool.map(get, jobs):
            done += 1
            if done % 25 == 0 or done == len(jobs):
                print(f"[fetch] {done}/{len(jobs)} files", flush=True)
    record_downloads(plans, args.root, args.tokens)
    print(f"[fetch] done, every file checked and recorded. next: MAXGPT_SOURCES_ROOT={args.root} "
          f"python scripts/prepare_data.py --mix {args.mix} ...")


if __name__ == "__main__":
    main()
