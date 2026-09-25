"""Sanity-check a finished anneal shard build before it is swapped into the run.

    python3 scripts/anneal_check.py data/shards_anneal_v3 [--min-tokens 5.9e9] [--min-chat 4] [--min-csn 6]

Prints one line: "OK <shares>" (exit 0) or "BAD <reasons> | <shares>" (exit 1). Standard-library
only, so it runs under the box's system python3 from a bash watcher. The thresholds encode what a
usable decay-phase mix needs: the token budget was met, the chat share is real (the first two
builds landed at 2.1% because the SFT file ran dry), and the CodeSearchNet languages are present.
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def check(shard_dir: str, min_tokens: float, min_chat: float, min_csn: float, min_shards: int = 10) -> tuple[bool, str]:
    path = os.path.join(shard_dir, "meta.json")
    if not os.path.exists(path):
        return False, f"BAD no meta.json in {shard_dir} (build not finished)"
    with open(path, encoding="utf-8") as f:
        m = json.load(f)
    t = int(m.get("total_tokens", 0))
    bs = {str(k): int(v) for k, v in m.get("by_source", {}).items()}
    chat = 100.0 * bs.get("chat-sft", 0) / max(1, t)
    csn = 100.0 * sum(v for k, v in bs.items() if k.startswith("csn-")) / max(1, t)
    problems = []
    if t < min_tokens:
        problems.append(f"total {t:,} < {int(min_tokens):,}")
    if chat < min_chat:
        problems.append(f"chat {chat:.1f}% < {min_chat:.1f}%")
    if csn < min_csn:
        problems.append(f"csn {csn:.1f}% < {min_csn:.1f}%")
    if len(m.get("shards", [])) < min_shards:
        problems.append(f"only {len(m.get('shards', []))} shards")
    shares = " ".join(f"{k}={100.0 * v / max(1, t):.1f}%" for k, v in sorted(bs.items(), key=lambda kv: -kv[1]))
    if problems:
        return False, "BAD " + "; ".join(problems) + " | " + shares
    return True, f"OK {t:,} tokens, {len(m['shards'])} shards | " + shares


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("shard_dir")
    ap.add_argument("--min-tokens", type=float, default=5.9e9)
    ap.add_argument("--min-chat", type=float, default=4.0, help="minimum chat-sft share, percent")
    ap.add_argument("--min-csn", type=float, default=6.0, help="minimum CodeSearchNet share (all languages), percent")
    args = ap.parse_args()
    ok, line = check(args.shard_dir, args.min_tokens, args.min_chat, args.min_csn)
    print(line)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
