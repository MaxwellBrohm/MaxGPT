"""Carve a held-out validation set out of a finished shard build WITHOUT copying data: symlink the
last --n-val shards into a val dir and the rest into a train dir, each with its own meta.json, so the
dashboard's val perplexity is measured on tokens the model never trains on.

  python scripts/holdout_shard.py --shards data/shards --train data/shards_train --val data/shards_val_ultra
"""
import argparse
import json
import os


def link_set(src: str, meta: dict, names: set[str], dst: str) -> dict:
    os.makedirs(dst, exist_ok=True)
    rows = []
    for s in meta["shards"]:
        if s["name"] not in names:
            continue
        link = os.path.join(dst, s["name"])
        if not os.path.lexists(link):
            os.symlink(os.path.abspath(os.path.join(src, s["name"])), link)
        rows.append({"name": s["name"], "tokens": int(s["tokens"])})
    out = {k: v for k, v in meta.items() if k not in ("shards", "total_tokens", "by_source")}
    out["shards"] = rows
    out["total_tokens"] = sum(r["tokens"] for r in rows)
    with open(os.path.join(dst, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="data/shards")
    ap.add_argument("--train", default="data/shards_train")
    ap.add_argument("--val", default="data/shards_val_ultra")
    ap.add_argument("--n-val", type=int, default=1)
    args = ap.parse_args()
    with open(os.path.join(args.shards, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    names = [s["name"] for s in meta["shards"]]
    assert len(names) > args.n_val, "need more shards than the held-out count"
    tr = link_set(args.shards, meta, set(names[:-args.n_val]), args.train)
    va = link_set(args.shards, meta, set(names[-args.n_val:]), args.val)
    print(f"train: {len(tr['shards'])} shards, {tr['total_tokens']:,} tokens -> {args.train}")
    print(f"val:   {len(va['shards'])} shards, {va['total_tokens']:,} tokens -> {args.val}")


if __name__ == "__main__":
    main()
