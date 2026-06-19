"""One-command data prep for the 5070 box: builds EVERYTHING the pipeline needs.

  python scripts/prepare_data.py --config configs/ultra.yaml

  1. trains the real ~49k tokenizer on a sample of the mix (skipped if one exists),
  2. streams the weighted mix and tokenizes it into shards, stopping at a token budget
     (default = the config's total_tokens) so the blend stays proportional and the disk
     stays bounded -- it will NOT fill up on one source,
  3. builds the SFT chat data (data/sft.jsonl) and DPO preference data (data/prefs.jsonl).

Then `train.py` / the GUI can run the whole pipeline. Use --smoke for a tiny local
dry-run (no network, writes to /tmp) that exercises all of the above.
"""
import argparse
import itertools
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # maxgpt-ultra/

import yaml

from model import ModelConfig
from tokenizer.tokenizer import train_tokenizer, UltraTokenizer
from data.prepare import tokenize_to_shards, stream_mixed, PRETRAIN_MIX
from posttrain.sft_data import build_sft_jsonl
from posttrain.dpo import build_pref_jsonl

SMOKE_DOCS = [
    "The quick brown fox jumps over the lazy dog while 3 cats watch from the fence.",
    "def add(a, b):\n    return a + b\n\nprint(add(2, 3))  # 5",
    "Transformers learn from text by predicting the next token over and over.",
    "Café, naïve, 北京, 🚀 — byte-level tokenization handles every one of these.",
] * 100
SMOKE_SFT = [{"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello!"}]}] * 20
SMOKE_PREF = [{"prompt": [{"role": "user", "content": "say a"}], "chosen": "a", "rejected": "b"}] * 20


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ultra.yaml")
    ap.add_argument("--tokenizer-out", default="tokenizer/maxgpt-ultra.tokenizer.json")
    ap.add_argument("--shards-out", default="data/shards")
    ap.add_argument("--sft-out", default="data/sft.jsonl")
    ap.add_argument("--pref-out", default="data/prefs.jsonl")
    ap.add_argument("--max-tokens", type=float, default=None, help="pretrain token budget (default: config total_tokens)")
    ap.add_argument("--max-docs", type=int, default=None, help="hard cap on docs (optional; for quick tests)")
    ap.add_argument("--tokenizer-sample-docs", type=int, default=2_000_000)
    ap.add_argument("--shard-size", type=int, default=100_000_000)
    ap.add_argument("--vocab-size", type=int, default=None)
    ap.add_argument("--sft-examples", type=int, default=100_000)
    ap.add_argument("--pref-examples", type=int, default=60_000)
    ap.add_argument("--skip-posttrain", action="store_true", help="only build the pretrain corpus")
    ap.add_argument("--smoke", action="store_true", help="tiny local dry-run, no network")
    args = ap.parse_args()

    raw = yaml.safe_load(open(args.config))
    mcfg = ModelConfig.from_yaml(args.config)
    vocab = args.vocab_size or mcfg.vocab_size
    max_tokens = int(args.max_tokens) if args.max_tokens else int(float(raw.get("train", {}).get("total_tokens", 1e11)))

    if args.smoke:
        args.tokenizer_out = "/tmp/mgu_prep_tok.json"
        args.shards_out = "/tmp/mgu_prep_shards"
        args.sft_out, args.pref_out = "/tmp/mgu_prep_sft.jsonl", "/tmp/mgu_prep_prefs.jsonl"
        vocab, sample_docs, shard_size, max_tokens = 2000, 5000, 4096, 20000
    else:
        sample_docs, shard_size = args.tokenizer_sample_docs, args.shard_size

    def text_stream(limit=None):                      # plain text, for tokenizer training
        it = iter(SMOKE_DOCS) if args.smoke else (t for t, _ in stream_mixed(PRETRAIN_MIX))
        return itertools.islice(it, limit) if limit else it

    def tagged_stream(limit=None):                    # (text, source), for sharding
        it = (((d, "smoke") for d in SMOKE_DOCS)) if args.smoke else stream_mixed(PRETRAIN_MIX)
        return itertools.islice(it, limit) if limit else it

    # 1) tokenizer
    if os.path.exists(args.tokenizer_out):
        print(f"[prepare] tokenizer already at {args.tokenizer_out}; skipping training")
    else:
        print(f"[prepare] training {vocab}-vocab tokenizer on up to {sample_docs:,} docs ...")
        train_tokenizer(text_stream(sample_docs), vocab_size=vocab, out_path=args.tokenizer_out)
    tok = UltraTokenizer(args.tokenizer_out)
    print(f"[prepare] tokenizer ready: vocab={tok.vocab_size}")

    # 2) pretrain shards (proportional mix, capped at the token budget)
    print(f"[prepare] tokenizing the mix -> {args.shards_out}/  (budget ~{max_tokens:,} tokens) ...")
    meta = tokenize_to_shards(tagged_stream(args.max_docs), tok, args.shards_out,
                              shard_size=shard_size, max_tokens=max_tokens)
    tot = meta["total_tokens"]
    print(f"[prepare] pretrain: {tot:,} tokens in {len(meta['shards'])} shard(s). mix:")
    for src, n in sorted(meta["by_source"].items(), key=lambda kv: -kv[1]):
        print(f"           {src:<22} {n:>14,}  ({100*n/max(1,tot):4.1f}%)")

    # 3) SFT + DPO data
    if not args.skip_posttrain:
        if args.smoke:
            for path, rows in ((args.sft_out, SMOKE_SFT), (args.pref_out, SMOKE_PREF)):
                with open(path, "w") as f:
                    for r in rows:
                        f.write(json.dumps(r) + "\n")
            print(f"[prepare] sft={len(SMOKE_SFT)} rows, prefs={len(SMOKE_PREF)} rows (smoke)")
        else:
            print(f"[prepare] building SFT chat data -> {args.sft_out} ...")
            ns = build_sft_jsonl(args.sft_out, n=args.sft_examples)
            print(f"[prepare] building DPO preference data -> {args.pref_out} ...")
            npf = build_pref_jsonl(args.pref_out, n=args.pref_examples)
            print(f"[prepare] sft={ns:,} rows, prefs={npf:,} rows")

    print("[prepare] done. next:  python gui/server.py --config " + args.config)


if __name__ == "__main__":
    main()
