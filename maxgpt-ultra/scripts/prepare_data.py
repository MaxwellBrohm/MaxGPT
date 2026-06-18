"""One-command data prep for the 5070 box.

  python scripts/prepare_data.py --config configs/ultra.yaml

Does the whole data stage so the PC just runs it and waits:
  1. trains the real ~49k tokenizer on a sample of the mix (skipped if one exists),
  2. streams the full mix and tokenizes it into memmapped shards under data/shards/.

Then `scripts/train.py` is ready to go. Use --smoke for a tiny local dry-run (no network,
writes to /tmp) that validates the wiring on the Mac.
"""
import argparse
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # maxgpt-ultra/

from model import ModelConfig
from tokenizer.tokenizer import train_tokenizer, UltraTokenizer
from data.prepare import tokenize_to_shards, stream_mixed, PRETRAIN_MIX

SMOKE_DOCS = [
    "The quick brown fox jumps over the lazy dog while 3 cats watch from the fence.",
    "def add(a, b):\n    return a + b\n\nprint(add(2, 3))  # 5",
    "Transformers learn from text by predicting the next token over and over.",
    "Café, naïve, 北京, 🚀 — byte-level tokenization handles every one of these.",
] * 100


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ultra.yaml")
    ap.add_argument("--tokenizer-out", default="tokenizer/maxgpt-ultra.tokenizer.json")
    ap.add_argument("--shards-out", default="data/shards")
    ap.add_argument("--tokenizer-sample-docs", type=int, default=2_000_000,
                    help="docs sampled to TRAIN the tokenizer")
    ap.add_argument("--max-docs", type=int, default=None, help="cap total corpus docs (None = all)")
    ap.add_argument("--shard-size", type=int, default=100_000_000, help="tokens per shard")
    ap.add_argument("--vocab-size", type=int, default=None, help="override config vocab")
    ap.add_argument("--smoke", action="store_true", help="tiny local dry-run, no network")
    args = ap.parse_args()

    mcfg = ModelConfig.from_yaml(args.config)
    vocab = args.vocab_size or mcfg.vocab_size

    if args.smoke:  # never touch the repo on a dry-run
        args.tokenizer_out = "/tmp/maxgpt_ultra_prep_tok.json"
        args.shards_out = "/tmp/maxgpt_ultra_prep_shards"
        vocab, sample_docs, shard_size = 2000, 5000, 4096
    else:
        sample_docs, shard_size = args.tokenizer_sample_docs, args.shard_size

    def corpus(limit=None):
        it = iter(SMOKE_DOCS) if args.smoke else stream_mixed(PRETRAIN_MIX)
        return itertools.islice(it, limit) if limit else it

    # 1) tokenizer
    if os.path.exists(args.tokenizer_out):
        print(f"[prepare] tokenizer already at {args.tokenizer_out}; skipping training")
    else:
        print(f"[prepare] training {vocab}-vocab tokenizer on up to {sample_docs:,} docs ...")
        train_tokenizer(corpus(sample_docs), vocab_size=vocab, out_path=args.tokenizer_out)
    tok = UltraTokenizer(args.tokenizer_out)
    print(f"[prepare] tokenizer ready: vocab={tok.vocab_size}")

    # 2) shards
    print(f"[prepare] tokenizing corpus -> shards in {args.shards_out}/ ...")
    meta = tokenize_to_shards(corpus(args.max_docs), tok, args.shards_out, shard_size=shard_size)
    print(f"[prepare] done: {meta['total_tokens']:,} tokens across {len(meta['shards'])} shard(s)")
    print(f"[prepare] next:  python scripts/train.py --config {args.config} --data {args.shards_out} --out runs/ultra")


if __name__ == "__main__":
    main()
