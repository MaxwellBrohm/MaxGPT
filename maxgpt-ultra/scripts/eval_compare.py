"""Compare two checkpoints on the fixed suite with question-level paired differences.

    python scripts/eval_compare.py --config configs/ultra_lambda_final.yaml \
        --a runs/sft/checkpoints/ckpt_X.pt --b runs/dpo_sweep/beta_0.3/checkpoints/ckpt_Y.pt \
        --suite-dir data/bench --device cuda --out runs/eval/compare_sft_vs_dpo03.json

Both models score the same examples; each task reports A's and B's accuracy, the paired delta
(B minus A) with its standard error and a 95% bootstrap interval, and how many examples flipped
each way. At 1.1B most of the movement between two recipes is noise, and pairing removes the
example-sampling part of it (docs/research_2026-09-22.md 3.4). Pick by this, never by
preference win-rate.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # maxgpt-ultra/
from cfg import cuda_alloc_conf   # noqa: E402
cuda_alloc_conf()

from scripts.eval_suite import load_weights   # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--a", required=True, help="checkpoint A (the baseline, e.g. the SFT model)")
    ap.add_argument("--b", required=True, help="checkpoint B (the candidate)")
    ap.add_argument("--tokenizer", default="tokenizer/maxgpt-ultra.tokenizer.json")
    ap.add_argument("--suite-dir", default="data/bench")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--precision", default="auto")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch
    from model import ModelConfig, MaxGPTUltra
    from tokenizer.tokenizer import UltraTokenizer
    from eval.suite import load_suite, run_suite, paired_compare, format_compare, format_table
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    suite = load_suite(args.suite_dir, n=args.limit)
    if not suite:
        ap.error(f"no task files in {args.suite_dir}; run scripts/eval_suite.py --fetch first")
    tok = UltraTokenizer(args.tokenizer)
    cfg = ModelConfig.from_yaml(args.config)
    results = {}
    for tag, path in (("a", args.a), ("b", args.b)):
        model = MaxGPTUltra(cfg)
        step = load_weights(model, path)
        model.to(device)
        print(f"[compare] {tag} = {path} (step {step})", flush=True)
        results[tag] = run_suite(model, tok, suite, device=device, batch_size=args.batch_size,
                                 precision=args.precision, details=True)
        print(format_table(results[tag]))
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    cmp = paired_compare(results["a"], results["b"])
    print("\n=== paired (B minus A) ===")
    print(format_compare(cmp, "A", "B"))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        slim = {k: {t: {kk: vv for kk, vv in m.items() if kk != "correct"} if isinstance(m, dict) else m
                    for t, m in v.items()} for k, v in results.items()}
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"a": args.a, "b": args.b, "results": slim, "paired": cmp}, f, indent=2)
        print(f"[compare] saved {args.out}")


if __name__ == "__main__":
    main()
