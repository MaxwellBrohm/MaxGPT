"""The fixed benchmark suite (eval/suite.py), as a command.

Fetch it once (needs the network; the box streams from HuggingFace unauthenticated):
    python scripts/eval_suite.py --fetch --suite-dir data/bench --n 1000 --seed 0
Score a checkpoint (a full ckpt_*.pt or a weights_*.pt snapshot) on a free card:
    CUDA_VISIBLE_DEVICES=1 python scripts/eval_suite.py --config configs/ultra_lambda_final.yaml \
        --checkpoint runs/pretrain/checkpoints/ckpt_00010000.pt --device cuda --out runs/eval/suite_10000.json
Same checkpoint, another subset (the noise floor): fetch with --seed 1 into another --suite-dir.

The training run also scores the suite by itself every --suite-every steps (scripts/train.py), on
rank 0's card, and logs it into the eval row of runs/pretrain/metrics.jsonl as "suite".
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # maxgpt-ultra/
from cfg import cuda_alloc_conf   # noqa: E402  (allocator setting, before torch is imported; as scripts/train.py)
cuda_alloc_conf()


def load_weights(model, path: str) -> int:
    """Load a full checkpoint or a weights-only snapshot into `model`; returns its step."""
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    fixed = {}
    for k, v in state.items():                     # tolerate compiled / DDP prefixes
        for pre in ("_orig_mod.", "module."):
            if k.startswith(pre):
                k = k[len(pre):]
        fixed[k] = v
    model.load_state_dict(fixed)
    return int(ck.get("step", -1)) if isinstance(ck, dict) else -1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite-dir", default="data/bench")
    ap.add_argument("--fetch", action="store_true", help="download a seeded subset of every task into --suite-dir")
    ap.add_argument("--n", type=int, default=1000, help="examples per task when fetching")
    ap.add_argument("--seed", type=int, default=0, help="subset seed when fetching")
    ap.add_argument("--tasks", default=",".join(__import__("eval.suite", fromlist=["TASKS"]).TASKS))
    ap.add_argument("--config", help="model YAML (scoring)")
    ap.add_argument("--checkpoint", help="ckpt_*.pt or weights_*.pt (scoring)")
    ap.add_argument("--tokenizer", default="tokenizer/maxgpt-ultra.tokenizer.json")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--precision", default="auto", help="auto|fp16|bf16|fp32 autocast on cuda")
    ap.add_argument("--limit", type=int, default=None, help="score only the first N examples per task (quick look)")
    ap.add_argument("--out", default=None, help="write the results JSON here")
    args = ap.parse_args()
    tasks = tuple(t for t in args.tasks.split(",") if t)

    from eval.suite import fetch_suite, load_suite, run_suite, format_table
    if args.fetch:
        fetch_suite(args.suite_dir, n=args.n, seed=args.seed, tasks=tasks)
        print(f"[suite] written to {args.suite_dir}/ (suite.json lists n + source per task)")
        if not args.checkpoint:
            return

    if not (args.config and args.checkpoint):
        ap.error("--config and --checkpoint are required for scoring (or use --fetch)")
    import torch
    from model import ModelConfig, MaxGPTUltra
    from tokenizer.tokenizer import UltraTokenizer
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    suite = load_suite(args.suite_dir, tasks=tasks, n=args.limit)
    if not suite:
        ap.error(f"no task files in {args.suite_dir}; run with --fetch first")
    model = MaxGPTUltra(ModelConfig.from_yaml(args.config))
    step = load_weights(model, args.checkpoint)
    model.to(device)
    tok = UltraTokenizer(args.tokenizer)
    print(f"[suite] step {step}: {sum(len(v) for v in suite.values())} examples over {list(suite)} on {device}", flush=True)
    res = run_suite(model, tok, suite, device=device, batch_size=args.batch_size, precision=args.precision)
    print(format_table(res))
    rec = {"step": step, "checkpoint": args.checkpoint, "suite_dir": args.suite_dir,
           "when": time.strftime("%Y-%m-%d %H:%M"), **res}
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=2)
        print(f"[suite] saved {args.out}")


if __name__ == "__main__":
    main()
