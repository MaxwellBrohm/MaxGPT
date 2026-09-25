"""Train MaxGPT-Ultra. The "run and walk away" entry point for the 5070 box.

  python scripts/train.py --config configs/ultra.yaml --data data/shards --out runs/ultra

Resumes automatically from the latest checkpoint in --out if one exists, so you can
stop (or lose power) and just run the same command again. Pause/play from the GUI uses
the same mechanism.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # maxgpt-ultra/
from cfg import configure_triton_ptxas, cuda_alloc_conf
cuda_alloc_conf()            # expandable-segments allocator where the driver handles it (less VRAM fragmentation)
configure_triton_ptxas()   # old driver + CUDA 11.8 ptxas -> torch.compile still works (Lambda box)

import torch

from model import ModelConfig, MaxGPTUltra, load_yaml
from data import PackedShardDataset
from train.trainer import Trainer
from train import dist as D


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="model+train YAML (e.g. configs/ultra.yaml)")
    ap.add_argument("--data", required=True, help="shard directory containing meta.json")
    ap.add_argument("--out", default="runs/ultra", help="output dir (checkpoints + metrics)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-steps", type=int, default=None, help="optional cap (for testing)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stop-file", default=None, help="if this file appears, checkpoint and exit (GUI pause)")
    ap.add_argument("--eval-data", default=None, help="held-out shard dir for periodic val perplexity")
    ap.add_argument("--eval-every", type=int, default=0, help="run eval every N steps")
    ap.add_argument("--tokenizer", default=None, help="tokenizer json (enables sample generations in eval)")
    ap.add_argument("--suite-dir", default=None, help="fixed benchmark suite dir (data/bench, see scripts/eval_suite.py)")
    ap.add_argument("--suite-every", type=int, default=10000,
                    help="score the suite at the eval on every Nth step, and at the first eval after a "
                         "(re)start when the last scored one is older than N steps")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dinfo = D.init_distributed(device)    # multi-GPU when launched by torchrun; else a no-op
    raw = load_yaml(args.config)
    mcfg = ModelConfig.from_yaml(args.config)
    tcfg = raw["train"]

    torch.manual_seed(args.seed)

    meta_path = os.path.join(args.data, "meta.json")
    if not os.path.exists(meta_path):
        if D.is_main():
            print(f"\n  ERROR: model / data not found  (no shards at '{args.data}').")
            print("  Nothing has been trained yet on this machine.")
            print("  Run  python scripts/prepare_data.py  to build the data, then press play again.\n")
        sys.exit(1)

    model = MaxGPTUltra(mcfg)
    data = PackedShardDataset(args.data, mcfg.seq_len)

    eval_fn = None
    if args.eval_data:
        from eval.harness import evaluate
        from tokenizer.tokenizer import UltraTokenizer
        val_data = PackedShardDataset(args.eval_data, mcfg.seq_len)
        etok = UltraTokenizer(args.tokenizer) if args.tokenizer else None
        prompts = ["The meaning of life is", "Once upon a time", "def add(a, b):"] if etok else None
        suite, last = None, [None]
        if args.suite_dir and etok:
            from eval.suite import load_suite, run_suite, last_suite_step
            suite = load_suite(args.suite_dir) or None
            if suite:
                last[0] = last_suite_step(os.path.join(args.out, "metrics.jsonl"))
                print(f"[train] benchmark suite: {sum(len(v) for v in suite.values())} examples over "
                      f"{list(suite)}, every {args.suite_every} steps (last scored: {last[0]})", flush=True)

        def eval_fn(m, step):
            rec = evaluate(m, tokenizer=etok, val_data=val_data, sample_prompts=prompts, device=device)
            due = suite and args.suite_every and (step % args.suite_every == 0 or last[0] is None
                                                  or step - last[0] >= args.suite_every)
            if due:                                    # the fixed suite rides in the same eval row
                rec["suite"] = run_suite(m, etok, suite, device=device)
                rec["suite_avg"] = rec["suite"]["avg"]
                last[0] = step
            return rec
        if args.eval_every:
            tcfg["eval_every"] = args.eval_every

    trainer = Trainer(model, data, tcfg, device, args.out, eval_fn=eval_fn,
                      seed=args.seed, stop_file=args.stop_file)
    resumed = trainer.resume_if_available()

    import signal
    def _graceful(*_):
        trainer.request_stop()
    signal.signal(signal.SIGINT, _graceful)
    try:
        signal.signal(signal.SIGTERM, _graceful)
    except (ValueError, OSError):
        pass

    if D.is_main():
        print(f"[train] device={device} gpus={dinfo['world']} params={model.num_params()/1e6:.1f}M "
              f"total_steps={trainer.total_steps} tokens/step={trainer.tokens_per_step} "
              f"resumed={resumed} (from step {trainer.step})", flush=True)
    trainer.train(max_steps=args.max_steps)
    if D.is_main():
        print(f"[train] done at step {trainer.step}; checkpoints + metrics in {args.out}", flush=True)
    D.cleanup()


if __name__ == "__main__":
    main()
