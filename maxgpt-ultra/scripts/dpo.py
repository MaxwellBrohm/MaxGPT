"""DPO preference tuning, after SFT.

  python scripts/dpo.py --config configs/ultra.yaml --init runs/sft/checkpoints/best.pt \
      --tokenizer tokenizer/maxgpt-ultra.tokenizer.json --data data/prefs.jsonl --out runs/dpo

Loads the SFT model as the policy, freezes a copy as the reference, and optimizes the DPO
objective on preference pairs. Memory note for the 1B on 12GB: two model copies live at
once, so use a small --batch-size + --grad-checkpointing.

Preference jsonl: one {"prompt": [messages], "chosen": str, "rejected": str} per line.
"""
import argparse
import copy
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # maxgpt-ultra/
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # less VRAM fragmentation

import torch

from model import ModelConfig, MaxGPTUltra
from tokenizer.tokenizer import UltraTokenizer
from posttrain.dpo import DPODataset, DPOTrainer
from train.checkpoint import load_checkpoint


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--init", required=True, help="SFT checkpoint (policy + reference start)")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--data", required=True, help="preference jsonl: {prompt, chosen, rejected}")
    ap.add_argument("--out", default="runs/dpo")
    ap.add_argument("--device", default=None)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--grad-checkpointing", action="store_true")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--stop-file", default=None)
    args = ap.parse_args()

    mcfg = ModelConfig.from_yaml(args.config)
    import yaml
    _raw = yaml.safe_load(open(args.config, encoding="utf-8"))
    _dpo = ((_raw.get("posttrain", {}) or {}).get("dpo", {}) or {})   # optional DPO speed knobs
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    tok = UltraTokenizer(args.tokenizer)
    with open(args.data, encoding="utf-8") as f:
        examples = [json.loads(l) for l in f if l.strip()]
    data = DPODataset(examples, tok, mcfg.seq_len)

    policy = MaxGPTUltra(mcfg)
    load_checkpoint(args.init, policy, optimizer=None, map_location=device)
    ref = copy.deepcopy(policy)   # frozen reference = the SFT model

    steps = max(1, int(args.epochs * len(data) / args.batch_size))
    tcfg = {"batch_size": args.batch_size, "grad_accum": args.grad_accum, "total_steps": steps,
            "warmup_steps": max(1, steps // 20), "lr": args.lr, "decay_frac": 0.1,
            "grad_checkpointing": args.grad_checkpointing, "weight_decay": 0.0,
            "autosave_minutes": 15, "log_every": 10,
            # speed (all loss-free): bf16 + compile like pretrain/SFT, cache the frozen reference
            # once then free it, and chunk the 49k-wide logprob math. Overridable per-config.
            "compile": _dpo.get("compile", True),
            "compile_modes": _dpo.get("compile_modes", ["max-autotune", "default"]),
            "precompute_ref": _dpo.get("precompute_ref", True),
            "logp_chunk": int(_dpo.get("logp_chunk", 0))}
    from eval.harness import evaluate
    _prompts = ["Hello! How are you?", "What is the capital of France?",
                "Write a short poem about stars.", "def reverse(s):"]
    eval_fn = lambda m, step: evaluate(m, tokenizer=tok, sample_prompts=_prompts, device=device, chat=True)
    trainer = DPOTrainer(policy, ref, data, tcfg, device, args.out, beta=args.beta,
                         seed=0, stop_file=args.stop_file, eval_fn=eval_fn, eval_every=100)
    resumed = trainer.resume_if_available()

    import signal
    signal.signal(signal.SIGINT, lambda *_: trainer.request_stop())
    try:
        signal.signal(signal.SIGTERM, lambda *_: trainer.request_stop())
    except (ValueError, OSError):
        pass
    print(f"[dpo] device={device} pairs={len(data)} steps={steps} beta={args.beta} resumed={resumed}")
    trainer.train(max_steps=args.max_steps)
    print(f"[dpo] done at step {trainer.step}; checkpoints in {args.out}")


if __name__ == "__main__":
    main()
