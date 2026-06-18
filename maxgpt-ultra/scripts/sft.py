"""Supervised fine-tuning: turn the pretrained base model into a chat assistant.

  python scripts/sft.py --config configs/ultra.yaml --init runs/ultra/checkpoints/best.pt \
      --tokenizer tokenizer/maxgpt-ultra.tokenizer.json --data data/sft.jsonl --out runs/sft

Loads the pretrained weights, then fine-tunes on chat data with assistant-only loss
masking (reusing the same trainer, just a lower LR and a short schedule). Auto-resumes
from --out, and supports the GUI stop-file for pause/play.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # maxgpt-ultra/

import torch

from model import ModelConfig, MaxGPTUltra
from tokenizer.tokenizer import UltraTokenizer
from posttrain.sft_data import SFTDataset, load_chat_jsonl
from train.trainer import Trainer
from train.checkpoint import load_checkpoint


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--init", default=None, help="pretrained checkpoint to start from (weights only)")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--data", required=True, help="chat jsonl: one {'messages': [...]} per line")
    ap.add_argument("--out", default="runs/sft")
    ap.add_argument("--device", default=None)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--stop-file", default=None)
    args = ap.parse_args()

    mcfg = ModelConfig.from_yaml(args.config)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    tok = UltraTokenizer(args.tokenizer)
    examples = load_chat_jsonl(args.data)
    ds = SFTDataset(examples, tok, mcfg.seq_len)

    model = MaxGPTUltra(mcfg)
    if args.init:
        load_checkpoint(args.init, model, optimizer=None, map_location=device)  # pretrained weights only

    tokens_per_step = args.micro_batch * args.grad_accum * mcfg.seq_len
    total_tokens = int(args.epochs * ds.n)
    tcfg = {"micro_batch": args.micro_batch, "grad_accum": args.grad_accum,
            "total_tokens": total_tokens, "warmup_tokens": int(0.03 * total_tokens),
            "lr": args.lr, "decay_frac": 0.1, "z_loss": 0.0, "grad_clip": 1.0,
            "autosave_minutes": 15, "log_every": 10, "keep_last_k": 2,
            "grad_checkpointing": False, "optimizer_8bit": False}

    trainer = Trainer(model, ds, tcfg, device, args.out, seed=0, stop_file=args.stop_file)
    resumed = trainer.resume_if_available()

    import signal
    signal.signal(signal.SIGINT, lambda *_: trainer.request_stop())
    try:
        signal.signal(signal.SIGTERM, lambda *_: trainer.request_stop())
    except (ValueError, OSError):
        pass

    print(f"[sft] device={device} examples={len(examples)} sft_tokens={ds.n:,} "
          f"steps={trainer.total_steps} resumed={resumed} init={'yes' if args.init else 'no'}")
    trainer.train(max_steps=args.max_steps)
    print(f"[sft] done at step {trainer.step}; checkpoints in {args.out}")


if __name__ == "__main__":
    main()
