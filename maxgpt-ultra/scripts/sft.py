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
from cfg import configure_triton_ptxas, cuda_alloc_conf
cuda_alloc_conf()            # expandable-segments allocator where the driver handles it (less VRAM fragmentation)
configure_triton_ptxas()   # old driver + CUDA 11.8 ptxas -> torch.compile still works (Lambda box)

import torch

from model import ModelConfig, MaxGPTUltra
from tokenizer.tokenizer import UltraTokenizer
from posttrain.sft_data import SFTDataset, ReplayBlend, load_chat_jsonl
from data.loader import PackedShardDataset
from model import load_yaml
from train.trainer import Trainer
from train.checkpoint import load_checkpoint
from train import dist as D


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
    ap.add_argument("--optimizer", default="config", help="config (the pretraining config's optimizer: matched, as the research report advises) | adamw | normuon")
    ap.add_argument("--replay-shards", default=None, help="pretraining shard dir to replay into SFT batches (data/shards_train)")
    ap.add_argument("--replay-frac", type=float, default=0.10, help="share of every batch that is replayed pretraining windows")
    ap.add_argument("--no-doc-mask", action="store_true", help="let packed conversations attend into each other (the old behaviour)")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dinfo = D.init_distributed(device)    # multi-GPU when launched by torchrun; else a no-op
    world = dinfo["world"]
    mcfg = ModelConfig.from_yaml(args.config)
    torch.manual_seed(0)

    tok = UltraTokenizer(args.tokenizer)
    examples = load_chat_jsonl(args.data)
    ds = SFTDataset(examples, tok, mcfg.seq_len)
    if args.replay_shards and args.replay_frac > 0:   # 10% pretraining windows in every batch (report 3.1)
        ds = ReplayBlend(ds, PackedShardDataset(args.replay_shards, mcfg.seq_len), frac=args.replay_frac)

    model = MaxGPTUltra(mcfg)
    if not args.no_doc_mask:                  # packed conversations must not attend into each other
        model.doc_mask, model.doc_sep_id = True, tok.eos_id
    if args.init:
        load_checkpoint(args.init, model, optimizer=None, map_location=device)  # pretrained weights only

    # --micro-batch x --grad-accum is the number of windows per optimizer step (what sets the
    # dynamics). Split it over the GPUs: each rank does grad_accum/world micro-steps of
    # micro_batch/... windows so the total per step stays the same on 1 or 8 cards.
    windows = args.micro_batch * args.grad_accum
    ga_per_rank = max(1, args.grad_accum // world)
    mb_per_rank = max(1, windows // (world * ga_per_rank))
    if D.is_main() and mb_per_rank * ga_per_rank * world != windows:
        print(f"[sft] note: {windows} windows/step does not split evenly over {world} GPUs; "
              f"using {mb_per_rank * ga_per_rank * world}", flush=True)
    total_tokens = int(args.epochs * ds.n)
    _tr = load_yaml(args.config).get("train", {}) or {}
    _opt = (_tr.get("optimizer", "adamw") if args.optimizer == "config" else args.optimizer).lower()
    tcfg = {"micro_batch": mb_per_rank, "grad_accum": ga_per_rank * world,
            "total_tokens": total_tokens, "warmup_tokens": int(0.03 * total_tokens),
            "lr": args.lr, "decay_frac": 0.1, "z_loss": 0.0, "grad_clip": 1.0,
            "autosave_minutes": 15, "log_every": 10, "keep_last_k": 2,
            "grad_checkpointing": False, "optimizer_8bit": False,
            "optimizer": _opt, "cautious_wd": _tr.get("cautious_wd", False), "muon_momentum": _tr.get("muon_momentum", 0.95),
            "compile": True, "eval_every": 200, "loss_chunk": 2048}

    # chat-formatted sample prompts so the dashboard shows the assistant's replies improving
    from eval.harness import evaluate
    _prompts = ["Hello! How are you?", "Explain photosynthesis in one sentence.",
                "Write a haiku about the ocean.", "def fizzbuzz(n):"]
    eval_fn = lambda m, step: evaluate(m, tokenizer=tok, sample_prompts=_prompts, device=device, chat=True)

    trainer = Trainer(model, ds, tcfg, device, args.out, eval_fn=eval_fn, seed=0, stop_file=args.stop_file)
    resumed = trainer.resume_if_available()

    import signal
    signal.signal(signal.SIGINT, lambda *_: trainer.request_stop())
    try:
        signal.signal(signal.SIGTERM, lambda *_: trainer.request_stop())
    except (ValueError, OSError):
        pass

    if D.is_main():
        print(f"[sft] device={device} gpus={world} examples={len(examples)} sft_tokens={ds.n:,} "
              f"steps={trainer.total_steps} resumed={resumed} init={'yes' if args.init else 'no'}", flush=True)
    trainer.train(max_steps=args.max_steps)
    if D.is_main():
        print(f"[sft] done at step {trainer.step}; checkpoints in {args.out}", flush=True)
    D.cleanup()


if __name__ == "__main__":
    main()
