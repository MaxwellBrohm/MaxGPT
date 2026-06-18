"""CPU smoke test for the training loop.

On a tiny model + tiny dataset it verifies the things that must be right before a
months-long run: the loss actually goes down, the WSD schedule has the right shape,
checkpoint+resume restores step and weights exactly, and the divergence guard detects a
NaN and rolls back.

Run from maxgpt-ultra/:  ../venv/bin/python scripts/test_train.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # maxgpt-ultra/

import torch

from model import ModelConfig, MaxGPTUltra
from tokenizer.tokenizer import train_tokenizer, UltraTokenizer
from data.prepare import tokenize_to_shards
from data.loader import PackedShardDataset
from train.schedule import wsd_lr
from train.trainer import Trainer

TOK = "/tmp/maxgpt_ultra_train_tok.json"
SHARDS = "/tmp/maxgpt_ultra_train_shards"
OUT = "/tmp/maxgpt_ultra_train_run"

DOCS = [
    "The cat sat on the mat while the dog ran in the yard near the old red barn.",
    "Numbers like 7 and 42 and 100 show up when we count things in the world.",
    "def square(n):\n    return n * n\n\nprint(square(9))",
    "Attention lets each token look at the others; that is the core transformer idea.",
] * 80


def main() -> None:
    print("=" * 72)
    print("MaxGPT-Ultra training-loop smoke test")
    print("=" * 72)

    print("\n[setup] tiny tokenizer + shards + model")
    train_tokenizer(iter(DOCS), vocab_size=1200, out_path=TOK)
    tok = UltraTokenizer(TOK)
    tokenize_to_shards(DOCS, tok, SHARDS, shard_size=1024)
    seq_len = 16
    cfg = ModelConfig(vocab_size=tok.vocab_size, d_model=128, n_layers=2, n_heads=4,
                      n_kv_heads=2, mlp_hidden=256, seq_len=seq_len)
    model = MaxGPTUltra(cfg)
    data = PackedShardDataset(SHARDS, seq_len)
    tcfg = {"micro_batch": 4, "grad_accum": 2, "total_tokens": 128 * 80,
            "warmup_tokens": 128 * 5, "lr": 3e-3, "decay_frac": 0.2, "grad_clip": 1.0,
            "z_loss": 1e-4, "autosave_minutes": 9999, "log_every": 5, "keep_last_k": 2}
    trainer = Trainer(model, data, tcfg, device="cpu", out_dir=OUT, seed=0)
    print(f"  params={model.num_params()/1e6:.2f}M  total_steps={trainer.total_steps}  warmup={trainer.warmup_steps}")

    print("\n[1] WSD schedule shape")
    lr0 = wsd_lr(0, total_steps=trainer.total_steps, warmup_steps=trainer.warmup_steps, decay_frac=0.2, max_lr=3e-3)
    lr_mid = wsd_lr(trainer.warmup_steps + 1, total_steps=trainer.total_steps, warmup_steps=trainer.warmup_steps, decay_frac=0.2, max_lr=3e-3)
    lr_end = wsd_lr(trainer.total_steps - 1, total_steps=trainer.total_steps, warmup_steps=trainer.warmup_steps, decay_frac=0.2, max_lr=3e-3)
    assert lr0 < lr_mid and abs(lr_mid - 3e-3) < 1e-9 and lr_end < lr_mid, (lr0, lr_mid, lr_end)
    print(f"  warmup {lr0:.2e} < stable {lr_mid:.2e} > decay {lr_end:.2e} ✓")

    print("\n[2] loss decreases over 30 steps")
    first = trainer.train_step()["loss"]
    for _ in range(29):
        trainer.train_step()
    last = trainer.train_step()["loss"]
    print(f"  loss {first:.3f} -> {last:.3f}")
    assert last < first - 0.5, f"loss did not fall enough ({first:.3f} -> {last:.3f})"
    print("  loss fell substantially ✓")

    print("\n[3] checkpoint + exact resume")
    trainer.save()
    step_at_save = trainer.step
    ref = next(p for p in model.parameters() if p.dim() >= 2).detach().clone()
    # train a bit more so the live model diverges from the checkpoint, then resume
    for _ in range(5):
        trainer.train_step()
    assert trainer.step != step_at_save
    trainer.resume_if_available()
    assert trainer.step == step_at_save, (trainer.step, step_at_save)
    now = next(p for p in model.parameters() if p.dim() >= 2).detach()
    assert torch.allclose(now, ref), "weights not restored on resume"
    print(f"  resumed to step {step_at_save} with weights restored exactly ✓")

    print("\n[4] divergence guard detects NaN and rolls back")
    trainer.save()
    safe_step = trainer.step
    p0 = next(p for p in model.parameters() if p.dim() >= 2).detach().clone()
    orig = trainer._micro_forward
    trainer._micro_forward = lambda: torch.tensor(float("nan"), requires_grad=True)
    rec = trainer.train_step()
    assert rec["diverged"] and trainer.step == safe_step, rec
    print(f"  NaN loss flagged diverged, step held at {safe_step} ✓")
    trainer._micro_forward = orig
    # corrupt a weight, then roll back to the good checkpoint
    with torch.no_grad():
        next(p for p in model.parameters() if p.dim() >= 2).add_(1.0)
    assert trainer._rollback()
    p1 = next(p for p in model.parameters() if p.dim() >= 2).detach()
    assert torch.allclose(p1, p0), "rollback did not restore weights"
    print("  rollback restored the last good weights ✓")

    print("\n" + "=" * 72)
    print("ALL CHECKS PASSED ✅")
    print("=" * 72)


if __name__ == "__main__":
    main()
