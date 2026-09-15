"""CPU smoke test for the training loop.

On a tiny model + tiny dataset it verifies the things that must be right before a
months-long run: the loss actually goes down, the WSD schedule has the right shape,
checkpoint+resume restores step and weights exactly, and the divergence guard detects a
NaN and rolls back.

Run from maxgpt-ultra/:  ../venv/bin/python scripts/test_train.py
"""
import math
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
    trainer._micro_forward = lambda sync=True: torch.tensor(float("nan"))   # a NaN loss, no backward
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

    print("\n[5] fp16 loss scaling is exact (same weights as the fp32 run, bit for bit)")
    # On CPU precision=fp16 runs the loss scaler without autocast: scaling by a power of two
    # commutes with every rounding, so after unscaling the gradients (and thus the weights) must
    # be identical to an unscaled run, not merely close.
    torch.manual_seed(1)
    m_plain = MaxGPTUltra(cfg)
    m_fp16 = MaxGPTUltra(cfg)
    m_fp16.load_state_dict(m_plain.state_dict())
    base = {**tcfg, "log_every": 1000}
    t_plain = Trainer(m_plain, PackedShardDataset(SHARDS, seq_len), {**base, "precision": "fp32"},
                      device="cpu", out_dir=OUT + "_plain", seed=0)
    t_fp16 = Trainer(m_fp16, PackedShardDataset(SHARDS, seq_len), {**base, "precision": "fp16"},
                     device="cpu", out_dir=OUT + "_fp16", seed=0)
    assert not t_plain.scaler.is_enabled() and t_fp16.scaler.is_enabled()
    for _ in range(8):
        r1, r2 = t_plain.train_step(), t_fp16.train_step()
    assert "loss_scale" in r2 and r2["loss_scale"] == 2.0 ** 16, r2
    for (n1, p1), (n2, p2) in zip(m_plain.named_parameters(), m_fp16.named_parameters()):
        assert torch.equal(p1, p2), f"{n1} differs between plain and loss-scaled runs"
    assert r1["loss"] == r2["loss"], (r1["loss"], r2["loss"])
    print(f"  8 steps, loss {r1['loss']:.4f} both ways, every weight tensor bit-identical ✓")

    print("\n[6] fp16 gradient overflow: step skipped, scale halved, NOT a divergence")
    before = [p.detach().clone() for p in m_fp16.parameters()]
    t_fp16.scaler.update(new_scale=2.0 ** 127)          # loss * 2^127 overflows fp32 -> inf grads
    step_before = t_fp16.step
    rec = t_fp16.train_step()
    assert rec.get("overflow") is True and rec["diverged"] is False, rec
    assert t_fp16.step == step_before + 1, "an overflow skips the update but still counts the step"
    assert rec["loss_scale"] == 2.0 ** 126, rec["loss_scale"]
    for p, b in zip(m_fp16.parameters(), before):
        assert torch.equal(p, b), "weights changed on an overflowed step"
    print(f"  overflow flagged, weights untouched, scale 2^127 -> 2^126 ✓")
    t_fp16.scaler.update(new_scale=2.0 ** 16)
    rec = t_fp16.train_step()
    assert not rec.get("overflow") and math.isfinite(rec["grad_norm"]), rec
    print("  next step trains normally ✓")

    print("\n[7] loss-scaler state survives checkpoint + resume")
    t_fp16.scaler.update(new_scale=2.0 ** 12)
    t_fp16.save()
    t_fp16.scaler.update(new_scale=2.0 ** 20)
    assert t_fp16.resume_if_available()
    assert t_fp16.scaler.get_scale() == 2.0 ** 12, t_fp16.scaler.get_scale()
    print("  scale restored from the checkpoint ✓")

    print("\n" + "=" * 72)
    print("ALL CHECKS PASSED ✅")
    print("=" * 72)


if __name__ == "__main__":
    main()
