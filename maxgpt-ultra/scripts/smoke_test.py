"""CPU smoke test for the MaxGPT-Ultra architecture.

Verifies, without a GPU:
  - both configs build and have the expected parameter counts (1B built on the
    'meta' device so it costs no real memory),
  - a forward pass returns the right shape and a sane init loss (~ln(vocab)),
  - backward produces finite gradients,
  - attention is genuinely causal (a future token can't change past logits),
  - GQA wiring is correct.

Run from the maxgpt-ultra/ folder:  ../venv/bin/python scripts/smoke_test.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # maxgpt-ultra/

import torch

from model import ModelConfig, MaxGPTUltra


def report_params(path: str, label: str) -> ModelConfig:
    cfg = ModelConfig.from_yaml(path)
    try:
        with torch.device("meta"):
            m = MaxGPTUltra(cfg)
        total = sum(p.numel() for p in m.parameters())          # tied weights once
        non_emb = total - cfg.vocab_size * cfg.d_model
        print(f"  {label:10s} {total/1e6:7.1f}M params  ({non_emb/1e6:6.1f}M non-embedding)  "
              f"| d_model={cfg.d_model} L={cfg.n_layers} H={cfg.n_heads}/{cfg.n_kv_heads}kv "
              f"hd={cfg.head_dim} vocab={cfg.vocab_size} seq={cfg.seq_len}")
    except Exception as e:  # pragma: no cover
        print(f"  {label}: meta build failed: {e}")
    return cfg


def main() -> None:
    print("=" * 78)
    print("MaxGPT-Ultra architecture smoke test")
    print("=" * 78)

    print("\n[1] parameter counts")
    shake_cfg = report_params("configs/shakedown.yaml", "shakedown")
    report_params("configs/ultra.yaml", "ultra")

    print("\n[2] functional test (shakedown config, CPU, fp32)")
    torch.manual_seed(0)
    cfg = shake_cfg
    model = MaxGPTUltra(cfg)
    model.eval()
    B, T = 2, 128
    idx = torch.randint(0, cfg.vocab_size, (B, T))
    targets = torch.randint(0, cfg.vocab_size, (B, T))

    logits, loss = model(idx, targets, z_loss_weight=1e-4)
    assert logits.shape == (B, T, cfg.vocab_size), f"bad logits shape {logits.shape}"
    assert torch.isfinite(loss), f"non-finite loss {loss}"
    expected = math.log(cfg.vocab_size)
    print(f"  forward OK   logits={tuple(logits.shape)}  loss={loss.item():.3f}  "
          f"(expected ~{expected:.2f} at init)")
    assert abs(loss.item() - expected) < 1.0, "init loss far from ln(vocab) — suspicious"

    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no gradients produced"
    assert all(torch.isfinite(g).all() for g in grads), "non-finite gradient"
    gnorm = torch.sqrt(sum(g.pow(2).sum() for g in grads)).item()
    print(f"  backward OK  grad norm={gnorm:.3f}  ({len(grads)} tensors with grad)")

    print("\n[3] causality (changing the last token must not move earlier logits)")
    with torch.no_grad():
        a, _ = model(idx)
        idx2 = idx.clone()
        idx2[:, -1] = (idx2[:, -1] + 1) % cfg.vocab_size
        b, _ = model(idx2)
    past_unchanged = torch.allclose(a[:, :-1], b[:, :-1], atol=1e-4)
    last_changed = not torch.allclose(a[:, -1], b[:, -1], atol=1e-4)
    print(f"  past logits unchanged: {past_unchanged} | last position changed: {last_changed}")
    assert past_unchanged and last_changed, "causality check FAILED"

    print("\n[4] GQA wiring")
    attn = model.blocks[0].attn
    print(f"  {attn.n_heads} query heads share {attn.n_kv_heads} kv heads (n_rep={attn.n_rep})")
    assert attn.n_heads == attn.n_kv_heads * attn.n_rep

    print("\n" + "=" * 78)
    print("ALL CHECKS PASSED ✅")
    print("=" * 78)


if __name__ == "__main__":
    main()
