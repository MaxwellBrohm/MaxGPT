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

import numpy as np

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

    print("\n[8] precision resolver: a card with only EMULATED bf16 (Turing) must get fp16")
    from train import trainer as T
    saved = (torch.cuda.is_available, torch.cuda.is_bf16_supported)
    try:
        torch.cuda.is_available = lambda: True
        def fake_bf16(including_emulation=True):    # what PyTorch reports on a Titan RTX / T4
            return True if including_emulation else False
        torch.cuda.is_bf16_supported = fake_bf16
        assert T.amp_dtype("auto", "cuda") == torch.float16, T.amp_dtype("auto", "cuda")
        assert T.amp_dtype("bf16", "cuda") == torch.float16      # explicit bf16 degrades to fp16 there
        assert T.amp_dtype("fp32", "cuda") is None and T.amp_dtype("auto", "cpu") is None
        torch.cuda.is_bf16_supported = lambda including_emulation=True: True   # an Ampere+ card
        assert T.amp_dtype("auto", "cuda") == torch.bfloat16
    finally:
        torch.cuda.is_available, torch.cuda.is_bf16_supported = saved
    print("  Turing -> fp16, Ampere+ -> bf16, fp32/cpu -> none ✓")

    print("\n[9] Muon / NorMuon + cautious decay: train, checkpoint round trip, param routing")
    from train.muon import Muon
    for kind, extra in (("muon", {}), ("normuon", {"cautious_wd": True})):
        torch.manual_seed(2)
        mm = MaxGPTUltra(cfg)
        tm = Trainer(mm, PackedShardDataset(SHARDS, seq_len), {**base, "optimizer": kind, **extra},
                     device="cpu", out_dir=OUT + "_" + kind, seed=0)
        assert isinstance(tm.optimizer, Muon)
        g = tm.optimizer.param_groups
        n_muon = sum(p.numel() for p in g[0]["params"])
        assert g[0]["use_muon"] and all(p.dim() == 2 for p in g[0]["params"])
        assert not g[1]["use_muon"] and mm.tok_emb.weight in g[1]["params"], "tied embedding must use AdamW"
        assert not g[2]["use_muon"] and all(p.dim() == 1 for p in g[2]["params"])
        first = tm.train_step()["loss"]
        for _ in range(25):
            rec = tm.train_step()
        assert rec["loss"] < first - 0.5, (kind, first, rec["loss"])
        tm.save()
        ref_p = next(p for p in mm.parameters() if p.dim() >= 2).detach().clone()
        st_ref = {k: v.clone() for k, v in tm.optimizer.state[g[0]["params"][0]].items() if torch.is_tensor(v)}
        for _ in range(3):
            tm.train_step()
        assert tm.resume_if_available()
        assert torch.equal(next(p for p in mm.parameters() if p.dim() >= 2).detach(), ref_p)
        st = tm.optimizer.state[g[0]["params"][0]]
        assert all(torch.equal(st[k], v) for k, v in st_ref.items()), "optimizer state not restored"
        print(f"  {kind:<8} loss {first:.3f} -> {rec['loss']:.3f} over 26 steps; {n_muon/1e3:.0f}k params via Muon; "
              f"resume restored weights + optimizer state ✓")

    print("\n[10] architecture tweaks (attn gate + value residual + norm scaling): identity at init,"
          " train, and KV-cache decode == full forward")
    torch.manual_seed(3)
    cfg_t = ModelConfig(vocab_size=tok.vocab_size, d_model=128, n_layers=3, n_heads=4, n_kv_heads=2,
                        mlp_hidden=256, seq_len=seq_len, attn_gate=True, value_residual=True, norm_scaling=True)
    mt = MaxGPTUltra(cfg_t)
    base_sd = {k: v for k, v in mt.state_dict().items()
               if not any(t in k for t in ("attn_gate_proj", "vr_scale", "vr_alpha"))}
    plain = MaxGPTUltra(ModelConfig(vocab_size=tok.vocab_size, d_model=128, n_layers=3, n_heads=4, n_kv_heads=2,
                                    mlp_hidden=256, seq_len=seq_len))
    plain.load_state_dict(base_sd)
    ids = torch.randint(0, tok.vocab_size, (2, seq_len))
    mt.eval(); plain.eval()
    with torch.no_grad():
        # with norm_scaling off the tweaks must be an exact identity at init; check gate + value residual alone
        cfg_id = ModelConfig(vocab_size=tok.vocab_size, d_model=128, n_layers=3, n_heads=4, n_kv_heads=2,
                             mlp_hidden=256, seq_len=seq_len, attn_gate=True, value_residual=True)
        mid = MaxGPTUltra(cfg_id); mid.load_state_dict(base_sd, strict=False); mid.eval()
        assert torch.allclose(mid(ids)[0], plain(ids)[0], atol=1e-5), "gate/value-residual not identity at init"
        full_logits = mt(ids)[0]
        k0 = seq_len // 2
        logits, past = mt(ids[:, :k0], use_cache=True)
        steps = [logits[:, -1]]
        for t in range(k0, seq_len - 1):
            logits, past = mt(ids[:, t:t + 1], past=past, use_cache=True)
            steps.append(logits[:, -1])
        inc = torch.stack(steps, dim=1)                       # predictions for positions k0-1 .. seq_len-2
        assert torch.allclose(inc, full_logits[:, k0 - 1:seq_len - 1], atol=1e-4), "KV-cache decode differs"
    mt.train()
    tt = Trainer(mt, PackedShardDataset(SHARDS, seq_len), base, device="cpu", out_dir=OUT + "_tweaks", seed=0)
    first = tt.train_step()["loss"]
    for _ in range(29):
        rec = tt.train_step()
    assert rec["loss"] < first - 0.5, (first, rec["loss"])
    assert any(n.endswith("attn_gate_proj.weight") for n, _ in mt.named_parameters())
    print(f"  identity at init ✓  cached decode matches ✓  loss {first:.3f} -> {rec['loss']:.3f} ✓")

    print("\n[11] a compiled tier failing in backward: step replays on the next tier, same data, same result")
    torch.manual_seed(5)
    m_ref, m_try = MaxGPTUltra(cfg), MaxGPTUltra(cfg)
    m_try.load_state_dict(m_ref.state_dict())
    t_ref = Trainer(m_ref, PackedShardDataset(SHARDS, seq_len), base, device="cpu", out_dir=OUT + "_ref2", seed=0)
    t_try = Trainer(m_try, PackedShardDataset(SHARDS, seq_len), base, device="cpu", out_dir=OUT + "_try2", seed=0)
    t_ref.train_step()
    calls = {"n": 0}
    real = t_try.net

    class Flaky(torch.nn.Module):                 # stands in for a compiled model whose backward blows up once
        def forward(self, x, y, **kw):
            calls["n"] += 1
            logits, loss = real(x, y, **kw)
            if calls["n"] == 2:                   # fail on the 2nd micro-step, i.e. after grads already accumulated
                return logits, loss * torch.tensor(float("nan")).requires_grad_() + _Boom.apply(loss)
            return logits, loss

    class _Boom(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            return x
        @staticmethod
        def backward(ctx, g):
            raise RuntimeError("accessing tensor output of CUDAGraphs that has been overwritten (simulated)")

    t_try._compile_modes = ["fake-tier"]
    t_try.fwd = Flaky()
    rec = t_try.train_step()
    assert t_try.fwd is t_try.net and not t_try._compile_modes, "did not fall back to eager"
    assert t_try.step == 1 and not rec["diverged"], rec
    assert t_try.data.pos == t_ref.data.pos, (t_try.data.pos, t_ref.data.pos)
    for (n1, p1), (n2, p2) in zip(m_ref.named_parameters(), m_try.named_parameters()):
        assert torch.equal(p1, p2), f"{n1} differs after the replayed step"
    print(f"  backward failure at micro-step 2 -> tier dropped, step replayed from the same data, weights identical ✓")

    print("\n[12] decay-phase annealing blend: off before the start step, exact share after, exact resume")
    import json as _json, shutil
    from data.loader import AnnealBlend
    ANN = SHARDS + "_anneal"
    shutil.rmtree(ANN, ignore_errors=True); os.makedirs(ANN)
    # anneal tokens all live in the top 50 ids of the vocab: a window is recognizably "anneal" (main text never is)
    V = tok.vocab_size
    rows = []
    for i in range(3):
        a = (V - 50 + (np.arange(4000) * 7 + i) % 50).astype(np.uint16); a.tofile(os.path.join(ANN, f"shard_{i:05d}.bin"))
        rows.append({"name": f"shard_{i:05d}.bin", "tokens": int(len(a))})
    _json.dump({"dtype": "uint16", "eot_id": 1, "shards": rows, "total_tokens": 12000}, open(os.path.join(ANN, "meta.json"), "w"))
    is_anneal = lambda w: bool((w >= V - 50).float().mean() > 0.9)
    cfg_a = {**base, "micro_batch": 4, "grad_accum": 2, "log_every": 1000,
             "anneal": {"shards": ANN, "frac": 0.5, "ramp_tokens": 0, "start": 3}}     # B = 8 windows/step, A = 4 from step 3
    torch.manual_seed(7)
    m_plain, m_ann = MaxGPTUltra(cfg), MaxGPTUltra(cfg)
    m_ann.load_state_dict(m_plain.state_dict())
    t_plain = Trainer(m_plain, PackedShardDataset(SHARDS, seq_len), {**cfg_a, "anneal": {}}, device="cpu", out_dir=OUT + "_noann", seed=0)
    t_ann = Trainer(m_ann, PackedShardDataset(SHARDS, seq_len), cfg_a, device="cpu", out_dir=OUT + "_ann", seed=0)
    assert isinstance(t_ann.data, AnnealBlend) and t_ann.data.start_step == 3
    seen = {"before": [], "after": []}
    orig_nb = t_ann.data.next_batch
    def spy(bs, device="cpu"):
        x, y = orig_nb(bs, device)
        seen["after" if t_ann.step >= 3 else "before"].extend(is_anneal(w) for w in x)
        return x, y
    t_ann.data.next_batch = spy
    for _ in range(3):
        t_plain.train_step(); t_ann.train_step()
    for (n1, p1), (n2, p2) in zip(m_plain.named_parameters(), m_ann.named_parameters()):
        assert torch.equal(p1, p2), f"{n1}: annealing changed training BEFORE its start step"
    assert not any(seen["before"]) and len(seen["before"]) == 24, seen["before"]
    for _ in range(4):
        rec = t_ann.train_step()
    assert rec.get("anneal_frac") == 0.5, rec
    assert sum(seen["after"]) == 16 and len(seen["after"]) == 32, (sum(seen["after"]), len(seen["after"]))   # 4 of every 8
    # 3 plain steps (8 main windows each) + 4 annealed steps (4 main + 4 anneal windows each)
    assert t_ann.data.main.pos == (3 * 8 + 4 * 4) * seq_len, (t_ann.data.main.pos, seq_len)
    assert t_ann.data.anneal.pos == 4 * 4 * seq_len, (t_ann.data.anneal.pos, seq_len)
    print(f"  bit-identical to no-anneal for 3 steps, then 4 of 8 windows per step from the anneal stream ✓")
    # exact resume, and a pre-annealing (main-only) state still loads
    t_ann.data.next_batch = orig_nb
    t_ann.save()
    st = t_ann.data.state_dict(); assert st["blend"] and st["block_i"] == 0
    x_ref, _ = t_ann.data.next_batch(4)
    t_ann.data.load_state_dict(st)
    x_again, _ = t_ann.data.next_batch(4)
    assert torch.equal(x_ref, x_again), "resume did not reproduce the next batch"
    t_ann.data.load_state_dict({"pos": 0, "epoch": 0})
    assert t_ann.data.main.pos == 0 and t_ann.data.state_dict()["block_i"] == 0
    print("  blend state round-trips; an old main-only checkpoint state loads ✓")

    print("\n[13] weights-only snapshots: only in the decay phase, on the cadence, with their own retention")
    import glob as _glob
    cfg_w = {**base, "micro_batch": 4, "grad_accum": 2, "total_tokens": 128 * 20, "decay_frac": 0.5,   # 20 steps, decay from step 10
             "decay_weights_every": 2, "decay_weights_keep": 3, "log_every": 1000}
    torch.manual_seed(9)
    mw = MaxGPTUltra(cfg)
    tw = Trainer(mw, PackedShardDataset(SHARDS, seq_len), cfg_w, device="cpu", out_dir=OUT + "_wts", seed=0)
    shutil.rmtree(os.path.join(OUT + "_wts", "checkpoints"), ignore_errors=True); os.makedirs(os.path.join(OUT + "_wts", "checkpoints"))
    tw.train(max_steps=9)
    assert not _glob.glob(os.path.join(OUT + "_wts", "checkpoints", "weights_*.pt")), "snapshot written before the decay"
    tw.train(max_steps=11)                                            # steps 10..20: due at 10,12,...,20 -> keep the last 3
    names = sorted(os.path.basename(p) for p in _glob.glob(os.path.join(OUT + "_wts", "checkpoints", "weights_*.pt")))
    assert names == ["weights_00000016.pt", "weights_00000018.pt", "weights_00000020.pt"], names
    w = torch.load(os.path.join(OUT + "_wts", "checkpoints", "weights_00000020.pt"), weights_only=False)
    assert "optimizer" not in w and w["step"] == 20 and set(w["model"]) == set(mw.state_dict())
    print(f"  none before step 10; after step 20 the last 3 of {{10..20 step 2}} remain: {names} (weights only) ✓")

    print("\n" + "=" * 72)
    print("ALL CHECKS PASSED ✅")
    print("=" * 72)


if __name__ == "__main__":
    main()
