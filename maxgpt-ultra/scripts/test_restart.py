"""Tests for the November 2026 restart features (docs/research_2026-11-18_restart.md).

fp32 attention per layer, attention-logit telemetry, Peri-LN, the ReLU-squared MLP, embedding
weight decay, the linear decay shape, milestone and decay-start checkpoints, the DONE marker, and
the gradient watch. CPU only, tiny models.

Run from maxgpt-ultra/:  ../venv/bin/python scripts/test_restart.py
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # maxgpt-ultra/

import torch

import model.model as MM
from model import ModelConfig, MaxGPTUltra
from tokenizer.tokenizer import train_tokenizer, UltraTokenizer
from data.prepare import tokenize_to_shards
from data.loader import PackedShardDataset
from train.schedule import wsd_lr
from train.trainer import Trainer, make_optimizer
from train.gradwatch import GradWatch

TOK = "/tmp/maxgpt_ultra_restart_tok.json"
SHARDS = "/tmp/maxgpt_ultra_restart_shards"
OUT = "/tmp/maxgpt_ultra_restart_run"
DOCS = ["the quick brown fox jumps over the lazy dog near the quiet river bank",
        "a small model learns to predict the next token from the tokens before it",
        "def add(a, b):\n    return a + b\n\nprint(add(2, 3))",
        "seven times eight is fifty six and nine times nine is eighty one"] * 80


def small(**kw):
    base = dict(vocab_size=300, d_model=64, n_layers=3, n_heads=4, n_kv_heads=2, mlp_hidden=128, seq_len=16)
    base.update(kw)
    return ModelConfig(**base)


def main() -> None:
    print("=" * 72)
    print("MaxGPT-Ultra restart-feature tests")
    print("=" * 72)
    torch.manual_seed(0)
    x = torch.randint(0, 300, (2, 16))

    print("\n[1] fp32 attention switch: only the listed layers leave bf16, results unchanged in fp32, mask honored")
    m_sw = MaxGPTUltra(small(attn_fp32_layers=(1,)))
    m_plain = MaxGPTUltra(small())
    m_plain.load_state_dict(m_sw.state_dict())
    with torch.no_grad():
        assert torch.equal(m_sw(x)[0], m_plain(x)[0]), "in plain fp32 the switch must not change anything"
    seen, real = [], MM.F.scaled_dot_product_attention

    def autocast_on() -> bool:
        try:
            return bool(torch.is_autocast_enabled("cpu"))
        except TypeError:                             # older torch
            return bool(torch.is_autocast_cpu_enabled())

    def spy(q, k, v, **kw):                       # record how attention was called, then run it in fp32
        seen.append((autocast_on(), {q.dtype, k.dtype, v.dtype} == {torch.float32}))
        return real(q.float(), k.float(), v.float(), **kw).to(v.dtype)

    MM.F.scaled_dot_product_attention = spy
    try:
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            m_sw(x)
    finally:
        MM.F.scaled_dot_product_attention = real
    # layers 0 and 2: called with autocast on (the fused kernel then runs in bf16);
    # layer 1: autocast switched off around the call and every input already fp32
    assert [a for a, _ in seen] == [True, False, True] and seen[1][1], seen
    m_all = MaxGPTUltra(small(attn_fp32_layers=(0, 1, 2)))
    m_all.doc_mask, m_all.doc_sep_id = True, 7
    xa = x.clone(); xa[0, 5] = 7                  # a document separator at position 5 of row 0
    xb = xa.clone(); xb[0, :5] = (xb[0, :5] + 11) % 300; xb[0, :5][xb[0, :5] == 7] = 8   # rewrite document A only
    with torch.no_grad():
        assert torch.equal(m_all(xa)[0][0, 6:], m_all(xb)[0][0, 6:]), "the fp32 path ignored the document mask"
    print(f"  autocast active at the attention call, per layer: {[a for a, _ in seen]}; layer 1 gets fp32 inputs ✓")

    print("\n[2] attention-logit telemetry: the causal maximum, computed from the model's own queries and keys")
    att = MaxGPTUltra(small(qk_norm=False)).blocks[0].attn
    full = MaxGPTUltra(small(qk_norm=False))
    cos, sin = full.rope_cos[:16], full.rope_sin[:16]
    found = None
    for seed in range(60):                        # a fixture where the largest score is one causal attention never sees
        g = torch.Generator().manual_seed(seed)
        h = torch.randn(1, 16, 64, generator=g)
        with torch.no_grad():
            q, k = att._qk(h, cos, sin)
            s = (q @ MM.repeat_kv(k, att.n_rep).transpose(-1, -2)).abs() / (att.head_dim ** 0.5)
        causal = torch.ones(16, 16, dtype=torch.bool).tril()
        cmax, umax = float(s.masked_fill(~causal, 0).max()), float(s.max())
        if umax > 1.05 * cmax:
            found = (h, cmax, umax)
            break
    assert found, "no fixture with a non-causal maximum found"
    h, cmax, umax = found
    got = att.max_logit(h, cos, sin)
    assert abs(got - cmax) < 1e-5 and got < umax, (got, cmax, umax)
    with torch.no_grad():
        base_no = att.max_logit(h, cos, sin)
        att.q_proj.weight.mul_(5.0)
        assert abs(att.max_logit(h, cos, sin) - 5 * base_no) < 1e-3 * base_no, "without QK-norm the score scales with the weights"
    att_n = MaxGPTUltra(small()).blocks[0].attn           # with QK-norm the score ignores the weight scale
    with torch.no_grad():
        b1 = att_n.max_logit(h, cos, sin)
        att_n.q_proj.weight.mul_(5.0)
        assert abs(att_n.max_logit(h, cos, sin) - b1) < 1e-2 * b1, "QK-norm must bound the score (5x weights, under 1% change)"
    print(f"  causal max {cmax:.3f} reported (the unmasked max was {umax:.3f}); QK-norm makes it scale-free ✓")

    print("\n[3] Peri-LN: a norm on every sublayer output and on the embeddings, same answers with the cache")
    cfg_p = small(peri_ln=True, mlp="relu2", mlp_hidden=192)
    m_p = MaxGPTUltra(cfg_p)
    m_0 = MaxGPTUltra(small(mlp="relu2", mlp_hidden=192))
    n_p, n_0 = sum(p.numel() for p in m_p.parameters()), sum(p.numel() for p in m_0.parameters())
    assert n_p - n_0 == 2 * 3 * 64 + 64, (n_p, n_0)
    rms, hooks = [], []
    for b in m_p.blocks:
        for norm in (b.attn_out_norm, b.mlp_out_norm):
            hooks.append(norm.register_forward_hook(lambda mod, i, o: rms.append(float(o.pow(2).mean(-1).sqrt().mean()))))
    emb_calls = []
    hooks.append(m_p.emb_norm.register_forward_hook(lambda mod, i, o: emb_calls.append(1)))
    with torch.no_grad():
        full_logits = m_p(x)[0]
    for hk in hooks:
        hk.remove()
    # 6 calls (attention and MLP output in each of 3 blocks). At init the scaled-down output projections
    # make the sublayer outputs tiny, so the norm's epsilon keeps their RMS a little under 1; a normal-sized
    # input comes out at exactly unit RMS.
    assert len(rms) == 6 and all(0.5 < r <= 1.0 + 1e-3 for r in rms), rms
    assert len(emb_calls) == 1
    with torch.no_grad():
        unit = m_p.blocks[0].mlp_out_norm(3.0 * torch.randn(2, 5, 64))
    assert abs(float(unit.pow(2).mean(-1).sqrt().mean()) - 1.0) < 1e-3
    with torch.no_grad():
        pre, cache = m_p(x[:, :10], use_cache=True)
        nxt, _ = m_p(x[:, 10:11], past=cache, use_cache=True)
    assert torch.allclose(pre, full_logits[:, :10], atol=1e-4) and torch.allclose(nxt[:, 0], full_logits[:, 10], atol=1e-4)
    print(f"  +{n_p - n_0} parameters; 6 sublayer outputs normalized (RMS {min(rms):.2f} to {max(rms):.2f} at init); cached decoding matches the full pass ✓")

    print("\n[4] ReLU-squared MLP: two matrices, relu(x)^2, selectable from the YAML")
    mlp = m_p.blocks[0].mlp
    assert isinstance(mlp, MM.ReLU2MLP) and sum(p.numel() for p in mlp.parameters()) == 2 * 64 * 192
    hh = torch.randn(2, 5, 64)
    with torch.no_grad():
        assert torch.allclose(mlp(hh), mlp.down_proj(torch.relu(mlp.up_proj(hh)) ** 2), atol=1e-6)
        assert not torch.allclose(mlp(hh), mlp.down_proj(torch.relu(mlp.up_proj(hh))), atol=1e-4), "sanity: the square matters"
    d = tempfile.mkdtemp(prefix="mgu_cfg_")
    with open(os.path.join(d, "r.yaml"), "w") as f:
        f.write("model:\n  d_model: 64\n  n_heads: 4\n  n_kv_heads: 2\n  mlp: relu2\n  mlp_hidden: 192\n  peri_ln: true\n  attn_fp32_layers: [0, 2]\n")
    c = ModelConfig.from_yaml(os.path.join(d, "r.yaml"))
    assert c.mlp == "relu2" and c.peri_ln is True and c.attn_fp32_layers == (0, 2), c
    with open(os.path.join(d, "bad.yaml"), "w") as f:
        f.write("model:\n  mlp: gelu\n")
    try:
        ModelConfig.from_yaml(os.path.join(d, "bad.yaml")); raise SystemExit("an unknown mlp must be rejected")
    except AssertionError:
        pass
    assert isinstance(MaxGPTUltra(small()).blocks[0].mlp, MM.SwiGLU), "the default stays SwiGLU"
    print("  output matches down(relu(up(x))^2); YAML selects it; unknown names are rejected ✓")

    print("\n[5] embedding weight decay: its own setting, for AdamW and for NorMuon")
    for kind in ("adamw", "normuon"):
        shrink = {}
        for embed_wd in (None, 0.0):
            mm = MaxGPTUltra(small())
            opt = make_optimizer(mm, 1e-2, (0.9, 0.95), 0.1, kind=kind, embed_wd=embed_wd)
            eg = [g for g in opt.param_groups if any(p is mm.tok_emb.weight for p in g["params"])]
            assert len(eg) == 1 and eg[0]["weight_decay"] == (0.1 if embed_wd is None else 0.0), (kind, embed_wd, eg[0]["weight_decay"])
            assert all(g["weight_decay"] == 0.1 for g in opt.param_groups
                       if any(p.dim() >= 2 and p is not mm.tok_emb.weight for p in g["params"])), "hidden matrices keep their decay"
            before = float(mm.tok_emb.weight.norm())
            for p in mm.parameters():
                p.grad = torch.zeros_like(p)      # no gradient: any change to the embedding is weight decay
            opt.step()
            shrink[embed_wd] = before - float(mm.tok_emb.weight.norm())
        assert shrink[None] > 1e-6 and abs(shrink[0.0]) < 1e-9, (kind, shrink)
    print("  default decays the embedding; embed_weight_decay 0 leaves it untouched; hidden matrices unaffected ✓")

    print("\n[6] linear decay shape")
    kw = dict(total_steps=1000, warmup_steps=10, decay_frac=0.2, max_lr=1.0)
    assert abs(wsd_lr(799, **kw, decay_shape="linear") - 1.0) < 1e-9 and abs(wsd_lr(850, **kw, decay_shape="linear") - 0.75) < 1e-9
    assert abs(wsd_lr(900, **kw, decay_shape="linear") - 0.5) < 1e-9 and wsd_lr(999, **kw, decay_shape="linear") < 0.006
    print("  1.0 until the decay starts, 0.75 a quarter in, 0.5 halfway, about 0 at the end ✓")

    print("\n[7] permanent checkpoints (milestones, decay start) survive pruning; DONE only at the true end")
    train_tokenizer(iter(DOCS), vocab_size=600, out_path=TOK)
    tok = UltraTokenizer(TOK)
    shutil.rmtree(SHARDS, ignore_errors=True)
    tokenize_to_shards(DOCS, tok, SHARDS, shard_size=2048)
    cfg7 = ModelConfig(vocab_size=tok.vocab_size, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2, mlp_hidden=128, seq_len=16)
    tps = 4 * 1 * 16
    tcfg = {"micro_batch": 4, "grad_accum": 1, "total_tokens": tps * 20, "warmup_tokens": tps * 2, "lr": 1e-3,
            "decay_frac": 0.2, "decay_shape": "linear", "z_loss": 0.0, "autosave_minutes": 9999, "log_every": 1,
            "keep_last_k": 1, "milestone_tokens": tps * 5, "decay_weights_every": 0}
    shutil.rmtree(OUT, ignore_errors=True)
    t7 = Trainer(MaxGPTUltra(cfg7), PackedShardDataset(SHARDS, 16), tcfg, device="cpu", out_dir=OUT, seed=0)
    assert t7.decay_start_step == 16 and t7.milestone_steps == 5
    t7.train(max_steps=12)
    ck = os.path.join(OUT, "checkpoints")
    names = lambda: sorted(os.path.basename(p) for p in glob.glob(os.path.join(ck, "*.pt")))
    assert not os.path.exists(os.path.join(OUT, "DONE")), "DONE must not appear when the run merely stopped early"
    assert "milestone_00000005.pt" in names() and "milestone_00000010.pt" in names(), names()
    assert len([n for n in names() if n.startswith("ckpt_")]) == 1, names()
    t7.train()
    done = json.load(open(os.path.join(OUT, "DONE")))
    assert done == {"step": 20, "total_steps": 20}, done
    for n in ("milestone_00000015.pt", "milestone_00000020.pt", "decay_start.pt"):
        assert n in names(), (n, names())
    assert int(torch.load(os.path.join(ck, "decay_start.pt"), map_location="cpu", weights_only=False)["step"]) == 16
    assert int(torch.load(os.path.join(ck, "milestone_00000005.pt"), map_location="cpu", weights_only=False)["step"]) == 5
    print(f"  kept: {[n for n in names() if not n.startswith('ckpt_')]}; one rolling checkpoint; DONE = {done} ✓")

    print("\n[8] gradient watch: steady is quiet, a sustained rise alerts once, a slow rise is still caught, a huge one halts")
    gw = GradWatch(window=100, min_baseline=20, alert_patience=5, halt_patience=30)
    g0 = torch.Generator().manual_seed(0)
    assert all(gw.feed(1.0 + 0.1 * float(torch.rand((), generator=g0))) is None for _ in range(300)), "steady norms must stay quiet"
    assert all(gw.feed(50.0) is None for _ in range(4)), "a spike shorter than the patience is not an alert"
    assert gw.feed(1.0) is None
    out = [gw.feed(20.0) for _ in range(12)]
    assert out.count("alert") == 1 and out.index("alert") == 4 and out.count("halt") == 0, out
    assert gw.feed(1.0) is None and [gw.feed(20.0) for _ in range(5)][-1] == "alert", "the alert re-arms after the norm recovers"
    slow = GradWatch(window=100, min_baseline=20, alert_patience=5, halt_factor=0)
    for _ in range(30):
        slow.feed(1.0)
    g, fired = 1.0, None
    for i in range(400):                           # 3% a step: any moving median keeps up with this
        g *= 1.03
        if slow.feed(g) == "alert":
            fired = (i, g)
            break
    assert fired and 10.0 < fired[1] < 14.0, f"a slow rise must alert soon after 10x the floor, got {fired}"
    big = GradWatch(window=100, min_baseline=20, alert_patience=5, halt_patience=30)
    for _ in range(30):
        big.feed(1.0)
    res = [big.feed(500.0) for _ in range(40)]
    assert res.count("halt") == 1 and res.index("halt") == 29 and res.count("alert") == 1, res
    mp = os.path.join(d, "metrics.jsonl")
    with open(mp, "w") as f:
        for i in range(120):
            f.write(json.dumps({"step": i, "loss": 2.0, "grad_norm": 0.5}) + "\n")
        f.write(json.dumps({"step": 120, "event": "eval", "grad_norm": 99.0}) + "\n")
        f.write(json.dumps({"step": 121, "loss": 9.0, "grad_norm": 77.0, "diverged": True}) + "\n")
    seeded = GradWatch(alert_patience=3)
    assert seeded.seed_from_metrics(mp) == 120 and abs(seeded.floor - 0.5) < 1e-9
    assert [seeded.feed(40.0) for _ in range(3)][-1] == "alert", "a restarted run must judge against the earlier level at once"
    print(f"  quiet when steady; alert at the 5th high step; slow rise caught at {fired[1]:.1f}x; halt at the 30th; reseeds from metrics ✓")

    print("\n[9] a halt stops the run cleanly and leaves GRAD_HALT for the job chain")
    out9 = OUT + "_halt"
    shutil.rmtree(out9, ignore_errors=True)
    tcfg9 = {**tcfg, "total_tokens": tps * 120, "milestone_tokens": 0, "grad_alert_patience": 3, "grad_halt_patience": 6}
    t9 = Trainer(MaxGPTUltra(cfg7), PackedShardDataset(SHARDS, 16), tcfg9, device="cpu", out_dir=out9, seed=0)
    real_step = t9.train_step

    def fake_step():                                # the gradient norm explodes from step 70 on
        rec = real_step()
        if rec["step"] >= 70:
            rec["grad_norm"] = 1e6
        return rec

    t9.train_step = fake_step
    t9.train()
    assert os.path.exists(os.path.join(out9, "GRAD_HALT")) and not os.path.exists(os.path.join(out9, "DONE"))
    assert 70 <= t9.step < 90, f"the run should stop a few steps after the blow-up, stopped at {t9.step}"
    events = [json.loads(l).get("event") for l in open(os.path.join(out9, "metrics.jsonl"))]
    assert "grad_alert" in events and "grad_halt" in events and events.index("grad_alert") < events.index("grad_halt")
    print(f"  stopped at step {t9.step} of 120 with grad_alert then grad_halt logged; GRAD_HALT written, no DONE ✓")

    print("\n[10] the decay phase never starts without its data: missing anneal shards stop the run there")
    out10 = OUT + "_noanneal"
    shutil.rmtree(out10, ignore_errors=True)
    tcfg10 = {**tcfg, "milestone_tokens": 0, "anneal": {"shards": "/tmp/maxgpt_no_such_anneal_shards", "frac": 0.4}}
    t10 = Trainer(MaxGPTUltra(cfg7), PackedShardDataset(SHARDS, 16), tcfg10, device="cpu", out_dir=out10, seed=0)
    t10.train()
    assert t10.step == t10.decay_start_step == 16, (t10.step, t10.decay_start_step)
    assert os.path.exists(os.path.join(out10, "ANNEAL_MISSING")) and not os.path.exists(os.path.join(out10, "DONE"))
    assert json.load(open(os.path.join(out10, "ANNEAL_MISSING")))["step"] == 16
    out10b = OUT + "_optanneal"
    shutil.rmtree(out10b, ignore_errors=True)
    tcfg10b = {**tcfg10, "anneal": {**tcfg10["anneal"], "optional": True}}
    t10b = Trainer(MaxGPTUltra(cfg7), PackedShardDataset(SHARDS, 16), tcfg10b, device="cpu", out_dir=out10b, seed=0)
    t10b.train()
    assert t10b.step == 20 and os.path.exists(os.path.join(out10b, "DONE")) and not os.path.exists(os.path.join(out10b, "ANNEAL_MISSING"))
    print("  stopped at step 16 of 20 with ANNEAL_MISSING and no DONE; anneal.optional lets a run finish without it ✓")

    print("\n[11] --anneal-data: the job script says where the decay-phase shards are, the config only that they are used")
    from scripts.train import with_anneal_data
    t11 = {"lr": 1.0, "anneal": {"shards": "data/shards_anneal", "frac": 0.4}}
    moved = with_anneal_data(t11, "/big/shards_anneal")
    assert moved["anneal"] == {"shards": "/big/shards_anneal", "frac": 0.4} and moved["lr"] == 1.0
    assert t11["anneal"]["shards"] == "data/shards_anneal", "the loaded config must not be edited in place"
    assert with_anneal_data(t11, None) == t11
    assert with_anneal_data({"lr": 1.0}, "/big/shards_anneal") == {"lr": 1.0}, "the flag must not switch annealing on"
    anneal11 = SHARDS + "_anneal"
    shutil.rmtree(anneal11, ignore_errors=True)
    tokenize_to_shards(DOCS[::-1], tok, anneal11, shard_size=2048)
    cfg11 = os.path.join(d, "tiny_anneal.yaml")          # JSON is valid YAML
    with open(cfg11, "w") as f:
        json.dump({"model": {"vocab_size": tok.vocab_size, "d_model": 64, "n_layers": 2, "n_heads": 4, "n_kv_heads": 2,
                             "mlp_hidden": 128, "seq_len": 16},
                   "train": {**tcfg, "milestone_tokens": 0, "compile": False,
                             "anneal": {"shards": "/tmp/maxgpt_no_such_anneal_shards", "frac": 0.4, "ramp_tokens": tps * 2}}}, f)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def launch(out, *extra):                              # the real entry point, as the job script runs it
        shutil.rmtree(out, ignore_errors=True)
        r = subprocess.run([sys.executable, os.path.join(root, "scripts", "train.py"), "--config", cfg11, "--data", SHARDS,
                            "--out", out, "--device", "cpu", *extra], cwd=root, capture_output=True, text=True, timeout=600)
        assert r.returncode == 0, (r.stdout + r.stderr)[-2000:]
        rows = [json.loads(l) for l in open(os.path.join(out, "metrics.jsonl"))]
        return r.stdout, max((x.get("anneal_frac", 0.0) for x in rows), default=0.0)

    out11 = OUT + "_cli_moved"
    text, top = launch(out11, "--anneal-data", anneal11)
    assert os.path.exists(os.path.join(out11, "DONE")) and not os.path.exists(os.path.join(out11, "ANNEAL_MISSING")), text[-1500:]
    assert f"annealing: {anneal11}" in text and abs(top - 0.4) < 1e-9, (top, text[-1500:])
    out11b = OUT + "_cli_config_path"
    text, top = launch(out11b)
    assert os.path.exists(os.path.join(out11b, "ANNEAL_MISSING")) and not os.path.exists(os.path.join(out11b, "DONE")), text[-1500:]
    assert top == 0.0 and json.load(open(os.path.join(out11b, "ANNEAL_MISSING")))["shards"] == "/tmp/maxgpt_no_such_anneal_shards"
    print("  with the flag the run blends the moved shards at 40% and finishes; without it the config's missing path stops it ✓")

    print("\n" + "=" * 72)
    print("ALL CHECKS PASSED ✅")
    print("=" * 72)


if __name__ == "__main__":
    main()
