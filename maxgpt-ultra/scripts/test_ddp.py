"""Multi-GPU (DDP) correctness test, on CPU with 2 processes (gloo), no GPU needed.

The claim under test: training on N ranks is the SAME optimization as training on one.
So we run the tiny model single-process, then under `torchrun --nproc_per_node=2` with the
same config, and compare the final weights. Also checked: the data loaders hand each rank a
disjoint slice whose union is exactly what one process would have read; fp16 loss scaling
under DDP matches fp32 DDP bit for bit; the sharded DPO reference precompute matches the
single-process one; a GUI pause (stop file seen by rank 0 only) stops every rank cleanly.

Run from maxgpt-ultra/:  ../venv/bin/python scripts/test_ddp.py
"""
import copy
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # maxgpt-ultra/
sys.path.insert(0, ROOT)

import torch

from model import ModelConfig, MaxGPTUltra
from tokenizer.tokenizer import train_tokenizer, UltraTokenizer
from data.prepare import tokenize_to_shards
from data.loader import PackedShardDataset
from posttrain.sft_data import SFTDataset
from posttrain.dpo import DPODataset, DPOTrainer
from train.trainer import Trainer
from train import dist as D

TOK = "/tmp/maxgpt_ultra_ddp_tok.json"
SHARDS = "/tmp/maxgpt_ultra_ddp_shards"
OUT = "/tmp/maxgpt_ultra_ddp"
SEQ = 16
STEPS = 6
DOCS = [
    "The cat sat on the mat while the dog ran in the yard near the old red barn.",
    "Numbers like 7 and 42 and 100 show up when we count things in the world.",
    "def square(n):\n    return n * n\n\nprint(square(9))",
    "Attention lets each token look at the others; that is the core transformer idea.",
    "say a b x y the assistant answers the user with a short reply yes no good",
] * 80
PREFS = [
    {"prompt": [{"role": "user", "content": "say a"}], "chosen": "a", "rejected": "b"},
    {"prompt": [{"role": "user", "content": "say x"}], "chosen": "x", "rejected": "y"},
] * 40
CHATS = [{"messages": [{"role": "user", "content": "say a"}, {"role": "assistant", "content": "a b x y"}]},
         {"messages": [{"role": "user", "content": "say x"}, {"role": "assistant", "content": "x y a b"}]}] * 30
TCFG = {"micro_batch": 2, "grad_accum": 4, "total_tokens": SEQ * 2 * 4 * 400, "warmup_tokens": SEQ * 2 * 4 * 5, "rank_shares_auto": False,
        "lr": 3e-3, "decay_frac": 0.2, "grad_clip": 1.0, "z_loss": 1e-4, "autosave_minutes": 9999,
        "log_every": 2, "keep_last_k": 1}
DPO_TCFG = {"batch_size": 4, "grad_accum": 2, "total_steps": 60, "warmup_steps": 2, "lr": 5e-4, "decay_frac": 0.1,
            "weight_decay": 0.0, "autosave_minutes": 9999, "log_every": 1000, "precompute_ref": True, "compile": False}


def build_model(vocab):
    torch.manual_seed(123)          # identical init in every process
    return MaxGPTUltra(ModelConfig(vocab_size=vocab, d_model=128, n_layers=2, n_heads=4,
                                   n_kv_heads=2, mlp_hidden=256, seq_len=SEQ))


def run_pretrain(out, precision, stop_file=None, optimizer="adamw", extra=None):
    tok = UltraTokenizer(TOK)
    model = build_model(tok.vocab_size)
    tr = Trainer(model, PackedShardDataset(SHARDS, SEQ), {**TCFG, "precision": precision, "optimizer": optimizer, **(extra or {})},
                 device="cpu", out_dir=out, seed=0, stop_file=stop_file)
    tr.train(max_steps=STEPS)
    return {k: v.detach().clone() for k, v in model.state_dict().items()}, tr.step


def run_dpo(out):
    tok = UltraTokenizer(TOK)
    policy = build_model(tok.vocab_size)
    ref = copy.deepcopy(policy)
    data = DPODataset(PREFS, tok, SEQ)
    tr = DPOTrainer(policy, ref, data, DPO_TCFG, "cpu", out, beta=0.1, seed=0)
    tr.train(max_steps=8)
    return ({k: v.detach().clone() for k, v in policy.state_dict().items()},
            torch.from_numpy(data.ref_c.copy()), torch.from_numpy(data.ref_r.copy()), tr.step)


def worker(mode: str) -> None:
    """Runs inside each torchrun process."""
    info = D.init_distributed("cpu")     # CPU ranks even on a GPU box (gloo)
    out = os.path.join(OUT, mode)
    if mode == "pause":
        stop = os.path.join(out, "STOP")
        if info["rank"] == 0:            # only rank 0 sees the pause request, as with the GUI
            os.makedirs(out, exist_ok=True)
            open(stop, "w").close()
        _, step = run_pretrain(out, "fp32", stop_file=stop)
        res = {"step": step}
    elif mode.startswith("pretrain-"):
        prec, _, opt = mode.split("-", 1)[1].partition("-")     # e.g. pretrain-fp32, pretrain-fp16-normuon, pretrain-fp32-shares
        extra = {"rank_shares": [3, 1]} if opt == "shares" else None    # unequal shares of the 4 micro-steps
        if opt == "auto":                                                   # auto-balance, re-deal every 2 steps
            extra = {"rank_shares_auto": True, "rank_shares_every": 2, "rank_shares_smooth": 1.0, "log_every": 1}
            os.environ["MAXGPT_TEST_SLOW_RANK"] = "1"
            os.environ["MAXGPT_TEST_SLOW_SECS"] = "0.15"
        w, step = run_pretrain(out, prec, optimizer=("adamw" if opt in ("", "shares", "auto") else opt), extra=extra)
        res = {"weights": w, "step": step}
    elif mode == "dpo":
        w, rc, rr, step = run_dpo(out)
        res = {"weights": w, "ref_c": rc, "ref_r": rr, "step": step}
    else:
        raise SystemExit(f"unknown worker mode {mode}")
    torch.save(res, os.path.join(out, f"rank{info['rank']}.pt"))
    D.cleanup()


def torchrun(mode: str) -> str:
    """Launch 2 workers. On Linux: real torchrun with the same flags the GUI server uses. On
    macOS torchrun's launcher stalls resolving the hostname on some networks, so there the
    workers get the same RANK/WORLD_SIZE/MASTER_* environment torchrun would set, directly
    (identical code path inside the trainer). Returns rank 0's combined output."""
    out = os.path.join(OUT, mode)
    os.makedirs(out, exist_ok=True)
    for f in ("rank0.pt", "rank1.pt", "metrics.jsonl"):
        try:
            os.remove(os.path.join(out, f))
        except OSError:
            pass
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    launcher = os.environ.get("MAXGPT_TEST_LAUNCHER", "env" if sys.platform == "darwin" else "torchrun")
    if launcher == "torchrun":
        cmd = [sys.executable, "-m", "torch.distributed.run", "--nnodes=1", "--rdzv-backend=c10d",
               "--rdzv-endpoint=127.0.0.1:0", "--nproc_per_node=2",
               "--redirects", "3", "--tee", "0:3", "--log-dir", os.path.join(out, "ranks"),
               os.path.abspath(__file__), "--worker", mode]
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, timeout=600, env=env)
        text = r.stdout + r.stderr
        assert r.returncode == 0, f"torchrun {mode} failed (rc={r.returncode}):\n{text[-3000:]}"
        return text
    import socket
    with socket.socket() as sk:
        sk.bind(("127.0.0.1", 0))
        port = sk.getsockname()[1]
    procs = []
    for rank in (0, 1):
        e = {**env, "RANK": str(rank), "LOCAL_RANK": str(rank), "WORLD_SIZE": "2",
             "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(port)}
        if sys.platform == "darwin":
            e["GLOO_SOCKET_IFNAME"] = "lo0"
        procs.append(subprocess.Popen([sys.executable, os.path.abspath(__file__), "--worker", mode],
                                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                      cwd=ROOT, env=e))
    outs = []
    for pr in procs:
        try:
            outs.append(pr.communicate(timeout=600)[0])
        except subprocess.TimeoutExpired:
            for q in procs:
                q.kill()
            raise AssertionError(f"{mode}: a rank hung (deadlock between ranks?)")
    for rank, (pr, o) in enumerate(zip(procs, outs)):
        assert pr.returncode == 0, f"{mode} rank {rank} failed (rc={pr.returncode}):\n{o[-3000:]}"
    return outs[0]


def load_ranks(mode):
    return [torch.load(os.path.join(OUT, mode, f"rank{r}.pt"), weights_only=False) for r in (0, 1)]


def same_weights(a, b, exact=False, atol=1e-5):
    for k in a:
        if exact:
            assert torch.equal(a[k], b[k]), f"{k} differs"
        else:
            assert torch.allclose(a[k], b[k], atol=atol, rtol=1e-4), \
                f"{k} differs (max abs {float((a[k] - b[k]).abs().max()):.2e})"


def rows(x):
    return sorted(tuple(r.tolist()) for r in x)


def main() -> None:
    if len(sys.argv) >= 3 and sys.argv[1] == "--worker":
        worker(sys.argv[2])
        return

    print("=" * 72)
    print("MaxGPT-Ultra multi-GPU (DDP) test: 2 CPU processes vs 1")
    print("=" * 72)
    print("\n[setup] tiny tokenizer + shards (fresh: a stale run dir would hand DPO a cached reference table)")
    import shutil
    shutil.rmtree(OUT, ignore_errors=True)
    train_tokenizer(iter(DOCS), vocab_size=1200, out_path=TOK)
    tok = UltraTokenizer(TOK)
    tokenize_to_shards(DOCS, tok, SHARDS, shard_size=1024)

    print("\n[1] loaders: two ranks read disjoint slices whose union is the single-process read")
    single = PackedShardDataset(SHARDS, SEQ)
    r0, r1 = PackedShardDataset(SHARDS, SEQ), PackedShardDataset(SHARDS, SEQ)
    r0.shard(0, 2)
    r1.shard(1, 2)
    one = [single.next_batch(2)[0] for _ in range(4)]                     # 4 micro-steps of 2
    two = [d.next_batch(2)[0] for d in (r0, r1) for _ in range(2)]        # 2 ranks x 2 micro-steps of 2
    assert rows(torch.cat(one)) == rows(torch.cat(two)), "packed loader: rank slices != single read"
    assert single.pos == r0.pos == r1.pos, (single.pos, r0.pos, r1.pos)
    print(f"  PackedShardDataset: 8 windows, same set, pos {single.pos} on every rank ✓")
    s1 = SFTDataset(CHATS, tok, SEQ)
    s0, s2 = SFTDataset(CHATS, tok, SEQ), SFTDataset(CHATS, tok, SEQ)
    s0.shard(0, 2)
    s2.shard(1, 2)
    one = [s1.next_batch(2)[0] for _ in range(4)]
    two = [d.next_batch(2)[0] for d in (s0, s2) for _ in range(2)]
    assert rows(torch.cat(one)) == rows(torch.cat(two)), "SFT loader: rank slices != single read"
    assert s1.pos == s0.pos == s2.pos
    print(f"  SFTDataset: same set, pos {s1.pos} on every rank ✓")
    d1 = DPODataset(PREFS, tok, SEQ)
    d0, d2 = DPODataset(PREFS, tok, SEQ), DPODataset(PREFS, tok, SEQ)
    d0.shard(0, 2)
    d2.shard(1, 2)
    one = [d1.next_batch(4)[0] for _ in range(2)]                          # bs 4 x 2 micro-steps
    two = [d.next_batch(2)[0] for _ in range(2) for d in (d0, d2)]         # bs 2/rank x 2 micro-steps
    assert rows(torch.cat(one)) == rows(torch.cat(two)) and d1.pos == d0.pos == d2.pos
    print(f"  DPODataset: same set, pos {d1.pos} on every rank ✓")
    # unequal shares: rank 0 takes 3 windows of every 4-window block, rank 1 takes 1
    u0, u1 = PackedShardDataset(SHARDS, SEQ), PackedShardDataset(SHARDS, SEQ)
    u0.shard(0, 2, shares=[3, 1]); u1.shard(1, 2, shares=[3, 1])
    single2 = PackedShardDataset(SHARDS, SEQ)
    one = [single2.next_batch(1)[0] for _ in range(8)]                    # 2 steps x 4 windows
    two = [u0.next_batch(1)[0] for _ in range(6)] + [u1.next_batch(1)[0] for _ in range(2)]
    assert rows(torch.cat(one)) == rows(torch.cat(two)), "unequal shares: union of rank windows != single read"
    assert single2.pos == u0.pos == u1.pos, (single2.pos, u0.pos, u1.pos)
    print(f"  unequal shares [3,1]: same 8 windows over 2 steps, pos {u0.pos} on both ranks ✓")

    print("\n[2] single-process reference runs")
    w_single, step_single = run_pretrain(os.path.join(OUT, "single"), "fp32")
    dpo_single = run_dpo(os.path.join(OUT, "dpo-single"))
    print(f"  pretrain {step_single} steps, dpo {dpo_single[3]} steps ✓")

    print("\n[3] torchrun x2 (fp32): ranks agree exactly, and match the single-process weights")
    text = torchrun("pretrain-fp32")
    a, b = load_ranks("pretrain-fp32")
    assert a["step"] == b["step"] == step_single, (a["step"], b["step"], step_single)
    same_weights(a["weights"], b["weights"], exact=True)
    same_weights(a["weights"], w_single)
    mx = max(float((a["weights"][k] - w_single[k]).abs().max()) for k in w_single)
    assert "[train] precision=fp32" in text and "gpus=2" in text, "rank-0 output did not reach stdout"
    with open(os.path.join(OUT, "pretrain-fp32", "metrics.jsonl"), encoding="utf-8") as f:
        metas = [json.loads(l) for l in f if '"meta"' in l]
    assert len(metas) == 1 and metas[0]["tokens_per_step"] == SEQ * 2 * 4, metas
    print(f"  rank0 == rank1 bit for bit; vs single max |diff| {mx:.1e}; one metrics writer ✓")

    print("\n[4] torchrun x2 (fp16 loss scaling): identical to the fp32 DDP run")
    torchrun("pretrain-fp16")
    c, _ = load_ranks("pretrain-fp16")
    same_weights(c["weights"], a["weights"], exact=True)
    print("  bit-identical ✓")

    print("\n[5] torchrun x2 DPO: sharded reference precompute + DDP policy match single-process")
    torchrun("dpo")
    p0, p1 = load_ranks("dpo")
    same_weights(p0["weights"], p1["weights"], exact=True)
    assert torch.allclose(p0["ref_c"], dpo_single[1], atol=1e-4) and torch.allclose(p0["ref_r"], dpo_single[2], atol=1e-4)
    assert torch.allclose(p1["ref_c"], dpo_single[1], atol=1e-4), "rank 1 did not receive the full reference table"
    same_weights(p0["weights"], dpo_single[0], atol=1e-4)
    assert p0["step"] == dpo_single[3]
    print(f"  reference logprobs match; policy after {p0['step']} steps matches ✓")

    print("\n[5b] torchrun x2 with unequal shares [3,1]: ranks agree exactly and match single-process")
    torchrun("pretrain-fp32-shares")
    s0, s1 = load_ranks("pretrain-fp32-shares")
    assert s0["step"] == s1["step"] == step_single
    same_weights(s0["weights"], s1["weights"], exact=True)
    same_weights(s0["weights"], w_single)
    mx = max(float((s0["weights"][k] - w_single[k]).abs().max()) for k in w_single)
    print(f"  rank0 == rank1 bit for bit; vs single max |diff| {mx:.1e} ✓")

    print("\n[5c] auto-balancing: a slow rank ends up with fewer micro-steps, results still match single-process")
    text = torchrun("pretrain-fp32-auto")
    a0, a1 = load_ranks("pretrain-fp32-auto")
    assert a0["step"] == a1["step"] == step_single
    same_weights(a0["weights"], a1["weights"], exact=True)
    same_weights(a0["weights"], w_single)
    import json as _json
    with open(os.path.join(OUT, "pretrain-fp32-auto", "metrics.jsonl"), encoding="utf-8") as f:
        shares = [_json.loads(l)["rank_shares"] for l in f if "rank_shares" in l]
    assert shares and shares[-1][0] > shares[-1][1], f"shares did not move toward the fast rank: {shares}"
    print(f"  shares over the run: {shares[0]} -> {shares[-1]} (rank 1 slowed artificially); weights match single ✓")

    print("\n[6] torchrun x2 with NorMuon: ranks agree exactly and match the single-process NorMuon run")
    w_nm, _ = run_pretrain(os.path.join(OUT, "single-normuon"), "fp32", optimizer="normuon")
    torchrun("pretrain-fp32-normuon")
    n0, n1 = load_ranks("pretrain-fp32-normuon")
    same_weights(n0["weights"], n1["weights"], exact=True)
    same_weights(n0["weights"], w_nm)
    mx = max(float((n0["weights"][k] - w_nm[k]).abs().max()) for k in w_nm)
    print(f"  rank0 == rank1 bit for bit; vs single max |diff| {mx:.1e} ✓")

    print("\n[7] pause: stop file seen by rank 0 only -> every rank exits cleanly")
    text = torchrun("pause")
    q0, q1 = load_ranks("pause")
    assert q0["step"] == q1["step"] == 0, (q0["step"], q1["step"])
    with open(os.path.join(OUT, "pause", "metrics.jsonl"), encoding="utf-8") as f:
        assert any('"paused"' in l for l in f), "no paused event logged"
    print("  both ranks stopped at step 0 with a paused event ✓")

    print("\n" + "=" * 72)
    print("ALL CHECKS PASSED ✅")
    print("=" * 72)


if __name__ == "__main__":
    main()
