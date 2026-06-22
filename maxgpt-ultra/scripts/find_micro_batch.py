"""Find the largest micro_batch that fits in VRAM, automatically and safely.

Why a script and not eyeballing it: the right micro_batch is the biggest one that fits,
and guessing leaves throughput on the table. Why subprocesses: a CUDA OOM can corrupt the
process's CUDA context, so each candidate is tested in a FRESH subprocess. If a size OOMs,
only that throwaway process dies and the search continues cleanly. Nothing here can wedge
your machine.

How it works: it runs REAL training steps (forward + backward + optimizer.step) at each
candidate size, using THIS config's exact memory settings (grad_checkpointing, 8-bit
optimizer, chunked loss, bf16 autocast), so the number it finds matches real training. It
doubles the batch (1, 2, 4, 8, ...) until the first OOM, then binary-searches between the
last size that fit and the first that didn't for the exact maximum.

  python scripts/find_micro_batch.py --config configs/ultra.yaml

Options:
  --compile     confirm under the real max-autotune compile (the authoritative number). It runs
                the fast eager search first, then compiles ONLY near that boundary (a few trials,
                minutes each) instead of compiling every size. Compiled training usually fits the
                same or MORE than eager, so this can hand you an even larger micro_batch.
  --max-batch N don't probe past N (default 128)
  --seq-len N   override the sequence length (default: the config's)
  --steps N     train steps per trial (default 4; >=2 so the 8-bit optimizer state allocates)
"""
import argparse
import os
import sys
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # maxgpt-ultra/
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")     # match the real run


def run_probe(micro_batch: int, args) -> None:
    """One trial in THIS process. Exit code: 0 = fit, 2 = OOM, 3 = no-CUDA/other error."""
    import torch
    import yaml
    if not torch.cuda.is_available():
        print("NOCUDA")
        sys.exit(3)
    from model import ModelConfig, MaxGPTUltra
    from train.trainer import make_optimizer, _enable_fast_math

    raw = yaml.safe_load(open(args.config, encoding="utf-8"))
    t = raw.get("train", {})
    mcfg = ModelConfig.from_yaml(args.config)
    seq_len = args.seq_len or mcfg.seq_len
    dev = "cuda"
    _enable_fast_math()
    try:
        torch.cuda.reset_peak_memory_stats()
        model = MaxGPTUltra(mcfg).to(dev)
        model.grad_checkpointing = bool(t.get("grad_checkpointing", False))
        model.loss_chunk = int(t.get("loss_chunk", 0))
        model.train()
        opt = make_optimizer(model, float(t.get("lr", 3e-4)), tuple(t.get("betas", [0.9, 0.95])),
                             float(t.get("weight_decay", 0.1)), bool(t.get("optimizer_8bit", False)))
        fwd = torch.compile(model, mode="max-autotune") if args.compile else model
        z = float(t.get("z_loss", 0.0))
        V = mcfg.vocab_size
        for _ in range(args.steps):
            x = torch.randint(0, V, (micro_batch, seq_len), device=dev)
            y = torch.randint(0, V, (micro_batch, seq_len), device=dev)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss = fwd(x, y, z_loss_weight=z)
            loss.backward()
            opt.step()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"FIT peak={peak:.2f}GB/{total:.2f}GB")
        sys.exit(0)
    except torch.cuda.OutOfMemoryError:
        print("OOM")
        sys.exit(2)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("OOM")
            sys.exit(2)
        print(f"ERROR {type(e).__name__}: {e}")
        sys.exit(3)


def spawn(micro_batch: int, args, use_compile: bool):
    """Run one probe in a clean subprocess. Returns (fit, peak_gb, total_gb, raw_line)."""
    cmd = [sys.executable, os.path.abspath(__file__), "--config", args.config,
           "--steps", str(args.steps), "--probe", str(micro_batch)]
    if args.seq_len:
        cmd += ["--seq-len", str(args.seq_len)]
    if use_compile:
        cmd += ["--compile"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    lines = (r.stdout + "\n" + r.stderr).strip().splitlines()
    line = next((l for l in lines if l.startswith(("FIT", "OOM", "ERROR", "NOCUDA"))), "")
    peak = total = None
    if line.startswith("FIT") and "peak=" in line:
        seg = line.split("peak=")[1]
        peak = float(seg.split("GB")[0])
        total = float(seg.split("/")[1].rstrip("GB"))
    return (r.returncode == 0, peak, total, line or "(no output)")


def refine_compile(args, eager_best: int):
    """Confirm + expand the eager result under the real max-autotune compile. Compiled
    training usually uses <= eager memory, so start at the eager max and expand upward; if
    it OOMs (compile used more) walk down. Returns (best, peak, total), or None if compile
    failed to build on this machine (then the eager number stands as a safe floor)."""
    print("\n[autotune] confirming under torch.compile (max-autotune) -- minutes per trial, hang tight...")

    def trial(mb):
        fit, peak, total, line = spawn(mb, args, use_compile=True)
        if line.startswith("ERROR"):
            print(f"  [compile] micro_batch={mb:<4} compile FAILED ({line[:60]})")
            return None
        print(f"  [compile] micro_batch={mb:<4} {'FIT' if fit else 'OOM'}" + (f"   peak {peak:.2f}GB" if peak else ""))
        return (fit, peak, total)

    r = trial(eager_best)
    if r is None:
        return None
    if not r[0]:                                   # compiled used MORE -> walk down to the first fit
        mb = eager_best
        while mb > 1:
            mb -= 1
            r = trial(mb)
            if r is None:
                return None
            if r[0]:
                return (mb, r[1], r[2])
        return (1, r[1], r[2])
    best, bpeak, btot = eager_best, r[1], r[2]     # fits -> expand upward (compile often frees room)
    mb = eager_best + 1
    while mb <= args.max_batch:
        r = trial(mb)
        if r is None or not r[0]:
            break
        best, bpeak, btot = mb, r[1], r[2]
        mb += 1
    return (best, bpeak, btot)


def main() -> None:
    ap = argparse.ArgumentParser(description="Auto-find the largest micro_batch that fits in VRAM.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", type=int, default=4, help="real train steps per trial (>=2)")
    ap.add_argument("--max-batch", type=int, default=128, help="don't probe past this")
    ap.add_argument("--seq-len", type=int, default=None, help="override seq_len (default: config)")
    ap.add_argument("--compile", action="store_true",
                    help="confirm under real max-autotune compile (the authoritative number; minutes per trial)")
    ap.add_argument("--probe", type=int, default=None, help=argparse.SUPPRESS)  # internal: one trial
    args = ap.parse_args()

    if args.probe is not None:
        run_probe(args.probe, args)
        return

    print(f"[autotune] config={args.config}  seq_len={args.seq_len or 'config'}  "
          f"steps/trial={args.steps}  compile={args.compile}")
    print("[autotune] each size runs real train steps in an isolated subprocess (OOM-safe)\n")

    peaks: dict[int, float] = {}
    total_gb = None

    # 1) double until the first OOM -> bracket [last_ok, first_oom]
    last_ok, first_oom, mb = 0, None, 1
    while mb <= args.max_batch:
        fit, peak, total, line = spawn(mb, args, use_compile=False)
        if line == "NOCUDA":
            print("[autotune] No CUDA GPU detected. Run this on your PC (the 5070 box), not the Mac.")
            return
        if line.startswith("ERROR"):
            print(f"[autotune] trial errored (not an OOM): {line}\n           fix this before trusting results.")
            return
        total_gb = total or total_gb
        print(f"  micro_batch={mb:<4} {'FIT' if fit else 'OOM'}" + (f"   peak {peak:.2f}GB" if peak else ""))
        if fit:
            peaks[mb] = peak
            last_ok = mb
            mb *= 2
        else:
            first_oom = mb
            break

    if last_ok == 0:
        print("\n[autotune] even micro_batch=1 did not fit. Turn on grad_checkpointing or lower seq_len.")
        return

    if first_oom is None:
        print(f"\n[autotune] reached the --max-batch cap ({args.max_batch}) without an OOM.")
        print(f"           raise --max-batch to search higher.")
        best = last_ok
    else:
        # 2) binary search between last_ok (fit) and first_oom (oom)
        lo, hi = last_ok, first_oom
        while hi - lo > 1:
            mid = (lo + hi) // 2
            fit, peak, total, line = spawn(mid, args, use_compile=False)
            total_gb = total or total_gb
            print(f"  micro_batch={mid:<4} {'FIT' if fit else 'OOM'}" + (f"   peak {peak:.2f}GB" if peak else ""))
            if fit:
                peaks[mid] = peak
                lo = mid
            else:
                hi = mid
        best = lo

    peak = peaks.get(best)
    tested_compiled = False

    # 2) (optional) confirm under the REAL max-autotune compile, near the eager boundary
    if args.compile:
        refined = refine_compile(args, best)
        if refined is None:
            print("[autotune] torch.compile didn't build on this machine; keeping the eager result (a safe floor).")
        else:
            best, peak, ctot = refined
            total_gb = ctot or total_gb
            tested_compiled = True

    # 3) report + recommend. Leave a little headroom: the real run also does periodic eval and a
    #    divergence-rollback checkpoint load, which spike memory transiently above a plain step.
    rec = best
    reason = ""
    if peak and total_gb and peak > 0.93 * total_gb:
        rec = max(1, best - 1)
        reason = "  (backed off 1: peak was within ~7% of total VRAM)"

    # suggest a grad_accum that keeps the SAME effective batch (so the LR schedule is unchanged)
    raw = __import__("yaml").safe_load(open(args.config, encoding="utf-8"))
    t = raw.get("train", {})
    cur_mb, cur_ga = int(t.get("micro_batch", 1)), int(t.get("grad_accum", 1))
    eff = cur_mb * cur_ga
    new_ga = max(1, round(eff / rec))

    print("\n[autotune] RESULT" + ("  (max-autotune compile -- the real-run number)" if tested_compiled
                                   else "  (eager estimate)"))
    print(f"  largest micro_batch that fits : {best}" + (f"   (peak {peak:.2f}GB / {total_gb:.2f}GB)" if peak else ""))
    print(f"  recommended micro_batch       : {rec}{reason}")
    if not tested_compiled:
        print(f"  note: eager estimate. The real run compiles, which usually fits the same or MORE,")
        print(f"        so for the authoritative number re-run with --compile.")
    print(f"\n  In {args.config} set:")
    print(f"      micro_batch: {rec}")
    print(f"      grad_accum:  {new_ga}      # keeps effective batch ~{rec * new_ga} (was {eff}): same dynamics, just faster")


if __name__ == "__main__":
    main()
