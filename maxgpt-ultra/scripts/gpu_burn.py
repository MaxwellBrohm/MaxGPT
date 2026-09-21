"""Controlled GPU load for the thermal ramp: sustained fp16 matmuls (~full power) on the given
cards for a set time, one process per card. Stop early with Ctrl-C / tmux kill-session.

  python scripts/gpu_burn.py --gpus 4,5 --minutes 40
"""
import argparse
import os
import subprocess
import sys
import time


def burn(minutes: float) -> None:
    import torch
    n = 8192
    a = torch.randn(n, n, device="cuda", dtype=torch.float16)
    b = torch.randn(n, n, device="cuda", dtype=torch.float16)
    end = time.time() + minutes * 60
    i = 0
    while time.time() < end:
        c = a @ b                       # tensor-core bound, ~max power
        if i % 50 == 0:
            torch.cuda.synchronize()
        i += 1
    torch.cuda.synchronize()
    print(f"burn on {os.environ.get('CUDA_VISIBLE_DEVICES')} done ({i} matmuls)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", required=True)
    ap.add_argument("--minutes", type=float, default=30)
    ap.add_argument("--one", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.one:
        burn(args.minutes)
        return
    procs = []
    for g in args.gpus.split(","):
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": g.strip()}
        procs.append(subprocess.Popen([sys.executable, os.path.abspath(__file__), "--gpus", g, "--minutes",
                                       str(args.minutes), "--one"], env=env))
    print(f"burning GPUs {args.gpus} for {args.minutes} min", flush=True)
    try:
        for p in procs:
            p.wait()
    except KeyboardInterrupt:
        for p in procs:
            p.terminate()


if __name__ == "__main__":
    main()
