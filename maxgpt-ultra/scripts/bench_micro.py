"""Measure real training throughput of a config on ONE GPU for a few candidate settings, so the
micro_batch / grad_checkpointing choice is made on tokens/s, not guesswork.

  python scripts/bench_micro.py --config configs/ultra_lambda.yaml \
      --variant mb1:ckpt0 --variant mb4:ckpt1 --variant mb8:ckpt1

Each variant runs in its own subprocess (an OOM cannot poison the others): build the model,
compile it (default mode: same relative ordering as max-autotune, minutes less compile), warm up,
then time `--steps` forward+backward micro-steps with the trainer's exact precision path
(autocast + loss scaling where the card has no bf16). Reports tokens/s per GPU and peak VRAM.
Under DDP every GPU runs the same micro-steps, so tokens/s scales with the GPU count.
"""
import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # maxgpt-ultra/
from cfg import configure_triton_ptxas, cuda_alloc_conf
cuda_alloc_conf()
configure_triton_ptxas()


def run_variant(args) -> None:
    import torch
    from contextlib import nullcontext
    from model import ModelConfig, MaxGPTUltra, load_yaml
    from train.trainer import make_optimizer, _enable_fast_math, amp_dtype
    mb, ckpt = args.probe.split(":")
    mb, ckpt = int(mb[2:]), bool(int(ckpt[4:]))
    t = load_yaml(args.config).get("train", {})
    mcfg = ModelConfig.from_yaml(args.config)
    _enable_fast_math()
    model = MaxGPTUltra(mcfg).cuda()
    model.grad_checkpointing = ckpt
    model.loss_chunk = int(t.get("loss_chunk", 0))
    model.train()
    opt = make_optimizer(model, float(t.get("lr", 3e-4)), tuple(t.get("betas", [0.9, 0.95])),
                         float(t.get("weight_decay", 0.1)), bool(t.get("optimizer_8bit", False)),
                         kind=str(t.get("optimizer", "adamw")).lower())
    dtype = amp_dtype(t.get("precision", "auto"), "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16))
    fwd = torch.compile(model, mode=(None if args.compile_mode == "default" else args.compile_mode)) \
        if args.compile_mode != "eager" else model
    V, T = mcfg.vocab_size, mcfg.seq_len
    z = float(t.get("z_loss", 0.0))

    def micro():
        x = torch.randint(0, V, (mb, T), device="cuda")
        y = torch.randint(0, V, (mb, T), device="cuda")
        with (torch.autocast("cuda", dtype=dtype) if dtype is not None else nullcontext()):
            _, loss = fwd(x, y, z_loss_weight=z)
        scaler.scale(loss).backward()

    try:
        for _ in range(args.warmup):          # compile + autotune + allocator warm-up
            micro()
        scaler.unscale_(opt); scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        for _ in range(args.steps):
            micro()
        torch.cuda.synchronize()
        dt = time.time() - t0
        toks = args.steps * mb * T
        print(f"RESULT {args.probe} tok/s_per_gpu={toks / dt:,.0f} ms/micro={1000 * dt / args.steps:.0f} "
              f"peak_gb={torch.cuda.max_memory_allocated() / 1e9:.2f} of {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}")
    except torch.OutOfMemoryError:
        print(f"RESULT {args.probe} OOM")
    except Exception as e:
        print(f"RESULT {args.probe} ERROR {type(e).__name__}: {str(e).splitlines()[0][:160]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--variant", action="append", default=[], help="mbN:ckpt0|1 (repeatable)")
    ap.add_argument("--compile-mode", default="default", help="default | max-autotune-no-cudagraphs | max-autotune | eager")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--probe", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.probe:
        run_variant(args)
        return
    variants = args.variant or ["mb1:ckpt0", "mb2:ckpt0", "mb4:ckpt1", "mb8:ckpt1"]
    print(f"[bench] {args.config} compile={args.compile_mode} steps={args.steps} (+{args.warmup} warm-up) per variant", flush=True)
    for v in variants:
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "--config", args.config, "--probe", v,
                            "--compile-mode", args.compile_mode, "--steps", str(args.steps), "--warmup", str(args.warmup)],
                           capture_output=True, text=True)
        line = next((l for l in (r.stdout + r.stderr).splitlines() if l.startswith("RESULT")), f"RESULT {v} (no output, rc={r.returncode})")
        print("  " + line, flush=True)


if __name__ == "__main__":
    main()
