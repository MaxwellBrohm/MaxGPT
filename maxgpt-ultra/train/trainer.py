"""The MaxGPT-Ultra training loop.

Config-driven (the `train:` block of a YAML). Handles: WSD learning-rate schedule, mixed
precision on CUDA (bf16 where the card has it, else fp16 + dynamic loss scaling; fp32 on
CPU), gradient accumulation to a large effective batch, gradient clipping, optional z-loss,
optional gradient checkpointing + 8-bit AdamW (for the 1B on 12GB), multi-GPU via DDP
(launch with torchrun; see train/dist.py), periodic JSONL metric logging (for the
dashboard), time-based autosave, a divergence guard that rolls back to the last good
checkpoint on a NaN/Inf, and exact resume-from-latest.

Batch semantics: `micro_batch` is per GPU, `grad_accum` is the TOTAL number of micro-steps
per optimizer step across all GPUs (so tokens/step, and therefore the LR schedule, do not
change with the GPU count). Each rank runs grad_accum / world_size micro-steps.
"""
from __future__ import annotations

import json
import math
import os
import platform
import time
from contextlib import nullcontext

import torch

from .schedule import wsd_lr
from .checkpoint import CheckpointManager, load_checkpoint
from . import dist as D


def make_optimizer(model, lr, betas, weight_decay, use_8bit=False, kind="adamw", muon_opts=None):
    # weight decay on 2D+ tensors (matmuls/embeddings), none on 1D (norms/gains)
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [{"params": decay, "weight_decay": weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    on_cuda = torch.cuda.is_available() and any(p.is_cuda for g in groups for p in g["params"])
    if kind in ("muon", "normuon"):
        from .muon import Muon
        opts = dict(muon_opts or {})
        # Muon for the block matrices; AdamW (same lr / decay rules) for embeddings, head, and 1D
        emb = {id(p) for n, p in model.named_parameters()
               if n.startswith(("tok_emb", "lm_head")) or n.endswith(("tok_emb.weight", "lm_head.weight"))}
        mats = [p for p in decay if id(p) not in emb]
        embs = [p for p in decay if id(p) in emb]
        groups = [{"params": mats, "weight_decay": weight_decay, "use_muon": True},
                  {"params": embs, "weight_decay": weight_decay, "use_muon": False},
                  {"params": no_decay, "weight_decay": 0.0, "use_muon": False}]
        ns_dtype = None
        if on_cuda:
            ns_dtype = torch.bfloat16 if cuda_has_native_bf16() else torch.float16
        return Muon(groups, lr=lr, weight_decay=weight_decay, betas=betas,
                    normalize=(kind == "normuon") or bool(opts.get("normalize", False)),
                    cautious=bool(opts.get("cautious_wd", False)), momentum=float(opts.get("momentum", 0.95)),
                    ns_dtype=ns_dtype)
    if use_8bit and on_cuda:
        try:
            import bitsandbytes as bnb
            # Paged 8-bit pages optimizer state to host RAM via CUDA unified memory, which only
            # oversubscribes on Linux. On Windows it cannot page and OOMs, so use plain 8-bit there.
            if platform.system() != "Windows":
                try:
                    return bnb.optim.PagedAdamW8bit(groups, lr=lr, betas=betas)   # frees ~2B/param of VRAM
                except Exception:
                    pass
            return bnb.optim.AdamW8bit(groups, lr=lr, betas=betas)
        except Exception as e:
            print(f"[train] bitsandbytes unavailable ({type(e).__name__}); using fp32 AdamW "
                  f"(optimizer state costs 8 bytes/param instead of 2)", flush=True)
    # fused=True runs the whole update in one kernel (CUDA only); same math as the default
    return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=on_cuda)


def cuda_has_native_bf16() -> bool:
    """True only when the GPU's tensor cores do bf16 (Ampere sm_80 and newer). PyTorch's
    is_bf16_supported() also says True for cards that merely EMULATE bf16 (Turing, Volta), and
    emulated bf16 is many times slower than fp16 there, so we ask about real hardware only."""
    if not torch.cuda.is_available():
        return False
    try:
        return bool(torch.cuda.is_bf16_supported(including_emulation=False))
    except TypeError:                                   # older torch: decide by compute capability
        major, _ = torch.cuda.get_device_capability()
        return major >= 8


def amp_dtype(precision: str | None, device: str):
    """Resolve the autocast dtype for this machine, or None for plain fp32.

    'auto' -> bf16 on cards that have it (Ampere+), else fp16; 'bf16' falls back to fp16 with
    a warning on cards without bf16 (Turing / Volta), because bf16 matmuls do not exist there.
    fp16 needs dynamic loss scaling (see Trainer) since its exponent range is tiny; the master
    weights, optimizer state and gradient accumulation stay fp32 in every mode, so the model
    quality is the same either way. CPU never autocasts.
    """
    p = (precision or "auto").lower()
    if device != "cuda" or p == "fp32":
        return None
    has_bf16 = cuda_has_native_bf16()
    if p == "fp16":
        return torch.float16
    if p == "bf16" and not has_bf16:
        print("[train] this GPU has no bf16; using fp16 + loss scaling instead", flush=True)
        return torch.float16
    if p in ("bf16", "auto"):
        return torch.bfloat16 if has_bf16 else torch.float16
    raise ValueError(f"unknown precision {precision!r} (use auto | bf16 | fp16 | fp32)")


def _enable_fast_math() -> None:
    """TF32 matmuls + cuDNN autotuning. Faster on Ampere/Blackwell at negligible precision
    cost (we already train in bf16). No-op / harmless on CPU and on pre-Ampere cards."""
    try:
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass


class Trainer:
    def __init__(self, model, data, tcfg: dict, device: str, out_dir: str,
                 eval_fn=None, seed: int = 0, stop_file: str | None = None):
        self.world = D.world_size()
        self.rank = D.rank()
        self.is_main = D.is_main()
        self.model = model.to(device)
        self.data = data
        self.tcfg = tcfg
        self.device = device
        self.eval_fn = eval_fn
        self.seed = seed
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.ckpt = CheckpointManager(os.path.join(out_dir, "checkpoints"),
                                      keep_last=int(tcfg.get("keep_last_k", 3)))
        self.log_path = os.path.join(out_dir, "metrics.jsonl")

        self.micro_batch = int(tcfg["micro_batch"])
        ga_total = int(tcfg["grad_accum"])
        # rank_shares: unequal micro-step counts per rank (in CUDA_VISIBLE_DEVICES order) for cards of
        # unequal speed; DDP waits for the slowest card every step, so a throttled card should carry
        # fewer micro-steps. Must sum to grad_accum. The gradient stays the exact same average over the
        # same grad_accum micro-batches (each rank scales its losses by the MEAN share, so the DDP
        # all-reduce mean equals the global mean); only the dealing changes.
        shares = tcfg.get("rank_shares")
        if shares and len(shares) == self.world and self.world > 1:
            shares = [int(n) for n in shares]
            if sum(shares) != ga_total or min(shares) < 1:
                raise ValueError(f"rank_shares {shares} must be >= 1 each and sum to grad_accum={ga_total}")
            self.rank_shares = shares
            self.grad_accum = shares[self.rank]             # micro-steps THIS rank runs per step
        else:
            if shares and self.world > 1 and self.is_main:
                print(f"[train] rank_shares has {len(shares)} entries but there are {self.world} ranks; using an equal split", flush=True)
            if ga_total % self.world:
                lo = (ga_total // self.world) * self.world
                raise ValueError(f"grad_accum={ga_total} must be a multiple of the GPU count ({self.world}); "
                                 f"use {max(lo, self.world)} or {lo + self.world}, or a GPU count that divides it")
            self.rank_shares = None
            self.grad_accum = ga_total // self.world        # micro-steps THIS rank runs per step
        self.loss_div = ga_total / self.world                # per-micro-step loss scale: mean share, so the all-reduced grad is the global mean
        self.ga_total = ga_total
        # auto-balance: every N steps each rank reports its measured micro-step speed and the shares are
        # re-dealt in proportion (smoothed), so throttling cards, a new power cap or a serviced card are
        # absorbed without anyone touching the config. Exact same gradient whatever the dealing.
        self.shares_auto = bool(tcfg.get("rank_shares_auto", True)) and self.world > 1
        self.shares_every = int(tcfg.get("rank_shares_every", 10))
        self.shares_smooth = float(tcfg.get("rank_shares_smooth", 0.5))   # weight of the new measurement
        self._micro_time = 0.0                                  # seconds spent in this step's no_sync micro-steps
        self._micro_n = 0
        self._speed_acc = []                                    # per-step measured speeds (micro-steps/s) since the last re-deal
        self.seq_len = data.seq_len
        self.tokens_per_step = self.micro_batch * ga_total * self.seq_len
        self.total_steps = max(1, int(float(tcfg["total_tokens"]) // self.tokens_per_step))
        self.warmup_steps = int(float(tcfg.get("warmup_tokens", 0)) // self.tokens_per_step)
        self.max_lr = float(tcfg["lr"])
        self.decay_frac = float(tcfg.get("decay_frac", 0.15))
        self.grad_clip = float(tcfg.get("grad_clip", 1.0))
        self.z_loss = float(tcfg.get("z_loss", 0.0))
        self.betas = tuple(tcfg.get("betas", [0.9, 0.95]))
        self.wd = float(tcfg.get("weight_decay", 0.1))
        self.autosave_s = float(tcfg.get("autosave_minutes", 15)) * 60.0
        self.log_every = int(tcfg.get("log_every", 10))
        self.eval_every = int(tcfg.get("eval_every", 0))

        # Decay-phase data annealing (docs/research_2026-09-22.md 2.3): once the shards exist, blend
        # them in from the decay start (or an explicit step), ramping up over ramp_tokens.
        an = tcfg.get("anneal") or {}
        if an.get("shards") and hasattr(data, "_read"):
            from data.loader import PackedShardDataset as _PSD, AnnealBlend
            adir = str(an["shards"])
            if os.path.exists(os.path.join(adir, "meta.json")):
                start = an.get("start", "decay")
                start_step = int(self.total_steps * (1.0 - self.decay_frac)) if start == "decay" else int(start)
                data = AnnealBlend(data, _PSD(adir, data.seq_len), frac=float(an.get("frac", 0.4)),
                                   start_step=start_step, ramp_tokens=float(an.get("ramp_tokens", 1e9)),
                                   tokens_per_step=self.tokens_per_step)
                self.data = data
                if self.is_main:
                    print(f"[train] annealing: {adir} blended in at {float(an.get('frac', 0.4)):.0%} of each step "
                          f"from step {start_step:,} (ramp over {data.ramp_steps:,} steps)", flush=True)
            elif self.is_main:
                print(f"[train] anneal shards not found at {adir}: annealing OFF", flush=True)
        if hasattr(data, "shard"):                          # each rank reads its own slice of every step
            if self.rank_shares:
                data.shard(self.rank, self.world, shares=[n * self.micro_batch for n in self.rank_shares])
            elif hasattr(data, "set_step"):                 # a blend always deals in step blocks
                data.shard(self.rank, self.world, shares=[self.grad_accum * self.micro_batch] * self.world)
            else:
                data.shard(self.rank, self.world)
        if self.shares_auto and not self.rank_shares and hasattr(data, "shard"):
            # start balanced; the first re-deal comes after shares_every measured steps
            self.rank_shares = [self.grad_accum] * self.world
            data.shard(self.rank, self.world, shares=[n * self.micro_batch for n in self.rank_shares])

        self.model.grad_checkpointing = bool(tcfg.get("grad_checkpointing", False))
        self.model.loss_chunk = int(tcfg.get("loss_chunk", 0))   # >0 -> chunked, memory-light loss
        self.amp_dtype = amp_dtype(tcfg.get("precision", "auto"), device)
        self.use_amp = self.amp_dtype is not None
        # fp16 has no room for small gradients, so the loss is multiplied by a large scale before
        # backward and the gradients divided by it before clipping/stepping (exact: powers of two).
        # A step whose gradients overflow is skipped and the scale shrinks; it grows back slowly.
        # On CPU with precision=fp16 the scaler runs without autocast (the test path).
        use_scaler = (self.amp_dtype == torch.float16) or \
                     (device == "cpu" and str(tcfg.get("precision", "")).lower() == "fp16")
        self.scaler = torch.amp.GradScaler(device, enabled=use_scaler,
                                           init_scale=float(tcfg.get("loss_scale_init", 2.0 ** 16)))
        self.optimizer = make_optimizer(self.model, self.max_lr, self.betas, self.wd,
                                        bool(tcfg.get("optimizer_8bit", False)),
                                        kind=str(tcfg.get("optimizer", "adamw")).lower(),
                                        muon_opts={"cautious_wd": tcfg.get("cautious_wd", False),
                                                   "momentum": tcfg.get("muon_momentum", 0.95)})
        _enable_fast_math()
        self.net = D.wrap_ddp(self.model)                   # DDP wrapper (or the model itself)
        self._ddp = self.net is not self.model
        self._grads_bound = False    # DDP: True once .grad tensors are views into the all-reduce buckets
        # torch.compile fuses kernels for a large throughput win (CUDA only). It shares params
        # with self.model, so the optimizer / save / load keep using the uncompiled handle. We try
        # the most aggressive mode first and drop a tier at a time on failure
        # (max-autotune -> default -> eager), so we always end up running.
        # tiers default to autotuned kernels WITHOUT CUDA graphs: with graphs, the checkpointed chunked
        # loss re-runs a compiled subgraph inside backward and overwrites graph outputs the rest of the
        # backward still needs (torch raises "accessing tensor output of CUDAGraphs that has been overwritten")
        self._compile_modes = (list(tcfg.get("compile_modes", ["max-autotune-no-cudagraphs", "default"]))
                               if bool(tcfg.get("compile", False)) and device == "cuda" else [])
        self.fwd = self._make_fwd()
        self.step = 0
        self._last_save = time.time()
        self._t0 = time.time()
        self._stop = False           # set by request_stop() (e.g. Ctrl-C)
        self.stop_file = stop_file    # GUI pause: presence of this file => checkpoint + exit
        self._rollbacks_at = (-1, 0)  # (step, count): give up after repeated rollbacks at one step
        if self.is_main:
            prec = {None: "fp32", torch.bfloat16: "bf16", torch.float16: "fp16"}[self.amp_dtype]
            prec += " + loss scaling" if self.scaler.is_enabled() else ""
            print(f"[train] precision={prec} optimizer={type(self.optimizer).__name__} "
                  f"gpus={self.world} micro_batch={self.micro_batch}/gpu grad_accum={ga_total} total "
                  f"({'shares ' + str(self.rank_shares) if self.rank_shares else str(self.grad_accum) + '/gpu'}) "
                  f"-> {self.tokens_per_step:,} tokens/step", flush=True)

    def _make_fwd(self):
        """Compile self.net with the next available mode; eager if none left."""
        while self._compile_modes:
            mode = self._compile_modes[0]
            try:
                f = torch.compile(self.net, mode=(None if mode == "default" else mode), dynamic=False)
                if self.is_main:
                    print(f"[train] torch.compile(mode={mode})", flush=True)
                return f
            except Exception as e:
                if self.is_main:
                    print(f"[train] compile setup mode={mode} failed ({type(e).__name__}); next tier", flush=True)
                self._compile_modes.pop(0)
        return self.net

    @staticmethod
    def _release_compile_memory() -> None:
        """A failed compiled attempt can leave autotune workspaces / a CUDA-graph pool behind;
        give that VRAM back before the next tier runs (a real OOM must not cascade)."""
        import gc
        try:
            import torch._dynamo
            torch._dynamo.reset()
        except Exception:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def request_stop(self) -> None:
        self._stop = True

    def _should_stop(self) -> bool:
        # rank 0 decides, everyone follows: a stop must be unanimous or the ranks desync
        mine = self._stop or bool(self.stop_file and os.path.exists(self.stop_file))
        return D.broadcast_flag(mine)

    # --- helpers ---
    def _set_lr(self, lr: float) -> None:
        for g in self.optimizer.param_groups:
            g["lr"] = lr

    def _micro_forward(self, sync: bool = True):
        if os.environ.get("MAXGPT_TEST_SLOW_RANK") == str(self.rank):      # test hook: pretend this card is slow
            time.sleep(float(os.environ.get("MAXGPT_TEST_SLOW_SECS", "0.05")))
        x, y = self.data.next_batch(self.micro_batch, self.device)
        ctx = (torch.autocast(device_type="cuda", dtype=self.amp_dtype)
               if self.use_amp else nullcontext())
        # DDP: skip the gradient all-reduce on every micro-step but the last one of the step
        nosync = self.net.no_sync() if (not sync and hasattr(self.net, "no_sync")) else nullcontext()
        with ctx, nosync:
            while True:
                try:
                    _, loss = self.fwd(x, y, z_loss_weight=self.z_loss)
                    break
                except Exception as e:
                    if self._compile_modes:             # a compiled mode failed at runtime -> next tier
                        if self.is_main:
                            print(f"[train] compiled forward failed ({type(e).__name__}: "
                                  f"{str(e).splitlines()[0][:160]}); dropping a compile tier", flush=True)
                        self._compile_modes.pop(0)
                        self._release_compile_memory()
                        self.fwd = self._make_fwd()
                    else:
                        raise
            loss_scaled = self.scaler.scale(loss / self.loss_div)
            loss_scaled.backward()
        # clone: under max-autotune the forward is a CUDA graph whose output buffers are reused by
        # the next replay; keeping the raw output alive across micro-steps raises "accessing tensor
        # output of CUDAGraphs that has been overwritten" (found on the Lambda box)
        return loss.detach().clone()

    def _log(self, rec: dict) -> None:
        if not self.is_main:
            return
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    def _bind_grads(self) -> None:
        """DDP memory: under no_sync autograd would allocate a second, standalone copy of every
        gradient (a whole model's worth) next to DDP's all-reduce buckets. One synced backward of a
        ZERO loss on the first batch binds each .grad to its bucket view; from then on grads are
        zeroed in place instead of dropped, so every micro-step accumulates straight into the
        buckets. Done once per run, as exactly one collective on every rank (which keeps ranks with
        unequal micro-step shares in lockstep); the batch is rewound and used again for real."""
        state = self.data.state_dict()
        x, y = self.data.next_batch(self.micro_batch, self.device)
        ctx = (torch.autocast(device_type="cuda", dtype=self.amp_dtype) if self.use_amp else nullcontext())
        with ctx:
            _, loss = self.fwd(x, y, z_loss_weight=self.z_loss)
        (loss * 0.0).backward()
        self.optimizer.zero_grad(set_to_none=False)
        self.data.load_state_dict(state)
        self._grads_bound = True

    def _accumulate(self):
        """grad_accum micro-steps of forward+backward; returns the summed (detached) loss."""
        if self._ddp and not self._grads_bound:
            self._bind_grads()
        self.optimizer.zero_grad(set_to_none=not self._ddp)
        loss_sum = None
        t0 = time.time()
        for i in range(self.grad_accum):
            loss = self._micro_forward(sync=(i == self.grad_accum - 1))   # all-reduce only on the last one
            # accumulate on-device (detached); one GPU->CPU sync per step instead of per micro-step
            loss_sum = loss if loss_sum is None else loss_sum + loss
            if i == self.grad_accum - 2:                        # time the no_sync micro-steps only (no all-reduce wait)
                if self.device == "cuda":
                    torch.cuda.synchronize()
                self._micro_time, self._micro_n = time.time() - t0, self.grad_accum - 1
        if self.grad_accum == 1:                                # a single micro-step: time the whole thing
            if self.device == "cuda":
                torch.cuda.synchronize()
            self._micro_time, self._micro_n = time.time() - t0, 1
        return loss_sum

    def _rebalance_shares(self) -> None:
        """Re-deal grad_accum across ranks in proportion to each rank's measured speed (all ranks
        compute the identical result from the same all-reduced vector). Called at step boundaries."""
        if not self.shares_auto or self._micro_n == 0:
            return
        self._speed_acc.append(self._micro_n / max(self._micro_time, 1e-6))
        if len(self._speed_acc) < self.shares_every:
            return
        my_speed = sum(self._speed_acc) / len(self._speed_acc)
        self._speed_acc = []
        vec = torch.zeros(self.world, dtype=torch.float64, device=D.reduce_device())
        vec[self.rank] = my_speed
        D.all_reduce_sum(vec)
        speeds = vec.tolist()
        target = [self.ga_total * sp / sum(speeds) for sp in speeds]
        blended = [(1 - self.shares_smooth) * old + self.shares_smooth * new for old, new in zip(self.rank_shares, target)]
        shares = [max(1, int(round(b))) for b in blended]
        while sum(shares) != self.ga_total:                     # fix rounding, one unit at a time
            i = max(range(self.world), key=lambda r: blended[r] - shares[r]) if sum(shares) < self.ga_total \
                else min(range(self.world), key=lambda r: (blended[r] - shares[r], -shares[r]))
            if sum(shares) > self.ga_total and shares[i] <= 1:
                i = max(range(self.world), key=lambda r: shares[r])
            shares[i] += 1 if sum(shares) < self.ga_total else -1
        if shares != self.rank_shares:
            if self.is_main:
                print(f"[train] rank shares {self.rank_shares} -> {shares} (measured speeds "
                      f"{', '.join(f'{sp:.2f}' for sp in speeds)} micro-steps/s)", flush=True)
            self.rank_shares = shares
            self.grad_accum = shares[self.rank]
            if hasattr(self.data, "shard"):
                self.data.shard(self.rank, self.world, shares=[n * self.micro_batch for n in shares])

    # --- one optimizer step (grad_accum micro-steps) ---
    def train_step(self) -> dict:
        lr = wsd_lr(self.step, total_steps=self.total_steps, warmup_steps=self.warmup_steps,
                    decay_frac=self.decay_frac, max_lr=self.max_lr)
        self._set_lr(lr)
        if hasattr(self.data, "set_step"):                  # annealing blend: the mix depends on the step
            self.data.set_step(self.step)
        data_state = self.data.state_dict()          # so a compile-tier failure can replay this exact step
        while True:
            try:
                loss_sum = self._accumulate()
                break
            except Exception as e:
                if not self._compile_modes:          # eager already: a real error
                    raise
                # a compiled tier failed somewhere in forward/backward: drop it and redo the whole step
                # on the next tier with the same batches (grads zeroed, loader rewound)
                if self.is_main:
                    print(f"[train] compiled step failed ({type(e).__name__}: {str(e).splitlines()[0][:160]}); "
                          f"dropping a compile tier and replaying the step", flush=True)
                self._compile_modes.pop(0)
                self._release_compile_memory()
                self.fwd = self._make_fwd()
                self.optimizer.zero_grad(set_to_none=not self._ddp)
                self.data.load_state_dict(data_state)
        self.scaler.unscale_(self.optimizer)                # grads back to true scale before clipping
        gnorm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip))
        # global mean loss over all micro-steps of the step (exact with unequal shares too)
        loss_avg = float(D.all_reduce_sum(loss_sum.clone()) / (self.loss_div * self.world))
        loss_ok, grad_ok = math.isfinite(loss_avg), math.isfinite(gnorm)
        # fp16: non-finite GRADIENTS are a routine overflow (the scaler skips the step and halves
        # the scale); only a non-finite LOSS is a real divergence. Without a scaler both are.
        overflow = self.scaler.is_enabled() and loss_ok and not grad_ok
        diverged = (not loss_ok) or (not grad_ok and not overflow)
        if not diverged:
            self.scaler.step(self.optimizer)                # no-op update when grads overflowed
            self.scaler.update()
            self.step += 1
        rec = {"step": self.step, "loss": loss_avg, "lr": lr, "grad_norm": gnorm, "diverged": diverged}
        if not diverged:
            self._rebalance_shares()
        if self.rank_shares and self.world > 1:
            rec["rank_shares"] = list(self.rank_shares)
        if hasattr(self.data, "frac_at") and self.data.frac_at() > 0:
            rec["anneal_frac"] = round(self.data.frac_at(), 4)
        if self.scaler.is_enabled():
            rec["loss_scale"] = float(self.scaler.get_scale())
            if overflow:
                rec["overflow"] = True
        return rec

    # --- checkpoint / resume ---
    def save(self, best: bool = False, metrics: dict | None = None) -> None:
        if self.is_main:                                    # one writer; every rank holds the same weights
            self.ckpt.save(model=self.model, optimizer=self.optimizer, step=self.step,
                           data_state=self.data.state_dict(),
                           model_cfg=dict(vars(self.model.cfg)), train_cfg=self.tcfg,
                           seed=self.seed, best=best, metrics=metrics,
                           extra={"scaler": self.scaler.state_dict()} if self.scaler.is_enabled() else None)
        self._last_save = time.time()

    def _rollback(self) -> bool:
        path = self.ckpt.latest_path()
        if not path:
            return False
        ck = load_checkpoint(path, self.model, self.optimizer, map_location=self.device)
        self.step = int(ck["step"])
        if ck.get("data_state"):
            self.data.load_state_dict(ck["data_state"])
        sc = (ck.get("extra") or {}).get("scaler")
        if sc and self.scaler.is_enabled():
            self.scaler.load_state_dict(sc)
        return True

    def resume_if_available(self) -> bool:
        return self._rollback()

    # --- main loop ---
    def train(self, max_steps: int | None = None) -> None:
        target = self.total_steps if max_steps is None else min(self.total_steps, self.step + max_steps)
        self._t0 = time.time()
        self._log({"step": self.step, "event": "meta", "total_steps": self.total_steps,
                   "tokens_per_step": self.tokens_per_step})
        while self.step < target:
            if self._should_stop():        # GUI pause / Ctrl-C: save and exit cleanly
                self._log({"step": self.step, "event": "paused"})
                break
            rec = self.train_step()
            if rec["diverged"]:
                self._log({**rec, "event": "divergence"})
                if not self._rollback():
                    raise RuntimeError(f"divergence at step {rec['step']} with no checkpoint to roll back to")
                at, n = self._rollbacks_at
                self._rollbacks_at = (self.step, n + 1 if at == self.step else 1)
                if self._rollbacks_at[1] >= 3:   # the same data keeps blowing up: stop instead of looping forever
                    raise RuntimeError(f"diverged 3 times in a row after rolling back to step {self.step}")
                continue
            if self.step % self.log_every == 0:
                dt = time.time() - self._t0
                self._t0 = time.time()
                rec["tok_per_s"] = self.tokens_per_step * self.log_every / max(dt, 1e-6)
                self._log(rec)
            if self.eval_every and self.eval_fn and self.step % self.eval_every == 0:
                if self.is_main:               # eval on rank 0 only; the others wait at the barrier
                    self._log({"step": self.step, "event": "eval", **self.eval_fn(self.model, self.step)})
                    self.model.train()  # eval_fn flips to eval(); restore train so chunked-loss/checkpoint stay active
                D.barrier()
            if time.time() - self._last_save >= self.autosave_s:
                self.save()
        self.save()  # final checkpoint
