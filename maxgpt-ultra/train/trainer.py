"""The MaxGPT-Ultra training loop.

Config-driven (the `train:` block of a YAML). Handles: WSD learning-rate schedule,
bf16 autocast on CUDA (fp32 elsewhere), gradient accumulation to a large effective batch,
gradient clipping, optional z-loss, optional gradient checkpointing + 8-bit AdamW (for
the 1B on 12GB), periodic JSONL metric logging (for the dashboard), time-based autosave,
a divergence guard that rolls back to the last good checkpoint on a NaN/Inf, and exact
resume-from-latest.
"""
from __future__ import annotations

import json
import math
import os
import time
from contextlib import nullcontext

import torch

from .schedule import wsd_lr
from .checkpoint import CheckpointManager, load_checkpoint


def make_optimizer(model, lr, betas, weight_decay, use_8bit=False):
    # weight decay on 2D+ tensors (matmuls/embeddings), none on 1D (norms/gains)
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [{"params": decay, "weight_decay": weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    if use_8bit and torch.cuda.is_available():
        try:
            import bitsandbytes as bnb
            return bnb.optim.AdamW8bit(groups, lr=lr, betas=betas)
        except Exception:
            pass  # fall back to regular AdamW if bitsandbytes is unavailable
    return torch.optim.AdamW(groups, lr=lr, betas=betas)


class Trainer:
    def __init__(self, model, data, tcfg: dict, device: str, out_dir: str,
                 eval_fn=None, seed: int = 0, stop_file: str | None = None):
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
        self.grad_accum = int(tcfg["grad_accum"])
        self.seq_len = data.seq_len
        self.tokens_per_step = self.micro_batch * self.grad_accum * self.seq_len
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

        self.model.grad_checkpointing = bool(tcfg.get("grad_checkpointing", False))
        self.use_amp = (device == "cuda")
        self.optimizer = make_optimizer(self.model, self.max_lr, self.betas, self.wd,
                                        bool(tcfg.get("optimizer_8bit", False)))
        self.step = 0
        self._last_save = time.time()
        self._t0 = time.time()
        self._stop = False           # set by request_stop() (e.g. Ctrl-C)
        self.stop_file = stop_file    # GUI pause: presence of this file => checkpoint + exit

    def request_stop(self) -> None:
        self._stop = True

    def _should_stop(self) -> bool:
        return self._stop or bool(self.stop_file and os.path.exists(self.stop_file))

    # --- helpers ---
    def _set_lr(self, lr: float) -> None:
        for g in self.optimizer.param_groups:
            g["lr"] = lr

    def _micro_forward(self):
        x, y = self.data.next_batch(self.micro_batch, self.device)
        ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
               if self.use_amp else nullcontext())
        with ctx:
            _, loss = self.model(x, y, z_loss_weight=self.z_loss)
        return loss

    def _log(self, rec: dict) -> None:
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    # --- one optimizer step (grad_accum micro-steps) ---
    def train_step(self) -> dict:
        lr = wsd_lr(self.step, total_steps=self.total_steps, warmup_steps=self.warmup_steps,
                    decay_frac=self.decay_frac, max_lr=self.max_lr)
        self._set_lr(lr)
        self.optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for _ in range(self.grad_accum):
            loss = self._micro_forward()
            (loss / self.grad_accum).backward()
            loss_sum += loss.item()
        gnorm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip))
        loss_avg = loss_sum / self.grad_accum
        diverged = not (math.isfinite(loss_avg) and math.isfinite(gnorm))
        if not diverged:
            self.optimizer.step()
            self.step += 1
        return {"step": self.step, "loss": loss_avg, "lr": lr, "grad_norm": gnorm, "diverged": diverged}

    # --- checkpoint / resume ---
    def save(self, best: bool = False, metrics: dict | None = None) -> None:
        self.ckpt.save(model=self.model, optimizer=self.optimizer, step=self.step,
                       data_state=self.data.state_dict(),
                       model_cfg=dict(vars(self.model.cfg)), train_cfg=self.tcfg,
                       seed=self.seed, best=best, metrics=metrics)
        self._last_save = time.time()

    def _rollback(self) -> bool:
        path = self.ckpt.latest_path()
        if not path:
            return False
        ck = load_checkpoint(path, self.model, self.optimizer, map_location=self.device)
        self.step = int(ck["step"])
        if ck.get("data_state"):
            self.data.load_state_dict(ck["data_state"])
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
                continue
            if self.step % self.log_every == 0:
                dt = time.time() - self._t0
                self._t0 = time.time()
                rec["tok_per_s"] = self.tokens_per_step * self.log_every / max(dt, 1e-6)
                self._log(rec)
            if self.eval_every and self.eval_fn and self.step % self.eval_every == 0:
                self._log({"step": self.step, "event": "eval", **self.eval_fn(self.model, self.step)})
            if time.time() - self._last_save >= self.autosave_s:
                self.save()
        self.save()  # final checkpoint
