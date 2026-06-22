"""DPO (Direct Preference Optimization) - post-training after SFT.

DPO improves response quality from preference pairs (prompt, chosen, rejected) without a
separate reward model or RL loop. It nudges the policy to raise the log-prob of `chosen`
and lower that of `rejected`, relative to a frozen reference (the SFT model), with a KL
anchor so it doesn't drift too far. Loss (Rafailov et al., 2023):

    L = -log sigmoid( beta * [ (logp_pol(chosen) - logp_ref(chosen))
                               - (logp_pol(rejected) - logp_ref(rejected)) ] )

Examples: {"prompt": [messages up to the user turn], "chosen": str, "rejected": str}.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from tokenizer.tokenizer import IM_END
from train.schedule import wsd_lr
from train.trainer import make_optimizer
from train.checkpoint import CheckpointManager, load_checkpoint


def sequence_logprobs(model, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Sum of log-probs of the response tokens (where mask is True) for each sequence.
    Right-padding is safe: causal attention means pad tokens never affect earlier tokens,
    and padded positions are masked out of the sum."""
    logits, _ = model(ids)
    logp = torch.log_softmax(logits.float(), dim=-1)
    tok_logp = logp[:, :-1, :].gather(-1, ids[:, 1:, None]).squeeze(-1)   # logprob of each next token
    m = mask[:, 1:].to(tok_logp.dtype)
    return (tok_logp * m).sum(-1)


def dpo_loss(policy, ref, batch, beta: float = 0.1):
    cids, cm, rids, rm = batch
    pol_c = sequence_logprobs(policy, cids, cm)
    pol_r = sequence_logprobs(policy, rids, rm)
    with torch.no_grad():
        ref_c = sequence_logprobs(ref, cids, cm)
        ref_r = sequence_logprobs(ref, rids, rm)
    logits = beta * ((pol_c - pol_r) - (ref_c - ref_r))
    loss = -F.logsigmoid(logits).mean()
    chosen_reward = float((beta * (pol_c - ref_c)).mean().detach())   # stats are logging-only -> detach
    rejected_reward = float((beta * (pol_r - ref_r)).mean().detach())
    stats = {"dpo_loss": float(loss.detach()), "reward_margin": chosen_reward - rejected_reward,
             "chosen_reward": chosen_reward, "rejected_reward": rejected_reward,
             "acc": float((logits > 0).float().mean())}
    return loss, stats


class DPODataset:
    def __init__(self, examples: list[dict], tok, seq_len: int):
        self.seq_len = seq_len
        self.pad = tok.pad_id
        self.items = []
        for ex in examples:
            p = tok.encode(tok.render_chat(ex["prompt"], add_generation_prompt=True))

            def build(text):
                r = tok.encode(f"{text}{IM_END}\n")
                ids = (p + r)[:seq_len]
                mask = ([False] * len(p) + [True] * len(r))[:seq_len]
                return ids, mask

            cid, cm = build(ex["chosen"])
            rid, rm = build(ex["rejected"])
            self.items.append((cid, cm, rid, rm))
        self.pos = 0

    def __len__(self):
        return len(self.items)

    def _pad(self, seqs, masks):
        T = max(len(s) for s in seqs)
        ids = np.full((len(seqs), T), self.pad, dtype=np.int64)
        mk = np.zeros((len(seqs), T), dtype=bool)
        for i, (s, m) in enumerate(zip(seqs, masks)):
            ids[i, :len(s)] = s
            mk[i, :len(m)] = m
        return torch.from_numpy(ids), torch.from_numpy(mk)

    def next_batch(self, bs: int, device: str = "cpu"):
        batch = [self.items[(self.pos + i) % len(self.items)] for i in range(bs)]
        self.pos = (self.pos + bs) % len(self.items)
        cids, cm = self._pad([b[0] for b in batch], [b[1] for b in batch])
        rids, rm = self._pad([b[2] for b in batch], [b[3] for b in batch])
        return (cids.to(device), cm.to(device), rids.to(device), rm.to(device))

    def state_dict(self):
        return {"pos": self.pos}

    def load_state_dict(self, s):
        self.pos = int(s.get("pos", 0)) % max(1, len(self.items))


class DPOTrainer:
    def __init__(self, policy, ref, data: DPODataset, tcfg: dict, device: str, out_dir: str,
                 beta: float = 0.1, seed: int = 0, stop_file: str | None = None,
                 eval_fn=None, eval_every: int = 0):
        self.policy = policy.to(device)
        self.ref = ref.to(device)
        for p in self.ref.parameters():
            p.requires_grad_(False)
        self.ref.eval()
        self.data = data
        self.device = device
        self.beta = beta
        self.seed = seed
        self.stop_file = stop_file
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.ckpt = CheckpointManager(os.path.join(out_dir, "checkpoints"), keep_last=int(tcfg.get("keep_last_k", 2)))
        self.log_path = os.path.join(out_dir, "metrics.jsonl")

        self.batch_size = int(tcfg.get("batch_size", 8))
        self.grad_accum = int(tcfg.get("grad_accum", 1))
        self.total_steps = int(tcfg["total_steps"])
        self.warmup_steps = int(tcfg.get("warmup_steps", max(1, self.total_steps // 20)))
        self.max_lr = float(tcfg.get("lr", 5e-6))
        self.decay_frac = float(tcfg.get("decay_frac", 0.1))
        self.grad_clip = float(tcfg.get("grad_clip", 1.0))
        self.autosave_s = float(tcfg.get("autosave_minutes", 15)) * 60.0
        self.log_every = int(tcfg.get("log_every", 10))
        self.policy.grad_checkpointing = bool(tcfg.get("grad_checkpointing", False))
        self.optimizer = make_optimizer(self.policy, self.max_lr, tuple(tcfg.get("betas", [0.9, 0.95])),
                                        float(tcfg.get("weight_decay", 0.0)), bool(tcfg.get("optimizer_8bit", False)))
        self.eval_fn = eval_fn
        self.eval_every = int(eval_every or tcfg.get("eval_every", 0))
        from train.trainer import _enable_fast_math
        _enable_fast_math()
        self.step = 0
        self._last_save = time.time()
        self._stop = False

    def request_stop(self):
        self._stop = True

    def _log(self, rec):
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    def step_once(self) -> dict:
        lr = wsd_lr(self.step, total_steps=self.total_steps, warmup_steps=self.warmup_steps,
                    decay_frac=self.decay_frac, max_lr=self.max_lr)
        for g in self.optimizer.param_groups:
            g["lr"] = lr
        self.optimizer.zero_grad(set_to_none=True)
        agg = {}
        for _ in range(self.grad_accum):
            loss, stats = dpo_loss(self.policy, self.ref, self.data.next_batch(self.batch_size, self.device), self.beta)
            (loss / self.grad_accum).backward()
            for k, v in stats.items():
                agg[k] = agg.get(k, 0.0) + v / self.grad_accum
        gnorm = float(torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip))
        import math
        if not (math.isfinite(agg["dpo_loss"]) and math.isfinite(gnorm)):
            return {"step": self.step, "diverged": True, **agg}
        self.optimizer.step()
        self.step += 1
        return {"step": self.step, "loss": agg["dpo_loss"], "lr": lr, "grad_norm": gnorm, "diverged": False, **agg}

    def save(self):
        self.ckpt.save(model=self.policy, optimizer=self.optimizer, step=self.step,
                       data_state=self.data.state_dict(), model_cfg=dict(vars(self.policy.cfg)),
                       train_cfg={"dpo": True, "beta": self.beta}, seed=self.seed)
        self._last_save = time.time()

    def resume_if_available(self) -> bool:
        path = self.ckpt.latest_path()
        if not path:
            return False
        ck = load_checkpoint(path, self.policy, self.optimizer, map_location=self.device)
        self.step = int(ck["step"])
        if ck.get("data_state"):
            self.data.load_state_dict(ck["data_state"])
        return True

    def train(self, max_steps: int | None = None):
        target = self.total_steps if max_steps is None else min(self.total_steps, self.step + max_steps)
        # meta line so the dashboard can draw the projection + progress bar + ETA for DPO too
        tps = self.batch_size * self.grad_accum * 2 * self.data.seq_len   # chosen+rejected tokens/step
        self._log({"step": self.step, "event": "meta", "total_steps": self.total_steps, "tokens_per_step": tps})
        t0 = time.time()
        while self.step < target:
            if self._stop or (self.stop_file and os.path.exists(self.stop_file)):
                self._log({"step": self.step, "event": "paused"})
                break
            rec = self.step_once()
            if rec.get("diverged"):
                self._log({**rec, "event": "divergence"})
                if not self.resume_if_available():
                    raise RuntimeError("DPO diverged with no checkpoint to roll back to")
                continue
            if self.step % self.log_every == 0:
                dt = time.time() - t0
                t0 = time.time()
                rec["tok_per_s"] = tps * self.log_every / max(dt, 1e-6)
                self._log(rec)
            if self.eval_every and self.eval_fn and self.step % self.eval_every == 0:
                self._log({"step": self.step, "event": "eval", **self.eval_fn(self.policy, self.step)})
                self.policy.train()
            if time.time() - self._last_save >= self.autosave_s:
                self.save()
        self.save()


def build_pref_jsonl(out_path: str, n: int = 60000,
                     name: str = "HuggingFaceH4/ultrafeedback_binarized", split: str = "train_prefs") -> int:
    """Stream a preference dataset into {"prompt":[...], "chosen":str, "rejected":str} jsonl
    for DPO (PC; needs datasets)."""
    from datasets import load_dataset
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    ds = load_dataset(name, split=split, streaming=True)
    written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in ds:
            prompt, chosen, rejected = ex.get("prompt"), ex.get("chosen"), ex.get("rejected")
            if not (prompt and chosen and rejected):
                continue
            try:
                c, r = chosen[-1]["content"], rejected[-1]["content"]
            except Exception:
                continue
            if not (c and r):
                continue
            f.write(json.dumps({"prompt": [{"role": "user", "content": prompt}],
                                "chosen": c, "rejected": r}) + "\n")
            written += 1
            if written >= n:
                break
    return written
