"""Atomic, resumable checkpointing.

A checkpoint captures everything needed for exact continuation: model weights, optimizer
state (Adam moments), step, the data-loader position, the configs, the seed, and RNG
state. Writes are atomic (temp file + os.replace) so a crash or power loss mid-save can
never corrupt a checkpoint. We keep the last K plus an optional best, and a `latest.json`
pointer for resume-from-latest.
"""
from __future__ import annotations

import glob
import json
import os

import torch


def _atomic_save(obj, path: str) -> None:
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)   # atomic on the same filesystem


def _atomic_save_json(obj, path: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


class CheckpointManager:
    def __init__(self, out_dir: str, keep_last: int = 3):
        self.dir = out_dir
        self.keep_last = keep_last
        os.makedirs(out_dir, exist_ok=True)

    def save(self, *, model, optimizer, step: int, data_state: dict, model_cfg: dict,
             train_cfg: dict, seed: int, best: bool = False, metrics: dict | None = None) -> str:
        payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "data_state": data_state,
            "model_cfg": model_cfg,
            "train_cfg": train_cfg,
            "seed": seed,
            "rng_torch": torch.get_rng_state(),
            "metrics": metrics or {},
        }
        if torch.cuda.is_available():
            payload["rng_cuda"] = torch.cuda.get_rng_state_all()
        path = os.path.join(self.dir, f"ckpt_{step:08d}.pt")
        _atomic_save(payload, path)
        _atomic_save_json({"path": os.path.basename(path), "step": step},
                          os.path.join(self.dir, "latest.json"))
        if best:
            _atomic_save(payload, os.path.join(self.dir, "best.pt"))
        self._prune()
        return path

    def _prune(self) -> None:
        cks = sorted(glob.glob(os.path.join(self.dir, "ckpt_*.pt")))
        for p in cks[:-self.keep_last] if self.keep_last > 0 else []:
            try:
                os.remove(p)
            except OSError:
                pass

    def latest_path(self) -> str | None:
        p = os.path.join(self.dir, "latest.json")
        if not os.path.exists(p):
            return None
        with open(p) as f:
            name = json.load(f)["path"]
        full = os.path.join(self.dir, name)
        return full if os.path.exists(full) else None


def load_checkpoint(path: str, model, optimizer=None, map_location: str = "cpu") -> dict:
    ck = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ck["model"])
    if optimizer is not None and ck.get("optimizer"):
        optimizer.load_state_dict(ck["optimizer"])
    return ck
