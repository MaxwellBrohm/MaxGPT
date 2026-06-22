"""Packed, resumable data loader over the memmapped token shards.

- **Packed**: documents are read as one continuous token stream and sliced into
  fixed `seq_len` windows, so there is no padding waste (windows flow across document
  boundaries, with the EOT marker separating docs).
- **Resumable**: the read position is a single integer we can save and restore, so
  pause/resume continues over the exact same data without replaying tokens. Shards are
  expected to be written from an already-shuffled stream, so reading sequentially is
  both correct and trivially resumable.

next_batch returns (x, y) where y is x shifted by one token (the next-token targets).
"""
from __future__ import annotations

import json
import os

import numpy as np

DTYPE = np.uint16


class PackedShardDataset:
    def __init__(self, data_dir: str, seq_len: int):
        with open(os.path.join(data_dir, "meta.json")) as f:
            meta = json.load(f)
        self.seq_len = seq_len
        self.shards = [
            np.memmap(os.path.join(data_dir, s["name"]), dtype=DTYPE, mode="r")
            for s in meta["shards"]
        ]
        self.shard_lens = [len(s) for s in self.shards]
        self.total = int(sum(self.shard_lens))
        self.cum = np.cumsum([0] + self.shard_lens)  # cum[i] = global start of shard i
        assert self.total > self.seq_len + 1, "not enough tokens for even one window"
        self.pos = 0
        self.epoch = 0

    def _read(self, start: int, n: int) -> np.ndarray:
        """Read n tokens starting at global index `start`, wrapping across shards/end."""
        out = np.empty(n, dtype=DTYPE)
        got = 0
        while got < n:
            gi = (start + got) % self.total
            si = int(np.searchsorted(self.cum, gi, side="right") - 1)
            local = gi - int(self.cum[si])
            take = min(n - got, self.shard_lens[si] - local)
            out[got:got + take] = self.shards[si][local:local + take]
            got += take
        return out

    def next_batch(self, batch_size: int, device: str = "cpu"):
        import torch
        xs, ys = [], []
        for _ in range(batch_size):
            chunk = self._read(self.pos, self.seq_len + 1).astype(np.int64)
            xs.append(chunk[:-1])
            ys.append(chunk[1:])
            self.pos += self.seq_len
            if self.pos >= self.total:
                self.pos -= self.total
                self.epoch += 1
        x = torch.from_numpy(np.stack(xs))
        y = torch.from_numpy(np.stack(ys))
        if device == "cuda":            # pinned + async copy overlaps the host->device transfer with compute
            return (x.pin_memory().to(device, non_blocking=True),
                    y.pin_memory().to(device, non_blocking=True))
        return x.to(device), y.to(device)

    # --- resume support (data-position tracking) ---
    def state_dict(self) -> dict:
        return {"pos": int(self.pos), "epoch": int(self.epoch)}

    def load_state_dict(self, state: dict) -> None:
        self.pos = int(state["pos"]) % self.total
        self.epoch = int(state.get("epoch", 0))
