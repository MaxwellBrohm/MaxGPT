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
    """Shards are memory-mapped LAZILY, a few at a time (sequential reading only ever touches one
    or two), so a 1,000-shard build does not need 1,000 open files: the per-process limit on a
    typical Linux login is 1,024, and mapping everything up front hit it on the Lambda box."""

    MAX_OPEN = 8

    def __init__(self, data_dir: str, seq_len: int):
        with open(os.path.join(data_dir, "meta.json")) as f:
            meta = json.load(f)
        self.seq_len = seq_len
        self.paths = [os.path.join(data_dir, s["name"]) for s in meta["shards"]]
        self.shard_lens = [int(s["tokens"]) for s in meta["shards"]]
        self._open: dict[int, np.memmap] = {}          # shard index -> memmap, small LRU
        self.total = int(sum(self.shard_lens))
        self.cum = np.cumsum([0] + self.shard_lens)  # cum[i] = global start of shard i
        assert self.total > self.seq_len + 1, "not enough tokens for even one window"
        self.pos = 0
        self.epoch = 0
        self.rank, self.world = 0, 1     # multi-GPU: see shard()
        self.shares = None               # unequal per-rank shares: see shard(shares=...)
        self._block_i = 0                # windows taken from the current step block (block mode)

    def shard(self, rank: int, world: int, shares=None) -> None:
        """Multi-GPU: rank r reads window r, r+world, r+2*world, ... of the stream. `pos` stays
        the GLOBAL stream position (identical on every rank, so checkpoints are the same file
        regardless of the GPU count), and one optimizer step across all ranks consumes exactly
        the windows a single GPU would have, so the gradient is the same average.

        shares=[n_0, ..., n_{world-1}]: unequal shares for cards of unequal speed (a throttled
        card gets fewer windows per step so the fast ones stop waiting for it). Every step is a
        block of sum(shares) consecutive windows; rank r takes the shares[r] windows starting at
        sum(shares[:r]). `pos` still advances identically on every rank (by the whole block)."""
        assert 0 <= rank < world
        self.rank, self.world = rank, world
        if shares is not None:
            assert len(shares) == world and all(int(n) >= 1 for n in shares), shares
            self.shares = [int(n) for n in shares]
            self._block_i = 0

    def _shard(self, si: int) -> np.memmap:
        mm = self._open.pop(si, None)
        if mm is None:
            mm = np.memmap(self.paths[si], dtype=DTYPE, mode="r")
            assert len(mm) == self.shard_lens[si], f"{self.paths[si]}: {len(mm)} tokens on disk, meta says {self.shard_lens[si]}"
            while len(self._open) >= self.MAX_OPEN:         # evict the least recently used
                old = next(iter(self._open))
                del self._open[old]
        self._open[si] = mm                                  # (re)insert as most recently used
        return mm

    def _read(self, start: int, n: int) -> np.ndarray:
        """Read n tokens starting at global index `start`, wrapping across shards/end."""
        out = np.empty(n, dtype=DTYPE)
        got = 0
        while got < n:
            gi = (start + got) % self.total
            si = int(np.searchsorted(self.cum, gi, side="right") - 1)
            local = gi - int(self.cum[si])
            take = min(n - got, self.shard_lens[si] - local)
            out[got:got + take] = self._shard(si)[local:local + take]
            got += take
        return out

    def _advance(self, n_windows: int) -> None:
        self.pos += n_windows * self.seq_len
        while self.pos >= self.total:
            self.pos -= self.total
            self.epoch += 1

    def next_batch(self, batch_size: int, device: str = "cpu"):
        import torch
        xs, ys = [], []
        for _ in range(batch_size):
            if self.shares is not None:                     # block mode (unequal shares)
                block = sum(self.shares)
                mine = self.shares[self.rank]
                start = self.pos + (sum(self.shares[:self.rank]) + self._block_i) * self.seq_len
                chunk = self._read(start, self.seq_len + 1).astype(np.int64)
                self._block_i += 1
                if self._block_i >= mine:                    # last window of my share: the block is done for me
                    self._advance(block)
                    self._block_i = 0
            else:
                chunk = self._read(self.pos + self.rank * self.seq_len, self.seq_len + 1).astype(np.int64)
                self._advance(self.world)
            xs.append(chunk[:-1])
            ys.append(chunk[1:])
        x = torch.from_numpy(np.stack(xs))
        y = torch.from_numpy(np.stack(ys))
        if device == "cuda":            # pinned + async copy overlaps the host->device transfer with compute
            return (x.pin_memory().to(device, non_blocking=True),
                    y.pin_memory().to(device, non_blocking=True))
        return x.to(device), y.to(device)

    # --- resume support (data-position tracking) ---
    def state_dict(self) -> dict:
        # block_i is this rank's position inside the current step block (unequal shares); it is 0 at
        # every step boundary, which is where checkpoints are taken, so saving it is safe on every rank
        return {"pos": int(self.pos), "epoch": int(self.epoch), "block_i": int(self._block_i)}

    def load_state_dict(self, state: dict) -> None:
        self.pos = int(state["pos"]) % self.total
        self.epoch = int(state.get("epoch", 0))
        self._block_i = int(state.get("block_i", 0))
