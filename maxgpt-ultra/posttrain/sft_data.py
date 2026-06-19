"""SFT (supervised fine-tuning) data.

Formats chat examples with the ChatML template and applies **assistant-only loss
masking**: the model is supervised only on the assistant's reply tokens, not on the
system/user prompt. Packs the masked stream into the same `(x, y)` batch interface the
trainer already uses (y is -100 wherever loss should be ignored).

Examples are dicts: {"messages": [{"role": "system"|"user"|"assistant", "content": str}, ...]}.
"""
from __future__ import annotations

import json

import numpy as np
import torch

from tokenizer.tokenizer import IM_START, IM_END


def encode_chat_example(messages: list[dict], tok) -> tuple[list[int], list[bool]]:
    """Return (token_ids, supervise_flags). supervise[i] is True iff token i is part of
    an assistant reply (and should contribute to the loss)."""
    tokens: list[int] = []
    sup: list[bool] = []
    for m in messages:
        header = tok.encode(f"{IM_START}{m['role']}\n")
        body = tok.encode(f"{m['content']}{IM_END}\n")
        tokens += header
        sup += [False] * len(header)            # never supervise the role header
        is_assistant = (m["role"] == "assistant")
        tokens += body
        sup += [is_assistant] * len(body)        # supervise assistant content + its <|im_end|>
    return tokens, sup


class SFTDataset:
    """Packs masked chat examples into fixed-length windows. Same interface as
    PackedShardDataset, so the existing Trainer drives SFT unchanged."""

    def __init__(self, examples: list[dict], tok, seq_len: int):
        self.seq_len = seq_len
        toks: list[int] = []
        sups: list[bool] = []
        for ex in examples:
            t, s = encode_chat_example(ex["messages"], tok)
            t.append(tok.eos_id)
            s.append(False)                      # separator between examples
            toks += t
            sups += s
        self.tokens = np.asarray(toks, dtype=np.int64)
        self.sup = np.asarray(sups, dtype=bool)
        self.n = len(self.tokens)
        assert self.n > seq_len + 1, "not enough SFT tokens for one window"
        self.pos = 0
        self.epoch = 0

    def next_batch(self, batch_size: int, device: str = "cpu"):
        xs, ys = [], []
        for _ in range(batch_size):
            if self.pos + self.seq_len + 1 > self.n:
                self.pos = 0
                self.epoch += 1
            wt = self.tokens[self.pos:self.pos + self.seq_len + 1]
            ws = self.sup[self.pos:self.pos + self.seq_len + 1]
            x = wt[:-1].copy()
            y = wt[1:].copy()
            y[~ws[1:]] = -100                    # supervise only assistant tokens
            xs.append(x)
            ys.append(y)
            self.pos += self.seq_len
        x = torch.from_numpy(np.stack(xs))
        y = torch.from_numpy(np.stack(ys))
        return x.to(device), y.to(device)

    def state_dict(self) -> dict:
        return {"pos": int(self.pos), "epoch": int(self.epoch)}

    def load_state_dict(self, s: dict) -> None:
        self.pos = int(s.get("pos", 0)) % self.n
        self.epoch = int(s.get("epoch", 0))


def load_chat_jsonl(path: str) -> list[dict]:
    """Local chat data: one JSON object per line, each {"messages": [...]}."""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def load_sft_hf(name: str = "HuggingFaceH4/ultrachat_200k", split: str = "train_sft", n: int | None = None):
    """Load an instruction/chat set from HuggingFace into the messages format (PC; needs
    `datasets`). UltraChat and OASST already store a 'messages' list; adjust per dataset."""
    from datasets import load_dataset
    ds = load_dataset(name, split=split, streaming=(n is None))
    out = []
    for i, ex in enumerate(ds):
        if n is not None and i >= n:
            break
        if "messages" in ex:
            out.append({"messages": ex["messages"]})
    return out


def build_sft_jsonl(out_path: str, n: int = 100000,
                    name: str = "HuggingFaceH4/ultrachat_200k", split: str = "train_sft") -> int:
    """Stream a chat dataset into a {"messages": [...]} jsonl for SFT (PC; needs datasets)."""
    import os
    from datasets import load_dataset
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    ds = load_dataset(name, split=split, streaming=True)
    written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in ds:
            msgs = ex.get("messages")
            if not msgs:
                continue
            f.write(json.dumps({"messages": msgs}) + "\n")
            written += 1
            if written >= n:
                break
    return written
