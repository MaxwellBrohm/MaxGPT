"""SFT (supervised fine-tuning) data.

Formats chat examples with the ChatML template and applies **assistant-only loss
masking**: the model is supervised only on the assistant's reply tokens, not on the
system/user prompt. Packs the masked stream into the same `(x, y)` batch interface the
trainer already uses (y is -100 wherever loss should be ignored).

Examples are dicts: {"messages": [{"role": "system"|"user"|"assistant", "content": str}, ...]}.
"""
from __future__ import annotations

import json
import os
import re

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
        self.rank, self.world = 0, 1     # multi-GPU: see shard()

    def shard(self, rank: int, world: int) -> None:
        """Multi-GPU: rank r reads window r of every group of `world` consecutive windows; `pos`
        stays the global position (same on every rank). Same scheme as PackedShardDataset."""
        assert 0 <= rank < world
        self.rank, self.world = rank, world

    def next_batch(self, batch_size: int, device: str = "cpu"):
        xs, ys = [], []
        for _ in range(batch_size):
            if self.pos + self.world * self.seq_len + 1 > self.n:   # the whole rank-group must fit
                self.pos = 0
                self.epoch += 1
            s = self.pos + self.rank * self.seq_len
            wt = self.tokens[s:s + self.seq_len + 1]
            ws = self.sup[s:s + self.seq_len + 1]
            x = wt[:-1].copy()
            y = wt[1:].copy()
            y[~ws[1:]] = -100                    # supervise only assistant tokens
            xs.append(x)
            ys.append(y)
            self.pos += self.world * self.seq_len
        x = torch.from_numpy(np.stack(xs))
        y = torch.from_numpy(np.stack(ys))
        return x.to(device), y.to(device)

    def state_dict(self) -> dict:
        return {"pos": int(self.pos), "epoch": int(self.epoch)}

    def load_state_dict(self, s: dict) -> None:
        self.pos = int(s.get("pos", 0)) % self.n
        self.epoch = int(s.get("epoch", 0))


class ReplayBlend:
    """SFT batches with a share of plain pretraining windows mixed in (research report 3.1:
    replaying 5-20% of pretraining data into SFT measured up to 1.87x target-data efficiency, and
    the gain is largest exactly when the target data was scarce in pretraining, as here). Every
    batch takes round(frac * B) windows from `replay` (a PackedShardDataset: ordinary next-token
    targets, nothing masked) and the rest from `main` (the SFTDataset). Same interface as both."""

    def __init__(self, main, replay, frac: float = 0.1):
        assert 0.0 <= frac < 1.0, frac
        self.main, self.replay, self.frac = main, replay, float(frac)
        self.seq_len = main.seq_len

    def n_replay(self, batch_size: int) -> int:
        return min(batch_size - 1, int(round(self.frac * batch_size))) if self.frac > 0 else 0

    def shard(self, rank: int, world: int, shares=None) -> None:
        self.main.shard(rank, world)
        self.replay.shard(rank, world)

    def next_batch(self, batch_size: int, device: str = "cpu"):
        k = self.n_replay(batch_size)
        x, y = self.main.next_batch(batch_size - k, device)
        if k:
            xr, yr = self.replay.next_batch(k, device)
            x, y = torch.cat([x, xr]), torch.cat([y, yr])
        return x, y

    @property
    def epoch(self):
        return self.main.epoch

    @property
    def n(self):
        return self.main.n

    def state_dict(self) -> dict:
        return {"blend": True, "main": self.main.state_dict(), "replay": self.replay.state_dict()}

    def load_state_dict(self, s: dict) -> None:
        if s.get("blend"):
            self.main.load_state_dict(s["main"])
            self.replay.load_state_dict(s["replay"])
        else:                                    # a state saved before replay existed
            self.main.load_state_dict(s)


class Decontaminator:
    """Drops training text that overlaps the evaluation suite. n-gram (default 13 words, the
    lm-eval convention) hashes of every text field in data/bench/*.jsonl are collected once;
    is_contaminated(text) is True when any n-gram of the text is among them. The research
    report's instruction: re-run decontamination against what WE evaluate on, since the
    datasets' own decontamination targeted their eval sets."""

    _FIELDS = ("context", "target", "continuation", "choices", "contexts")

    def __init__(self, bench_dir: str, n: int = 13):
        self.n = n
        self.grams: set[int] = set()
        self.sources = 0
        if not os.path.isdir(bench_dir):
            return
        for fn in sorted(os.listdir(bench_dir)):
            if not fn.endswith(".jsonl"):
                continue
            with open(os.path.join(bench_dir, fn), encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    for k in self._FIELDS:
                        v = row.get(k)
                        for t in (v if isinstance(v, list) else [v]):
                            if isinstance(t, str):
                                self.grams.update(self._ngrams(t))
                    self.sources += 1

    @staticmethod
    def _words(text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())

    def _ngrams(self, text: str):
        w = self._words(text)
        return (hash(tuple(w[i:i + self.n])) for i in range(0, len(w) - self.n + 1))

    def is_contaminated(self, text: str) -> bool:
        return self.grams and any(g in self.grams for g in self._ngrams(text))

    def conversation_contaminated(self, messages: list[dict]) -> bool:
        return any(self.is_contaminated(m.get("content") or "") for m in messages)


# SmolTalk2, no-think subsets only (docs/research_2026-09-22.md 3.1): the SFT mix that measured
# best for a small base model in the one controlled comparison, minus the parts a 1.1B model at
# seq 2048 cannot use (thinking traces, 64k long-context, tool calling) and the multilingual
# subsets (this model is English). Counts are per subset; the total is ~350k conversations,
# in the report's 200k-400k range. OpenAssistant is appended on top for human-written chat.
SMOLTALK2_MIX = {
    "smoltalk_smollm3_smol_magpie_ultra_no_think": 120_000,       # general instructions + chat
    "OpenHermes_2.5_no_think": 60_000,                            # broad instruction following
    "smoltalk_smollm3_systemchats_30k_no_think": 34_000,          # system-prompt persona chats (all)
    "smoltalk_smollm3_everyday_conversations_no_think": 2_300,    # small talk (all)
    "smoltalk_smollm3_smol_rewrite_no_think": 20_000,
    "smoltalk_smollm3_smol_summarize_no_think": 20_000,
    "smoltalk_smollm3_explore_instruct_rewriting_no_think": 10_000,
    "tulu_3_sft_personas_instruction_following_no_think": 30_000, # IFEval-style constraints (all)
    "Mixture_of_Thoughts_science_no_think": 15_000,
    "OpenThoughts3_1.2M_no_think_no_think": 30_000,               # math / code answers, no traces
    "table_gpt_no_think": 5_000,
}
_ROLES = {"system", "user", "assistant"}


def clean_messages(messages) -> list[dict] | None:
    """The conversation as [{role, content}] with only system/user/assistant turns, non-empty
    content, ending in an assistant turn; None if it cannot be used."""
    if not isinstance(messages, list):
        return None
    out = []
    for m in messages:
        role, content = (m or {}).get("role"), (m or {}).get("content")
        if role not in _ROLES or not isinstance(content, str) or not content.strip():
            return None
        out.append({"role": role, "content": content})
    if not out or out[-1]["role"] != "assistant" or not any(m["role"] == "user" for m in out):
        return None
    return out


def build_sft_jsonl_smoltalk2(out_path: str, mix: dict | None = None, seed: int = 0,
                              decontaminator: Decontaminator | None = None, append: bool = False) -> dict:
    """Stream the chosen SmolTalk2 no-think subsets into a {"messages": [...]} jsonl (needs
    datasets + network). Each subset is shuffled with a buffer before taking its count, rows are
    cleaned by clean_messages, and rows overlapping the evaluation suite are dropped when a
    Decontaminator is given. Returns {subset: written, ..., "_dropped_contaminated": n}."""
    from datasets import load_dataset
    import os as _os
    mix = dict(mix or SMOLTALK2_MIX)
    _os.makedirs(_os.path.dirname(out_path) or ".", exist_ok=True)
    counts, dropped = {}, 0
    with open(out_path, "a" if append else "w", encoding="utf-8") as f:
        for subset, n in mix.items():
            ds = load_dataset("HuggingFaceTB/smoltalk2", "SFT", split=subset, streaming=True)
            ds = ds.shuffle(seed=seed, buffer_size=10_000)
            written = 0
            for ex in ds:
                msgs = clean_messages(ex.get("messages"))
                if msgs is None:
                    continue
                if decontaminator is not None and decontaminator.conversation_contaminated(msgs):
                    dropped += 1
                    continue
                f.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")
                written += 1
                if written >= n:
                    break
            counts[subset] = written
            print(f"[sft] {subset}: {written:,} conversations", flush=True)
    counts["_dropped_contaminated"] = dropped
    return counts


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


def _oasst_threads(rows, lang: str = "en") -> list[dict]:
    """Reconstruct OpenAssistant's message TREE into flat chat threads.

    OASST rows are tree nodes (message_id / parent_id / role / text / rank). We index
    them, then from each English prompter root we walk down, always taking the best-ranked
    reply at each step, building [user, assistant, user, ...] threads. Pure function (takes
    a list of row dicts) so it's testable without the network."""
    nodes, children = {}, {}
    for r in rows:
        if r.get("deleted"):
            continue
        if lang and r.get("lang") != lang:
            continue
        mid = r.get("message_id")
        if not mid:
            continue
        nodes[mid] = r
        children.setdefault(r.get("parent_id"), []).append(mid)

    def rank_key(i):                      # rank 0 = best; missing rank sorts last
        rk = nodes[i].get("rank")
        return rk if rk is not None else 1e9

    threads = []
    roots = [m for m in nodes if nodes[m].get("parent_id") is None and nodes[m].get("role") == "prompter"]
    for root in roots:
        msgs, cur, expect = [], root, "prompter"
        while cur in nodes and nodes[cur].get("role") == expect:
            msgs.append({"role": "user" if expect == "prompter" else "assistant",
                         "content": nodes[cur].get("text") or ""})
            nxt = "assistant" if expect == "prompter" else "prompter"
            kids = [k for k in children.get(cur, []) if k in nodes and nodes[k].get("role") == nxt]
            if not kids:
                break
            cur, expect = sorted(kids, key=rank_key)[0], nxt
        if msgs and msgs[-1]["role"] == "user":      # a dangling prompt with no reply teaches nothing
            msgs.pop()                                # (26% of threads in the 2026-09-26 dry run)
        msgs = clean_messages(msgs)
        if msgs is not None and len(msgs) >= 2:
            threads.append({"messages": msgs})
    return threads


def append_oasst_jsonl(out_path: str, names=("OpenAssistant/oasst1", "OpenAssistant/oasst2"),
                       lang: str = "en") -> int:
    """APPEND OpenAssistant conversations onto an existing SFT jsonl (adds the casual,
    varied chat that UltraChat lacks). OASST is small, so we take all English threads
    (PC; needs `datasets`). Returns how many threads were appended."""
    from datasets import load_dataset
    written = 0
    with open(out_path, "a", encoding="utf-8") as f:
        for name in names:
            try:
                ds = load_dataset(name, split="train")
            except Exception as e:
                print(f"[sft] WARNING: could not load {name}: {type(e).__name__}: {e}")
                continue
            for t in _oasst_threads(list(ds), lang):
                f.write(json.dumps(t) + "\n")
                written += 1
    return written
