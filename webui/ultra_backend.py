"""MaxGPT-Ultra family backend for the web UI (MaxGPT-Mini + MaxGPT-Ultra).

These models use the *new* architecture (RoPE / RMSNorm / SwiGLU / GQA), the 49k
UltraTokenizer, ChatML prompts, and the CheckpointManager checkpoint format, none of
which the legacy GPT-2-style loader in app.py understands. Rather than fork app.py's
loader, this module reuses maxgpt-ultra's own (already-tested) classes and exposes a
tiny surface the web UI delegates to for the "ultra" family:

    available()                          -> can we import the maxgpt-ultra code at all?
    resolve_checkpoint(candidates)       -> first existing checkpoint (file or run dir)
    load(ckpt, tok, cfg_yaml, device)    -> {model, tokenizer, context_window, step, stop_id}
    render_chatml(history, msg, tok, ...) -> (ChatML prompt, turns_dropped)
    stream(model, tok, prompt, ...)      -> yields decoded text chunks (token-by-token)

The streaming loop mirrors app.py's legacy one (MaxGPTUltra.forward also returns
(logits, loss) and UltraTokenizer has .encode/.decode), the only real differences are
the ChatML prompt and stopping on the <|im_end|> token id instead of the "USER:" string.
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Generator

import torch

ULTRA_ROOT = Path(__file__).resolve().parent.parent / "maxgpt-ultra"

_M = None   # cached SimpleNamespace of the imported classes


def _mod():
    """Lazily import maxgpt-ultra's classes (adds the package root to sys.path once).
    Cached so repeated calls are free; raises if maxgpt-ultra isn't importable."""
    global _M
    if _M is not None:
        return _M
    if str(ULTRA_ROOT) not in sys.path:
        sys.path.insert(0, str(ULTRA_ROOT))
    from model.model import MaxGPTUltra
    from model.config import ModelConfig
    from tokenizer.tokenizer import UltraTokenizer, IM_END
    _M = types.SimpleNamespace(MaxGPTUltra=MaxGPTUltra, ModelConfig=ModelConfig,
                               UltraTokenizer=UltraTokenizer, IM_END=IM_END)
    return _M


def available() -> bool:
    try:
        _mod()
        return True
    except Exception:
        return False


def resolve_checkpoint(candidates) -> Path | None:
    """Return the first usable checkpoint among `candidates` (in priority order).
    A candidate may be a direct .pt file, or a run dir containing checkpoints/latest.json."""
    for c in candidates:
        c = Path(c)
        if c.is_file():
            return c
        latest = c / "checkpoints" / "latest.json"
        if latest.exists():
            try:
                name = json.loads(latest.read_text())["path"]
            except Exception:
                continue
            p = c / "checkpoints" / name
            if p.exists():
                return p
    return None


def load(ckpt_path, tok_path, config_yaml, device: str) -> dict:
    """Build a MaxGPTUltra from the checkpoint and load the UltraTokenizer.
    The checkpoint is all dicts/tensors (no pickled custom classes), so no sys.modules
    aliasing games are needed, unlike the legacy loader."""
    M = _mod()
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    try:                                            # the trained config is the source of truth
        cfg = M.ModelConfig(**ck["model_cfg"])
    except Exception:
        cfg = M.ModelConfig.from_yaml(str(config_yaml))
    model = M.MaxGPTUltra(cfg)
    sd = ck["model"]
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.to(device).eval()
    tok = M.UltraTokenizer(str(tok_path))
    stop_id = tok.specials.get(M.IM_END, tok.eos_id)   # ChatML assistant turns end with <|im_end|>
    return {"model": model, "tokenizer": tok,
            "context_window": int(getattr(cfg, "seq_len", 1024)),
            "step": int(ck.get("step", -1)), "stop_id": stop_id}


def render_chatml(history: list[dict], user_input: str, tokenizer,
                  context_window: int, max_new_tokens: int) -> tuple[str, int]:
    """Build a ChatML prompt, dropping oldest turns until it fits the token budget."""
    def to_msgs(turns):
        m = []
        for t in turns:
            m.append({"role": "user", "content": t["user"]})
            m.append({"role": "assistant", "content": t["assistant"]})
        m.append({"role": "user", "content": user_input})
        return m

    budget = max(1, context_window - max_new_tokens)
    turns, dropped = list(history), 0
    while True:
        prompt = tokenizer.render_chat(to_msgs(turns), add_generation_prompt=True)
        if len(tokenizer.encode(prompt)) <= budget or not turns:
            return prompt, dropped
        turns.pop(0)
        dropped += 1


def stream(model, tokenizer, prompt: str, device: str, context_window: int, *,
           max_new_tokens: int, temperature: float, top_k: int,
           repetition_penalty: float = 1.15, repetition_window: int = 64,
           stop_id: int | None = None) -> Generator[str, None, None]:
    """Token-by-token streaming for the ultra family. Stops cleanly on the stop token
    (no string holdback needed) and never renders special tokens."""
    ids = tokenizer.encode(prompt)
    tokens = torch.tensor([ids], dtype=torch.long, device=device)
    gen: list[int] = []
    decoded, last = "", 0
    with torch.no_grad():
        for _ in range(max_new_tokens):
            inp = tokens[:, -context_window:] if tokens.size(1) > context_window else tokens
            logits, _ = model(inp)
            logits = logits[:, -1, :]
            if repetition_penalty != 1.0:
                recent = tokens[0, -repetition_window:] if tokens.size(1) > repetition_window else tokens[0]
                uniq = torch.unique(recent)
                rl = logits[0, uniq]
                logits[0, uniq] = torch.where(rl > 0, rl / repetition_penalty, rl * repetition_penalty)
            logits = logits / max(1e-6, temperature)
            if top_k:
                tv, _ = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
                logits = torch.where(logits < tv[:, -1:], float("-inf"), logits)
            probs = torch.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            tid = int(nxt.item())
            if stop_id is not None and tid == stop_id:
                break
            tokens = torch.cat([tokens, nxt], dim=1)
            gen.append(tid)
            decoded = tokenizer.decode(gen, skip_special=True)
            if len(decoded) > last:
                yield decoded[last:]
                last = len(decoded)
    if last < len(decoded):
        yield decoded[last:]
