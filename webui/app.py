"""
MaxGPT Web UI — chat with and compare from-scratch language models.

Run from this directory:
    streamlit run app.py
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import datetime
import queue
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator

import streamlit as st
import torch

import ultra_backend   # MaxGPT-Mini / MaxGPT-Ultra (new-architecture "ultra" family)


# ============================================================================
# Model registry
#
# Each entry is one selectable model in the UI. MaxGPT-3.5 shares the maxgpt-3
# folder (same architecture + tokenizer, produced by finetune.py which saves
# its weights as `final_sft.pt` next to the base `final.pt`).
#
# The `summary` + `bullets` fields are shown in an "About this model" panel
# so anyone using the UI can see *why* the models behave differently —
# architecture deltas, training-data deltas, base vs SFT.
# ============================================================================

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class ModelSpec:
    folder: Path
    checkpoint_filename: str   # "final.pt" or "final_sft.pt" — sits in folder/checkpoints/
    namespace: str             # sys.modules namespace for the import-collision loader
    color: str                 # accent hex used in dots, badges, compare borders
    blurb: str                 # 1-3 words shown inside the badge
    summary: str               # one-line architecture summary
    bullets: tuple             # 2-3 bullets explaining what's different vs. the others
    # --- "ultra" family only (MaxGPT-Mini / MaxGPT-Ultra): new arch, ChatML, 49k tokenizer ---
    family: str = "legacy"             # "legacy" (GPT-2 style) or "ultra" (MaxGPTUltra)
    ckpt_candidates: tuple = ()        # checkpoints to try in order (a .pt file or a run dir)
    tokenizer_candidates: tuple = ()   # UltraTokenizer json paths to try in order
    config_yaml: object = None         # model config (fallback if the checkpoint lacks one)
    approx_params: int = 0             # param count for the arch panel (no GPT-2 formula)


MODEL_SPECS: dict[str, ModelSpec] = {
    "MaxGPT-1": ModelSpec(
        folder=ROOT / "maxgpt-1",
        checkpoint_filename="final.pt",
        namespace="maxgpt_1",
        color="#10B981",  # emerald
        blurb="pre-trained baseline",
        summary="23M params · 6 layers × 384 dim · 6 heads · ctx 256",
        bullets=(
            "Smallest baseline. Trained on 3 datasets: TinyStories, OASST1, UltraChat.",
            "~500M token-passes (1 epoch, Chinchilla-optimal for 23M params).",
            "Short 256-token context. Outputs are short and simple — proves the stack works.",
        ),
    ),
    "MaxGPT-2": ModelSpec(
        folder=ROOT / "maxgpt-2",
        checkpoint_filename="final.pt",
        namespace="maxgpt_2",
        color="#3B82F6",  # blue
        blurb="scaled pre-trained",
        summary="110M params · 12 layers × 768 dim · 12 heads · ctx 512",
        bullets=(
            "~4.8× the parameters of MaxGPT-1. 2× deeper, 2× wider, 2× context (512).",
            "Same 3-dataset mix as MaxGPT-1 (TinyStories + OASST1 + UltraChat), "
            "trained for ~2B token-passes (4 epochs ≈ Chinchilla-optimal for 110M).",
            "Visibly better multi-turn coherence and broader knowledge than MaxGPT-1.",
        ),
    ),
    "MaxGPT-3": ModelSpec(
        folder=ROOT / "maxgpt-3",
        checkpoint_filename="final.pt",
        namespace="maxgpt_3",
        color="#8B5CF6",  # violet
        blurb="large pre-trained",
        summary="235M params · 16 layers × 1024 dim · 16 heads · ctx 1024",
        bullets=(
            "~2.1× the parameters of MaxGPT-2. 16 layers × 1024 dim, 2× context (1024).",
            "7-dataset mix (~5B unique tokens): TinyStories, OASST1+2, UltraChat, "
            "Cosmopedia v2, OpenOrca, WildChat-1M, HH-RLHF.",
            "Targets ~20B token-passes (~4× Chinchilla for 235M) for maximum output quality.",
        ),
    ),
    "MaxGPT-3.5": ModelSpec(
        folder=ROOT / "maxgpt-3",  # shares maxgpt-3's code + tokenizer
        checkpoint_filename="final_sft.pt",
        namespace="maxgpt_3",
        color="#F59E0B",  # amber
        blurb="SFT chat-tuned",
        summary="235M params · same architecture as MaxGPT-3 · SFT-fine-tuned",
        bullets=(
            "Starts from MaxGPT-3's weights, then continues on chat-only data "
            "(~2.4B tokens, 300K SFT steps at lr=5e-5, loss masked to assistant turns).",
            "Best at sustained conversation: resolves cross-turn references "
            "(\"feed HIM\" → the puppy) ~4× more reliably than base, and stays on-topic in follow-ups.",
            "A trade-off (per blind eval): more assistant-like and factually structured, but "
            "more verbose — base MaxGPT-3 still wins on concise, open-ended single-turn replies.",
        ),
    ),
    "MaxGPT-Mini": ModelSpec(
        folder=ROOT / "maxgpt-ultra",
        checkpoint_filename="",            # ultra family resolves its own checkpoint
        namespace="maxgpt_mini",
        color="#06B6D4",  # cyan
        blurb="modern · compact",
        summary="113M params · RoPE/RMSNorm/SwiGLU/GQA · 49k vocab · SFT+DPO+RAG",
        bullets=(
            "First of the new generation: modern decoder (RoPE, RMSNorm, SwiGLU, GQA, QK-norm) "
            "instead of the GPT-2-style stack used by MaxGPT-1/2/3.",
            "Pretrained on FineWeb-Edu + Cosmopedia + code + math (49k tokenizer), then SFT + DPO, "
            "with retrieval (RAG) available at inference.",
            "Half the size of MaxGPT-3 (113M vs 235M) but built to punch above it on clean output, "
            "code, and grounded answers.",
        ),
        family="ultra",
        ckpt_candidates=(
            ROOT / "maxgpt-ultra" / "models" / "mini.pt",
            ROOT / "maxgpt-ultra" / "runs" / "dpo",
            ROOT / "maxgpt-ultra" / "runs" / "sft",
            ROOT / "maxgpt-ultra" / "runs" / "pretrain",
        ),
        tokenizer_candidates=(
            ROOT / "maxgpt-ultra" / "models" / "mini.tokenizer.json",
            ROOT / "maxgpt-ultra" / "tokenizer" / "maxgpt-ultra.tokenizer.json",
        ),
        config_yaml=ROOT / "maxgpt-ultra" / "configs" / "shakedown.yaml",
        approx_params=113_000_000,
    ),
    "MaxGPT-Ultra": ModelSpec(
        folder=ROOT / "maxgpt-ultra",
        checkpoint_filename="",
        namespace="maxgpt_ultra",
        color="#EF4444",  # red
        blurb="modern · flagship",
        summary="~1.1B params · same modern arch as Mini · ~100B tokens · SFT+DPO+RAG",
        bullets=(
            "The flagship: ~1.1B params trained on ~100B tokens of FineWeb-Edu / Cosmopedia / code / math.",
            "Same modern architecture as MaxGPT-Mini, scaled ~10x. Built to be the first genuinely "
            "daily-usable model in the lineage.",
            "Copy its finished checkpoint to maxgpt-ultra/models/ultra.pt (and its tokenizer) to load it here.",
        ),
        family="ultra",
        ckpt_candidates=(
            ROOT / "maxgpt-ultra" / "models" / "ultra.pt",
        ),
        tokenizer_candidates=(
            ROOT / "maxgpt-ultra" / "models" / "ultra.tokenizer.json",
            ROOT / "maxgpt-ultra" / "tokenizer" / "maxgpt-ultra.tokenizer.json",
        ),
        config_yaml=ROOT / "maxgpt-ultra" / "configs" / "ultra.yaml",
        approx_params=1_100_000_000,
    ),
}

STOP_STRING = "USER:"


# ============================================================================
# Import collision handling
# ----------------------------------------------------------------------------
# Each maxgpt-N/ folder has its own config.py, model.py, tokenizer.py with the
# SAME class names (Config, Transformer, BPETokenizer). A normal
# `from config import Config` would shadow whichever was imported last.
#
# Fix: use importlib.util to register each model's files in sys.modules under a
# uniquely-namespaced name ("maxgpt_1.model" vs "maxgpt_2.model" vs ...).
#
# Works because the three .py files in each folder are self-contained — they
# don't import from each other, only from stdlib and torch.
# ============================================================================

def _load_file_as_module(file_path: Path, unique_name: str):
    spec = importlib.util.spec_from_file_location(unique_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[unique_name] = module
    spec.loader.exec_module(module)
    return module


@dataclass
class ModelClasses:
    Config: type
    Transformer: type
    BPETokenizer: type


def load_model_classes(folder: Path, namespace: str) -> ModelClasses:
    cfg = _load_file_as_module(folder / "config.py", f"{namespace}.config")
    tok = _load_file_as_module(folder / "tokenizer.py", f"{namespace}.tokenizer")
    mdl = _load_file_as_module(folder / "model.py", f"{namespace}.model")
    return ModelClasses(Config=cfg.Config, Transformer=mdl.Transformer, BPETokenizer=tok.BPETokenizer)


# ============================================================================
# Device + checkpoint discovery
# ============================================================================

def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def has_files(name: str) -> bool:
    spec = MODEL_SPECS[name]
    if spec.family == "ultra":
        if not ultra_backend.available():
            return False
        ckpt = ultra_backend.resolve_checkpoint(spec.ckpt_candidates)
        tok = next((p for p in spec.tokenizer_candidates if Path(p).is_file()), None)
        return ckpt is not None and tok is not None
    return (
        (spec.folder / "checkpoints" / spec.checkpoint_filename).exists()
        and (spec.folder / "data" / "tokenizer.json").exists()
    )


def availability() -> dict[str, bool]:
    return {name: has_files(name) for name in MODEL_SPECS}


# ============================================================================
# Cheap architecture summary (no weights loaded — just reads Config + maths)
# ============================================================================

@st.cache_data(show_spinner=False)
def arch_info(name: str) -> dict:
    spec = MODEL_SPECS[name]
    if spec.family == "ultra":
        p = spec.approx_params
        info = {"n_params": p, "params_str": (f"{p/1e9:.1f}B" if p >= 1e9 else f"{p/1e6:.0f}M"),
                "context_window": 0, "hidden_dim": 0, "num_blocks": 0, "num_heads": 0}
        try:                                    # dims from the config YAML (always present in the repo)
            if ultra_backend.available() and spec.config_yaml and Path(spec.config_yaml).exists():
                cfg = ultra_backend._mod().ModelConfig.from_yaml(str(spec.config_yaml))
                info.update(context_window=cfg.seq_len, hidden_dim=cfg.d_model,
                            num_blocks=cfg.n_layers, num_heads=cfg.n_heads)
        except Exception:
            pass
        return info
    classes = load_model_classes(spec.folder, spec.namespace)
    cfg = classes.Config()
    V, D, L = cfg.vocab_size, cfg.hidden_dim, cfg.num_blocks
    # Standard transformer parameter count: 2VD (token+lm_head) + CD (pos embed)
    # + L * (12D^2 + 13D) per block + 2D (final layernorm)
    n_params = 2 * V * D + cfg.context_window * D + L * (12 * D * D + 13 * D) + 2 * D
    return {
        "n_params": n_params,
        "params_str": f"{n_params / 1e6:.0f}M",
        "context_window": cfg.context_window,
        "hidden_dim": D,
        "num_blocks": L,
        "num_heads": cfg.num_heads,
    }


# ============================================================================
# Model loading (cached by Streamlit so reruns don't reload from disk)
# ============================================================================

@dataclass
class LoadedModel:
    name: str
    model: torch.nn.Module
    tokenizer: object
    context_window: int
    device: str
    checkpoint_step: int
    family: str = "legacy"            # "ultra" routes generation/prompting through ultra_backend
    stop_id: int | None = None        # ultra: token id to stop on (<|im_end|>)


@st.cache_resource(show_spinner=False)
def load_model(name: str, device: str) -> LoadedModel:
    spec = MODEL_SPECS[name]
    if spec.family == "ultra":
        ckpt = ultra_backend.resolve_checkpoint(spec.ckpt_candidates)
        tok = next((p for p in spec.tokenizer_candidates if Path(p).is_file()), None)
        if ckpt is None or tok is None:
            raise FileNotFoundError(f"{name}: checkpoint or tokenizer not found yet")
        u = ultra_backend.load(ckpt, tok, spec.config_yaml, device)
        return LoadedModel(name=name, model=u["model"], tokenizer=u["tokenizer"],
                           context_window=u["context_window"], device=device,
                           checkpoint_step=u["step"], family="ultra", stop_id=u["stop_id"])
    classes = load_model_classes(spec.folder, spec.namespace)
    config = classes.Config()

    # Only pass use_flash if this model's Transformer signature accepts it
    # (older MaxGPT-1 builds predate the Flash-Attention toggle).
    sig = inspect.signature(classes.Transformer.__init__)
    kwargs = dict(
        vocab_size=config.vocab_size,
        context_window=config.context_window,
        hidden_dim=config.hidden_dim,
        num_heads=config.num_heads,
        num_blocks=config.num_blocks,
    )
    if "use_flash" in sig.parameters:
        # Flash/Efficient SDPA backends are CUDA-only. On MPS/CPU (e.g. running
        # the demo on the Mac) force the manual-attention path, otherwise the
        # forced sdpa_kernel([FLASH, EFFICIENT]) context in model.py raises
        # "no viable backend." Manual attention is fine for inference.
        kwargs["use_flash"] = getattr(config, "use_flash_attention", True) and device == "cuda"
    model = classes.Transformer(**kwargs)

    ckpt_path = spec.folder / "checkpoints" / spec.checkpoint_filename

    # weights_only=False because the checkpoint contains the pickled Config
    # dataclass. Its pickle references "config.Config" (the original bare
    # module path), so we temporarily alias our namespaced modules under
    # the bare names pickle expects, then restore in finally.
    alias_targets = ("config", "model", "tokenizer")
    alias_backup = {n: sys.modules.get(n) for n in alias_targets}
    for bare in alias_targets:
        full = f"{spec.namespace}.{bare}"
        if full in sys.modules:
            sys.modules[bare] = sys.modules[full]
    try:
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    finally:
        for bare, prev in alias_backup.items():
            if prev is None:
                sys.modules.pop(bare, None)
            else:
                sys.modules[bare] = prev

    # If the model was saved while wrapped by torch.compile, every key in the
    # state_dict is prefixed with "_orig_mod." (the wrapper's attribute name).
    # train.py/finetune.py try to unwrap on save, but some older checkpoints
    # (notably MaxGPT-3's final.pt) were saved before that fix landed. Strip
    # the prefix here so the loader is defensive — no-op when keys are clean.
    state_dict = checkpoint["model_state"]
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device).eval()

    tokenizer = classes.BPETokenizer()
    tokenizer.load(str(spec.folder / "data" / "tokenizer.json"))

    return LoadedModel(
        name=name,
        model=model,
        tokenizer=tokenizer,
        context_window=config.context_window,
        device=device,
        checkpoint_step=checkpoint.get("step", -1),
    )


# ============================================================================
# Streaming generation
#
# Mirrors generate() in each sample.py but yields decoded chunks as new tokens
# are sampled, instead of returning the full string at the end.
#
# Holdback trick: don't yield the last few characters of decoded text until we
# know they aren't forming "USER:". The stop string never flashes on screen
# before we cut the generation.
# ============================================================================

def streaming_generate(
    loaded: LoadedModel,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    repetition_penalty: float = 1.2,
    repetition_window: int = 64,
    stop_string: str = STOP_STRING,
) -> Generator[str, None, None]:
    if loaded.family == "ultra":            # MaxGPT-Mini / MaxGPT-Ultra: delegate to the ultra backend
        yield from ultra_backend.stream(
            loaded.model, loaded.tokenizer, prompt, loaded.device, loaded.context_window,
            max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k,
            repetition_penalty=repetition_penalty, repetition_window=repetition_window,
            stop_id=loaded.stop_id)
        return
    model = loaded.model
    tokenizer = loaded.tokenizer
    device = loaded.device
    context_window = loaded.context_window

    prompt_ids = tokenizer.encode(prompt)
    tokens = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    generated_ids: list[int] = []
    decoded_full = ""
    last_yielded_len = 0
    holdback = len(stop_string)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            input_tokens = tokens[:, -context_window:] if tokens.size(1) > context_window else tokens
            logits, _ = model(input_tokens)
            logits = logits[:, -1, :]

            if repetition_penalty != 1.0:
                recent = tokens[0, -repetition_window:] if tokens.size(1) > repetition_window else tokens[0]
                unique_recent = torch.unique(recent)
                recent_logits = logits[0, unique_recent]
                logits[0, unique_recent] = torch.where(
                    recent_logits > 0,
                    recent_logits / repetition_penalty,
                    recent_logits * repetition_penalty,
                )

            logits = logits / temperature

            if top_k is not None:
                top_values, _ = torch.topk(logits, top_k, dim=-1)
                threshold = top_values[:, -1].unsqueeze(-1)
                logits = torch.where(logits < threshold, float("-inf"), logits)

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_token], dim=1)
            generated_ids.append(next_token.item())

            decoded_full = tokenizer.decode(generated_ids)

            stop_idx = decoded_full.find(stop_string)
            if stop_idx >= 0:
                if stop_idx > last_yielded_len:
                    yield decoded_full[last_yielded_len:stop_idx]
                return

            safe_len = max(last_yielded_len, len(decoded_full) - holdback)
            if safe_len > last_yielded_len:
                yield decoded_full[last_yielded_len:safe_len]
                last_yielded_len = safe_len

    if last_yielded_len < len(decoded_full):
        yield decoded_full[last_yielded_len:]


# ============================================================================
# Multi-turn prompt building with sliding-window history truncation
# ============================================================================

def build_chat_prompt(
    history: list[dict],
    user_input: str,
    tokenizer,
    context_window: int,
    max_new_tokens: int,
) -> tuple[str, int]:
    """Returns (prompt, num_turns_dropped)."""
    if hasattr(tokenizer, "render_chat"):     # ultra family (UltraTokenizer) -> ChatML prompt
        return ultra_backend.render_chatml(history, user_input, tokenizer, context_window, max_new_tokens)
    def render(turns: list[dict]) -> str:
        parts = [f"USER: {t['user']}\nASSISTANT: {t['assistant']}\n" for t in turns]
        parts.append(f"USER: {user_input}\nASSISTANT:")
        return "".join(parts)

    budget = max(1, context_window - max_new_tokens)
    turns = list(history)
    dropped = 0

    while True:
        prompt = render(turns)
        if len(tokenizer.encode(prompt)) <= budget or not turns:
            return prompt, dropped
        turns.pop(0)
        dropped += 1


# ============================================================================
# TimedGen — wraps a generator and captures elapsed-time + chunk-count as side
# effects. Used so chat mode can render the same status pill as compare mode
# AFTER st.write_stream finishes consuming the generator.
# ============================================================================

class TimedGen:
    def __init__(self, gen):
        self._gen = gen
        self.chunks = 0
        self.elapsed = 0.0

    def __iter__(self):
        t0 = time.time()
        for chunk in self._gen:
            self.chunks += 1
            yield chunk
        self.elapsed = time.time() - t0


# ============================================================================
# Compare-mode worker — runs streaming_generate in a thread, pushes results
# to a queue. Only the main thread touches the Streamlit API.
# ============================================================================

def compare_worker(loaded: LoadedModel, prompt: str, gen_kwargs: dict, out_q: queue.Queue):
    start = time.time()
    try:
        full_text = ""
        chunks = 0
        for chunk in streaming_generate(loaded, prompt, **gen_kwargs):
            full_text += chunk
            chunks += 1
            out_q.put(("chunk", loaded.name, full_text, time.time() - start, chunks))
        out_q.put(("done", loaded.name, full_text, time.time() - start, chunks))
    except Exception as e:
        out_q.put(("error", loaded.name, repr(e), time.time() - start, 0))


# ============================================================================
# Styling — one CSS block, injected once. Keeps the rest of the UI plain
# Streamlit widgets so functionality is identical.
#
# All visual indicators are CSS shapes (rounded divs) — no emoji.
# ============================================================================

CSS = """
<style>
  .block-container { padding-top: 2.2rem; }

  /* ---------- Hero header ---------- */
  /* Dark slate -> deep indigo so the four model accent colors really pop
     against it, instead of competing with the pastel purple/pink gradient. */
  .mg-hero {
      background: linear-gradient(135deg, #0F172A 0%, #1E1B4B 55%, #312E81 100%);
      padding: 1.6rem 1.9rem 1.4rem;
      border-radius: 18px;
      margin-bottom: 1.4rem;
      color: white;
      box-shadow: 0 12px 40px rgba(15, 23, 42, 0.45);
  }
  .mg-hero h1 {
      margin: 0; color: white;
      font-size: 2.4rem; font-weight: 800;
      letter-spacing: -0.025em; line-height: 1.1;
  }
  .mg-hero .tagline {
      margin-top: 0.45rem;
      opacity: 0.92; font-size: 0.98rem;
  }
  .mg-hero .pills {
      margin-top: 0.95rem;
      display: flex; gap: 0.35rem; flex-wrap: wrap;
  }
  .mg-hero .pill {
      background: rgba(255,255,255,0.18);
      padding: 0.22rem 0.7rem;
      border-radius: 999px;
      font-size: 0.78rem; font-weight: 500;
      backdrop-filter: blur(6px);
      display: inline-flex; align-items: center; gap: 0.35rem;
  }
  .mg-hero .pill.muted { opacity: 0.5; }

  /* Colored dot used inside hero pills — small circle with a white halo
     so it pops against the gradient background. */
  .mg-hero-dot {
      width: 7px; height: 7px;
      border-radius: 50%;
      background: var(--c, #888);
      box-shadow: 0 0 0 2px rgba(255,255,255,0.45);
      flex-shrink: 0;
  }

  /* ---------- Chat-mode meta badges ---------- */
  .mg-meta {
      display: flex; gap: 0.5rem; flex-wrap: wrap;
      align-items: center; margin: 0.2rem 0 0.9rem;
  }
  .mg-model-pill {
      padding: 0.32rem 0.95rem;
      border-radius: 999px;
      font-size: 0.92rem; font-weight: 700;
      letter-spacing: 0.005em;
      display: inline-flex; align-items: center; gap: 0.45rem;
      border: 1px solid color-mix(in srgb, var(--c, #888) 40%, transparent);
  }
  .mg-pill-dot {
      width: 8px; height: 8px; border-radius: 50%;
      background: var(--c, #888);
      flex-shrink: 0;
  }
  .mg-pill {
      background: rgba(120,120,120,0.11);
      padding: 0.3rem 0.7rem;
      border-radius: 8px;
      font-size: 0.81rem;
      color: rgba(110,110,110,0.95);
      font-family: 'SF Mono', 'Monaco', 'Consolas', monospace;
  }

  /* ---------- Pulsing streaming status ---------- */
  .mg-status {
      display: flex; align-items: center; gap: 0.5rem;
      font-size: 0.82rem;
      color: rgba(120,120,120,0.9);
      font-family: 'SF Mono', 'Monaco', 'Consolas', monospace;
      margin-bottom: 0.5rem;
  }
  .mg-dot {
      width: 9px; height: 9px; border-radius: 50%;
      background: var(--c, #10B981);
      animation: mg-pulse 1.25s ease-in-out infinite;
  }
  .mg-dot.done { animation: none; opacity: 0.5; }
  @keyframes mg-pulse {
      0%, 100% { opacity: 1; transform: scale(1); }
      50%      { opacity: 0.4; transform: scale(0.82); }
  }

  /* ---------- Compare-mode column header ---------- */
  .mg-col {
      border-top: 4px solid var(--c, #6B7280);
      background: linear-gradient(180deg,
          color-mix(in srgb, var(--c) 8%, transparent) 0%,
          transparent 70%);
      padding: 0.85rem 0.7rem 0.55rem;
      border-radius: 4px 4px 0 0;
      margin-bottom: 0.5rem;
  }
  .mg-col-name {
      font-size: 1.22rem; font-weight: 800;
      color: var(--c, #6B7280);
      letter-spacing: -0.01em;
      display: flex; align-items: center; gap: 0.5rem;
  }
  .mg-col-square {
      width: 11px; height: 11px;
      background: var(--c, #888);
      border-radius: 3px;
      flex-shrink: 0;
  }
  .mg-col-sub {
      font-size: 0.76rem;
      color: rgba(120,120,120,0.85);
      margin-top: 0.18rem;
      font-family: 'SF Mono', 'Monaco', 'Consolas', monospace;
  }

  /* ---------- Sidebar polish ---------- */
  [data-testid="stSidebar"] h2 { letter-spacing: -0.01em; }
  [data-testid="stSidebar"] .stSlider { padding: 0.15rem 0; }

  /* Accent strip on the selected sidebar model — added to the radio group's
     focused option. Subtle, doesn't change layout. */
  [data-testid="stSidebar"] [role="radiogroup"] label[data-baseweb="radio"] {
      padding: 0.15rem 0;
  }

  /* ---------- Family groupings (Classic / Neo) in the hero ---------- */
  .mg-hero { border: 1px solid rgba(255,255,255,0.08); }
  .mg-hero .tagline { max-width: 60rem; line-height: 1.5; }
  .mg-fam { margin-top: 0.95rem; }
  .mg-fam-label {
      display: block;
      font-size: 0.7rem; font-weight: 700;
      text-transform: uppercase; letter-spacing: 0.14em;
      color: rgba(255,255,255,0.6);
      margin-bottom: 0.45rem;
  }
  .mg-fam-sub { font-weight: 500; text-transform: none; letter-spacing: 0; opacity: 0.65; }
  .mg-fam .pills { margin-top: 0; }
</style>
"""


# ---------- HTML render helpers ----------

def hero(active_set: dict[str, bool]) -> str:
    def pill(n: str) -> str:
        spec = MODEL_SPECS[n]
        muted = "" if active_set[n] else " muted"
        return (f'<span class="pill{muted}">'
                f'<span class="mg-hero-dot" style="--c: {spec.color};"></span>{n}</span>')

    classic = "".join(pill(n) for n, s in MODEL_SPECS.items() if s.family == "legacy")
    neo = "".join(pill(n) for n, s in MODEL_SPECS.items() if s.family == "ultra")
    return f"""
    <div class="mg-hero">
      <h1>MaxGPT</h1>
      <div class="tagline">A family of language models built from scratch in PyTorch &mdash; my own
      tokenizer, transformer, and training loop. No pretrained weights, no transformers library.
      Chat with one, or race them side by side.</div>
      <div class="mg-fam">
        <span class="mg-fam-label">Classic <span class="mg-fam-sub">&middot; the GPT-2-style lineage</span></span>
        <div class="pills">{classic}</div>
      </div>
      <div class="mg-fam">
        <span class="mg-fam-label">Neo <span class="mg-fam-sub">&middot; modern architecture &middot; retrieval-augmented</span></span>
        <div class="pills">{neo}</div>
      </div>
    </div>
    """


def model_badge_html(name: str, loaded: LoadedModel, info: dict) -> str:
    spec = MODEL_SPECS[name]
    color = spec.color
    step = f"{loaded.checkpoint_step:,}" if loaded.checkpoint_step >= 0 else "?"
    return f"""
    <div class="mg-meta">
      <span class="mg-model-pill"
            style="--c: {color}; background: color-mix(in srgb, {color} 14%, transparent); color: {color};">
        <span class="mg-pill-dot" style="--c: {color};"></span>{name}
      </span>
      <span class="mg-pill">{info['params_str']} params</span>
      <span class="mg-pill">{info['num_blocks']}L × {info['hidden_dim']}d</span>
      <span class="mg-pill">ctx {loaded.context_window}</span>
      <span class="mg-pill">step {step}</span>
      <span class="mg-pill">{loaded.device}</span>
      <span class="mg-pill">{spec.blurb}</span>
    </div>
    """


def col_header_html(name: str, info: dict) -> str:
    spec = MODEL_SPECS[name]
    return f"""
    <div class="mg-col" style="--c: {spec.color};">
      <div class="mg-col-name">
        <span class="mg-col-square" style="--c: {spec.color};"></span>{name}
      </div>
      <div class="mg-col-sub">{info['params_str']} params · ctx {info['context_window']} · {spec.blurb}</div>
    </div>
    """


def status_html(name: str, elapsed: float, n_chunks: int, done: bool) -> str:
    spec = MODEL_SPECS[name]
    tps = n_chunks / elapsed if elapsed > 0 else 0.0
    state = "done" if done else "streaming"
    dot_class = "mg-dot done" if done else "mg-dot"
    return f"""
    <div class="mg-status">
      <span class="{dot_class}" style="--c: {spec.color};"></span>
      <span>{state} · {elapsed:.1f}s · {n_chunks} chunks · {tps:.1f} ch/s</span>
    </div>
    """


def about_panel(name: str) -> None:
    """Expander explaining what makes this model different — shown below the
    badge in chat mode and below the prompt in compare mode."""
    spec = MODEL_SPECS[name]
    info = arch_info(name)
    with st.expander(f"About {name}", expanded=False):
        st.markdown(f"**{spec.summary}**")
        for b in spec.bullets:
            st.markdown(f"- {b}")
        heads = info['num_heads'] or 1
        fam = "Neo" if spec.family == "ultra" else "Classic"
        vocab = "49,152 (byte-level BPE)" if spec.family == "ultra" else "16,000 (BPE)"
        st.caption(
            f"{fam} family · {info['num_heads']} attention heads · head dim "
            f"{info['hidden_dim'] // heads} · vocab {vocab}"
        )


# ============================================================================
# Page
# ============================================================================

# Page icon: small SVG favicon (dark indigo "M") — sits next to app.py.
_ICON = Path(__file__).parent / "favicon.svg"
st.set_page_config(
    page_title="MaxGPT",
    page_icon=str(_ICON) if _ICON.exists() else "M",
    layout="wide",
)
st.markdown(CSS, unsafe_allow_html=True)

device = pick_device()
avail = availability()

st.markdown(hero(avail), unsafe_allow_html=True)


# ---------- Sidebar ----------
with st.sidebar:
    st.header("Settings")
    mode = st.radio("Mode", ["Chat", "Compare"], horizontal=True)
    st.divider()

    selected_model: str | None = None
    selected_models: list[str] = []

    def picker_label(name: str) -> str:
        info = arch_info(name)
        spec = MODEL_SPECS[name]
        fam = "Neo" if spec.family == "ultra" else "Classic"
        suffix = "" if avail[name] else "   (no checkpoint)"
        return f"{name}  ·  {fam}  ·  {info['params_str']}{suffix}"

    if mode == "Chat":
        options = list(MODEL_SPECS.keys())
        # Default to MaxGPT-3.5 (flagship chatbot) if available, else the
        # largest available model.
        _pref = next(
            (n for n in ["MaxGPT-3.5", "MaxGPT-3", "MaxGPT-2", "MaxGPT-1"] if avail.get(n)),
            options[0],
        )
        picked = st.radio("Model", options, index=options.index(_pref), format_func=picker_label)
        if avail[picked]:
            selected_model = picked
        else:
            st.warning(
                f"`{picked}` is missing "
                f"`checkpoints/{MODEL_SPECS[picked].checkpoint_filename}` "
                "or `data/tokenizer.json`. Drop the files in and reload."
            )
        if picked:
            st.caption(MODEL_SPECS[picked].summary)
    else:
        loadable = [n for n in MODEL_SPECS if avail[n]]
        not_loadable = [n for n in MODEL_SPECS if not avail[n]]
        # Default to the flagship matchup: MaxGPT-3.5 vs MaxGPT-3 (SFT vs base).
        # Fall back to the first two available if that pair isn't present.
        _pair = [n for n in ["MaxGPT-3.5", "MaxGPT-3"] if n in loadable]
        _compare_default = _pair if len(_pair) == 2 else loadable[: min(2, len(loadable))]
        selected_models = st.multiselect(
            "Models to compare",
            loadable,
            default=_compare_default,
            format_func=picker_label,
        )
        if not_loadable:
            st.caption("Unavailable: " + ", ".join(not_loadable))

    st.divider()
    st.subheader("Sampling")
    temperature        = st.slider("Temperature",        0.1, 1.5, 0.8, 0.05, key="temperature")
    top_k              = st.slider("Top-K",              1,   100, 50,  1,    key="top_k")
    repetition_penalty = st.slider("Repetition penalty", 1.0, 2.0, 1.2, 0.05, key="repetition_penalty")
    max_new_tokens     = st.slider("Max new tokens",     50,  300, 200, 10,   key="max_new_tokens")

    # Snap-back for live demos. Same values as sample.py — these are the
    # settings the models were tested against, so they're the "known good"
    # baseline. Defined as an on_click callback so session_state is mutated
    # BEFORE the next render — Streamlit forbids modifying a widget's value
    # mid-render, but pre-render mutations are honored.
    def _reset_demo_defaults():
        st.session_state["temperature"]        = 0.8
        st.session_state["top_k"]              = 50
        st.session_state["repetition_penalty"] = 1.2
        st.session_state["max_new_tokens"]     = 200

    st.button(
        "Reset to demo defaults",
        on_click=_reset_demo_defaults,
        use_container_width=True,
        help="Snaps sampling to temp 0.8 · top-k 50 · rep 1.2 · 200 tokens (the sample.py defaults).",
    )

    st.divider()
    st.caption(f"Device: `{device}`")

    if mode == "Chat" and selected_model is not None:
        st.session_state["collect_feedback"] = st.toggle(
            "Collect feedback (training data)", value=True,
            help="Logs your 1-5 ratings to webui/feedback.jsonl so you can fine-tune on them later. Does not train the model live.",
        )
        if st.button("Clear chat", use_container_width=True):
            st.session_state.get("chat_history", {}).pop(selected_model, None)
            st.rerun()


gen_kwargs = dict(
    max_new_tokens=max_new_tokens,
    temperature=temperature,
    top_k=top_k,
    repetition_penalty=repetition_penalty,
)


# ============================================================================
# Feedback collection. Logs ratings to webui/feedback.jsonl for LATER offline
# fine-tuning (this does NOT train the model live; it only gathers data, the
# safe approach we settled on: collect now, batch-train on the good ones later).
# ============================================================================
FEEDBACK_PATH = Path(__file__).parent / "feedback.jsonl"


def log_feedback(model: str, prompt: str, response: str, rating: int) -> None:
    rec = {
        "ts": datetime.datetime.now().isoformat(timespec="seconds"),
        "model": model,
        "rating": rating,            # 1 (bad) to 5 (great), magnitude for future LR scaling
        "prompt": prompt,
        "response": response,
    }
    with open(FEEDBACK_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def feedback_ui(model: str, idx: int, turn: dict) -> None:
    logged = st.session_state.setdefault("fb_logged", set())
    fid = f"{model}::{idx}"
    if fid in logged:
        st.caption("rating saved, thanks!")
        return
    st.caption("Rate this reply (1 = bad, 5 = great):")
    c1, c2 = st.columns([4, 1])
    rating = c1.slider("rating", 1, 5, 3, key=f"rate_{fid}", label_visibility="collapsed")
    if c2.button("Save", key=f"save_{fid}", use_container_width=True):
        log_feedback(model, f"USER: {turn['user']}\nASSISTANT:", turn["assistant"], rating)
        logged.add(fid)
        st.rerun()


# ---------- Chat mode ----------
if mode == "Chat":
    if selected_model is None:
        st.info("Pick a model with a checkpoint to start chatting.")
        st.stop()

    with st.spinner(f"Loading {selected_model}..."):
        loaded = load_model(selected_model, device)

    info = arch_info(selected_model)
    st.markdown(model_badge_html(selected_model, loaded, info), unsafe_allow_html=True)
    about_panel(selected_model)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = {}
    history = st.session_state.chat_history.setdefault(selected_model, [])

    collect_feedback = st.session_state.get("collect_feedback", True)
    for i, turn in enumerate(history):
        with st.chat_message("user"):
            st.write(turn["user"])
        with st.chat_message("assistant"):
            st.write(turn["assistant"])
            if collect_feedback:
                feedback_ui(selected_model, i, turn)

    user_input = st.chat_input(f"Talk to {selected_model}")
    if user_input:
        with st.chat_message("user"):
            st.write(user_input)

        prompt, dropped = build_chat_prompt(
            history, user_input, loaded.tokenizer,
            loaded.context_window, max_new_tokens,
        )

        with st.chat_message("assistant"):
            if dropped > 0:
                noun = "turn" if dropped == 1 else "turns"
                st.caption(f"{dropped} earlier {noun} dropped to fit context window")
            tg = TimedGen(streaming_generate(loaded, prompt, **gen_kwargs))
            response = st.write_stream(tg)
            st.markdown(
                status_html(selected_model, tg.elapsed, tg.chunks, done=True),
                unsafe_allow_html=True,
            )
            new_turn = {"user": user_input, "assistant": response}
            history.append(new_turn)
            if st.session_state.get("collect_feedback", True):
                feedback_ui(selected_model, len(history) - 1, new_turn)


# ---------- Compare mode (one thread per model) ----------
else:
    if not selected_models:
        st.info("Pick at least one model with a checkpoint.")
        st.stop()

    loaded_map: dict[str, LoadedModel] = {}
    with st.spinner(f"Loading {len(selected_models)} model(s)..."):
        for name in selected_models:
            loaded_map[name] = load_model(name, device)

    # Per-model "About" expanders inline, so viewers can see what's different
    # between the columns they're about to compare.
    with st.expander(
        f"About these {len(selected_models)} models",
        expanded=False,
    ):
        for name in selected_models:
            spec = MODEL_SPECS[name]
            st.markdown(
                f"**{name}** — <span style='color:{spec.color};font-weight:600;'>"
                f"{spec.blurb}</span>",
                unsafe_allow_html=True,
            )
            st.caption(spec.summary)
            for b in spec.bullets:
                st.markdown(f"- {b}")
            st.markdown("")

    user_input = st.chat_input("Prompt all selected models")
    if user_input:
        st.markdown(
            f"<div class='mg-pill' style='display:inline-block;margin-bottom:0.6rem;'>"
            f"prompt · {user_input}</div>",
            unsafe_allow_html=True,
        )

        cols = st.columns(len(selected_models))
        text_placeholders: dict[str, "st.delta_generator.DeltaGenerator"] = {}
        stat_placeholders: dict[str, "st.delta_generator.DeltaGenerator"] = {}
        for col, name in zip(cols, selected_models):
            with col:
                st.markdown(col_header_html(name, arch_info(name)), unsafe_allow_html=True)
                stat_placeholders[name] = st.empty()
                text_placeholders[name] = st.empty()

        q: queue.Queue = queue.Queue()
        for name in selected_models:
            lm = loaded_map[name]
            if lm.family == "ultra":          # build each model's prompt in its own format
                prompt, _ = ultra_backend.render_chatml([], user_input, lm.tokenizer,
                                                        lm.context_window, gen_kwargs["max_new_tokens"])
            else:
                prompt = f"USER: {user_input}\nASSISTANT:"
            threading.Thread(
                target=compare_worker,
                args=(lm, prompt, gen_kwargs, q),
                daemon=True,
            ).start()

        texts       = {n: "" for n in selected_models}
        elapsed     = {n: 0.0 for n in selected_models}
        chunks_seen = {n: 0 for n in selected_models}
        done        = {n: False for n in selected_models}

        while not all(done.values()):
            try:
                while True:
                    msg_type, name, text_or_err, t_elapsed, n_chunks = q.get_nowait()
                    if msg_type == "error":
                        texts[name] = f"[ERROR] {text_or_err}"
                        done[name] = True
                    else:
                        texts[name] = text_or_err
                        elapsed[name] = t_elapsed
                        chunks_seen[name] = n_chunks
                        if msg_type == "done":
                            done[name] = True
            except queue.Empty:
                pass

            for name in selected_models:
                cursor = "" if done[name] else "&#9612;"
                text_placeholders[name].markdown(texts[name] + cursor)
                stat_placeholders[name].markdown(
                    status_html(name, elapsed[name], chunks_seen[name], done[name]),
                    unsafe_allow_html=True,
                )

            time.sleep(0.03)
