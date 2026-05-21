"""
MaxGPT Web UI — chat with and compare three from-scratch language models.

Run from this directory:
    streamlit run app.py
"""

from __future__ import annotations

import importlib.util
import inspect
import queue
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

import streamlit as st
import torch


# ============================================================================
# Paths and constants
# ============================================================================

ROOT = Path(__file__).resolve().parent.parent

MODEL_FOLDERS = {
    "MaxGPT-1": ROOT / "maxgpt-1",
    "MaxGPT-2": ROOT / "maxgpt-2",
    "MaxGPT-3": ROOT / "maxgpt-3",
}

STOP_STRING = "USER:"


# ============================================================================
# Import collision handling
# ----------------------------------------------------------------------------
# Each maxgpt-N/ folder has its own config.py, model.py, tokenizer.py defining
# classes with the SAME names (Config, Transformer, BPETokenizer). A normal
# `from config import Config` after a `sys.path.insert(0, "maxgpt-1")` would
# work for one model — but the second `sys.path.insert(0, "maxgpt-2")` followed
# by another `from config import Config` would either pick up the cached
# maxgpt-1 module or shadow it, depending on the order.
#
# Fix: use importlib.util to register each model's files in sys.modules under a
# uniquely-namespaced name ("maxgpt_1.model" vs "maxgpt_2.model" vs ...). This
# gives us three distinct Transformer classes that can coexist in one process.
#
# This works because the three .py files in each folder are self-contained —
# they don't import from each other, only from stdlib and torch.
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
    return ModelClasses(
        Config=cfg.Config,
        Transformer=mdl.Transformer,
        BPETokenizer=tok.BPETokenizer,
    )


# ============================================================================
# Device + checkpoint discovery
# ============================================================================

def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def has_files(folder: Path) -> bool:
    return (folder / "checkpoints" / "final.pt").exists() and (folder / "data" / "tokenizer.json").exists()


def availability() -> dict[str, bool]:
    return {name: has_files(folder) for name, folder in MODEL_FOLDERS.items()}


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


@st.cache_resource(show_spinner=False)
def load_model(name: str, device: str) -> LoadedModel:
    folder = MODEL_FOLDERS[name]
    namespace = name.lower().replace("-", "_")
    classes = load_model_classes(folder, namespace)

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
        kwargs["use_flash"] = getattr(config, "use_flash_attention", True)
    model = classes.Transformer(**kwargs)

    ckpt_path = folder / "checkpoints" / "final.pt"
    # weights_only=False because the checkpoint contains the Config dataclass,
    # not just tensor state. Safe — we created these files.
    #
    # The pickled Config inside the checkpoint references its original module
    # path ("config.Config"), so we temporarily alias our namespaced modules
    # ("maxgpt_1.config", etc.) under the bare names pickle expects. Restored
    # in finally so loading another model later doesn't see stale aliases.
    alias_targets = ("config", "model", "tokenizer")
    alias_backup = {n: sys.modules.get(n) for n in alias_targets}
    for bare in alias_targets:
        full = f"{namespace}.{bare}"
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
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()

    tokenizer = classes.BPETokenizer()
    tokenizer.load(str(folder / "data" / "tokenizer.json"))

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
# Holdback trick: we don't yield the last few characters of decoded text until
# we know they aren't forming "USER:". That way the stop string never flashes
# on screen before we cut the generation.
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
) -> tuple[str, bool]:
    def render(turns: list[dict]) -> str:
        parts = [f"USER: {t['user']}\nASSISTANT: {t['assistant']}\n" for t in turns]
        parts.append(f"USER: {user_input}\nASSISTANT:")
        return "".join(parts)

    budget = max(1, context_window - max_new_tokens)
    turns = list(history)
    truncated = False

    while True:
        prompt = render(turns)
        if len(tokenizer.encode(prompt)) <= budget or not turns:
            return prompt, truncated
        turns.pop(0)
        truncated = True


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
# UI
# ============================================================================

st.set_page_config(page_title="MaxGPT", page_icon=":robot_face:", layout="wide")
st.title("MaxGPT")

device = pick_device()
avail = availability()

with st.sidebar:
    st.header("Settings")
    mode = st.radio("Mode", ["Chat", "Compare"], horizontal=True)
    st.divider()

    selected_model = None
    selected_models: list[str] = []

    if mode == "Chat":
        options = list(MODEL_FOLDERS.keys())
        picked = st.radio(
            "Model",
            options,
            format_func=lambda n: n if avail[n] else f"{n}  (no checkpoint)",
        )
        if avail[picked]:
            selected_model = picked
        else:
            st.warning(
                f"`{picked}` is missing `checkpoints/final.pt` or `data/tokenizer.json`. "
                "Add the files and reload the page."
            )
    else:
        loadable = [n for n, ok in avail.items() if ok]
        not_loadable = [n for n, ok in avail.items() if not ok]
        selected_models = st.multiselect(
            "Models to compare",
            loadable,
            default=loadable[: min(2, len(loadable))],
        )
        if not_loadable:
            st.caption(f"Unavailable (no checkpoint): {', '.join(not_loadable)}")

    st.divider()
    st.subheader("Sampling")
    temperature = st.slider("Temperature", 0.1, 1.5, 0.8, 0.05)
    top_k = st.slider("Top-K", 1, 100, 50, 1)
    repetition_penalty = st.slider("Repetition penalty", 1.0, 2.0, 1.2, 0.05)
    max_new_tokens = st.slider("Max new tokens", 50, 300, 200, 10)

    st.divider()
    st.caption(f"Device: `{device}`")

    if mode == "Chat" and selected_model is not None:
        if st.button("Clear chat", use_container_width=True):
            st.session_state.get("chat_history", {}).pop(selected_model, None)
            st.rerun()


gen_kwargs = dict(
    max_new_tokens=max_new_tokens,
    temperature=temperature,
    top_k=top_k,
    repetition_penalty=repetition_penalty,
)


# --------------------------------------------------------------------------- #
# Chat mode                                                                   #
# --------------------------------------------------------------------------- #
if mode == "Chat":
    if selected_model is None:
        st.info("Select a model with a checkpoint to start chatting.")
        st.stop()

    with st.spinner(f"Loading {selected_model}..."):
        loaded = load_model(selected_model, device)

    st.caption(
        f"`{selected_model}`  ·  ctx={loaded.context_window}  ·  "
        f"step={loaded.checkpoint_step}  ·  device={loaded.device}"
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = {}
    history = st.session_state.chat_history.setdefault(selected_model, [])

    for turn in history:
        with st.chat_message("user"):
            st.write(turn["user"])
        with st.chat_message("assistant"):
            st.write(turn["assistant"])

    user_input = st.chat_input(f"Talk to {selected_model}")
    if user_input:
        with st.chat_message("user"):
            st.write(user_input)

        prompt, truncated = build_chat_prompt(
            history, user_input, loaded.tokenizer,
            loaded.context_window, max_new_tokens,
        )

        with st.chat_message("assistant"):
            if truncated:
                st.caption("...earlier turns dropped to fit context window")
            response = st.write_stream(streaming_generate(loaded, prompt, **gen_kwargs))

        history.append({"user": user_input, "assistant": response})


# --------------------------------------------------------------------------- #
# Compare mode — each selected model streams in its own thread                #
# --------------------------------------------------------------------------- #
else:
    if not selected_models:
        st.info("Pick at least one model with a checkpoint.")
        st.stop()

    loaded_map: dict[str, LoadedModel] = {}
    with st.spinner(f"Loading {len(selected_models)} model(s)..."):
        for name in selected_models:
            loaded_map[name] = load_model(name, device)

    user_input = st.chat_input("Prompt all selected models")
    if user_input:
        st.markdown(f"**Prompt:** {user_input}")

        cols = st.columns(len(selected_models))
        text_placeholders = {}
        stat_placeholders = {}
        for col, name in zip(cols, selected_models):
            with col:
                st.subheader(name)
                stat_placeholders[name] = st.empty()
                text_placeholders[name] = st.empty()

        q: queue.Queue = queue.Queue()
        for name in selected_models:
            prompt = f"USER: {user_input}\nASSISTANT:"
            t = threading.Thread(
                target=compare_worker,
                args=(loaded_map[name], prompt, gen_kwargs, q),
                daemon=True,
            )
            t.start()

        texts = {n: "" for n in selected_models}
        elapsed = {n: 0.0 for n in selected_models}
        chunks_seen = {n: 0 for n in selected_models}
        done = {n: False for n in selected_models}

        while not all(done.values()):
            try:
                while True:
                    msg_type, name, text_or_err, t_elapsed, n_chunks = q.get_nowait()
                    if msg_type == "error":
                        texts[name] = f":warning: {text_or_err}"
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
                tps = chunks_seen[name] / elapsed[name] if elapsed[name] > 0 else 0.0
                state = "done" if done[name] else "streaming"
                stat_placeholders[name].caption(
                    f"{state}  ·  {elapsed[name]:.1f}s  ·  ~{tps:.1f} chunks/s"
                )

            time.sleep(0.03)
