"""
MaxGPT-3 SFT data preparation — chat-only subset for the fine-tuning phase.

Produces:
    data/chat_train.bin
    data/chat_val.bin

Uses the EXISTING tokenizer (data/tokenizer.json) — no retraining needed.
Only loads the 5 conversational datasets from the original 7. Output is
~2.4B train tokens + ~265M val tokens.

For the pre-train → SFT pipeline:
  1. prepare_data.py     → train.bin + val.bin     (all 7 datasets, used by train.py)
  2. prepare_chat_data.py → chat_train.bin + chat_val.bin (5 chat datasets, used by finetune.py)
  3. train.py             → checkpoints/final.pt  (base pre-trained model)
  4. finetune.py          → checkpoints/final_sft.pt (chat-specialized model)
"""

from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import os
import multiprocessing as mp
from collections import defaultdict
from tokenizer import BPETokenizer

# ====================
# CONFIG
# ====================
DOCUMENT_SEPARATOR = "\n\n###\n\n"
TRAIN_VAL_SPLIT = 0.9
OUTPUT_DIR = "data"

# Per-dataset character targets. Same as prepare_data.py for these 5 datasets,
# so the proportions stay consistent.
TARGET_CHARS = {
    "oasst":          400_000_000,    # OASST1 + OASST2, ~3.6M tokens
    "ultrachat":    7_000_000_000,    # full UltraChat, ~1.35B tokens
    "openorca":     3_500_000_000,    # subset, ~823M tokens
    "wildchat":     2_000_000_000,    # full English WildChat-1M, ~527M tokens
    "hhrlhf":         500_000_000,    # full Anthropic HH-RLHF, ~32M tokens
}

# Multiprocessing config (same as prepare_data.py, learned-the-hard-way values)
N_WORKERS = 8
ENCODE_CHUNK_SIZE = 500_000


# ====================
# DATASET LOADERS (subset of prepare_data.py — chat-only)
# ====================

def format_oasst_conversation(tree_root, children_by_parent):
    current = tree_root
    text_parts = []
    while current is not None:
        role_label = "USER" if current["role"] == "prompter" else "ASSISTANT"
        text_parts.append(f"{role_label}: {current['text']}")
        children = children_by_parent.get(current["message_id"], [])
        valid = [c for c in children if c["rank"] is not None]
        if valid:
            current = min(valid, key=lambda c: c["rank"])
        else:
            current = children[0] if children else None
    return "\n".join(text_parts)


def load_oasst():
    formatted_conversations = []
    for hf_name in ["OpenAssistant/oasst1", "OpenAssistant/oasst2"]:
        try:
            print(f"  Loading {hf_name}...")
            ds = load_dataset(hf_name, split="train")
            ds = ds.filter(lambda row: row.get("lang") == "en")
            children_by_parent = defaultdict(list)
            for m in ds:
                children_by_parent[m["parent_id"]].append(m)
            roots = children_by_parent[None]
            for root in roots:
                formatted_conversations.append(format_oasst_conversation(root, children_by_parent))
        except Exception as e:
            print(f"  Skipping {hf_name}: {e}")
    return DOCUMENT_SEPARATOR.join(formatted_conversations)


def load_ultrachat():
    ds = load_dataset("stingning/ultrachat", split="train", streaming=True)
    target = TARGET_CHARS["ultrachat"]
    parts = []
    chars = 0
    for example in tqdm(ds, desc="UltraChat"):
        turns = example.get("data", [])
        if not turns:
            continue
        formatted_turns = []
        for i, turn in enumerate(turns):
            role = "USER" if i % 2 == 0 else "ASSISTANT"
            formatted_turns.append(f"{role}: {turn}")
        text = "\n".join(formatted_turns)
        parts.append(text)
        chars += len(text)
        if chars >= target:
            break
    return DOCUMENT_SEPARATOR.join(parts)


def load_openorca():
    ds = load_dataset("Open-Orca/OpenOrca", split="train", streaming=True)
    target = TARGET_CHARS["openorca"]
    parts = []
    chars = 0
    for example in tqdm(ds, desc="OpenOrca"):
        question = example.get("question", "")
        response = example.get("response", "")
        if not question or not response:
            continue
        text = f"USER: {question}\nASSISTANT: {response}"
        parts.append(text)
        chars += len(text)
        if chars >= target:
            break
    return DOCUMENT_SEPARATOR.join(parts)


def load_wildchat():
    ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
    target = TARGET_CHARS["wildchat"]
    parts = []
    chars = 0
    for example in tqdm(ds, desc="WildChat"):
        if example.get("language") != "English":
            continue
        conv = example.get("conversation", [])
        if not conv:
            continue
        formatted_turns = []
        for msg in conv:
            role = "USER" if msg.get("role") == "user" else "ASSISTANT"
            content = msg.get("content", "")
            if content:
                formatted_turns.append(f"{role}: {content}")
        text = "\n".join(formatted_turns)
        parts.append(text)
        chars += len(text)
        if chars >= target:
            break
    return DOCUMENT_SEPARATOR.join(parts)


def load_hhrlhf():
    ds = load_dataset("Anthropic/hh-rlhf", split="train")
    target = TARGET_CHARS["hhrlhf"]
    parts = []
    chars = 0
    for example in tqdm(ds, desc="HH-RLHF"):
        raw = example.get("chosen", "")
        if not raw:
            continue
        text = raw.replace("\n\nHuman:", "\nUSER:").replace("\n\nAssistant:", "\nASSISTANT:")
        text = text.strip()
        parts.append(text)
        chars += len(text)
        if chars >= target:
            break
    return DOCUMENT_SEPARATOR.join(parts)


DATASET_LOADERS = {
    "oasst":     load_oasst,
    "ultrachat": load_ultrachat,
    "openorca":  load_openorca,
    "wildchat":  load_wildchat,
    "hhrlhf":    load_hhrlhf,
}


# ====================
# MULTIPROCESSING ENCODING (same as prepare_data.py)
# ====================
_worker_tokenizer = None


def _init_worker(tokenizer_path):
    global _worker_tokenizer
    _worker_tokenizer = BPETokenizer()
    _worker_tokenizer.load(tokenizer_path)


def _encode_chunk(chunk):
    return _worker_tokenizer.encode(chunk)


def chunked(s, n):
    for i in range(0, len(s), n):
        yield s[i:i + n]


def encode_parallel_streaming(text, tokenizer_path, label, output_path):
    """Stream tokens directly to disk as each chunk completes — no big in-memory list."""
    chunks = list(chunked(text, ENCODE_CHUNK_SIZE))
    print(f"  Encoding {label} ({len(text):,} chars in {len(chunks)} chunks across {N_WORKERS} workers)...")

    total_tokens = 0
    with open(output_path, "wb") as out_f:
        with mp.Pool(
            processes=N_WORKERS,
            initializer=_init_worker,
            initargs=(tokenizer_path,),
        ) as pool:
            for chunk_tokens in tqdm(
                pool.imap(_encode_chunk, chunks),
                total=len(chunks),
                desc=label,
            ):
                np.array(chunk_tokens, dtype=np.uint16).tofile(out_f)
                total_tokens += len(chunk_tokens)
    return total_tokens


# ====================
# HELPERS
# ====================

def split_90_10(tokens):
    idx = int(len(tokens) * TRAIN_VAL_SPLIT)
    return tokens[:idx], tokens[idx:]


# ====================
# MAIN
# ====================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tokenizer_path = os.path.join(OUTPUT_DIR, "tokenizer.json")

    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(
            f"No tokenizer at {tokenizer_path}. Run prepare_data.py first to "
            "train the tokenizer — this script reuses it."
        )

    # PHASE 1: Load + format all 5 chat datasets
    all_texts = {}
    print("\n" + "=" * 70 + "\nPHASE 1: Loading + formatting chat datasets\n" + "=" * 70)
    for name, loader in DATASET_LOADERS.items():
        print(f"\nLoading {name}...")
        try:
            all_texts[name] = loader()
            print(f"  {name}: {len(all_texts[name]):,} chars")
        except Exception as e:
            print(f"  ERROR loading {name}: {e}")
            all_texts[name] = ""

    total_chars = sum(len(t) for t in all_texts.values())
    print(f"\nTotal chat corpus: {total_chars:,} chars")

    # PHASE 2: Encode each dataset in parallel, write to intermediate bin files
    print("\n" + "=" * 70 + "\nPHASE 2: Parallel encoding\n" + "=" * 70)
    token_counts = {}
    for name in DATASET_LOADERS:
        text = all_texts.get(name, "")
        if not text:
            continue
        intermediate_path = os.path.join(OUTPUT_DIR, f"_chat_intermediate_{name}.bin")
        if os.path.exists(intermediate_path):
            existing = np.fromfile(intermediate_path, dtype=np.uint16)
            print(f"\nSkipping {name} — cached ({len(existing):,} tokens)")
            token_counts[name] = len(existing)
            all_texts[name] = ""
            continue
        print(f"\n--- {name} ---")
        n_tokens = encode_parallel_streaming(text, tokenizer_path, name, intermediate_path)
        token_counts[name] = n_tokens
        all_texts[name] = ""

    total_tokens = sum(token_counts.values())
    print(f"\n=== Tokens per chat dataset ===")
    for name, count in token_counts.items():
        if total_tokens > 0:
            print(f"  {name:10s}: {count:>14,} ({count/total_tokens:.1%})")
    print(f"  TOTAL     : {total_tokens:>14,}")

    # PHASE 3: Per-dataset 90/10 split + combine
    print("\n" + "=" * 70 + "\nPHASE 3: Per-dataset 90/10 split + combine\n" + "=" * 70)
    train_arrays = []
    val_arrays = []
    for name in DATASET_LOADERS:
        if name not in token_counts:
            continue
        intermediate_path = os.path.join(OUTPUT_DIR, f"_chat_intermediate_{name}.bin")
        tokens = np.fromfile(intermediate_path, dtype=np.uint16)
        train_t, val_t = split_90_10(tokens.tolist())
        train_arrays.append(np.array(train_t, dtype=np.uint16))
        val_arrays.append(np.array(val_t, dtype=np.uint16))
        del tokens, train_t, val_t

    train_array = np.concatenate(train_arrays)
    val_array = np.concatenate(val_arrays)
    del train_arrays, val_arrays

    train_array.tofile(os.path.join(OUTPUT_DIR, "chat_train.bin"))
    val_array.tofile(os.path.join(OUTPUT_DIR, "chat_val.bin"))

    print(f"\n=== Final ===")
    print(f"  chat_train.bin: {len(train_array):,} tokens")
    print(f"  chat_val.bin:   {len(val_array):,} tokens")

    # Cleanup intermediates
    print("\nCleaning up intermediate files...")
    for name in DATASET_LOADERS:
        path = os.path.join(OUTPUT_DIR, f"_chat_intermediate_{name}.bin")
        if os.path.exists(path):
            os.remove(path)
    print("Done!")


if __name__ == "__main__":
    main()
