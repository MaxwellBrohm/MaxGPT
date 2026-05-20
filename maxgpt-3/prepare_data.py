"""
MaxGPT-3 data preparation pipeline.

7 datasets, ~5B total unique tokens, per-dataset 90/10 split,
multiprocessing-accelerated encoding (~20x faster than single-threaded).

Each dataset is loaded, formatted, encoded, and saved separately as an
intermediate .bin file. After all 7 are processed, they're combined into
the final train.bin / val.bin with proportional 90/10 split per dataset.

Targets ~5B unique tokens for ~4 epochs over MaxGPT-3's 20B-pass training.
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
VOCAB_SIZE = 16000
DOCUMENT_SEPARATOR = "\n\n###\n\n"
TRAIN_VAL_SPLIT = 0.9
OUTPUT_DIR = "data"

# Per-dataset character targets. Roughly chosen so each dataset contributes
# the planned proportion of total tokens (assuming ~5x compression on average).
# TinyStories compresses ~7x because of its limited vocab; chat data ~3.5x.
TARGET_CHARS = {
    "tinystories":  1_900_000_000,   # full corpus, ~270M tokens
    "oasst":          400_000_000,    # OASST1 + OASST2 combined, ~150M tokens
    "ultrachat":    7_000_000_000,   # full UltraChat, ~2B tokens (the bulk of chat data)
    "cosmopedia":   7_500_000_000,   # subset of 25B-token corpus, ~1.5B tokens
    "openorca":     3_500_000_000,   # subset of 4M examples, ~700M tokens
    "wildchat":     2_000_000_000,   # full WildChat-1M English subset, ~400M tokens
    "hhrlhf":         500_000_000,    # full Anthropic HH-RLHF, ~100M tokens
}

# Tokenizer training sample sizes — balanced across all 7 datasets so BPE merges
# capture vocabulary from each style. Total ~70MB sample.
BPE_SAMPLE_SIZES = {
    "tinystories":  20_000_000,   # 20MB
    "oasst":         5_000_000,   # 5MB (all of OASST basically)
    "ultrachat":    15_000_000,   # 15MB
    "cosmopedia":   15_000_000,   # 15MB
    "openorca":      8_000_000,   # 8MB
    "wildchat":      5_000_000,   # 5MB
    "hhrlhf":        2_000_000,   # 2MB
}

# Multiprocessing config
# Each worker imports pandas/numpy/datasets (~300MB each), so too many workers
# eats RAM fast. 8 workers gives ~8x speedup with ~2.5GB total worker overhead.
N_WORKERS = 8
ENCODE_CHUNK_SIZE = 500_000   # 500KB per worker chunk (was 1MB — reduces per-call memory)


# ====================
# DATASET LOADERS / FORMATTERS
# ====================

def format_oasst_conversation(tree_root, children_by_parent):
    """Walk down the OASST tree picking best-ranked child at each step.
    Returns 'USER: ... \nASSISTANT: ...' formatted text."""
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


def load_tinystories():
    """Full TinyStories — narrative coherence at small scale."""
    ds = load_dataset("roneneldan/TinyStories", split="train")
    target = TARGET_CHARS["tinystories"]
    parts = []
    chars = 0
    for example in tqdm(ds, desc="TinyStories"):
        text = example["text"]
        parts.append(text)
        chars += len(text)
        if chars >= target:
            break
    return DOCUMENT_SEPARATOR.join(parts)


def load_oasst():
    """OASST1 + OASST2 combined. Real human-assistant conversations."""
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
                formatted_conversations.append(
                    format_oasst_conversation(root, children_by_parent)
                )
        except Exception as e:
            print(f"  Skipping {hf_name}: {e}")

    return DOCUMENT_SEPARATOR.join(formatted_conversations)


def load_ultrachat():
    """Full UltraChat (stingning/ultrachat). 1.5M synthetic chat conversations.
    Each row's 'data' field is a list of strings alternating user/assistant."""
    ds = load_dataset("stingning/ultrachat", split="train", streaming=True)
    target = TARGET_CHARS["ultrachat"]
    parts = []
    chars = 0
    for example in tqdm(ds, desc="UltraChat"):
        # Convert the list-of-turns to USER:/ASSISTANT: format
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


def load_cosmopedia():
    """Cosmopedia v2 — synthetic educational content. Subset of HuggingFaceTB/smollm-corpus."""
    ds = load_dataset(
        "HuggingFaceTB/smollm-corpus",
        "cosmopedia-v2",
        split="train",
        streaming=True,
    )
    target = TARGET_CHARS["cosmopedia"]
    parts = []
    chars = 0
    for example in tqdm(ds, desc="Cosmopedia"):
        text = example.get("text", "")
        if not text:
            continue
        parts.append(text)
        chars += len(text)
        if chars >= target:
            break
    return DOCUMENT_SEPARATOR.join(parts)


def load_openorca():
    """OpenOrca — instruction-response pairs. 4M examples; we subsample."""
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
    """WildChat-1M — real ChatGPT conversations (consented).
    Each row has 'conversation' field (list of {role, content} dicts) + 'language'."""
    ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
    target = TARGET_CHARS["wildchat"]
    parts = []
    chars = 0
    for example in tqdm(ds, desc="WildChat"):
        # Filter to English
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
    """Anthropic HH-RLHF helpful/harmless. Uses 'chosen' (the preferred response).
    Format is '\n\nHuman: ...\n\nAssistant: ...' which we convert to USER:/ASSISTANT:."""
    ds = load_dataset("Anthropic/hh-rlhf", split="train")
    target = TARGET_CHARS["hhrlhf"]
    parts = []
    chars = 0
    for example in tqdm(ds, desc="HH-RLHF"):
        raw = example.get("chosen", "")
        if not raw:
            continue
        # Convert "Human:" → "USER:" and "Assistant:" → "ASSISTANT:"
        text = raw.replace("\n\nHuman:", "\nUSER:").replace("\n\nAssistant:", "\nASSISTANT:")
        text = text.strip()
        parts.append(text)
        chars += len(text)
        if chars >= target:
            break
    return DOCUMENT_SEPARATOR.join(parts)


DATASET_LOADERS = {
    "tinystories": load_tinystories,
    "oasst":       load_oasst,
    "ultrachat":   load_ultrachat,
    "cosmopedia":  load_cosmopedia,
    "openorca":    load_openorca,
    "wildchat":    load_wildchat,
    "hhrlhf":      load_hhrlhf,
}


# ====================
# MULTIPROCESSING ENCODING
# ====================
# Each worker process loads its own tokenizer once via the initializer,
# then receives chunks via imap and returns lists of token IDs.

_worker_tokenizer = None

def _init_worker(tokenizer_path):
    """Initializer for each worker process — loads tokenizer once per worker."""
    global _worker_tokenizer
    _worker_tokenizer = BPETokenizer()
    _worker_tokenizer.load(tokenizer_path)


def _encode_chunk(chunk):
    """Worker function — encodes a single chunk using the per-worker tokenizer."""
    return _worker_tokenizer.encode(chunk)


def chunked(s, n):
    """Yield n-char chunks from string s."""
    for i in range(0, len(s), n):
        yield s[i:i + n]


def encode_parallel_streaming(text, tokenizer_path, label, output_path):
    """Encode a large text using N_WORKERS processes in parallel.
    Streams tokens DIRECTLY TO DISK as each chunk completes — never holds the
    full token list in memory. Critical for large datasets (Cosmopedia / UltraChat
    can produce 1B+ tokens which would be 28GB+ as a Python list).
    Returns the total token count."""
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
                # Convert to uint16 and write to disk immediately
                np.array(chunk_tokens, dtype=np.uint16).tofile(out_f)
                total_tokens += len(chunk_tokens)
                # chunk_tokens goes out of scope here, freed by GC
    return total_tokens


# ====================
# HELPERS
# ====================

def split_90_10(tokens):
    """Split a token list into train (90%) and val (10%)."""
    idx = int(len(tokens) * TRAIN_VAL_SPLIT)
    return tokens[:idx], tokens[idx:]


def save_intermediate(tokens, name):
    """Save a per-dataset token bin to disk for later combining."""
    path = os.path.join(OUTPUT_DIR, f"_intermediate_{name}.bin")
    np.array(tokens, dtype=np.uint16).tofile(path)
    return path


def load_intermediate(name):
    """Load a per-dataset token bin saved by save_intermediate."""
    path = os.path.join(OUTPUT_DIR, f"_intermediate_{name}.bin")
    return np.fromfile(path, dtype=np.uint16)


# ====================
# MAIN PIPELINE
# ====================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tokenizer_path = os.path.join(OUTPUT_DIR, "tokenizer.json")

    # =========================================================
    # PHASE 1: Load + format all 7 datasets (keep in memory only
    # long enough to train tokenizer + encode each one)
    # =========================================================
    all_texts = {}
    print("\n" + "=" * 70 + "\nPHASE 1: Loading + formatting datasets\n" + "=" * 70)
    for name, loader in DATASET_LOADERS.items():
        print(f"\nLoading {name}...")
        try:
            all_texts[name] = loader()
            print(f"  {name}: {len(all_texts[name]):,} chars")
        except Exception as e:
            print(f"  ERROR loading {name}: {e}")
            print(f"  Skipping {name} — continuing with remaining datasets.")
            all_texts[name] = ""

    total_chars = sum(len(t) for t in all_texts.values())
    print(f"\nTotal raw corpus: {total_chars:,} chars")
    for name, text in all_texts.items():
        if total_chars > 0:
            print(f"  {name:12s}: {len(text):>14,} chars ({len(text)/total_chars:.1%})")

    # =========================================================
    # PHASE 2: Train tokenizer (or load if already cached)
    # =========================================================
    print("\n" + "=" * 70 + "\nPHASE 2: Tokenizer\n" + "=" * 70)

    if os.path.exists(tokenizer_path):
        print(f"Loading existing tokenizer from {tokenizer_path}")
        tokenizer = BPETokenizer()
        tokenizer.load(tokenizer_path)
    else:
        print("Building balanced sample from all 7 datasets for tokenizer training...")
        sample_parts = []
        for name in DATASET_LOADERS:
            text = all_texts.get(name, "")
            n = min(BPE_SAMPLE_SIZES[name], len(text))
            sample_parts.append(text[:n])
        tokenizer_sample = (DOCUMENT_SEPARATOR + DOCUMENT_SEPARATOR).join(sample_parts)
        print(f"Tokenizer sample: {len(tokenizer_sample):,} chars")

        print("Training tokenizer (this takes ~30-60 min for 16K vocab)...")
        tokenizer = BPETokenizer()
        tokenizer.train(tokenizer_sample, VOCAB_SIZE)
        tokenizer.save(tokenizer_path)
        print(f"Saved tokenizer to {tokenizer_path}")
        del tokenizer_sample

    # =========================================================
    # PHASE 3: Encode each dataset in parallel, save intermediate bin files
    # =========================================================
    print("\n" + "=" * 70 + "\nPHASE 3: Parallel encoding (one dataset at a time)\n" + "=" * 70)

    token_counts = {}
    for name in DATASET_LOADERS:
        text = all_texts.get(name, "")
        if not text:
            print(f"\nSkipping {name} (no text loaded)")
            continue

        intermediate_path = os.path.join(OUTPUT_DIR, f"_intermediate_{name}.bin")
        if os.path.exists(intermediate_path):
            # Resume support — if we already encoded this dataset, skip re-encoding
            existing = np.fromfile(intermediate_path, dtype=np.uint16)
            print(f"\nSkipping {name} encoding — already done ({len(existing):,} tokens cached)")
            token_counts[name] = len(existing)
            # Free the text since we don't need it anymore
            all_texts[name] = ""
            continue

        print(f"\n--- {name} ---")
        # Stream tokens directly to disk — never builds the full token list in memory
        n_tokens = encode_parallel_streaming(text, tokenizer_path, name, intermediate_path)
        token_counts[name] = n_tokens
        print(f"  {name}: {n_tokens:,} tokens saved")
        # Free the original text now that it's encoded
        all_texts[name] = ""

    total_tokens = sum(token_counts.values())
    print(f"\n=== Tokens per dataset ===")
    for name, count in token_counts.items():
        if total_tokens > 0:
            print(f"  {name:12s}: {count:>14,} ({count/total_tokens:.1%})")
    print(f"  TOTAL       : {total_tokens:>14,}")

    # =========================================================
    # PHASE 4: Per-dataset 90/10 split, then combine into final train/val bins
    # =========================================================
    print("\n" + "=" * 70 + "\nPHASE 4: Per-dataset 90/10 split + combine\n" + "=" * 70)

    train_arrays = []
    val_arrays = []
    for name in DATASET_LOADERS:
        if name not in token_counts:
            continue
        tokens = load_intermediate(name)
        train_t, val_t = split_90_10(tokens.tolist())
        train_arrays.append(np.array(train_t, dtype=np.uint16))
        val_arrays.append(np.array(val_t, dtype=np.uint16))
        del tokens, train_t, val_t

    train_array = np.concatenate(train_arrays)
    val_array = np.concatenate(val_arrays)
    del train_arrays, val_arrays

    train_array.tofile(os.path.join(OUTPUT_DIR, "train.bin"))
    val_array.tofile(os.path.join(OUTPUT_DIR, "val.bin"))

    print(f"\n=== Final ===")
    print(f"  train.bin: {len(train_array):,} tokens")
    print(f"  val.bin:   {len(val_array):,} tokens")

    # Cleanup intermediates (comment this out if you want to keep them for debugging)
    print("\nCleaning up intermediate files...")
    for name in DATASET_LOADERS:
        path = os.path.join(OUTPUT_DIR, f"_intermediate_{name}.bin")
        if os.path.exists(path):
            os.remove(path)
    print("Done!")


if __name__ == "__main__":
    main()
