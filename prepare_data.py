from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import os
from tokenizer import BPETokenizer
from collections import defaultdict

VOCAB_SIZE = 16000
DOCUMENT_SEPARATOR = "\n\n###\n\n"
TRAIN_VAL_SPLIT = 0.9
OUTPUT_DIR = "data"

# Subsample TinyStories — the full 1.9B chars would dominate the corpus.
# 1/6 keeps it as a meaningful chunk (~15% of final corpus) without drowning out the chat data.
TINYSTORIES_FRACTION = 1 / 6

# Tokenizer training sample sizes — balanced across the three datasets so the
# BPE merges capture vocabulary from each style (narrative, real dialogue, synthetic chat)
BPE_SAMPLE_FROM_TINYSTORIES = 25_000_000   # 25MB
BPE_SAMPLE_FROM_OASST = 6_000_000          # ~6MB (all of OASST basically)
BPE_SAMPLE_FROM_ULTRACHAT = 19_000_000     # 19MB

def format_oasst_conversation(tree_root, children_by_parent):
    # Given a root prompt message + a dict of all messages keyed by ID,
    # walk down the tree picking the best-ranked child at each step.
    # Returns a single string in "USER: ... \nASSISTANT: ... \n" format.
    
    current = tree_root
    text_parts = []
    
    while current is not None:
        role_label = "USER" if current["role"] == "prompter" else "ASSISTANT"
        text_parts.append(f"{role_label}: {current['text']}")
        
        # Find children of current message
        children = children_by_parent.get(current["message_id"], [])
        valid = [c for c in children if c["rank"] is not None]
        if valid:
            current = min(valid, key=lambda c: c["rank"])
        else:
            current = children[0] if children else None
    
    return "\n".join(text_parts)

def load_and_format_tinystories():
    # Use HuggingFace datasets to load TinyStories
    ds = load_dataset("roneneldan/TinyStories", split="train")
    # Join all stories with document separator
    # return DOCUMENT_SEPARATOR.join(
    #     story["text"] for i, story in enumerate(ds) if i < 1000
    # )
    return DOCUMENT_SEPARATOR.join(story["text"] for story in ds)

def load_and_format_oasst():
    # Load OASST1
    ds = load_dataset("OpenAssistant/oasst1", split="train")

    # Filter to English only
    ds = ds.filter(lambda row: row["lang"] == "en")

    # Build lookup: message_id -> message
    children_by_parent = defaultdict(list)
    for m in ds:
        children_by_parent[m["parent_id"]].append(m)

    # Find root messages (parent_id is None)
    roots = children_by_parent[None]

    # Format each tree
    formatted_conversations = []
    for root in roots:
        conversation_text = format_oasst_conversation(root, children_by_parent)
        formatted_conversations.append(conversation_text)

    return DOCUMENT_SEPARATOR.join(formatted_conversations)

def format_ultrachat_conversation(messages):
    """
    UltraChat stores each conversation as a list of {role, content} dicts.
    Convert to the same USER:/ASSISTANT: format we use for OASST so the model
    sees ONE consistent chat format across all conversational data.
    """
    parts = []
    for msg in messages:
        role_label = "USER" if msg["role"] == "user" else "ASSISTANT"
        parts.append(f"{role_label}: {msg['content']}")
    return "\n".join(parts)

def load_and_format_ultrachat():
    # Load the curated 200K-conversation version (smaller + filtered version of
    # full UltraChat — ~200K conversations, suitable for SFT). The "train_sft"
    # split is the one HuggingFace recommends for instruction-tuning models.
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")

    formatted = []
    for row in ds:
        formatted.append(format_ultrachat_conversation(row["messages"]))

    return DOCUMENT_SEPARATOR.join(formatted)

def encode_in_chunks(text, tokenizer, label):
    """Encode a text corpus chunk-by-chunk with a progress bar."""
    print(f"\nEncoding {label}...")
    tokens = []
    chunk_size = 1_000_000
    total_chunks = max(1, (len(text) + chunk_size - 1) // chunk_size)
    for chunk in tqdm(chunked(text, chunk_size), total=total_chunks, desc=label):
        tokens.extend(tokenizer.encode(chunk))
    return tokens

def split_90_10(tokens):
    """Split a token list into train (first 90%) and val (last 10%)."""
    idx = int(len(tokens) * TRAIN_VAL_SPLIT)
    return tokens[:idx], tokens[idx:]

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # === LOAD ALL THREE DATASETS ===
    print("Loading TinyStories...")
    tinystories_text = load_and_format_tinystories()
    print(f"TinyStories (full): {len(tinystories_text):,} chars")

    # Subsample TinyStories — full corpus is ~1.9B chars which would dominate
    # the mix. Take only TINYSTORIES_FRACTION so it ends up as a meaningful
    # but not overwhelming chunk (~15% of final corpus).
    keep_chars = int(len(tinystories_text) * TINYSTORIES_FRACTION)
    tinystories_text = tinystories_text[:keep_chars]
    print(f"TinyStories (subsampled to {TINYSTORIES_FRACTION:.0%}): {len(tinystories_text):,} chars")

    print("Loading OASST1...")
    oasst_text = load_and_format_oasst()
    print(f"OASST: {len(oasst_text):,} chars")

    print("Loading UltraChat...")
    ultrachat_text = load_and_format_ultrachat()
    print(f"UltraChat: {len(ultrachat_text):,} chars")

    total_chars = len(tinystories_text) + len(oasst_text) + len(ultrachat_text)
    print(f"\nTotal corpus: {total_chars:,} chars")
    print(f"  TinyStories: {len(tinystories_text)/total_chars:.1%}")
    print(f"  OASST:       {len(oasst_text)/total_chars:.1%}")
    print(f"  UltraChat:   {len(ultrachat_text)/total_chars:.1%}")

    # === TOKENIZER (load if cached, else train on balanced 3-dataset sample) ===
    tokenizer_path = os.path.join(OUTPUT_DIR, "tokenizer.json")
    tokenizer = BPETokenizer()

    if os.path.exists(tokenizer_path):
        print(f"\nLoading existing tokenizer from {tokenizer_path}")
        tokenizer.load(tokenizer_path)
    else:
        print("\nTraining tokenizer on balanced sample from all 3 datasets...")
        tokenizer_sample = (
            tinystories_text[:BPE_SAMPLE_FROM_TINYSTORIES]
            + DOCUMENT_SEPARATOR
            + oasst_text[:BPE_SAMPLE_FROM_OASST]
            + DOCUMENT_SEPARATOR
            + ultrachat_text[:BPE_SAMPLE_FROM_ULTRACHAT]
        )
        print(f"Tokenizer training sample: {len(tokenizer_sample):,} chars")
        tokenizer.train(tokenizer_sample, VOCAB_SIZE)
        tokenizer.save(tokenizer_path)

    # === ENCODE EACH DATASET SEPARATELY ===
    # Separately so we can split each 90/10 — ensures train AND val both have
    # representative samples of ALL three sources (fixes the bug where OASST
    # ended up entirely in val because it was concatenated at the end).
    ts_tokens = encode_in_chunks(tinystories_text, tokenizer, "TinyStories")
    oasst_tokens = encode_in_chunks(oasst_text, tokenizer, "OASST")
    ultrachat_tokens = encode_in_chunks(ultrachat_text, tokenizer, "UltraChat")

    # Free the corpus strings — we're done with them and they're huge
    del tinystories_text, oasst_text, ultrachat_text

    total_tokens = len(ts_tokens) + len(oasst_tokens) + len(ultrachat_tokens)
    print(f"\n=== Token counts ===")
    print(f"  TinyStories: {len(ts_tokens):>14,} ({len(ts_tokens)/total_tokens:.1%})")
    print(f"  OASST:       {len(oasst_tokens):>14,} ({len(oasst_tokens)/total_tokens:.1%})")
    print(f"  UltraChat:   {len(ultrachat_tokens):>14,} ({len(ultrachat_tokens)/total_tokens:.1%})")
    print(f"  TOTAL:       {total_tokens:>14,}")

    # === SPLIT EACH DATASET 90/10 (per-dataset, not on the combined stream) ===
    # This guarantees both train.bin and val.bin contain proportional samples
    # of ALL three datasets, not just the one that happened to come last.
    ts_train, ts_val = split_90_10(ts_tokens)
    oasst_train, oasst_val = split_90_10(oasst_tokens)
    ultrachat_train, ultrachat_val = split_90_10(ultrachat_tokens)

    # === CONCATENATE INTO FINAL TRAIN/VAL ===
    train_tokens = ts_train + oasst_train + ultrachat_train
    val_tokens = ts_val + oasst_val + ultrachat_val

    # Free intermediates
    del ts_tokens, oasst_tokens, ultrachat_tokens
    del ts_train, ts_val, oasst_train, oasst_val, ultrachat_train, ultrachat_val

    print(f"\n=== Final split ===")
    print(f"  Train: {len(train_tokens):,} tokens")
    print(f"  Val:   {len(val_tokens):,} tokens")

    # === SAVE AS UINT16 BIN FILES ===
    train_array = np.array(train_tokens, dtype=np.uint16)
    val_array = np.array(val_tokens, dtype=np.uint16)
    train_array.tofile(os.path.join(OUTPUT_DIR, "train.bin"))
    val_array.tofile(os.path.join(OUTPUT_DIR, "val.bin"))

    print(f"\nSaved train.bin ({len(train_tokens):,} tokens) and val.bin ({len(val_tokens):,} tokens)")

def chunked(s, n):
    for i in range(0, len(s), n):
        yield s[i:i+n]

if __name__ == "__main__":
    main()