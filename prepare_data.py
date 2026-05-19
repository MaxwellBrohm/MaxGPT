from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import os
from tokenizer import BPETokenizer
from collections import defaultdict

VOCAB_SIZE = 16000                         # was 500 for smoke test
DOCUMENT_SEPARATOR = "\n\n###\n\n"
BPE_SAMPLE_FROM_TINYSTORIES = 35_000_000   # 35MB
BPE_SAMPLE_FROM_OASST = 15_000_000         # 15MB
TRAIN_VAL_SPLIT = 0.9
OUTPUT_DIR = "data"

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

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Loading TinyStories...")
    tinystories_text = load_and_format_tinystories()
    print(f"TinyStories: {len(tinystories_text):,} chars")
    
    print("Loading OASST1...")
    oasst_text = load_and_format_oasst()
    print(f"OASST: {len(oasst_text):,} chars")
    
    # Build a BALANCED sample for tokenizer training (the fix!)
    # Take 35MB from TinyStories + 15MB from OASST so the tokenizer
    # learns vocab from both styles, not just TinyStories
    tokenizer_path = os.path.join(OUTPUT_DIR, "tokenizer.json")
    tokenizer = BPETokenizer()

    if os.path.exists(tokenizer_path):
        print(f"Loading existing tokenizer from {tokenizer_path}")
        tokenizer.load(tokenizer_path)
    else:
        print("Training tokenizer on balanced sample...")
        tokenizer_sample = (
            tinystories_text[:BPE_SAMPLE_FROM_TINYSTORIES]
            + DOCUMENT_SEPARATOR
            + oasst_text[:BPE_SAMPLE_FROM_OASST]
        )
        tokenizer.train(tokenizer_sample, VOCAB_SIZE)
        tokenizer.save(tokenizer_path)
        
    # Combine the FULL corpus (now that tokenizer is trained, encode everything)
    full_corpus = tinystories_text + DOCUMENT_SEPARATOR + oasst_text
    print(f"Full corpus: {len(full_corpus):,} chars")
    
    # Tokenize the full corpus in chunks with a progress bar
    print("Encoding full corpus (this will take ~1-2 hours)...")
    chunk_size = 1_000_000   # 1MB chunks
    all_tokens = []
    for chunk in tqdm(chunked(full_corpus, chunk_size), total=len(full_corpus)//chunk_size):
        chunk_tokens = tokenizer.encode(chunk)
        all_tokens.extend(chunk_tokens)
    
    print(f"Total tokens: {len(all_tokens):,}")
    
    # Train/val split (90/10)
    split_index = int(len(all_tokens) * TRAIN_VAL_SPLIT)
    train_tokens = all_tokens[:split_index]
    val_tokens = all_tokens[split_index:]
    
    # Save as binary files (uint16 since vocab fits in 16 bits)
    train_array = np.array(train_tokens, dtype=np.uint16)
    val_array = np.array(val_tokens, dtype=np.uint16)
    train_array.tofile(os.path.join(OUTPUT_DIR, "train.bin"))
    val_array.tofile(os.path.join(OUTPUT_DIR, "val.bin"))
    
    print(f"Saved train.bin ({len(train_tokens):,} tokens) and val.bin ({len(val_tokens):,} tokens)")

def chunked(s, n):
    for i in range(0, len(s), n):
        yield s[i:i+n]

if __name__ == "__main__":
    main()