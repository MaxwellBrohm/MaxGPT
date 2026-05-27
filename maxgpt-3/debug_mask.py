"""Debug script for the loss masking — figures out why 100% is masked.

Compares the OLD marker patterns (with "\n" prefix) vs the NEW patterns
(without "\n"). If the OLD count is 0 but the NEW count is high, that
confirms the byte-level BPE bug: the "\n" gets absorbed into a merge with
the preceding character, breaking the [10, ...] prefix match.
"""

import numpy as np
import os
from tokenizer import BPETokenizer

OUTPUT_DIR = "data"

# 1. Check file sizes
print("=" * 60)
print("FILE SIZES")
print("=" * 60)
for name in ["chat_train.bin", "chat_val.bin", "tokenizer.json"]:
    path = os.path.join(OUTPUT_DIR, name)
    if os.path.exists(path):
        sz = os.path.getsize(path)
        print(f"  {name}: {sz:,} bytes ({sz / 1e6:.1f} MB)")
    else:
        print(f"  {name}: MISSING")

# 2. Load tokenizer and encode markers (BOTH variants)
tok = BPETokenizer()
tok.load(os.path.join(OUTPUT_DIR, "tokenizer.json"))
print(f"\nTokenizer vocab size: {len(tok.vocab):,}")

# OLD pattern (with leading \n) — what finetune_data.py used to use
am_old = tok.encode("\nASSISTANT:")
um_old = tok.encode("\nUSER:")
# NEW pattern (no leading \n) — what finetune_data.py uses now
am_new = tok.encode("ASSISTANT:")
um_new = tok.encode("USER:")

print(f"\n--- Marker encodings ---")
print(f"OLD '\\nASSISTANT:' -> {am_old}")
print(f"NEW   'ASSISTANT:' -> {am_new}")
print(f"OLD '\\nUSER:'      -> {um_old}")
print(f"NEW   'USER:'      -> {um_new}")

print(f"\n--- Decoding individual tokens (to verify what each represents) ---")
for label, tokens in [
    ("\\nASSISTANT:", am_old),
    ("ASSISTANT:",    am_new),
    ("\\nUSER:",      um_old),
    ("USER:",         um_new),
]:
    decoded_parts = [repr(tok.decode([t])) for t in tokens]
    print(f"  {label!r:18s} = {tokens} -> {' + '.join(decoded_parts)}")

# 3. Load chat_train.bin and check structure
data = np.memmap(os.path.join(OUTPUT_DIR, "chat_train.bin"), dtype=np.uint16, mode="r")
print(f"\nchat_train.bin: {len(data):,} tokens")

# 4. Show what the first 500 tokens decode to (should look like chat data!)
sample = data[:500].tolist()
print(f"\n--- First 500 tokens decoded ---")
print("---")
print(tok.decode(sample))
print("---")

# 5. Count marker appearances in first 100K tokens (BOTH variants)
def find_pattern_positions(tokens, pattern):
    n, m = len(tokens), len(pattern)
    if m == 0 or n < m:
        return []
    matches = np.ones(n - m + 1, dtype=bool)
    for k in range(m):
        matches &= (tokens[k : n - m + 1 + k] == pattern[k])
    return np.where(matches)[0]

check_window = min(100_000, len(data))
sample_data = np.array(data[:check_window], dtype=np.uint16)

print(f"\n--- Marker counts in first {check_window:,} tokens ---")
for label, pattern_old, pattern_new in [
    ("ASSISTANT", am_old, am_new),
    ("USER",      um_old, um_new),
]:
    old_arr = np.array(pattern_old, dtype=np.uint16)
    new_arr = np.array(pattern_new, dtype=np.uint16)
    old_positions = find_pattern_positions(sample_data, old_arr)
    new_positions = find_pattern_positions(sample_data, new_arr)
    print(f"  OLD {label:9s} {pattern_old}: {len(old_positions):5d} matches")
    print(f"  NEW {label:9s} {pattern_new}: {len(new_positions):5d} matches")
    print()

# 6. Show context around first occurrence of the NEW pattern
new_am_arr = np.array(am_new, dtype=np.uint16)
new_am_positions = find_pattern_positions(sample_data, new_am_arr)
if len(new_am_positions) > 0:
    p = int(new_am_positions[0])
    ctx_start = max(0, p - 30)
    ctx_end = min(check_window, p + 30)
    print(f"--- Context around first NEW ASSISTANT match (position {p}) ---")
    print(f"  Tokens: {sample_data[ctx_start:ctx_end].tolist()}")
    print(f"  Decoded: {repr(tok.decode(sample_data[ctx_start:ctx_end].tolist()))}")
    # Specifically show the token RIGHT BEFORE the marker — that's the one
    # that absorbed the "\n" and broke the old pattern matching.
    if p > 0:
        prev_tok = int(sample_data[p - 1])
        prev_bytes = tok.vocab[prev_tok]
        print(f"\n  Token immediately before ASSISTANT marker:")
        print(f"    ID:    {prev_tok}")
        print(f"    Bytes: {prev_bytes!r}")
        print(f"    (If this contains '\\n', that's the merge that absorbed our anchor!)")
