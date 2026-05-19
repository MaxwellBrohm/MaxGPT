import numpy as np
import os

OUTPUT_DIR = "data"

# Load existing bins and combine
train = np.fromfile(os.path.join(OUTPUT_DIR, "train.bin"), dtype=np.uint16)
val = np.fromfile(os.path.join(OUTPUT_DIR, "val.bin"), dtype=np.uint16)
all_tokens = np.concatenate([train, val])
print(f"Total tokens: {len(all_tokens):,}")

# Shuffle in moderately-sized chunks (preserves local coherence — model still 
# sees coherent sentences during training, but datasets get interleaved)
chunk_size = 10_000   # ~10K tokens per chunk
n = len(all_tokens)
num_chunks = n // chunk_size
chunks = [all_tokens[i*chunk_size:(i+1)*chunk_size] for i in range(num_chunks)]
remainder = all_tokens[num_chunks*chunk_size:]

np.random.seed(42)   # reproducible
np.random.shuffle(chunks)

shuffled = np.concatenate(chunks + [remainder])

# Resplit 90/10
split_idx = int(len(shuffled) * 0.9)
new_train = shuffled[:split_idx]
new_val = shuffled[split_idx:]

# Save
new_train.tofile(os.path.join(OUTPUT_DIR, "train.bin"))
new_val.tofile(os.path.join(OUTPUT_DIR, "val.bin"))
print(f"New train.bin: {len(new_train):,} tokens, val.bin: {len(new_val):,} tokens")
print(f"OASST tokens are now distributed across both splits")