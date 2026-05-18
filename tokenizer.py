from collections import Counter
from tqdm import tqdm
import json

class BPETokenizer:
    def __init__(self):
        # state: self.merges (dict mapping pair -> new id) and self.vocab (dict mapping id -> bytes)
        self.merges = {}
        # will map pair-of-token-IDs -> new token ID
        # example: {(116, 104): 256} means "the pair (t, h) becomes token 256"
    
        self.vocab = {i: bytes([i]) for i in range(256)}
        # the initial 256 tokens are just the 256 possible byte values
        # i.e., vocab[97] == b"a", vocab[98] == b"b", etc.
    
    def train(self, text: str, vocab_size: int) -> None:
        # ============================================================
        # OPTIMIZED BPE TRAINING
        # Per-merge cost: O(occurrences of merged pair), not O(total tokens)
        # ============================================================
        
        # --- STEP 1: Initialize tokens + linked list pointers ---
        tokens = list(text.encode("utf-8"))
        n = len(tokens)
        
        # Doubly-linked list pointers so we can splice elements in O(1)
        prev = [i - 1 for i in range(n)]      # prev[0] = -1 (sentinel: no prev)
        next_ = [i + 1 for i in range(n)]     # next_[n-1] = n (sentinel: no next)
        is_dead = [False] * n                 # tracks spliced-out positions
        
        # --- STEP 2: One-time O(n) pass to populate pair counts + positions ---
        pair_counts = {}          # pair -> count
        pair_positions = {}       # pair -> SET of positions (left index)
        
        for i in range(n - 1):
            pair = (tokens[i], tokens[i+1])
            pair_counts[pair] = pair_counts.get(pair, 0) + 1
            pair_positions.setdefault(pair, set()).add(i)
        
        num_merges = vocab_size - 256

        print(f"Initial corpus: {n:,} tokens, {len(pair_counts):,} unique pairs")
        print(f"Training {num_merges:,} merges...")
        
        # --- STEP 3: The merge loop ---
        for merge_index in tqdm(range(num_merges), desc="Training BPE"):
            
            # 3a. Find max pair (and bail if no pairs left)
            if not pair_counts: 
                break
            (a, b) = max(pair_counts, key=pair_counts.get)
            if pair_counts[(a, b)] <= 0: 
                break
            
            new_id = 256 + merge_index
            
            # 3b. Snapshot positions in sorted order, then process each
            positions_to_process = sorted(pair_positions[(a, b)])
            
            for pos in positions_to_process:
                
                # 3c. Validity checks — skip dead or stale positions
                if is_dead[pos]: 
                    continue
                
                j = next_[pos]    # position of the right half of the pair
                if j >= n or is_dead[j]: 
                    continue
                if tokens[pos] != a or tokens[j] != b: 
                    continue
                
                # 3d. Find neighbors
                left_idx = prev[pos]      # -1 if no left
                right_idx = next_[j]      # n if no right
                
                # 3e. Update LEFT neighbor pair counts
                # Before: (tokens[left_idx], a)  ->  After: (tokens[left_idx], new_id)
                if left_idx >= 0:
                    old_left = (tokens[left_idx], a)
                    new_left = (tokens[left_idx], new_id)
                    pair_counts[old_left] -= 1
                    pair_positions[old_left].discard(left_idx)
                    pair_counts[new_left] = pair_counts.get(new_left, 0) + 1
                    pair_positions.setdefault(new_left, set()).add(left_idx)
                
                # 3f. Update RIGHT neighbor pair counts
                # Before: (b, tokens[right_idx]) at position j
                # After:  (new_id, tokens[right_idx]) at position pos
                if right_idx < n:
                    old_right = (b, tokens[right_idx])
                    new_right = (new_id, tokens[right_idx])
                    pair_counts[old_right] -= 1
                    pair_positions[old_right].discard(j)
                    pair_counts[new_right] = pair_counts.get(new_right, 0) + 1
                    pair_positions.setdefault(new_right, set()).add(pos)
                
                # 3g. Decrement the merged pair's count for THIS position
                pair_counts[(a, b)] -= 1
                
                # 3h. Splice — replace pos with new_id, remove j from the list
                tokens[pos] = new_id
                next_[pos] = right_idx
                if right_idx < n:
                    prev[right_idx] = pos
                is_dead[j] = True
            
            # 3i. Clean up: clear the merged pair's position set
            pair_positions[(a, b)] = set()
            pair_counts[(a, b)] = 0
            
            # 3j. Record the merge in tokenizer state
            self.merges[(a, b)] = new_id
            self.vocab[new_id] = self.vocab[a] + self.vocab[b]

        print(f"Done! Final vocab size: {len(self.vocab):,}")
        # All done. tokens, prev, next_, is_dead, pair_counts, pair_positions can be discarded.

    def train_naive(self, text: str, vocab_size: int) -> None:
        """Original naive O(n) per merge implementation. Kept for benchmarking."""
        # Reset state (so we can call this on a fresh tokenizer)
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        
        tokens = list(text.encode("utf-8"))
        num_merges = vocab_size - 256
        
        for i in tqdm(range(num_merges), desc="Training BPE (naive)"):
            pair_counts = Counter(zip(tokens, tokens[1:]))
            if not pair_counts:
                break
            most_common_pair = pair_counts.most_common(1)[0][0]
            new_id = 256 + i
            tokens = self._replace_pair(tokens, most_common_pair, new_id)
            self.merges[most_common_pair] = new_id
            self.vocab[new_id] = self.vocab[most_common_pair[0]] + self.vocab[most_common_pair[1]]
            
    def encode(self, text: str) -> list[int]:
        # apply learned merges to new text    
        # Start with the byte representation
        tokens = list(text.encode("utf-8"))
        
        # Apply learned merges, always picking the one learned EARLIEST 
        # (i.e., lowest new_id in self.merges) that applies to the current tokens
        
        while len(tokens) >= 2:
            # Find all adjacent pairs currently in tokens
            pairs = set(zip(tokens, tokens[1:]))
            
            # Of those, find the pair that's in self.merges with the LOWEST id
            # (= was learned earliest during training)
            # If no pair in `pairs` is in self.merges, we're done
            
            applicable = pairs & self.merges.keys()
            if not applicable:
                break
            
            best_pair = min(applicable, key=lambda p: self.merges[p])
            
            # Apply the merge (reuse the helper from training)
            tokens = self._replace_pair(tokens, best_pair, self.merges[best_pair])
        
        return tokens
    
    def decode(self, ids: list[int]) -> str:
        # ids → bytes → utf-8 string
        # Look up each ID's bytes, concatenate, decode as UTF-8
        
        all_bytes = b"".join(self.vocab[id] for id in ids)
        
        return all_bytes.decode("utf-8", errors="replace")
            # errors="replace" is important — if a model generates a weird
            # token sequence that doesn't decode as valid UTF-8, we get a 
            # replacement character (�) instead of crashing
    
    def save(self, path: str) -> None:
        # Convert self.merges (which has tuple keys, not JSON-friendly) 
        # into a list of plain records that preserve the training order
        # Note: dicts preserve insertion order in Python 3.7+,
        # so iterating gives merges in the order they were learned
        merges_list = [
            {"pair": [a, b], "id": new_id}
            for (a, b), new_id in self.merges.items()
        ]
        data = {"merges": merges_list}
        
        with open(path, "w") as f: 
            json.dump(data, f)

    def load(self, path: str) -> None:
        with open(path) as f:
            data = json.load(f)
        
        # Reset state to fresh
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        
        # Replay each merge in training order, rebuilding both merges and vocab
        for entry in data["merges"]:
            a, b = entry["pair"]
            new_id = entry["id"]
            pair = (a, b)
            
            self.merges[pair] = new_id
            # Vocab entry = concatenation of the two halves' bytes
            self.vocab[new_id] = self.vocab[a] + self.vocab[b]

    def _replace_pair(self, tokens, pair, new_id):
        # Walks the tokens list, replacing every occurrence of `pair` with `new_id`.
        # Returns a new list (doesn't mutate the input).
    
        result = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i+1] == pair[1]:
                result.append(new_id)
                i = i + 2    # skip past both elements we just merged
            else:
                result.append(tokens[i])
                i = i + 1
        return result

if __name__ == "__main__":
    import urllib.request
    import os
    import time
    
    # Download Tiny Shakespeare (cache locally so we only download once)
    cache_path = "/tmp/tinyshakespeare.txt"
    if not os.path.exists(cache_path):
        print("Downloading Tiny Shakespeare...")
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        urllib.request.urlretrieve(url, cache_path)
    
    with open(cache_path) as f:
        text = f.read()
    print(f"Benchmark text: {len(text):,} chars\n")
    
    vocab_size = 1000   # 744 merges
    
    # === NEW algorithm ===
    tok_new = BPETokenizer()
    start = time.time()
    tok_new.train(text, vocab_size=vocab_size)
    new_time = time.time() - start
    print(f"\nNEW algorithm: {new_time:.2f}s\n")
    
    # === OLD (naive) algorithm ===
    tok_old = BPETokenizer()
    start = time.time()
    tok_old.train_naive(text, vocab_size=vocab_size)
    old_time = time.time() - start
    print(f"\nOLD algorithm: {old_time:.2f}s\n")
    
    # === RESULTS ===
    print(f"=== Speedup: {old_time / new_time:.1f}x ===")
    
    # Quick correctness check
    sample = "To be or not to be"
    enc_new = tok_new.encode(sample)
    enc_old = tok_old.encode(sample)
    print(f"New algorithm encodes '{sample}' as: {enc_new}")
    print(f"Old algorithm encodes '{sample}' as: {enc_old}")
    print(f"(Different token IDs are expected due to tie-breaking — both are valid)")