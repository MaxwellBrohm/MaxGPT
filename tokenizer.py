from collections import Counter
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
        # implement the algorithm above
        # Convert input text into a list of byte integers (0..255)
        tokens = list(text.encode("utf-8"))
        
        num_merges = vocab_size - 256
            # because the first 256 IDs are already the byte vocab
        
        for i in range(num_merges):
            
            # 1. Count every adjacent pair in tokens
            pair_counts = Counter(zip(tokens, tokens[1:]))
            if not pair_counts:
                break
            
            # 2. Find the most common pair
            most_common_pair = pair_counts.most_common(1)[0][0]
            
            # 3. Mint a new token ID for it
            new_id = 256 + i
            
            # 4. Replace every occurrence of that pair in tokens with new_id
            tokens = self._replace_pair(tokens, most_common_pair, new_id)
            
            # 5. Record the merge for later use in encode()
            self.merges[most_common_pair] = new_id
            
            # 6. Record the new vocab entry — the bytes are the concatenation
            #    of the bytes of the two halves
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
    # Quick sanity test — feel free to write whatever test you want
    text = "the quick brown fox jumps over the lazy dog. " * 100
    tok = BPETokenizer()
    tok.train(text, vocab_size=300)  # 300 = 256 byte vocab + 44 merges
    
    encoded = tok.encode("the quick fox")
    decoded = tok.decode(encoded)
    
    print(f"Original: 'the quick fox'")
    print(f"Encoded:  {encoded}")
    print(f"Decoded:  '{decoded}'")
    print(f"Match: {decoded == 'the quick fox'}")

    # After your existing test
    tok.save("test_tokenizer.json")

    tok2 = BPETokenizer()
    tok2.load("test_tokenizer.json")

    encoded_again = tok2.encode("the quick fox")
    print(f"After save/load: {encoded_again}")
    print(f"Same as before: {encoded == encoded_again}")