"""
Data loader for the SFT (fine-tuning) phase — with PROPER LOSS MASKING.

For SFT, the standard practice is:
  - Train ONLY on the assistant's response tokens
  - Mask out (don't compute loss on) the user's tokens and role markers
  - The user tokens are still SEEN by the model (they're context for predicting
    the response), but no gradient flows back to "improve" the user-side
    predictions. For a small-capacity model, this is the difference between
    "spending half its capacity learning to imitate the user" vs. "all gradient
    focused on producing good replies."

Implementation:
  - At module load: encode the role marker token sequences ("\nASSISTANT:" and
    "\nUSER:") once. These are the anchors we use to identify role boundaries
    in the token stream.
  - When sampling a batch: extend each chunk by 2000 tokens of LOOKBACK so we
    can correctly determine the role state at the start of every sampled chunk.
    Otherwise a chunk starting mid-way through an assistant response would get
    incorrectly masked.
  - Build a 0/1 mask: 1 = assistant content (train on this), 0 = user content
    or role markers (mask out).
  - Apply the mask by setting masked positions in y to -100. PyTorch's
    cross_entropy uses ignore_index=-100 by default, so masked positions
    contribute nothing to the loss.
"""

import numpy as np
import torch
from tokenizer import BPETokenizer

# ====================
# MODULE-LEVEL SETUP
# ====================

# Memmap the chat-only binary files (lazy disk reads, no full load into RAM)
train_data = np.memmap("data/chat_train.bin", dtype=np.uint16, mode="r")
val_data = np.memmap("data/chat_val.bin", dtype=np.uint16, mode="r")

# Load the tokenizer once so we can encode role markers
_tokenizer = BPETokenizer()
_tokenizer.load("data/tokenizer.json")

# Role marker BYTES — we search at the BYTE level after decoding tokens, which
# is robust to ALL of BPE's merge-by-rank weirdness around marker boundaries.
#
# Why this is necessary (token-level matching kept failing):
#   Byte-level BPE has no pre-tokenization regex — merges fire purely by global
#   rank. So the same substring tokenizes DIFFERENTLY depending on surrounding
#   bytes. Two merges in particular wreck token-level marker matching:
#     1. Preceding "\n" gets eaten: ".\n", "?\n", "!\n", " \n" are common merges
#        from pre-training, so the "\n" before "\nASSISTANT:" never survives as
#        a standalone token in real data.
#     2. Trailing ": " gets eaten: (":", " ") merges in most chat positions
#        because "X: " is everywhere. So "ASSISTANT:" → [504, 58] standalone
#        becomes [504, <": "_merged>, ...] in real data, and "USER:" → [8193]
#        gets eaten into [<"USER: "_merged>, ...].
#   debug_mask.py confirmed both: only 1/9 markers found at the token level even
#   though the data visibly contains hundreds of role boundaries.
#
# Byte-level search bypasses both issues completely.
ASSISTANT_MARKER_BYTES = b"ASSISTANT:"
USER_MARKER_BYTES = b"USER:"

# Pre-compute the byte length of every vocab token once at module load. Used to
# derive per-chunk byte offsets via a vectorized cumsum (no Python loop).
_VOCAB_BYTE_LEN = np.array(
    [len(_tokenizer.vocab[i]) for i in range(len(_tokenizer.vocab))],
    dtype=np.int64,
)

# How far to look back in the token stream when sampling a chunk, so we know
# the role state at the chunk's start. 2000 tokens reliably catches at least
# one role marker for our average ~200-token turns.
LOOKBACK = 2000

print(f"[finetune_data] Loaded tokenizer (vocab {len(_tokenizer.vocab):,})")
print(f"[finetune_data] Role markers (byte-level search): {ASSISTANT_MARKER_BYTES!r}, {USER_MARKER_BYTES!r}")
print(f"[finetune_data] Lookback for state detection: {LOOKBACK} tokens")


# ====================
# MASK BUILDING
# ====================

def _build_loss_mask(tokens: np.ndarray, initial_state: bool = False) -> np.ndarray:
    """Build a 0/1 mask the same length as `tokens`:
       1 = assistant content (train on this)
       0 = user content, role markers, or pre-first-role region

    Detection strategy: decode the chunk to raw bytes, find marker substrings
    via bytes.find(), then map byte positions back to token indices via a
    cumsum lookup. This is robust to BPE's merge-by-rank tokenization (which
    makes token-level pattern matching unreliable — see the comment block at
    module load for the full story).
    """
    n = len(tokens)
    if n == 0:
        return np.zeros(0, dtype=np.uint8)

    mask = np.zeros(n, dtype=np.uint8)

    # 1. Decode each token's bytes and stitch into a single byte stream
    token_bytes_list = [_tokenizer.vocab[int(t)] for t in tokens]
    full_bytes = b"".join(token_bytes_list)

    # 2. Per-token byte offsets via vectorized cumsum
    #    byte_offsets[i] = byte position where token i begins
    #    byte_offsets[n] = total byte length
    byte_lens = _VOCAB_BYTE_LEN[tokens]
    byte_offsets = np.concatenate(([0], np.cumsum(byte_lens))).astype(np.int64)

    # 3. Find every marker occurrence at the BYTE level, then map back to tokens
    boundaries = []  # (first_marker_token, role_after_marker, num_marker_tokens)
    for marker_bytes, role_after in [
        (ASSISTANT_MARKER_BYTES, True),
        (USER_MARKER_BYTES, False),
    ]:
        m = len(marker_bytes)
        start = 0
        while True:
            idx = full_bytes.find(marker_bytes, start)
            if idx == -1:
                break
            # Map byte positions back to token indices:
            #   tok_start = the token containing the FIRST byte of the marker
            #   tok_end   = the token containing the LAST  byte of the marker
            # Any token that overlaps the marker — even partially — is treated
            # as a marker token (masked out). This is correct: a token spanning
            # "X" + "ASSI" shouldn't be trained on as user content, nor should
            # a token spanning ":" + " r" be trained on as assistant content.
            tok_start = int(np.searchsorted(byte_offsets, idx, side="right")) - 1
            tok_end = int(np.searchsorted(byte_offsets, idx + m - 1, side="right")) - 1
            boundaries.append((tok_start, role_after, tok_end - tok_start + 1))
            start = idx + m

    boundaries.sort(key=lambda b: b[0])

    # 4. Walk boundaries left-to-right, filling regions based on current state
    in_assistant = initial_state
    cursor = 0
    for tok_pos, role_after, marker_tok_len in boundaries:
        # Defensive: skip markers overlapping one we already passed (e.g. a
        # boundary token shared with the previous marker's tail).
        if tok_pos < cursor:
            continue
        if in_assistant:
            mask[cursor:tok_pos] = 1
        # Marker tokens [tok_pos, tok_pos + marker_tok_len) stay 0
        in_assistant = role_after
        cursor = tok_pos + marker_tok_len

    # Trailing region after the last marker
    if in_assistant:
        mask[cursor:n] = 1

    return mask


# ====================
# BATCH SAMPLING
# ====================

def get_batch(split, batch_size, context_window, device):
    """Sample a batch of (input, target) chunks from the chat data, with loss
    masking applied: user positions in the target have value -100 so they're
    ignored by cross_entropy.

    Each sampled chunk is extended with LOOKBACK tokens before the chunk start
    so we can correctly determine the role state at position 0 of the chunk.
    Without this, chunks that start mid-assistant-response would get incorrectly
    masked out.
    """
    data = train_data if split == "train" else val_data

    # Sample starting positions. We need at least LOOKBACK tokens before the
    # chunk and at least (context_window + 1) tokens after for the y shift.
    ix = torch.randint(
        low=LOOKBACK,
        high=len(data) - context_window - 1,
        size=(batch_size,),
    )

    x_list = []
    y_masked_list = []

    for i in ix:
        i_int = int(i.item())

        # Extract extended chunk: [i - LOOKBACK, i + context_window + 1]
        ext_start = i_int - LOOKBACK
        ext_end = i_int + context_window + 1
        ext_chunk = np.array(data[ext_start:ext_end], dtype=np.uint16)

        # Build mask across the extended chunk so we know the state correctly
        ext_mask = _build_loss_mask(ext_chunk, initial_state=False)

        # Extract the actual chunk and its mask
        chunk_offset = LOOKBACK   # position of the actual chunk start in ext_chunk
        x_chunk = ext_chunk[chunk_offset : chunk_offset + context_window].astype(np.int64)
        y_chunk = ext_chunk[chunk_offset + 1 : chunk_offset + 1 + context_window].astype(np.int64)
        y_mask = ext_mask[chunk_offset + 1 : chunk_offset + 1 + context_window]

        # Apply mask: positions where y_mask == 0 → target -100 (ignored by loss)
        y_masked = np.where(y_mask == 1, y_chunk, -100)

        x_list.append(x_chunk)
        y_masked_list.append(y_masked)

    x = torch.from_numpy(np.stack(x_list))
    y = torch.from_numpy(np.stack(y_masked_list))

    if device.startswith("cuda"):
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)

    return x, y


if __name__ == "__main__":
    # Smoke test: sample a batch and verify the mask is working.
    # Should see most assistant content positions surviving, and most user
    # content positions set to -100.
    import sys

    x, y = get_batch("train", batch_size=2, context_window=256, device="cpu")
    print(f"\nInput shape:  {x.shape}    # should be (2, 256)")
    print(f"Target shape: {y.shape}    # should be (2, 256)")

    n_total = y.numel()
    n_masked = (y == -100).sum().item()
    n_trained = n_total - n_masked
    print(f"\nMask stats over batch:")
    print(f"  Total target positions: {n_total}")
    print(f"  Trained on (assistant): {n_trained} ({n_trained/n_total:.1%})")
    print(f"  Masked out (user/markers): {n_masked} ({n_masked/n_total:.1%})")
    print(f"\nExpected: roughly 30-60% of tokens trained on (assistant responses)")
    print(f"If this is way off (e.g., 0% or 100%), something's wrong with marker detection.")

    # Show a chunk of the actual target tokens and which ones survived masking
    print(f"\nFirst 30 positions of y[0]:")
    print(f"  Targets: {y[0, :30].tolist()}")
    print(f"  (-100 means masked out, otherwise the actual next-token target)")
