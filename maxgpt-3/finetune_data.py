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

# Encode role marker patterns. We use the "\n" prefix because EVERY role marker
# in our training data is preceded by a newline (either from joining turns with
# "\n" or from the document separator "\n\n###\n\n"). This makes detection
# unambiguous — bare "USER:" appearing in some assistant response content won't
# trigger a false role boundary.
ASSISTANT_MARKER = np.array(_tokenizer.encode("\nASSISTANT:"), dtype=np.uint16)
USER_MARKER = np.array(_tokenizer.encode("\nUSER:"), dtype=np.uint16)

# How far to look back in the token stream when sampling a chunk, so we know
# the role state at the chunk's start. 2000 tokens reliably catches at least
# one role marker for our average ~200-token turns.
LOOKBACK = 2000

print(f"[finetune_data] Loaded tokenizer (vocab {len(_tokenizer.vocab):,})")
print(f"[finetune_data] ASSISTANT marker = {len(ASSISTANT_MARKER)} tokens: {ASSISTANT_MARKER.tolist()}")
print(f"[finetune_data] USER marker = {len(USER_MARKER)} tokens: {USER_MARKER.tolist()}")
print(f"[finetune_data] Lookback for state detection: {LOOKBACK} tokens")


# ====================
# MASK BUILDING
# ====================

def _find_pattern_positions(tokens: np.ndarray, pattern: np.ndarray) -> np.ndarray:
    """Find all starting positions where `pattern` appears in `tokens`.
    Vectorized — much faster than a Python loop over positions."""
    n, m = len(tokens), len(pattern)
    if m == 0 or n < m:
        return np.array([], dtype=np.int64)
    matches = np.ones(n - m + 1, dtype=bool)
    for k in range(m):
        matches &= (tokens[k : n - m + 1 + k] == pattern[k])
    return np.where(matches)[0]


def _build_loss_mask(tokens: np.ndarray, initial_state: bool = False) -> np.ndarray:
    """Build a 0/1 mask the same length as `tokens`:
       1 = assistant content (train on this)
       0 = user content, role markers, or pre-first-role region

    Walk through pattern matches in order, toggling the assistant/user state at
    each role marker. Region BETWEEN markers gets the mask value of the current
    state. The marker tokens themselves are always 0 (we don't train on
    "ASSISTANT:" being predicted — at inference we prompt it, not generate it).
    """
    n = len(tokens)
    mask = np.zeros(n, dtype=np.uint8)

    # Find every position of each marker
    am_positions = _find_pattern_positions(tokens, ASSISTANT_MARKER)
    um_positions = _find_pattern_positions(tokens, USER_MARKER)

    # Build a sorted list of all boundary events:
    # (position, role_after_marker, marker_length)
    boundaries = []
    am_len = len(ASSISTANT_MARKER)
    um_len = len(USER_MARKER)
    for p in am_positions:
        boundaries.append((int(p), True, am_len))   # True = assistant
    for p in um_positions:
        boundaries.append((int(p), False, um_len))  # False = user
    boundaries.sort(key=lambda b: b[0])

    # Walk through boundaries, marking the regions between them
    in_assistant = initial_state
    cursor = 0
    for pos, role_after, marker_len in boundaries:
        # If the marker overlaps with an earlier marker we already passed, skip.
        # (Shouldn't happen in practice — role markers can't overlap — but
        # defensive coding.)
        if pos < cursor:
            continue

        # Mark region [cursor, pos) based on CURRENT state (before this marker)
        if in_assistant:
            mask[cursor:pos] = 1
        # The marker positions [pos, pos + marker_len) stay 0 (don't train on markers)

        # Update state to what's AFTER this marker
        in_assistant = role_after
        cursor = pos + marker_len

    # Mark the final region after the last boundary
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
