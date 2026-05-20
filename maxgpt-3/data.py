import numpy as np
import torch

# Module-level: memmap the binary files once when the module is imported.
# (Cheap operation — no actual reading happens until we slice.)
train_data = np.memmap("data/train.bin", dtype=np.uint16, mode="r")
val_data = np.memmap("data/val.bin", dtype=np.uint16, mode="r")

def get_batch(split, batch_size, context_window, device):
    """
    Returns (inputs, targets) tensors, each of shape (batch_size, context_window).
    
    split: "train" or "val"
    batch_size: how many independent sequences per batch
    context_window: T, the sequence length
    device: where to put the tensors ("cuda", "mps", or "cpu")
    """
    
    # Pick which split we're sampling from
    data = train_data if split == "train" else val_data
    
    # Sample batch_size random starting indices.
    # The valid range is [0, len(data) - context_window - 1]
    # (the -1 is because we need T+1 tokens for input+target)
    ix = torch.randint(
        low=0,
        high=len(data) - context_window - 1,
        size=(batch_size,)
    )
    
    # Build input and target tensors by slicing the memmap at each random index.
    # Important: cast to int64 because PyTorch embedding layers need int64 indices,
    # but we stored data as uint16 to save disk space.
    x = torch.stack([
        torch.from_numpy(data[i : i + context_window].astype(np.int64))
        for i in ix
    ])
    y = torch.stack([
        torch.from_numpy(data[i + 1 : i + 1 + context_window].astype(np.int64))
        for i in ix
    ])
    
    # Move to the right device (CPU/GPU/MPS)
    # MaxGPT-3 optimization: pin memory + non_blocking transfers. Pinned ("page-locked")
    # CPU memory lets the GPU copy DMA directly, and non_blocking=True lets the CPU
    # keep doing things while the transfer happens. Small per-step speedup (~1-3%)
    # that adds up over millions of steps.
    if device.startswith("cuda"):
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)

    return x, y

if __name__ == "__main__":
    x, y = get_batch("train", batch_size=4, context_window=256, device="cpu")
    print(f"Input shape:  {x.shape}    # should be (4, 256)")
    print(f"Target shape: {y.shape}    # should be (4, 256)")
    print(f"Input dtype:  {x.dtype}    # should be torch.int64")
    print(f"First input row [first 10]:  {x[0, :10].tolist()}")
    print(f"First target row [first 10]: {y[0, :10].tolist()}")
    print(f"Notice target row is input row shifted by one position")