"""
Diagnostic: check which scaled_dot_product_attention backends are usable on this hardware.

Run from any directory with a working torch+CUDA install:
    python check_attention.py

Prints which of (FLASH_ATTENTION, EFFICIENT_ATTENTION, CUDNN_ATTENTION, MATH)
can actually run a small attention call on the current GPU.
"""

import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend


print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Compute capability: {torch.cuda.get_device_capability(0)}")
else:
    print("No CUDA — exiting.")
    raise SystemExit(0)

# Match our MaxGPT-2 attention shapes
batch_size = 4
num_heads = 12
seq_len = 512
head_dim = 64
dtype = torch.bfloat16   # we use bf16 in training

q = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda", dtype=dtype)
k = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda", dtype=dtype)
v = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda", dtype=dtype)

backends = [
    ("FLASH_ATTENTION", SDPBackend.FLASH_ATTENTION),
    ("EFFICIENT_ATTENTION", SDPBackend.EFFICIENT_ATTENTION),
    ("CUDNN_ATTENTION", SDPBackend.CUDNN_ATTENTION),
    ("MATH", SDPBackend.MATH),
]

print(f"\nTesting attention backends on shapes ({batch_size}, {num_heads}, {seq_len}, {head_dim}) in {dtype}:\n")

for name, backend in backends:
    try:
        with sdpa_kernel(backend):
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        torch.cuda.synchronize()
        print(f"  {name:25s} OK   (output shape: {tuple(out.shape)})")
    except Exception as e:
        msg = f"{type(e).__name__}: {str(e)[:150]}"
        print(f"  {name:25s} FAILED — {msg}")

print("\nDone.")
