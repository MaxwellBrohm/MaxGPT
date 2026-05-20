from dataclasses import dataclass

@dataclass
class Config:
    # Architecture (MaxGPT-3 — ~235M params, ~10x bigger than MaxGPT-1's 23M)
    # Clean scaling: 23M (MaxGPT-1) → 110M (MaxGPT-2) → 235M (MaxGPT-3)
    # Same vocab + context family as MaxGPT-2 so all three models can share the
    # same evaluation prompts and tokenizer-style comparisons.
    vocab_size: int = 16000
    context_window: int = 512        # same as MaxGPT-2
    hidden_dim: int = 1024           # was 768 in MaxGPT-2 (+33% wider)
    num_heads: int = 16              # was 12 — each head still has dim 64 (1024/16)
    num_blocks: int = 16             # was 12 — 33% deeper

    # Training hyperparameters
    batch_size: int = 16             # SMOKE TEST FIRST! If VRAM stays under ~10GB, batch=16 is fine.
                                     # If it spills past 11GB or hits shared memory, drop to 8.
                                     # 235M is ~2x MaxGPT-2's static state, so memory will be tighter
                                     # even with EFFICIENT_ATTENTION. May need to fall back to 8.
    learning_rate: float = 2e-4      # was 3e-4 in MaxGPT-2 — slightly smaller for the bigger model
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # Schedule — targets ~20B token-passes (~4x Chinchilla for 235M)
    # 2.44M steps × 8192 tokens/step (16 batch * 512 ctx) = 20B passes
    # ~4 epochs over the planned ~5B-token data mix
    # Time estimate: ~5-7 days depending on whether torch.compile / WSL+Flash work out
    max_steps: int = 2_440_000
    warmup_steps: int = 2000         # longer warmup for the bigger model (was 500 in MaxGPT-2)
    eval_interval: int = 5000        # less frequent for the long run (was 500)
    eval_iters: int = 25             # was 50 — halved to reduce eval overhead
    log_interval: int = 500          # was 50 — less frequent to keep log file manageable

    # Checkpointing — fewer checkpoints since training is super long
    checkpoint_interval: int = 50000 # ~49 total checkpoints across the 2.44M-step run
    checkpoint_dir: str = "checkpoints"

    # Data
    data_dir: str = "data"

    # Device — auto-detect, but allow override (set by train.py)
    device: str = "auto"

    # Attention backend toggle (False = manual computation, for the presentation slides)
    use_flash_attention: bool = True

    # torch.compile — JITs the model into fused kernels for ~20-40% speedup.
    # First iteration is slow (compilation, ~30-90 sec). Falls back to eager if there's
    # any issue (some sdpa_kernel + autocast combos confuse the compiler on Blackwell).
    # Over 2.44M steps the compilation overhead is nothing and the per-step speedup
    # saves 1-2 days of total training time. Recommend leaving on.
    use_torch_compile: bool = True
