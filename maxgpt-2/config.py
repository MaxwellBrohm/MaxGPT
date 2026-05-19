from dataclasses import dataclass

@dataclass
class Config:
    # Architecture (MaxGPT-2 — ~110M params, ~4.5x bigger than MaxGPT-1)
    # Scaled hidden_dim, blocks, heads, AND context window. Same vocab so the
    # two models can be compared on identical tokenization.
    vocab_size: int = 16000
    context_window: int = 512        # was 256 — longer context for better multi-turn coherence
    hidden_dim: int = 768            # was 384 — 2x wider
    num_heads: int = 12              # was 6 — each head still has dim 64 (768/12)
    num_blocks: int = 12             # was 6 — 2x deeper

    # Training hyperparameters
    batch_size: int = 32             # MaxGPT-1 used 128. Dropped to 32 because of MaxGPT-2's
                                     # 2x longer context + 4.5x bigger model + 16K-vocab logits
                                     # tensor (1GB+ at batch=64). With EFFICIENT_ATTENTION + bf16,
                                     # batch=32 fits in 12GB VRAM with margin. Tokens-per-step is
                                     # 16K (32 * 512), half of MaxGPT-1's, so we 2x max_steps to
                                     # preserve total compute.
    learning_rate: float = 3e-4      # was 4e-4 — more conservative for larger model (Karpathy constant)
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95              # modern LLM standard (not the PyTorch default 0.999)
    grad_clip: float = 1.0

    # Schedule
    max_steps: int = 60000           # 2x more steps than 30K because batch_size dropped from 64→32.
                                     # At 16K tokens/step × 60K steps = 983M token-passes, which is
                                     # ~2 epochs over the 482M-token dataset. Still under full
                                     # Chinchilla (2B for a 100M model) but reusing MaxGPT-1's data
                                     # caps us here — MaxGPT-3 will get more data + push past
                                     # Chinchilla on a bigger model.
    warmup_steps: int = 500          # was 200 — longer warmup helps larger model stabilize early
    eval_interval: int = 500
    eval_iters: int = 50
    log_interval: int = 50

    # Checkpointing
    checkpoint_interval: int = 3000  # was 2000 — fewer total checkpoints since training is longer
    checkpoint_dir: str = "checkpoints"

    # Data
    data_dir: str = "data"

    # Device — auto-detect, but allow override (set by train.py)
    device: str = "auto"

    # toggle for Flash Attention vs manual attention
    # switch to False to show the manual version during presentations
    use_flash_attention: bool = True
