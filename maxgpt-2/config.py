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
    batch_size: int = 16             # MaxGPT-1 used 128. Dropped to 16 (was 32) after training
                                     # crashed at step 5000 from VRAM spillover into shared memory.
                                     # Even at batch=32 the margin was too thin — Parsec or any
                                     # other GPU activity would push past 12GB. At batch=16 the
                                     # peak usage should be ~6-8GB with a comfortable safety
                                     # buffer. Tokens-per-step = 16*512 = 8K, so we 2x max_steps
                                     # again to preserve total compute.
    learning_rate: float = 3e-4      # was 4e-4 — more conservative for larger model (Karpathy constant)
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95              # modern LLM standard (not the PyTorch default 0.999)
    grad_clip: float = 1.0

    # Schedule
    max_steps: int = 244000          # Targets full ~2B Chinchilla token-passes at batch=16, ctx=512
                                     # (244K * 8K tokens/step = 2.0B passes). This is ~4 epochs over
                                     # the 482M-token dataset — fine for small models (Phi/TinyLlama
                                     # train many epochs past Chinchilla on smaller data routinely).
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
