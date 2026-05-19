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
    batch_size: int = 64             # was 128 — smaller because each step processes 2x longer
                                     # sequences and the model is 4.5x bigger. Tokens-per-step is
                                     # still 32K (64 * 512), same as MaxGPT-1's 128 * 256
    learning_rate: float = 3e-4      # was 4e-4 — more conservative for larger model (Karpathy constant)
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95              # modern LLM standard (not the PyTorch default 0.999)
    grad_clip: float = 1.0

    # Schedule
    max_steps: int = 30000           # was 15000 — 2x more steps. At 32K tokens/step that's ~960M
                                     # token-passes, which is roughly 2 epochs over the 482M-token
                                     # dataset. Slight overshoot of Chinchilla compute (460M optimal
                                     # for a 23M model, ~2B for 100M) — but we reuse MaxGPT-1's data
                                     # rather than re-prepping, so we accept being slightly data-bound.
                                     # MaxGPT-3 will get more data + push past Chinchilla.
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
