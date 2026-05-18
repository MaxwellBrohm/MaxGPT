from dataclasses import dataclass

@dataclass
class Config:
    # Architecture (MaxGPT-1)
    vocab_size: int = 16000
    context_window: int = 256
    hidden_dim: int = 384
    num_heads: int = 6
    num_blocks: int = 6
    
    # Training hyperparameters
    batch_size: int = 32                # number of sequences per step
    learning_rate: float = 3e-4           # AdamW base learning rate
    weight_decay: float = 0.1             # AdamW weight decay
    beta1: float = 0.9                    # AdamW beta1
    beta2: float = 0.95                   # AdamW beta2 (modern LLM standard, not 0.999)
    grad_clip: float = 1.0                # max gradient norm before clipping
    
    # Schedule
    max_steps: int = 20000              # total training steps
    warmup_steps: int = 200             # linear warmup at start
    eval_interval: int = 500            # how often to compute val loss
    eval_iters: int = 50                # how many batches to average for val loss
    log_interval: int = 50              # how often to print training loss
    
    # Checkpointing
    checkpoint_interval: int = 2000     # save every N steps
    checkpoint_dir: str = "checkpoints"
    
    # Data
    data_dir: str = "data"
    
    # Device — auto-detect, but allow override
    # Will be set by train.py based on what's available
    device: str = "auto"