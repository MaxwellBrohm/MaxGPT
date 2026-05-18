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
    batch_size: int = 128                # number of sequences per step, was bumped from 32 to 128 after bf16+Flash freed VRAM
    learning_rate: float = 4e-4           # AdamW base learning rate, was 3e-4 but was bumped slightly higher for the larger batch
    weight_decay: float = 0.1             # AdamW weight decay
    beta1: float = 0.9                    # AdamW beta1
    beta2: float = 0.95                   # AdamW beta2 (modern LLM standard, not 0.999)
    grad_clip: float = 1.0                # max gradient norm before clipping
    
    # Schedule
    max_steps: int = 10000              # total training steps, was 2000 but was halved since each step now sees 4x more tokens
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

    # toggle for Flash Attention vs manual attention
    use_flash_attention: bool = True # switch to False to use the manual version (for presentations)