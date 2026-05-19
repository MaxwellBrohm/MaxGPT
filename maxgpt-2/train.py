import os, math, time
import torch
from config import Config
from model import Transformer
from data import get_batch

# Attention backend is forced inside model.py via sdpa_kernel context manager.
# (Our PyTorch wheel doesn't include Flash Attention on Blackwell, and auto-selection
# would silently fall back to the slow MATH backend. EFFICIENT_ATTENTION gives Flash-
# like O(T) memory and runs reliably.)

def get_device():
    """Pick the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def get_lr(step, config):
    """
    Learning rate schedule:
    - Linear warmup from 0 to peak over warmup_steps
    - Then cosine decay from peak down to 10% of peak over remaining steps
    
    Why: linear warmup prevents early instability when gradients are wild.
    Cosine decay gradually reduces LR so the model can settle into a minimum.
    """
    # Warmup phase
    if step < config.warmup_steps:
        return config.learning_rate * (step + 1) / config.warmup_steps
    
    # Cosine decay phase
    progress = (step - config.warmup_steps) / (config.max_steps - config.warmup_steps)
    progress = min(progress, 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))   # 1.0 → 0.0
    return config.learning_rate * (0.1 + 0.9 * coeff)      # peak_lr → 10% of peak

@torch.no_grad()    # decorator: disables gradient tracking, saves memory + speed
def estimate_loss(model, config, device):
    """
    Eval helper: estimates loss on train and val by averaging multiple batches.
    More accurate than just looking at single-batch training loss.
    """
    model.eval()   # disables dropout etc. (we don't have any, but good practice)
    losses = {}
    for split in ["train", "val"]:
        batch_losses = []
        for _ in range(config.eval_iters):
            x, y = get_batch(split, config.batch_size, config.context_window, device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                _, loss = model(x, y)
            batch_losses.append(loss.item())
        losses[split] = sum(batch_losses) / len(batch_losses)
    model.train()  # back to training mode
    return losses

def main():
    config = Config()
    
    # Device setup
    device = get_device() if config.device == "auto" else config.device
    print(f"Using device: {device}")
    
    # Build model
    model = Transformer(
        vocab_size=config.vocab_size,
        context_window=config.context_window,
        hidden_dim=config.hidden_dim,
        num_heads=config.num_heads,
        num_blocks=config.num_blocks,
        use_flash=config.use_flash_attention,
    )
    model = model.to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    
    # AdamW optimizer
    # Note the `betas` is a TUPLE of (beta1, beta2)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,      # will be overridden each step by scheduler
        weight_decay=config.weight_decay,
        betas=(config.beta1, config.beta2),
    )
    
    # Make sure checkpoint dir exists
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    # Training loop
    print("Starting training...")
    start_time = time.time()
    
    for step in range(config.max_steps):
        # Set learning rate for this step
        lr = get_lr(step, config)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        
        # Sample a batch
        x, y = get_batch("train", config.batch_size, config.context_window, device)
        
        # Forward pass with bf16 mixed precision
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            _, loss = model(x, y)
        
        # Backward pass
        # set_to_none=True is faster than the default (zero_grad sets to None instead of 0 tensors)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        # Gradient clipping (BEFORE optimizer step)
        # Clips the total gradient norm to max grad_clip
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        
        # Optimizer step
        optimizer.step()
        
        # Logging training loss
        if step % config.log_interval == 0:
            elapsed = time.time() - start_time
            print(f"step {step:5d} | loss {loss.item():.4f} | lr {lr:.2e} | elapsed {elapsed:.0f}s")
        
        # Periodic eval (more accurate than single-batch loss)
        if step % config.eval_interval == 0 and step > 0:
            losses = estimate_loss(model, config, device)
            print(f"  >> EVAL step {step}: train={losses['train']:.4f}, val={losses['val']:.4f}")
        
        # Periodic checkpoint
        if step % config.checkpoint_interval == 0 and step > 0:
            checkpoint = {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "step": step,
                "config": config,
            }
            path = os.path.join(config.checkpoint_dir, f"checkpoint_{step}.pt")
            torch.save(checkpoint, path)
            print(f"  >> Saved checkpoint: {path}")
    
    # Final checkpoint
    final_checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "step": config.max_steps,
        "config": config,
    }
    torch.save(final_checkpoint, os.path.join(config.checkpoint_dir, "final.pt"))
    print(f"Training complete! Final checkpoint: {config.checkpoint_dir}/final.pt")

if __name__ == "__main__":
    main()