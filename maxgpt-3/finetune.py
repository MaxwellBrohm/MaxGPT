"""
SFT (Supervised Fine-Tuning) phase for MaxGPT-3.

Loads the pre-trained checkpoint (final.pt) and continues training on chat-only
data with a much lower learning rate. This concentrates gradient updates on
conversational patterns and instruction-following, which typically gives the
biggest visible quality bump for a chatbot demo.

Modern small-LM pipeline:
    pre-train (raw text + chat mixed) → SFT (chat-only, low LR)

Run with:
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    python -u finetune.py 2>&1 | tee finetune_run.log

Expected runtime: ~18-24 hours for ~1 epoch over the chat data.
Output: checkpoints/final_sft.pt (preserves base model at final.pt).
"""

import os, math, time
import torch
from config import Config
from model import Transformer
# IMPORTANT: imports from finetune_data, not data — uses chat_train.bin / chat_val.bin
from finetune_data import get_batch

# Attention backend forced inside model.py via sdpa_kernel context manager.


# ====================
# SFT-SPECIFIC HYPERPARAMETERS
# ====================
# These OVERRIDE the values in Config, because SFT wants very different settings
# than pre-training:
#   - LR ~10-40x smaller (model is already trained, smaller updates)
#   - Short warmup or none (model isn't starting cold)
#   - Fewer steps (1 epoch over the chat data is plenty)
#   - Different checkpoint output (don't overwrite the base model)

SFT_LEARNING_RATE = 5e-5          # was 2e-4 in pre-training — ~4x smaller
SFT_WARMUP_STEPS = 100            # was 2000 — much shorter, model is already warm
SFT_MAX_STEPS = 300_000           # ~1 epoch over the chat data (~2.4B tokens at 8192/step)
SFT_CHECKPOINT_INTERVAL = 25_000  # ~12 SFT checkpoints
SFT_LOG_INTERVAL = 500
SFT_EVAL_INTERVAL = 2_500
SFT_EVAL_ITERS = 25
SFT_OUTPUT_FILE = "final_sft.pt"
BASE_CHECKPOINT_FILE = "final.pt"


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def get_sft_lr(step):
    """SFT LR schedule: short linear warmup, then cosine decay to 10% of peak."""
    if step < SFT_WARMUP_STEPS:
        return SFT_LEARNING_RATE * (step + 1) / SFT_WARMUP_STEPS
    progress = (step - SFT_WARMUP_STEPS) / (SFT_MAX_STEPS - SFT_WARMUP_STEPS)
    progress = min(progress, 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return SFT_LEARNING_RATE * (0.1 + 0.9 * coeff)


@torch.no_grad()
def estimate_loss(model, config, device):
    """Eval helper — averages loss over SFT_EVAL_ITERS batches of chat data."""
    model.eval()
    losses = {}
    for split in ["train", "val"]:
        batch_losses = []
        for _ in range(SFT_EVAL_ITERS):
            x, y = get_batch(split, config.batch_size, config.context_window, device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                _, loss = model(x, y)
            batch_losses.append(loss.item())
        losses[split] = sum(batch_losses) / len(batch_losses)
    model.train()
    return losses


def main():
    config = Config()

    # Device setup
    device = get_device() if config.device == "auto" else config.device
    print(f"Using device: {device}")

    # Build model (same architecture as pre-training)
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

    # === LOAD PRE-TRAINED CHECKPOINT (the only big difference vs train.py) ===
    base_ckpt_path = os.path.join(config.checkpoint_dir, BASE_CHECKPOINT_FILE)
    if not os.path.exists(base_ckpt_path):
        raise FileNotFoundError(
            f"No base checkpoint at {base_ckpt_path}. Run train.py first to "
            "produce a pre-trained model, then run this script to fine-tune it."
        )
    print(f"Loading pre-trained checkpoint from {base_ckpt_path}...")
    checkpoint = torch.load(base_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    base_step = checkpoint.get("step", 0)
    print(f"  Starting SFT from pre-training step {base_step:,}")

    # Compile after loading weights (compile + load_state_dict order matters)
    if config.use_torch_compile and device.startswith("cuda"):
        try:
            print("Compiling model with torch.compile (first step will be slow)...")
            model = torch.compile(model)
        except Exception as e:
            print(f"  torch.compile failed, falling back to eager mode: {e}")

    # Fresh AdamW (don't carry over optimizer state from pre-training — SFT wants
    # to forget the pre-train momentum and start fresh with much smaller LR)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=SFT_LEARNING_RATE,
        weight_decay=config.weight_decay,
        betas=(config.beta1, config.beta2),
        fused=(device.startswith("cuda")),
    )

    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # SFT training loop
    print(f"\nStarting SFT — {SFT_MAX_STEPS:,} steps at LR {SFT_LEARNING_RATE} on chat data...")
    start_time = time.time()

    for step in range(SFT_MAX_STEPS):
        lr = get_sft_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Sample from chat data (not the full pre-training mix)
        x, y = get_batch("train", config.batch_size, config.context_window, device)

        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        if step % SFT_LOG_INTERVAL == 0:
            elapsed = time.time() - start_time
            print(f"SFT step {step:6d} | loss {loss.item():.4f} | lr {lr:.2e} | elapsed {elapsed:.0f}s")

        if step % SFT_EVAL_INTERVAL == 0 and step > 0:
            losses = estimate_loss(model, config, device)
            print(f"  >> EVAL SFT step {step}: train={losses['train']:.4f}, val={losses['val']:.4f}")

        if step % SFT_CHECKPOINT_INTERVAL == 0 and step > 0:
            ckpt = {
                "model_state": (model._orig_mod.state_dict() if hasattr(model, "_orig_mod") else model.state_dict()),
                "optimizer_state": optimizer.state_dict(),
                "step": step,
                "base_pretrain_step": base_step,
                "config": config,
                "phase": "sft",
            }
            path = os.path.join(config.checkpoint_dir, f"sft_checkpoint_{step}.pt")
            torch.save(ckpt, path)
            print(f"  >> Saved SFT checkpoint: {path}")

    # Final SFT checkpoint
    final_ckpt = {
        "model_state": (model._orig_mod.state_dict() if hasattr(model, "_orig_mod") else model.state_dict()),
        "optimizer_state": optimizer.state_dict(),
        "step": SFT_MAX_STEPS,
        "base_pretrain_step": base_step,
        "config": config,
        "phase": "sft",
    }
    final_path = os.path.join(config.checkpoint_dir, SFT_OUTPUT_FILE)
    torch.save(final_ckpt, final_path)
    print(f"\nSFT complete! Final checkpoint: {final_path}")
    print(f"(Base pre-trained model still at {base_ckpt_path})")


if __name__ == "__main__":
    main()
