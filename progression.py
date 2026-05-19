"""
Run a fixed set of test prompts through every saved checkpoint to see how
MaxGPT evolves over training. The output (printed and saved to progression.txt)
shows side-by-side what the model produces at each training step — perfect
"evolution from gibberish to coherent" material for the writeup.

Usage:
    python progression.py

Reads from:
    checkpoints/checkpoint_*.pt  (and checkpoints/final.pt if present)
    data/tokenizer.json

Writes to:
    progression.txt
"""

import os
import re
import torch

from config import Config
from model import Transformer
from tokenizer import BPETokenizer
from sample import generate   # reuse the same generate function as the chat script


# === FIXED TEST PROMPTS ===
# Run the SAME prompts through every checkpoint so we can compare apples-to-apples.
# Mix of narrative prompts and chat prompts to test both modes.
TEST_PROMPTS = [
    "Once upon a time",
    "The little cat",
    "USER: Hi, how are you?\nASSISTANT:",
    "USER: Tell me a short story about a dog.\nASSISTANT:",
    "USER: What is your favorite color?\nASSISTANT:",
    "USER: Explain how photosynthesis works.\nASSISTANT:",
    "USER: Write a poem about the ocean.\nASSISTANT:",
]

# Sampling settings — fixed across all checkpoints for fair comparison
MAX_NEW_TOKENS = 120
TEMPERATURE = 0.8
TOP_K = 50

OUTPUT_FILE = "progression.txt"


def list_checkpoints(checkpoint_dir):
    """
    Find all checkpoint files in the directory, sorted by training step.
    `final.pt` (if present) is sorted last regardless of name.
    Returns a list of (step, filename) tuples.
    """
    if not os.path.exists(checkpoint_dir):
        return []

    checkpoints = []
    for f in os.listdir(checkpoint_dir):
        if not f.endswith(".pt"):
            continue
        match = re.match(r"checkpoint_(\d+)\.pt", f)
        if match:
            checkpoints.append((int(match.group(1)), f))
        elif f == "final.pt":
            # Sort final.pt last by giving it a sentinel "infinity" step
            checkpoints.append((float("inf"), f))

    checkpoints.sort(key=lambda x: x[0])
    return checkpoints


def load_model_from_checkpoint(checkpoint_path, config, device):
    """Build a fresh model and load its weights from the given checkpoint."""
    model = Transformer(
        vocab_size=config.vocab_size,
        context_window=config.context_window,
        hidden_dim=config.hidden_dim,
        num_heads=config.num_heads,
        num_blocks=config.num_blocks,
        use_flash=config.use_flash_attention,
    )
    # weights_only=False because our checkpoints contain the Config dataclass
    # (not just tensor weights). Safe because we created these files ourselves.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)
    model.eval()
    # checkpoint["step"] tells us where in training this snapshot was saved
    step = checkpoint.get("step", None)
    return model, step


def format_label(step, filename):
    if step is None:
        return filename
    if filename == "final.pt":
        return f"Step {step} (final)"
    return f"Step {step}"


def main():
    config = Config()
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    # Load tokenizer once — it's the same for every checkpoint
    tokenizer_path = os.path.join(config.data_dir, "tokenizer.json")
    if not os.path.exists(tokenizer_path):
        print(f"ERROR: No tokenizer found at {tokenizer_path}")
        print("Run prepare_data.py first.")
        return

    tokenizer = BPETokenizer()
    tokenizer.load(tokenizer_path)
    print(f"Loaded tokenizer (vocab size: {len(tokenizer.vocab):,})")

    # Find all checkpoints
    checkpoints = list_checkpoints(config.checkpoint_dir)
    if not checkpoints:
        print(f"ERROR: No checkpoints found in {config.checkpoint_dir}")
        print("Train the model first with: python train.py")
        return

    print(f"Found {len(checkpoints)} checkpoint(s) to process\n")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        # Write a header to the output file
        header = "=" * 70 + "\n"
        header += "MaxGPT-1 — Training progression samples\n"
        header += f"Settings: max_new_tokens={MAX_NEW_TOKENS}, "
        header += f"temperature={TEMPERATURE}, top_k={TOP_K}\n"
        header += "=" * 70 + "\n"
        print(header)
        out.write(header)

        for sort_key, filename in checkpoints:
            checkpoint_path = os.path.join(config.checkpoint_dir, filename)
            print(f"Loading {filename}...")

            model, step = load_model_from_checkpoint(checkpoint_path, config, device)
            label = format_label(step, filename)

            section_header = f"\n\n{'=' * 70}\n{label}\n{'=' * 70}\n"
            print(section_header)
            out.write(section_header)

            for prompt in TEST_PROMPTS:
                with torch.no_grad():
                    output = generate(
                        model=model,
                        tokenizer=tokenizer,
                        prompt=prompt,
                        max_new_tokens=MAX_NEW_TOKENS,
                        temperature=TEMPERATURE,
                        top_k=TOP_K,
                        device=device,
                    )

                block = (
                    f"\n--- Prompt: {prompt!r} ---\n"
                    f"{output}\n"
                )
                print(block)
                out.write(block)

            # Free GPU memory before loading the next checkpoint
            del model
            if device == "cuda":
                torch.cuda.empty_cache()

        footer = f"\n\n{'=' * 70}\nDone. Output saved to {OUTPUT_FILE}\n"
        print(footer)
        out.write(footer)


if __name__ == "__main__":
    main()
