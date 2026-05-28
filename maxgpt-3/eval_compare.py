"""
Multi-sample evaluation harness — compares models on the same prompt set
with both greedy (deterministic) and stochastic (temperature 0.8) sampling.

Designed to feed into parallel-agent rubric scoring downstream:
  - 1 greedy sample per (model, prompt) → deterministic comparison
  - 5 random samples per (model, prompt) → measures sampling variance

Output format is structured with explicit delimiters so it can be parsed
mechanically by scoring agents.

Run AFTER finetune.py finishes:
  python eval_compare.py
Or to test the current setup while training is still running:
  CUDA_VISIBLE_DEVICES="" python eval_compare.py    # CPU mode, doesn't fight training

Output: evaluation_results.txt
"""

import os
import time
import torch

from config import Config
from model import Transformer
from tokenizer import BPETokenizer
from sample import generate


# ====================
# CONFIG
# ====================

# Which checkpoints to evaluate. Comment out any that don't exist yet.
MODELS_TO_TEST = [
    ("base",     "final.pt"),                    # the pre-trained base
    # Add intermediate SFT checkpoints if you want trajectory data:
    # ("sft_75k",  "sft_checkpoint_75000.pt"),
    # ("sft_150k", "sft_checkpoint_150000.pt"),
    # ("sft_225k", "sft_checkpoint_225000.pt"),
    ("sft_final", "final_sft.pt"),               # the fully trained SFT
]

# Test prompts — same set as progression.txt so we have continuity.
# Mix of narrative completion + chat-format prompts.
PROMPTS = [
    "Once upon a time",
    "The little cat",
    "USER: Hi, how are you?\nASSISTANT:",
    "USER: Tell me a short story about a dog.\nASSISTANT:",
    "USER: What is your favorite color?\nASSISTANT:",
    "USER: Explain how photosynthesis works.\nASSISTANT:",
    "USER: Write a poem about the ocean.\nASSISTANT:",
]

# Sampling settings
N_RANDOM_SAMPLES = 5             # temp=0.8 samples per (model, prompt)
TEMPERATURE_RANDOM = 0.8
TOP_K = 50
MAX_NEW_TOKENS = 200             # bumped from 120 so poems/stories can end naturally
REPETITION_PENALTY = 1.2

OUTPUT_FILE = "evaluation_results.txt"


# ====================
# HELPERS
# ====================

def load_model(checkpoint_filename, config, device):
    """Build a fresh model and load weights from the given checkpoint filename."""
    model = Transformer(
        vocab_size=config.vocab_size,
        context_window=config.context_window,
        hidden_dim=config.hidden_dim,
        num_heads=config.num_heads,
        num_blocks=config.num_blocks,
        use_flash=config.use_flash_attention,
    )
    path = os.path.join(config.checkpoint_dir, checkpoint_filename)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state"]
    # Strip torch.compile's "_orig_mod." prefix if present (some older checkpoints
    # have it, current ones don't — this no-ops cleanly when absent)
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    step = checkpoint.get("step", None)
    return model, step


def sample_response(model, tokenizer, prompt, device, greedy=False):
    """Generate one response. Greedy mode uses top_k=1 (deterministic argmax)."""
    if greedy:
        return generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=1.0,          # doesn't matter with top_k=1
            top_k=1,                  # top_k=1 → only highest-logit token → deterministic
            device=device,
            repetition_penalty=REPETITION_PENALTY,
        )
    else:
        return generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE_RANDOM,
            top_k=TOP_K,
            device=device,
            repetition_penalty=REPETITION_PENALTY,
        )


def extract_response(full_text, prompt):
    """Strip the prompt from the generated text + cut at next USER: if present."""
    # Remove the prompt itself — generate() returns prompt + completion
    if full_text.startswith(prompt):
        generated = full_text[len(prompt):]
    else:
        generated = full_text

    # For chat-format prompts, cut at the next USER: marker (model self-stops)
    if "USER:" in generated:
        generated = generated.split("USER:")[0]

    return generated.rstrip()


# ====================
# MAIN
# ====================

def main():
    config = Config()

    # Device: respect CUDA_VISIBLE_DEVICES so user can force CPU during training
    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Device: {device}")

    # Load tokenizer once (same for all models)
    tokenizer = BPETokenizer()
    tokenizer.load(os.path.join(config.data_dir, "tokenizer.json"))
    print(f"Loaded tokenizer (vocab {len(tokenizer.vocab):,})")

    # Verify all checkpoints exist before starting
    missing = []
    for label, filename in MODELS_TO_TEST:
        path = os.path.join(config.checkpoint_dir, filename)
        if not os.path.exists(path):
            missing.append((label, path))
    if missing:
        print("\nERROR: Missing checkpoints:")
        for label, path in missing:
            print(f"  {label}: {path}")
        print("\nEither generate them first or comment them out of MODELS_TO_TEST.")
        return

    n_per_model = len(PROMPTS) * (1 + N_RANDOM_SAMPLES)
    n_total = n_per_model * len(MODELS_TO_TEST)
    print(f"\nEvaluation plan:")
    print(f"  Models:  {len(MODELS_TO_TEST)} ({', '.join(m[0] for m in MODELS_TO_TEST)})")
    print(f"  Prompts: {len(PROMPTS)}")
    print(f"  Samples per (model, prompt): 1 greedy + {N_RANDOM_SAMPLES} random = {1 + N_RANDOM_SAMPLES}")
    print(f"  Total generations: {n_total} ({n_per_model} per model)")
    print()

    start_time = time.time()

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        # Header so the scoring agent has context
        out.write("=" * 75 + "\n")
        out.write("MaxGPT-3 — Multi-sample evaluation\n")
        out.write("=" * 75 + "\n")
        out.write(f"Models:  {', '.join(m[0] for m in MODELS_TO_TEST)}\n")
        out.write(f"Prompts: {len(PROMPTS)}\n")
        out.write(f"Samples per (model, prompt): 1 greedy (deterministic) + "
                  f"{N_RANDOM_SAMPLES} random (temp={TEMPERATURE_RANDOM}, top_k={TOP_K})\n")
        out.write(f"Max new tokens: {MAX_NEW_TOKENS}, repetition_penalty: {REPETITION_PENALTY}\n")
        out.write("\n")
        out.write("Format note: each generation is wrapped in ===MODEL===, ===PROMPT===,\n")
        out.write("===SAMPLE===, and ===END=== delimiters so it can be parsed mechanically.\n")
        out.write("=" * 75 + "\n\n")

        gen_num = 0
        for model_label, checkpoint_filename in MODELS_TO_TEST:
            print(f"\n=== Loading {model_label} ({checkpoint_filename}) ===")
            model, step = load_model(checkpoint_filename, config, device)
            print(f"  Loaded (training step: {step})")

            for prompt_idx, prompt in enumerate(PROMPTS, 1):
                # Generate 1 greedy + N random samples
                samples = [("greedy", sample_response(model, tokenizer, prompt, device, greedy=True))]
                for i in range(N_RANDOM_SAMPLES):
                    samples.append(
                        (f"random_{i+1}", sample_response(model, tokenizer, prompt, device, greedy=False))
                    )

                # Write all samples for this (model, prompt) pair
                for sample_label, full_text in samples:
                    gen_num += 1
                    response = extract_response(full_text, prompt)

                    out.write("===MODEL=== " + model_label + "\n")
                    out.write("===PROMPT=== " + repr(prompt) + "\n")
                    out.write("===SAMPLE=== " + sample_label + "\n")
                    out.write(response + "\n")
                    out.write("===END===\n\n")
                    out.flush()

                elapsed = time.time() - start_time
                progress = gen_num / n_total
                eta = (elapsed / progress - elapsed) if progress > 0 else 0
                print(f"  [{gen_num:3d}/{n_total}] {model_label} prompt {prompt_idx}/{len(PROMPTS)} "
                      f"({progress*100:.0f}%, ETA {eta:.0f}s)")

            # Free this model's GPU memory before loading the next one
            del model
            if device == "cuda":
                torch.cuda.empty_cache()

        total_elapsed = time.time() - start_time
        out.write("\n" + "=" * 75 + "\n")
        out.write(f"Complete. {gen_num} generations in {total_elapsed:.0f}s.\n")
        out.write("=" * 75 + "\n")

    print(f"\nDone! {gen_num} generations in {total_elapsed:.0f}s.")
    print(f"Output saved to {OUTPUT_FILE}")
    print(f"\nReady for parallel-agent scoring against the 4-axis rubric.")


if __name__ == "__main__":
    main()
