"""
Multi-turn coherence evaluation — tests whether a model USES context from
earlier turns when answering a later turn.

Each test is a SCRIPTED conversation: the prior turns (both USER and ASSISTANT)
are fixed/canned so every model sees the identical history. Only the FINAL
assistant turn is generated. The final user question is designed so that
answering it correctly REQUIRES information from an earlier turn — this
isolates context-tracking ("does it remember/use what was said before?").

Categories tested:
  - memory:      recall a fact stated earlier ("What's my favorite animal?")
  - reference:   resolve a pronoun to an earlier entity ("What should I feed him?")
  - continuity:  follow up on the same topic ("Can you explain that more simply?")
  - adaptation:  modify a previous output ("Now make it about summer instead")
  - correction:  incorporate a user correction ("Actually I meant Italy")

Run AFTER finetune.py finishes (needs final_sft.pt):
  python eval_multiturn.py
Or force CPU if the GPU is busy:
  CUDA_VISIBLE_DEVICES="" python eval_multiturn.py

Output: multiturn_results.txt  (delimiter-wrapped for parallel-agent scoring)
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

# Two-way comparison by default: pre-trained base vs fully fine-tuned SFT.
# Uncomment intermediates if you want the trajectory.
MODELS_TO_TEST = [
    ("base",      "final.pt"),
    # ("sft_75k",  "sft_checkpoint_75000.pt"),
    # ("sft_150k", "sft_checkpoint_150000.pt"),
    # ("sft_225k", "sft_checkpoint_225000.pt"),
    ("sft_final", "final_sft.pt"),
]

# Scripted multi-turn conversations. Each `history` ends with "ASSISTANT:" so
# the model generates ONLY the final assistant turn. `key` tells the scorer
# what counts as correctly using the prior context.
CONVERSATIONS = [
    {
        "id": "memory_animal", "category": "memory",
        "history": "USER: My favorite animal is the dolphin.\nASSISTANT: Dolphins are wonderful — intelligent, social, and playful creatures!\nUSER: What's my favorite animal?\nASSISTANT:",
        "key": "Must answer 'dolphin' (stated in turn 1).",
    },
    {
        "id": "memory_name", "category": "memory",
        "history": "USER: Hi! My name is Max.\nASSISTANT: Nice to meet you, Max! How can I help you today?\nUSER: What is my name?\nASSISTANT:",
        "key": "Must answer 'Max'.",
    },
    {
        "id": "memory_city", "category": "memory",
        "history": "USER: I live in Seattle.\nASSISTANT: Seattle is a beautiful city — known for its coffee and rainy weather!\nUSER: Which city did I say I live in?\nASSISTANT:",
        "key": "Must answer 'Seattle'.",
    },
    {
        "id": "memory_two_hobbies", "category": "memory",
        "history": "USER: I love hiking on weekends.\nASSISTANT: Hiking is a great way to stay active and enjoy nature!\nUSER: I also enjoy painting.\nASSISTANT: Painting is a wonderful creative outlet.\nUSER: What two hobbies did I mention?\nASSISTANT:",
        "key": "Must recall BOTH hiking and painting (3-turn memory).",
    },
    {
        "id": "reference_puppy", "category": "reference",
        "history": "USER: I just adopted a puppy named Rex.\nASSISTANT: Congratulations on adopting Rex! Puppies bring so much joy.\nUSER: What should I feed him?\nASSISTANT:",
        "key": "'him' = Rex/puppy; should give dog-feeding advice.",
    },
    {
        "id": "reference_guitar", "category": "reference",
        "history": "USER: I'm learning to play the guitar.\nASSISTANT: That's awesome! The guitar is a rewarding instrument to learn.\nUSER: How long does it usually take to get good at it?\nASSISTANT:",
        "key": "'it' = guitar; should answer about guitar skill timeline.",
    },
    {
        "id": "reference_cookies", "category": "reference",
        "history": "USER: I'm making chocolate chip cookies.\nASSISTANT: Yum! Chocolate chip cookies are a classic. Need any tips?\nUSER: Yes — how long should I bake them?\nASSISTANT:",
        "key": "'them' = cookies; should give a cookie baking time.",
    },
    {
        "id": "continuity_rainbow", "category": "continuity",
        "history": "USER: Explain how rainbows form.\nASSISTANT: Rainbows form when sunlight passes through raindrops, which bend and split the light into its colors.\nUSER: Can you explain that more simply?\nASSISTANT:",
        "key": "Should simplify the RAINBOW explanation, not switch topics.",
    },
    {
        "id": "continuity_mars", "category": "continuity",
        "history": "USER: Tell me about the planet Mars.\nASSISTANT: Mars is the fourth planet from the Sun, often called the Red Planet for its iron-rich soil.\nUSER: Is it bigger or smaller than Earth?\nASSISTANT:",
        "key": "'it' = Mars; correct = smaller than Earth (and stays on Mars topic).",
    },
    {
        "id": "adapt_poem", "category": "adaptation",
        "history": "USER: Write a one-line poem about winter.\nASSISTANT: Snow blankets the silent world in white.\nUSER: Now write one about summer instead.\nASSISTANT:",
        "key": "Should write a SUMMER line (adapting from the winter request).",
    },
    {
        "id": "adapt_catname", "category": "adaptation",
        "history": "USER: Suggest a name for a black cat.\nASSISTANT: How about Shadow? It suits a sleek black cat perfectly.\nUSER: What about for a white one?\nASSISTANT:",
        "key": "Should suggest a name fitting a WHITE cat (e.g. Snow/Cloud).",
    },
    {
        "id": "correction_capital", "category": "correction",
        "history": "USER: What's the capital of France?\nASSISTANT: The capital of France is Paris.\nUSER: Sorry, I actually meant Italy.\nASSISTANT:",
        "key": "Should now answer Rome (capital of Italy).",
    },
]

# Sampling
N_RANDOM_SAMPLES = 3             # temp=0.8 samples per (model, conversation)
TEMPERATURE_RANDOM = 0.8
TOP_K = 50
MAX_NEW_TOKENS = 150             # answers should be fairly short
REPETITION_PENALTY = 1.2

# Stop markers — cut the generated turn at the first sign of drift/next-turn so
# we score the actual reply, not the runaway tail.
STOP_MARKERS = ["USER:", "\nASSISTANT:", "[Your Name", "Best regards", "\nDear ",
                "Subject:", "/imagine", "```", "/*", "\n###"]

OUTPUT_FILE = "multiturn_results.txt"


# ====================
# HELPERS
# ====================

def load_model(checkpoint_filename, config, device):
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
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model, checkpoint.get("step", None)


def extract_reply(full_text, history):
    """Strip the scripted history, then cut at the first stop marker."""
    reply = full_text[len(history):] if full_text.startswith(history) else full_text
    cut = len(reply)
    for marker in STOP_MARKERS:
        idx = reply.find(marker)
        if idx != -1:
            cut = min(cut, idx)
    return reply[:cut].strip()


def sample_reply(model, tokenizer, history, device, greedy=False):
    return generate(
        model=model, tokenizer=tokenizer, prompt=history,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=(1.0 if greedy else TEMPERATURE_RANDOM),
        top_k=(1 if greedy else TOP_K),
        device=device, repetition_penalty=REPETITION_PENALTY,
    )


# ====================
# MAIN
# ====================

def main():
    config = Config()
    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = BPETokenizer()
    tokenizer.load(os.path.join(config.data_dir, "tokenizer.json"))
    print(f"Loaded tokenizer (vocab {len(tokenizer.vocab):,})")

    # Verify checkpoints exist
    missing = [(l, os.path.join(config.checkpoint_dir, f))
               for l, f in MODELS_TO_TEST
               if not os.path.exists(os.path.join(config.checkpoint_dir, f))]
    if missing:
        print("\nERROR: missing checkpoints:")
        for l, p in missing:
            print(f"  {l}: {p}")
        return

    n_total = len(MODELS_TO_TEST) * len(CONVERSATIONS) * (1 + N_RANDOM_SAMPLES)
    print(f"\nPlan: {len(MODELS_TO_TEST)} models x {len(CONVERSATIONS)} conversations "
          f"x {1 + N_RANDOM_SAMPLES} samples = {n_total} generations\n")

    start = time.time()
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        out.write("=" * 75 + "\n")
        out.write("MaxGPT-3 — Multi-turn coherence evaluation\n")
        out.write("=" * 75 + "\n")
        out.write(f"Models: {', '.join(m[0] for m in MODELS_TO_TEST)}\n")
        out.write(f"Conversations: {len(CONVERSATIONS)} (scripted history, model generates final turn only)\n")
        out.write(f"Samples per (model, conversation): 1 greedy + {N_RANDOM_SAMPLES} random (temp={TEMPERATURE_RANDOM})\n")
        out.write("Each ===CONTEXT_CHECK=== line says what counts as correctly using prior context.\n")
        out.write("=" * 75 + "\n\n")

        n = 0
        for model_label, ckpt in MODELS_TO_TEST:
            print(f"=== Loading {model_label} ({ckpt}) ===")
            model, step = load_model(ckpt, config, device)
            print(f"  loaded (step {step})")

            for conv in CONVERSATIONS:
                samples = [("greedy", sample_reply(model, tokenizer, conv["history"], device, greedy=True))]
                for i in range(N_RANDOM_SAMPLES):
                    samples.append((f"random_{i+1}", sample_reply(model, tokenizer, conv["history"], device)))

                for sample_label, full_text in samples:
                    n += 1
                    reply = extract_reply(full_text, conv["history"])
                    out.write("===MODEL=== " + model_label + "\n")
                    out.write("===CONV_ID=== " + conv["id"] + " (" + conv["category"] + ")\n")
                    out.write("===HISTORY=== " + repr(conv["history"]) + "\n")
                    out.write("===CONTEXT_CHECK=== " + conv["key"] + "\n")
                    out.write("===SAMPLE=== " + sample_label + "\n")
                    out.write(reply + "\n")
                    out.write("===END===\n\n")
                    out.flush()

                elapsed = time.time() - start
                print(f"  [{n:3d}/{n_total}] {model_label} :: {conv['id']}")

            del model
            if device == "cuda":
                torch.cuda.empty_cache()

        out.write("\n" + "=" * 75 + "\n")
        out.write(f"Complete. {n} generations in {time.time() - start:.0f}s.\n")
        out.write("=" * 75 + "\n")

    print(f"\nDone! {n} generations in {time.time() - start:.0f}s. Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
