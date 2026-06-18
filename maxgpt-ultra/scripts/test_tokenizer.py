"""Smoke test for the MaxGPT-Ultra tokenizer.

Trains a tiny tokenizer on a small sample (the real run uses vocab=49152 on a corpus
sample) and verifies the properties we actually depend on:
  - lossless round-trip on prose, code, numbers, and unseen unicode/emoji (byte fallback),
  - every digit is its own token,
  - special tokens encode atomically and keep low, stable ids,
  - chat rendering produces the special tokens we expect.

Run from maxgpt-ultra/:  ../venv/bin/python scripts/test_tokenizer.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # maxgpt-ultra/

from tokenizer.tokenizer import (
    train_tokenizer, UltraTokenizer, SPECIAL_TOKENS, END, PAD, IM_START, IM_END, THINK_START,
)

# A small but varied sample so BPE has something to learn (repeated to give it volume).
SAMPLE_BASE = [
    "The quick brown fox jumps over the lazy dog. Language models learn from text.",
    "She sold 3 apples and 27 oranges for $4.50 on March 15, 2026 at 9:45am.",
    "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a",
    "Attention is all you need; transformers process tokens in parallel with self-attention.",
    "Café au lait, jalapeño, naïve, Zürich, Москва, 北京, 東京, and emoji like 🚀☕🤖🔥.",
    "import torch\nx = torch.randn(2, 8)\nprint(x.mean(), x.std())  # quick check",
    "Reasoning step by step: first we add, then we multiply, finally we compare the results.",
]
SAMPLE = SAMPLE_BASE * 200

OUT = "/tmp/maxgpt_ultra_test_tokenizer.json"


def main() -> None:
    print("=" * 72)
    print("MaxGPT-Ultra tokenizer smoke test")
    print("=" * 72)

    print("\n[train] tiny vocab=2000 on the sample ...")
    train_tokenizer(iter(SAMPLE), vocab_size=2000, out_path=OUT)
    tk = UltraTokenizer(OUT)
    print(f"  vocab_size={tk.vocab_size}  eos={tk.eos_id} pad={tk.pad_id}")

    print("\n[1] lossless round-trip (incl. unseen unicode / emoji via byte fallback)")
    cases = [
        "Hello, world!",
        "café ☕ 北京 🚀 — naïve résumé",          # none of this was 'learned', byte fallback must cover it
        "def f(x): return x * 2  # comment",
        "Order 66 executed at 3:30pm on 7/4/1776.",
        "\t\n  weird   whitespace\n\n",
    ]
    for s in cases:
        out = tk.decode(tk.encode(s))
        assert out == s, f"round-trip FAILED\n  in:  {s!r}\n  out: {out!r}"
    print(f"  all {len(cases)} cases round-trip exactly ✓")

    print("\n[2] digit-splitting (each digit its own token)")
    digits = "1234567890"
    ids = tk.encode(digits)
    assert len(ids) == 10, f"expected 10 tokens for {digits!r}, got {len(ids)}: {ids}"
    assert tk.decode(ids) == digits
    print(f"  '{digits}' -> {len(ids)} tokens ✓")

    print("\n[3] special tokens encode atomically")
    for t in [END, PAD, IM_START, IM_END, THINK_START]:
        ids = tk.encode(t)
        assert len(ids) == 1, f"{t!r} not atomic: {ids}"
    print(f"  all {len(SPECIAL_TOKENS)} special tokens present; sampled ones are single tokens ✓")

    print("\n[4] special ids are reserved low")
    max_special_id = max(tk.specials.values())
    assert max_special_id < len(SPECIAL_TOKENS), f"special ids not low: max={max_special_id}"
    print(f"  special ids occupy 0..{max_special_id} (count={len(SPECIAL_TOKENS)}) ✓")

    print("\n[5] chat rendering produces ChatML special tokens")
    msgs = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "What is 2 + 2?"},
    ]
    ids = tk.encode_chat(msgs)
    assert tk.specials[IM_START] in ids and tk.specials[IM_END] in ids
    rendered = tk.render_chat(msgs)
    assert rendered.endswith(f"{IM_START}assistant\n")
    print("  render_chat() emits <|im_start|>/<|im_end|> and an assistant prompt ✓")

    print("\n" + "=" * 72)
    print("ALL CHECKS PASSED ✅")
    print("=" * 72)


if __name__ == "__main__":
    main()
