"""CPU smoke test for SFT (supervised fine-tuning).

Checks the three things that must be right: assistant-only loss masking, the all-masked
batch can't NaN, and a tiny model actually learns to produce the assistant reply after
fine-tuning on chat data.

Run from maxgpt-ultra/:  ../venv/bin/python scripts/test_sft.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # maxgpt-ultra/

import torch

from model import ModelConfig, MaxGPTUltra, generate
from tokenizer.tokenizer import train_tokenizer, UltraTokenizer
from posttrain.sft_data import encode_chat_example, SFTDataset
from train.trainer import Trainer

TOK = "/tmp/maxgpt_ultra_sft_tok.json"
OUT = "/tmp/maxgpt_ultra_sft_run"

# enough text to train a usable tiny tokenizer that covers our chat tokens
CORPUS = ["ping pong hello world the assistant replies politely and helpfully to the user",
          "You are a terse assistant. ping pong hello world test reply done."] * 200

EXAMPLES = [
    {"messages": [{"role": "user", "content": "ping"}, {"role": "assistant", "content": "pong"}]},
    {"messages": [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "world"}]},
] * 80


def main() -> None:
    print("=" * 72)
    print("MaxGPT-Ultra SFT smoke test")
    print("=" * 72)

    train_tokenizer(iter(CORPUS), vocab_size=800, out_path=TOK)
    tok = UltraTokenizer(TOK)

    print("\n[1] assistant-only loss masking")
    toks, sup = encode_chat_example(
        [{"role": "user", "content": "ping"}, {"role": "assistant", "content": "pong"}], tok)
    supervised = tok.decode([t for t, s in zip(toks, sup) if s])
    unsupervised = tok.decode([t for t, s in zip(toks, sup) if not s])
    print(f"  supervised tokens decode to: {supervised!r}")
    assert "pong" in supervised and "ping" not in supervised, "masking supervised the wrong tokens"
    assert "ping" in unsupervised, "user text should be unsupervised"
    print("  loss is on the assistant reply only ✓")

    print("\n[2] all-masked batch does not NaN")
    cfg = ModelConfig(vocab_size=tok.vocab_size, d_model=96, n_layers=2, n_heads=4,
                      n_kv_heads=2, mlp_hidden=192, seq_len=24)
    m = MaxGPTUltra(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    y = torch.full((2, 16), -100)
    _, loss = m(x, y)
    assert torch.isfinite(loss), "all-masked batch produced non-finite loss"
    print(f"  all-masked loss = {float(loss):.3f} (finite) ✓")

    print("\n[3] model learns the assistant replies")
    ds = SFTDataset(EXAMPLES, tok, seq_len=24)
    tcfg = {"micro_batch": 8, "grad_accum": 1, "total_tokens": 24 * 8 * 200,
            "warmup_tokens": 24 * 8 * 5, "lr": 3e-3, "decay_frac": 0.1, "z_loss": 0.0,
            "autosave_minutes": 9999, "log_every": 1000}
    trainer = Trainer(m, ds, tcfg, device="cpu", out_dir=OUT, seed=0)
    for _ in range(120):
        trainer.train_step()

    def reply(user):
        prompt = tok.render_chat([{"role": "user", "content": user}], add_generation_prompt=True)
        ids = torch.tensor([tok.encode(prompt)])
        gen = generate(m, ids, max_new_tokens=8, temperature=0.0, eos_id=tok.eos_id)
        return tok.decode(gen[0, ids.shape[1]:].tolist(), skip_special=True)

    r_ping, r_hello = reply("ping"), reply("hello")
    print(f"  user 'ping'  -> assistant '{r_ping.strip()}'")
    print(f"  user 'hello' -> assistant '{r_hello.strip()}'")
    assert "pong" in r_ping and "world" in r_hello, "model did not learn the chat mapping"
    print("  learned ping->pong and hello->world ✓")

    print("\n" + "=" * 72)
    print("ALL CHECKS PASSED ✅")
    print("=" * 72)


if __name__ == "__main__":
    main()
