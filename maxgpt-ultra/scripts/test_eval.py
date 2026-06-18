"""CPU smoke test for the eval harness.

Trains a tiny model to memorize simple word patterns, then checks: validation perplexity
is low (it learned), greedy generation reproduces a memorized continuation, multiple-
choice scoring prefers the correct continuation, and the sample-prompt runner returns
timed completions.

Run from maxgpt-ultra/:  ../venv/bin/python scripts/test_eval.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # maxgpt-ultra/

import torch

from model import ModelConfig, MaxGPTUltra, generate
from tokenizer.tokenizer import train_tokenizer, UltraTokenizer
from data.prepare import tokenize_to_shards
from data.loader import PackedShardDataset
from train.trainer import Trainer
from eval.harness import eval_perplexity, eval_multiple_choice, run_sample_prompts, evaluate

TOK = "/tmp/maxgpt_ultra_eval_tok.json"
SHARDS = "/tmp/maxgpt_ultra_eval_shards"
OUT = "/tmp/maxgpt_ultra_eval_run"

DOCS = [
    "alpha beta gamma delta epsilon",
    "one two three four five",
    "red green blue yellow purple",
    "north south east west center",
] * 120


def main() -> None:
    print("=" * 72)
    print("MaxGPT-Ultra eval-harness smoke test")
    print("=" * 72)

    print("\n[setup] tiny tokenizer + shards + model; memorize for 80 steps")
    train_tokenizer(iter(DOCS), vocab_size=1000, out_path=TOK)
    tok = UltraTokenizer(TOK)
    tokenize_to_shards(DOCS, tok, SHARDS, shard_size=2048)
    seq_len = 16
    cfg = ModelConfig(vocab_size=tok.vocab_size, d_model=128, n_layers=2, n_heads=4,
                      n_kv_heads=2, mlp_hidden=256, seq_len=seq_len)
    model = MaxGPTUltra(cfg)
    data = PackedShardDataset(SHARDS, seq_len)
    tcfg = {"micro_batch": 8, "grad_accum": 1, "total_tokens": 128 * 200, "warmup_tokens": 128 * 5,
            "lr": 3e-3, "decay_frac": 0.2, "z_loss": 0.0, "autosave_minutes": 9999, "log_every": 1000}
    trainer = Trainer(model, data, tcfg, device="cpu", out_dir=OUT, seed=0)
    for _ in range(80):
        trainer.train_step()

    print("\n[1] validation perplexity is low (it learned)")
    ppl = eval_perplexity(model, data, n_batches=10, batch_size=8, device="cpu")
    print(f"  val_loss={ppl['val_loss']:.3f}  val_ppl={ppl['val_ppl']:.2f}")
    assert ppl["val_loss"] < 2.0, "model did not learn the tiny dataset"

    print("\n[2] greedy generation reproduces a memorized continuation")
    ids = torch.tensor([tok.encode("alpha beta gamma delta")])
    gen = generate(model, ids, max_new_tokens=4, temperature=0.0, eos_id=tok.eos_id)
    completion = tok.decode(gen[0, ids.shape[1]:].tolist(), skip_special=True)
    print(f"  'alpha beta gamma delta' -> '{completion.strip()}'")
    assert gen.shape[1] <= ids.shape[1] + 4
    assert "epsilon" in completion, "did not greedily continue the memorized pattern"

    print("\n[3] multiple-choice picks the correct continuation")
    mc = [
        {"context": "alpha beta gamma", "choices": [" delta epsilon", " four five"], "answer": 0},
        {"context": "one two three", "choices": [" delta epsilon", " four five"], "answer": 1},
        {"context": "red green blue", "choices": [" yellow purple", " south east"], "answer": 0},
        {"context": "north south", "choices": [" east west center", " three four five"], "answer": 0},
    ]
    res = eval_multiple_choice(model, tok, mc, device="cpu")
    print(f"  mc_acc={res['mc_acc']:.2f} over {res['mc_n']} questions")
    assert res["mc_acc"] >= 0.75, "multiple-choice scoring is not preferring correct continuations"

    print("\n[4] sample-prompt runner returns timed completions")
    samples = run_sample_prompts(model, tok, ["one two", "red green"], device="cpu",
                                 max_new_tokens=6, temperature=0.7)
    for s in samples:
        assert isinstance(s["completion"], str) and 0 < s["new_tokens"] <= 6 and s["tok_per_s"] > 0
    print(f"  generated {len(samples)} samples, e.g. 'one two' -> '{samples[0]['completion'].strip()}'")

    print("\n[5] evaluate() bundles everything into one dict")
    bundle = evaluate(model, tokenizer=tok, val_data=data, mc_examples=mc,
                      sample_prompts=["alpha beta"], device="cpu", n_batches=5)
    assert "val_loss" in bundle and "mc_acc" in bundle and "samples" in bundle
    print(f"  keys: {sorted(bundle.keys())}")

    print("\n" + "=" * 72)
    print("ALL CHECKS PASSED ✅")
    print("=" * 72)


if __name__ == "__main__":
    main()
