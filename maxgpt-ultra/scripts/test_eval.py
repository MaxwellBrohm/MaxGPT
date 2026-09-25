"""CPU smoke test for the eval harness.

Trains a tiny model to memorize simple word patterns, then checks: validation perplexity
is low (it learned), greedy generation reproduces a memorized continuation, multiple-
choice scoring prefers the correct continuation, and the sample-prompt runner returns
timed completions.

Run from maxgpt-ultra/:  ../venv/bin/python scripts/test_eval.py
"""
import json
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
    torch.manual_seed(0)      # the model is built before the Trainer seeds torch: pin the init, or the
                              # toy memorizes a different subset of the sentences on every run
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
    print("\n[6] validation reads the SAME held-out slice every call (comparable across evals)")
    a = eval_perplexity(model, data, n_batches=4, batch_size=2, device="cpu")
    data.pos = (data.pos + 5 * seq_len) % data.total      # someone moved the stream in between
    b = eval_perplexity(model, data, n_batches=4, batch_size=2, device="cpu")
    assert a["val_loss"] == b["val_loss"], f"eval drifted with the stream position: {a['val_loss']} vs {b['val_loss']}"
    assert a["val_tokens"] == 4 * 2 * seq_len, a["val_tokens"]
    print(f"  two evals around a moved stream position agree exactly (val_loss={a['val_loss']:.4f}, "
          f"{a['val_tokens']} tokens) ✓")

    print("\n[7] batched scoring (right-padded) equals one-at-a-time scoring")
    from eval.suite import score_continuations, run_suite, write_suite, load_suite, last_suite_step, format_table
    for _ in range(60):                 # memorize harder: at 80 steps a sentence or two is still marginal,
        trainer.train_step()            # and which ones flips with CPU numeric noise between runs
    # only the first two sentences, with 2+ context tokens: the 2-layer toy learns those cold
    # (log-prob ~ -0.002); the other two sentences, and any 1-token context (position 0 never
    # follows an end-of-text here, unlike in training), flip with CPU numeric noise between runs.
    # Unequal total lengths (3, 4, 5, 5, 5 tokens) so the batched path really pads.
    pairs = [("alpha beta", " gamma"), ("one two three", " four"), ("alpha beta gamma", " delta epsilon"),
             ("one two", " three four five"), ("one two three four", " five")]
    one = score_continuations(model, tok, pairs, device="cpu", batch_size=1)
    many = score_continuations(model, tok, pairs, device="cpu", batch_size=5)
    for (a, ka, ba, ga), (b, kb, bb, gb) in zip(one, many):
        assert abs(a - b) < 1e-4 and (ka, ba, ga) == (kb, bb, gb), (a, b, ga, gb)
    bad = [p for p, (_, _, _, g) in zip(pairs, many) if not g]
    assert not bad, f"memorized continuations should all be greedy matches; not greedy: {bad}"
    print(f"  {len(pairs)} pairs of unequal length: log-probs agree within 1e-4, greedy flags agree ✓")

    print("\n[8] suite scorers, fixed-suite files, cadence helper")
    suite = {
        "lambada": [{"context": d.rsplit(" ", 1)[0], "target": " " + d.rsplit(" ", 1)[1]} for d in DOCS[:2]],
        "piqa": [{"context": "one two three", "choices": [" four five", " purple", " west center"], "answer": 0},
                 {"context": "alpha beta", "choices": [" west", " gamma delta epsilon"], "answer": 1}],
        "winogrande": [{"contexts": ["red green blue", "alpha beta gamma"], "continuation": " yellow purple", "answer": 0},
                       {"contexts": ["alpha beta", "one two"], "continuation": " three four five", "answer": 1}],
    }
    res = run_suite(model, tok, suite, device="cpu", batch_size=4)
    assert res["lambada"]["acc"] == 1.0 and res["lambada"]["ppl"] < 1.5, res["lambada"]
    assert res["piqa"]["acc"] == 1.0 and res["piqa"]["acc_norm"] == 1.0, res["piqa"]
    assert res["winogrande"]["acc"] == 1.0, res["winogrande"]
    assert abs(res["avg"] - 1.0) < 1e-9 and res["precision"] == "fp32", (res["avg"], res["precision"])
    bench = "/tmp/maxgpt_ultra_eval_bench"
    write_suite(bench, suite, seed=0)
    assert load_suite(bench) == suite, "suite files did not round-trip"
    assert load_suite(bench, n=1)["piqa"] == suite["piqa"][:1]
    mp = "/tmp/maxgpt_ultra_eval_metrics.jsonl"
    with open(mp, "w") as f:
        f.write(json.dumps({"step": 10, "loss": 1.0}) + "\n")
        f.write(json.dumps({"step": 500, "event": "eval", "val_loss": 1.0, "suite": {"avg": 0.5}}) + "\n")
        f.write(json.dumps({"step": 1000, "event": "eval", "val_loss": 1.0}) + "\n")
    assert last_suite_step(mp) == 500 and last_suite_step("/tmp/does_not_exist.jsonl") is None
    import eval.suite as ES                            # acc vs acc_norm on a case where they disagree
    real_scorer = ES.score_continuations
    # choice 0: raw -1.0 over 2 bytes (-0.5 per byte); choice 1: raw -3.0 over 30 bytes (-0.1 per byte)
    ES.score_continuations = lambda m, t, pairs, **kw: [(-1.0, 1, 2, True), (-3.0, 3, 30, False)]
    try:
        r2 = ES.eval_mc(None, None, [{"context": "c", "choices": ["ab", "x" * 30], "answer": 1}])
    finally:
        ES.score_continuations = real_scorer
    assert r2["acc"] == 0.0 and r2["acc_norm"] == 1.0, r2
    print("  " + format_table(res).replace("\n", "\n  "))
    print("  memorized model scores 100% in all three formats; files round-trip; last suite step = 500 ✓")

    print("ALL CHECKS PASSED ✅")
    print("=" * 72)


if __name__ == "__main__":
    main()
