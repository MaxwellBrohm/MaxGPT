"""CPU smoke test for DPO.

With the policy initialized equal to the frozen reference, the DPO loss should start at
about ln(2) and the reward margin at ~0. After a few steps of preferring `chosen` over
`rejected`, the loss should drop and the margin should grow positive.

Run from maxgpt-ultra/:  ../venv/bin/python scripts/test_dpo.py
"""
import copy
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # maxgpt-ultra/

import torch

from model import ModelConfig, MaxGPTUltra
from tokenizer.tokenizer import train_tokenizer, UltraTokenizer
from posttrain.dpo import DPODataset, DPOTrainer, dpo_loss, sequence_logprobs

TOK = "/tmp/maxgpt_ultra_dpo_tok.json"
OUT = "/tmp/maxgpt_ultra_dpo_run"

CORPUS = ["say a b x y the assistant answers the user with a short reply yes no good",
          "You are a helpful assistant. say a then b, say x then y, reply done."] * 200

PREFS = [
    {"prompt": [{"role": "user", "content": "say a"}], "chosen": "a", "rejected": "b"},
    {"prompt": [{"role": "user", "content": "say x"}], "chosen": "x", "rejected": "y"},
] * 40


def main() -> None:
    print("=" * 72)
    print("MaxGPT-Ultra DPO smoke test")
    print("=" * 72)

    train_tokenizer(iter(CORPUS), vocab_size=700, out_path=TOK)
    tok = UltraTokenizer(TOK)
    cfg = ModelConfig(vocab_size=tok.vocab_size, d_model=96, n_layers=2, n_heads=4,
                      n_kv_heads=2, mlp_hidden=192, seq_len=24)
    policy = MaxGPTUltra(cfg)
    ref = copy.deepcopy(policy)            # reference == policy at the start
    data = DPODataset(PREFS, tok, seq_len=24)

    print("\n[1] sequence_logprobs shape")
    b = data.next_batch(4)
    lp = sequence_logprobs(policy, b[0], b[1])
    assert lp.shape == (4,), lp.shape
    print(f"  per-sequence logprobs shape {tuple(lp.shape)} ✓")

    print("\n[2] initial loss ~ ln(2), margin ~ 0 (policy == reference)")
    _, s0 = dpo_loss(policy, ref, data.next_batch(8), beta=0.1)
    print(f"  loss={s0['dpo_loss']:.3f} (ln2={math.log(2):.3f})  margin={s0['reward_margin']:+.3f}")
    assert abs(s0["dpo_loss"] - math.log(2)) < 0.05 and abs(s0["reward_margin"]) < 1e-3

    print("\n[3] after DPO, loss falls and the preference margin grows")
    data.pos = 0
    tcfg = {"batch_size": 4, "grad_accum": 1, "total_steps": 60, "warmup_steps": 5,
            "lr": 1e-3, "decay_frac": 0.1, "autosave_minutes": 9999, "log_every": 1000}
    trainer = DPOTrainer(policy, ref, data, tcfg, device="cpu", out_dir=OUT, beta=0.1, seed=0)
    trainer.train(max_steps=60)
    _, sN = dpo_loss(policy, ref, data.next_batch(8), beta=0.1)
    print(f"  loss {s0['dpo_loss']:.3f} -> {sN['dpo_loss']:.3f}   "
          f"margin {s0['reward_margin']:+.3f} -> {sN['reward_margin']:+.3f}   acc={sN['acc']:.2f}")
    assert sN["dpo_loss"] < s0["dpo_loss"] - 0.05, "DPO loss did not decrease"
    assert sN["reward_margin"] > 0.02, "preference margin did not grow"
    assert sN["acc"] >= 0.75, "policy does not prefer chosen on most pairs"
    print("  policy now prefers the chosen responses ✓")

    print("\n" + "=" * 72)
    print("ALL CHECKS PASSED ✅")
    print("=" * 72)


if __name__ == "__main__":
    main()
