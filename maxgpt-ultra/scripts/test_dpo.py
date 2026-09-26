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

    print("\n[4] length-normalized DPO: the log-ratio gap is per token; vanilla is per sequence")
    import posttrain.dpo as PD
    real_slp = PD.sequence_logprobs
    # chosen: 10 scored tokens, policy -10 vs reference -11 (gap +1, +0.1 per token)
    # rejected: 2 scored tokens, policy -3 vs reference -3 (gap 0)
    cm = torch.zeros(1, 11, dtype=torch.bool); cm[0, 1:] = True          # 10 response tokens after position 0
    rm = torch.zeros(1, 3, dtype=torch.bool);  rm[0, 1:] = True          # 2 response tokens
    fake = {"c": torch.tensor([-10.0]), "r": torch.tensor([-3.0])}
    PD.sequence_logprobs = lambda model, ids, mask, chunk=0: fake["c"] if ids.size(1) == 11 else fake["r"]
    try:
        batch = (torch.zeros(1, 11, dtype=torch.long), cm, torch.zeros(1, 3, dtype=torch.long), rm,
                 torch.tensor([-11.0]), torch.tensor([-3.0]))
        ln_loss, ln_stats = PD.dpo_loss(None, None, batch, beta=0.5, length_norm=True)
        va_loss, va_stats = PD.dpo_loss(None, None, batch, beta=0.5, length_norm=False)
        expect_ln = -math.log(torch.sigmoid(torch.tensor(0.5 * (1.0 / 10 - 0.0))).item())
        expect_va = -math.log(torch.sigmoid(torch.tensor(0.5 * (1.0 - 0.0))).item())
        assert abs(float(ln_loss) - expect_ln) < 1e-6 and abs(float(va_loss) - expect_va) < 1e-6, (float(ln_loss), expect_ln, float(va_loss), expect_va)
        assert ln_stats["chosen_len"] == 10.0 and ln_stats["rejected_len"] == 2.0 and abs(ln_stats["chosen_nll"] - 1.0) < 1e-6
        aux_loss, _ = PD.dpo_loss(None, None, batch, beta=0.5, length_norm=True, sft_weight=0.3)
        assert abs(float(aux_loss) - (expect_ln + 0.3 * 1.0)) < 1e-6, float(aux_loss)
    finally:
        PD.sequence_logprobs = real_slp
    assert PD.pair_margin_ok({"chosen_instruct_reward": 2.0, "rejected_instruct_reward": 1.0}, 0.0)
    assert not PD.pair_margin_ok({"chosen_instruct_reward": 1.0, "rejected_instruct_reward": 2.0}, 0.0)
    assert PD.pair_margin_ok({"prompt": "no reward columns"}, 0.0)
    print(f"  LN loss {float(ln_loss):.4f} (gap 0.1/token) vs vanilla {float(va_loss):.4f} (gap 1/sequence); aux NLL adds 0.3 x 1.0; margin filter ✓")

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
