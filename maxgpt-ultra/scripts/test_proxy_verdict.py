"""Tests for scripts/proxy_verdict.py (the proxy A/B decision rule). Standard library only.

Run from maxgpt-ultra/:  ../venv/bin/python scripts/test_proxy_verdict.py
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # maxgpt-ultra/

from scripts.proxy_verdict import decide, final_val_loss, read_runs, verdict


def write_run(root, name, losses, suite, done=True):
    d = os.path.join(root, name)
    os.makedirs(d)
    with open(os.path.join(d, "metrics.jsonl"), "w") as f:
        f.write(json.dumps({"step": 10, "loss": 9.0, "lr": 1e-3}) + "\n")           # a training row, not an eval
        for i, v in enumerate(losses):
            f.write(json.dumps({"step": 250 * (i + 1), "event": "eval", "val_loss": v}) + "\n")
    if suite is not None:
        with open(os.path.join(d, "suite.json"), "w") as f:
            json.dump({"avg": suite}, f)
    if done:
        open(os.path.join(d, "DONE"), "w").close()


def main() -> None:
    print("[1] the final loss is the mean of the last three evaluations")
    root = tempfile.mkdtemp(prefix="mgu_proxy_")
    write_run(root, "base_s0", [5.0, 3.02, 3.00, 2.98], 0.400)
    assert abs(final_val_loss(os.path.join(root, "base_s0", "metrics.jsonl")) - 3.00) < 1e-12
    assert final_val_loss(os.path.join(root, "nope", "metrics.jsonl")) is None

    print("[2] the rule: beyond two noise floors and benchmarks intact, or the baseline stays")
    bl, bs = [3.000, 3.002, 2.998], [0.400, 0.402, 0.398]           # sd 0.002 on both; suite floor 0.005 applies
    win = decide(bl, bs, 2.990, 0.401)
    assert win["adopt"] and abs(win["loss_gain"] - 0.010) < 1e-9, win
    assert not decide(bl, bs, 2.997, 0.410)["adopt"], "a gain inside the noise must not be adopted"
    assert not decide(bl, bs, 3.020, 0.450)["adopt"], "a WORSE loss must never be adopted, whatever the benchmarks say"
    assert not decide(bl, bs, 2.980, 0.385)["adopt"], "lower loss with clearly worse benchmarks must not be adopted"
    assert decide(bl, bs, 2.980, 0.392)["adopt"], "a small benchmark dip inside its noise is fine"
    tight = decide([3.0, 3.0, 3.0], bs, 2.9985, 0.40)                 # zero spread: the 0.001 floor decides
    assert not tight["adopt"] and tight["loss_noise"] == 0.001, tight
    assert decide([3.0, 3.0, 3.0], bs, 2.9975, 0.40)["adopt"]
    assert not decide(bl[:2], bs[:2], 2.5, 0.9)["adopt"], "fewer than three baseline seeds: no decision"
    assert not decide(bl, bs, None, None)["adopt"]

    print("[3] the data arm is judged on benchmarks alone")
    assert decide(bl, bs, None, 0.420, suite_only=True)["adopt"]
    assert not decide(bl, bs, None, 0.405, suite_only=True)["adopt"]
    assert not decide(bl, bs, 1.0, 0.380, suite_only=True)["adopt"], "a low loss on other data proves nothing"

    print("[4] reading run folders and the full verdict")
    write_run(root, "base_s1", [3.1, 3.004, 3.002, 3.000], 0.402)
    write_run(root, "base_s2", [3.1, 3.000, 2.998, 2.996], 0.398)
    write_run(root, "peri_ln_s0", [3.1, 2.992, 2.990, 2.988], 0.401)
    write_run(root, "relu2_s0", [3.1, 3.001, 2.999, 3.000], 0.400)
    write_run(root, "oldmix_s0", [2.5, 2.4, 2.4, 2.4], 0.380)
    write_run(root, "lr_4.5e-4_s0", [3.1, 3.0, 3.0], None, done=False)
    runs = read_runs(root)
    assert set(runs) == {"base", "peri_ln", "relu2", "oldmix", "lr_4.5e-4"} and len(runs["base"]) == 3, runs.keys()
    v = verdict(runs)
    got = {arm: d["adopt"] for arm, d in v["arms"].items()}
    assert got == {"peri_ln": True, "relu2": False, "oldmix": False, "lr_4.5e-4": False}, got
    assert "not finished" in v["arms"]["lr_4.5e-4"]["why"]
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
