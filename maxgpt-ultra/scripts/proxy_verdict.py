"""Read the finished proxy A/B runs and apply the decision rule that was fixed before they ran.

    python scripts/proxy_verdict.py --runs runs/proxy [--out runs/proxy/verdict.json]

Layout it expects (written by slurm/proxy_ab.sbatch): runs/proxy/<arm>_s<seed>/metrics.jsonl (the
trainer's log, with held-out `val_loss` in its eval rows) and .../suite.json (scripts/eval_suite.py,
its "avg"). The baseline arm is `base` with three seeds.

The rule (docs/research_2026-11-18_restart.md section 7):
  * noise floor = the standard deviation of the baseline's final held-out loss over its seeds, never
    taken below 0.001 nats (published seed noise at this size is 0.001 to 0.002);
  * a change is ADOPTED only if its final held-out loss beats the baseline mean by more than 2 noise
    floors AND its benchmark average is not more than 2 baseline standard deviations below the
    baseline's (a floor of 0.005 on that too);
  * the data arm (`oldmix`) trains on other data, so its held-out loss is not comparable: it is judged
    on the benchmark average alone, and the NEW mix is confirmed if the old one is not clearly better;
  * with fewer than three baseline seeds nothing is adopted.
"Final held-out loss" is the mean of the last three evaluations, as in the September A/B.
Standard library only.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics
import sys

LOSS_FLOOR, SUITE_FLOOR, K = 0.001, 0.005, 2.0
SUITE_ONLY = {"oldmix"}                      # arms whose held-out loss is on different data


def final_val_loss(metrics_path: str, last: int = 3) -> float | None:
    vals = []
    if os.path.exists(metrics_path):
        with open(metrics_path, encoding="utf-8") as f:
            for line in f:
                if '"val_loss"' not in line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("event") == "eval" and isinstance(r.get("val_loss"), (int, float)):
                    vals.append(float(r["val_loss"]))
    return sum(vals[-last:]) / len(vals[-last:]) if vals else None


def read_runs(runs_dir: str) -> dict:
    """{arm: [{"seed", "loss", "suite", "done"}, ...]} for every run directory named <arm>_s<seed>."""
    out: dict = {}
    for d in sorted(glob.glob(os.path.join(runs_dir, "*_s*"))):
        m = re.match(r"^(.*)_s(\d+)$", os.path.basename(d))
        if not m or not os.path.isdir(d):
            continue
        suite = None
        sp = os.path.join(d, "suite.json")
        if os.path.exists(sp):
            with open(sp, encoding="utf-8") as f:
                suite = json.load(f).get("avg")
        out.setdefault(m.group(1), []).append({"seed": int(m.group(2)), "loss": final_val_loss(os.path.join(d, "metrics.jsonl")),
                                               "suite": suite, "done": os.path.exists(os.path.join(d, "DONE"))})
    return out


def decide(base_losses: list, base_suites: list, arm_loss, arm_suite, suite_only: bool = False) -> dict:
    """Apply the rule to one arm. Returns {"adopt": bool, "why": str, plus the numbers used}."""
    base_losses = [x for x in base_losses if x is not None]
    base_suites = [x for x in base_suites if x is not None]
    if len(base_losses) < 3 or len(base_suites) < 3:
        return {"adopt": False, "why": "fewer than three finished baseline seeds: no decision"}
    lm, ls = statistics.mean(base_losses), max(statistics.stdev(base_losses), LOSS_FLOOR)
    sm, ss = statistics.mean(base_suites), max(statistics.stdev(base_suites), SUITE_FLOOR)
    res = {"base_loss": lm, "loss_noise": ls, "base_suite": sm, "suite_noise": ss, "arm_loss": arm_loss, "arm_suite": arm_suite}
    if arm_suite is None or (arm_loss is None and not suite_only):
        return {**res, "adopt": False, "why": "the arm has not finished"}
    suite_gain = arm_suite - sm
    if suite_only:
        better = suite_gain > K * ss
        return {**res, "adopt": better, "suite_gain": suite_gain,
                "why": ("benchmark average clearly above the baseline" if better
                        else "benchmark average not clearly above the baseline")}
    loss_gain = lm - arm_loss                     # positive = the arm's loss is lower
    if loss_gain <= K * ls:
        return {**res, "adopt": False, "loss_gain": loss_gain, "suite_gain": suite_gain,
                "why": f"loss gain {loss_gain:+.4f} is within {K:g} noise floors ({K * ls:.4f})"}
    if suite_gain < -K * ss:
        return {**res, "adopt": False, "loss_gain": loss_gain, "suite_gain": suite_gain,
                "why": f"lower loss but the benchmark average fell by {-suite_gain:.4f} (limit {K * ss:.4f})"}
    return {**res, "adopt": True, "loss_gain": loss_gain, "suite_gain": suite_gain,
            "why": f"loss gain {loss_gain:+.4f} exceeds {K:g} noise floors ({K * ls:.4f}) and the benchmarks hold"}


def verdict(runs: dict) -> dict:
    base = runs.get("base", [])
    bl, bs = [r["loss"] for r in base], [r["suite"] for r in base]
    out = {"baseline": {"seeds": len(base), "losses": bl, "suites": bs}, "arms": {}}
    for arm, rs in sorted(runs.items()):
        if arm == "base":
            continue
        r = rs[0]
        out["arms"][arm] = decide(bl, bs, r["loss"], r["suite"], suite_only=arm in SUITE_ONLY)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs/proxy")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    runs = read_runs(args.runs)
    if not runs:
        sys.exit(f"no runs found under {args.runs}")
    v = verdict(runs)
    b = v["baseline"]
    print(f"baseline: {b['seeds']} seed(s), final held-out loss {b['losses']}, benchmark average {b['suites']}")
    for arm, d in v["arms"].items():
        note = " (data arm: here ADOPT would mean the OLD mix is better)" if arm in SUITE_ONLY else ""
        print(f"  {arm:<18} {'ADOPT' if d['adopt'] else 'keep the baseline':<18} {d['why']}{note}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(v, f, indent=2)
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
