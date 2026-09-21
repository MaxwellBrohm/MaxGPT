"""Decide the Ultra recipe from the finished shakedown A/B runs, and write the final config.

  python scripts/ab_verdict.py --runs runs/ab --base configs/ultra_lambda.yaml --out configs/ultra_lambda_final.yaml

Score = mean held-out loss of each variant's last --last evals. The winner's flags are adopted only
if it beats the baseline recipe (adamw) by at least --min-gain (relative); otherwise the baseline
stands. A variant that did not finish (or has no evals) is ignored. Prints the table and the
decision; exit code 0 either way, so an unattended launcher can always proceed.
"""
import argparse
import glob
import json
import os
import statistics

FLAGS = {   # variant -> (train overrides, model overrides)
    "adamw": ({}, {}),
    "normuon": ({"optimizer": "normuon", "cautious_wd": True}, {}),
    "adamw_arch": ({}, {"attn_gate": True, "value_residual": True, "norm_scaling": True}),
    "normuon_arch": ({"optimizer": "normuon", "cautious_wd": True},
                     {"attn_gate": True, "value_residual": True, "norm_scaling": True}),
}


def score(path: str, last: int):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    meta = next((r for r in rows if r.get("event") == "meta"), None)
    evals = [r["val_loss"] for r in rows if r.get("event") == "eval" and "val_loss" in r]
    steps = [r["step"] for r in rows if "loss" in r and r.get("event") is None]
    if not meta or not evals or not steps:
        return None
    finished = max(steps) >= int(meta["total_steps"]) - 1
    return {"val": statistics.mean(evals[-last:]), "n_evals": len(evals), "step": max(steps),
            "total": int(meta["total_steps"]), "finished": finished}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/ab")
    ap.add_argument("--base", default="configs/ultra_lambda.yaml")
    ap.add_argument("--out", default="configs/ultra_lambda_final.yaml")
    ap.add_argument("--last", type=int, default=3)
    ap.add_argument("--min-gain", type=float, default=0.005, help="relative val-loss gain needed to leave the baseline recipe")
    ap.add_argument("--allow-unfinished", action="store_true", help="score runs that did not reach their last step")
    args = ap.parse_args()

    results = {}
    for path in sorted(glob.glob(os.path.join(args.runs, "*", "metrics.jsonl"))):
        name = os.path.basename(os.path.dirname(path))
        if name not in FLAGS:
            continue
        s = score(path, args.last)
        if s and (s["finished"] or args.allow_unfinished):
            results[name] = s
    print(f"{'variant':<14} {'val loss':>9} {'evals':>6} {'step':>12}")
    for n, s in sorted(results.items(), key=lambda kv: kv[1]["val"]):
        print(f"{n:<14} {s['val']:>9.4f} {s['n_evals']:>6} {s['step']:>6}/{s['total']}{'' if s['finished'] else ' (unfinished)'}")
    base_name = "adamw"
    winner, reason = base_name, "baseline recipe"
    if results:
        best = min(results, key=lambda n: results[n]["val"])
        if base_name in results and best != base_name:
            gain = (results[base_name]["val"] - results[best]["val"]) / results[base_name]["val"]
            if gain >= args.min_gain:
                winner, reason = best, f"beats adamw by {100 * gain:.2f}% held-out loss (>= {100 * args.min_gain:.1f}%)"
            else:
                reason = f"best variant {best} only {100 * gain:.2f}% better than adamw (< {100 * args.min_gain:.1f}%): keeping the baseline"
        elif base_name not in results and best != base_name:
            winner, reason = best, "adamw run missing; best available variant"
    else:
        reason = "no finished A/B runs: baseline recipe"
    train_over, model_over = FLAGS[winner]
    base_rel = os.path.relpath(os.path.abspath(args.base), os.path.dirname(os.path.abspath(args.out))) or os.path.basename(args.base)
    lines = [f"# written by scripts/ab_verdict.py: winner = {winner} ({reason})", f"extends: {base_rel}", "name: maxgpt-ultra"]
    if model_over:
        lines.append("model:")
        lines += [f"  {k}: {str(v).lower()}" for k, v in model_over.items()]
    if train_over:
        lines.append("train:")
        lines += [f"  {k}: {str(v).lower() if isinstance(v, bool) else v}" for k, v in train_over.items()]
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nDECISION: {winner}  ({reason})\nwrote {args.out}")


if __name__ == "__main__":
    main()
