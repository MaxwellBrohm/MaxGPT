"""Summarize the shakedown A/B runs: one line per variant with the latest step, train loss,
held-out loss, throughput, and (once several evals exist) the mean of the last few val losses.

  python scripts/ab_report.py            # reads runs/ab/*/metrics.jsonl
  python scripts/ab_report.py --runs runs/ab --last 3
"""
import argparse
import glob
import json
import os
import statistics


def load(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/ab")
    ap.add_argument("--last", type=int, default=3, help="average this many latest val losses")
    args = ap.parse_args()
    print(f"{'variant':<14} {'step':>6} {'train':>7} {'val':>7} {'val(avg last)':>14} {'tok/s':>9} {'note':>10}")
    for path in sorted(glob.glob(os.path.join(args.runs, "*", "metrics.jsonl"))):
        name = os.path.basename(os.path.dirname(path))
        rows = load(path)
        meta = next((r for r in rows if r.get("event") == "meta"), {})
        train = [r for r in rows if "loss" in r and r.get("event") is None]
        evals = [r for r in rows if r.get("event") == "eval" and "val_loss" in r]
        last = train[-1] if train else {}
        val = evals[-1]["val_loss"] if evals else float("nan")
        val_avg = statistics.mean(e["val_loss"] for e in evals[-args.last:]) if evals else float("nan")
        note = f"{last['step']}/{meta.get('total_steps', '?')}" if last else "starting"
        ov = sum(1 for r in train if r.get("overflow"))
        if ov:
            note += f" ov{ov}"
        print(f"{name:<14} {last.get('step', 0):>6} {last.get('loss', float('nan')):>7.3f} {val:>7.3f} "
              f"{val_avg:>14.4f} {last.get('tok_per_s', 0):>9,.0f} {note:>10}")


if __name__ == "__main__":
    main()
