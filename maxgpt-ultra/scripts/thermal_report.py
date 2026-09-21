"""Summarize thermal.csv from thermal_watch.py: per 5-minute window, how many cards were loaded,
the hottest card, the thermometer (idle) cards' mean, total power, and the CPU sensor. Then the
plateau per load level and the cool-down after the last load.

  python scripts/thermal_report.py --log ~/MaxGPT/thermal.csv --thermometer 8,9
"""
import argparse
import csv
import statistics
from collections import defaultdict
from datetime import datetime


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--thermometer", default="8,9")
    ap.add_argument("--window", type=int, default=300, help="seconds per row")
    args = ap.parse_args()
    therm = {int(x) for x in args.thermometer.split(",") if x.strip()}
    samples = defaultdict(list)                       # time -> rows
    with open(args.log, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            samples[r["time"]].append(r)
    if not samples:
        print("no samples")
        return
    t0 = datetime.strptime(min(samples), "%Y-%m-%d %H:%M:%S")
    windows = defaultdict(list)
    for ts, rows in samples.items():
        t = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
        k = int((t - t0).total_seconds() // args.window)
        loaded = [r for r in rows if float(r["util_pct"]) >= 50]
        hot = max(float(r["temp_c"]) for r in rows)
        th = [float(r["temp_c"]) for r in rows if int(r["gpu"]) in therm]
        windows[k].append((len(loaded), hot, statistics.mean(th) if th else float("nan"),
                           sum(float(r["power_w"]) for r in rows), float(rows[0]["cpu_c"] or "nan")))
    print(f"{'t+min':>6} {'loaded':>6} {'hottest':>8} {'therm':>6} {'power':>7} {'cpu':>5}")
    levels = defaultdict(list)
    for k in sorted(windows):
        w = windows[k]
        n = round(statistics.mean(x[0] for x in w))
        hot = max(x[1] for x in w)
        th = statistics.mean(x[2] for x in w)
        pw = statistics.mean(x[3] for x in w)
        cpu = statistics.mean(x[4] for x in w if x[4] == x[4]) if any(x[4] == x[4] for x in w) else float("nan")
        levels[n].append((hot, th))
        print(f"{k * args.window // 60:>6} {n:>6} {hot:>7.0f}C {th:>5.1f}C {pw:>6.0f}W {cpu:>4.0f}C")
    print("\nplateau per load level (last 2 windows at that level):")
    for n in sorted(levels):
        tail = levels[n][-2:]
        print(f"  {n} cards loaded: hottest {max(h for h, _ in tail):.0f}C, thermometer {statistics.mean(t for _, t in tail):.1f}C")


if __name__ == "__main__":
    main()
