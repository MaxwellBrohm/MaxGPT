"""Plot thermal_watch.py logs: per-card temperature, the idle "thermometer" cards, total power, and
the CPU sensor over time, with the number of loaded cards shaded underneath. One PNG per log.

  python scripts/thermal_plot.py --log thermal.level3.csv --thermometer 6,8 --out level3.png --title "8 cards"
"""
import argparse
import csv
from collections import defaultdict
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--thermometer", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="")
    args = ap.parse_args()
    therm = {int(x) for x in args.thermometer.split(",") if x.strip()}
    by_gpu = defaultdict(list)
    per_time = defaultdict(list)
    for r in csv.DictReader(open(args.log, encoding="utf-8")):
        t = datetime.strptime(r["time"], "%Y-%m-%d %H:%M:%S")
        by_gpu[int(r["gpu"])].append((t, float(r["temp_c"])))
        per_time[t].append(r)
    if not by_gpu:
        raise SystemExit("empty log")
    t0 = min(per_time)
    mins = lambda t: (t - t0).total_seconds() / 60
    times = sorted(per_time)
    loaded = [sum(1 for r in per_time[t] if float(r["util_pct"]) >= 50) for t in times]
    power = [sum(float(r["power_w"]) for r in per_time[t]) for t in times]
    cpu = [float(per_time[t][0]["cpu_c"]) if per_time[t][0]["cpu_c"] not in ("", "nan") else float("nan") for t in times]

    fig, ax = plt.subplots(3, 1, figsize=(11, 9), sharex=True, gridspec_kw={"height_ratios": [3, 1.2, 1]})
    ax[0].fill_between([mins(t) for t in times], 0, [l * 10 for l in loaded], color="#dddddd", step="mid", label="loaded cards x10")
    for g in sorted(by_gpu):
        xs = [mins(t) for t, _ in by_gpu[g]]
        ys = [v for _, v in by_gpu[g]]
        if g in therm:
            ax[0].plot(xs, ys, lw=2.4, label=f"GPU {g} (idle thermometer)")
        else:
            ax[0].plot(xs, ys, lw=1, alpha=0.8, label=f"GPU {g}")
    ax[0].axhline(89, color="red", ls="--", lw=0.8)
    ax[0].text(0.2, 89.5, "89 C max operating", color="red", fontsize=8)
    ax[0].set_ylabel("GPU temperature (C)")
    ax[0].set_ylim(20, 100)
    ax[0].legend(ncol=4, fontsize=7, loc="lower right")
    ax[0].set_title(args.title or args.log)
    ax[1].plot([mins(t) for t in times], power, color="black")
    ax[1].set_ylabel("total GPU power (W)")
    ax[2].plot([mins(t) for t in times], cpu, color="tab:purple")
    ax[2].set_ylabel("CPU sensor (C)")
    ax[2].set_xlabel("minutes")
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}: {len(times)} samples, {len(by_gpu)} GPUs, {mins(times[-1]):.0f} min")


if __name__ == "__main__":
    main()
