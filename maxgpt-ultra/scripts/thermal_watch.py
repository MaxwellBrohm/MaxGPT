"""Thermal watchdog + logger for a shared GPU box. Runs ON the box (tmux), independent of anyone's
laptop connection, so a load test can never outlive its safety cutoff.

  python scripts/thermal_watch.py --thermometer 8,9 --kill-sessions ab_adamw,ab_normuon,burn

Every --interval seconds it logs one CSV row per GPU (temp, power, util, fan, SM clock, throttle
flags) plus the hottest CPU sensor, and prints a one-line summary. It TRIPS (kills the named tmux
sessions and every train/burn/bench process, then keeps logging so the cool-down is recorded) when:
  - any GPU reaches --max-temp (default: 5 C under the driver's own SHUTDOWN temperature; Turing
    cards normally sit at 85-89 C under sustained load and throttle their clocks at ~91 C, which
    is their own protection working, not an emergency),
  - the idle "thermometer" GPUs rise --idle-rise C above their baseline (room heating up),
  - a GPU reports a hardware slowdown (power brake) at all, or a thermal slowdown continuously
    for --throttle-secs, or a CPU sensor passes --max-cpu.
Baseline = the mean of the thermometer GPUs' first --baseline-samples readings.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import subprocess
import sys
import time

FIELDS = ["index", "temperature.gpu", "power.draw", "utilization.gpu", "fan.speed", "clocks.sm",
          "clocks_throttle_reasons.hw_slowdown", "clocks_throttle_reasons.hw_thermal_slowdown",
          "clocks_throttle_reasons.sw_thermal_slowdown"]
# throttle column: "" (none) | "sw" (thermal clock reduction, normal at the limit) | "hw" (power brake / hw thermal)


def read_gpus(smi: str) -> list[dict]:
    out = subprocess.run([smi, f"--query-gpu={','.join(FIELDS)}", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=10).stdout
    rows = []
    for line in out.strip().splitlines():
        v = [x.strip() for x in line.split(",")]
        if len(v) < len(FIELDS):
            continue
        def num(x):
            try:
                return float(x)
            except ValueError:
                return float("nan")
        hw = any(x.lower().startswith("active") for x in v[6:8])
        sw = v[8].lower().startswith("active")
        rows.append({"gpu": int(v[0]), "temp": num(v[1]), "power": num(v[2]), "util": num(v[3]),
                     "fan": num(v[4]), "sm": num(v[5]), "throttle": "hw" if hw else ("sw" if sw else "")})
    return rows


def driver_temps(smi: str) -> dict:
    """The card's own thresholds: {'slowdown': C, 'shutdown': C} (whichever nvidia-smi reports)."""
    out = {}
    try:
        text = subprocess.run([smi, "-q", "-d", "TEMPERATURE"], capture_output=True, text=True, timeout=10).stdout
        for line in text.splitlines():
            for key in ("slowdown", "shutdown"):
                if f"GPU {key.capitalize()} Temp" in line and "C" in line and key not in out:
                    out[key] = float(line.split(":")[1].strip().split()[0])
    except Exception:
        pass
    return out


def cpu_temp() -> float:
    best = float("nan")
    for path in glob.glob("/sys/class/hwmon/hwmon*/temp*_input"):
        try:
            t = int(open(path).read().strip()) / 1000.0
        except Exception:
            continue
        if t > 0 and (best != best or t > best):     # nan-safe max
            best = t
    return best


def trip(reason: str, sessions: list[str], log) -> None:
    log(f"TRIP: {reason}  -> killing load")
    for s in sessions:
        subprocess.run(["tmux", "kill-session", "-t", s], capture_output=True)
    # belt and braces: any straggler training / burn / bench process (pattern written so it never matches itself)
    subprocess.run(["pkill", "-f", "[s]cripts/(train|sft|dpo|gpu_burn|bench_micro|find_micro_batch)\\.py"], capture_output=True)
    subprocess.run(["pkill", "-f", "[t]orch.distributed.run"], capture_output=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=os.path.expanduser("~/MaxGPT/thermal.csv"))
    ap.add_argument("--events", default=os.path.expanduser("~/MaxGPT/thermal_events.log"))
    ap.add_argument("--interval", type=float, default=15.0)
    ap.add_argument("--thermometer", default="8,9", help="idle GPUs whose rise measures the room")
    ap.add_argument("--baseline-samples", type=int, default=8)
    ap.add_argument("--idle-rise", type=float, default=10.0, help="trip when a thermometer GPU rises this much (C)")
    ap.add_argument("--max-temp", type=float, default=None, help="trip temp for any GPU (default: shutdown temp - 5)")
    ap.add_argument("--throttle-secs", type=float, default=0.0,
                    help="trip after this long of continuous thermal throttling (0 = log only: Turing cards in a dense "
                         "chassis throttle as their steady state, which is their own regulator working)")
    ap.add_argument("--max-cpu", type=float, default=92.0)
    ap.add_argument("--kill-sessions", default="", help="comma-separated tmux sessions to kill on a trip")
    ap.add_argument("--smi", default="nvidia-smi", help=argparse.SUPPRESS)      # test hook
    ap.add_argument("--once", action="store_true", help=argparse.SUPPRESS)      # test hook
    args = ap.parse_args()

    therm = [int(x) for x in args.thermometer.split(",") if x.strip()]
    sessions = [s for s in args.kill_sessions.split(",") if s.strip()]
    dt = driver_temps(args.smi)
    if args.max_temp is not None:
        max_temp = args.max_temp
    elif "shutdown" in dt:
        max_temp = dt["shutdown"] - 5.0
    elif "slowdown" in dt:
        max_temp = dt["slowdown"] + 1.0
    else:
        max_temp = 92.0

    def log(msg):
        line = f"{time.strftime('%F %T')} {msg}"
        print(line, flush=True)
        with open(args.events, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    log(f"watch start: max_temp={max_temp:.0f}C (driver slowdown {dt.get('slowdown')} / shutdown {dt.get('shutdown')}), "
        f"idle_rise={args.idle_rise}C on GPUs {therm}, thermal throttle > {args.throttle_secs:.0f}s, "
        f"cpu<{args.max_cpu}C, kill={sessions or 'none'}")
    throttle_since: dict[int, float] = {}
    new = not os.path.exists(args.log)
    baseline: dict[int, list[float]] = {g: [] for g in therm}
    base_val: float | None = None
    tripped = False
    while True:
        try:
            gpus = read_gpus(args.smi)
        except Exception as e:
            log(f"nvidia-smi failed ({type(e).__name__}); retrying")
            time.sleep(args.interval)
            continue
        ct = cpu_temp()
        ts = time.strftime("%F %T")
        with open(args.log, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["time", "gpu", "temp_c", "power_w", "util_pct", "fan_pct", "sm_mhz", "throttle", "cpu_c"])
                new = False
            for g in gpus:
                w.writerow([ts, g["gpu"], g["temp"], g["power"], g["util"], g["fan"], g["sm"], g["throttle"], ct])
        if base_val is None:
            for g in gpus:
                if g["gpu"] in baseline:
                    baseline[g["gpu"]].append(g["temp"])
            if all(len(v) >= args.baseline_samples for v in baseline.values()) and baseline:
                base_val = sum(sum(v) / len(v) for v in baseline.values()) / len(baseline)
                log(f"thermometer baseline {base_val:.1f}C")
        loaded = [g for g in gpus if g["util"] >= 50]
        tmean = (sum(g["temp"] for g in gpus if g["gpu"] in therm) / max(1, len(therm))) if therm else float("nan")
        hottest = max(gpus, key=lambda g: g["temp"]) if gpus else None
        total_w = sum(g["power"] for g in gpus)
        rise = (tmean - base_val) if base_val is not None else float("nan")
        now = time.time()
        for g in gpus:                                   # how long each card has been thermally throttling
            if g["throttle"] == "sw":
                throttle_since.setdefault(g["gpu"], now)
            else:
                throttle_since.pop(g["gpu"], None)
        thr = ",".join(f"{g['gpu']}{g['throttle']}" for g in gpus if g["throttle"]) or "-"
        print(f"{ts} loaded={len(loaded)} hottest=gpu{hottest['gpu'] if hottest else '?'} "
              f"{hottest['temp'] if hottest else float('nan'):.0f}C room(therm)={tmean:.1f}C rise={rise:+.1f} "
              f"total={total_w:.0f}W cpu={ct:.0f}C throttle={thr}", flush=True)
        if not tripped:
            reason = None
            long_thr = ([g for g, t in throttle_since.items() if now - t >= args.throttle_secs]
                        if args.throttle_secs > 0 else [])       # 0 = throttling is logged, never a trip
            if hottest and hottest["temp"] >= max_temp:
                reason = f"gpu{hottest['gpu']} at {hottest['temp']:.0f}C >= {max_temp:.0f}C"
            elif any(g["throttle"] == "hw" for g in gpus):
                reason = "hardware slowdown (power brake / hw thermal) on GPU " + ",".join(str(g["gpu"]) for g in gpus if g["throttle"] == "hw")
            elif long_thr:
                reason = f"thermal throttling for over {args.throttle_secs:.0f}s on GPU {','.join(map(str, long_thr))}"
            elif base_val is not None and rise >= args.idle_rise:
                reason = f"thermometer GPUs rose {rise:.1f}C over baseline {base_val:.1f}C (room heating)"
            elif ct == ct and ct >= args.max_cpu:
                reason = f"CPU sensor {ct:.0f}C >= {args.max_cpu:.0f}C"
            if reason:
                trip(reason, sessions, log)
                tripped = True
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
