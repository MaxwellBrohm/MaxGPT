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

Pause/resume mode (the real run): with --pause-cmd / --resume-cmd set, a trip runs the pause
command (the dashboard's pause: the trainer checkpoints and exits) instead of killing, waits for
the load to disappear (falls back to killing after --pause-grace seconds), and once the thermometer
is back within --resume-rise C of baseline and every card is under --resume-max-temp C it runs the
resume command. More than --max-trips-per-hour trips means something is wrong: it stops resuming
and leaves the run paused.
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


def cpu_temp(override: str | None = None) -> float:
    best = float("nan")
    paths = [override] if override else glob.glob("/sys/class/hwmon/hwmon*/temp*_input")
    for path in paths:
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
    ap.add_argument("--baseline-after-load", type=float, default=0.0,
                    help="take the baseline only after the load has been on for this many seconds (lets the "
                         "neighbour-heat settle first, so the rise measures the ROOM)")
    ap.add_argument("--idle-rise", type=float, default=10.0, help="trip when a thermometer GPU rises this much (C)")
    ap.add_argument("--max-temp", type=float, default=None, help="trip temp for any GPU (default: shutdown temp - 5)")
    ap.add_argument("--throttle-secs", type=float, default=0.0,
                    help="trip after this long of continuous thermal throttling (0 = log only: Turing cards in a dense "
                         "chassis throttle as their steady state, which is their own regulator working)")
    ap.add_argument("--max-cpu", type=float, default=92.0)
    ap.add_argument("--cpu-rise", type=float, default=0.0,
                    help="trip when the CPU sensor rises this much over its settled baseline (a room reading that is "
                         "not a GPU; 0 = off)")
    ap.add_argument("--state-file", default=os.path.expanduser("~/MaxGPT/thermal_state"),
                    help="the watchdog's state (armed|pausing|cooling|stopped) is written here every sample")
    ap.add_argument("--cpu-file", default=None, help=argparse.SUPPRESS)   # test hook
    ap.add_argument("--kill-sessions", default="", help="comma-separated tmux sessions to kill on a trip")
    ap.add_argument("--pause-cmd", default="", help="run this instead of killing on a trip (e.g. curl -s -X POST http://127.0.0.1:8800/api/pause)")
    ap.add_argument("--resume-cmd", default="", help="run this when cooled down again (e.g. curl -s -X POST http://127.0.0.1:8800/api/start)")
    ap.add_argument("--pause-grace", type=float, default=300.0, help="seconds to wait for the pause to take before killing")
    ap.add_argument("--resume-rise", type=float, default=2.0, help="resume when the thermometer rise is back under this (C)")
    ap.add_argument("--resume-max-temp", type=float, default=60.0, help="... and every GPU is under this (C)")
    ap.add_argument("--max-trips-per-hour", type=int, default=3)
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
    state = "armed"                 # armed | pausing | cooling | stopped
    load_since: float | None = None
    cpu_base_samples: list[float] = []
    cpu_base: float | None = None
    trip_times: list[float] = []
    paused_at = 0.0

    def run_cmd(cmd: str, what: str) -> None:
        try:
            r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
            log(f"{what}: {cmd!r} -> rc={r.returncode} {r.stdout.strip()[:120]}")
        except Exception as e:
            log(f"{what}: {cmd!r} failed ({type(e).__name__})")
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
        ct = cpu_temp(args.cpu_file)
        ts = time.strftime("%F %T")
        with open(args.log, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["time", "gpu", "temp_c", "power_w", "util_pct", "fan_pct", "sm_mhz", "throttle", "cpu_c"])
                new = False
            for g in gpus:
                w.writerow([ts, g["gpu"], g["temp"], g["power"], g["util"], g["fan"], g["sm"], g["throttle"], ct])
        loaded_now = [g for g in gpus if g["util"] >= 50]
        if args.baseline_after_load > 0:
            load_since = (load_since or time.time()) if loaded_now else None
            settled = load_since is not None and time.time() - load_since >= args.baseline_after_load
        else:
            settled = True
        if settled and cpu_base is None and ct == ct:
            cpu_base_samples.append(ct)
            if len(cpu_base_samples) >= args.baseline_samples:
                cpu_base = sum(cpu_base_samples) / len(cpu_base_samples)
                log(f"cpu baseline {cpu_base:.1f}C")
        if base_val is None and settled:
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
        cpu_rise = (ct - cpu_base) if (cpu_base is not None and ct == ct) else float("nan")
        try:
            with open(args.state_file, "w", encoding="utf-8") as sf:
                sf.write(f"{state} {ts} rise={rise:+.1f} cpu_rise={cpu_rise:+.1f} loaded={len(loaded_now)}\n")
        except Exception:
            pass
        now = time.time()
        for g in gpus:                                   # how long each card has been thermally throttling
            if g["throttle"] == "sw":
                throttle_since.setdefault(g["gpu"], now)
            else:
                throttle_since.pop(g["gpu"], None)
        thr = ",".join(f"{g['gpu']}{g['throttle']}" for g in gpus if g["throttle"]) or "-"
        print(f"{ts} loaded={len(loaded)} hottest=gpu{hottest['gpu'] if hottest else '?'} "
              f"{hottest['temp'] if hottest else float('nan'):.0f}C room(therm)={tmean:.1f}C rise={rise:+.1f} "
              f"total={total_w:.0f}W cpu={ct:.0f}C({cpu_rise:+.1f}) throttle={thr} [{state}]", flush=True)
        if state == "pausing":                       # waiting for the trainer to checkpoint and exit
            if not loaded:
                state = "cooling"
                log("load is gone; waiting for the cards and the room to cool")
            elif now - paused_at >= args.pause_grace:
                trip("pause did not take within the grace period", sessions, log)
                state = "cooling"
        elif state == "cooling":
            if not loaded and (base_val is None or rise <= args.resume_rise) and hottest and hottest["temp"] <= args.resume_max_temp:
                recent = [t for t in trip_times if now - t < 3600]
                if len(recent) >= args.max_trips_per_hour:
                    log(f"{len(recent)} trips in the last hour: NOT resuming (something needs a human)")
                    state = "stopped"
                else:
                    log(f"cooled down (rise {rise:+.1f}C, hottest {hottest['temp']:.0f}C): resuming")
                    run_cmd(args.resume_cmd, "resume")
                    state = "armed"
                    tripped = False
        if not tripped and state == "armed":
            reason = None
            long_thr = ([g for g, t in throttle_since.items() if now - t >= args.throttle_secs]
                        if args.throttle_secs > 0 else [])       # 0 = throttling is logged, never a trip
            if hottest and hottest["temp"] >= max_temp:
                reason = f"gpu{hottest['gpu']} at {hottest['temp']:.0f}C >= {max_temp:.0f}C"
            elif any(g["throttle"] == "hw" for g in gpus):
                reason = "hardware slowdown (power brake / hw thermal) on GPU " + ",".join(str(g["gpu"]) for g in gpus if g["throttle"] == "hw")
            elif long_thr:
                reason = f"thermal throttling for over {args.throttle_secs:.0f}s on GPU {','.join(map(str, long_thr))}"
            elif args.cpu_rise > 0 and cpu_base is not None and cpu_rise >= args.cpu_rise:
                reason = f"CPU sensor rose {cpu_rise:.1f}C over its settled baseline {cpu_base:.1f}C (room heating)"
            elif base_val is not None and rise >= args.idle_rise:
                reason = f"thermometer GPUs rose {rise:.1f}C over baseline {base_val:.1f}C (room heating)"
            elif ct == ct and ct >= args.max_cpu:
                reason = f"CPU sensor {ct:.0f}C >= {args.max_cpu:.0f}C"
            if reason:
                tripped = True
                trip_times.append(now)
                if args.pause_cmd and args.resume_cmd:
                    log(f"TRIP: {reason}  -> pausing the run")
                    run_cmd(args.pause_cmd, "pause")
                    state, paused_at = "pausing", now
                else:
                    trip(reason, sessions, log)
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
