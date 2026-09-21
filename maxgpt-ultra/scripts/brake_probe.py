"""Find the box's electrical ceiling safely: burn N cards at full draw while sampling every second
for the hardware POWER BRAKE / HW slowdown flags, and stop the burn the moment the brake is
sustained (or a card reaches --max-temp). Reports brake samples (transient at start-up vs
sustained), total power, and per-card clocks.

  python scripts/brake_probe.py --gpus 6,7 --seconds 150
"""
import argparse
import os
import subprocess
import sys
import time

FIELDS = "index,power.draw,temperature.gpu,clocks.sm,clocks_throttle_reasons.hw_slowdown," \
         "clocks_throttle_reasons.hw_power_brake_slowdown,clocks_throttle_reasons.sw_power_cap," \
         "clocks_throttle_reasons.sw_thermal_slowdown"


def sample():
    out = subprocess.run(["nvidia-smi", f"--query-gpu={FIELDS}", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=10).stdout
    rows = []
    for line in out.strip().splitlines():
        v = [x.strip() for x in line.split(",")]
        if len(v) < 8:
            continue
        act = lambda x: x.lower().startswith("active")
        rows.append({"gpu": int(v[0]), "w": float(v[1]), "t": float(v[2]), "sm": float(v[3]),
                     "hw": act(v[4]), "brake": act(v[5]), "pcap": act(v[6]), "therm": act(v[7])})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", required=True)
    ap.add_argument("--seconds", type=int, default=150)
    ap.add_argument("--max-temp", type=float, default=92.0)
    ap.add_argument("--sustained", type=int, default=5, help="consecutive brake samples that stop the burn")
    ap.add_argument("--log", default=os.path.expanduser("~/MaxGPT/brake_probe.csv"))
    args = ap.parse_args()
    burn = subprocess.Popen([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "gpu_burn.py"),
                             "--gpus", args.gpus, "--minutes", str(args.seconds / 60 + 2)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    loaded = {int(g) for g in args.gpus.split(",")}
    consec, brake_total, hw_total, peak_w, stop_reason = 0, 0, 0, 0.0, None
    first_brake_t = None
    t0 = time.time()
    try:
        with open(args.log, "a", encoding="utf-8") as f:
            for i in range(args.seconds):
                rows = sample()
                now = time.time() - t0
                f.write(f"{time.strftime('%F %T')},{args.gpus}," + ";".join(
                    f"{r['gpu']}:{r['w']:.0f}W:{r['t']:.0f}C:{r['sm']:.0f}:{'B' if r['brake'] else ''}{'H' if r['hw'] else ''}{'P' if r['pcap'] else ''}{'T' if r['therm'] else ''}"
                    for r in rows) + "\n")
                total_w = sum(r["w"] for r in rows)
                peak_w = max(peak_w, total_w)
                braking = [r["gpu"] for r in rows if r["brake"] or r["hw"]]
                brake_total += sum(1 for r in rows if r["brake"])
                hw_total += sum(1 for r in rows if r["hw"])
                consec = consec + 1 if braking else 0
                if braking and first_brake_t is None:
                    first_brake_t = now
                if i % 10 == 0 or braking:
                    ld = [r for r in rows if r["gpu"] in loaded]
                    print(f"t+{now:5.0f}s total={total_w:5.0f}W loaded: " +
                          " ".join(f"g{r['gpu']}={r['w']:.0f}W/{r['t']:.0f}C/{r['sm']:.0f}MHz{'!BRAKE' if r['brake'] else ''}{'!HW' if r['hw'] and not r['brake'] else ''}{'/pcap' if r['pcap'] else ''}{'/therm' if r['therm'] else ''}" for r in ld),
                          flush=True)
                hot = max((r["t"] for r in rows), default=0)
                if consec >= args.sustained:
                    stop_reason = f"power brake / HW slowdown sustained {consec}s on GPU {braking}"
                    break
                if hot >= args.max_temp:
                    stop_reason = f"a card reached {hot:.0f}C"
                    break
                time.sleep(max(0.0, 1.0 - (time.time() - t0 - now)))
    finally:
        burn.terminate()
        try:
            burn.wait(timeout=10)
        except subprocess.TimeoutExpired:
            burn.kill()
        subprocess.run(["pkill", "-f", "[g]pu_burn.py"], capture_output=True)
    verdict = stop_reason or "no sustained brake"
    print(f"RESULT gpus={args.gpus} n={len(loaded)} peak_total={peak_w:.0f}W brake_samples={brake_total} hw_samples={hw_total} "
          f"first_brake={'none' if first_brake_t is None else f'{first_brake_t:.0f}s'} -> {verdict}", flush=True)


if __name__ == "__main__":
    main()
