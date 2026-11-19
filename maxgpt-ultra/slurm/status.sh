#!/bin/bash
# Where the run stands, in a few lines. Run from maxgpt-ultra/ on the cluster:  bash slurm/status.sh
cd "$(dirname "$0")/.." || exit 1
[ -f slurm/env.sh ] && source slurm/env.sh
RUN="${MAXGPT_RUN:-runs/pretrain}"
echo "== queue =="; squeue -u "$USER" -o "%.10i %.14j %.8T %.10M %.10l %.20S %R" 2>/dev/null || echo "(squeue not available here)"
echo "== run: $RUN =="
[ -f "$RUN/DONE" ] && echo "DONE: $(cat "$RUN/DONE")"
[ -f "$RUN/GRAD_HALT" ] && echo "GRAD_HALT: $(cat "$RUN/GRAD_HALT")  (see NAIRR.md)"
[ -f "$RUN/ANNEAL_MISSING" ] && echo "ANNEAL_MISSING: $(cat "$RUN/ANNEAL_MISSING")  (the decay-phase shards were not found; see NAIRR.md)"
python3 - "$RUN/metrics.jsonl" <<'PY'
import json, os, sys
p = sys.argv[1]
if not os.path.exists(p):
    print("no metrics yet"); sys.exit()
last = meta = ev = suite = None
alerts = 0
for line in open(p, encoding="utf-8"):
    try:
        r = json.loads(line)
    except Exception:
        continue
    e = r.get("event")
    if e == "meta": meta = r
    elif e == "eval":
        ev = r
        if "suite" in r: suite = r
    elif e in ("grad_alert", "grad_halt"): alerts += 1
    elif not e and "loss" in r: last = r
if last and meta:
    total, tps = meta["total_steps"], meta["tokens_per_step"]
    left = (total - last["step"]) * tps / max(1.0, last.get("tok_per_s", 1.0)) / 3600
    print(f"step {last['step']:,} of {total:,} ({100 * last['step'] / total:.1f}%)  loss {last['loss']:.3f}  "
          f"{last.get('tok_per_s', 0):,.0f} tok/s  about {left:.0f} compute-hours left  grad norm {last.get('grad_norm', 0):.3g}")
if ev: print(f"last eval: step {ev['step']:,}  val loss {ev['val_loss']:.3f}" + (f"  max attention logit {max(ev['act_max_logit']):.1f}" if ev.get("act_max_logit") else ""))
if suite: print(f"last benchmark suite: step {suite['step']:,}  average {suite['suite_avg']:.3f}")
if alerts: print(f"gradient-watch events: {alerts} (grep grad_ {p})")
PY
ls "$RUN/checkpoints" 2>/dev/null | grep -c "milestone_" | xargs -I{} echo "milestone checkpoints: {}"
