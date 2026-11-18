"""Watch the gradient norm for a sustained rise above its own long-run level.

Why this exists: fused bf16 attention can go wrong late in training without a single NaN. In
arXiv 2609.34272 a 450M model trained normally for 25B tokens, then its gradient norm rose about a
thousandfold and it finished 0.2 nats worse than the same run with fp32 attention. The loss alone
does not show it in time. Their runs had QK-norm off (ours bounds the quantity that grows), so
this is insurance, not an expected event.

Two levels, both judged against the FLOOR: the lowest value the baseline (the median of the last
`window` normal steps) has taken so far.
  alert  g > alert_factor x floor for alert_patience steps in a row  -> log it, keep training
  halt   g > halt_factor  x floor for halt_patience  steps in a row  -> stop cleanly, so a
         multi-day unattended run does not spend its GPU-hours training a corrupted model
The floor, not the moving baseline, is the reference because a blow-up can be slow: a norm that
climbs a fraction of a percent per step drags any moving median along with it and would never
look high against it. The watch is re-seeded from metrics.jsonl after a restart (a 48-hour job
chain restarts often).
"""
from __future__ import annotations

import json
import math
import os
import statistics
from collections import deque


class GradWatch:
    def __init__(self, alert_factor: float = 10.0, alert_patience: int = 20, halt_factor: float = 100.0,
                 halt_patience: int = 200, window: int = 20000, min_baseline: int = 50):
        self.alert_factor, self.alert_patience = float(alert_factor), int(alert_patience)
        self.halt_factor, self.halt_patience = float(halt_factor), int(halt_patience)
        self.min_baseline = int(min_baseline)
        self.base: deque[float] = deque(maxlen=int(window))
        self.floor: float | None = None                  # lowest baseline seen: the reference level
        self.high_alert = self.high_halt = 0
        self.alerted = self.halted = False

    def baseline(self) -> float | None:
        return statistics.median(self.base) if len(self.base) >= self.min_baseline else None

    def seed(self, values) -> None:
        """Start from earlier normal gradient norms (after a restart)."""
        good = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(v) and v > 0]
        self.base.extend(good)
        for i in range(0, len(good) - self.min_baseline + 1, max(1, self.min_baseline)):   # floor from chunk medians
            m = statistics.median(good[i:i + self.min_baseline])
            self.floor = m if self.floor is None else min(self.floor, m)

    def seed_from_metrics(self, path: str, last: int = 4000) -> int:
        """Seed from the grad_norm column of a metrics.jsonl; returns how many values were used."""
        if not os.path.exists(path):
            return 0
        vals = deque(maxlen=last)
        with open(path, encoding="utf-8") as f:
            for line in f:
                if '"grad_norm"' not in line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if not r.get("event") and not r.get("diverged"):
                    vals.append(r.get("grad_norm"))
        self.seed(vals)
        return len(self.base)

    def feed(self, g: float) -> str | None:
        """One step's pre-clip gradient norm. Returns "alert" or "halt" the first time each level
        is crossed (alert re-arms once the norm is back to normal), else None."""
        if not math.isfinite(g):
            return None                                   # non-finite steps are the divergence guard's job
        b = self.baseline()
        if b is None:
            self.base.append(g)
            return None
        self.floor = b if self.floor is None else min(self.floor, b)
        ref = self.floor
        is_high = bool(self.alert_factor) and g > self.alert_factor * ref
        if is_high:
            self.high_alert += 1
        else:
            self.high_alert = 0
            self.alerted = False
            self.base.append(g)                           # only normal steps move the baseline
        self.high_halt = self.high_halt + 1 if (self.halt_factor and g > self.halt_factor * ref) else 0
        if self.halt_factor and self.high_halt >= self.halt_patience and not self.halted:
            self.halted = True
            return "halt"
        if is_high and self.high_alert >= self.alert_patience and not self.alerted:
            self.alerted = True
            return "alert"
        return None
