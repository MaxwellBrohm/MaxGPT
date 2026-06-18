"""Warmup-Stable-Decay (WSD) learning-rate schedule.

Three phases:
  - warmup: linear 0 -> max_lr over `warmup_steps` (lets the optimizer's moment
    estimates settle before taking big steps),
  - stable: constant max_lr for the long middle (this is the phase we can extend or
    pause/resume freely, since nothing about it depends on the total length),
  - decay: cosine from max_lr down to min_lr over the last `decay_frac` of steps
    (the end-of-run loss drop, taken only when we decide to finish).

Unlike a single cosine curve, WSD does not need the final step count baked in up front,
which is exactly why it fits an open-ended, pause-heavy run.
"""
import math


def wsd_lr(step: int, *, total_steps: int, warmup_steps: int, decay_frac: float,
           max_lr: float, min_lr_ratio: float = 0.0) -> float:
    warmup_steps = max(1, warmup_steps)
    decay_steps = max(1, int(total_steps * decay_frac))
    stable_end = max(warmup_steps, total_steps - decay_steps)

    if step < warmup_steps:                       # linear warmup
        return max_lr * (step + 1) / warmup_steps
    if step < stable_end:                          # constant
        return max_lr
    t = min(1.0, (step - stable_end) / max(1, total_steps - stable_end))  # 0 -> 1
    min_lr = max_lr * min_lr_ratio
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * t))  # cosine decay
