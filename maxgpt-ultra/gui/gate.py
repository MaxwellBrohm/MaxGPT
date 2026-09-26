"""One gate for everything that starts or resumes training on the box: the dashboard's start (which
is also what the keeper's and the watchdog's "play" call), the dashboard's auto-advance to the next
stage, and the keeper's restarts.

  HOLD file   MAXGPT_HOLD_FILE, default ~/MaxGPT/HOLD. While it exists nothing starts and nothing
              is auto-restarted, and the keeper pauses a run that is still going. `touch ~/MaxGPT/HOLD`
              is the off switch for an administrator or for Max; `rm` it to allow runs again.
              Written after 2026-09-25, when the keeper could not tell an admin stop from a crash
              and would have brought the run back within 10 minutes of IS killing it.
  RUN_HOURS   MAXGPT_RUN_HOURS, e.g. "22-07": training may only run inside this window of local
              hours (22:00 to 06:59, wrapping midnight); outside it the keeper pauses the run and
              nothing starts. Empty means always.

Standard library only, so the bash keeper can call it with the system python:
  python3 gui/gate.py   -> prints "ok" (exit 0) or the reason (exit 1)
"""
from __future__ import annotations

import datetime
import os
import sys

HOLD_FILE = os.environ.get("MAXGPT_HOLD_FILE") or os.path.expanduser("~/MaxGPT/HOLD")
RUN_HOURS = os.environ.get("MAXGPT_RUN_HOURS", "")


def in_window(spec: str, hour: int) -> bool:
    """'22-07' is 22:00 to 06:59 (wraps midnight); '9-17' is 09:00 to 16:59; '' or 'a-a' is always."""
    spec = (spec or "").strip()
    if not spec:
        return True
    a, b = (int(x) for x in spec.split("-"))
    if a == b:
        return True
    return a <= hour < b if a < b else (hour >= a or hour < b)


def allowed(now: datetime.datetime | None = None, hold_file: str | None = None,
            run_hours: str | None = None) -> tuple[bool, str]:
    """(True, 'ok') when a run may start or continue; otherwise (False, why)."""
    hold_file = HOLD_FILE if hold_file is None else hold_file
    run_hours = RUN_HOURS if run_hours is None else run_hours
    if os.path.exists(hold_file):
        return False, f"HOLD file present ({hold_file}); remove it to allow runs"
    hour = (now or datetime.datetime.now()).hour
    if not in_window(run_hours, hour):
        return False, f"outside the allowed hours {run_hours} (it is {hour:02d}:xx)"
    return True, "ok"


if __name__ == "__main__":
    ok, why = allowed()
    print(why)
    sys.exit(0 if ok else 1)
