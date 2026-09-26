"""Backend smoke test for the pipeline GUI.

Uses dummy stage processes (print + write metrics + honor the stop-file) to verify the
orchestration without a GPU: stages auto-advance in order, pause checkpoints+halts the
current stage, the REST endpoints serve per-stage data, and the dashboard is served.

Run from maxgpt-ultra/:  ../venv/bin/python scripts/test_gui.py
"""
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # maxgpt-ultra/

from fastapi.testclient import TestClient

import gui.server as server
from gui.server import Stage, Pipeline

DUMMY = r'''
import sys, os, time, json
run_out, mode = sys.argv[1], sys.argv[2]
os.makedirs(run_out, exist_ok=True)
mp = os.path.join(run_out, "metrics.jsonl"); stop = os.path.join(run_out, "STOP")
print(f"dummy {mode} start", flush=True)
i = 0
while True:
    if os.path.exists(stop): print("stop seen; checkpoint + exit", flush=True); break
    i += 1
    print(f"step {i}", flush=True)
    with open(mp, "a") as f: f.write(json.dumps({"step": i, "loss": 5.0/(1+i), "lr": 1e-3}) + "\n")
    if mode == "quick" and i >= 3: break
    time.sleep(0.15)
print("dummy done", flush=True)
'''


def dummy(run_out, mode):
    return Stage(mode + "_" + os.path.basename(run_out), mode.title(),
                 (lambda r=run_out, m=mode: [sys.executable, "-c", DUMMY, r, m]), run_out)


def wait_for(fn, timeout=8.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if fn():
            return True
        time.sleep(0.1)
    return False


def main():
    print("=" * 72)
    print("MaxGPT-Ultra pipeline GUI smoke test")
    print("=" * 72)
    d = tempfile.mkdtemp(prefix="mgu_pipe_")

    print("\n[1] stages auto-advance in order")
    a, b = dummy(os.path.join(d, "a"), "quick"), dummy(os.path.join(d, "b"), "quick")
    server.PIPE = Pipeline([a, b])
    client = TestClient(server.app)
    assert client.get("/").status_code == 200
    assert client.post("/api/start").json()["ok"] is True
    assert wait_for(lambda: a.status == "done" and b.status == "done", 12), \
        f"stages did not both finish (a={a.status}, b={b.status})"
    print(f"  stage A -> done, then auto-started B -> done ✓")

    print("\n[2] per-stage REST data")
    sa = client.get("/api/stage/" + a.key).json()
    assert any("step" in l for l in sa["log"]) and len(sa["metrics"]) > 0
    pipe = client.get("/api/pipeline").json()
    assert [s["status"] for s in pipe["stages"]] == ["done", "done"]
    print(f"  /api/stage returns log+metrics; /api/pipeline shows both done ✓")

    print("\n[3] pause halts the current stage (checkpoint + exit)")
    longst = dummy(os.path.join(d, "long"), "loop")
    server.PIPE = Pipeline([longst])
    client.post("/api/start")
    assert wait_for(lambda: server.PIPE.running(), 5), "long stage never started"
    time.sleep(0.4)
    assert client.post("/api/pause").json()["ok"] is True
    assert wait_for(lambda: longst.status == "paused", 8), f"did not pause (status={longst.status})"
    assert not server.PIPE.running()
    print("  pause -> stage checkpointed and halted (status=paused) ✓")

    print("\n[4] stop aborts")
    st = dummy(os.path.join(d, "s"), "loop")
    server.PIPE = Pipeline([st])
    client.post("/api/start")
    assert wait_for(lambda: server.PIPE.running(), 5)
    client.post("/api/stop")
    assert wait_for(lambda: not server.PIPE.running(), 5)
    print("  stop -> pipeline aborted ✓")

    print("\n" + "=" * 72)

    print("\n[5] the HOLD file stops every launch (start refused, reason visible), removing it allows again")
    import datetime
    from gui import gate
    hold = os.path.join(d, "HOLD")
    gate.HOLD_FILE, gate.RUN_HOURS = hold, ""
    open(hold, "w").close()
    h5 = dummy(os.path.join(d, "held"), "quick")
    server.PIPE = Pipeline([h5])
    assert client.post("/api/start").json()["ok"] is False, "start must be refused while HOLD exists"
    assert not server.PIPE.running() and h5.status == "paused"
    assert any("not starting" in l and "HOLD" in l for l in h5.lines()), h5.lines()[-3:]
    assert "HOLD" in client.get("/api/pipeline").json()["gate"]
    os.remove(hold)
    assert client.post("/api/start").json()["ok"] is True
    assert wait_for(lambda: h5.status == "done", 12), h5.status
    assert client.get("/api/pipeline").json()["gate"] == "ok"
    print("  refused with the HOLD reason, ran to completion once the file was gone ✓")

    print("\n[6] allowed hours: outside the window nothing starts; inside it does; window arithmetic")
    assert gate.in_window("22-07", 23) and gate.in_window("22-07", 3) and not gate.in_window("22-07", 12)
    assert gate.in_window("9-17", 9) and not gate.in_window("9-17", 17) and gate.in_window("", 5) and gate.in_window("5-5", 5)
    now_h = datetime.datetime.now().hour
    gate.RUN_HOURS = f"{(now_h + 2) % 24}-{(now_h + 3) % 24}"          # a window that excludes now
    h6 = dummy(os.path.join(d, "hours"), "quick")
    server.PIPE = Pipeline([h6])
    assert client.post("/api/start").json()["ok"] is False and "hours" in client.get("/api/pipeline").json()["gate"]
    gate.RUN_HOURS = f"{now_h}-{(now_h + 1) % 24}"                      # a window that includes now
    assert client.post("/api/start").json()["ok"] is True
    assert wait_for(lambda: h6.status == "done", 12), h6.status
    gate.RUN_HOURS = ""
    print(f"  hour {now_h}: refused outside the window, ran inside it ✓")

    print("ALL CHECKS PASSED ✅")
    print("=" * 72)


if __name__ == "__main__":
    main()
