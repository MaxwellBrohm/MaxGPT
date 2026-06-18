"""Backend smoke test for the mission-control GUI.

Uses a dummy "training" subprocess (prints lines, writes metrics.jsonl, respects the stop
file) so we can exercise the supervisor mechanics without a GPU: serve the dashboard,
start, capture terminal + metrics, pause via the stop-file (process checkpoints + exits),
and stream over the websocket. The real training integration runs on the 5070 box.

Run from maxgpt-ultra/:  ../venv/bin/python scripts/test_gui.py
"""
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # maxgpt-ultra/

from fastapi.testclient import TestClient

import gui.server as server

# stands in for scripts/train.py: emits logs + metrics, exits when the stop file appears
DUMMY = r'''
import sys, os, time, json
run_out, stop_file = sys.argv[1], sys.argv[2]
os.makedirs(run_out, exist_ok=True)
mp = os.path.join(run_out, "metrics.jsonl")
print("dummy training started", flush=True)
i = 0
while True:
    if os.path.exists(stop_file):
        print("stop file seen; checkpointing and exiting", flush=True)
        break
    i += 1
    print(f"hello step {i}", flush=True)
    with open(mp, "a") as f:
        f.write(json.dumps({"step": i, "loss": 5.0 / (1 + i), "lr": 1e-3,
                            "tok_per_s": 1000, "grad_norm": 1.0}) + "\n")
    time.sleep(0.1)
print("dummy exited", flush=True)
'''


def main() -> None:
    print("=" * 72)
    print("MaxGPT-Ultra GUI backend smoke test")
    print("=" * 72)

    run_out = tempfile.mkdtemp(prefix="mgu_gui_")
    stop_file = os.path.join(run_out, "STOP")
    cmd = [sys.executable, "-c", DUMMY, run_out, stop_file]
    server.SUP = server.Supervisor(cmd, run_out, stop_file)
    client = TestClient(server.app)

    print("\n[1] dashboard + status served")
    r = client.get("/")
    assert r.status_code == 200 and "MaxGPT-Ultra" in r.text and "mission control" in r.text
    assert client.get("/api/status").json()["status"] == "idle"
    print("  GET / returns the dashboard; status=idle ✓")

    print("\n[2] start -> running, terminal + metrics captured")
    assert client.post("/api/start").json()["ok"] is True
    time.sleep(1.6)
    assert server.SUP.running(), "process not running after start"
    lines, _ = server.SUP.lines_since(0)
    assert any("hello step" in l for l in lines), "terminal output not captured"
    assert len(server.SUP.metrics_tail()) > 0, "metrics.jsonl not picked up"
    print(f"  running; captured {len(lines)} log lines and {len(server.SUP.metrics_tail())} metric rows ✓")

    print("\n[3] websocket streams log/metrics/status")
    with client.websocket_connect("/ws") as wsc:
        first = wsc.receive_json()
        assert first["type"] == "log"
        got = {first["type"]}
        for _ in range(3):
            got.add(wsc.receive_json()["type"])
    assert "metrics" in got and "status" in got
    print(f"  websocket delivered {sorted(got)} ✓")

    print("\n[4] pause -> stop file -> process checkpoints + exits, VRAM freed")
    assert client.post("/api/pause").json()["ok"] is True
    assert not server.SUP.running(), "process still running after pause"
    assert server.SUP.snapshot()["status"] == "paused"
    lines, _ = server.SUP.lines_since(0)
    assert any("exiting" in l for l in lines), "did not see graceful exit"
    print("  paused: process exited cleanly on the stop-file ✓")

    print("\n[5] play again -> fresh process runs")
    assert client.post("/api/start").json()["ok"] is True
    time.sleep(0.6)
    assert server.SUP.running()
    client.post("/api/stop")
    print("  restarted, then stopped ✓")

    print("\n" + "=" * 72)
    print("ALL CHECKS PASSED ✅")
    print("=" * 72)


if __name__ == "__main__":
    main()
