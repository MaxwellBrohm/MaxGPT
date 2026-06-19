"""MaxGPT-Ultra mission-control dashboard (backend).

Supervises training as a subprocess with pause/play, and streams the live terminal +
metrics to the browser. Run on the training box:

  python gui/server.py --config configs/ultra.yaml --data data/shards --out runs/ultra
  # then open http://127.0.0.1:8800

Pause writes a STOP file into the run dir; the trainer notices it, checkpoints, and exits
(freeing VRAM so you can game). Play removes it and relaunches train.py, which
auto-resumes from the latest checkpoint. Cross-platform (no signals needed).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import threading
from collections import deque

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


class Supervisor:
    """Manages one training subprocess: start / pause (checkpoint + exit) / stop, with a
    ring buffer of terminal lines and a reader of the run's metrics.jsonl."""

    def __init__(self, cmd: list[str], run_out: str, stop_file: str, data_path: str | None = None):
        self.cmd = cmd
        self.run_out = run_out
        self.stop_file = stop_file
        self.data_path = data_path
        self.proc: subprocess.Popen | None = None
        self.log: deque[tuple[int, str]] = deque(maxlen=4000)
        self._seq = 0
        self.status = "idle"
        self._lock = threading.Lock()

    def _append(self, text: str) -> None:
        self._seq += 1
        self.log.append((self._seq, text))

    def _read_stdout(self, proc: subprocess.Popen) -> None:
        for line in iter(proc.stdout.readline, ""):
            self._append(line.rstrip("\n"))
        try:
            proc.stdout.close()
        except Exception:
            pass

    def start(self) -> bool:
        with self._lock:
            if self.running():
                return False
            if self.data_path and not os.path.exists(os.path.join(self.data_path, "meta.json")):
                self._append("$ start requested")
                self._append(f"  ✗ ERROR: model / data not found  (no shards at '{self.data_path}').")
                self._append("  Nothing has been trained on this machine yet.")
                self._append("  Run  python scripts/prepare_data.py  to build the data, then press play again.")
                self.status = "error"
                return False
            for p in (self.stop_file,):
                try:
                    os.remove(p)
                except OSError:
                    pass
            os.makedirs(self.run_out, exist_ok=True)
            self._append("$ " + " ".join(self.cmd))
            self.proc = subprocess.Popen(self.cmd, stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT, text=True, bufsize=1, cwd=ROOT)
            threading.Thread(target=self._read_stdout, args=(self.proc,), daemon=True).start()
            self.status = "running"
            return True

    def pause(self) -> bool:
        with self._lock:
            if not self.running():
                self.status = "paused"
                return False
            open(self.stop_file, "w").close()
            self._append("[gui] pause requested: checkpointing and stopping ...")
            proc = self.proc
        try:
            proc.wait(timeout=600)
        except subprocess.TimeoutExpired:
            proc.terminate()
        try:
            os.remove(self.stop_file)
        except OSError:
            pass
        self.status = "paused"
        self._append("[gui] paused (VRAM freed). Play to resume.")
        return True

    def stop(self) -> None:
        with self._lock:
            if self.running():
                self.proc.terminate()
            self.status = "stopped"
            self._append("[gui] stopped.")

    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def lines_since(self, seq: int) -> tuple[list[str], int]:
        items = [(s, t) for (s, t) in list(self.log) if s > seq]
        last = items[-1][0] if items else seq
        return [t for _, t in items], last

    def metrics_tail(self, n: int = 3000) -> list[dict]:
        path = os.path.join(self.run_out, "metrics.jsonl")
        out: list[dict] = []
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    try:
                        out.append(json.loads(line))
                    except Exception:
                        pass
        return out[-n:]

    def snapshot(self) -> dict:
        if self.status == "running" and not self.running():
            self.status = "exited"
        return {"status": self.status, "running": self.running()}


SUP: Supervisor | None = None
app = FastAPI()


@app.get("/")
def index() -> HTMLResponse:
    with open(os.path.join(HERE, "static", "index.html")) as f:
        return HTMLResponse(f.read())


@app.post("/api/start")
def api_start():
    return {"ok": SUP.start()}


@app.post("/api/pause")
def api_pause():
    return {"ok": SUP.pause()}


@app.post("/api/stop")
def api_stop():
    SUP.stop()
    return {"ok": True}


@app.get("/api/status")
def api_status():
    return SUP.snapshot()


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    last_seq = 0
    try:
        lines, last_seq = SUP.lines_since(0)
        await websocket.send_json({"type": "log", "lines": lines})
        await websocket.send_json({"type": "metrics", "data": SUP.metrics_tail()})
        await websocket.send_json({"type": "status", **SUP.snapshot()})
        while True:
            await asyncio.sleep(0.6)
            lines, last_seq = SUP.lines_since(last_seq)
            if lines:
                await websocket.send_json({"type": "log", "lines": lines})
            await websocket.send_json({"type": "metrics", "data": SUP.metrics_tail()})
            await websocket.send_json({"type": "status", **SUP.snapshot()})
    except WebSocketDisconnect:
        pass


def build_supervisor(config: str, data: str, out: str) -> Supervisor:
    stop_file = os.path.join(out, "STOP")
    cmd = [sys.executable, os.path.join("scripts", "train.py"),
           "--config", config, "--data", data, "--out", out, "--stop-file", stop_file]
    return Supervisor(cmd, out, stop_file, data_path=data)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ultra.yaml")
    ap.add_argument("--data", default="data/shards")
    ap.add_argument("--out", default="runs/ultra")
    ap.add_argument("--port", type=int, default=8800)
    args = ap.parse_args()
    global SUP
    SUP = build_supervisor(args.config, args.data, args.out)
    print(f"[gui] dashboard at http://127.0.0.1:{args.port}  (run dir: {args.out})")
    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
