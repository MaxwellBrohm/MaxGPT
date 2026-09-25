"""MaxGPT-Ultra mission-control: a full-pipeline reactor.

One play button runs the whole pipeline in order -- data -> pretrain -> SFT -> DPO --
each stage as its own subprocess, auto-advancing when the previous one finishes. A final
"chat" stage lets you talk to the trained model (RAG). Pause checkpoints the current stage
and frees VRAM; play resumes it; stop aborts. Each stage keeps its own terminal log +
metrics, so you can click back/forward through the stepper to see how each one did.

Run on the 5070 box:  python gui/server.py   (then open http://127.0.0.1:8800)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
from collections import deque

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from cfg import load_yaml   # torch-free config reader (handles `extends:`)


def visible_gpu_count() -> int:
    """How many GPUs a stage may use: CUDA_VISIBLE_DEVICES if set (the etiquette knob on a
    shared box), else every card nvidia-smi lists; 1 when there is no NVIDIA GPU."""
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    if vis is not None:
        return max(1, len([v for v in vis.split(",") if v.strip() != ""]))
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=5)
        return max(1, len([l for l in out.stdout.splitlines() if l.startswith("GPU ")]))
    except Exception:
        return 1


def resolve_gpus(setting) -> int:
    """train.gpus from the config: an int, or 'auto' (= every visible card)."""
    if isinstance(setting, bool) or setting is None:
        return 1
    if isinstance(setting, int):
        return max(1, setting)
    return visible_gpu_count() if str(setting).lower() == "auto" else max(1, int(setting))


def launch_cmd(py, script_and_args, n_gpus, run_out):
    """Single process for 1 GPU; torchrun (one process per GPU, DDP) for more. Rank 0's output
    is what the dashboard shows; every rank's full log lands in <run_out>/ranks/."""
    if n_gpus <= 1:
        return [py] + script_and_args
    return [py, "-m", "torch.distributed.run", "--nnodes=1", "--rdzv-backend=c10d",
            "--rdzv-endpoint=127.0.0.1:0", f"--nproc_per_node={n_gpus}",
            "--redirects", "3", "--tee", "0:3", "--log-dir", os.path.join(run_out, "ranks")] + script_and_args


class Stage:
    def __init__(self, key, label, build, run_out, kind="train", done_when=None):
        self.key = key
        self.label = label
        self.build = build          # () -> argv list (resolved at launch; may raise FriendlyError)
        self.run_out = run_out
        self.kind = kind            # "data" | "train" | "chat"
        self.done_when = done_when  # optional () -> bool: outputs already on disk, so skip this stage
        self.status = "pending"     # pending | running | done | paused | error | stopped | ready
        self.log: deque[tuple[int, str]] = deque(maxlen=4000)
        self._seq = 0

    def append(self, text):
        self._seq += 1
        self.log.append((self._seq, text))

    def lines(self):
        return [t for _, t in self.log]

    def metrics(self):
        path = os.path.join(self.run_out, "metrics.jsonl")
        out = []
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    try:
                        out.append(json.loads(line))
                    except Exception:
                        pass
        return out[-4000:]

    def stop_file(self):
        return os.path.join(self.run_out, "STOP")


class FriendlyError(Exception):
    pass


class Pipeline:
    def __init__(self, stages: list[Stage]):
        self.stages = stages
        self.run_stages = [s for s in stages if s.kind in ("data", "train")]
        self.idx = 0                # index into run_stages (current auto-run stage)
        self.proc: subprocess.Popen | None = None
        self.aborted = False
        self._lock = threading.Lock()

    def sync_from_disk(self):
        """Promote a stage to 'done' when its outputs already exist, so a fresh launch
        skips finished work (mainly: data prep). Only runs while idle; training stages
        have no done_when and instead resume from their checkpoints when re-run."""
        if self.running():
            return
        for s in self.run_stages:
            if s.status == "pending" and s.done_when and s.done_when():
                s.status = "done"

    def current(self) -> Stage | None:
        return self.run_stages[self.idx] if self.idx < len(self.run_stages) else None

    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    # --- subprocess plumbing ---
    def _read(self, st: Stage, proc):
        for line in iter(proc.stdout.readline, ""):
            st.append(line.rstrip("\n"))
        try:
            proc.stdout.close()
        except Exception:
            pass

    def _run(self, i: int):
        st = self.run_stages[i]
        os.makedirs(st.run_out, exist_ok=True)
        try:
            cmd = st.build()
        except FriendlyError as e:
            st.append(f"  ERROR: {e}")
            st.status = "error"
            return
        try:
            os.remove(st.stop_file())
        except OSError:
            pass
        st.append("$ " + " ".join(cmd))
        st.status = "running"
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}     # live logs even through torchrun's pipe
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, bufsize=1, cwd=ROOT, env=env)
        threading.Thread(target=self._read, args=(st, self.proc), daemon=True).start()
        threading.Thread(target=self._wait, args=(i, self.proc), daemon=True).start()

    def _wait(self, i: int, proc):
        rc = proc.wait()
        st = self.run_stages[i]
        if os.path.exists(st.stop_file()):          # paused: checkpointed + exited
            try:
                os.remove(st.stop_file())
            except OSError:
                pass
            st.status = "paused"
            st.append("[gui] paused (VRAM freed). Play to resume.")
            return
        if self.aborted:
            st.status = "stopped"
            return
        if rc == 0:
            st.status = "done"
            st.append("[gui] stage complete.")
            if i + 1 < len(self.run_stages):        # auto-advance to the next stage
                self.idx = i + 1
                self._run(self.idx)
            else:
                # last training stage done -> chat stage becomes ready
                for s in self.stages:
                    if s.kind == "chat":
                        s.status = "ready"
        else:
            st.status = "error"
            st.append(f"[gui] stage exited with code {rc}; pipeline halted.")

    # --- controls ---
    def start(self) -> bool:
        with self._lock:
            if self.running():
                return False
            self.aborted = False
            self.sync_from_disk()        # skip stages whose outputs already exist (e.g. data)
            while self.idx < len(self.run_stages) and self.run_stages[self.idx].status == "done":
                self.idx += 1
            if self.idx >= len(self.run_stages):
                return False
            self._run(self.idx)
            return True

    def pause(self) -> bool:
        st = self.current()
        if not (st and self.running()):
            return False
        open(st.stop_file(), "w").close()           # current stage checkpoints + exits
        return True

    def stop(self):
        self.aborted = True
        if self.running():
            self.proc.terminate()

    def snapshot(self) -> dict:
        self.sync_from_disk()        # reflect already-built artifacts in the UI (data shows done)
        cur = self.current()
        return {"stages": [{"key": s.key, "label": s.label, "status": s.status, "kind": s.kind}
                           for s in self.stages],
                "current": cur.key if cur else None,
                "running": self.running()}


# --------------------------------------------------------------------------- #
PIPE: Pipeline | None = None
CHAT_CFG: dict = {}
app = FastAPI()


@app.get("/")
def index():
    with open(os.path.join(HERE, "static", "index.html"), encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.post("/api/start")
def api_start():
    return {"ok": PIPE.start()}


@app.post("/api/pause")
def api_pause():
    return {"ok": PIPE.pause()}


@app.post("/api/stop")
def api_stop():
    PIPE.stop()
    return {"ok": True}


@app.get("/api/pipeline")
def api_pipeline():
    return PIPE.snapshot()


def _read_gpu() -> dict:
    """GPU stats via nvidia-smi (no extra deps; ships with the NVIDIA driver). On a multi-GPU
    box the cards in CUDA_VISIBLE_DEVICES (all of them if unset) are aggregated: utilization
    averaged, VRAM and power summed, temperature max."""
    try:
        q = "index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw"
        out = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=3)
        rows = [[x.strip() for x in line.split(",")] for line in out.stdout.strip().splitlines() if line.strip()]
        vis = os.environ.get("CUDA_VISIBLE_DEVICES")
        if vis is not None:
            want = {v.strip() for v in vis.split(",") if v.strip() != ""}
            rows = [r for r in rows if r[0] in want] or rows
        keys = ["gpu_util", "vram_used_mb", "vram_total_mb", "gpu_temp", "gpu_power"]
        cols = {k: [] for k in keys}
        for r in rows:
            for k, x in zip(keys, r[1:]):
                try:
                    cols[k].append(float(x))
                except Exception:
                    pass            # a field may report [N/A]; keep the rest
        d = {}
        if cols["gpu_util"]:
            d["gpu_util"] = sum(cols["gpu_util"]) / len(cols["gpu_util"])
        for k in ("vram_used_mb", "vram_total_mb", "gpu_power"):
            if cols[k]:
                d[k] = sum(cols[k])
        if cols["gpu_temp"]:
            d["gpu_temp"] = max(cols["gpu_temp"])
        d["gpu_count"] = len(rows)
        return d
    except Exception:
        return {}               # no NVIDIA GPU / nvidia-smi not found


def _read_cpu_ram() -> dict:
    """CPU + system RAM via psutil if installed (pip install psutil to enable)."""
    try:
        import psutil
        vm = psutil.virtual_memory()
        return {"cpu_pct": psutil.cpu_percent(interval=None),
                "ram_used_gb": vm.used / 1e9, "ram_total_gb": vm.total / 1e9}
    except Exception:
        return {}


@app.get("/api/sysstats")
def api_sysstats():
    return {**_read_gpu(), **_read_cpu_ram()}


@app.get("/api/stage/{key}")
def api_stage(key: str):
    for s in PIPE.stages:
        if s.key == key:
            return {"key": s.key, "label": s.label, "status": s.status, "kind": s.kind,
                    "log": s.lines(), "metrics": s.metrics() if s.kind in ("train", "data") else []}
    return JSONResponse({"error": "no such stage"}, status_code=404)


_chat_state: dict = {}


@app.post("/api/chat")
async def api_chat(body: dict):
    """Lazy-load the latest trained model and answer (RAG-ready). Graceful if untrained."""
    msg = (body or {}).get("message", "").strip()
    if not msg:
        return {"reply": ""}
    try:
        if "model" not in _chat_state:
            _load_chat_model()
        from rag.chat import respond
        reply = respond(_chat_state["model"], _chat_state["tok"], msg,
                        retriever=_chat_state.get("retriever"), max_new_tokens=200,
                        device=_chat_state["device"])
        return {"reply": reply}
    except FileNotFoundError:
        return {"reply": "(no trained model yet -- run the pipeline first)"}
    except Exception as e:
        return {"reply": f"(chat error: {e})"}


def _load_chat_model():
    import torch
    sys.path.insert(0, ROOT)
    from model import ModelConfig, MaxGPTUltra
    from tokenizer.tokenizer import UltraTokenizer
    from train.checkpoint import load_checkpoint
    cfg = ModelConfig.from_yaml(CHAT_CFG["config"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # prefer the most-trained checkpoint available
    ckpt = None
    for run in ("runs/dpo", "runs/sft", "runs/pretrain"):
        latest = os.path.join(ROOT, run, "checkpoints", "latest.json")
        if os.path.exists(latest):
            name = json.load(open(latest, encoding="utf-8"))["path"]
            ckpt = os.path.join(ROOT, run, "checkpoints", name)
            break
    if not ckpt:
        raise FileNotFoundError("no checkpoint")
    model = MaxGPTUltra(cfg)
    load_checkpoint(ckpt, model, map_location=device)
    _chat_state.update(model=model.to(device).eval(),
                       tok=UltraTokenizer(CHAT_CFG["tokenizer"]), device=device)


def build_default_pipeline(config, tokenizer, shards, sft_data, pref_data, runs, eval_shards=None) -> Pipeline:
    py = sys.executable
    try:
        n_gpus = resolve_gpus((load_yaml(os.path.join(ROOT, config)).get("train") or {}).get("gpus", 1))
    except Exception:
        n_gpus = 1
    print(f"[gui] training stages will use {n_gpus} GPU(s)" + (" via torchrun" if n_gpus > 1 else ""), flush=True)

    def train_cmd(script_and_args, run_out):
        return launch_cmd(py, script_and_args, n_gpus, run_out)

    def latest_ckpt(run_dir):
        latest = os.path.join(ROOT, run_dir, "checkpoints", "latest.json")
        if not os.path.exists(latest):
            raise FriendlyError(f"previous stage produced no checkpoint in {run_dir} "
                                f"(it may not have finished). Nothing to start from.")
        return os.path.join(run_dir, "checkpoints", json.load(open(latest, encoding="utf-8"))["path"])

    def data_done():   # data prep already built the shards + SFT/DPO files -> jump to pretrain
        return (os.path.exists(os.path.join(ROOT, shards, "meta.json"))
                and os.path.exists(os.path.join(ROOT, sft_data))
                and os.path.exists(os.path.join(ROOT, pref_data)))

    def suite_flags():   # the fixed benchmark suite, once it has been fetched (scripts/eval_suite.py --fetch)
        return (["--suite-dir", "data/bench", "--suite-every", "10000"]
                if os.path.exists(os.path.join(ROOT, "data", "bench", "suite.json")) else [])

    stages = [
        Stage("data", "Data", lambda: [py, "-u", "scripts/prepare_data.py", "--config", config,
              "--metrics-out", os.path.join(runs, "data", "metrics.jsonl")],
              os.path.join(runs, "data"), kind="data", done_when=data_done),
        Stage("pretrain", "Pretrain", lambda: train_cmd(["scripts/train.py", "--config", config,
              "--data", shards, "--out", os.path.join(runs, "pretrain"), "--tokenizer", tokenizer,
              "--eval-data", (eval_shards or shards), "--eval-every", "500", *suite_flags(),
              "--stop-file", os.path.join(runs, "pretrain", "STOP")], os.path.join(runs, "pretrain")),
              os.path.join(runs, "pretrain")),
        Stage("sft", "SFT", lambda: train_cmd(["scripts/sft.py", "--config", config,
              "--init", latest_ckpt(os.path.join(runs, "pretrain")), "--tokenizer", tokenizer,
              "--data", sft_data, "--out", os.path.join(runs, "sft"),
              "--stop-file", os.path.join(runs, "sft", "STOP")], os.path.join(runs, "sft")),
              os.path.join(runs, "sft")),
        Stage("dpo", "DPO", lambda: train_cmd(["scripts/dpo.py", "--config", config,
              "--init", latest_ckpt(os.path.join(runs, "sft")), "--tokenizer", tokenizer,
              "--data", pref_data, "--out", os.path.join(runs, "dpo"),
              "--stop-file", os.path.join(runs, "dpo", "STOP")], os.path.join(runs, "dpo")),
              os.path.join(runs, "dpo")),
        Stage("chat", "Chat", lambda: [], os.path.join(runs, "dpo"), kind="chat"),
    ]
    pipe = Pipeline(stages)
    pipe.sync_from_disk()   # so the very first page load already shows finished stages as done
    return pipe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ultra.yaml")
    ap.add_argument("--tokenizer", default="tokenizer/maxgpt-ultra.tokenizer.json")
    ap.add_argument("--shards", default="data/shards")
    ap.add_argument("--eval-shards", default=None, help="held-out shard dir for val perplexity (scripts/holdout_shard.py); default: the training shards")
    ap.add_argument("--sft-data", default="data/sft.jsonl")
    ap.add_argument("--pref-data", default="data/prefs.jsonl")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--port", type=int, default=8800)
    args = ap.parse_args()
    global PIPE, CHAT_CFG
    PIPE = build_default_pipeline(args.config, args.tokenizer, args.shards,
                                  args.sft_data, args.pref_data, args.runs, eval_shards=args.eval_shards)
    CHAT_CFG = {"config": args.config, "tokenizer": args.tokenizer}
    print(f"[gui] reactor at http://127.0.0.1:{args.port}")
    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
