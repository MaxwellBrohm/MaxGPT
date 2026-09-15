"""Config YAML loading (torch-free, so the GUI server can use it without importing torch).

A top-level `extends: other.yaml` (relative to the file) pulls that file in first and overlays
this one on top, so a per-machine config (configs/ultra_lambda.yaml) can change training
knobs while the model block in configs/ultra.yaml stays the one source of truth.
"""
from __future__ import annotations

import os

import yaml


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_yaml(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    base = raw.pop("extends", None)
    if not base:
        return raw
    base_path = base if os.path.isabs(base) else os.path.join(os.path.dirname(os.path.abspath(path)), base)
    return _deep_merge(load_yaml(base_path), raw)


def configure_triton_ptxas() -> str | None:
    """Old NVIDIA drivers (< 525, CUDA 11.x era) cannot load the kernels Triton builds with its
    bundled CUDA 12 ptxas ("device kernel image is invalid"), which silently costs the whole
    torch.compile speedup. If such a driver is found and the CUDA 11.8 ptxas is installed in this
    environment (pip: nvidia-cuda-nvcc-cu11), point Triton at it. Returns the path used, or None.
    Torch-free and cheap, so every training script can call it first thing."""
    import glob
    import subprocess
    import sys
    if os.environ.get("TRITON_PTXAS_PATH"):
        return os.environ["TRITON_PTXAS_PATH"]
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=5).stdout.strip().splitlines()
        major = int(out[0].split(".")[0]) if out else 0
    except Exception:
        return None
    if major == 0 or major >= 525:
        return None
    hits = glob.glob(os.path.join(sys.prefix, "lib", "python*", "site-packages", "nvidia", "cuda_nvcc", "bin", "ptxas"))
    if not hits:
        print(f"[setup] driver {major} is too old for Triton's ptxas; install nvidia-cuda-nvcc-cu11==11.8.89 "
              f"to get torch.compile (running without it)", flush=True)
        return None
    os.environ["TRITON_PTXAS_PATH"] = hits[0]
    return hits[0]


def _driver_major() -> int:
    import subprocess
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=5).stdout.strip().splitlines()
        return int(out[0].split(".")[0]) if out else 0
    except Exception:
        return 0


def cuda_alloc_conf() -> None:
    """Enable PyTorch's expandable-segments allocator (less fragmentation, so a bigger micro_batch
    fits) on drivers that support it well. On old drivers (< 525) it reports a near-OOM as
    "CUDA driver error: invalid argument" and can corrupt the allocator after a failed kernel
    launch (seen on the Lambda box), so there we keep the default allocator. Never overrides an
    explicit PYTORCH_CUDA_ALLOC_CONF."""
    if "PYTORCH_CUDA_ALLOC_CONF" in os.environ:
        return
    major = _driver_major()
    if major == 0 or major >= 525:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
