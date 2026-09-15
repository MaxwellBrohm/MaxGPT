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
