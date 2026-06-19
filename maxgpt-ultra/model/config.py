"""Model configuration for MaxGPT-Ultra.

A single dataclass describes the architecture. It can be built by hand or loaded
from a YAML file in configs/ (the `model:` block). The training-loop reads the
`train:` block separately.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import yaml


@dataclass
class ModelConfig:
    # --- core dimensions ---
    vocab_size: int = 49152      # ~49k byte-level BPE (49152 = 3 * 2^14, tensor-core aligned)
    d_model: int = 768           # residual stream width
    n_layers: int = 12           # number of transformer blocks
    n_heads: int = 12            # query heads
    n_kv_heads: int = 4          # key/value heads (GQA: n_heads must be a multiple of this)
    mlp_hidden: int = 2048       # SwiGLU inner width (~8/3 * d_model)
    seq_len: int = 1024          # max context the RoPE table is built for

    # --- knobs ---
    rope_theta: float = 10000.0  # RoPE base frequency
    qk_norm: bool = True         # RMSNorm on per-head Q and K before attention
    tie_embeddings: bool = True  # share the token embedding with the output head
    rms_eps: float = 1e-5        # RMSNorm epsilon
    init_std: float = 0.02       # std for the normal weight init

    @property
    def head_dim(self) -> int:
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        return self.d_model // self.n_heads

    def __post_init__(self) -> None:
        assert self.n_heads % self.n_kv_heads == 0, \
            f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads}) for GQA"
        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"

    @classmethod
    def from_yaml(cls, path: str) -> "ModelConfig":
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        m = dict(raw.get("model", {}))
        # These are design commitments, not knobs - assert the YAML agrees.
        assert m.pop("norm", "rmsnorm") == "rmsnorm", "this architecture is RMSNorm-only"
        assert m.pop("pos", "rope") == "rope", "this architecture is RoPE-only"
        assert m.pop("mlp", "swiglu") == "swiglu", "this architecture is SwiGLU-only"
        valid = {f.name for f in dataclasses.fields(cls)}
        kwargs = {k: v for k, v in m.items() if k in valid}
        return cls(**kwargs)
