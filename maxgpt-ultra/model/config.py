"""Model configuration for MaxGPT-Ultra.

A single dataclass describes the architecture. It can be built by hand or loaded
from a YAML file in configs/ (the `model:` block). The training-loop reads the
`train:` block separately.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from cfg import load_yaml


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

    # --- optional tweaks from the 2025/26 small-model literature (all off by default; A/B'd before use) ---
    attn_gate: bool = False      # per-head sigmoid gate on the attention output (Qwen "Gated Attention" 2025, IMU-1 2026)
    value_residual: bool = False # normalized value residual: mix layer 1's values into every later layer (IMU-1 2026)
    norm_scaling: bool = False   # scale each block's pre-norm outputs by 1/sqrt(layer index) ("LayerNorm scaling", IMU-1)

    # --- options for the November 2026 restart (docs/research_2026-11-18_restart.md); off = the September model ---
    mlp: str = "swiglu"          # "swiglu" (3 matrices, hidden ~8/3 d) or "relu2" (2 matrices, relu(x)^2; set mlp_hidden to 1.5x the SwiGLU width to match parameters)
    peri_ln: bool = False        # Peri-LN: also normalize each sublayer's OUTPUT and the embeddings (Gemma 2/3, OLMo 2); an A/B candidate
    attn_fp32_layers: tuple = () # layers whose attention runs in fp32 even under bf16 autocast (the fix if bf16 attention gradients blow up late)

    @property
    def head_dim(self) -> int:
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        return self.d_model // self.n_heads

    def __post_init__(self) -> None:
        assert self.n_heads % self.n_kv_heads == 0, \
            f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads}) for GQA"
        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"
        assert self.mlp in ("swiglu", "relu2"), f"unknown mlp {self.mlp!r}: swiglu | relu2"
        self.attn_fp32_layers = tuple(int(i) for i in (self.attn_fp32_layers or ()))

    @classmethod
    def from_yaml(cls, path: str) -> "ModelConfig":
        raw = load_yaml(path)
        m = dict(raw.get("model", {}))
        # These are design commitments, not knobs - assert the YAML agrees.
        assert m.pop("norm", "rmsnorm") == "rmsnorm", "this architecture is RMSNorm-only"
        assert m.pop("pos", "rope") == "rope", "this architecture is RoPE-only"
        assert m.get("mlp", "swiglu") in ("swiglu", "relu2"), "mlp must be swiglu or relu2"
        if "attn_fp32_layers" in m:
            m["attn_fp32_layers"] = tuple(int(i) for i in (m["attn_fp32_layers"] or ()))
        valid = {f.name for f in dataclasses.fields(cls)}
        kwargs = {k: v for k, v in m.items() if k in valid}
        return cls(**kwargs)
