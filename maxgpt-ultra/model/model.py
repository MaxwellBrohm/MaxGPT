"""MaxGPT-Ultra - a modern decoder-only transformer.

Design (Llama / Qwen / SmolLM2 lineage), every choice documented in TECHNIQUES.md:
  - RoPE rotary positions (no learned position embeddings)
  - RMSNorm, pre-norm, no biases anywhere
  - SwiGLU feed-forward
  - Grouped-query attention (GQA)
  - QK-norm (RMSNorm on per-head Q/K) for high-LR stability
  - Tied input/output embeddings
  - z-loss available in the loss for logit-scale control

Attention uses torch's scaled_dot_product_attention, which dispatches to
FlashAttention on supported GPUs (and a math fallback on CPU).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from .config import ModelConfig


# --------------------------------------------------------------------------- #
# Normalization
# --------------------------------------------------------------------------- #
class RMSNorm(nn.Module):
    """Root-mean-square norm: rescale by RMS over the feature dim, then a learned
    per-feature gain. No mean-centering and no bias, so it's cheaper than LayerNorm
    and empirically just as stable. Computed in fp32 for numerical safety."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x.to(dtype) * self.weight


# --------------------------------------------------------------------------- #
# Rotary position embeddings (RoPE)
# --------------------------------------------------------------------------- #
def precompute_rope(head_dim: int, seq_len: int, theta: float,
                    dtype: torch.dtype = torch.float32):
    """Build the (seq_len, head_dim) cos/sin tables once. Rotating Q/K by these
    angles injects *relative* position into the dot product, which generalizes
    better than absolute learned positions."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))  # (hd/2,)
    pos = torch.arange(seq_len).float()                                            # (T,)
    freqs = torch.outer(pos, inv_freq)                                             # (T, hd/2)
    emb = torch.cat((freqs, freqs), dim=-1)                                        # (T, hd)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, n_heads, T, head_dim); cos/sin: (T, head_dim)
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return (x * cos) + (rotate_half(x) * sin)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand (B, n_kv, T, hd) -> (B, n_kv*n_rep, T, hd) so each KV head is shared
    by n_rep query heads (the GQA trick: fewer KV heads = smaller KV cache + less
    memory bandwidth, at almost no quality cost)."""
    if n_rep == 1:
        return x
    B, n_kv, T, hd = x.shape
    return (x[:, :, None, :, :]
            .expand(B, n_kv, n_rep, T, hd)
            .reshape(B, n_kv * n_rep, T, hd))


# --------------------------------------------------------------------------- #
# Attention (GQA + QK-norm + RoPE)
# --------------------------------------------------------------------------- #
class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.n_rep = cfg.n_heads // cfg.n_kv_heads
        self.head_dim = cfg.head_dim

        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * self.head_dim, cfg.d_model, bias=False)

        self.qk_norm = cfg.qk_norm
        if cfg.qk_norm:
            self.q_norm = RMSNorm(self.head_dim, cfg.rms_eps)
            self.k_norm = RMSNorm(self.head_dim, cfg.rms_eps)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim)

        # QK-norm: normalize each head's Q and K (over head_dim) before RoPE. Keeps
        # the attention logits well-scaled so we can train at a higher learning rate.
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = q.transpose(1, 2)  # (B, n_heads,    T, hd)
        k = k.transpose(1, 2)  # (B, n_kv_heads, T, hd)
        v = v.transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        # Causal flash/SDPA attention.
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
        return self.o_proj(out)


# --------------------------------------------------------------------------- #
# SwiGLU feed-forward
# --------------------------------------------------------------------------- #
class SwiGLU(nn.Module):
    """FFN(x) = down( silu(gate(x)) * up(x) ). The gated activation outperforms a
    plain GELU MLP at the same parameter budget; hidden width is ~8/3 * d_model so
    the three projections roughly match a 4x GELU MLP's parameter count."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.d_model, cfg.mlp_hidden, bias=False)
        self.up_proj = nn.Linear(cfg.d_model, cfg.mlp_hidden, bias=False)
        self.down_proj = nn.Linear(cfg.mlp_hidden, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# --------------------------------------------------------------------------- #
# Transformer block (pre-norm)
# --------------------------------------------------------------------------- #
class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.attn = Attention(cfg)
        self.mlp_norm = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.mlp = SwiGLU(cfg)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # Pre-norm residuals: normalize the input to each sublayer, add the result back.
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.mlp(self.mlp_norm(x))
        return x


# --------------------------------------------------------------------------- #
# Full model
# --------------------------------------------------------------------------- #
class MaxGPTUltra(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.grad_checkpointing = False   # trainer sets True to trade compute for memory (fits the 1B on 12GB)
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight  # weight tying

        cos, sin = precompute_rope(cfg.head_dim, cfg.seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # Deep-net residual init: scale the *output* projection of each sublayer by
        # 1/sqrt(2 * n_layers) so the residual stream doesn't blow up with depth.
        for name, p in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("down_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=cfg.init_std / math.sqrt(2 * cfg.n_layers))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None,
                z_loss_weight: float = 0.0):
        B, T = idx.shape
        assert T <= self.cfg.seq_len, f"sequence length {T} exceeds seq_len {self.cfg.seq_len}"

        x = self.tok_emb(idx)
        cos = self.rope_cos[:T].to(x.dtype)
        sin = self.rope_sin[:T].to(x.dtype)
        for block in self.blocks:
            if self.grad_checkpointing and self.training:
                x = grad_checkpoint(block, x, cos, sin, use_reentrant=False)
            else:
                x = block(x, cos, sin)
        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            logits_flat = logits.view(-1, logits.size(-1)).float()  # fp32 for the softmax
            targets_flat = targets.view(-1)
            if (targets_flat != -100).any():
                loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=-100)
            else:
                loss = logits_flat.sum() * 0.0   # batch had no supervised tokens (SFT masking edge case)
            if z_loss_weight > 0.0:
                # z-loss penalizes large logsumexp, keeping logits from drifting huge
                lse = torch.logsumexp(logits_flat, dim=-1)
                keep = targets_flat != -100
                z = (lse[keep].pow(2).mean() if keep.any() else lse.pow(2).mean())
                loss = loss + z_loss_weight * z
        return logits, loss

    @torch.no_grad()
    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())  # tied weights counted once
        if non_embedding:
            n -= self.tok_emb.weight.numel()
            if self.lm_head.weight is not self.tok_emb.weight:
                n -= self.lm_head.weight.numel()
        return n
