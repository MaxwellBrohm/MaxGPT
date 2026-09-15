"""Muon optimizer (with the NorMuon and cautious-weight-decay options), Moonlight-style.

Muon (Jordan et al., 2024) replaces Adam's per-coordinate normalization for MATRIX weights
with an orthogonalized momentum: the momentum matrix M is mapped to the nearest
orthogonal matrix (all singular values -> 1) by five Newton-Schulz iterations, so every
direction in the weight space gets an equal-sized step instead of the few dominant ones.
Moonlight (Liu et al., 2025) showed it scales to multi-billion models with two additions
that we adopt: decoupled weight decay, and a per-matrix update scale that makes the update
RMS 0.2 (what AdamW's update typically has), so AdamW's learning rate and weight decay carry
over unchanged. Reported ~2x compute efficiency vs AdamW at matched quality.

Options:
  normalize=True   NorMuon (Li et al., 2025): after orthogonalizing, divide each ROW (output
                   neuron) by a running RMS of its own update, so no neuron hogs the step;
                   +~11% over Muon at 1.1B in the paper. Costs one scalar per row.
  cautious=True    Cautious weight decay (ICLR 2026): decay a coordinate only where the update
                   and the weight have the same sign (the step is already shrinking it);
                   lower loss at every scale tested, no new hyperparameters.

Only >=2D weights of the transformer blocks go through Muon; embeddings (the tied
embedding/head), norm gains and other 1D params use AdamW in a second param group, as in
every published Muon recipe. Under DDP each rank sees identical (all-reduced) gradients,
so it computes the identical update: no extra communication.
"""
from __future__ import annotations

import math

import torch

_NS_COEFFS = (3.4445, -4.7750, 2.0315)   # the quintic Newton-Schulz polynomial from the Muon paper


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, dtype=None) -> torch.Tensor:
    """Orthogonalize G: return the matrix with G's singular vectors and all singular values ~1.
    Runs in the given low-precision dtype (Muon's NS iteration is stable there) on tall-or-wide
    matrices alike (the iteration is applied to the smaller Gram side)."""
    assert G.ndim == 2
    a, b, c = _NS_COEFFS
    X = G.to(dtype) if dtype is not None else G
    X = X / (X.norm() + 1e-7)                      # spectral norm <= 1 so the polynomial converges
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    """Param groups may set `use_muon=False` to be optimized with AdamW instead (embeddings, 1D)."""

    def __init__(self, params, lr=3e-4, weight_decay=0.1, momentum=0.95, nesterov=True, ns_steps=5,
                 normalize=False, cautious=False, beta2=0.95, betas=(0.9, 0.95), eps=1e-8, ns_dtype=None):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov,
                        ns_steps=ns_steps, normalize=normalize, cautious=cautious, beta2=beta2,
                        betas=betas, eps=eps, use_muon=True)
        super().__init__(params, defaults)
        self.ns_dtype = ns_dtype

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            if group["use_muon"]:
                self._muon_group(group)
            else:
                self._adamw_group(group)
        return loss

    def _muon_group(self, group) -> None:
        lr, wd = group["lr"], group["weight_decay"]
        for p in group["params"]:
            if p.grad is None:
                continue
            g = p.grad
            if g.ndim != 2:                                   # e.g. a conv weight: flatten trailing dims
                g = g.reshape(g.size(0), -1)
            st = self.state[p]
            if "momentum" not in st:
                st["momentum"] = torch.zeros_like(g)
                if group["normalize"]:
                    st["row_v"] = torch.zeros(g.size(0), device=g.device, dtype=torch.float32)
            buf = st["momentum"]
            buf.mul_(group["momentum"]).add_(g)
            m = g.add(buf, alpha=group["momentum"]) if group["nesterov"] else buf
            O = zeropower_via_newtonschulz5(m, group["ns_steps"], self.ns_dtype).float()
            if group["normalize"]:                            # NorMuon: per-row second moment
                v = st["row_v"]
                v.mul_(group["beta2"]).add_(O.pow(2).mean(dim=1), alpha=1 - group["beta2"])
                O = O / (v.sqrt().unsqueeze(1) + group["eps"])
            # RMS-match to AdamW (Moonlight): scale so the update's RMS is 0.2 whatever the shape
            O = O * (0.2 * math.sqrt(O.numel()) / (O.norm() + group["eps"]))
            O = O.reshape(p.shape).to(p.dtype)
            if wd:
                if group["cautious"]:                         # decay only where the step already shrinks |p|
                    p.mul_(1 - lr * wd * (O * p > 0).to(p.dtype))
                else:
                    p.mul_(1 - lr * wd)
            p.add_(O, alpha=-lr)

    def _adamw_group(self, group) -> None:
        lr, wd, (b1, b2), eps = group["lr"], group["weight_decay"], group["betas"], group["eps"]
        for p in group["params"]:
            if p.grad is None:
                continue
            g = p.grad
            st = self.state[p]
            if "step" not in st:
                st["step"] = 0
                st["exp_avg"] = torch.zeros_like(p)
                st["exp_avg_sq"] = torch.zeros_like(p)
            st["step"] += 1
            st["exp_avg"].mul_(b1).add_(g, alpha=1 - b1)
            st["exp_avg_sq"].mul_(b2).addcmul_(g, g, value=1 - b2)
            bc1, bc2 = 1 - b1 ** st["step"], 1 - b2 ** st["step"]
            u = (st["exp_avg"] / bc1) / ((st["exp_avg_sq"] / bc2).sqrt() + eps)
            if wd:
                if group["cautious"]:
                    p.mul_(1 - lr * wd * (u * p > 0).to(p.dtype))
                else:
                    p.mul_(1 - lr * wd)
            p.add_(u, alpha=-lr)
