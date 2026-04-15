"""Direction 3 — mixed-geometry attention at d_head=2.

The substrate's `d_head=2` constraint makes 2D attention the bottleneck —
but 2D is also the natural home for several non-Euclidean geometries that
flat dot-product attention can't access:

  * Euclidean (current) — Q · K, dot product. Good for arbitrary linear
    similarity. Compiled programs assume this.
  * Hyperbolic (Poincaré disk) — exponentially expanding distances. Encodes
    hierarchies and trees with logarithmic capacity. Natural for is-a
    relationships, taxonomies, code call graphs.
  * Spherical — angle-only similarity (cosine). Ignores magnitude. Natural
    for "concept similarity regardless of intensity."
  * Toroidal — periodic wraparound. Natural for cyclic patterns (time of
    day, day of week, calendar).
  * Lattice — keys constrained to integer grid; exact discrete index lookup.
    Already used implicitly by parabolic-key compiled programs.

Per-layer dispatch via `layer_geometries` config. Default uniform Euclidean
preserves all current Small2DTransformer behavior, including all 15
compiled programs.

At d_head ≠ 2, several of these geometries lose their special properties
or require per-element tricks; at d_head=2 they're uniquely accessible
because 2D is small enough for closed-form geometric operations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from calm.llm_computer.model import Small2DConfig, Small2DTransformer


# ---- Geometry score functions ----
# Each takes Q (B, H, S, 2) and K (B, H, S, 2), returns scores (B, H, S, S).
# All assume d_head=2 in the last dim.

def euclidean_score(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Standard dot product. Current substrate behavior."""
    return torch.einsum("bhid,bhjd->bhij", q, k)


def hyperbolic_score(q: torch.Tensor, k: torch.Tensor,
                     epsilon: float = 1e-5) -> torch.Tensor:
    """Negative Poincaré disk distance. Closer points → higher score.

    Q and K are projected into the open unit disk via tanh(norm) scaling
    so they stay in (-1, 1)². The Poincaré distance is
        d(u, v) = arcosh(1 + 2|u-v|²/((1-|u|²)(1-|v|²)))
    Negate to give an attention-style score (higher = closer).
    """
    # Constrain to unit disk via radial tanh.
    qn = q / (1 + q.norm(dim=-1, keepdim=True))
    kn = k / (1 + k.norm(dim=-1, keepdim=True))
    qsq = (qn * qn).sum(dim=-1, keepdim=True)            # (B, H, S, 1)
    ksq = (kn * kn).sum(dim=-1, keepdim=True)            # (B, H, S, 1)
    # Pairwise diff norms via expansion.
    qkd = qn.unsqueeze(-2) - kn.unsqueeze(-3)            # (B, H, Sq, Sk, 2)
    diff_sq = (qkd * qkd).sum(dim=-1)                    # (B, H, Sq, Sk)
    denom = (1 - qsq) * (1 - ksq).transpose(-2, -1) + epsilon  # (B, H, Sq, Sk)
    arg = 1 + 2 * diff_sq / denom
    # arcosh(arg) for arg >= 1; clamp to avoid NaN at exactly 1.
    arg = torch.clamp(arg, min=1.0 + epsilon)
    distance = torch.acosh(arg)
    return -distance


def spherical_score(q: torch.Tensor, k: torch.Tensor,
                    epsilon: float = 1e-8) -> torch.Tensor:
    """Cosine similarity. Magnitude-invariant."""
    qn = q / (q.norm(dim=-1, keepdim=True) + epsilon)
    kn = k / (k.norm(dim=-1, keepdim=True) + epsilon)
    return torch.einsum("bhid,bhjd->bhij", qn, kn)


def toroidal_score(q: torch.Tensor, k: torch.Tensor,
                   period: float = 6.283185307) -> torch.Tensor:
    """Negative wrapped distance on a 2-torus. Treats Q and K as angles
    in [0, 2π) per dimension; computes minimum wrapped Euclidean distance
    between query and key, negated for attention semantics."""
    qd = q.unsqueeze(-2)                  # (B, H, Sq, 1, 2)
    kd = k.unsqueeze(-3)                  # (B, H, 1, Sk, 2)
    diff = qd - kd
    # Wrap each dim into [-period/2, period/2)
    diff = torch.remainder(diff + period / 2, period) - period / 2
    distance = torch.sqrt((diff * diff).sum(dim=-1) + 1e-8)
    return -distance


def lattice_score(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Snap K to nearest integer lattice point, then Euclidean.

    Used by compiled programs with parabolic-key construction. Snaps in K
    only because the program-emitted keys are exactly integer-valued and
    we want any small perturbation to round-trip.
    """
    k_snapped = k.round()
    return torch.einsum("bhid,bhjd->bhij", q, k_snapped)


GEOMETRY_DISPATCH: dict[str, Callable] = {
    "euclidean": euclidean_score,
    "hyperbolic": hyperbolic_score,
    "spherical": spherical_score,
    "toroidal": toroidal_score,
    "lattice": lattice_score,
}


# ---- Mixed-geometry transformer subclass ----

@dataclass
class MixedGeometryConfig(Small2DConfig):
    """Small2DConfig + per-layer geometry assignment.

    `layer_geometries` is None → uniform Euclidean (parent behavior).
    Else it must be a list of length `n_layers`, each entry one of the
    keys in GEOMETRY_DISPATCH.
    """
    layer_geometries: Optional[list[str]] = None


class MixedGeometrySmall2DTransformer(Small2DTransformer):
    """Per-layer geometric attention. Each layer's `_attention` step uses
    its own geometry dispatch from `cfg.layer_geometries`.

    Compiled programs that assume Euclidean keep working when
    layer_geometries=None or when a layer's entry is "euclidean".
    """

    def __init__(self, config: MixedGeometryConfig):
        super().__init__(config)
        if config.layer_geometries is not None:
            assert len(config.layer_geometries) == config.n_layers, (
                f"layer_geometries must have length n_layers="
                f"{config.n_layers}, got {len(config.layer_geometries)}"
            )
            for g in config.layer_geometries:
                assert g in GEOMETRY_DISPATCH, (
                    f"unknown geometry {g!r}; available: "
                    f"{list(GEOMETRY_DISPATCH)}"
                )

    def _attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                   hard_max: bool, geometry: str = "euclidean") -> torch.Tensor:
        """Causal attention with configurable geometry."""
        B, H, S, Dh = q.shape
        score_fn = GEOMETRY_DISPATCH[geometry]
        scores = score_fn(q, k)
        mask = torch.triu(
            torch.ones(S, S, dtype=torch.bool, device=q.device), diagonal=1
        )
        scores = scores.masked_fill(mask, float("-inf"))
        if hard_max:
            idx = scores.argmax(dim=-1, keepdim=True)
            weights = torch.zeros_like(scores)
            weights.scatter_(-1, idx, 1.0)
        else:
            weights = F.softmax(scores, dim=-1)
        return torch.einsum("bhij,bhjd->bhid", weights, v)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        if self.config.layer_geometries is None:
            return super().forward(idx)

        B, S = idx.shape
        cfg = self.config
        pos_idx = torch.arange(S, device=idx.device)
        x = self.tok(idx) + self.pos(pos_idx)
        for layer in range(cfg.n_layers):
            qkv = self.W_qkv[layer](x)
            qkv = qkv.reshape(B, S, 3, cfg.n_heads, cfg.d_head)
            q, k, v = qkv.permute(2, 0, 3, 1, 4)
            attn = self._attention(
                q, k, v,
                hard_max=cfg.use_hard_max,
                geometry=cfg.layer_geometries[layer],
            )
            attn = attn.transpose(1, 2).reshape(B, S, cfg.d_model)
            x = x + self.W_out[layer](attn)
            gate, val = self.ff_in[layer](x).chunk(2, dim=-1)
            x = x + self.ff_out[layer](F.relu(gate) * val)
        return self.head(x)
