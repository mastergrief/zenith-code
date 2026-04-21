"""UnifiedSubstrate — per-layer sub-head partition across trained mechanisms.

Round 8 of the runtime-recurrence research track. Generalizes the session-
30 Level-5 pattern (softmax + hard-max + grouped attention modes coexisting
in one layer via per-sub-head dispatch) from COMPILED attention variants
to TRAINED mechanisms:

  Layer partition (n_heads=32, d_head=2):
    sub-heads  0..15   softmax attention  — general context mixing
    sub-heads 16..23   DeltaNet recurrence — associative memory
    sub-heads 24..31   copy pointer       — prefix-masked input retrieval

All mechanisms share the layer's W_qkv projection. Each processes its
sub-head slice with its own attention kernel. Outputs concatenate along
the sub-head axis and pass through a shared W_out → residual. One forward
pass, three mechanisms, unified hidden state.

Differs from Round 6a (hybrid PT+DeltaNet):
  * 6a: DeltaNet backbone produces x → copy mechanism runs in separate
        pass with dedicated copy_q_proj/copy_k_proj → logits blended at
        OUTPUT in probability space.
  * R8: softmax/delta/copy all run in the SAME attention step on their
        sub-head slices → outputs sum at RESIDUAL level → single head
        projection. No log-prob blend; standard CE loss.

Design choice notes:
  * Copy sub-heads use standard softmax attention with a prefix-only mask
    (ignores positions after <sep>). Content-based attention on input
    embeddings → attended V carries token-identity; the shared head
    projects to vocab. No explicit scatter-to-vocab step (Way-1 pointer).
  * Delta sub-heads use L2-normed Q/K + SiLU + Householder recurrence
    per delta_rule.py. One β_head per layer, takes the layer's x as input,
    β_t broadcasts across delta sub-heads.
  * Softmax sub-heads use the causal softmax from the base Small2DTransformer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from calm.llm_computer.model import Small2DConfig, Small2DTransformer


MechKind = Literal["softmax", "delta", "copy"]


@dataclass
class MechSpec:
    kind: MechKind
    sh_lo: int   # sub-head range [lo, hi)
    sh_hi: int


@dataclass
class UnifiedSubstrateConfig(Small2DConfig):
    """Small2DConfig + per-layer sub-head partition.

    mech_specs applies uniformly to every layer. Per-layer variation can
    be added later by making this a list-of-lists. Ranges must cover
    [0, n_heads) without overlap.
    """
    mech_specs: list[MechSpec] = field(default_factory=list)
    sep_token_id: int = 3


class UnifiedSubstrateTransformer(Small2DTransformer):
    """Sub-head-partitioned transformer: softmax + delta + copy in one pass."""

    def __init__(self, config: UnifiedSubstrateConfig):
        super().__init__(config)
        self.u_config = config
        self._validate_specs()

        # One β_head per layer for the delta mechanism (learned write
        # strength taking the layer's residual x as input).
        self.beta_heads = nn.ModuleList([
            nn.Linear(config.d_model, 1, bias=True)
            for _ in range(config.n_layers)
        ])
        for h in self.beta_heads:
            with torch.no_grad():
                h.bias.fill_(0.0)

    def _validate_specs(self):
        specs = self.u_config.mech_specs
        if not specs:
            raise ValueError("UnifiedSubstrateConfig needs at least one MechSpec")
        covered = [False] * self.u_config.n_heads
        for s in specs:
            if not (0 <= s.sh_lo < s.sh_hi <= self.u_config.n_heads):
                raise ValueError(f"bad spec range: {s}")
            for i in range(s.sh_lo, s.sh_hi):
                if covered[i]:
                    raise ValueError(f"sub-head {i} covered twice")
                covered[i] = True
        if not all(covered):
            missing = [i for i, c in enumerate(covered) if not c]
            raise ValueError(f"sub-heads uncovered: {missing}")

    @staticmethod
    def _build_prefix_mask(idx: torch.Tensor, sep_id: int) -> torch.Tensor:
        """True for positions before the first <sep>; False elsewhere."""
        B, S = idx.shape
        is_sep = (idx == sep_id)
        has_sep = is_sep.any(dim=1)
        sep_pos = is_sep.float().argmax(dim=1)
        sep_pos = torch.where(has_sep, sep_pos, torch.tensor(S, device=idx.device))
        positions = torch.arange(S, device=idx.device).unsqueeze(0)
        return positions < sep_pos.unsqueeze(1)

    @staticmethod
    def _softmax_kernel(q, k, v):
        """Causal softmax attention on sub-head slice. Shapes: (B, H', S, Dh)."""
        B, H, S, Dh = q.shape
        scores = torch.einsum("bhid,bhjd->bhij", q, k)
        causal = torch.triu(
            torch.ones(S, S, dtype=torch.bool, device=q.device), diagonal=1
        )
        scores = scores.masked_fill(causal, float("-inf"))
        w = F.softmax(scores, dim=-1)
        return torch.einsum("bhij,bhjd->bhid", w, v)

    @staticmethod
    def _copy_kernel(q, k, v, prefix_mask):
        """Softmax attention over INPUT PREFIX positions only.

        prefix_mask: (B, S) — True for copyable prefix positions.
        Causal mask still applies (decode pos can't see future prefix).
        """
        B, H, S, Dh = q.shape
        scores = torch.einsum("bhid,bhjd->bhij", q, k)
        causal = torch.triu(
            torch.ones(S, S, dtype=torch.bool, device=q.device), diagonal=1
        )
        scores = scores.masked_fill(causal, float("-inf"))
        # Only attend to prefix positions: broadcast (B, S) → (B, H, S, S)
        non_prefix = ~prefix_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, S)
        non_prefix = non_prefix.expand(-1, H, S, -1)
        scores = scores.masked_fill(non_prefix, float("-inf"))
        # Positions where ALL keys are masked out → softmax gives uniform NaN;
        # zero out those rows.
        all_masked = non_prefix.all(dim=-1, keepdim=True) | causal.unsqueeze(0).unsqueeze(0).all(dim=-1, keepdim=True)
        scores = scores.masked_fill(
            (scores == float("-inf")).all(dim=-1, keepdim=True),
            0.0,  # dummy; we'll zero the output below
        )
        w = F.softmax(scores, dim=-1)
        out = torch.einsum("bhij,bhjd->bhid", w, v)
        # Zero positions that had no valid attention target (e.g. position 0
        # when the prefix is empty of preceding tokens).
        no_valid = all_masked.squeeze(-1).unsqueeze(-1)  # (B, H, S, 1)
        out = out.masked_fill(no_valid, 0.0)
        return out

    @staticmethod
    def _delta_kernel_slice(q, k, v, beta, eps=1e-6):
        """Householder recurrence on sub-head slice; state lives in (B, sub_D, sub_D).

        q/k/v: (B, H', S, Dh). beta: (B, S, 1).

        Flattens sub-head axis into a combined sub_D = H' * Dh dimension,
        runs DeltaNet Householder recurrence with L2-norm + SiLU on K/Q,
        reshapes back to (B, H', S, Dh).
        """
        B, H, S, Dh = q.shape
        D = H * Dh
        q_f = q.transpose(1, 2).reshape(B, S, D)
        k_f = k.transpose(1, 2).reshape(B, S, D)
        v_f = v.transpose(1, 2).reshape(B, S, D)

        q_feat = F.normalize(F.silu(q_f), p=2, dim=-1, eps=eps)
        k_feat = F.normalize(F.silu(k_f), p=2, dim=-1, eps=eps)

        S_state = torch.zeros(B, D, D, device=q.device, dtype=q.dtype)
        outs = []
        for t in range(S):
            k_t = k_feat[:, t, :]          # (B, D)
            v_t = v_f[:, t, :]             # (B, D)
            q_t = q_feat[:, t, :]          # (B, D)
            beta_t = beta[:, t, :]         # (B, 1)
            v_old = torch.einsum("bij,bj->bi", S_state, k_t)
            delta = (v_old - v_t) * beta_t
            update = torch.einsum("bi,bj->bij", delta, k_t)
            S_state = S_state - update
            out_t = torch.einsum("bij,bj->bi", S_state, q_t)
            outs.append(out_t)
        out_flat = torch.stack(outs, dim=1)  # (B, S, D)
        return out_flat.reshape(B, S, H, Dh).transpose(1, 2)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, S = idx.shape
        cfg = self.u_config
        pos_idx = torch.arange(S, device=idx.device)
        x = self.tok(idx) + self.pos(pos_idx)

        # Only build prefix mask if any copy spec exists.
        has_copy = any(s.kind == "copy" for s in cfg.mech_specs)
        prefix_mask = self._build_prefix_mask(idx, cfg.sep_token_id) if has_copy else None

        for layer in range(cfg.n_layers):
            qkv = self.W_qkv[layer](x)
            qkv = qkv.reshape(B, S, 3, cfg.n_heads, cfg.d_head)
            q, k, v = qkv.permute(2, 0, 3, 1, 4)  # 3 × (B, H, S, Dh)

            attn_out = torch.zeros_like(v)  # (B, H, S, Dh)

            for spec in cfg.mech_specs:
                q_s = q[:, spec.sh_lo:spec.sh_hi]
                k_s = k[:, spec.sh_lo:spec.sh_hi]
                v_s = v[:, spec.sh_lo:spec.sh_hi]
                if spec.kind == "softmax":
                    out_s = self._softmax_kernel(q_s, k_s, v_s)
                elif spec.kind == "delta":
                    beta = torch.sigmoid(self.beta_heads[layer](x))  # (B, S, 1)
                    out_s = self._delta_kernel_slice(q_s, k_s, v_s, beta)
                elif spec.kind == "copy":
                    out_s = self._copy_kernel(q_s, k_s, v_s, prefix_mask)
                else:
                    raise ValueError(f"unknown mech: {spec.kind}")
                attn_out[:, spec.sh_lo:spec.sh_hi] = out_s

            attn_out = attn_out.transpose(1, 2).reshape(B, S, cfg.d_model)
            x = x + self.W_out[layer](attn_out)
            gate, val = self.ff_in[layer](x).chunk(2, dim=-1)
            x = x + self.ff_out[layer](F.relu(gate) * val)

        return self.head(x)


def build_unified_substrate(
    vocab_size: int = 82, d_model: int = 64, n_heads: int = 32,
    n_layers: int = 4, d_ffn: int = 128, max_len: int = 96,
    sep_token_id: int = 3, use_hard_max: bool = False,
    partition: str = "16-8-8",  # softmax-delta-copy sub-head counts
) -> UnifiedSubstrateTransformer:
    """Canonical R8 config: 16 softmax + 8 delta + 8 copy sub-heads."""
    parts = [int(x) for x in partition.split("-")]
    assert len(parts) == 3, "partition must be 'a-b-c' for softmax/delta/copy"
    assert sum(parts) == n_heads, f"partition {parts} must sum to n_heads={n_heads}"
    s_lo = 0; s_hi = s_lo + parts[0]
    d_lo = s_hi; d_hi = d_lo + parts[1]
    c_lo = d_hi; c_hi = c_lo + parts[2]
    specs = [
        MechSpec(kind="softmax", sh_lo=s_lo, sh_hi=s_hi),
        MechSpec(kind="delta",   sh_lo=d_lo, sh_hi=d_hi),
        MechSpec(kind="copy",    sh_lo=c_lo, sh_hi=c_hi),
    ]
    cfg = UnifiedSubstrateConfig(
        vocab_size=vocab_size, d_model=d_model, n_heads=n_heads,
        n_layers=n_layers, d_ffn=d_ffn, max_len=max_len,
        use_hard_max=use_hard_max,
        mech_specs=specs, sep_token_id=sep_token_id,
    )
    assert cfg.d_head == 2, f"d_head must be 2, got {cfg.d_head}"
    return UnifiedSubstrateTransformer(cfg)
