"""DeltaNet subclass of Small2DTransformer — generalized Householder recurrence.

Round 5 of the runtime-weight-addition research track. Succeeds the Round-4
null (fast_weights delta+gate stuck at 10.5-12.2% on n=10 recall) by
implementing DeltaNet per Yang et al. 2024 (NeurIPS, arXiv:2406.06484).

Update rule:

    S_t = S_{t-1} (I - β_t k_t k_t^T) + β_t v_t k_t^T
        = S_{t-1} - β_t (S_{t-1} k_t - v_t) k_t^T      (SGD form)

Key differences from fast_weights.py's `use_delta_rule=True`:

  (1) No multiplicative decay λ. Only the specific k_t direction is
      erased; all other directions are preserved perfectly. The paper's
      ablation: when β_t = 1 and ‖k_t‖ = 1, `I − k_t k_t^T` is a clean
      projection matrix. The Round-4 variant had λ=0.95 which quietly
      decayed ALL stored bindings by ~40% over 10 steps.

  (2) L2 normalization on K and Q. Ensures eigenvalues of
      `I − β_t k_t k_t^T` land in [0, 1]. Paper ablation: L1→L2 alone
      worth +2pp zero-shot and ~3× on FDA.

  (3) SiLU feature map on K and Q (paper: Qin 2022 / Dao-Gu 2024).

  (4) Learned data-dependent β_t = σ(W_β x_t). Write strength is per-
      position and per-sample, not a fixed hyperparameter.

Read-after-write ordering: `o_t = S_t q_t` (state reflects the current
position's write). This is the paper's convention; contrast fast_weights
which uses read-before-write (S_{t-1} q_t). Makes a material difference
at n=query_at_position when the model wants to recall what it JUST wrote.

The recurrent form below is O(L·d²) — fine at L≤64 where this test runs.
The paper's contribution is the chunkwise parallel form (see
RESEARCH/DELTA-RULE/02_Chunkwise_Parallel_Algorithm.md) which matters
at L≥2K; not needed for this round.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from calm.llm_computer.model import Small2DConfig, Small2DTransformer


@dataclass
class DeltaNetConfig(Small2DConfig):
    """Small2DConfig + DeltaNet hyperparameters.

    Architecture matches FastWeightConfig so Round 4 baseline vs Round 5
    DeltaNet is apples-to-apples (same d_model, heads, layers, head dim).

    use_softmax_attn=False (default) matches the paper's Figure-2 architecture:
    DeltaNet REPLACES self-attention entirely. When True, softmax attention
    runs in parallel with DeltaNet (Round-5a ablation; regressed at n≥5
    because the optimizer routed through softmax and left DeltaNet
    untrained).
    """
    use_delta_net: bool = True
    use_softmax_attn: bool = False  # paper-canonical: DeltaNet replaces attn
    use_short_conv: bool = False    # paper's short 1D conv after QKV; optional at this scale
    use_l2_norm: bool = True        # L2 on K/Q per paper; ablation: try False
    use_silu_feat: bool = True      # SiLU feature map on K/Q per paper


class DeltaNetSmall2DTransformer(Small2DTransformer):
    """Small2DTransformer + per-layer DeltaNet recurrence alongside attention.

    Adds a β_t linear head per layer (d_model + 1 params per layer).
    Runs DeltaNet recurrence in parallel with softmax attention; outputs
    are summed into the residual stream, same pattern as fast_weights.py.
    """

    def __init__(self, config: DeltaNetConfig):
        super().__init__(config)
        self.beta_head = nn.ModuleList([
            nn.Linear(config.d_model, 1, bias=True)
            for _ in range(config.n_layers)
        ])
        # Initialize β bias to 0 so σ(0) = 0.5 at init — balanced read/write
        # strength before training. Paper initializes differently for LM
        # training; 0.5 works for this short-sequence synthetic.
        for h in self.beta_head:
            with torch.no_grad():
                h.bias.fill_(0.0)

    @staticmethod
    def _delta_step(
        S: torch.Tensor,
        q_t: torch.Tensor,
        k_t: torch.Tensor,
        v_t: torch.Tensor,
        beta_t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One DeltaNet update.

        Args:
            S: (B, D, D) hidden state before the write.
            q_t, k_t, v_t: (B, D) — k_t and q_t assumed L2-normalized
                and SiLU-mapped. v_t is raw (no feature map).
            beta_t: (B, 1) in [0, 1] — data-dependent write strength.

        Returns:
            S_new: (B, D, D)
            out: (B, D) — `S_new @ q_t` (read-after-write).
        """
        # Householder / SGD form: S_new = S - β·(S@k - v) k^T
        v_old = torch.einsum("bij,bj->bi", S, k_t)           # (B, D)
        delta = (v_old - v_t) * beta_t                        # (B, D)
        update = torch.einsum("bi,bj->bij", delta, k_t)       # (B, D, D)
        S_new = S - update
        out = torch.einsum("bij,bj->bi", S_new, q_t)
        return S_new, out

    def _forward_backbone(self, idx: torch.Tensor) -> torch.Tensor:
        """Run the DeltaNet backbone; return pre-head hidden states x (B, S, D).

        Factored out so subclasses (e.g. CopyAugmentedDeltaNet) can use the
        DeltaNet recurrence without the final vocab projection. Has the same
        use_delta_net=False fallback as forward().
        """
        if not getattr(self.config, "use_delta_net", True):
            # Vanilla path — replicate Small2DTransformer.forward minus the head
            B, S = idx.shape
            cfg = self.config
            pos_idx = torch.arange(S, device=idx.device)
            x = self.tok(idx) + self.pos(pos_idx)
            for layer in range(cfg.n_layers):
                qkv = self.W_qkv[layer](x)
                qkv = qkv.reshape(B, S, 3, cfg.n_heads, cfg.d_head)
                q, k, v = qkv.permute(2, 0, 3, 1, 4)
                attn = self._attention(q, k, v, hard_max=cfg.use_hard_max)
                attn = attn.transpose(1, 2).reshape(B, S, cfg.d_model)
                x = x + self.W_out[layer](attn)
                gate, val = self.ff_in[layer](x).chunk(2, dim=-1)
                x = x + self.ff_out[layer](F.relu(gate) * val)
            return x

        B, S = idx.shape
        cfg = self.config
        pos_idx = torch.arange(S, device=idx.device)
        x = self.tok(idx) + self.pos(pos_idx)

        for layer in range(cfg.n_layers):
            qkv = self.W_qkv[layer](x)                         # (B, S, 3D)
            qkv = qkv.reshape(B, S, 3, cfg.n_heads, cfg.d_head)
            q, k, v = qkv.permute(2, 0, 3, 1, 4)               # 3 × (B, H, S, Dh)

            if cfg.use_softmax_attn:
                attn = self._attention(q, k, v, hard_max=cfg.use_hard_max)
                attn = attn.transpose(1, 2).reshape(B, S, cfg.d_model)
            else:
                attn = None

            # Flatten heads back to (B, S, D) for the DeltaNet recurrence,
            # which treats the whole d_model vector as one "head" (the
            # sub-head invariant lives in the attention path only).
            q_flat = q.transpose(1, 2).reshape(B, S, cfg.d_model)
            k_flat = k.transpose(1, 2).reshape(B, S, cfg.d_model)
            v_flat = v.transpose(1, 2).reshape(B, S, cfg.d_model)

            # Paper feature-map + L2-norm on K and Q. V untouched.
            # Both ablatable — the paper's d_head=128 regime may not transfer
            # to the substrate's d_head=2 invariant.
            if cfg.use_silu_feat:
                q_feat = F.silu(q_flat)
                k_feat = F.silu(k_flat)
            else:
                q_feat = q_flat
                k_feat = k_flat
            if cfg.use_l2_norm:
                q_feat = F.normalize(q_feat, p=2, dim=-1, eps=1e-6)
                k_feat = F.normalize(k_feat, p=2, dim=-1, eps=1e-6)

            # Per-position learned β_t ∈ (0, 1).
            beta = torch.sigmoid(self.beta_head[layer](x))     # (B, S, 1)

            S_state = torch.zeros(
                B, cfg.d_model, cfg.d_model,
                device=x.device, dtype=x.dtype,
            )
            reads = []
            for t in range(S):
                S_state, out_t = self._delta_step(
                    S_state,
                    q_feat[:, t, :],
                    k_feat[:, t, :],
                    v_flat[:, t, :],
                    beta[:, t, :],
                )
                reads.append(out_t)
            delta_out = torch.stack(reads, dim=1)              # (B, S, D)

            if attn is not None:
                x = x + self.W_out[layer](attn) + self.W_out[layer](delta_out)
            else:
                x = x + self.W_out[layer](delta_out)
            gate, val = self.ff_in[layer](x).chunk(2, dim=-1)
            x = x + self.ff_out[layer](F.relu(gate) * val)

        return x

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """idx: (B, S). Returns logits (B, S, vocab)."""
        return self.head(self._forward_backbone(idx))
