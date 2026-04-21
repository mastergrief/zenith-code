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
    use_chunkwise: bool = False     # chunkwise parallel form (UT transform, paper §4)
    chunk_size: int = 32            # C in paper — 32 is sweet spot at seq≤128


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

    @staticmethod
    def _delta_chunkwise(
        S: torch.Tensor,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        beta: torch.Tensor,
        chunk_size: int = 32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Chunkwise parallel delta-rule (paper §3-4 — UT transform).

        Args:
            S:     (B, D, D) initial state (typically zeros).
            Q/K/V: (B, L, D) per-position queries, keys, values.
                   K/Q are assumed L2-normalized + SiLU (caller applies).
            beta:  (B, L, 1) per-position write strength in (0, 1).
            chunk_size: C — number of tokens processed in parallel per
                   chunk. Trade-off: bigger C = more parallelism, more
                   SRAM per chunk. 32 is sweet spot at L≤128; bump to
                   64 or 128 for longer sequences.

        Returns:
            S_final: (B, D, D) final state after all positions processed.
            reads:   (B, L, D) per-position read-after-write outputs,
                     matches `_delta_step` one-position-at-a-time output
                     to float32 numerical epsilon.

        Math (for chunk [t] of size C starting from prior state S):
            K_c K_c^T                                  (C, C)
            A = I + tril(diag(β) K K^T, -1)            (C, C) lower-tri
            T = A^-1 · diag(β)                          (C, C) tri solve
            W = T K,  U = T V                          (C, D) each
            U' = U - W S^T                              (C, D) — prior-state adjusted
            O = Q S^T + (Q K^T ⊙ M_causal) U'          (C, D) output
            S_next = S + U'^T K                        (D, D) state update
        """
        B, L, D = Q.shape
        device = Q.device
        dtype = Q.dtype

        # Precompute the strict-lower-triangular mask once (outside chunk loop).
        # For the causal Q K^T mask we want lower-triangular INCLUDING diagonal
        # (position r reads its own k_r after write at position r).
        C = chunk_size

        reads_chunks = []
        for start in range(0, L, C):
            end = min(start + C, L)
            Cc = end - start

            Q_c = Q[:, start:end, :]                        # (B, Cc, D)
            K_c = K[:, start:end, :]                        # (B, Cc, D)
            V_c = V[:, start:end, :]                        # (B, Cc, D)
            beta_c = beta[:, start:end, :].squeeze(-1)      # (B, Cc)

            # K K^T intra-chunk Gram matrix.
            Kkt = torch.matmul(K_c, K_c.transpose(-2, -1))  # (B, Cc, Cc)

            # A = I + tril(diag(β) K K^T, -1)
            # Row r of (diag(β) K K^T) is β_r · K_r K^T. Strict lower tri only.
            eye_Cc = torch.eye(Cc, device=device, dtype=dtype).unsqueeze(0).expand(B, Cc, Cc)
            strict_tril = torch.tril(
                torch.ones(Cc, Cc, device=device, dtype=dtype), diagonal=-1,
            )
            A_mat = eye_Cc + (beta_c.unsqueeze(-1) * Kkt) * strict_tril

            # T · A = diag(β)  →  T = A^-1 · diag(β)
            # Using torch.linalg.solve_triangular: solves A X = RHS for X.
            rhs = torch.diag_embed(beta_c)                  # (B, Cc, Cc)
            T_mat = torch.linalg.solve_triangular(A_mat, rhs, upper=False)

            W_c = torch.matmul(T_mat, K_c)                  # (B, Cc, D)
            U_c = torch.matmul(T_mat, V_c)                  # (B, Cc, D)

            # Prior-state adjustment. S^T is (B, D, D).
            S_t = S.transpose(-2, -1)                        # (B, D, D)
            U_prime = U_c - torch.matmul(W_c, S_t)           # (B, Cc, D)

            # Intra-chunk output with causal (lower-tri incl. diagonal) mask.
            causal_incl_diag = torch.tril(
                torch.ones(Cc, Cc, device=device, dtype=dtype), diagonal=0,
            )
            Qkt = torch.matmul(Q_c, K_c.transpose(-2, -1))   # (B, Cc, Cc)
            Qkt_masked = Qkt * causal_incl_diag
            O_c = (
                torch.matmul(Q_c, S_t)                        # prior-state reads
                + torch.matmul(Qkt_masked, U_prime)           # intra-chunk reads
            )                                                 # (B, Cc, D)
            reads_chunks.append(O_c)

            # State update: S_next = S + (U')^T K
            S = S + torch.matmul(U_prime.transpose(-2, -1), K_c)  # (B, D, D)

        reads = torch.cat(reads_chunks, dim=1)               # (B, L, D)
        return S, reads

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
            if getattr(cfg, "use_chunkwise", False):
                S_state, delta_out = self._delta_chunkwise(
                    S_state, q_feat, k_feat, v_flat, beta,
                    chunk_size=getattr(cfg, "chunk_size", 32),
                )
            else:
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
                delta_out = torch.stack(reads, dim=1)          # (B, S, D)

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
