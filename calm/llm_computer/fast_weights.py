"""Fast-weights subclass of Small2DTransformer — Round 1 of runtime weight addition.

Schlag-style asymmetric fast weights (Schlag et al. 2021, building on
Ba et al. 2016 and Schmidhuber 1992). Per-layer `W_fast` matrix updates
during inference via an outer-product Hebbian rule:

    W_fast_t = λ · W_fast_{t-1} + η · outer(v_t, k_t)

Read via `W_fast @ q_t`, added to the residual stream alongside standard
attention. No gradient descent, no training loop — weights evolve during
the forward pass and reset at sequence boundaries.

The substrate constraint is d_head = 2 (Percepta paper). This module's
novel experimental axis: fast-weight associative recall has never been
measured at this narrow head dimension. Round 1 binary decision in
scripts/experiment_fast_weights.py.

Read-before-write ordering at each position: state at position t reflects
writes from positions < t (not t itself). Matches delta-rule semantics and
prevents trivial self-retrieval during training.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from calm.llm_computer.model import Small2DConfig, Small2DTransformer


@dataclass
class FastWeightConfig(Small2DConfig):
    """Small2DConfig + fast-weights hyperparameters.

    When `use_fast_weights=False`, forward falls through to the parent
    Small2DTransformer.forward and produces bitwise-equal output (given
    matching parameters). This is the regression-test invariant.

    `normalize_outer=True` scales outer(v,k) by 1/d_model inside forward so
    the fast-weight matrix stays bounded at initialization — analogous to
    attention's 1/sqrt(d_k) scaling. Without it, random-init fast reads
    dominate the residual stream and initial loss blows up.
    """
    lambda_decay: float = 0.95
    eta_write: float = 0.5
    use_fast_weights: bool = True
    normalize_outer: bool = True


class FastWeightSmall2DTransformer(Small2DTransformer):
    """Small2DTransformer + per-layer Schlag-style fast weights.

    Per layer, per position t:
      1. Standard attention unchanged (softmax or hard-max, causal).
      2. fast_read_t = W_fast @ q_t     (state BEFORE position-t write)
      3. W_fast ← λ·W_fast + η·outer(v_t, k_t)
      4. Residual: x ← x + W_out(attn) + fast_read
      5. FFN unchanged.

    No new nn.Parameters — `W_fast` is a runtime tensor allocated per
    forward call and discarded at sequence end. This means the subclass's
    state_dict matches the parent's exactly, so weight-loading across
    classes works without strict=False tricks.
    """

    @staticmethod
    def _fast_weight_step(
        W_fast: torch.Tensor,
        q_t: torch.Tensor,
        k_t: torch.Tensor,
        v_t: torch.Tensor,
        lambda_decay: float,
        eta_write: float,
        outer_scale: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One fast-weight update: read, then write.

        Args:
            W_fast: (B, D, D) current state.
            q_t, k_t, v_t: (B, D) projections at current position.
            outer_scale: multiplier on outer(v, k) before integration. The
                model's forward passes 1/d_model here when normalize_outer
                is True; tests pass default 1.0 for clean Schlag math.

        Returns:
            new_W_fast: (B, D, D)
            read: (B, D) — W_fast @ q_t using pre-update state.
        """
        read = torch.einsum("bij,bj->bi", W_fast, q_t)
        outer = torch.einsum("bi,bj->bij", v_t, k_t) * outer_scale
        new_W_fast = lambda_decay * W_fast + eta_write * outer
        return new_W_fast, read

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """idx: (B, S). Returns logits (B, S, vocab).

        When `use_fast_weights=False`, delegates to parent for bit-identical output.
        """
        if not self.config.use_fast_weights:
            return super().forward(idx)

        B, S = idx.shape
        cfg = self.config
        pos_idx = torch.arange(S, device=idx.device)
        x = self.tok(idx) + self.pos(pos_idx)

        for layer in range(cfg.n_layers):
            qkv = self.W_qkv[layer](x)  # (B, S, 3*D)
            qkv = qkv.reshape(B, S, 3, cfg.n_heads, cfg.d_head)
            q, k, v = qkv.permute(2, 0, 3, 1, 4)  # 3 × (B, H, S, Dh)
            attn = self._attention(q, k, v, hard_max=cfg.use_hard_max)
            attn = attn.transpose(1, 2).reshape(B, S, cfg.d_model)  # (B, S, D)

            # Fast-weights: per-position rollout. State is per-batch-element.
            # Reuse the same Q/K/V projections that feed standard attention —
            # one projection matrix serves both paths (substrate invariant).
            q_flat = q.transpose(1, 2).reshape(B, S, cfg.d_model)
            k_flat = k.transpose(1, 2).reshape(B, S, cfg.d_model)
            v_flat = v.transpose(1, 2).reshape(B, S, cfg.d_model)
            W_fast = torch.zeros(
                B, cfg.d_model, cfg.d_model, device=x.device, dtype=x.dtype,
            )
            outer_scale = 1.0 / cfg.d_model if cfg.normalize_outer else 1.0
            fast_reads = []
            for t in range(S):
                W_fast, read_t = self._fast_weight_step(
                    W_fast,
                    q_flat[:, t, :],
                    k_flat[:, t, :],
                    v_flat[:, t, :],
                    cfg.lambda_decay,
                    cfg.eta_write,
                    outer_scale,
                )
                fast_reads.append(read_t)
            fast_out = torch.stack(fast_reads, dim=1)  # (B, S, D)

            x = x + self.W_out[layer](attn) + fast_out
            gate, val = self.ff_in[layer](x).chunk(2, dim=-1)
            x = x + self.ff_out[layer](F.relu(gate) * val)

        return self.head(x)
