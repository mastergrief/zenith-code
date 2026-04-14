"""Fast-weights subclass of Small2DTransformer — runtime weight addition.

Schlag-style asymmetric fast weights (Schlag et al. 2021, building on
Ba et al. 2016 and Schmidhuber 1992). Per-layer `W_fast` matrix updates
during inference via an outer-product Hebbian rule:

    W_fast_t = λ · W_fast_{t-1} + η · g_t · outer(Δv_t, k_t)

where `Δv_t = v_t − W_fast @ k_t` under the delta rule (Round 4), and
`g_t ∈ [0, 1]` is a per-position learned write gate (Round 4). With both
disabled this reduces to Round 1's plain form `W_fast ← λ·W_fast + η·outer(v, k)`.

Read via `W_fast @ q_t`, added to the residual stream alongside standard
attention. No gradient descent on `W_fast` — it evolves during the forward
pass and resets at sequence boundaries.

The substrate constraint is d_head = 2 (Percepta paper).

Read-before-write ordering at each position: state at position t reflects
writes from positions < t (not t itself), preventing trivial self-retrieval.

Round history:
  1  plain Schlag asymmetric fast weights; 99.1% on 3-pair recall
  2  fusion coexistence with compiled programs (no interference)
  3  d_model capacity scaling null at n=10 (+1.8pp on 4x capacity)
  4  delta rule + write gate — mechanism changes targeting interference
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
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
    attention's 1/sqrt(d_k) scaling.

    Round 4 additions (both default False — Round 1 behavior preserved):
      use_delta_rule: subtract current W_fast@k from v before the outer
        product. Re-binding the same key overwrites instead of stacking.
      use_write_gate: per-layer tiny MLP emits a per-position sigmoid gate
        in (0, 1) that scales the update. Gate≈0 → silent position;
        gate≈1 → full write. Lets the model learn to write only at KV
        positions, not at SEP / query tokens. Adds ~d_model × gate_hidden
        parameters per layer.
    """
    lambda_decay: float = 0.95
    eta_write: float = 0.5
    use_fast_weights: bool = True
    normalize_outer: bool = True
    use_delta_rule: bool = False
    use_write_gate: bool = False
    gate_hidden: int = 16


class FastWeightSmall2DTransformer(Small2DTransformer):
    """Small2DTransformer + per-layer Schlag fast weights with optional
    delta rule and learned write gate.

    State_dict invariants:
      - use_write_gate=False  → no extra nn.Parameters; state_dict matches
        the parent class exactly (Round 1 regression test).
      - use_write_gate=True   → per-layer `gate_mlp` ModuleList is added.
        Loading from a parent's state_dict requires strict=False.
    """

    def __init__(self, config: FastWeightConfig):
        super().__init__(config)
        if config.use_write_gate:
            # Small per-layer MLP: x (B, D) -> gate (B, 1) via sigmoid.
            # Final bias initialized to +2 so initial sigmoid ≈ 0.88 — gates
            # start mostly open, preserving Round 1 behavior at init; the
            # model can learn to close gates at non-KV positions.
            gate_mlps = []
            for _ in range(config.n_layers):
                mlp = nn.Sequential(
                    nn.Linear(config.d_model, config.gate_hidden),
                    nn.ReLU(),
                    nn.Linear(config.gate_hidden, 1),
                )
                with torch.no_grad():
                    mlp[-1].bias.fill_(2.0)
                gate_mlps.append(mlp)
            self.gate_mlp = nn.ModuleList(gate_mlps)

    @staticmethod
    def _fast_weight_step(
        W_fast: torch.Tensor,
        q_t: torch.Tensor,
        k_t: torch.Tensor,
        v_t: torch.Tensor,
        lambda_decay: float,
        eta_write: float,
        outer_scale: float = 1.0,
        *,
        use_delta_rule: bool = False,
        write_gate: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One fast-weight update: read, then write.

        Args:
            W_fast: (B, D, D) current state.
            q_t, k_t, v_t: (B, D) projections at current position.
            outer_scale: multiplier on outer(v, k) before integration.
            use_delta_rule: if True, subtract `W_fast @ k` from `v` before
                the outer product. Overwrites same-key bindings.
            write_gate: optional (B,) or (B, 1) tensor in [0, 1]. Scales the
                update term (not the decay). None = always write.

        Returns:
            new_W_fast: (B, D, D)
            read: (B, D) — `W_fast @ q_t` using pre-update state.
        """
        read = torch.einsum("bij,bj->bi", W_fast, q_t)

        if use_delta_rule:
            v_old = torch.einsum("bij,bj->bi", W_fast, k_t)
            write_v = v_t - v_old
        else:
            write_v = v_t

        outer = torch.einsum("bi,bj->bij", write_v, k_t) * outer_scale
        if write_gate is not None:
            outer = outer * write_gate.reshape(-1, 1, 1)

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

        outer_scale = 1.0 / cfg.d_model if cfg.normalize_outer else 1.0

        for layer in range(cfg.n_layers):
            qkv = self.W_qkv[layer](x)  # (B, S, 3*D)
            qkv = qkv.reshape(B, S, 3, cfg.n_heads, cfg.d_head)
            q, k, v = qkv.permute(2, 0, 3, 1, 4)  # 3 × (B, H, S, Dh)
            attn = self._attention(q, k, v, hard_max=cfg.use_hard_max)
            attn = attn.transpose(1, 2).reshape(B, S, cfg.d_model)  # (B, S, D)

            q_flat = q.transpose(1, 2).reshape(B, S, cfg.d_model)
            k_flat = k.transpose(1, 2).reshape(B, S, cfg.d_model)
            v_flat = v.transpose(1, 2).reshape(B, S, cfg.d_model)

            # Precompute per-position gates from the residual input x (B, S, D)
            # before the fast-weights loop. One MLP call for the whole sequence.
            gates = None
            if cfg.use_write_gate:
                gates = torch.sigmoid(self.gate_mlp[layer](x))  # (B, S, 1)

            W_fast = torch.zeros(
                B, cfg.d_model, cfg.d_model, device=x.device, dtype=x.dtype,
            )
            fast_reads = []
            for t in range(S):
                gate_t = gates[:, t, 0] if gates is not None else None
                W_fast, read_t = self._fast_weight_step(
                    W_fast,
                    q_flat[:, t, :],
                    k_flat[:, t, :],
                    v_flat[:, t, :],
                    cfg.lambda_decay,
                    cfg.eta_write,
                    outer_scale,
                    use_delta_rule=cfg.use_delta_rule,
                    write_gate=gate_t,
                )
                fast_reads.append(read_t)
            fast_out = torch.stack(fast_reads, dim=1)  # (B, S, D)

            x = x + self.W_out[layer](attn) + fast_out
            gate, val = self.ff_in[layer](x).chunk(2, dim=-1)
            x = x + self.ff_out[layer](F.relu(gate) * val)

        return self.head(x)
