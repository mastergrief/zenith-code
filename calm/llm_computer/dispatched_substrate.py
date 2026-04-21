"""DispatchedSubstrate — full-capacity mechanisms + learned per-position gate.

Round 9 — the ultimate unified architecture per the R5-R8 arc:

  * Every mechanism runs at FULL d_model capacity (no partition).
  * Mechanisms compose additively into the residual stream.
  * A LEARNED per-position soft gate routes contribution mass between them.
  * Compiled programs can slot in as frozen mechanisms alongside trained.
  * One forward pass, N mechanisms in parallel, shared Q/K/V projections.

R5-R8 lessons applied:
  R5:  pure DeltaNet at d_head=2 caps at ~20% on random-KV (capacity).
  R6a: PT + DeltaNet additive hybrid → 100%. Copy path routes around
       DeltaNet's weakness. Parallel composition works.
  R7:  pure DeltaNet without copy → 92%. Mechanism alone sufficient
       but copy adds the last 8pp.
  R8:  Sub-head PARTITION of softmax+delta+copy → 44% @ ep25. Capacity
       sharing loses. COMPETITIVE composition hurts.
  R9:  Full-capacity parallel mechanisms + learned gate → hypothesis:
       matches or exceeds R6a's 100%.

Architecture per layer:

  q, k, v = W_qkv(x).chunk(3, -1)      # shared across mechanisms
  out_softmax = causal_softmax_attn(q, k, v)              # full capacity
  out_delta   = householder_delta(L2(SiLU(q)), L2(SiLU(k)), v, β(x))  # full
  out_copy    = prefix_masked_softmax_attn(q, k, v, mask) # full

  gates = softmax(W_gate(x))                              # (B, S, 3)
  attn  = gates[..., 0:1] * out_softmax
        + gates[..., 1:2] * out_delta
        + gates[..., 2:3] * out_copy

  x = x + W_out(attn)
  x = x + FFN(x)

Parameters added over Small2DTransformer:
  * 1× beta_head per layer (d_model × 1 + 1 bias)
  * 1× gate_head per layer (d_model × n_mechanisms + n_mechanisms bias)
  Total: ~260-1000 params per layer × n_layers

The gate learns WHICH mechanism each position needs. At a digit-copy
position, gates might concentrate on copy. At an operator-generate
position, gates might concentrate on softmax+delta. Compiled-mechanism
slots (frozen) get routing learned for them without modifying their
weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from calm.llm_computer.model import Small2DConfig, Small2DTransformer


@dataclass
class DispatchedSubstrateConfig(Small2DConfig):
    """Small2DConfig + dispatched-mechanisms parameters.

    mechanisms: list of mechanism names to run per layer. Order matters —
    gate output index i corresponds to mechanisms[i].
    Supported: "softmax", "delta", "copy".
    """
    mechanisms: list[str] = field(
        default_factory=lambda: ["softmax", "delta", "copy"]
    )
    sep_token_id: int = 3


class DispatchedSubstrateTransformer(Small2DTransformer):
    """Full-capacity parallel mechanisms + learned per-position dispatch."""

    def __init__(self, config: DispatchedSubstrateConfig):
        super().__init__(config)
        self.d_config = config
        n_mech = len(config.mechanisms)

        # β_head per layer for delta mechanism (scalar sigmoid).
        self.beta_heads = nn.ModuleList([
            nn.Linear(config.d_model, 1, bias=True)
            for _ in range(config.n_layers)
        ])
        for h in self.beta_heads:
            with torch.no_grad():
                h.bias.fill_(0.0)

        # Gate_head per layer: emits logits over mechanisms.
        self.gate_heads = nn.ModuleList([
            nn.Linear(config.d_model, n_mech, bias=True)
            for _ in range(config.n_layers)
        ])
        # Initialize gate biases to zero → uniform dispatch at init.
        for h in self.gate_heads:
            with torch.no_grad():
                h.bias.zero_()

    @staticmethod
    def _build_prefix_mask(idx: torch.Tensor, sep_id: int) -> torch.Tensor:
        B, S = idx.shape
        is_sep = (idx == sep_id)
        has_sep = is_sep.any(dim=1)
        sep_pos = is_sep.float().argmax(dim=1)
        sep_pos = torch.where(has_sep, sep_pos, torch.tensor(S, device=idx.device))
        positions = torch.arange(S, device=idx.device).unsqueeze(0)
        return positions < sep_pos.unsqueeze(1)

    @staticmethod
    def _softmax_attn(q, k, v):
        """(B, H, S, Dh) × (B, H, S, Dh) → (B, H, S, Dh). Causal mask."""
        S = q.shape[-2]
        scores = torch.einsum("bhid,bhjd->bhij", q, k)
        causal = torch.triu(
            torch.ones(S, S, dtype=torch.bool, device=q.device), diagonal=1
        )
        scores = scores.masked_fill(causal, float("-inf"))
        w = F.softmax(scores, dim=-1)
        return torch.einsum("bhij,bhjd->bhid", w, v)

    @staticmethod
    def _copy_attn(q, k, v, prefix_mask):
        """Softmax attention over INPUT PREFIX positions only."""
        B, H, S, Dh = q.shape
        scores = torch.einsum("bhid,bhjd->bhij", q, k)
        causal = torch.triu(
            torch.ones(S, S, dtype=torch.bool, device=q.device), diagonal=1
        )
        scores = scores.masked_fill(causal, float("-inf"))
        non_prefix = ~prefix_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, S)
        scores = scores.masked_fill(non_prefix.expand(-1, H, S, -1), float("-inf"))
        # Identify rows with no valid key; zero output there (not NaN).
        no_valid = (scores == float("-inf")).all(dim=-1, keepdim=True)
        scores = scores.masked_fill(no_valid, 0.0)
        w = F.softmax(scores, dim=-1)
        out = torch.einsum("bhij,bhjd->bhid", w, v)
        out = out.masked_fill(no_valid, 0.0)
        return out

    def _delta_attn(self, q, k, v, beta):
        """Householder recurrence on FULL (B, D, D) state. Shapes (B,H,S,Dh)."""
        B, H, S, Dh = q.shape
        D = H * Dh
        q_f = q.transpose(1, 2).reshape(B, S, D)
        k_f = k.transpose(1, 2).reshape(B, S, D)
        v_f = v.transpose(1, 2).reshape(B, S, D)

        q_feat = F.normalize(F.silu(q_f), p=2, dim=-1, eps=1e-6)
        k_feat = F.normalize(F.silu(k_f), p=2, dim=-1, eps=1e-6)

        S_state = torch.zeros(B, D, D, device=q.device, dtype=q.dtype)
        outs = []
        for t in range(S):
            k_t = k_feat[:, t, :]
            v_t = v_f[:, t, :]
            q_t = q_feat[:, t, :]
            beta_t = beta[:, t, :]
            v_old = torch.einsum("bij,bj->bi", S_state, k_t)
            delta = (v_old - v_t) * beta_t
            update = torch.einsum("bi,bj->bij", delta, k_t)
            S_state = S_state - update
            out_t = torch.einsum("bij,bj->bi", S_state, q_t)
            outs.append(out_t)
        out_flat = torch.stack(outs, dim=1)  # (B, S, D)
        return out_flat.reshape(B, S, H, Dh).transpose(1, 2)

    def forward(self, idx: torch.Tensor, return_gates: bool = False):
        """Returns (logits, gate_history). gate_history[l] is (B, S, n_mech)."""
        B, S = idx.shape
        cfg = self.d_config
        pos_idx = torch.arange(S, device=idx.device)
        x = self.tok(idx) + self.pos(pos_idx)

        has_copy = "copy" in cfg.mechanisms
        prefix_mask = self._build_prefix_mask(idx, cfg.sep_token_id) if has_copy else None
        gate_history = []

        for layer in range(cfg.n_layers):
            qkv = self.W_qkv[layer](x)
            qkv = qkv.reshape(B, S, 3, cfg.n_heads, cfg.d_head)
            q, k, v = qkv.permute(2, 0, 3, 1, 4)  # 3 × (B, H, S, Dh)

            # Compute each mechanism's output at full capacity.
            outs = []
            for m in cfg.mechanisms:
                if m == "softmax":
                    o = self._softmax_attn(q, k, v)
                elif m == "delta":
                    beta = torch.sigmoid(self.beta_heads[layer](x))
                    o = self._delta_attn(q, k, v, beta)
                elif m == "copy":
                    o = self._copy_attn(q, k, v, prefix_mask)
                else:
                    raise ValueError(f"unknown mechanism: {m}")
                outs.append(o)

            # Learned per-position soft dispatch.
            gate_logits = self.gate_heads[layer](x)  # (B, S, n_mech)
            gates = F.softmax(gate_logits, dim=-1)
            if return_gates:
                gate_history.append(gates.detach())

            # Weighted sum over mechanisms; gate broadcasts across (H, Dh).
            # outs[i]: (B, H, S, Dh); gates[..., i:i+1]: (B, S, 1)
            # Move S axis to match: gates → (B, 1, S, 1)
            g = gates.transpose(-1, -2).unsqueeze(-1)  # (B, n_mech, S, 1)
            attn = sum(
                g[:, i:i+1, :, :] * outs[i]
                for i in range(len(outs))
            )
            attn = attn.transpose(1, 2).reshape(B, S, cfg.d_model)

            x = x + self.W_out[layer](attn)
            gate, val = self.ff_in[layer](x).chunk(2, dim=-1)
            x = x + self.ff_out[layer](F.relu(gate) * val)

        logits = self.head(x)
        if return_gates:
            return logits, gate_history
        return logits


def build_dispatched_substrate(
    vocab_size: int = 82, d_model: int = 64, n_heads: int = 32,
    n_layers: int = 4, d_ffn: int = 128, max_len: int = 96,
    sep_token_id: int = 3, use_hard_max: bool = False,
    mechanisms: list[str] = None,
) -> DispatchedSubstrateTransformer:
    """Canonical R9 config: softmax + delta + copy mechanisms."""
    if mechanisms is None:
        mechanisms = ["softmax", "delta", "copy"]
    cfg = DispatchedSubstrateConfig(
        vocab_size=vocab_size, d_model=d_model, n_heads=n_heads,
        n_layers=n_layers, d_ffn=d_ffn, max_len=max_len,
        use_hard_max=use_hard_max,
        mechanisms=mechanisms, sep_token_id=sep_token_id,
    )
    assert cfg.d_head == 2, f"d_head must be 2, got {cfg.d_head}"
    return DispatchedSubstrateTransformer(cfg)
