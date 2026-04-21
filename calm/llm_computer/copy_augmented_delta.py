"""Copy-augmented DeltaNet — PT backbone swapped for DeltaNet recurrence.

Round 6a hybrid: pointer-copy transducer with DeltaNet sequence-mixing
instead of softmax attention. The copy path (copy_gate + pointer attn
over input positions) is unchanged; only the backbone producing
hidden states `h` is swapped.

Hypothesis (vs Round-5c null): PT's copy mechanism is a parallel path
that bypasses the backbone's generate-side output. When DeltaNet's
hidden states are noisy (as Round 5c showed on random-KV recall), the
copy gate leans on the copy path and downweights the generate path.
For digit-heavy structured outputs (PT's canonical workload), ~70-90%
of emitted tokens are copies — so backbone noise on the generate path
is partially shielded. Whether this is enough to preserve PT's 95-100%
autoregressive accuracy is the binary Round-6a gate.

Architecture — inherits `DeltaNetSmall2DTransformer`, which itself
inherits `Small2DTransformer`. The copy machinery (copy_gate,
copy_q_proj, copy_k_proj) is additive on top of DeltaNet's hidden
states. Forward:

    x = self._forward_backbone(idx)          # DeltaNet recurrence
    gen_logits = self.head(x)                # generate-path logits
    copy_scores over input prefix            # unchanged from PT
    blended = p_copy · copy_dist + (1-p_copy) · gen_probs
    return log(blended + ε)

State dict compatibility: this is NOT a drop-in load from a plain PT
checkpoint — the copy machinery is compatible but the backbone layers
differ (DeltaNet has β_head Linear per layer; softmax PT doesn't). So
a fresh training run is required. Regression tests live against the
existing PT checkpoint separately.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from calm.llm_computer.delta_rule import DeltaNetConfig, DeltaNetSmall2DTransformer


@dataclass
class CopyAugmentedDeltaConfig(DeltaNetConfig):
    """DeltaNetConfig + copy mechanism parameters (mirrors CopyAugmentedConfig)."""
    n_copy_heads: int = 4
    sep_token_id: int = 3


class CopyAugmentedDeltaNet(DeltaNetSmall2DTransformer):
    """DeltaNet backbone + pointer-copy decode mechanism."""

    def __init__(self, config: CopyAugmentedDeltaConfig):
        super().__init__(config)
        self.copy_config = config
        d = config.d_model

        self.copy_gate = nn.Linear(d, 1, bias=True)
        nn.init.constant_(self.copy_gate.bias, -2.0)

        copy_dim = config.n_copy_heads * config.d_head
        self.copy_q_proj = nn.Linear(d, copy_dim, bias=False)
        self.copy_k_proj = nn.Linear(d, copy_dim, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """idx: (B, S). Returns log-probs (B, S, vocab)."""
        B, S = idx.shape
        cfg = self.config

        # DeltaNet backbone produces per-position hidden states.
        x = self._forward_backbone(idx)

        # Generation distribution.
        gen_logits = self.head(x)

        # Copy mechanism (identical to CopyAugmentedTransformer).
        sep_id = self.copy_config.sep_token_id
        prefix_mask = self._build_prefix_mask(idx, sep_id)

        p_copy = torch.sigmoid(self.copy_gate(x))

        n_ch = self.copy_config.n_copy_heads
        dh = cfg.d_head
        cq = self.copy_q_proj(x).reshape(B, S, n_ch, dh)
        ck = self.copy_k_proj(x).reshape(B, S, n_ch, dh)

        copy_scores = torch.einsum("bihd,bjhd->bhij", cq, ck)

        causal = torch.triu(
            torch.ones(S, S, dtype=torch.bool, device=idx.device), diagonal=1,
        )
        prefix_block = ~prefix_mask.unsqueeze(1).unsqueeze(1).expand_as(copy_scores)
        copy_scores = copy_scores.masked_fill(causal, float("-inf"))
        copy_scores = copy_scores.masked_fill(prefix_block, float("-inf"))

        copy_scores_avg = copy_scores.mean(dim=1)
        copy_attn = F.softmax(copy_scores_avg, dim=-1)

        copy_logits = torch.zeros_like(gen_logits)
        src_tokens = idx.unsqueeze(1).expand(B, S, S)
        copy_logits.scatter_add_(2, src_tokens, copy_attn)

        gen_probs = F.softmax(gen_logits, dim=-1)
        blended = p_copy * copy_logits + (1 - p_copy) * gen_probs

        return torch.log(blended + 1e-10)

    @staticmethod
    def _build_prefix_mask(idx: torch.Tensor, sep_id: int) -> torch.Tensor:
        """Positions before first <sep> marked True (copyable input prefix)."""
        B, S = idx.shape
        is_sep = (idx == sep_id)
        has_sep = is_sep.any(dim=1)
        sep_pos = is_sep.float().argmax(dim=1)
        sep_pos = torch.where(has_sep, sep_pos, torch.tensor(S, device=idx.device))
        positions = torch.arange(S, device=idx.device).unsqueeze(0)
        return positions < sep_pos.unsqueeze(1)


def build_copy_augmented_delta(
    vocab_size: int = 80, d_model: int = 64, n_heads: int = 32,
    n_layers: int = 4, d_ffn: int = 128, max_len: int = 96,
    n_copy_heads: int = 4, sep_token_id: int = 3,
    use_hard_max: bool = False,
    use_softmax_attn: bool = False,
) -> CopyAugmentedDeltaNet:
    """Build a CopyAugmentedDeltaNet mirroring PT's default sizing."""
    cfg = CopyAugmentedDeltaConfig(
        vocab_size=vocab_size, d_model=d_model, n_heads=n_heads,
        n_layers=n_layers, d_ffn=d_ffn, max_len=max_len,
        n_copy_heads=n_copy_heads, sep_token_id=sep_token_id,
        use_hard_max=use_hard_max,
        use_delta_net=True, use_softmax_attn=use_softmax_attn,
    )
    assert cfg.d_head == 2, f"d_head must be 2, got {cfg.d_head}"
    return CopyAugmentedDeltaNet(cfg)
