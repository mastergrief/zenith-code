"""Copy-augmented Small2DTransformer for pointer-copy transduction.

Adds a learned copy gate + pointer attention to the base substrate model.
At each decode step the model chooses: generate from vocabulary OR copy
from an input position. Digits get copied exactly (no transposition errors);
operators and structure tokens get generated from vocabulary.

Substrate-native: same d_head=2 invariant, same .pt format, same forward
signature (idx → logits). The copy mechanism is additive — removing it
recovers the base Small2DTransformer behavior.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from calm.llm_computer.model import Small2DConfig, Small2DTransformer


@dataclass
class CopyAugmentedConfig(Small2DConfig):
    """Extends Small2DConfig with copy mechanism parameters."""
    n_copy_heads: int = 4  # number of sub-heads dedicated to copy attention
    sep_token_id: int = 3  # <sep> token — boundary between prefix and decode


class CopyAugmentedTransformer(Small2DTransformer):
    """Small2DTransformer with pointer-copy mechanism.

    The model learns a per-position gate: p_copy ∈ [0, 1].
    - p_copy ≈ 1: copy token from input prefix via pointer attention
    - p_copy ≈ 0: generate token from vocabulary distribution
    - Blended: P(token) = p_copy * P_copy + (1-p_copy) * P_gen

    The copy attention uses dedicated heads (Q from decoder state,
    K/V from prefix positions). At d_head=2 each copy head is a tiny
    2D attention that learns to point at the right input position.
    """

    def __init__(self, config: CopyAugmentedConfig):
        super().__init__(config)
        self.copy_config = config
        d = config.d_model

        # Copy gate: hidden → scalar → sigmoid
        self.copy_gate = nn.Linear(d, 1, bias=True)
        # Initialize gate bias slightly negative so model starts by
        # preferring generation (existing behavior) and learns to copy.
        nn.init.constant_(self.copy_gate.bias, -2.0)

        # Pointer attention: project decoder hidden → copy Q/K
        # K is projected from the same hidden states but we'll mask
        # to only attend over prefix positions.
        copy_dim = config.n_copy_heads * config.d_head  # e.g. 4 * 2 = 8
        self.copy_q_proj = nn.Linear(d, copy_dim, bias=False)
        self.copy_k_proj = nn.Linear(d, copy_dim, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """idx: (B, S). Returns logits (B, S, vocab).

        Logits blend vocabulary generation and copy distributions.
        """
        B, S = idx.shape
        cfg = self.config

        # --- Standard transformer forward (same as parent) ---
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

        # --- Generation distribution (standard) ---
        gen_logits = self.head(x)  # (B, S, vocab)

        # --- Copy mechanism ---
        # Find <sep> positions to build prefix mask
        sep_id = self.copy_config.sep_token_id
        # For each batch element, prefix = positions before <sep>
        # prefix_mask: (B, S) — True for copyable prefix positions
        prefix_mask = self._build_prefix_mask(idx, sep_id)  # (B, S)

        # Copy gate: per-position decision
        p_copy = torch.sigmoid(self.copy_gate(x))  # (B, S, 1)

        # Pointer attention scores over prefix
        n_ch = self.copy_config.n_copy_heads
        dh = cfg.d_head
        cq = self.copy_q_proj(x).reshape(B, S, n_ch, dh)  # (B, S, H_c, dh)
        ck = self.copy_k_proj(x).reshape(B, S, n_ch, dh)  # (B, S, H_c, dh)

        # Scores: each decode position attends to all prefix positions
        # (B, S_decode, H_c, dh) × (B, S_prefix, H_c, dh) → (B, H_c, S, S)
        copy_scores = torch.einsum("bihd,bjhd->bhij", cq, ck)  # (B, H_c, S, S)

        # Mask: only attend to prefix positions, and respect causality
        causal = torch.triu(
            torch.ones(S, S, dtype=torch.bool, device=idx.device), diagonal=1
        )
        # ~prefix_mask: can't copy from non-prefix positions
        prefix_block = ~prefix_mask.unsqueeze(1).unsqueeze(1).expand_as(copy_scores)
        copy_scores = copy_scores.masked_fill(causal, float("-inf"))
        copy_scores = copy_scores.masked_fill(prefix_block, float("-inf"))

        # Average across copy heads, get distribution over source positions
        copy_scores_avg = copy_scores.mean(dim=1)  # (B, S, S)
        copy_attn = F.softmax(copy_scores_avg, dim=-1)  # (B, S, S)

        # Convert copy attention to vocabulary distribution:
        # scatter source token identities weighted by attention
        copy_logits = torch.zeros_like(gen_logits)  # (B, S, vocab)
        src_tokens = idx.unsqueeze(1).expand(B, S, S)  # (B, S, S) — token at each source pos
        copy_logits.scatter_add_(2, src_tokens, copy_attn)

        # Blend: p_copy * copy_dist + (1-p_copy) * gen_dist
        gen_probs = F.softmax(gen_logits, dim=-1)
        blended = p_copy * copy_logits + (1 - p_copy) * gen_probs  # (B, S, vocab)

        # Return log-probs for CE loss compatibility
        # Add small epsilon to avoid log(0)
        return torch.log(blended + 1e-10)

    def _build_prefix_mask(self, idx: torch.Tensor, sep_id: int) -> torch.Tensor:
        """Build mask: True for positions in the NL prefix (before <sep>).

        For each batch element, finds <sep> and marks all positions before it.
        """
        B, S = idx.shape
        # Find first <sep> in each batch element
        is_sep = (idx == sep_id)  # (B, S)
        # If no sep found, treat entire sequence as prefix (fallback)
        has_sep = is_sep.any(dim=1)  # (B,)
        # argmax on bool gives first True position
        sep_pos = is_sep.float().argmax(dim=1)  # (B,) — position of first <sep>
        # For sequences without <sep>, set sep_pos to S (nothing is prefix)
        sep_pos = torch.where(has_sep, sep_pos, torch.tensor(S, device=idx.device))

        # mask[b, i] = True if i < sep_pos[b]
        positions = torch.arange(S, device=idx.device).unsqueeze(0)  # (1, S)
        mask = positions < sep_pos.unsqueeze(1)  # (B, S)
        return mask


def build_copy_augmented_hrm(
    vocab_size: int = 80, d_model: int = 64, n_heads: int = 32,
    n_layers: int = 4, d_ffn: int = 128, max_len: int = 96,
    n_copy_heads: int = 4, sep_token_id: int = 3,
    use_hard_max: bool = False,
) -> CopyAugmentedTransformer:
    """Build a copy-augmented substrate HRM for training."""
    cfg = CopyAugmentedConfig(
        vocab_size=vocab_size, d_model=d_model, n_heads=n_heads,
        n_layers=n_layers, d_ffn=d_ffn, max_len=max_len,
        n_copy_heads=n_copy_heads, sep_token_id=sep_token_id,
        use_hard_max=use_hard_max,
    )
    assert cfg.d_head == 2, f"d_head must be 2, got {cfg.d_head}"
    return CopyAugmentedTransformer(cfg)
