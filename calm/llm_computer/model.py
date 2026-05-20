"""Small 2D-head transformer matching the Percepta paper.

Vanilla PyTorch. The architectural constraint is `d_head=2`; everything
else is standard `nn.MultiheadAttention` + gated-ReLU FFN + causal mask +
learned positional embeddings. What makes this a computer is the weights,
not the architecture — the weights are compiled from a gate graph (see
`compile.py`), not trained.

Supports `use_hard_max=True` to replace softmax attention with argmax
attention. Hard-max is what the paper's analytical correctness proofs
assume; softmax with large scaling temperature approximates it to
arbitrary precision but is slower to reason about.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Small2DConfig:
    """Configuration for Small2DTransformer.

    Paper defaults: d_model=36, n_heads=18, n_layers=7, d_ffn=36.
    We use tiny defaults (d_model=8, n_heads=4, n_layers=2) for the
    first prototype so we can test compile/execute cycles in ms on CPU.

    HRM-Text-derived flags (default off — bit-equivalence preserved):
      use_gated_attention: apply `sigmoid(gate) * attn_output` per Qwen3Next /
        HRM-Text 1B. Adds one Linear(d_model, d_model) per layer. Cheap.
      use_lecun_init: re-initialize every nn.Linear weight in the module
        tree with LeCun normal (std=sqrt(1/fan_in)). Per HRM-Text:
        better matched to sigmoid/tanh-gated paths than PyTorch's default
        kaiming_uniform(a=sqrt(5)) which assumes leaky-ReLU shape.
        Bias channels are NOT touched (preserves init contracts like
        CopyAugmentedDeltaNet's copy_gate_bias_init).
      use_prefix_lm: relax causal mask within the prompt prefix per HRM-Text
        `prefix_lm=true`. Tokens at positions [0, prefix_end) attend
        bidirectionally to each other; tokens at or after prefix_end stay
        causal w.r.t. everything. Requires a subclass-specific way to
        identify the prefix — base Small2DTransformer returns None from
        `_compute_prefix_mask(idx)`, so the flag is inert here.
        CopyAugmentedDeltaNet overrides the hook to derive the prefix
        from sep_token_id.
    """
    vocab_size: int = 32
    d_model: int = 8
    n_heads: int = 4
    n_layers: int = 2
    d_ffn: int = 8
    max_len: int = 256
    use_hard_max: bool = True
    use_gated_attention: bool = False
    use_lecun_init: bool = False
    use_prefix_lm: bool = False

    @property
    def d_head(self) -> int:
        assert self.d_model % self.n_heads == 0, "d_model must divide n_heads"
        return self.d_model // self.n_heads


class Small2DTransformer(nn.Module):
    """Minimal transformer compatible with compiled-weight execution.

    Matches the paper's `VanillaTransformer` snippet (doc 02 §8): learned
    token + position embeddings, MultiheadAttention layers, gated ReLU
    FFN, causal mask, linear head.

    `use_hard_max=True` replaces softmax with argmax — each query picks
    exactly one past position, matching the paper's analytical-correctness
    regime.
    """

    def __init__(self, config: Small2DConfig):
        super().__init__()
        self.config = config
        self.tok = nn.Embedding(config.vocab_size, config.d_model)
        self.pos = nn.Embedding(config.max_len, config.d_model)

        # Split QKV/out/FFN into pieces so a compiler can populate them per-head.
        self.W_qkv = nn.ModuleList([
            nn.Linear(config.d_model, 3 * config.d_model, bias=False)
            for _ in range(config.n_layers)
        ])
        self.W_out = nn.ModuleList([
            nn.Linear(config.d_model, config.d_model, bias=False)
            for _ in range(config.n_layers)
        ])
        self.ff_in = nn.ModuleList([
            nn.Linear(config.d_model, 2 * config.d_ffn, bias=False)
            for _ in range(config.n_layers)
        ])
        self.ff_out = nn.ModuleList([
            nn.Linear(config.d_ffn, config.d_model, bias=False)
            for _ in range(config.n_layers)
        ])
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # HRM-Text-derived: gated attention (Qwen3Next-style). Opt-in via config.
        # When use_gated_attention=False (default), these are not allocated.
        if config.use_gated_attention:
            self.attn_gate_proj = nn.ModuleList([
                nn.Linear(config.d_model, config.d_model, bias=False)
                for _ in range(config.n_layers)
            ])
        else:
            self.attn_gate_proj = None  # explicit; surfaces flag-state in repr

        # LeCun-normal init: re-init every Linear weight if flag set.
        # Subclasses that add their own Linear layers AFTER super().__init__()
        # must invoke `self._apply_lecun_init()` again at the end of their
        # __init__ if `config.use_lecun_init` is on.
        if config.use_lecun_init:
            self._apply_lecun_init()

    def _apply_lecun_init(self) -> None:
        """LeCun normal: weight ~ N(0, 1/fan_in). Touches nn.Linear weights
        only — leaves biases alone so subclass init contracts (e.g.
        CopyAugmentedDeltaNet's copy_gate_bias_init) stay intact.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                fan_in = module.weight.shape[1]
                std = (1.0 / max(1, fan_in)) ** 0.5
                nn.init.normal_(module.weight, mean=0.0, std=std)

    def _apply_lecun_init_to(self, roots) -> None:
        """Scoped LeCun normal: walk only the supplied module roots and
        re-init the nn.Linear weights within them. Used by Slice 8+ when a
        new weight bank is allocated AFTER an earlier broad `_apply_lecun_init`
        — calling the broad version again would re-init the existing L bank
        a second time with fresh RNG, silently breaking the n_iters=1/
        h_cycles=1 bit-equivalence guard for the new flag.

        `roots` may be a single nn.Module, a ModuleList, or any iterable
        of Modules. Bias-preserving as always.
        """
        if isinstance(roots, nn.Module):
            roots = [roots]
        for root in roots:
            for module in root.modules():
                if isinstance(module, nn.Linear):
                    fan_in = module.weight.shape[1]
                    std = (1.0 / max(1, fan_in)) ** 0.5
                    nn.init.normal_(module.weight, mean=0.0, std=std)

    def _compute_prefix_mask(self, idx: torch.Tensor) -> Optional[torch.Tensor]:
        """Hook: return (B, S) bool tensor where True marks prefix positions.

        Subclasses with a sep/marker concept (e.g. CopyAugmentedDeltaNet)
        override to derive the prefix from idx tokens. Base
        Small2DTransformer has no notion of prefix, returns None — which
        makes the use_prefix_lm flag a no-op at this level.
        """
        return None

    def _attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                   hard_max: bool,
                   gate: Optional[torch.Tensor] = None,
                   prefix_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Causal multi-head attention. q,k,v: (B, H, S, D_head).

        If `gate` is supplied (B, H, S, D_head), applies
        `sigmoid(gate) * out` before returning — HRM-Text-style gating.
        If `prefix_mask` is supplied (B, S) bool, queries within the prefix
        attend BIDIRECTIONALLY to keys within the prefix (causal mask is
        relaxed for the (prefix × prefix) block). Tokens at or after the
        prefix stay causal.
        When both gate=None and prefix_mask=None, behavior is identical
        to original causal softmax/argmax attention.
        """
        B, H, S, Dh = q.shape
        scores = torch.einsum("bhid,bhjd->bhij", q, k)  # (B, H, S, S)
        # Causal mask: position i can only see j <= i.
        causal = torch.triu(
            torch.ones(S, S, dtype=torch.bool, device=q.device), diagonal=1
        )
        if prefix_mask is not None:
            # prefix_mask: (B, S) — True where position is in the prefix block.
            # Build (B, S, S) "both endpoints in prefix" block; allow attention
            # there even when causal would have masked it.
            assert prefix_mask.shape == (B, S), (
                f"prefix_mask shape {tuple(prefix_mask.shape)} must be (B, S)=({B},{S})"
            )
            prefix_pair = prefix_mask.unsqueeze(2) & prefix_mask.unsqueeze(1)  # (B, S, S)
            # Final mask: causal AND NOT (both endpoints in prefix).
            mask = causal.unsqueeze(0) & ~prefix_pair  # (B, S, S)
            mask = mask.unsqueeze(1)  # (B, 1, S, S) — broadcast on head dim
        else:
            mask = causal
        scores = scores.masked_fill(mask, float("-inf"))
        if hard_max:
            # Replace softmax with one-hot over argmax. Tie-breaks: first match.
            idx = scores.argmax(dim=-1, keepdim=True)  # (B, H, S, 1)
            weights = torch.zeros_like(scores)
            weights.scatter_(-1, idx, 1.0)
        else:
            weights = F.softmax(scores, dim=-1)
        out = torch.einsum("bhij,bhjd->bhid", weights, v)  # (B, H, S, D_head)
        if gate is not None:
            out = torch.sigmoid(gate) * out
        return out

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """idx: (B, S). Returns logits (B, S, vocab)."""
        B, S = idx.shape
        cfg = self.config
        pos_idx = torch.arange(S, device=idx.device)
        x = self.tok(idx) + self.pos(pos_idx)
        # Compute prefix mask once per forward (used by every layer's attention
        # when use_prefix_lm is set AND a subclass populates _compute_prefix_mask).
        prefix_mask = (
            self._compute_prefix_mask(idx) if cfg.use_prefix_lm else None
        )
        for layer in range(cfg.n_layers):
            qkv = self.W_qkv[layer](x)  # (B, S, 3*D)
            qkv = qkv.reshape(B, S, 3, cfg.n_heads, cfg.d_head)
            q, k, v = qkv.permute(2, 0, 3, 1, 4)  # 3 × (B, H, S, Dh)
            # HRM-Text gated attention: compute gate per layer when flag on.
            gate = None
            if self.attn_gate_proj is not None:
                gate = self.attn_gate_proj[layer](x)  # (B, S, D)
                gate = gate.reshape(B, S, cfg.n_heads, cfg.d_head).transpose(1, 2)  # (B, H, S, Dh)
            attn = self._attention(
                q, k, v, hard_max=cfg.use_hard_max, gate=gate,
                prefix_mask=prefix_mask,
            )
            attn = attn.transpose(1, 2).reshape(B, S, cfg.d_model)  # (B, S, D)
            x = x + self.W_out[layer](attn)
            gate, val = self.ff_in[layer](x).chunk(2, dim=-1)
            x = x + self.ff_out[layer](F.relu(gate) * val)
        return self.head(x)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
