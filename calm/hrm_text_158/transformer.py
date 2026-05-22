"""HRM-Text-1.58 Transformer (backbone stack).

Source: sapientinc/HRM-Text SHA 056c4ec, `models/transformer.py:65-128`.
"""
from __future__ import annotations

from typing import Optional

import torch.nn.functional as F
from torch import Tensor, nn

from calm.hrm_text_158.config import TransformerConfig
from calm.hrm_text_158.layers import (
    Attention,
    RotaryEmbedding,
    SwiGLU,
)


class TransformerBlock(nn.Module):
    """Pre-norm or post-norm attention + SwiGLU block.

    Port of `sapientinc/HRM-Text/models/transformer.py:65-96`.
    """
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        init_cfg = config.init_config
        self.attn = Attention(
            hidden_size=config.hidden_size,
            head_dim=config.hidden_size // config.num_heads,
            num_heads=config.num_heads,
            num_key_value_heads=config.num_heads,
            attn_type=config.attn_type,
            init_std_in=init_cfg.in_std,
            init_std_out=init_cfg.attn_out_std,
        )
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            init_std_in=init_cfg.in_std,
            init_std_out=init_cfg.ff_out_std,
        )
        self._norm_type = config.norm_type
        self._norm_eps = config.norm_eps

    def norm(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.shape[-1],), eps=self._norm_eps)

    def forward(self, x: Tensor, **seq_info) -> Tensor:
        if self._norm_type == "pre":
            x = x + self.attn(self.norm(x), **seq_info)
            return x + self.mlp(self.norm(x))
        elif self._norm_type == "post":
            x = self.norm(x + self.attn(x, **seq_info))
            return self.norm(x + self.mlp(x))
        else:
            raise NotImplementedError(f"norm_type={self._norm_type!r}")


class Transformer(nn.Module):
    """Stack of TransformerBlocks with optional RoPE + final norm.

    Port of `sapientinc/HRM-Text/models/transformer.py:99-128`.
    """
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        init_cfg = config.init_config
        # Hint for LMHead init: same dim/std for input and output (per upstream)
        self.head_hint = {
            "in": {"dim": config.hidden_size, "init_std": init_cfg.in_std},
            "out": {"dim": config.hidden_size, "init_std": init_cfg.in_std},
        }
        # Position embeddings
        if config.pos_emb_type == "rope":
            assert config.rope_theta is not None
            self.rotary_emb: Optional[RotaryEmbedding] = RotaryEmbedding(
                config.hidden_size // config.num_heads,
                config.max_seq_len,
                base=config.rope_theta,
            )
        else:
            self.rotary_emb = None
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self._norm_type = config.norm_type
        self._norm_eps = config.norm_eps

    def norm_f(self, x: Tensor) -> Tensor:
        """Final norm: applied if pre-norm. Port of `transformer.py:113-116`."""
        if self._norm_type == "pre":
            return F.rms_norm(x, (x.shape[-1],), eps=self._norm_eps)
        return x

    def forward(
        self,
        x: Tensor,
        position_ids: Optional[Tensor] = None,
        sep_positions: Optional[Tensor] = None,
        **seq_info,
    ) -> Tensor:
        cos_sin = self.rotary_emb(position_ids) if self.rotary_emb is not None else None
        for layer in self.layers:
            x = layer(x, cos_sin=cos_sin, sep_positions=sep_positions, **seq_info)
        return self.norm_f(x)
