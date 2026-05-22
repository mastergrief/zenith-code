"""HRM-Text-1.58 configs.

Source: sapientinc/HRM-Text SHA 056c4ec.
- TransformerConfig: ports `models/transformer.py:19-37`
- HierarchicalReasoningModelConfig: ports `models/baselines/hrm_nocarry_bp_warmup.py:11-23`
- LMHeadConfig: ports `models/lm_head.py:14-15`

Deviation from upstream: dataclasses instead of pydantic.BaseModel
(simpler, no extra dependency). Behaviorally equivalent on the
fields we use; we lose pydantic validation but our test harness
asserts shapes/values explicitly.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class InitConfig:
    """Computed initialization scales. Ported from
    `sapientinc/HRM-Text/models/transformer.py:12-16` (InitConfig).
    """
    in_std: float
    attn_out_std: float
    ff_out_std: float


def find_multiple(a: int, b: int) -> int:
    """Round a up to next multiple of b.
    Port of `sapientinc/HRM-Text/models/layers.py:19-20`.
    """
    return (-(a // -b)) * b


@dataclass
class TransformerConfig:
    """Backbone transformer config.

    Port of `sapientinc/HRM-Text/models/transformer.py:19-37` (TransformerConfig).
    Behavioral differences from upstream:
    - dataclass not pydantic.BaseModel (D1.8 implicit — minimal-dep deviation)
    """
    max_seq_len: int
    n_layers: int
    hidden_size: int
    num_heads: int
    expansion: float

    # Upstream defaults `attn_type: AttnType = "prefixlm"` (layers.py:16).
    attn_type: Literal["causal", "prefixlm"] = "prefixlm"

    init_type: Literal["fixed_normal", "lecun_normal", "megatron"] = "lecun_normal"
    init_std: Optional[float] = None

    norm_type: Literal["pre", "post"] = "pre"
    norm_eps: float = 1e-6

    pos_emb_type: Literal["rope", "none"] = "rope"
    rope_theta: Optional[float] = 10000.0

    @property
    def intermediate_size(self) -> int:
        """Automatic compute from expansion. Port of `transformer.py:42-46`.
        Matches GLU param count to vanilla transformer at same expansion.
        Rounded to multiple of 256.
        """
        return find_multiple(round(self.expansion * self.hidden_size * 2 / 3), 256)

    @property
    def init_config(self) -> InitConfig:
        """Init scales per init_type. Port of `transformer.py:48-62`."""
        if self.init_type == "fixed_normal":
            std = self.init_std if self.init_std is not None else 0.02
            in_std = attn_out_std = ff_out_std = std
        elif self.init_type == "lecun_normal":
            in_std = attn_out_std = 1.0 / math.sqrt(self.hidden_size)
            ff_out_std = 1.0 / math.sqrt(self.intermediate_size)
        elif self.init_type == "megatron":
            std = self.init_std if self.init_std is not None else 1.0 / math.sqrt(self.hidden_size)
            in_std = std
            attn_out_std = ff_out_std = std / math.sqrt(2.0 * self.n_layers)
        else:
            raise NotImplementedError(f"init_type={self.init_type!r} not supported")
        return InitConfig(in_std=in_std, attn_out_std=attn_out_std, ff_out_std=ff_out_std)


@dataclass
class HierarchicalReasoningModelConfig(TransformerConfig):
    """HRM-Text top-level config.

    Port of `sapientinc/HRM-Text/models/baselines/hrm_nocarry_bp_warmup.py:11-23`.
    Inherits TransformerConfig and adds H/L recurrence + bp_warmup fields.
    """
    half_layers: bool = False  # Divide n_layers by 2, split evenly H/L
    H_cycles: int = 2
    L_cycles: int = 3

    bp_warmup_ratio: float = 0.0
    bp_min_steps: int = 2
    bp_max_steps: int = 5

    # Per-config override for H_level. Upstream uses dict; we use dict.
    H_override: dict = field(default_factory=dict)


@dataclass
class LMHeadConfig:
    """LMHead config. Port of `models/lm_head.py:14-15`."""
    vocab_size: int
