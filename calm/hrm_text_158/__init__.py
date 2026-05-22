"""HRM-Text-1.58: source-faithful port of sapientinc/HRM-Text.

Per task #51, codex msg 1779451257744 (Phase 1 implement +1).

Upstream source pinned at SHA 056c4ecad217933b9db33dfb22e30a2f511315ed.
Faithful audit: RESEARCH/HRM-Text-1.58/00_ARCHITECTURE.md.
Deviation list:  RESEARCH/HRM-Text-1.58/01_DEVIATIONS.md.

This module CONTAINS the HRM-Text architecture only. NO DeltaNet,
NO copy-transducer, NO Small2DTransformer substrate, NO ACT/Q-head.
Substrate compilation is preserved as adjacent card/facade layer,
NOT integrated internally here.
"""
from calm.hrm_text_158.config import (
    TransformerConfig,
    HierarchicalReasoningModelConfig,
    LMHeadConfig,
)
from calm.hrm_text_158.layers import (
    LinearInit,
    ScaledEmbeddingInit,
    RotaryEmbedding,
    Attention,
    SwiGLU,
    build_prefix_lm_mask,
    trunc_normal_init_,
    find_multiple,
)
from calm.hrm_text_158.transformer import TransformerBlock, Transformer
from calm.hrm_text_158.hrm import (
    HierarchicalReasoningModelRecurrentBlock,
    HierarchicalReasoningModel,
)
from calm.hrm_text_158.lm_head import LMHead

__all__ = [
    "TransformerConfig",
    "HierarchicalReasoningModelConfig",
    "LMHeadConfig",
    "LinearInit",
    "ScaledEmbeddingInit",
    "RotaryEmbedding",
    "Attention",
    "SwiGLU",
    "build_prefix_lm_mask",
    "trunc_normal_init_",
    "find_multiple",
    "TransformerBlock",
    "Transformer",
    "HierarchicalReasoningModelRecurrentBlock",
    "HierarchicalReasoningModel",
    "LMHead",
]
