"""LLM-Computer prototype — compile gate-graph programs to transformer weights.

Based on Percepta's March 2026 research (see RESEARCH/01-03). Implements:
  - Small2DTransformer: d_head=2 hard-max transformer, executable by weights.
  - HullKVCache: online 2D convex-hull KV cache for O(log t) attention.
  - Gate-graph IR (LookUp + ReGLU + linear wiring) and a simple compiler
    that instantiates Small2DTransformer weights from a gate graph.

Designed as a library so it can later be folded into HRM's decoder
(see plan: "embed into HRM decoder" — Round 1e+).
"""

from calm.llm_computer.model import Small2DTransformer, Small2DConfig
from calm.llm_computer.hull_cache import HullKVCache

__all__ = ["Small2DTransformer", "Small2DConfig", "HullKVCache"]
