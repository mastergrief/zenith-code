"""
Substrate evaluation defaults — centralized ctx + thinking-budget constants.

Every substrate eval script (R51+, R52+, R53+) imports from here. Changing
these two numbers changes every eval consistently.

- EVAL_CTX_SIZE: pre-allocated KVCacheTq4 ceiling. Allocates tq4 KV for this
  many tokens regardless of prompt length. tq4 KV is ~3.6× smaller than
  fp16 KV, so 32K is ~700 MB additional VRAM at the 8 GB budget.
- EVAL_MAX_TOKENS: AdaptiveBudget output-token clamp. Tiered (trivial 2K
  → deep 32K) but clamped here to give a predictable upper bound.

Usage:

    from calm.llm_computer.eval_defaults import (
        EVAL_CTX_SIZE, EVAL_MAX_TOKENS, get_adaptive_budget,
    )

    budget, est = get_adaptive_budget(prompt)
    out = m.generate(prompt, tok, max_tokens=budget,
                     kv_max_len=EVAL_CTX_SIZE, use_tq4_kv=True)

Rationale: R53.25 receipt showed MAX_TOKENS 400 → 900 alone lifts
log_level_counts 0/0 → 6/6 — output budget was masking real capability.
Current defaults give AdaptiveBudget room to scale without ever truncating
real coding problems. Gemma 4 E4B is NIAH-validated to 200K single-needle
and trained at 131K, so 32K prompt+output budget is comfortably inside.
"""

from __future__ import annotations

from typing import Optional, Tuple

# Pre-allocated KV cache ceiling (prompt + output tokens, tq4 storage).
EVAL_CTX_SIZE: int = 32768

# AdaptiveBudget clamp — cap the per-prompt output-token budget here.
EVAL_MAX_TOKENS: int = 16384


def get_adaptive_budget(prompt: str,
                        precomputed: Optional[dict] = None,
                        pre_analysis: Optional[dict] = None,
                        ceiling: Optional[int] = None):
    """Per-prompt output budget via AdaptiveBudget, clamped to ceiling.

    Returns (budget, estimate). The estimate carries tier + reasoning for
    logging. Falls back to the ceiling if AdaptiveBudget import fails (so
    eval scripts keep working even if calm/adaptive.py is broken).
    """
    cap = ceiling if ceiling is not None else EVAL_MAX_TOKENS
    try:
        from calm.adaptive import AdaptiveBudget
        est = AdaptiveBudget().estimate(
            prompt, precomputed=precomputed, pre_analysis=pre_analysis,
        )
        return min(est.budget, cap), est
    except Exception:
        return cap, None


def budget_only(prompt: str, ceiling: Optional[int] = None) -> int:
    """Shortcut when only the budget int is needed (no estimate metadata)."""
    budget, _ = get_adaptive_budget(prompt, ceiling=ceiling)
    return budget
