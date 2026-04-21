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

# Problem-count tiers for hypothesis-test-iterate loop discipline.
# workflow.md §"The loop should be fast — under 5 min per round":
# use ITERATION_N during fix rounds (fast feedback), FINAL_N for
# the commit-ready baseline measurement. Every eval script with a
# configurable N parameter should default to ITERATION_N and bump
# to FINAL_N only for the round that goes into a commit receipt.
ITERATION_N: int = 5      # fast iteration — ~10 min wall time target
FINAL_N: int = 20         # commit-ready baseline — ~40 min wall time

# Rotation-state config file (daemon-friendly — no env var needed since
# the gemma daemon execs scripts with a fresh namespace but doesn't
# inherit env vars set by `bin/gemma-run`). Eval scripts read this
# file at exec time to pick their problem window. Empty / missing
# file ⇒ window 0 (first ITERATION_N problems, clean deltas across
# rounds). Set window=1,2,3... to rotate for generalization checks.
ROTATION_STATE_PATH: str = "/tmp/substrate_eval_rotation.json"


def read_rotation_state() -> dict:
    """Return {'window': int, 'n': Optional[int], 'final': bool}.
    Missing file or parse error ⇒ defaults (window=0, n=None, final=False).
    """
    import json
    from pathlib import Path
    p = Path(ROTATION_STATE_PATH)
    if not p.exists():
        return {"window": 0, "n": None, "final": False}
    try:
        cfg = json.loads(p.read_text())
        return {
            "window": int(cfg.get("window", 0)),
            "n": cfg.get("n"),
            "final": bool(cfg.get("final", False)),
        }
    except Exception:
        return {"window": 0, "n": None, "final": False}


def write_rotation_state(window: int = 0, n=None,
                         final: bool = False) -> None:
    """Set the rotation state for the next eval run."""
    import json
    from pathlib import Path
    Path(ROTATION_STATE_PATH).write_text(
        json.dumps({"window": window, "n": n, "final": final})
    )


def resolve_problem_window(default_n: int = ITERATION_N,
                           final_n: int = FINAL_N) -> tuple:
    """Return (n, skip) for the next eval run, reading rotation state.

    Semantics:
      final=True       → (final_n, 0)            full-size baseline
      n overridden     → (n, window * n)         explicit N + offset
      default          → (default_n, window * default_n)
    """
    cfg = read_rotation_state()
    if cfg["final"]:
        return (final_n, 0)
    n = cfg["n"] if cfg["n"] is not None else default_n
    skip = cfg["window"] * n
    return (n, skip)


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
