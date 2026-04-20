"""MILP scheduler — stub + graceful-fallback wrapper.

Upstream reference: sjmoran/transformer-vm @ 6cfee30 (Percepta Core).
File: `transformer_vm/scheduler/milp.py` (814 LOC).

The greedy `auto_schedule()` in `schedule.py` handles the project's
current compiled-program set (29 programs, ≤ 20 gates each) without
slot pressure. CLAUDE.md explicitly defers MILP: "MILP scheduling
(RESEARCH/03 §6) deferred until programs hit ~30+ gates with real
slot pressure."

This module provides the API surface for the deferred port:

    from calm.llm_computer.milp_schedule import milp_schedule

    plan = milp_schedule(graph, max_layers=None, max_ffn=None)
    if plan is None:
        plan = auto_schedule(graph)     # greedy fallback

When PuLP is unavailable (current state), `milp_schedule()` returns
`None` and the caller falls back cleanly. When PuLP is installed and
the full upstream port lands, the same call returns an optimal
phase-assigned schedule that minimizes `d_model = 2 * D_half` across
the 4 phases: attention (LookUp), persist1, FFN (ReGLU), persist2.

Upstream algorithm (for reference when porting):
  1. Build dependency graph from _all_dims + _all_lookups
  2. For each gate G and each layer L, introduce LpBinary(G, L)
  3. Constraint: each gate in exactly one layer
  4. Constraint: dependency gates scheduled ≤ dependent gates
  5. Minimize: max over all (layer, phase) boundaries of
       ceil(active_dims / 2) + n_lookup_heads
  6. Solve (default CBC), read back LpVariable values as assignments
  7. Apply interval-coloring for register allocation

Port blockers:
  - PuLP dep (5 MB wheel) + optional CBC solver binary
  - Refactor `_all_dims`/_all_lookups` globals to graph-local lists
    (the project's `GateGraph.add()` already uses per-graph lists;
    the port would thread through `GateGraph` explicitly)
  - Adapt upstream's `ProgramGraph.position/inv_log_pos/position_sq`
    to the project's `PosEmbed` primitive

See `.claude/rules/tracing_roadmap.md` for when to pull this trigger.
"""

from __future__ import annotations

from typing import Any, Optional


# Detect PuLP at import — cheap check; actual solver pick happens per
# call in a full port.
try:
    import pulp                           # noqa: F401 — import-only probe
    _PULP_AVAILABLE = True
except ImportError:
    _PULP_AVAILABLE = False


def is_available() -> bool:
    """True iff the MILP solver dependency (PuLP) is importable.

    Callers should branch on this before invoking `milp_schedule` to
    decide whether to delegate to greedy. Equivalent to checking the
    return value of `milp_schedule()` but avoids the call overhead.
    """
    return _PULP_AVAILABLE


def milp_schedule(
    graph: Any,                           # GateGraph — untyped to avoid circular import
    *,
    max_layers: Optional[int] = None,
    max_ffn: Optional[int] = None,
    log: Optional[Any] = None,
) -> Optional[Any]:
    """Compute an optimal phase-assigned schedule for `graph`.

    Returns:
        - a schedule dict (format matching the port target) when PuLP
          is installed AND the port has landed
        - None when PuLP is unavailable OR the port hasn't been done
          yet; caller should fall back to `schedule.auto_schedule(graph)`

    The stub currently returns None unconditionally — this reserves the
    API contract for the future port without forcing a dependency now.
    Upgrade path: replace this function body with the ported upstream
    `milp_schedule()` once PuLP is added to requirements.
    """
    if not _PULP_AVAILABLE:
        return None

    # PuLP is available but the full port hasn't landed yet.
    # Return None so callers fall back to greedy cleanly.
    # TODO: port upstream milp.py:milp_schedule — see module docstring.
    return None
