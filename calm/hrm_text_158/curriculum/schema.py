"""Phase 3 retention probe schema.

Per codex msg 1779457170889 + 1779458774209 routing: each rung's probe
must report parsed/exact/too_long PER PRIOR RUNG + canonical 17×23,
to detect catastrophic forgetting on prior rungs after new rung
training.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RungProbeResult:
    """Result of probing one ckpt across new-rung + ALL prior-rung held-outs.

    Per-rung dicts use the rung name (e.g. "R0", "R1", ..., "R7") as the key.
    """
    rung: str                                          # e.g. "R2"
    ckpt_path: str
    step: int
    n_params: int

    # Per-rung accuracy on held-out (parsed_correct / cap)
    rung_accuracy: dict[str, float] = field(default_factory=dict)
    rung_parsed: dict[str, int] = field(default_factory=dict)
    rung_exact: dict[str, int] = field(default_factory=dict)
    rung_too_long: dict[str, int] = field(default_factory=dict)
    rung_cap: dict[str, int] = field(default_factory=dict)

    # Canonical 17×23 probe (multiplication-rung mastery gate)
    canonical_17x23: dict = field(default_factory=dict)  # {decoded, parsed, exact_ok, parsed_ok, too_long}

    # Retention deltas vs PREVIOUS rung's probe of the same prior rungs
    # (None for the rung being currently trained; populated for all rungs prior to it)
    retention_delta: dict[str, float] = field(default_factory=dict)

    # Run wall + finite-check
    elapsed_sec: float = 0.0
    finite: bool = True
