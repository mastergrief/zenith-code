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

    # Keyed per-rung one_digit exhaustive audits (codex msg
    # 1779523412979-ff88b885 after R1b5 design). Each audit-eligible
    # rung's 9-row exhaustive check stored under its own key in
    # `one_digit_audits`. Required because R1b5 probe includes R1b4v2
    # in priors AND R1b5 itself — both must have audits stored, not
    # one overwriting the other.
    # Shape: {<rung_name>: {"exact": int, "cap": 9, "parsed": int,
    #   "too_long": int, "finite": bool, "rows": [...]}, ...}.
    # Empty dict if no audit-eligible rungs probed.
    one_digit_audits: dict[str, dict] = field(default_factory=dict)

    # Backcompat alias for legacy R1b4v2-only field (codex msg
    # 1779483673737-20ff22ab). Mirrors `one_digit_audits["R1b4v2"]`
    # when present, for older receipts that read this singular field.
    # NEW code should consume `one_digit_audits[<rung>]` directly.
    one_digit_audit: dict = field(default_factory=dict)

    # Retention deltas vs PREVIOUS rung's probe of the same prior rungs
    # (None for the rung being currently trained; populated for all rungs prior to it)
    retention_delta: dict[str, float] = field(default_factory=dict)

    # Run wall + finite-check
    elapsed_sec: float = 0.0
    finite: bool = True

    # R4a batched probe/eval diagnostics (codex msg 1779534977172-88a0cb6c).
    # Populated only when --use-batched-probe-eval is set. Maps actual
    # chunk batch_size → count of chunks at that size, aggregated across
    # all rung loops + canonical + audits in this probe. Empty dict on
    # the scalar (B=1) path. JSON keys are stringified ints.
    batched_chunk_size_hist: dict[int, int] = field(default_factory=dict)
