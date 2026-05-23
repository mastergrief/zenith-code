"""Exhaustive finite-support audit specs for the Phase 3 math curriculum.

Per codex msg 1779552750209-3218959b +1 implement after R1b8 commit 1a14a09
where A0 exhaustive 1071/1072 (R1b7 baseline) and 1161/1164 (R1b8 candidate)
audits caught the R1b2 boundary hole + digit-7 cluster + 0-plus-N cluster
that sampled probes hid. Promoted from /tmp helper to committed tooling
per codex's "standard gate before R1b9 or language" framing.

This module is PURE DATA assembly — builds the exhaustive (question, expected)
list per active math rung. Zero model deps; unit-testable in isolation.

Active rungs are listed explicitly via `EXHAUSTIVE_ACTIVE_RUNGS` rather than
derived generically from RUNG_NAMES because R1b4v2 is K=3 (not K=4 by name)
and R1b2 is subtraction (not addition). Explicit per-rung specs avoid
naming drift as the chain extends.
"""
from __future__ import annotations

from typing import Callable


# Active math chain for exhaustive audit (codex msg 1779552750209 explicit
# list spec; matches `build_rung_splits` default minus diagnosis-only +
# minus R3/R4/R5/R6 which are not yet at finite-support training stage).
EXHAUSTIVE_ACTIVE_RUNGS: tuple[str, ...] = (
    "R0", "R1", "R1b1", "R1b2", "R1b3",
    "R1b4v2", "R1b5", "R1b6", "R1b7", "R1b8", "R1b9",
)

# R1b10 is PARKED (codex msg 1779558351771-055c2265 after 3 failed
# promotion attempts: R1b10 K=9 supervision destabilizes R1b2 K=-1
# subtraction. K=9 support builder remains in `_BUILDERS` for explicit
# diagnostic probes via `--curriculum-rungs R1b10` but is NOT in
# `EXHAUSTIVE_ACTIVE_RUNGS`, so default A0 aggregate reverts to the
# R1b9 chain total (1255). Tests pin both invariants.
PARKED_DIAGNOSTIC_RUNGS: tuple[str, ...] = ("R1b10",)


def _r0_support() -> list[tuple[str, int]]:
    """R0: `what is N?` -> N for N in [0, 99] = 100 rows."""
    return [(f"what is {n}?", n) for n in range(0, 100)]


def _r1_support() -> list[tuple[str, int]]:
    """R1: identity-bridge 3 templates × A in [0, 99] = 300 rows.

    Templates: `A plus 0`, `0 plus A`, `A minus 0` — all output A.
    """
    rows: list[tuple[str, int]] = []
    for a in range(0, 100):
        rows.append((f"what is {a} plus 0?", a))
        rows.append((f"what is 0 plus {a}?", a))
        rows.append((f"what is {a} minus 0?", a))
    return rows


def _r1b1_support() -> list[tuple[str, int]]:
    """R1b1: K=1 plus, A in [1, 98] = 98 rows."""
    return [(f"what is {a} plus 1?", a + 1) for a in range(1, 99)]


def _r1b2_support() -> list[tuple[str, int]]:
    """R1b2: K=-1 minus (subtraction), A in [1, 99] = 99 rows."""
    return [(f"what is {a} minus 1?", a - 1) for a in range(1, 100)]


def _r1b3_support() -> list[tuple[str, int]]:
    """R1b3: K=2 plus, A in [1, 97] = 97 rows."""
    return [(f"what is {a} plus 2?", a + 2) for a in range(1, 98)]


def _r1b4v2_support() -> list[tuple[str, int]]:
    """R1b4v2: K=3 plus (active-chain successor to R1b4), A in [1, 96] = 96 rows.

    R1b4v2 name reflects the 'v2' partition (one-digit-exhaustive) over
    K=3, not K=4 — the K value comes from the template, not the name.
    """
    return [(f"what is {a} plus 3?", a + 3) for a in range(1, 97)]


def _r1b5_support() -> list[tuple[str, int]]:
    """R1b5: K=4 plus, A in [1, 95] = 95 rows."""
    return [(f"what is {a} plus 4?", a + 4) for a in range(1, 96)]


def _r1b6_support() -> list[tuple[str, int]]:
    """R1b6: K=5 plus, A in [1, 94] = 94 rows."""
    return [(f"what is {a} plus 5?", a + 5) for a in range(1, 95)]


def _r1b7_support() -> list[tuple[str, int]]:
    """R1b7: K=6 plus, A in [1, 93] = 93 rows."""
    return [(f"what is {a} plus 6?", a + 6) for a in range(1, 94)]


def _r1b8_support() -> list[tuple[str, int]]:
    """R1b8: K=7 plus, A in [1, 92] = 92 rows."""
    return [(f"what is {a} plus 7?", a + 7) for a in range(1, 93)]


def _r1b9_support() -> list[tuple[str, int]]:
    """R1b9: K=8 plus, A in [1, 91] = 91 rows."""
    return [(f"what is {a} plus 8?", a + 8) for a in range(1, 92)]


def _r1b10_support() -> list[tuple[str, int]]:
    """R1b10: K=9 plus, A in [1, 90] = 90 rows."""
    return [(f"what is {a} plus 9?", a + 9) for a in range(1, 91)]


_BUILDERS: dict[str, Callable[[], list[tuple[str, int]]]] = {
    "R0": _r0_support,
    "R1": _r1_support,
    "R1b1": _r1b1_support,
    "R1b2": _r1b2_support,
    "R1b3": _r1b3_support,
    "R1b4v2": _r1b4v2_support,
    "R1b5": _r1b5_support,
    "R1b6": _r1b6_support,
    "R1b7": _r1b7_support,
    "R1b8": _r1b8_support,
    "R1b9": _r1b9_support,
    "R1b10": _r1b10_support,
}

# Expected per-rung row counts (constant; tests assert this matches builders).
EXHAUSTIVE_EXPECTED_COUNTS: dict[str, int] = {
    "R0": 100,
    "R1": 300,
    "R1b1": 98,
    "R1b2": 99,
    "R1b3": 97,
    "R1b4v2": 96,
    "R1b5": 95,
    "R1b6": 94,
    "R1b7": 93,
    "R1b8": 92,
    "R1b9": 91,
    "R1b10": 90,
}
# Aggregate is computed from ACTIVE rungs only, not every known count.
# Parked diagnostic rungs (e.g. R1b10) keep their entries in
# `EXHAUSTIVE_EXPECTED_COUNTS` for explicit per-rung probes/tests but
# are excluded from the active-chain aggregate so default A0 reverts
# to the R1b9 chain head total.
EXHAUSTIVE_EXPECTED_AGGREGATE: int = sum(
    EXHAUSTIVE_EXPECTED_COUNTS[r] for r in EXHAUSTIVE_ACTIVE_RUNGS
)  # 1255


def build_exhaustive_supports() -> dict[str, list[tuple[str, int]]]:
    """Build exhaustive (question, expected) lists for every active rung.

    Returns dict keyed by rung name in `EXHAUSTIVE_ACTIVE_RUNGS` order.
    Per-rung counts match `EXHAUSTIVE_EXPECTED_COUNTS`.

    Pure: no model deps, deterministic, side-effect-free.
    """
    return {rung: _BUILDERS[rung]() for rung in EXHAUSTIVE_ACTIVE_RUNGS}


def validate_watch_rows(watch_rows: list[dict]) -> list[dict]:
    """Validate user-supplied watch-rows JSON schema.

    Each row must be a dict with required fields:
      - `key`: str (e.g. "R0:what_is_7"), human-readable label
      - `question`: str (the exact prompt sent to the model)
      - `expected`: int (the canonical correct answer)

    Raises ValueError on schema violation. Codex msg 1779552750209
    guardrail: fail loudly BEFORE ckpt load.
    """
    if not isinstance(watch_rows, list):
        raise ValueError(
            f"watch_rows must be a JSON list; got {type(watch_rows).__name__}"
        )
    for i, row in enumerate(watch_rows):
        if not isinstance(row, dict):
            raise ValueError(
                f"watch_rows[{i}] must be a JSON object; got {type(row).__name__}"
            )
        for field, expected_type in (("key", str), ("question", str), ("expected", int)):
            if field not in row:
                raise ValueError(
                    f"watch_rows[{i}] missing required field {field!r}"
                )
            if not isinstance(row[field], expected_type):
                raise ValueError(
                    f"watch_rows[{i}][{field!r}] must be {expected_type.__name__}; "
                    f"got {type(row[field]).__name__}"
                )
    return watch_rows
