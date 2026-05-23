"""Language-wrapper finite-support audit specs (codex msg
1779559495228-f863199b +1 implement L0a as the first language-axis
rung over validated R0..R1b9 math primitives).

PARALLEL audit surface to `exhaustive_supports.py` (math A0). Kept in
a separate module so:

- Math A0 export (`EXHAUSTIVE_ACTIVE_RUNGS`, `EXHAUSTIVE_EXPECTED_AGGREGATE=1255`)
  stays pure and stable as the R1b9 chain head baseline.
- Language audit grows independently as L0a -> L0b -> L0c rungs land.
- Probe JSON reports `math` and `language` aggregates separately
  (no blended single-number aggregate).

Slice E.1 adds L0c (`<expr> equals what?` interrogative-suffix form)
as the third language-axis rung over R0..R1b9 primitives. Codex msg
1779571151811-d3f6bc4f +1 implement; LANGUAGE_EXPECTED_AGGREGATE
extends 460 -> 690 with L0c contributing its own 230-row support.

L0a/L0b/L0c each carry a BOUNDED STRATIFIED 230-row support with a
single paraphrase template over R0..R1b9 math primitives. Multiplicity
under the default training recipe (n_train=10000, replay_ratio=0.65)
is ~19x against unique_train_count=184, well above the 10x floor
required to disambiguate language acquisition from undercoverage.

Pure data assembly; zero model deps.
"""
from __future__ import annotations

from typing import Callable

from calm.hrm_text_158.curriculum.generators import (
    _enumerate_partition_l0a,
    _enumerate_partition_l0b,
    _enumerate_partition_l0c,
)


LANGUAGE_ACTIVE_RUNGS: tuple[str, ...] = ("L0a", "L0b", "L0c")


def _l0a_support(seed: int = 42) -> list[tuple[str, int, str]]:
    """Full 230-row L0a bounded stratified support (train + held).

    Each row is a (question, expected, source_rung) triple. source_rung
    identifies which R0..R1b9 math primitive the L0a row wraps, used
    for per-source-rung breakdown in audit JSON.

    Counts:
      R0:           20 (10 one_digit + 10 two_digit)
      R1_plus_0:    10 (A plus 0)
      R1_0_plus_A:  10 (0 plus A)
      R1_minus_0:   10 (A minus 0)
      R1b1..R1b9:   each 20 (9 one_digit + 11 two_digit)
      TOTAL:        230

    Source-rung labels for R1's three sub-templates are distinct
    ("R1_plus_0", "R1_0_plus_A", "R1_minus_0") so per-bucket reporting
    can separately track each identity-bridge variant.
    """
    train, held = _enumerate_partition_l0a(seed)
    rows: list[tuple[str, int, str]] = []
    for r in train + held:
        rows.append((r["question"], r["expected"], r["source_rung"]))
    return rows


def _l0b_support(seed: int = 42) -> list[tuple[str, int, str]]:
    """Full 230-row L0b bounded stratified support (train + held).

    Codex msg 1779567887201-1cf4f485 +1 Slice D.1 implement. L0b mirrors
    L0a's shape exactly; only the question-string template differs
    (`calculate <expr>.` instead of `what's <expr>?`). Same 13 source
    buckets, same per-bucket counts. Aggregate adds 230 on top of L0a's
    230 → LANGUAGE_EXPECTED_AGGREGATE = 460 (extended again to 690 when
    L0c lands in Slice E.1).
    """
    train, held = _enumerate_partition_l0b(seed)
    rows: list[tuple[str, int, str]] = []
    for r in train + held:
        rows.append((r["question"], r["expected"], r["source_rung"]))
    return rows


def _l0c_support(seed: int = 42) -> list[tuple[str, int, str]]:
    """Full 230-row L0c bounded stratified support (train + held).

    Codex msg 1779571151811-d3f6bc4f +1 Slice E.1 implement. L0c mirrors
    L0a/L0b shape exactly; only the question-string template differs
    (`<expr> equals what?` interrogative-suffix form, instead of L0a's
    `what's <expr>?` question-prefix or L0b's `calculate <expr>.`
    imperative-prefix). Same 13 source buckets, same per-bucket counts.
    Aggregate adds 230 on top of L0a+L0b's 460 →
    LANGUAGE_EXPECTED_AGGREGATE = 690.
    """
    train, held = _enumerate_partition_l0c(seed)
    rows: list[tuple[str, int, str]] = []
    for r in train + held:
        rows.append((r["question"], r["expected"], r["source_rung"]))
    return rows


_BUILDERS: dict[str, Callable[[int], list[tuple[str, int, str]]]] = {
    "L0a": _l0a_support,
    "L0b": _l0b_support,
    "L0c": _l0c_support,
}

LANGUAGE_EXPECTED_COUNTS: dict[str, int] = {
    "L0a": 230,
    "L0b": 230,
    "L0c": 230,
}
LANGUAGE_EXPECTED_AGGREGATE: int = sum(
    LANGUAGE_EXPECTED_COUNTS[r] for r in LANGUAGE_ACTIVE_RUNGS
)  # 690 (L0a + L0b + L0c, each 230)


def build_language_supports(seed: int = 42) -> dict[str, list[tuple[str, int, str]]]:
    """Build full finite-support audit list for every active language rung.

    Returns dict keyed by language rung name in `LANGUAGE_ACTIVE_RUNGS`
    order. Per-rung counts match `LANGUAGE_EXPECTED_COUNTS`. Each row
    is (question, expected, source_rung).

    Pure: no model deps, deterministic per seed.
    """
    return {rung: _BUILDERS[rung](seed) for rung in LANGUAGE_ACTIVE_RUNGS}


def language_source_rung_buckets(rung: str) -> list[str]:
    """Return the list of unique source-rung labels for a language rung,
    in canonical reporting order. Used by probe to render per-bucket
    breakdowns.
    """
    if rung in ("L0a", "L0b", "L0c"):
        # L0b/L0c mirror L0a's bucket order exactly (codex msg
        # 1779567887201-1cf4f485 Slice D.1 + 1779571151811-d3f6bc4f
        # Slice E.1: same 13 source buckets, same per-bucket counts,
        # only the question-template surface differs).
        return [
            "R0",
            "R1_plus_0", "R1_0_plus_A", "R1_minus_0",
            "R1b1", "R1b2", "R1b3", "R1b4v2",
            "R1b5", "R1b6", "R1b7", "R1b8", "R1b9",
        ]
    raise ValueError(f"unknown language rung {rung!r}; valid: {LANGUAGE_ACTIVE_RUNGS}")
