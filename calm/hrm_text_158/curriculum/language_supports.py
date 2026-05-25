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
    _enumerate_partition_l0c1,
    _enumerate_partition_l0c2,
    _enumerate_partition_l0c2k1,
    _enumerate_partition_l0c2k1_edge,
    _enumerate_partition_l0c2k2,
    _enumerate_partition_l0c2k3,
    l0c2_band_expected_count,
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


# ---------------------------------------------------------------------------
# Exhaustive L0c — ONE language wrapper at MATH density (codex msg
# 1779692896701 plan-gate: "language to math density" = the `<expr> equals
# what?` wrapper over the FULL math-A0 exhaustive set, not the bounded 230
# stratified sample and not all three wrappers at once). Derived by
# transforming each math-A0 row (`what is <expr>?` -> `<expr> equals what?`),
# so count / per-source-rung counts / expected values match math A0 by
# construction (1255), and L0c1's 121 one_digit rows are a subset.
# SEED-INDEPENDENT (exhaustive). Keyed by source rung like
# build_exhaustive_supports() so the audit can do per-bucket reporting.
# ---------------------------------------------------------------------------
_L0C_MATH_PREFIX = "what is "


def _math_q_to_l0c(math_q: str) -> str:
    """`what is <expr>?` (math-A0 surface) -> `<expr> equals what?` (L0c
    surface). The math-A0 question format is invariant across
    build_exhaustive_supports(); assert it to fail LOUD rather than silently
    mis-wrap. Produces byte-identical strings to the L0c partition templates
    (e.g. `10 minus 1 equals what?`, `7 equals what?`)."""
    if not (math_q.startswith(_L0C_MATH_PREFIX) and math_q.endswith("?")):
        raise ValueError(f"unexpected math-A0 question format: {math_q!r}")
    expr = math_q[len(_L0C_MATH_PREFIX):-1]
    return f"{expr} equals what?"


def build_exhaustive_l0c_supports() -> dict[str, list[tuple[str, int]]]:
    """Exhaustive L0c support: the `<expr> equals what?` wrapper applied to
    the full math-A0 exhaustive support (R0..R1b9, 1255 rows). Returns a
    dict keyed by source rung — same shape as build_exhaustive_supports() —
    so the audit reports a per-source-rung breakdown parallel to math A0.
    Pure data assembly; zero model deps; seed-independent."""
    from calm.hrm_text_158.curriculum.exhaustive_supports import (
        build_exhaustive_supports,
    )
    return {
        rung: [(_math_q_to_l0c(q), e) for (q, e) in rows]
        for rung, rows in build_exhaustive_supports().items()
    }


L0C_EXHAUSTIVE_EXPECTED_COUNT: int = 1255


# ---------------------------------------------------------------------------
# L0c1 — one_digit-STRATUM precursor SUBSET of L0c (codex msg 1779636434289 Slice F.1).
# SEPARATE diagnostic/audit surface: deliberately NOT in LANGUAGE_ACTIVE_RUNGS
# and NOT in the canonical 690 aggregate. Audited via --l0c1-audit on its own
# JSON surface, never blended into --language-supports. 121 rows = exactly
# L0c's one_digit stratum, so L0c1 ⊂ L0c.
# ---------------------------------------------------------------------------
L0C1_EXPECTED_COUNT: int = 121


def _l0c1_support(seed: int = 42) -> list[tuple[str, int, str]]:
    """121-row L0c1 one_digit-stratum-precursor support (train + held).

    The one_digit-stratum subset of L0c (L0c1 ⊂ L0c). Same template surface
    (`<expr> equals what?`) and same 13 source buckets as L0c; only the
    two_digit-stratum rows are excluded (the R1 identity-bridge stratum
    stays, even where its sampled A is two-digit). Codex msg 1779636434289
    +1 Slice F.1 implement.
    """
    train, held = _enumerate_partition_l0c1(seed)
    rows: list[tuple[str, int, str]] = []
    for r in train + held:
        rows.append((r["question"], r["expected"], r["source_rung"]))
    return rows


def build_l0c1_support(seed: int = 42) -> dict[str, list[tuple[str, int, str]]]:
    """Build the standalone L0c1 audit surface: {"L0c1": [...121 rows...]}.

    Parallel to `build_language_supports` but for the L0c1 precursor audit.
    Single-key dict so the probe's language-audit machinery iterates it
    uniformly under `surface="l0c1"`. Deliberately NOT merged into
    `build_language_supports` (keeps the canonical 690 language aggregate
    and the L0c1 surface separate, per codex Slice F.1 constraint).
    """
    return {"L0c1": _l0c1_support(seed)}


# ---------------------------------------------------------------------------
# L0c2 — bounded-2-digit stair-step rung audit surface (F.4a rung + F.4-audit).
# SEPARATE surface like L0c1: deliberately NOT in LANGUAGE_ACTIVE_RUNGS, so the
# canonical 690 language aggregate (L0a+L0b+L0c) is preserved. Audited via
# --l0c2-audit on its own JSON surface. 230 rows = the F.4a stratified hard
# subset; the third field is the COMPOSITE `source_rung:operator` bucket so the
# R1b2:minus / `10 minus 1 -> 9` operator-specific failure class cannot hide.
# ---------------------------------------------------------------------------
L0C2_AUDIT_EXPECTED_COUNT: int = 230
L0C2K1_AUDIT_EXPECTED_COUNT: int = 24
L0C2K2_AUDIT_EXPECTED_COUNT: int = 79
L0C2K3_AUDIT_EXPECTED_COUNT: int = 127


def _l0c2_support(seed: int = 42) -> list[tuple[str, int, str]]:
    """230-row L0c2 bounded-2-digit audit support (train + held). Same template
    surface (`<expr> equals what?`) as L0c, but every row is 2-digit-hard and
    the bucket label is the composite `source_rung:operator` (e.g. `R1b2:minus`)
    so --l0c2-audit reports per-(source_rung x operator), preserving the
    operator-specific failure class. Mirrors `_l0c1_support`; the partition is
    the F.4a `_enumerate_partition_l0c2`."""
    train, held = _enumerate_partition_l0c2(seed)
    rows: list[tuple[str, int, str]] = []
    for r in train + held:
        rows.append((r["question"], r["expected"], f"{r['source_rung']}:{r['operator']}"))
    return rows


def build_l0c2_support(seed: int = 42) -> dict[str, list[tuple[str, int, str]]]:
    """Build the standalone L0c2 audit surface: {"L0c2": [...230 rows...]}.

    Parallel to `build_l0c1_support` but for the F.4 bounded-2-digit stair-step
    rung. Single-key dict so the probe's language-audit machinery iterates it
    uniformly under `surface="l0c2"`. Deliberately NOT merged into
    `build_language_supports` (keeps the canonical 690 language aggregate and
    the L0c2 surface separate, mirroring the L0c1 constraint)."""
    return {"L0c2": _l0c2_support(seed)}


def _l0c2_band_support(partition_builder, seed: int = 42) -> list[tuple[str, int, str]]:
    """Build an L0c2 K-band audit support from a banded partition."""
    train, held = partition_builder(seed)
    rows: list[tuple[str, int, str]] = []
    for r in train + held:
        rows.append((r["question"], r["expected"], f"{r['source_rung']}:{r['operator']}"))
    return rows


def _l0c2k1_support(seed: int = 42) -> list[tuple[str, int, str]]:
    return _l0c2_band_support(_enumerate_partition_l0c2k1, seed)


def _l0c2k2_support(seed: int = 42) -> list[tuple[str, int, str]]:
    return _l0c2_band_support(_enumerate_partition_l0c2k2, seed)


def _l0c2k3_support(seed: int = 42) -> list[tuple[str, int, str]]:
    return _l0c2_band_support(_enumerate_partition_l0c2k3, seed)


def build_l0c2k1_support(seed: int = 42) -> dict[str, list[tuple[str, int, str]]]:
    """Standalone L0c2-K1 audit surface (seed-42 count 24)."""
    return {"L0c2-K1": _l0c2k1_support(seed)}


def build_l0c2k2_support(seed: int = 42) -> dict[str, list[tuple[str, int, str]]]:
    """Standalone L0c2-K2 audit surface (seed-42 count 79)."""
    return {"L0c2-K2": _l0c2k2_support(seed)}


def build_l0c2k3_support(seed: int = 42) -> dict[str, list[tuple[str, int, str]]]:
    """Standalone L0c2-K3 audit surface (seed-42 count 127)."""
    return {"L0c2-K3": _l0c2k3_support(seed)}


def l0c2_band_audit_expected_count(seed: int, band: str) -> int:
    """Seed-aware expected row count for an original L0c2 K-band audit surface
    (K1/K2/K3). Resolves the seed-17-vs-seed-42 mismatch: a seed-17 full-K1
    audit expects 29, not the seed-42 reference 24. The probe evaluates this at
    the RESOLVED audit_seed so the reported expected_aggregate matches the
    actual built rows on any seed."""
    return l0c2_band_expected_count(seed, band)


# --------------------------------------------------------------------------- #
# F.4d-edge — L0c2-K1-edge held-generalization micro-slice audit surface.
# SEPARATE finite surface (codex_2 design 1779728324177 + co-lead finite-train
# amendment). Exposed as TWO sub-surfaces so the gate reports per surface:
#   train 52/52 (strict-exact — training samples with replacement, so the only
#     way to prove all 52 unique train rows cleared is a finite enumeration),
#   held  13/13, with the held bucket axis splitting legacy(4) vs fresh(9).
# Counts are FIXED across seeds (only WHICH rows are fresh-held varies), so no
# seed-aware count helper is needed here (unlike K1/K2/K3). Default seed 17
# (active chain); the probe overrides with the resolved audit_seed.
# --------------------------------------------------------------------------- #
L0C2K1_EDGE_TRAIN_AUDIT_COUNT: int = 52
L0C2K1_EDGE_HELD_AUDIT_COUNT: int = 13
L0C2K1_EDGE_AUDIT_EXPECTED_COUNT: int = 65


def _l0c2k1_edge_train_support(seed: int = 17) -> list[tuple[str, int, str]]:
    """52 finite train rows as (question, expected, stratum)."""
    train, _held = _enumerate_partition_l0c2k1_edge(seed)
    return [(r["question"], r["expected"], r["stratum"]) for r in train]


def _l0c2k1_edge_held_support(seed: int = 17) -> list[tuple[str, int, str]]:
    """13 finite held rows as (question, expected, hold_kind) where hold_kind is
    'legacy' (the 4 pinned edges) or 'fresh' (9 generalization rows)."""
    _train, held = _enumerate_partition_l0c2k1_edge(seed)
    return [(r["question"], r["expected"], r["hold_kind"]) for r in held]


def build_l0c2k1_edge_support(seed: int = 17) -> dict[str, list[tuple[str, int, str]]]:
    """Standalone L0c2-K1-edge audit surface as TWO finite sub-surfaces so the
    probe reports pass/fail per surface: train 52/52 + held 13/13 (held bucket
    axis = legacy/fresh). Parallel to build_l0c2k1_support but two-keyed."""
    return {
        "L0c2-K1-edge-train": _l0c2k1_edge_train_support(seed),
        "L0c2-K1-edge-held": _l0c2k1_edge_held_support(seed),
    }


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
    if rung in ("L0a", "L0b", "L0c", "L0c1"):
        # L0b/L0c mirror L0a's bucket order exactly; L0c1 (Slice F.1) spans
        # the same 13 source buckets (it is L0c's one_digit subset). (codex msg
        # 1779567887201-1cf4f485 Slice D.1 + 1779571151811-d3f6bc4f
        # Slice E.1: same 13 source buckets, same per-bucket counts,
        # only the question-template surface differs).
        return [
            "R0",
            "R1_plus_0", "R1_0_plus_A", "R1_minus_0",
            "R1b1", "R1b2", "R1b3", "R1b4v2",
            "R1b5", "R1b6", "R1b7", "R1b8", "R1b9",
        ]
    if rung in ("L0c2", "L0c2-K1", "L0c2-K2", "L0c2-K3"):
        # F.4/F.4d bounded hard surfaces: report by COMPOSITE
        # source_rung:operator buckets, NOT collapsed source-rungs. K bands use
        # the full L0c2 bucket axis so non-default seeds cannot KeyError when a
        # small band happens to include a different sparse bucket.
        return [
            "R0:identity",
            "R1:minus", "R1:plus",
            "R1b1:plus", "R1b2:minus", "R1b3:plus", "R1b4v2:plus",
            "R1b5:plus", "R1b6:plus", "R1b7:plus", "R1b8:plus", "R1b9:plus",
        ]
    if rung == "L0c2-K1-edge-train":
        # F.4d-edge finite train surface: bucketed by template stratum.
        return ["identity", "plus_m0", "plus_m1", "plus_m2_m4", "plus_m5_m9"]
    if rung == "L0c2-K1-edge-held":
        # F.4d-edge finite held surface: 4 legacy pinned edges vs 9 fresh rows.
        return ["legacy", "fresh"]
    raise ValueError(
        f"unknown language rung {rung!r}; valid: {LANGUAGE_ACTIVE_RUNGS} "
        f"+ 'L0c1' / 'L0c2' / 'L0c2-K1..K3' (separate audit surfaces)"
    )
