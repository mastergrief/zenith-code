"""Retention-anchor V0 sentinel sets for HRM-Text-1.58 curriculum
training (codex msg 1779563870477-1b2cff63 +1 Slice A only).

PARALLEL audit surface to `exhaustive_supports.py` (math A0, 1255)
and `language_supports.py` (L0a, 230). Kept in a separate module so:

- Math A0 export stays pure (R0..R1b9 = 1255).
- Language audit stays pure (L0a = 230).
- Anchor audit grows independently as `_v2`, `_v3`, ... ship.
- Probe JSON reports `math`, `language`, and `anchor` aggregates
  separately (no blended single-number aggregate).

The V0 set `MATH_FRAGILE_V1` targets known parent-relative fragile
boundary rows observed during L0a slice runs:

- R1b2 `what is 10 minus 1?` -> 9 (parent decodes leading-zero "09";
  the L0a rr=0.65 lr=5e-4 final regressed this to value-wrong "0").
- R1 zero-left boundary (`what is 0 plus N?` for N=0..9): the rr=0.80
  L0a final regressed `0 plus 4?` to value-wrong "44".
- R1 zero-right boundary (`what is N plus 0?` for N=0..9): symmetric
  counterpart; included for full zero-boundary coverage.

Total: 1 + 10 + 10 = 21 anchor entries; 20 unique question strings
(`what is 0 plus 0?` appears under both R1_zero_left and R1_zero_right
buckets to keep symmetric audit accounting; downstream tooling MUST
key on `anchor_id` not `question`).

Default training behavior is unchanged: anchors compose only when
`--retention-anchor-set` is explicitly set to a non-`none` value.
Slice A ships this module + tests ONLY; trainer/probe wiring is
deferred to Slice B / C with separate design gates per codex msg
1779563870477-1b2cff63 (LMHead loss-reduction constraint discovered
on live source check).

Pure data assembly; zero model deps.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AnchorRow:
    """One retention-anchor entry.

    Fields:
        question: exact prompt text (matches existing rung surface
            format; same string the trainer tokenizes for the prompt
            half of a (prompt, response) pair).
        expected: integer expected answer; the response half tokenizes
            from `str(expected)`.
        source_rung: bucket label for audit-JSON per-source breakdown
            and operator-reading provenance.
        anchor_id: unique stable identifier; downstream tooling MUST
            use this rather than `question` because some questions
            (e.g. `what is 0 plus 0?`) appear under more than one
            source_rung bucket.
    """
    question: str
    expected: int
    source_rung: str
    anchor_id: str


MATH_FRAGILE_V1: tuple[AnchorRow, ...] = (
    # R1b2 known-fragile parent-shape carry-forward
    AnchorRow(question="what is 10 minus 1?", expected=9,
              source_rung="R1b2", anchor_id="r1b2:10_minus_1"),

    # R1 zero-left boundary: `0 plus N?` for N=0..9
    AnchorRow(question="what is 0 plus 0?", expected=0,
              source_rung="R1_zero_left", anchor_id="r1_zl:0_plus_0"),
    AnchorRow(question="what is 0 plus 1?", expected=1,
              source_rung="R1_zero_left", anchor_id="r1_zl:0_plus_1"),
    AnchorRow(question="what is 0 plus 2?", expected=2,
              source_rung="R1_zero_left", anchor_id="r1_zl:0_plus_2"),
    AnchorRow(question="what is 0 plus 3?", expected=3,
              source_rung="R1_zero_left", anchor_id="r1_zl:0_plus_3"),
    AnchorRow(question="what is 0 plus 4?", expected=4,
              source_rung="R1_zero_left", anchor_id="r1_zl:0_plus_4"),
    AnchorRow(question="what is 0 plus 5?", expected=5,
              source_rung="R1_zero_left", anchor_id="r1_zl:0_plus_5"),
    AnchorRow(question="what is 0 plus 6?", expected=6,
              source_rung="R1_zero_left", anchor_id="r1_zl:0_plus_6"),
    AnchorRow(question="what is 0 plus 7?", expected=7,
              source_rung="R1_zero_left", anchor_id="r1_zl:0_plus_7"),
    AnchorRow(question="what is 0 plus 8?", expected=8,
              source_rung="R1_zero_left", anchor_id="r1_zl:0_plus_8"),
    AnchorRow(question="what is 0 plus 9?", expected=9,
              source_rung="R1_zero_left", anchor_id="r1_zl:0_plus_9"),

    # R1 zero-right boundary: `N plus 0?` for N=0..9
    # `0 plus 0?` reappears here under the right-bucket label to keep
    # symmetric audit accounting (each bucket reports 10 entries).
    AnchorRow(question="what is 0 plus 0?", expected=0,
              source_rung="R1_zero_right", anchor_id="r1_zr:0_plus_0"),
    AnchorRow(question="what is 1 plus 0?", expected=1,
              source_rung="R1_zero_right", anchor_id="r1_zr:1_plus_0"),
    AnchorRow(question="what is 2 plus 0?", expected=2,
              source_rung="R1_zero_right", anchor_id="r1_zr:2_plus_0"),
    AnchorRow(question="what is 3 plus 0?", expected=3,
              source_rung="R1_zero_right", anchor_id="r1_zr:3_plus_0"),
    AnchorRow(question="what is 4 plus 0?", expected=4,
              source_rung="R1_zero_right", anchor_id="r1_zr:4_plus_0"),
    AnchorRow(question="what is 5 plus 0?", expected=5,
              source_rung="R1_zero_right", anchor_id="r1_zr:5_plus_0"),
    AnchorRow(question="what is 6 plus 0?", expected=6,
              source_rung="R1_zero_right", anchor_id="r1_zr:6_plus_0"),
    AnchorRow(question="what is 7 plus 0?", expected=7,
              source_rung="R1_zero_right", anchor_id="r1_zr:7_plus_0"),
    AnchorRow(question="what is 8 plus 0?", expected=8,
              source_rung="R1_zero_right", anchor_id="r1_zr:8_plus_0"),
    AnchorRow(question="what is 9 plus 0?", expected=9,
              source_rung="R1_zero_right", anchor_id="r1_zr:9_plus_0"),
)


RETENTION_ANCHOR_SETS: dict[str, tuple[AnchorRow, ...]] = {
    "math_fragile_v1": MATH_FRAGILE_V1,
}


RETENTION_ANCHOR_EXPECTED_COUNTS: dict[str, int] = {
    "math_fragile_v1": 21,
}


def load_anchor_set(name: str) -> tuple[AnchorRow, ...]:
    """Return the anchor-row tuple for a named set.

    `name="none"` returns an empty tuple (default-off contract).
    `name` not in the registry raises ValueError.

    Pure: no model deps, no side effects, no I/O.
    """
    if name == "none":
        return ()
    if name not in RETENTION_ANCHOR_SETS:
        raise ValueError(
            f"unknown retention-anchor set {name!r}; "
            f"valid: 'none' or one of {tuple(RETENTION_ANCHOR_SETS)}"
        )
    return RETENTION_ANCHOR_SETS[name]


def anchor_set_source_rung_buckets(name: str) -> list[str]:
    """Return per-source-rung bucket labels in canonical reporting
    order for a named anchor set. Used by probe to render per-bucket
    breakdowns parallel to `language_source_rung_buckets`.
    """
    if name == "math_fragile_v1":
        return ["R1b2", "R1_zero_left", "R1_zero_right"]
    if name == "none":
        return []
    raise ValueError(
        f"unknown retention-anchor set {name!r}; "
        f"valid: 'none' or one of {tuple(RETENTION_ANCHOR_SETS)}"
    )
