"""Slice A tests for `calm.hrm_text_158.curriculum.retention_anchors`.

Pins the 21-entry `MATH_FRAGILE_V1` golden table + load/dispatch API
shape. Trainer/probe integration tests are deferred to Slice B / C
per codex msg 1779563870477-1b2cff63 (LMHead loss-reduction
constraint requires separate Slice B replan: row-repeat oversample
OR explicit LMHead weighted-loss API).

Anchor-set design notes pinned by these tests:
- 21 entries by count, 20 unique question strings (`what is 0 plus 0?`
  appears under both R1_zero_left and R1_zero_right buckets).
- Downstream tooling MUST key on `anchor_id` not `question`.
- Default-off contract: `load_anchor_set("none")` returns ().
"""
from __future__ import annotations

import dataclasses

import pytest

from calm.hrm_text_158.curriculum.retention_anchors import (
    AnchorRow,
    MATH_FRAGILE_V1,
    RETENTION_ANCHOR_SETS,
    RETENTION_ANCHOR_EXPECTED_COUNTS,
    anchor_set_source_rung_buckets,
    load_anchor_set,
)


def _expected_golden_table() -> tuple[AnchorRow, ...]:
    """Independent reconstruction of the golden 21-row table.

    Mirrors the literal codex V0 spec (1 R1b2 + 10 zero-left +
    10 zero-right). Any drift between this and the module's
    `MATH_FRAGILE_V1` is caught by `test_math_fragile_v1_golden_table`.
    """
    rows: list[AnchorRow] = []
    rows.append(AnchorRow(
        question="what is 10 minus 1?", expected=9,
        source_rung="R1b2", anchor_id="r1b2:10_minus_1",
    ))
    for n in range(10):
        rows.append(AnchorRow(
            question=f"what is 0 plus {n}?", expected=n,
            source_rung="R1_zero_left", anchor_id=f"r1_zl:0_plus_{n}",
        ))
    for n in range(10):
        rows.append(AnchorRow(
            question=f"what is {n} plus 0?", expected=n,
            source_rung="R1_zero_right", anchor_id=f"r1_zr:{n}_plus_0",
        ))
    return tuple(rows)


def test_math_fragile_v1_count_equals_21():
    assert len(MATH_FRAGILE_V1) == 21, (
        f"V0 spec requires exactly 21 entries; got {len(MATH_FRAGILE_V1)}"
    )


def test_math_fragile_v1_unique_question_count_equals_20():
    """`what is 0 plus 0?` appears under both R1_zero_left and
    R1_zero_right buckets => 21 entries, 20 unique question strings."""
    unique_q = {row.question for row in MATH_FRAGILE_V1}
    assert len(unique_q) == 20, (
        f"V0 spec has natural dup of `what is 0 plus 0?`; expected 20 "
        f"unique question strings, got {len(unique_q)}"
    )


def test_math_fragile_v1_anchor_ids_unique():
    """anchor_ids must disambiguate the natural-dup case so downstream
    tooling can key on them."""
    ids = [row.anchor_id for row in MATH_FRAGILE_V1]
    assert len(set(ids)) == 21, (
        f"All anchor_ids must be unique; got {len(set(ids))} unique of "
        f"{len(ids)} total. Duplicates: "
        f"{[i for i in set(ids) if ids.count(i) > 1]}"
    )


def test_math_fragile_v1_golden_table():
    """Exact 21-row Q/A/source/anchor_id pin against independent
    reconstruction. Any drift fails here."""
    expected = _expected_golden_table()
    assert len(MATH_FRAGILE_V1) == len(expected)
    for i, (got, want) in enumerate(zip(MATH_FRAGILE_V1, expected)):
        assert got == want, (
            f"Row {i} drift:\n  got:  {got}\n  want: {want}"
        )


def test_math_fragile_v1_contains_10_minus_1():
    """Explicit pin for the known R1b2 fragile row (L0a rr=0.65
    lr=5e-4 final regressed this)."""
    matches = [r for r in MATH_FRAGILE_V1
               if r.question == "what is 10 minus 1?"]
    assert len(matches) == 1, (
        f"Exactly one entry must match the R1b2 fragile row; "
        f"got {len(matches)}"
    )
    row = matches[0]
    assert row.expected == 9
    assert row.source_rung == "R1b2"
    assert row.anchor_id == "r1b2:10_minus_1"


def test_math_fragile_v1_contains_zero_plus_4():
    """Explicit pin for the L0a rr=0.80 final regression row
    (`0 plus 4?` → '44' value-wrong) under R1_zero_left."""
    matches = [r for r in MATH_FRAGILE_V1
               if r.question == "what is 0 plus 4?"]
    assert len(matches) == 1, (
        f"Exactly one entry must match the rr=0.80 fragile row; "
        f"got {len(matches)}"
    )
    row = matches[0]
    assert row.expected == 4
    assert row.source_rung == "R1_zero_left"
    assert row.anchor_id == "r1_zl:0_plus_4"


def test_math_fragile_v1_buckets_sum_to_21():
    """Per-bucket counts: R1b2=1, R1_zero_left=10, R1_zero_right=10
    summing to the literal codex V0 spec."""
    buckets: dict[str, int] = {}
    for row in MATH_FRAGILE_V1:
        buckets[row.source_rung] = buckets.get(row.source_rung, 0) + 1
    assert buckets == {
        "R1b2": 1,
        "R1_zero_left": 10,
        "R1_zero_right": 10,
    }, f"Bucket count drift: {buckets}"
    assert sum(buckets.values()) == 21


def test_math_fragile_v1_expected_value_matches_question_semantics():
    """Sanity: every expected matches the obvious arithmetic of the
    question (so a hand-written typo in expected= would fail here)."""
    import re
    pat_plus = re.compile(r"^what is (\d+) plus (\d+)\?$")
    pat_minus = re.compile(r"^what is (\d+) minus (\d+)\?$")
    for row in MATH_FRAGILE_V1:
        m = pat_plus.match(row.question)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            assert row.expected == a + b, (
                f"Plus-row arithmetic mismatch: {row.question} "
                f"expected={row.expected} vs a+b={a + b}"
            )
            continue
        m = pat_minus.match(row.question)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            assert row.expected == a - b, (
                f"Minus-row arithmetic mismatch: {row.question} "
                f"expected={row.expected} vs a-b={a - b}"
            )
            continue
        pytest.fail(f"Unrecognized question pattern: {row.question!r}")


def test_anchor_row_is_frozen_dataclass():
    """AnchorRow must be immutable so the golden table can't be
    mutated at runtime."""
    row = MATH_FRAGILE_V1[0]
    with pytest.raises(dataclasses.FrozenInstanceError):
        row.expected = 999  # type: ignore[misc]


def test_load_anchor_set_none_returns_empty_tuple():
    """Default-off contract: 'none' yields no anchors."""
    result = load_anchor_set("none")
    assert result == ()
    assert isinstance(result, tuple)


def test_load_anchor_set_math_fragile_v1_returns_21_rows():
    """Named-set dispatch returns the 21-entry golden table."""
    result = load_anchor_set("math_fragile_v1")
    assert result is MATH_FRAGILE_V1
    assert len(result) == 21


def test_load_anchor_set_unknown_raises_value_error():
    """Bad name fails fast at load-time, not silently later."""
    with pytest.raises(ValueError, match=r"unknown retention-anchor set"):
        load_anchor_set("bogus_set_name")


def test_retention_anchor_expected_counts_matches_set_lengths():
    """The declared expected-count table must agree with the actual
    tuple lengths in `RETENTION_ANCHOR_SETS`."""
    for name, expected_count in RETENTION_ANCHOR_EXPECTED_COUNTS.items():
        assert name in RETENTION_ANCHOR_SETS, (
            f"declared count for {name!r} but set not registered"
        )
        assert len(RETENTION_ANCHOR_SETS[name]) == expected_count, (
            f"Set {name!r} has "
            f"{len(RETENTION_ANCHOR_SETS[name])} rows but declared "
            f"count is {expected_count}"
        )


def test_anchor_set_source_rung_buckets_math_fragile_v1():
    """Per-bucket canonical reporting order is exposed for probe
    audit JSON rendering."""
    buckets = anchor_set_source_rung_buckets("math_fragile_v1")
    assert buckets == ["R1b2", "R1_zero_left", "R1_zero_right"]


def test_anchor_set_source_rung_buckets_none_returns_empty():
    """`none` has no buckets — symmetric with `load_anchor_set('none')`."""
    assert anchor_set_source_rung_buckets("none") == []


def test_anchor_set_source_rung_buckets_unknown_raises():
    """Bad name fails fast."""
    with pytest.raises(ValueError, match=r"unknown retention-anchor set"):
        anchor_set_source_rung_buckets("bogus")
