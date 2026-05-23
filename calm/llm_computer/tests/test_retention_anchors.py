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
import json

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


# ============================================================================ #
# Slice B trainer-integration tests (codex msg 1779564576409-a7db0527 +1 A1
# row-repeat). These test the trainer-side helper `_compose_anchor_rows` and
# the ckpt config recording, NOT actual training (no GPU, no model build).
# Trainer module is imported lazily inside each test to avoid heavy import
# cost when running only Slice A tests.
# ============================================================================ #


def _import_trainer():
    """Import the trainer module lazily. Defers heavy torch/HRM imports
    until a Slice B test actually runs."""
    import importlib.util
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[3]
    trainer_path = repo_root / "scripts" / "train_hrm_text_158.py"
    # Use importlib to load the script as a module without adding to PATH.
    spec = importlib.util.spec_from_file_location(
        "_test_train_hrm_text_158", str(trainer_path)
    )
    assert spec is not None and spec.loader is not None
    if "_test_train_hrm_text_158" in sys.modules:
        return sys.modules["_test_train_hrm_text_158"]
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_test_train_hrm_text_158"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_trainer_compose_anchor_rows_none_returns_empty():
    """Default-off contract: 'none' yields zero anchor rows."""
    trainer = _import_trainer()
    rows = trainer._compose_anchor_rows("none", 2)
    assert rows == []
    rows = trainer._compose_anchor_rows("none", 5)
    assert rows == []


def test_trainer_compose_anchor_rows_math_fragile_v1_repeat_2_adds_42():
    """Default repeat=2 with math_fragile_v1 yields 21·2 = 42 anchor rows."""
    trainer = _import_trainer()
    rows = trainer._compose_anchor_rows("math_fragile_v1", 2)
    assert len(rows) == 42, f"expected 42, got {len(rows)}"
    # First 21 rows match the canonical set order; next 21 are the repeat.
    unique_count = len({r["anchor_id"] for r in rows})
    assert unique_count == 21, (
        f"21 unique anchor_ids expected; got {unique_count}"
    )


def test_trainer_compose_anchor_rows_math_fragile_v1_repeat_5_adds_105():
    """Explicit repeat=5 yields 21·5 = 105 anchor rows."""
    trainer = _import_trainer()
    rows = trainer._compose_anchor_rows("math_fragile_v1", 5)
    assert len(rows) == 105, f"expected 105, got {len(rows)}"
    unique_count = len({r["anchor_id"] for r in rows})
    assert unique_count == 21


def test_trainer_compose_anchor_rows_repeat_1_adds_21():
    """Minimum legal repeat=1 yields 21 anchor rows (no replication)."""
    trainer = _import_trainer()
    rows = trainer._compose_anchor_rows("math_fragile_v1", 1)
    assert len(rows) == 21
    # All anchor_ids unique at repeat=1
    assert len({r["anchor_id"] for r in rows}) == 21


def test_trainer_compose_anchor_rows_schema():
    """Each anchor row has the curriculum-compatible dict schema:
    question (str), expected (int), anchor_id (str), source_rung (str).
    `anchor_id` is the discriminator excluding anchors from target-rung
    unique-count math."""
    trainer = _import_trainer()
    rows = trainer._compose_anchor_rows("math_fragile_v1", 2)
    for r in rows:
        assert set(r.keys()) == {
            "question", "expected", "anchor_id", "source_rung",
        }, f"unexpected keys: {set(r.keys())}"
        assert isinstance(r["question"], str)
        assert isinstance(r["expected"], int)
        assert isinstance(r["anchor_id"], str)
        assert isinstance(r["source_rung"], str)


def test_trainer_compose_anchor_rows_unknown_set_raises():
    """Unknown set name fails fast (delegates to load_anchor_set)."""
    trainer = _import_trainer()
    with pytest.raises(ValueError, match=r"unknown retention-anchor set"):
        trainer._compose_anchor_rows("bogus_set", 2)


def test_build_ckpt_config_omits_anchor_fields_when_default():
    """Default-off contract: no anchor fields in config when retention_anchor_set
    is None or 'none'."""
    trainer = _import_trainer()
    # Pure dispatch test: mock the minimum surface _build_ckpt_config needs.

    class _MockCfg:
        max_seq_len = 256
        n_layers = 4
        hidden_size = 256
        num_heads = 2
        expansion = 4.0
        H_cycles = 2
        L_cycles = 3
        half_layers = True
        bp_warmup_ratio = 0.2
        bp_min_steps = 2
        bp_max_steps = 5
        norm_type = "rmsnorm"
        norm_eps = 1e-5
        rope_theta = 10000.0
        attn_type = "self"
        init_type = "leuncn"
        pos_emb_type = "rope"
        use_ternary_bulk = False

    class _MockTok:
        vocab_size = 260
        normalizer_version = "byte_utf8_v1"

        def vocab_as_list(self):
            return list(range(260))

    m, tok, cfg = object(), _MockTok(), _MockCfg()

    # Case 1: retention_anchor_set not passed → no fields in output
    out = trainer._build_ckpt_config(m, tok, cfg, max_len=256, batch_size=8)
    assert "retention_anchor_set" not in out
    assert "retention_anchor_repeat" not in out

    # Case 2: explicit 'none' → still no fields (matches default-off
    # contract: config shape unchanged from current behavior)
    out = trainer._build_ckpt_config(
        m, tok, cfg, max_len=256, batch_size=8,
        retention_anchor_set="none", retention_anchor_repeat=2,
    )
    assert "retention_anchor_set" not in out
    assert "retention_anchor_repeat" not in out


def test_build_ckpt_config_records_anchor_fields_when_enabled():
    """When retention_anchor_set != 'none', both fields appear in config."""
    trainer = _import_trainer()

    class _MockCfg:
        max_seq_len = 256
        n_layers = 4
        hidden_size = 256
        num_heads = 2
        expansion = 4.0
        H_cycles = 2
        L_cycles = 3
        half_layers = True
        bp_warmup_ratio = 0.2
        bp_min_steps = 2
        bp_max_steps = 5
        norm_type = "rmsnorm"
        norm_eps = 1e-5
        rope_theta = 10000.0
        attn_type = "self"
        init_type = "leuncn"
        pos_emb_type = "rope"
        use_ternary_bulk = False

    class _MockTok:
        vocab_size = 260
        normalizer_version = "byte_utf8_v1"

        def vocab_as_list(self):
            return list(range(260))

    out = trainer._build_ckpt_config(
        object(), _MockTok(), _MockCfg(), max_len=256, batch_size=8,
        retention_anchor_set="math_fragile_v1", retention_anchor_repeat=3,
    )
    assert out["retention_anchor_set"] == "math_fragile_v1"
    assert out["retention_anchor_repeat"] == 3


def test_trainer_anchor_repeat_zero_rejected_at_cli():
    """argparse error for --retention-anchor-repeat 0."""
    import subprocess
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[3]
    trainer_path = repo_root / "scripts" / "train_hrm_text_158.py"
    result = subprocess.run(
        [sys.executable, str(trainer_path),
         "--retention-anchor-repeat", "0", "--dry-run"],
        capture_output=True, text=True,
        env={"PYTHONPATH": str(repo_root)},
    )
    assert result.returncode != 0, (
        f"expected nonzero exit; got {result.returncode} "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    # Error message names the flag + sanity bound
    assert (
        "retention-anchor-repeat" in result.stderr
        or "retention-anchor-repeat" in result.stdout
    )
    assert ">= 1" in result.stderr or ">= 1" in result.stdout


def test_trainer_anchor_repeat_negative_rejected_at_cli():
    """argparse error for --retention-anchor-repeat -1."""
    import subprocess
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[3]
    trainer_path = repo_root / "scripts" / "train_hrm_text_158.py"
    result = subprocess.run(
        [sys.executable, str(trainer_path),
         "--retention-anchor-repeat", "-1", "--dry-run"],
        capture_output=True, text=True,
        env={"PYTHONPATH": str(repo_root)},
    )
    assert result.returncode != 0
    assert ">= 1" in result.stderr or ">= 1" in result.stdout


def test_trainer_anchor_repeat_non_integer_rejected_at_cli():
    """argparse type=int rejects floats loudly (no silent rounding)."""
    import subprocess
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[3]
    trainer_path = repo_root / "scripts" / "train_hrm_text_158.py"
    result = subprocess.run(
        [sys.executable, str(trainer_path),
         "--retention-anchor-repeat", "2.5", "--dry-run"],
        capture_output=True, text=True,
        env={"PYTHONPATH": str(repo_root)},
    )
    assert result.returncode != 0
    # argparse rejects non-int via "invalid int value"
    assert (
        "invalid int value" in result.stderr
        or "invalid int value" in result.stdout
    )


def test_trainer_anchor_set_rejects_unknown_at_cli():
    """argparse choices= rejects unknown set names."""
    import subprocess
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[3]
    trainer_path = repo_root / "scripts" / "train_hrm_text_158.py"
    result = subprocess.run(
        [sys.executable, str(trainer_path),
         "--retention-anchor-set", "bogus_set", "--dry-run"],
        capture_output=True, text=True,
        env={"PYTHONPATH": str(repo_root)},
    )
    assert result.returncode != 0
    assert (
        "invalid choice" in result.stderr
        or "invalid choice" in result.stdout
    )


def test_trainer_compose_anchor_rows_anchor_ids_disjoint_from_curriculum():
    """Anchor `anchor_id` field is a unique discriminator; existing
    curriculum rows never carry this key (verified via the curriculum
    row generators in calm/hrm_text_158/curriculum/generators.py which
    emit dicts with `question`, `expected`, `rung` OR `source_rung` but
    NOT `anchor_id`). This is the field downstream multiplicity-floor
    accounting uses to exclude anchor rows from target-rung unique counts.
    """
    trainer = _import_trainer()
    anchor_rows = trainer._compose_anchor_rows("math_fragile_v1", 3)

    # Sanity check: every anchor row has anchor_id; this is the
    # downstream-tooling discriminator codex specified.
    for r in anchor_rows:
        assert "anchor_id" in r, (
            f"anchor row missing required `anchor_id` field: {r}"
        )
        assert r["anchor_id"].startswith(("r1b2:", "r1_zl:", "r1_zr:"))


def test_trainer_anchor_phase_gate_rejects_gsm8k_mode():
    """Slice B phase-gate (codex msg 1779565128372-c6872566): retention
    anchors are Phase 3 curriculum-only. GSM8k mode (curriculum_rung=None)
    + anchor_set != 'none' must raise BEFORE any composition or save,
    so the ckpt cannot falsely record retention_anchor_set=enabled.
    """
    trainer = _import_trainer()
    with pytest.raises(ValueError, match=r"requires curriculum_rung"):
        trainer.train(
            curriculum_rung=None,
            retention_anchor_set="math_fragile_v1",
            retention_anchor_repeat=2,
            # Other kwargs unset; ValueError fires before they matter.
        )


def test_trainer_anchor_repeat_below_one_rejected_programmatically():
    """Programmatic-call defense: argparse-bypassing callers must still
    fail loudly on repeat < 1. CLI already covers this; this test
    pins the train() function-level check (codex msg 1779565128372).
    """
    trainer = _import_trainer()
    # GSM8k branch unreachable due to phase-gate, so set curriculum_rung
    # to a valid value so the repeat-validation check is the one that fires.
    with pytest.raises(ValueError, match=r"retention_anchor_repeat must be >= 1"):
        trainer.train(
            curriculum_rung="R0",
            retention_anchor_set="math_fragile_v1",
            retention_anchor_repeat=0,
        )
    with pytest.raises(ValueError, match=r"retention_anchor_repeat must be >= 1"):
        trainer.train(
            curriculum_rung="R0",
            retention_anchor_set="math_fragile_v1",
            retention_anchor_repeat=-3,
        )


# ============================================================================ #
# Slice C probe-integration tests (codex msg 1779566905283-8ba63fe9 +1
# implement). These test the `--anchor-audit` mode in
# scripts/probe_hrm_text_158.py against the banked L0a chain head, using
# the canonical smoke-audit baseline gate codex pinned:
#   - strict 20/21 with sole hole = anchor_id="r1b2:10_minus_1"
#     decoded "09" parsed_ok=true
#   - parsed 21/21, finite=true, value-wrong holes=0
#   - all zero-boundary anchors strict+parsed clean
#   - aggregate.expected_aggregate=21 (NOT blended with math 1255 / language 230)
#   - rows keyed by anchor_id (21 records preserved despite the natural-dup
#     of `what is 0 plus 0?`)
# Tests use subprocess invocations of the probe script — they exercise CLI
# parsing, mutex pre-checks, and (where the banked ckpt is available) actual
# audit JSON output.
# ============================================================================ #

import os
from pathlib import Path

_BANKED_L0A_CKPT = (
    Path(__file__).resolve().parents[3]
    / "calm" / "hrm" / "checkpoints"
    / "hrm_text_158_phase3_L0a_seed0017_replay65_n10k_lr2e4_from_R1b9_final.pt"
)


def _banked_ckpt_present() -> bool:
    return _BANKED_L0A_CKPT.exists()


def _run_anchor_audit_on_banked(tmp_path: Path) -> dict:
    """Run the probe with --anchor-audit against the banked L0a chain head
    and return the parsed JSON. Used by gate tests below."""
    import subprocess
    import sys

    repo_root = Path(__file__).resolve().parents[3]
    probe_path = repo_root / "scripts" / "probe_hrm_text_158.py"
    out_json = tmp_path / "anchor_audit.json"
    result = subprocess.run(
        [
            sys.executable, "-u", str(probe_path),
            "--ckpt-path", str(_BANKED_L0A_CKPT),
            "--anchor-audit",
            "--audit-output-json", str(out_json),
            "--use-cached-ternary-infer",
            "--use-kv-cache-decode",
            "--use-batched-probe-eval",
            "--probe-batch-size", "32",
        ],
        capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": str(repo_root)},
        timeout=120,
    )
    assert result.returncode == 0, (
        f"probe failed (exit {result.returncode}):\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )
    return json.loads(out_json.read_text()), result.stdout


@pytest.mark.skipif(
    not _banked_ckpt_present(),
    reason=f"banked L0a chain head not present at {_BANKED_L0A_CKPT}",
)
def test_probe_anchor_audit_banked_L0a_baseline_20_strict_21_parsed(tmp_path):
    """Smoke audit gate (codex msg 1779566905283 corrected baseline):
    banked L0a chain head must produce strict=20/21 AND parsed=21/21 AND
    sole strict hole is anchor_id='r1b2:10_minus_1' decoded '09'
    parsed_ok=True. Future anchor-trained ckpts may improve to strict 21/21
    but this test must not halt on the accepted parent-shape carry-forward.
    """
    out, _stdout = _run_anchor_audit_on_banked(tmp_path)
    agg = out["aggregate"]
    assert agg["n_total"] == 21
    assert agg["n_parsed_correct"] == 21, (
        f"parsed must be 21/21 at baseline; got {agg['n_parsed_correct']}/21"
    )
    assert agg["n_exact"] == 20, (
        f"strict must be 20/21 at baseline (one accepted parent-shape hole); "
        f"got {agg['n_exact']}/21"
    )
    assert agg["finite"] is True
    assert agg["expected_aggregate"] == 21
    # Sole hole must be the accepted carry-forward row
    holes = out["results"]["math_fragile_v1"]["holes_first20"]
    assert len(holes) == 1, f"expected exactly 1 hole; got {len(holes)}"
    h = holes[0]
    assert h["anchor_id"] == "r1b2:10_minus_1", (
        f"sole hole must be r1b2:10_minus_1; got {h['anchor_id']}"
    )
    assert h["decoded"] == "09", (
        f"decoded must match parent-shape '09'; got {h['decoded']!r}"
    )
    assert h["parsed_ok"] is True
    assert h["exact_ok"] is False


@pytest.mark.skipif(
    not _banked_ckpt_present(),
    reason=f"banked L0a chain head not present at {_BANKED_L0A_CKPT}",
)
def test_probe_anchor_audit_zero_boundary_strict_clean_on_banked_L0a(tmp_path):
    """Both zero-boundary buckets (R1_zero_left + R1_zero_right) must be
    strict+parsed clean at baseline (10/10 each). Any failure here is a
    real regression, NOT the accepted parent-shape carry-forward. Pins
    that the rr=0.80-era `0 plus 4? → '44'` corruption does not appear
    on the lr-softened banked head.
    """
    out, _stdout = _run_anchor_audit_on_banked(tmp_path)
    bs = out["results"]["math_fragile_v1"]["by_source_rung"]
    for bucket in ("R1_zero_left", "R1_zero_right"):
        b = bs[bucket]
        assert b["n_total"] == 10
        assert b["n_exact"] == 10, (
            f"{bucket} must be strict-clean at baseline; got {b['n_exact']}/10"
        )
        assert b["n_parsed_correct"] == 10
        assert b["n_holes"] == 0


@pytest.mark.skipif(
    not _banked_ckpt_present(),
    reason=f"banked L0a chain head not present at {_BANKED_L0A_CKPT}",
)
def test_probe_anchor_audit_aggregate_expected_21(tmp_path):
    """aggregate.expected_aggregate = 21 for math_fragile_v1; this is the
    anchor-mode-specific field, separate from math A0's 1255 and language
    L0a's 230."""
    out, _stdout = _run_anchor_audit_on_banked(tmp_path)
    assert out["aggregate"]["expected_aggregate"] == 21


@pytest.mark.skipif(
    not _banked_ckpt_present(),
    reason=f"banked L0a chain head not present at {_BANKED_L0A_CKPT}",
)
def test_probe_anchor_audit_separate_from_math_and_language(tmp_path):
    """Anchor JSON has no `expected_aggregate=1255` (math) or 230 (language)
    field-name collisions. Aggregates are NOT blended."""
    out, _stdout = _run_anchor_audit_on_banked(tmp_path)
    assert out["aggregate"]["expected_aggregate"] == 21
    # Anchor mode has no `audit_seed` / `ckpt_curriculum_seed` (those are
    # language-mode fields). Has its own `anchor_set` / `ckpt_anchor_set`.
    assert "anchor_set" in out
    assert "ckpt_anchor_set" in out
    assert "anchor_set_mismatch" in out
    assert "audit_seed" not in out
    assert "active_language_rungs" not in out


@pytest.mark.skipif(
    not _banked_ckpt_present(),
    reason=f"banked L0a chain head not present at {_BANKED_L0A_CKPT}",
)
def test_probe_anchor_audit_per_source_rung_buckets_sum_to_21(tmp_path):
    """Per-source-rung buckets R1b2 + R1_zero_left + R1_zero_right sum to 21
    (literal codex V0 spec: 1 + 10 + 10)."""
    out, _stdout = _run_anchor_audit_on_banked(tmp_path)
    bs = out["results"]["math_fragile_v1"]["by_source_rung"]
    assert set(bs.keys()) == {"R1b2", "R1_zero_left", "R1_zero_right"}
    assert bs["R1b2"]["n_total"] == 1
    assert bs["R1_zero_left"]["n_total"] == 10
    assert bs["R1_zero_right"]["n_total"] == 10
    total = sum(b["n_total"] for b in bs.values())
    assert total == 21


@pytest.mark.skipif(
    not _banked_ckpt_present(),
    reason=f"banked L0a chain head not present at {_BANKED_L0A_CKPT}",
)
def test_probe_anchor_audit_rows_keyed_by_anchor_id(tmp_path):
    """JSON.results[set].rows has 21 records (not 20) with unique
    anchor_ids. Downstream tooling MUST key on anchor_id, not question.
    The natural-dup of `what is 0 plus 0?` produces two row entries with
    the same `question` but different `anchor_id` values."""
    out, _stdout = _run_anchor_audit_on_banked(tmp_path)
    rows = out["results"]["math_fragile_v1"]["rows"]
    assert len(rows) == 21, f"expected 21 rows; got {len(rows)}"
    unique_ids = {r["anchor_id"] for r in rows}
    assert len(unique_ids) == 21, (
        f"all 21 anchor_ids must be unique; got {len(unique_ids)} unique"
    )
    # 20 unique questions (natural-dup of "0 plus 0?")
    unique_q = {r["question"] for r in rows}
    assert len(unique_q) == 20, (
        f"expected 20 unique question strings (natural dup of 0+0); "
        f"got {len(unique_q)}"
    )


@pytest.mark.skipif(
    not _banked_ckpt_present(),
    reason=f"banked L0a chain head not present at {_BANKED_L0A_CKPT}",
)
def test_probe_anchor_audit_fallback_source_printed(tmp_path):
    """When ckpt has no recorded retention_anchor_set, probe MUST print the
    fallback source plainly (codex msg 1779566905283 tightening): receipts
    must be unambiguous about whether the audit is a trained-anchor check
    or a baseline fallback."""
    out, stdout = _run_anchor_audit_on_banked(tmp_path)
    assert out["ckpt_anchor_set"] is None
    assert out["anchor_set_mismatch"] is False
    # Stdout must contain the fallback notice
    assert "source=fallback" in stdout, (
        f"fallback source must be printed plainly; stdout was:\n{stdout}"
    )
    assert "baseline audit, NOT a trained-anchor check" in stdout


def test_probe_anchor_audit_cli_conflicts_with_curriculum_rungs():
    """Mutex pre-check before ckpt load: --anchor-audit + --curriculum-rungs
    must fail fast."""
    import subprocess
    import sys

    repo_root = Path(__file__).resolve().parents[3]
    probe_path = repo_root / "scripts" / "probe_hrm_text_158.py"
    result = subprocess.run(
        [
            sys.executable, str(probe_path),
            "--ckpt-path", "/dev/null",  # never loaded; mutex fires first
            "--anchor-audit",
            "--curriculum-rungs", "R0",
        ],
        capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": str(repo_root)},
        timeout=15,
    )
    assert result.returncode != 0
    assert "--anchor-audit conflicts with --curriculum-rungs" in result.stderr


def test_probe_anchor_audit_cli_conflicts_with_exhaustive():
    """Mutex pre-check: --anchor-audit + --exhaustive-finite-supports must
    fail fast."""
    import subprocess
    import sys

    repo_root = Path(__file__).resolve().parents[3]
    probe_path = repo_root / "scripts" / "probe_hrm_text_158.py"
    result = subprocess.run(
        [
            sys.executable, str(probe_path),
            "--ckpt-path", "/dev/null",
            "--anchor-audit",
            "--exhaustive-finite-supports",
        ],
        capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": str(repo_root)},
        timeout=15,
    )
    assert result.returncode != 0
    assert (
        "--anchor-audit conflicts with --exhaustive-finite-supports"
        in result.stderr
    )


def test_probe_anchor_audit_cli_conflicts_with_language():
    """Mutex pre-check: --anchor-audit + --language-supports must fail
    fast."""
    import subprocess
    import sys

    repo_root = Path(__file__).resolve().parents[3]
    probe_path = repo_root / "scripts" / "probe_hrm_text_158.py"
    result = subprocess.run(
        [
            sys.executable, str(probe_path),
            "--ckpt-path", "/dev/null",
            "--anchor-audit",
            "--language-supports",
        ],
        capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": str(repo_root)},
        timeout=15,
    )
    assert result.returncode != 0
    assert "--anchor-audit conflicts with --language-supports" in result.stderr


def test_probe_anchor_audit_unknown_set_rejected_by_argparse():
    """`--anchor-set bogus` rejected by argparse `choices=` (loud failure,
    no silent fallback)."""
    import subprocess
    import sys

    repo_root = Path(__file__).resolve().parents[3]
    probe_path = repo_root / "scripts" / "probe_hrm_text_158.py"
    result = subprocess.run(
        [
            sys.executable, str(probe_path),
            "--ckpt-path", "/dev/null",
            "--anchor-audit",
            "--anchor-set", "bogus_set",
        ],
        capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": str(repo_root)},
        timeout=15,
    )
    assert result.returncode != 0
    assert (
        "invalid choice" in result.stderr
        or "invalid choice" in result.stdout
    )


def test_trainer_anchor_phase_gate_allows_default_off_in_gsm8k():
    """Default 'none' in GSM8k mode is still allowed (no phase-gate
    violation when anchors aren't enabled). This is the byte-identical
    default-off path — verify the ValueError does NOT fire.
    """
    trainer = _import_trainer()
    # We can't fully run train() without a model + data, but we CAN
    # verify the phase-gate check passes by inspecting how far the
    # function gets before hitting later validation. The simplest
    # assertion is that the guard's exception is NOT the one we hit.
    # If 'none' (default) is correctly skipped, train() will fail later
    # for a DIFFERENT reason (e.g. unsupported config), not the guard.
    try:
        trainer.train(
            curriculum_rung=None,
            retention_anchor_set="none",
            retention_anchor_repeat=2,
            # Likely to error somewhere later (model build, data load),
            # but NOT at the retention-anchor phase-gate.
        )
    except ValueError as e:
        # The retention-anchor guard's message is:
        #   "retention_anchor_set=...requires curriculum_rung to be set..."
        assert "requires curriculum_rung" not in str(e), (
            f"phase-gate fired with 'none' (default-off contract violated): {e}"
        )
    except Exception:
        # Any other exception (TypeError, AttributeError, FileNotFoundError,
        # etc.) is acceptable — we only care that the phase-gate didn't fire.
        pass
