"""Unit tests for language-wrapper finite-support audit infrastructure
(codex msg 1779559495228-f863199b +1 implement L0a as first
language-axis rung).

Pure tests: no model inference, no ckpt load. Asserts L0a support
shape, per-source-rung bucket counts, multiplicity floor at default
recipe, parallel-aggregate independence from math A0.
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

from calm.hrm_text_158.curriculum.exhaustive_supports import (
    EXHAUSTIVE_ACTIVE_RUNGS,
    EXHAUSTIVE_EXPECTED_AGGREGATE,
)
from calm.hrm_text_158.curriculum.language_supports import (
    LANGUAGE_ACTIVE_RUNGS,
    LANGUAGE_EXPECTED_AGGREGATE,
    LANGUAGE_EXPECTED_COUNTS,
    build_language_supports,
    language_source_rung_buckets,
)


def test_language_active_rungs_contains_l0a_and_l0b() -> None:
    """Active language rungs are L0a and L0b after Slice D.1 (codex msg
    1779567887201). L0c+ are future slices."""
    assert LANGUAGE_ACTIVE_RUNGS == ("L0a", "L0b")


def test_language_aggregate_equals_460() -> None:
    """Two language rungs × 230 each = 460. Per codex msg 1779567887201
    Slice D.1: LANGUAGE_EXPECTED_AGGREGATE extends 230 → 460 when L0b
    lands; per-rung counts unchanged."""
    assert LANGUAGE_EXPECTED_AGGREGATE == 460
    assert LANGUAGE_EXPECTED_COUNTS["L0a"] == 230
    assert LANGUAGE_EXPECTED_COUNTS["L0b"] == 230


def test_build_language_supports_l0a_shape() -> None:
    """L0a 230 (question, expected, source_rung) triples."""
    supports = build_language_supports()
    assert "L0a" in supports
    assert len(supports["L0a"]) == 230
    for row in supports["L0a"]:
        assert isinstance(row, tuple) and len(row) == 3, (
            f"row must be (question, expected, source_rung) triple; got {row!r}"
        )
        q, exp, src = row
        assert isinstance(q, str) and q.startswith("what's ")
        assert isinstance(exp, int)
        assert isinstance(src, str)


def test_l0a_per_source_rung_counts() -> None:
    """Per-source-rung counts match the bounded stratified spec."""
    supports = build_language_supports()
    by_source: dict[str, int] = {}
    for q, exp, src in supports["L0a"]:
        by_source[src] = by_source.get(src, 0) + 1
    expected = {
        "R0": 20,
        "R1_plus_0": 10,
        "R1_0_plus_A": 10,
        "R1_minus_0": 10,
        "R1b1": 20, "R1b2": 20, "R1b3": 20, "R1b4v2": 20,
        "R1b5": 20, "R1b6": 20, "R1b7": 20, "R1b8": 20, "R1b9": 20,
    }
    assert by_source == expected, f"per-source counts mismatch: {by_source}"
    assert sum(expected.values()) == 230


def test_l0a_buckets_helper_returns_canonical_order() -> None:
    buckets = language_source_rung_buckets("L0a")
    assert buckets == [
        "R0",
        "R1_plus_0", "R1_0_plus_A", "R1_minus_0",
        "R1b1", "R1b2", "R1b3", "R1b4v2",
        "R1b5", "R1b6", "R1b7", "R1b8", "R1b9",
    ]


def test_l0a_buckets_unknown_rung_raises() -> None:
    # L0b is now valid (Slice D.1). Use a definitely-unknown name.
    with pytest.raises(ValueError, match="unknown language rung"):
        language_source_rung_buckets("L0z_does_not_exist")


def test_math_a0_unchanged_at_1255() -> None:
    """Codex msg 1779559495228 invariant: math A0 export stays pure
    and stable; language supports are a PARALLEL audit surface, not
    blended into math aggregate."""
    assert EXHAUSTIVE_EXPECTED_AGGREGATE == 1255, (
        f"math A0 aggregate must stay 1255 (R0..R1b9); got {EXHAUSTIVE_EXPECTED_AGGREGATE}"
    )
    # And L0a-as-source-rungs are NOT in math active rungs.
    assert "L0a" not in EXHAUSTIVE_ACTIVE_RUNGS


def test_l0a_train_held_disjoint() -> None:
    """L0a train ∩ L0a held = ∅ (within-L0a disjoint invariant)."""
    from calm.hrm_text_158.curriculum.generators import _enumerate_partition_l0a
    train, held = _enumerate_partition_l0a(seed=42)
    train_qs = {r["question"] for r in train}
    held_qs = {r["question"] for r in held}
    overlap = train_qs & held_qs
    assert not overlap, f"L0a train ∩ held overlap: {sorted(overlap)[:5]}"


def test_l0a_partition_counts_184_train_46_held() -> None:
    """Exact partition spec: 184 train + 46 held = 230 total."""
    from calm.hrm_text_158.curriculum.generators import _enumerate_partition_l0a
    train, held = _enumerate_partition_l0a(seed=42)
    assert len(train) == 184, f"L0a train: {len(train)}"
    assert len(held) == 46, f"L0a held: {len(held)}"


def test_l0a_per_bucket_train_held_split() -> None:
    """Per-source-rung train/held split matches codex spec:
    R0 16/4, R1 sub-templates 8/2 each, R1bN each 16/4."""
    from calm.hrm_text_158.curriculum.generators import _enumerate_partition_l0a
    train, held = _enumerate_partition_l0a(seed=42)

    def count_by_source(rows: list[dict]) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in rows:
            out[r["source_rung"]] = out.get(r["source_rung"], 0) + 1
        return out

    train_by = count_by_source(train)
    held_by = count_by_source(held)

    assert train_by["R0"] == 16 and held_by["R0"] == 4
    for sub in ("R1_plus_0", "R1_0_plus_A", "R1_minus_0"):
        assert train_by[sub] == 8, f"{sub} train: {train_by[sub]}"
        assert held_by[sub] == 2, f"{sub} held: {held_by[sub]}"
    for rung in ("R1b1", "R1b2", "R1b3", "R1b4v2", "R1b5", "R1b6", "R1b7", "R1b8", "R1b9"):
        assert train_by[rung] == 16, f"{rung} train: {train_by[rung]}"
        assert held_by[rung] == 4, f"{rung} held: {held_by[rung]}"


def test_l0a_one_digit_exhaustive_in_train() -> None:
    """All R0 A∈{0..9} and R1bN A∈{1..9} one_digit picks must be in
    train (codex spec: "include all one-digit ...")."""
    from calm.hrm_text_158.curriculum.generators import _enumerate_partition_l0a
    train, held = _enumerate_partition_l0a(seed=42)
    train_qs = {r["question"] for r in train}
    held_qs = {r["question"] for r in held}

    # R0 one_digit: A=0..9
    for a in range(0, 10):
        assert f"what's {a}?" in train_qs, f"R0 one_digit {a} missing from train"
        assert f"what's {a}?" not in held_qs

    # R1bN one_digit: A=1..9 for each K=1..K=8 plus rung + K=-1 minus
    r1b_ops = [(" plus 1",), (" minus 1",), (" plus 2",), (" plus 3",),
               (" plus 4",), (" plus 5",), (" plus 6",), (" plus 7",), (" plus 8",)]
    for (op,) in r1b_ops:
        for a in range(1, 10):
            q = f"what's {a}{op}?"
            assert q in train_qs, f"{q!r} missing from L0a train"
            assert q not in held_qs


def test_l0a_template_shape_whats_math() -> None:
    """Every L0a row starts with `what's ` (contraction, NOT canonical
    `what is `). Distinguishes L0a paraphrase from R0..R1b9 math rows."""
    from calm.hrm_text_158.curriculum.generators import _enumerate_partition_l0a
    train, held = _enumerate_partition_l0a(seed=42)
    for r in train + held:
        q = r["question"]
        assert q.startswith("what's "), f"L0a row must start with `what's `: {q!r}"
        assert not q.startswith("what is "), f"L0a row must NOT start with `what is `: {q!r}"


def test_l0a_math_semantics_preserved() -> None:
    """Expected values match parent R0..R1b9 math semantics exactly.
    L0a does NOT introduce new math operations or values."""
    from calm.hrm_text_158.curriculum.generators import _enumerate_partition_l0a
    train, held = _enumerate_partition_l0a(seed=42)
    for r in train + held:
        q, exp, src = r["question"], r["expected"], r["source_rung"]
        # Parse semantics by template shape
        if src == "R0":
            # `what's N?` → N
            n = int(q[len("what's "):-1])
            assert exp == n, f"R0 row expected mismatch: {r}"
        elif src == "R1_plus_0":
            a = int(q[len("what's "):-len(" plus 0?")])
            assert exp == a
        elif src == "R1_0_plus_A":
            a = int(q[len("what's 0 plus "):-1])
            assert exp == a
        elif src == "R1_minus_0":
            a = int(q[len("what's "):-len(" minus 0?")])
            assert exp == a
        elif src.startswith("R1b"):
            # `what's A {plus,minus} K?`
            if " plus " in q:
                a_str, k_str = q[len("what's "):-1].split(" plus ")
                a, k = int(a_str), int(k_str)
                assert exp == a + k, f"{src} row expected mismatch: {r}"
            elif " minus " in q:
                a_str, k_str = q[len("what's "):-1].split(" minus ")
                a, k = int(a_str), int(k_str)
                assert exp == a - k, f"{src} row expected mismatch: {r}"


def test_l0a_multiplicity_meets_10x_floor() -> None:
    """Default recipe (n_train=10000, rr=0.65) yields n_new=3500
    against unique_train_count=184; multiplicity >= 10x floor."""
    n_train = 10000
    replay_ratio = 0.65
    n_new = int(n_train * (1.0 - replay_ratio))
    from calm.hrm_text_158.curriculum.generators import _enumerate_partition_l0a
    train, _ = _enumerate_partition_l0a(seed=42)
    unique_train = len(train)
    multiplicity = n_new / unique_train
    assert multiplicity >= 10.0, (
        f"L0a multiplicity {multiplicity:.2f}x below 10x floor; "
        f"n_new={n_new} unique_train={unique_train}"
    )
    # Sanity: expected ~19x
    assert 18.0 <= multiplicity <= 20.0, (
        f"L0a expected ~19x multiplicity; got {multiplicity:.2f}x"
    )


def test_l0a_partition_stable_across_pythonhashseed() -> None:
    """Deterministic partition: must produce identical train/held
    sequences across PYTHONHASHSEED values (using _stable_seed infra)."""
    snippet = (
        "from calm.hrm_text_158.curriculum.generators import _enumerate_partition_l0a; "
        "train, held = _enumerate_partition_l0a(seed=42); "
        "ts = sorted(r['question'] for r in train); "
        "hs = sorted(r['question'] for r in held); "
        "print('||'.join(ts) + '###' + '||'.join(hs))"
    )
    out1 = subprocess.check_output(
        [sys.executable, "-c", snippet],
        env={**os.environ, "PYTHONHASHSEED": "0"},
    ).decode().strip()
    out2 = subprocess.check_output(
        [sys.executable, "-c", snippet],
        env={**os.environ, "PYTHONHASHSEED": "777"},
    ).decode().strip()
    out3 = subprocess.check_output(
        [sys.executable, "-c", snippet],
        env={**os.environ, "PYTHONHASHSEED": "random"},
    ).decode().strip()
    assert out1 == out2 == out3, "L0a partition diverged across PYTHONHASHSEED"


def test_l0a_partition_changes_with_seed() -> None:
    """Different seeds produce different two_digit picks (one_digit
    exhaustive coverage is the same; two_digit sampling is seed-deterministic)."""
    from calm.hrm_text_158.curriculum.generators import _enumerate_partition_l0a
    train_a, _ = _enumerate_partition_l0a(seed=42)
    train_b, _ = _enumerate_partition_l0a(seed=17)
    qs_a = {r["question"] for r in train_a}
    qs_b = {r["question"] for r in train_b}
    # Some overlap (one_digit exhaustive), but not identical
    assert qs_a != qs_b, "L0a partition must depend on seed"


# ============================================================================ #
# Slice 2 probe integration tests (codex msg 1779560820500-88e4e540 +1 implement
# slice 2 with 2-file scope: probe + tests). Seed contract: probe defaults to
# ckpt's curriculum_seed, explicit override warns on mismatch + writes BOTH
# values to JSON, missing-seed-without-override fails BEFORE ckpt load.
# ============================================================================ #


def test_probe_language_default_seed_from_ckpt() -> None:
    """Codex msg 1779560820500: probe defaults to ckpt's stored
    curriculum_seed (no hardcoded 42)."""
    from unittest.mock import patch, MagicMock
    import scripts.probe_hrm_text_158 as probe_mod

    fake_ckpt = {"step": 1500, "config": {"max_seq_len": 384, "curriculum_seed": 17}}
    with patch.multiple(probe_mod,
                        torch=MagicMock(load=MagicMock(return_value=fake_ckpt)),
                        _build_model_from_ckpt=MagicMock(return_value=(MagicMock(), MagicMock())),
                        _decode_greedy_no_cache=MagicMock(
                            return_value=("0", False, True))):
        out = probe_mod.probe_language_finite_supports(
            ckpt_path="dummy",
            device="cpu",
            use_cached_ternary_infer=False,
            use_kv_cache_decode=False,
            use_batched_probe_eval=False,
        )
    assert out["audit_seed"] == 17, f"audit_seed should default to ckpt's curriculum_seed=17"
    assert out["ckpt_curriculum_seed"] == 17
    assert out["seed_mismatch"] is False


def test_probe_language_explicit_override_with_mismatch_warns() -> None:
    """Codex msg 1779560820500: explicit --language-audit-seed that
    differs from ckpt seed warns + records BOTH values in JSON."""
    from unittest.mock import patch, MagicMock
    import scripts.probe_hrm_text_158 as probe_mod

    fake_ckpt = {"step": 1500, "config": {"max_seq_len": 384, "curriculum_seed": 17}}
    with patch.multiple(probe_mod,
                        torch=MagicMock(load=MagicMock(return_value=fake_ckpt)),
                        _build_model_from_ckpt=MagicMock(return_value=(MagicMock(), MagicMock())),
                        _decode_greedy_no_cache=MagicMock(
                            return_value=("0", False, True))):
        out = probe_mod.probe_language_finite_supports(
            ckpt_path="dummy",
            audit_seed=137,
            device="cpu",
            use_cached_ternary_infer=False,
            use_kv_cache_decode=False,
            use_batched_probe_eval=False,
        )
    assert out["audit_seed"] == 137
    assert out["ckpt_curriculum_seed"] == 17
    assert out["seed_mismatch"] is True, "mismatch flag must be True when seeds differ"


def test_probe_language_explicit_override_match_no_mismatch() -> None:
    """Explicit override that matches ckpt seed is fine, no mismatch flag."""
    from unittest.mock import patch, MagicMock
    import scripts.probe_hrm_text_158 as probe_mod

    fake_ckpt = {"step": 1500, "config": {"max_seq_len": 384, "curriculum_seed": 17}}
    with patch.multiple(probe_mod,
                        torch=MagicMock(load=MagicMock(return_value=fake_ckpt)),
                        _build_model_from_ckpt=MagicMock(return_value=(MagicMock(), MagicMock())),
                        _decode_greedy_no_cache=MagicMock(
                            return_value=("0", False, True))):
        out = probe_mod.probe_language_finite_supports(
            ckpt_path="dummy",
            audit_seed=17,
            device="cpu",
            use_cached_ternary_infer=False,
            use_kv_cache_decode=False,
            use_batched_probe_eval=False,
        )
    assert out["audit_seed"] == 17
    assert out["ckpt_curriculum_seed"] == 17
    assert out["seed_mismatch"] is False


def test_probe_language_missing_seed_no_override_fails() -> None:
    """Codex msg 1779560820500: if ckpt config has no curriculum_seed AND
    no --language-audit-seed override, probe fails BEFORE ckpt load is
    interpreted as inference-runnable. No silent fallback to 42."""
    from unittest.mock import patch, MagicMock
    import scripts.probe_hrm_text_158 as probe_mod

    fake_ckpt = {"step": 1500, "config": {"max_seq_len": 384}}  # NO curriculum_seed
    with patch.multiple(probe_mod,
                        torch=MagicMock(load=MagicMock(return_value=fake_ckpt))):
        with pytest.raises(ValueError, match="curriculum_seed"):
            probe_mod.probe_language_finite_supports(
                ckpt_path="dummy",
                audit_seed=None,
                device="cpu",
                use_cached_ternary_infer=False,
                use_kv_cache_decode=False,
                use_batched_probe_eval=False,
            )


def test_probe_language_per_source_rung_buckets_sum_to_230() -> None:
    """Per-source-rung breakdown in audit JSON must sum to 230
    (matches LANGUAGE_EXPECTED_AGGREGATE)."""
    from unittest.mock import patch, MagicMock
    import scripts.probe_hrm_text_158 as probe_mod

    fake_ckpt = {"step": 1500, "config": {"max_seq_len": 384, "curriculum_seed": 17}}
    with patch.multiple(probe_mod,
                        torch=MagicMock(load=MagicMock(return_value=fake_ckpt)),
                        _build_model_from_ckpt=MagicMock(return_value=(MagicMock(), MagicMock())),
                        _decode_greedy_no_cache=MagicMock(
                            return_value=("0", False, True))):
        out = probe_mod.probe_language_finite_supports(
            ckpt_path="dummy",
            audit_seed=17,
            device="cpu",
            use_cached_ternary_infer=False,
            use_kv_cache_decode=False,
            use_batched_probe_eval=False,
        )
    l0a = out["results"]["L0a"]
    assert l0a["n_total"] == 230, f"L0a total: {l0a['n_total']}"
    bucket_total = sum(b["n_total"] for b in l0a["by_source_rung"].values())
    assert bucket_total == 230, (
        f"per-source-rung bucket sum must equal 230; got {bucket_total}"
    )
    # All 13 source-rung buckets are present.
    expected_buckets = {
        "R0", "R1_plus_0", "R1_0_plus_A", "R1_minus_0",
        "R1b1", "R1b2", "R1b3", "R1b4v2",
        "R1b5", "R1b6", "R1b7", "R1b8", "R1b9",
    }
    assert set(l0a["by_source_rung"].keys()) == expected_buckets


def test_probe_language_aggregate_separate_from_math() -> None:
    """Codex msg 1779559495228 invariant + 1779567887201 D.1 extension:
    language audit emits its own aggregate separate from math A0.
    JSON `aggregate.expected_aggregate` equals LANGUAGE_EXPECTED_AGGREGATE
    (= 460 after L0b lands), NOT blended with math 1255."""
    from unittest.mock import patch, MagicMock
    import scripts.probe_hrm_text_158 as probe_mod

    fake_ckpt = {"step": 1500, "config": {"max_seq_len": 384, "curriculum_seed": 17}}
    with patch.multiple(probe_mod,
                        torch=MagicMock(load=MagicMock(return_value=fake_ckpt)),
                        _build_model_from_ckpt=MagicMock(return_value=(MagicMock(), MagicMock())),
                        _decode_greedy_no_cache=MagicMock(
                            return_value=("0", False, True))):
        out = probe_mod.probe_language_finite_supports(
            ckpt_path="dummy",
            audit_seed=17,
            device="cpu",
            use_cached_ternary_infer=False,
            use_kv_cache_decode=False,
            use_batched_probe_eval=False,
        )
    assert out["aggregate"]["expected_aggregate"] == 460, (
        f"language aggregate must be 460 (L0a + L0b, NOT blended with math 1255); "
        f"got {out['aggregate']['expected_aggregate']}"
    )
    assert out["aggregate"]["n_total"] == 460
    # `active_language_rungs` distinguishes language from math; L0a then L0b in order
    assert out["active_language_rungs"] == ["L0a", "L0b"]


def test_probe_language_cli_conflicts_with_curriculum_rungs() -> None:
    """--language-supports conflicts with --curriculum-rungs (mutually
    exclusive); pre-check fails BEFORE ckpt load."""
    snippet = (
        "import subprocess, sys; "
        "r = subprocess.run([sys.executable, '-m', 'scripts.probe_hrm_text_158', "
        "'--ckpt-path', '/nonexistent.pt', "
        "'--curriculum-rungs', 'R0', "
        "'--language-supports'], capture_output=True, text=True); "
        "print('EXIT', r.returncode); "
        "print('STDERR', r.stderr[:500])"
    )
    out = subprocess.check_output([sys.executable, "-c", snippet], cwd=os.getcwd()).decode()
    assert "EXIT 0" not in out, f"expected nonzero exit; got: {out}"
    assert ("conflicts with --curriculum-rungs" in out
            or "mutually exclusive" in out), (
        f"expected explicit conflict error; got: {out}"
    )


def test_probe_language_cli_conflicts_with_exhaustive() -> None:
    """--language-supports conflicts with --exhaustive-finite-supports
    (math and language are separate probe modes per codex spec)."""
    snippet = (
        "import subprocess, sys; "
        "r = subprocess.run([sys.executable, '-m', 'scripts.probe_hrm_text_158', "
        "'--ckpt-path', '/nonexistent.pt', "
        "'--exhaustive-finite-supports', "
        "'--language-supports'], capture_output=True, text=True); "
        "print('EXIT', r.returncode); "
        "print('STDERR', r.stderr[:500])"
    )
    out = subprocess.check_output([sys.executable, "-c", snippet], cwd=os.getcwd()).decode()
    assert "EXIT 0" not in out, f"expected nonzero exit; got: {out}"
    assert ("conflicts with --exhaustive-finite-supports" in out
            or "mutually exclusive" in out
            or "separate probe modes" in out), (
        f"expected explicit conflict error; got: {out}"
    )


def test_probe_language_dispatch_path_selection() -> None:
    """Verify dispatch_path string matches existing exhaustive convention."""
    from unittest.mock import patch, MagicMock
    import scripts.probe_hrm_text_158 as probe_mod

    fake_ckpt = {"step": 1500, "config": {"max_seq_len": 384, "curriculum_seed": 17}}
    common = {
        "torch": MagicMock(load=MagicMock(return_value=fake_ckpt)),
        "_build_model_from_ckpt": MagicMock(return_value=(MagicMock(), MagicMock())),
        "_decode_greedy_no_cache": MagicMock(return_value=("0", False, True)),
        "_decode_greedy_cached": MagicMock(return_value=("0", False, True)),
        "_run_rows_batched": MagicMock(
            return_value=([("0", False, True)] * 230, {})
        ),
    }
    # Scenario A: no flags → scalar_no_cache
    with patch.multiple(probe_mod, **common):
        out = probe_mod.probe_language_finite_supports(
            ckpt_path="dummy", audit_seed=17, device="cpu",
            use_cached_ternary_infer=False, use_kv_cache_decode=False,
            use_batched_probe_eval=False,
        )
        assert out["dispatch_path"] == "scalar_no_cache"

    # Scenario B: kv only → scalar_kv_cache
    with patch.multiple(probe_mod, **common):
        out = probe_mod.probe_language_finite_supports(
            ckpt_path="dummy", audit_seed=17, device="cpu",
            use_cached_ternary_infer=False, use_kv_cache_decode=True,
            use_batched_probe_eval=False,
        )
        assert out["dispatch_path"] == "scalar_kv_cache"

    # Scenario C: both → batched_kv_cache
    with patch.multiple(probe_mod, **common):
        out = probe_mod.probe_language_finite_supports(
            ckpt_path="dummy", audit_seed=17, device="cpu",
            use_cached_ternary_infer=False, use_kv_cache_decode=True,
            use_batched_probe_eval=True,
        )
        assert out["dispatch_path"] == "batched_kv_cache"


def test_probe_language_output_json_creates_parent_dirs(tmp_path) -> None:
    """Output JSON writer mkdir -p's parent dir, mirroring exhaustive."""
    from unittest.mock import patch, MagicMock
    import json as json_mod
    import scripts.probe_hrm_text_158 as probe_mod

    fake_ckpt = {"step": 1500, "config": {"max_seq_len": 384, "curriculum_seed": 17}}
    nested = tmp_path / "deep" / "nested" / "lang_audit.json"
    assert not nested.parent.exists()
    with patch.multiple(probe_mod,
                        torch=MagicMock(load=MagicMock(return_value=fake_ckpt)),
                        _build_model_from_ckpt=MagicMock(return_value=(MagicMock(), MagicMock())),
                        _decode_greedy_no_cache=MagicMock(
                            return_value=("0", False, True))):
        probe_mod.probe_language_finite_supports(
            ckpt_path="dummy", audit_seed=17, device="cpu",
            output_json=str(nested),
            use_cached_ternary_infer=False, use_kv_cache_decode=False,
            use_batched_probe_eval=False,
        )
    assert nested.exists()
    payload = json_mod.loads(nested.read_text())
    assert "language_l0a" not in payload  # not a blended key
    assert "results" in payload and "L0a" in payload["results"]
    # Slice D.1 extends: L0b also present, aggregate now 460
    assert "L0b" in payload["results"]
    assert payload["aggregate"]["expected_aggregate"] == 460


# ============================================================================ #
# Slice D.1 L0b tests (codex msg 1779567887201-1cf4f485 +1 implement second
# language-axis rung as L0a mirror with template `calculate <math>.`).
# Parallels every L0a finite-support invariant: 230 rows = 184 train + 46 held,
# 13 source-rung buckets, train ∩ held = ∅, math A0 unchanged at 1255.
# ============================================================================ #


def test_build_language_supports_l0b_shape() -> None:
    """L0b 230 (question, expected, source_rung) triples; template
    `calculate <math>.` (not L0a's `what's <math>?`)."""
    supports = build_language_supports()
    assert "L0b" in supports
    assert len(supports["L0b"]) == 230
    for row in supports["L0b"]:
        assert isinstance(row, tuple) and len(row) == 3, (
            f"row must be (question, expected, source_rung) triple; got {row!r}"
        )
        q, exp, src = row
        assert isinstance(q, str) and q.startswith("calculate ")
        assert q.endswith(".")
        assert isinstance(exp, int)
        assert isinstance(src, str)


def test_l0b_per_source_rung_counts() -> None:
    """Per-source-rung counts identical to L0a (only template differs)."""
    supports = build_language_supports()
    by_source: dict[str, int] = {}
    for q, exp, src in supports["L0b"]:
        by_source[src] = by_source.get(src, 0) + 1
    expected = {
        "R0": 20,
        "R1_plus_0": 10,
        "R1_0_plus_A": 10,
        "R1_minus_0": 10,
        "R1b1": 20, "R1b2": 20, "R1b3": 20, "R1b4v2": 20,
        "R1b5": 20, "R1b6": 20, "R1b7": 20, "R1b8": 20, "R1b9": 20,
    }
    assert by_source == expected, f"per-source counts mismatch: {by_source}"
    assert sum(expected.values()) == 230


def test_l0b_buckets_helper_returns_canonical_order() -> None:
    """L0b returns the same bucket order as L0a (mirrored shape)."""
    buckets = language_source_rung_buckets("L0b")
    assert buckets == [
        "R0",
        "R1_plus_0", "R1_0_plus_A", "R1_minus_0",
        "R1b1", "R1b2", "R1b3", "R1b4v2",
        "R1b5", "R1b6", "R1b7", "R1b8", "R1b9",
    ]


def test_l0b_train_held_disjoint() -> None:
    """L0b train ∩ L0b held = ∅."""
    from calm.hrm_text_158.curriculum.generators import _enumerate_partition_l0b
    train, held = _enumerate_partition_l0b(seed=42)
    train_qs = {r["question"] for r in train}
    held_qs = {r["question"] for r in held}
    overlap = train_qs & held_qs
    assert not overlap, f"L0b train ∩ held overlap: {sorted(overlap)[:5]}"


def test_l0b_partition_counts_184_train_46_held() -> None:
    """Exact partition spec: 184 train + 46 held = 230 total (L0a mirror)."""
    from calm.hrm_text_158.curriculum.generators import _enumerate_partition_l0b
    train, held = _enumerate_partition_l0b(seed=42)
    assert len(train) == 184, f"L0b train: {len(train)}"
    assert len(held) == 46, f"L0b held: {len(held)}"


def test_l0b_template_shape_calculate_math() -> None:
    """L0b emits ONLY `calculate <expr>.` (period terminator). Distinct
    from L0a's `what's <expr>?` AND from canonical math `what is <expr>?`."""
    from calm.hrm_text_158.curriculum.generators import _enumerate_partition_l0b
    train, held = _enumerate_partition_l0b(seed=42)
    for row in train + held:
        q = row["question"]
        assert q.startswith("calculate "), f"L0b row must start with `calculate `: {q!r}"
        assert q.endswith("."), f"L0b row must end `.`: {q!r}"
        assert not q.startswith("what's "), f"L0b must NOT use L0a's prefix: {q!r}"
        assert not q.startswith("what is "), f"L0b must NOT use canonical math prefix: {q!r}"


def test_l0b_math_semantics_preserved() -> None:
    """L0b expected values match the wrapped math primitive (sanity:
    parse the expression and verify it equals expected).
    """
    import re
    from calm.hrm_text_158.curriculum.generators import _enumerate_partition_l0b
    train, held = _enumerate_partition_l0b(seed=42)
    pat_plus = re.compile(r"^calculate (\d+) plus (\d+)\.$")
    pat_minus = re.compile(r"^calculate (\d+) minus (\d+)\.$")
    pat_id = re.compile(r"^calculate (\d+)\.$")
    for row in train + held:
        q = row["question"]
        m = pat_plus.match(q)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            assert row["expected"] == a + b, (
                f"L0b plus row arithmetic mismatch: {q} expected={row['expected']} vs a+b={a+b}"
            )
            continue
        m = pat_minus.match(q)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            assert row["expected"] == a - b, (
                f"L0b minus row arithmetic mismatch: {q} expected={row['expected']} vs a-b={a-b}"
            )
            continue
        m = pat_id.match(q)
        if m:
            n = int(m.group(1))
            assert row["expected"] == n, (
                f"L0b identity row mismatch: {q} expected={row['expected']} vs n={n}"
            )
            continue
        raise AssertionError(f"Unrecognized L0b question pattern: {q!r}")


def test_l0b_multiplicity_meets_10x_floor() -> None:
    """At default recipe (n_new=3500 in train, unique=184):
    multiplicity = 3500/184 ≈ 19x, well above 10x floor."""
    from calm.hrm_text_158.curriculum.generators import _enumerate_partition_l0b
    train, _ = _enumerate_partition_l0b(seed=42)
    unique = len({r["question"] for r in train})
    assert unique == 184
    # Default recipe n_new at L0b training = 3500 (rr=0.65 * 10000 = 3500 new)
    multiplicity = 3500 / unique
    assert multiplicity > 10.0, (
        f"L0b multiplicity below floor: {multiplicity:.2f}x at n_new=3500/unique=184"
    )


def test_l0b_partition_stable_across_runs() -> None:
    """Same seed → identical partition (deterministic seed namespace)."""
    from calm.hrm_text_158.curriculum.generators import _enumerate_partition_l0b
    train1, held1 = _enumerate_partition_l0b(seed=17)
    train2, held2 = _enumerate_partition_l0b(seed=17)
    assert train1 == train2
    assert held1 == held2


def test_l0b_partition_changes_with_seed() -> None:
    """Different seed → different two_digit picks (one_digit exhaustive
    stays identical by construction)."""
    from calm.hrm_text_158.curriculum.generators import _enumerate_partition_l0b
    train_a, _ = _enumerate_partition_l0b(seed=17)
    train_b, _ = _enumerate_partition_l0b(seed=42)
    # one_digit rows are exhaustive across seeds; differ on two_digit picks.
    # At minimum, the two complete sets should differ.
    assert train_a != train_b, (
        "Different seeds must produce different L0b partitions"
    )


def test_l0a_l0b_partition_seed_namespaces_disjoint() -> None:
    """L0a and L0b use distinct seed namespaces (`L0a_partition` vs
    `L0b_partition`). Pin via the two_digit pick determinism: same seed
    must produce DIFFERENT two_digit slices for the two rungs (otherwise
    the seed namespaces are collapsing)."""
    from calm.hrm_text_158.curriculum.generators import (
        _enumerate_partition_l0a,
        _enumerate_partition_l0b,
    )
    l0a_train, _ = _enumerate_partition_l0a(seed=17)
    l0b_train, _ = _enumerate_partition_l0b(seed=17)
    # Extract two_digit R0 rows from each (those go through the seeded sampler)
    def _r0_two_digit_questions(rows):
        # R0 two_digit rows are the ones with source_rung=R0 AND numeric value >= 10
        return sorted(
            r["expected"] for r in rows
            if r["source_rung"] == "R0" and r["expected"] >= 10
        )
    l0a_r0_two = _r0_two_digit_questions(l0a_train)
    l0b_r0_two = _r0_two_digit_questions(l0b_train)
    # Both partitions should pick 6 R0 two_digit rows for train, but the
    # specific values should differ given distinct seed namespaces.
    assert l0a_r0_two != l0b_r0_two, (
        f"L0a/L0b two_digit R0 picks must differ under same outer seed; "
        f"L0a={l0a_r0_two} L0b={l0b_r0_two}"
    )
