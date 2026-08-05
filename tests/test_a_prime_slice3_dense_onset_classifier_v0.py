"""Battery for A′ slice-3 dense onset/shape classifier (plan v5)."""
from __future__ import annotations

import io
import json
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest

from scripts.a_prime_slice3_onset_reducer_v0 import (
    BASELINE_COUNT,
    CLASS_PRIORITY,
    HORIZONS,
    classify_from_counts,
    classify_suite,
    extract_horizon_point,
    ELIGIBLE_MODULE_DEFAULT,
)
from scripts.a_prime_slice3_dense_onset_classifier_v0 import (
    finalize_dual_key,
    main as classifier_main,
)


def _counts_map(seq_incl_step0: list[int]) -> dict[int, int]:
    """seq = [step0, N1, N5, N10, N20, N35, N50]."""
    assert len(seq_incl_step0) == 7
    return {n: seq_incl_step0[i + 1] for i, n in enumerate(HORIZONS)}


def test_counterexample_unclassified_shape() -> None:
    # plan v5 battery required fixture
    seq = [1484, 1084, 684, 704, 304, 62, 62]
    r = classify_from_counts(_counts_map(seq), liveness_ok=True, prefix_ok=True)
    assert r["branch"] == "UNCLASSIFIED_SHAPE"


def test_collapse_at_step_1() -> None:
    # D=1484-62=1422; need count_N1 <= 1484 - 0.5*1422 = 773
    seq = [1484, 700, 200, 100, 80, 70, 62]
    r = classify_from_counts(_counts_map(seq), liveness_ok=True, prefix_ok=True)
    assert r["branch"] == "COLLAPSE_AT_STEP_1"


def test_nonmonotone_recovery_before_collapse() -> None:
    # large N1 drop would be COLLAPSE, but recovery >=30 after forces NONMONOTONE first
    seq = [1484, 700, 750, 100, 80, 70, 62]  # gain 50 from N1→N5
    r = classify_from_counts(_counts_map(seq), liveness_ok=True, prefix_ok=True)
    assert r["branch"] == "NONMONOTONE_OR_MULTI_CLIFF"


def test_threshold_cliff() -> None:
    # one bin carries >=50% of D; no second >=30%
    # D=1422; 50%=711; 30%=426.6
    # put big drop 1→5 of 800, then small remainder
    seq = [1484, 1480, 680, 650, 640, 630, 62]
    # drops: 1→5:800, 5→10:30, 10→20:10, 20→35:10, 35→50:568 — 568>=426 → multi mid
    # adjust so only one big and no second mid
    seq = [1484, 1480, 700, 690, 680, 670, 62]
    # drops: 4,780,10,10,10,608 → 780>=711 big, 608>=426 mid second → multi cliff
    seq = [1484, 1480, 760, 750, 740, 730, 62]
    # drops: 4,720,10,10,10,668 → 720 big, 668 mid → still multi
    # need second largest < 0.30*D ≈ 426.6
    seq = [1484, 1480, 760, 750, 740, 400, 62]
    # drops: 4,720,10,10,340,338 → one big 720, mids: only 720 >=426 → one mid OK
    r = classify_from_counts(_counts_map(seq), liveness_ok=True, prefix_ok=True)
    assert r["branch"] == "THRESHOLD_CLIFF"


def test_gradual_drift() -> None:
    # non-increasing, every drop < 0.5*D, D>0, N50 in band
    seq = [1484, 1400, 1200, 1000, 800, 400, 62]
    r = classify_from_counts(_counts_map(seq), liveness_ok=True, prefix_ok=True)
    assert r["branch"] == "GRADUAL_DRIFT"


def test_no_reproduction_endpoint_drift() -> None:
    seq = [1484, 1400, 1200, 1000, 800, 400, 10]  # 10 outside [47,77]
    r = classify_from_counts(_counts_map(seq), liveness_ok=True, prefix_ok=True)
    assert r["branch"] == "NO_REPRODUCTION_OR_ENDPOINT_DRIFT"


def test_liveness_and_prefix_priority() -> None:
    seq = [1484, 1400, 1200, 1000, 800, 400, 62]
    r = classify_from_counts(_counts_map(seq), liveness_ok=False, prefix_ok=True)
    assert r["branch"] == "LIVENESS_OR_INSTRUMENT_FAIL"
    r2 = classify_from_counts(_counts_map(seq), liveness_ok=True, prefix_ok=False)
    assert r2["branch"] == "PREFIX_EQUIVALENCE_FAIL"


def test_class_priority_order_constant() -> None:
    assert CLASS_PRIORITY[0] == "LIVENESS_OR_INSTRUMENT_FAIL"
    assert CLASS_PRIORITY[3] == "NONMONOTONE_OR_MULTI_CLIFF"
    assert CLASS_PRIORITY[4] == "COLLAPSE_AT_STEP_1"
    assert CLASS_PRIORITY[-1] == "UNCLASSIFIED_SHAPE"


def _minimal_step(k: int, *, tag: str = "ok") -> dict:
    mod = ELIGIBLE_MODULE_DEFAULT
    return {
        "bp_steps": min(k + 1, 5),
        "global_horizon": 50,
        "loss": float(k),
        "metrics": {"exact_accuracy": [0, 1], "accuracy": [1, 3], "loss": [1.0, 3]},
        "q_changed_count": 4096,
        "support_batch": {
            "batch_content_hash16": f"batch{k:02d}{tag}",
            "row_ids": [f"{k}:row"],
        },
        "step_result": {
            "tensor_stats": {
                mod: {
                    "q_sha256_before": f"qb{k:02d}{tag}",
                    "q_sha256_after": f"qa{k:02d}{tag}",
                    "votes_sha256": f"vv{k:02d}{tag}",
                    "applied_flat_indices_hash16": f"ap{k:02d}{tag}",
                }
            }
        },
    }


def _receipt(
    n: int,
    *,
    final_count: int,
    start_count: int = 1484,
    steps_completed: int | None = None,
    parent_ok: bool = True,
    global_horizon: int = 50,
    step_tag: str = "ok",
    corrupt_step: int | None = None,
) -> dict:
    l0b_start = min(230, start_count)
    math_start = start_count - l0b_start
    # split final similarly proportional-ish
    l0b_final = min(230, final_count)
    math_final = max(0, final_count - l0b_final)
    if final_count > 230:
        l0b_final = 15 if final_count == 62 else min(230, final_count // 10)
        math_final = final_count - l0b_final
    step_reports = {str(k): _minimal_step(k, tag=step_tag) for k in range(1, n + 1)}
    if corrupt_step is not None and str(corrupt_step) in step_reports:
        step_reports[str(corrupt_step)]["loss"] = -999.0
    parent = (
        "9b4e311a22787e7d4808bde7bc2953568d767a2ee8ac648942a3f5dbb7b4d5ec"
        if parent_ok
        else "deadbeef" * 8
    )
    return {
        "steps_completed": n if steps_completed is None else steps_completed,
        "parent_hash_before": parent,
        "parent_hash_after": parent if parent_ok else ("cafebabe" * 8),
        "parent_hash_unchanged": parent_ok,
        "terminal_status": {
            "planned_return_code": 0,
            "producer_clean_completion": True,
        },
        "step_reports": step_reports,
        "prior_audit": {
            "enabled": True,
            "requested_supports": ["L0b", "math_a0"],
            "start_reports": {
                "L0b": {
                    "strict_exact_count": 230 if start_count >= 1484 else l0b_start,
                    "strict_exact_total": 230,
                    "strict_exact": f"{230 if start_count >= 1484 else l0b_start}/230",
                },
                "math_a0": {
                    "strict_exact_count": 1254 if start_count >= 1484 else math_start,
                    "strict_exact_total": 1255,
                    "strict_exact": f"{1254 if start_count >= 1484 else math_start}/1255",
                },
            },
            "final_reports": {
                "L0b": {
                    "strict_exact_count": l0b_final,
                    "strict_exact_total": 230,
                    "strict_exact": f"{l0b_final}/230",
                },
                "math_a0": {
                    "strict_exact_count": math_final,
                    "strict_exact_total": 1255,
                    "strict_exact": f"{math_final}/1255",
                },
            },
        },
    }


def _suite_from_seq(seq: list[int], **kwargs) -> dict[int, dict]:
    cm = _counts_map(seq)
    return {n: _receipt(n, final_count=cm[n], **kwargs) for n in HORIZONS}


def test_classify_suite_prefix_fail() -> None:
    seq = [1484, 1400, 1200, 1000, 800, 400, 62]
    recs = _suite_from_seq(seq)
    # corrupt N=5 step 1 vs reference
    recs[5]["step_reports"]["1"]["loss"] = -1.0
    r = classify_suite(recs, skip_prefix=False)
    assert r["branch"] == "PREFIX_EQUIVALENCE_FAIL"


def test_classify_suite_liveness_fail_steps() -> None:
    seq = [1484, 1400, 1200, 1000, 800, 400, 62]
    recs = _suite_from_seq(seq)
    recs[10]["steps_completed"] = 9
    r = classify_suite(recs, skip_prefix=True)
    assert r["branch"] == "LIVENESS_OR_INSTRUMENT_FAIL"


def test_classify_suite_gradual_with_prefix_ok() -> None:
    seq = [1484, 1400, 1200, 1000, 800, 400, 62]
    recs = _suite_from_seq(seq)
    r = classify_suite(recs, skip_prefix=False)
    assert r["branch"] == "GRADUAL_DRIFT"


def test_finalize_dual_key_packet_terminal(tmp_path: Path) -> None:
    seq = [1484, 1400, 1200, 1000, 800, 400, 62]
    classification = classify_from_counts(
        _counts_map(seq), liveness_ok=True, prefix_ok=True
    )
    classification = {
        "schema": "a_prime_slice3_onset_classification/v0",
        "branch": classification["branch"],
        "class_priority": list(CLASS_PRIORITY),
        "horizons": list(HORIZONS),
        "details": classification.get("details", {}),
        "claim_boundary": {"morphology_only": True, "pre_cause": True, "pre_carrier": True},
    }
    run_root = tmp_path / "run"
    buf = io.StringIO()
    err = io.StringIO()
    with redirect_stdout(buf), redirect_stderr(err):
        rc = finalize_dual_key(run_root, classification, source_shas={"input/meta": "abc"})
    assert rc == 0
    out = buf.getvalue()
    assert out.count("PACKET_TERMINAL") == 1
    assert f"PACKET_TERMINAL {classification['branch']}" in out
    assert "INCOMPLETE_FINALIZATION" not in err.getvalue()
    assert (run_root / "terminal_manifest.json").is_file()
    assert (run_root / "terminal_receipt.json").is_file()
    man = json.loads((run_root / "terminal_manifest.json").read_text())
    assert man["branch"] == classification["branch"]
    assert man["terminal_authority"] == "manifest+marker"


def test_finalize_hostile_candidate_branch(tmp_path: Path) -> None:
    classification = {
        "schema": "a_prime_slice3_onset_classification/v0",
        "branch": "GRADUAL_DRIFT",
        "class_priority": list(CLASS_PRIORITY),
        "horizons": list(HORIZONS),
        "details": {},
        "claim_boundary": {"morphology_only": True},
    }
    err = io.StringIO()
    with redirect_stdout(io.StringIO()), redirect_stderr(err):
        rc = finalize_dual_key(
            tmp_path / "run2",
            classification,
            source_shas={},
            inject_candidate_branch="UNCLASSIFIED_SHAPE",
        )
    assert rc == 2
    assert "INCOMPLETE_FINALIZATION" in err.getvalue()
    assert "PACKET_TERMINAL" not in err.getvalue()


def test_extract_horizon_point_start_band() -> None:
    rec = _receipt(1, final_count=1000, start_count=1484)
    pt = extract_horizon_point(rec, expected_n=1)
    assert pt["ok"] is True
    assert pt["final_count"] == 1000


def test_missing_horizon_receipt_liveness() -> None:
    """Unknown-input: drop N=20 from suite → LIVENESS_OR_INSTRUMENT_FAIL."""
    seq = [1484, 1400, 1200, 1000, 800, 400, 62]
    recs = _suite_from_seq(seq)
    del recs[20]
    r = classify_suite(recs, skip_prefix=True)
    assert r["branch"] == "LIVENESS_OR_INSTRUMENT_FAIL"


def test_malformed_cli_input_no_packet_terminal(tmp_path: Path) -> None:
    """Malformed/unknown CLI input fail-closed: nonzero rc, no PACKET_TERMINAL."""
    import io
    from contextlib import redirect_stdout, redirect_stderr

    buf = io.StringIO()
    err = io.StringIO()
    with redirect_stdout(buf), redirect_stderr(err):
        rc = classifier_main(
            [
                "--run-root",
                str(tmp_path / "run"),
                "--horizon-receipt",
                "not-a-pair",
            ]
        )
    assert rc != 0
    assert "PACKET_TERMINAL" not in buf.getvalue()
    assert "PACKET_TERMINAL" not in err.getvalue()


def test_malformed_json_receipt_no_packet_terminal(tmp_path: Path) -> None:
    """Malformed JSON receipt path fail-closed via CLI loader."""
    import io
    from contextlib import redirect_stdout, redirect_stderr

    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    # still need enough args; only N=1 malformed is enough to fail closed before finalize marker
    buf = io.StringIO()
    err = io.StringIO()
    args = ["--run-root", str(tmp_path / "run2"), "--horizon-receipt", f"1={bad}"]
    with redirect_stdout(buf), redirect_stderr(err):
        rc = classifier_main(args)
    assert rc != 0
    assert "PACKET_TERMINAL" not in buf.getvalue()


def test_terminal_status_failure_liveness() -> None:
    """C5a: terminal_status rc!=0 or clean=False → LIVENESS."""
    seq = [1484, 1400, 1200, 1000, 800, 400, 62]
    recs = _suite_from_seq(seq)
    recs[5]["terminal_status"] = {
        "planned_return_code": 1,
        "producer_clean_completion": True,
    }
    r = classify_suite(recs, skip_prefix=True)
    assert r["branch"] == "LIVENESS_OR_INSTRUMENT_FAIL"
    recs2 = _suite_from_seq(seq)
    recs2[10]["terminal_status"] = {
        "planned_return_code": 0,
        "producer_clean_completion": False,
    }
    r2 = classify_suite(recs2, skip_prefix=True)
    assert r2["branch"] == "LIVENESS_OR_INSTRUMENT_FAIL"


def test_incomplete_n50_step_coverage_liveness() -> None:
    """C5b: incomplete N=50 step coverage → LIVENESS."""
    seq = [1484, 1400, 1200, 1000, 800, 400, 62]
    recs = _suite_from_seq(seq)
    del recs[50]["step_reports"]["25"]
    r = classify_suite(recs, skip_prefix=True)
    assert r["branch"] == "LIVENESS_OR_INSTRUMENT_FAIL"


def test_later_step_wrong_global_horizon_liveness() -> None:
    """C5c: later-step wrong global_horizon → LIVENESS."""
    seq = [1484, 1400, 1200, 1000, 800, 400, 62]
    recs = _suite_from_seq(seq)
    recs[20]["step_reports"]["10"]["global_horizon"] = 20
    r = classify_suite(recs, skip_prefix=True)
    assert r["branch"] == "LIVENESS_OR_INSTRUMENT_FAIL"


def test_cli_has_no_skip_prefix() -> None:
    """C5d: --skip-prefix is not a CLI option."""
    import argparse
    from scripts.a_prime_slice3_dense_onset_classifier_v0 import main as _m
    # build parser the same way main does by attempting parse
    with pytest.raises(SystemExit):
        # argparse raises SystemExit on unknown args when using parse_args in main
        # call main with --skip-prefix which must fail
        classifier_main(["--run-root", "/tmp/x", "--skip-prefix"])
    # also ensure help text path: use internal parser recreation
    ap = argparse.ArgumentParser()
    # mirror: flag must not be in the module's main defaults — parse known_args via subprocess-less check
    src = Path("scripts/a_prime_slice3_dense_onset_classifier_v0.py").read_text(encoding="utf-8")
    assert '"--skip-prefix"' not in src and "'--skip-prefix'" not in src


def test_unknown_horizon_key_fail_closed(tmp_path: Path) -> None:
    """C5e: unknown horizon key → nonzero rc, no PACKET_TERMINAL."""
    import io
    from contextlib import redirect_stdout, redirect_stderr

    good = tmp_path / "r.json"
    good.write_text("{}", encoding="utf-8")
    buf = io.StringIO()
    err = io.StringIO()
    with redirect_stdout(buf), redirect_stderr(err):
        rc = classifier_main(
            ["--run-root", str(tmp_path / "run"), "--horizon-receipt", f"2={good}"]
        )
    assert rc != 0
    assert "PACKET_TERMINAL" not in buf.getvalue()


def test_duplicate_horizon_key_fail_closed(tmp_path: Path) -> None:
    """C5f: duplicate horizon key → nonzero rc, no PACKET_TERMINAL."""
    import io
    from contextlib import redirect_stdout, redirect_stderr

    good = tmp_path / "r.json"
    good.write_text("{}", encoding="utf-8")
    buf = io.StringIO()
    err = io.StringIO()
    with redirect_stdout(buf), redirect_stderr(err):
        rc = classifier_main(
            [
                "--run-root",
                str(tmp_path / "run"),
                "--horizon-receipt",
                f"1={good}",
                "--horizon-receipt",
                f"1={good}",
            ]
        )
    assert rc != 0
    assert "PACKET_TERMINAL" not in buf.getvalue()

