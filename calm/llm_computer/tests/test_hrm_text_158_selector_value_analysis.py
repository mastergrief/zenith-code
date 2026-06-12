"""CPU tests for paired selector-value identity/outcome receipt analysis."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from calm.hrm_text_158.native_full_stack.selector_value_analysis import (
    CLASSIFY_WHY_CANNOT_CLAIMS,
    CLASSIFY_WHY_PRIMARY_VERDICTS,
    DEFAULT_STATE_KEY,
    OFF_SCORE_SEMANTICS,
    ON_SCORE_SEMANTICS,
    build_identity_tables,
    build_outcome_tables,
    check_schedule_guards,
    descriptive_step_delta_association,
    extract_cap_window_steps,
    identity_verdict,
    load_paired_receipts,
    outcome_verdict,
    overlap_band_characterization,
    run_classify_why_analysis,
    run_full_analysis,
    run_identity_analysis,
)
from scripts.hrm_text_158_selector_value_analysis import main as orchestrator_main


def _support_batch(hash16: str = "hash_shared", row_ids: list[str] | None = None) -> dict[str, Any]:
    return {
        "batch_content_hash16": hash16,
        "row_ids": row_ids or ["row:a"],
    }


def _tensor_stats(
    *,
    applied_indices: list[int],
    q_sha: str,
    votes_sha: str,
    score_semantics: str,
    score_p50: float = 1.0,
    cap_jaccard: float | None = 0.0,
    candidate_count: int | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "applied_indices": applied_indices,
        "applied_flat_indices_hash16": f"h{len(applied_indices)}",
        "applied_count": len(applied_indices),
        "q_sha256_after": q_sha,
        "votes_sha256": votes_sha,
        "applied_selection_score_p50": score_p50,
        "applied_selection_score_p95": score_p50,
        "applied_selection_score_semantics": score_semantics,
    }
    if cap_jaccard is not None:
        payload["cap_window_jaccard_vs_prior_step"] = cap_jaccard
    if candidate_count is not None:
        payload["candidate_count"] = candidate_count
    return {DEFAULT_STATE_KEY: payload}


def _step_report(
    *,
    step: int,
    loss: float,
    applied_indices: list[int],
    q_sha: str,
    votes_sha: str,
    score_semantics: str,
    support_hash: str = "hash_shared",
    row_ids: list[str] | None = None,
    exact_accuracy: list[int] | None = None,
    cap_jaccard: float | None = 0.0,
    candidate_count: int | None = None,
    crossing_count: int | None = None,
    drift_tau: float | None = None,
    drift_sign_agreement: float | None = None,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "loss": loss,
        "loss_finite": True,
        "metrics": {
            "exact_accuracy": exact_accuracy or [0, 1],
            "loss": [loss, 1],
        },
        "support_batch": _support_batch(support_hash, row_ids),
        "step_result": {
            "tensor_stats": _tensor_stats(
                applied_indices=applied_indices,
                q_sha=q_sha,
                votes_sha=votes_sha,
                score_semantics=score_semantics,
                cap_jaccard=cap_jaccard,
                candidate_count=candidate_count,
            )
        },
    }
    if crossing_count is not None:
        report["grad_proxy_ingress"] = {
            "crossing_count_by_state_key": {DEFAULT_STATE_KEY: crossing_count}
        }
    if drift_tau is not None:
        report["proxy_oracle_drift"] = {
            "proxy_oracle_drift_step": step,
            "proxy_oracle_drift_tau": drift_tau,
            "proxy_oracle_drift_sign_agreement": drift_sign_agreement,
            "proxy_oracle_drift_top8_overlap": 0.0,
            "proxy_oracle_drift_gating": "sparse_anchor",
        }
    return report


def _receipt(
    *,
    arm: str,
    step_reports: dict[int, dict[str, Any]],
    batch_seed: int = 44,
    support_order_seed: int = 44,
) -> dict[str, Any]:
    return {
        "steps_completed": 10,
        "science_arm": arm,
        "batch": {"seed": batch_seed, "support_order_seed": support_order_seed},
        "step_reports": {str(step): report for step, report in step_reports.items()},
    }


def _primary_applied(step: int, arm: str) -> list[int]:
    base = step * 1000
    if arm == "on":
        return list(range(base, base + 4096))
    return list(range(base + 200, base + 4296))


def _build_primary_receipts(
    *,
    same_indices: bool = False,
    partial_overlap: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    on_reports: dict[int, dict[str, Any]] = {}
    off_reports: dict[int, dict[str, Any]] = {}
    for step in range(1, 11):
        if step <= 2:
            on_applied: list[int] = []
            off_applied = list(range(step * 10, step * 10 + 4096))
        elif same_indices:
            on_applied = list(range(step * 10, step * 10 + 4096))
            off_applied = list(on_applied)
        elif partial_overlap and step == 3:
            on_applied = list(range(0, 4096))
            off_applied = list(range(0, 2048)) + list(range(5000, 7048))
        elif partial_overlap:
            on_applied = list(range(step * 10000, step * 10000 + 4096))
            off_applied = list(range(step * 10000 + 5000, step * 10000 + 9096))
        else:
            on_applied = _primary_applied(step, "on")
            off_applied = _primary_applied(step, "off")
        q_on = "q_same" if same_indices else f"q_on_{step}"
        q_off = "q_same" if same_indices else f"q_off_{step}"
        on_reports[step] = _step_report(
            step=step,
            loss=1.0,
            applied_indices=on_applied,
            q_sha=q_on,
            votes_sha=f"v_on_{step}",
            score_semantics=ON_SCORE_SEMANTICS,
        )
        off_reports[step] = _step_report(
            step=step,
            loss=1.0,
            applied_indices=off_applied,
            q_sha=q_off,
            votes_sha=f"v_off_{step}",
            score_semantics=OFF_SCORE_SEMANTICS,
        )
    return _receipt(arm="on", step_reports=on_reports), _receipt(
        arm="off", step_reports=off_reports
    )


def _write_paired_run(tmp_path: Path, on: dict[str, Any], off: dict[str, Any]) -> Path:
    (tmp_path / "on").mkdir(parents=True)
    (tmp_path / "off").mkdir(parents=True)
    (tmp_path / "on" / "receipt.json").write_text(json.dumps(on), encoding="utf-8")
    (tmp_path / "off" / "receipt.json").write_text(json.dumps(off), encoding="utf-8")
    return tmp_path


def test_tight_all_primary_measurable_signal() -> None:
    on, off = _build_primary_receipts()
    summary = run_identity_analysis(on, off)
    assert summary["verdict"] == "selector_value_measurable_signal"
    assert summary["verdict_meta"]["partial_overlap_present"] is False


def test_committed_null_identity() -> None:
    on, off = _build_primary_receipts(same_indices=True)
    summary = run_identity_analysis(on, off)
    assert summary["verdict"] == "selector_value_committed_null"


def test_partial_overlap_unresolved() -> None:
    on, off = _build_primary_receipts(partial_overlap=True)
    on_steps = extract_cap_window_steps(on)
    off_steps = extract_cap_window_steps(off)
    tables = build_identity_tables(on_steps, off_steps)
    assert tables["verdict"] == "selector_value_different_unresolved"
    assert tables["verdict_meta"]["partial_overlap_present"] is True


def test_identity_verdict_explicit_q_rows_no_free_primary() -> None:
    primary = [
        {"step": 3, "cross_arm_jaccard": 0.5, "on_applied_count": 1, "off_applied_count": 1},
        {"step": 4, "cross_arm_jaccard": 0.25, "on_applied_count": 1, "off_applied_count": 1},
    ]
    q_rows = {3: {"q_match": False}, 4: {"q_match": False}}
    verdict, meta = identity_verdict(primary, q_rows)
    assert verdict == "selector_value_measurable_signal"
    assert meta["partial_overlap_present"] is False


def test_outcome_indistinguishable() -> None:
    on, off = _build_primary_receipts(same_indices=True)
    outcome = build_outcome_tables(on, off)
    assert outcome["verdict"] == "outcome_indistinguishable"


def test_outcome_trajectory_favors_off() -> None:
    on_reports: dict[int, dict[str, Any]] = {}
    off_reports: dict[int, dict[str, Any]] = {}
    for step in range(1, 11):
        delta = 0.0 if step == 1 else 1.0
        on_reports[step] = _step_report(
            step=step,
            loss=2.0 + delta,
            applied_indices=list(range(10)),
            q_sha=f"q_on_{step}",
            votes_sha=f"v_on_{step}",
            score_semantics=ON_SCORE_SEMANTICS,
        )
        off_reports[step] = _step_report(
            step=step,
            loss=2.0,
            applied_indices=list(range(10)),
            q_sha=f"q_off_{step}",
            votes_sha=f"v_off_{step}",
            score_semantics=OFF_SCORE_SEMANTICS,
        )
    outcome = build_outcome_tables(
        _receipt(arm="on", step_reports=on_reports),
        _receipt(arm="off", step_reports=off_reports),
    )
    assert outcome["verdict"] == "outcome_trajectory_favors_OFF"
    assert outcome["accuracy_tie_caveat"] is True


def test_outcome_trajectory_favors_on() -> None:
    on_reports: dict[int, dict[str, Any]] = {}
    off_reports: dict[int, dict[str, Any]] = {}
    for step in range(1, 11):
        delta = 0.0 if step == 1 else 1.0
        on_reports[step] = _step_report(
            step=step,
            loss=2.0,
            applied_indices=list(range(10)),
            q_sha=f"q_on_{step}",
            votes_sha=f"v_on_{step}",
            score_semantics=ON_SCORE_SEMANTICS,
        )
        off_reports[step] = _step_report(
            step=step,
            loss=2.0 + delta,
            applied_indices=list(range(10)),
            q_sha=f"q_off_{step}",
            votes_sha=f"v_off_{step}",
            score_semantics=OFF_SCORE_SEMANTICS,
        )
    outcome = build_outcome_tables(
        _receipt(arm="on", step_reports=on_reports),
        _receipt(arm="off", step_reports=off_reports),
    )
    assert outcome["verdict"] == "outcome_trajectory_favors_ON"


def test_exact_accuracy_tie_caveat_does_not_override_loss_direction() -> None:
    on_reports: dict[int, dict[str, Any]] = {}
    off_reports: dict[int, dict[str, Any]] = {}
    for step in range(1, 11):
        delta = 0.0 if step == 1 else 1.0
        on_reports[step] = _step_report(
            step=step,
            loss=3.0 + delta,
            applied_indices=list(range(10)),
            q_sha=f"q_on_{step}",
            votes_sha=f"v_on_{step}",
            score_semantics=ON_SCORE_SEMANTICS,
            exact_accuracy=[0, 1],
        )
        off_reports[step] = _step_report(
            step=step,
            loss=3.0,
            applied_indices=list(range(10)),
            q_sha=f"q_off_{step}",
            votes_sha=f"v_off_{step}",
            score_semantics=OFF_SCORE_SEMANTICS,
            exact_accuracy=[0, 1],
        )
    outcome = build_outcome_tables(
        _receipt(arm="on", step_reports=on_reports),
        _receipt(arm="off", step_reports=off_reports),
    )
    assert outcome["verdict"] == "outcome_trajectory_favors_OFF"
    assert outcome["accuracy_tie_caveat"] is True


def test_opposite_metric_direction_unresolved() -> None:
    on_reports: dict[int, dict[str, Any]] = {}
    off_reports: dict[int, dict[str, Any]] = {}
    for step in range(1, 11):
        on_loss = 5.0 if step >= 2 else 4.0
        off_loss = 4.0
        on_reports[step] = _step_report(
            step=step,
            loss=on_loss,
            applied_indices=list(range(10)),
            q_sha=f"q_on_{step}",
            votes_sha=f"v_on_{step}",
            score_semantics=ON_SCORE_SEMANTICS,
            exact_accuracy=[1, 1] if step >= 2 else [0, 1],
        )
        off_reports[step] = _step_report(
            step=step,
            loss=off_loss,
            applied_indices=list(range(10)),
            q_sha=f"q_off_{step}",
            votes_sha=f"v_off_{step}",
            score_semantics=OFF_SCORE_SEMANTICS,
            exact_accuracy=[0, 1],
        )
    outcome = build_outcome_tables(
        _receipt(arm="on", step_reports=on_reports),
        _receipt(arm="off", step_reports=off_reports),
    )
    assert outcome["verdict"] == "outcome_diverges_direction_unresolved"
    assert outcome["unresolved_reason"] == "opposite_metric_direction"


def test_seed_support_mismatch_invalid() -> None:
    on, off = _build_primary_receipts()
    off["batch"]["seed"] = 17
    guards = check_schedule_guards(on, off)
    assert guards.ok is False
    outcome = build_outcome_tables(on, off)
    assert outcome["verdict"] == "outcome_analysis_insufficient_surface"


def test_overlap_band_subordinate_only() -> None:
    on, off = _build_primary_receipts()
    on_steps = extract_cap_window_steps(on)
    off_steps = extract_cap_window_steps(off)
    rows = overlap_band_characterization(on_steps, off_steps)
    assert rows
    assert all(row["subordinate_non_headline"] is True for row in rows)


def test_orchestrator_writes_new_artifacts_only(tmp_path: Path) -> None:
    on, off = _build_primary_receipts()
    run_root = _write_paired_run(tmp_path, on, off)
    (run_root / "analysis").mkdir()
    legacy = run_root / "analysis" / "stage_c_summary.json"
    legacy.write_text('{"legacy": true}', encoding="utf-8")

    rc = orchestrator_main([str(run_root), "--mode", "full", "--repo-head", "deadbeef"])
    assert rc == 0
    assert json.loads(legacy.read_text(encoding="utf-8")) == {"legacy": True}
    assert (run_root / "analysis" / "stage_c_identity_summary.json").exists()
    assert (run_root / "analysis" / "stage_c_outcome_summary.json").exists()
    assert (run_root / "analysis" / "stage_c_outcome_memo.md").exists()
    assert (run_root / "analysis" / "run_manifest.json").exists()


def test_load_paired_receipts(tmp_path: Path) -> None:
    on, off = _build_primary_receipts()
    run_root = _write_paired_run(tmp_path, on, off)
    loaded_on, loaded_off = load_paired_receipts(run_root)
    assert loaded_on["science_arm"] == "on"
    assert loaded_off["science_arm"] == "off"


def test_run_full_analysis_shape() -> None:
    on, off = _build_primary_receipts()
    payload = run_full_analysis(on, off)
    assert "identity" in payload and "outcome" in payload


def _build_classify_h2_primary_receipts() -> tuple[dict[str, Any], dict[str, Any]]:
    on_reports: dict[int, dict[str, Any]] = {}
    off_reports: dict[int, dict[str, Any]] = {}
    for step in range(1, 11):
        if step <= 2:
            on_applied: list[int] = []
            off_applied = list(range(step * 10, step * 10 + 4096))
            on_loss = off_loss = 1.0
            crossing = None
            candidate = None
        else:
            on_applied = list(range(step * 1000, step * 1000 + 4096))
            off_applied = list(range(step * 1000 + 5000, step * 1000 + 9096))
            on_loss = 2.0 + (step - 2) * 0.5
            off_loss = 1.0
            crossing = step * 10
            candidate = 4096 + (step % 3) * 500
        on_reports[step] = _step_report(
            step=step,
            loss=on_loss,
            applied_indices=on_applied,
            q_sha=f"q_on_{step}",
            votes_sha=f"v_on_{step}",
            score_semantics=ON_SCORE_SEMANTICS,
            cap_jaccard=0.0 if step >= 3 else None,
            crossing_count=crossing,
        )
        off_reports[step] = _step_report(
            step=step,
            loss=off_loss,
            applied_indices=off_applied,
            q_sha=f"q_off_{step}",
            votes_sha=f"v_off_{step}",
            score_semantics=OFF_SCORE_SEMANTICS,
            candidate_count=candidate,
        )
    return _receipt(arm="on", step_reports=on_reports), _receipt(
        arm="off", step_reports=off_reports
    )


def _build_classify_h3_primary_receipts() -> tuple[dict[str, Any], dict[str, Any]]:
    on_reports: dict[int, dict[str, Any]] = {}
    off_reports: dict[int, dict[str, Any]] = {}
    for step in range(1, 11):
        if step <= 2:
            on_applied: list[int] = []
            off_applied = list(range(step * 10, step * 10 + 4096))
            on_loss = off_loss = 1.0
        else:
            on_applied = list(range(step * 1000, step * 1000 + 4096))
            off_applied = list(range(step * 1000 + 5000, step * 1000 + 9096))
            on_loss = 4.0 if step in (5, 6, 7) else 1.5
            off_loss = 1.0
        drift_tau = None
        drift_sign = None
        if step == 5:
            drift_tau = 0.081
            drift_sign = 0.5
        if step == 10:
            drift_tau = 0.886
            drift_sign = 0.5
        on_reports[step] = _step_report(
            step=step,
            loss=on_loss,
            applied_indices=on_applied,
            q_sha=f"q_on_{step}",
            votes_sha=f"v_on_{step}",
            score_semantics=ON_SCORE_SEMANTICS,
            cap_jaccard=0.0 if step >= 3 else None,
            drift_tau=drift_tau,
            drift_sign_agreement=drift_sign,
        )
        off_reports[step] = _step_report(
            step=step,
            loss=off_loss,
            applied_indices=off_applied,
            q_sha=f"q_off_{step}",
            votes_sha=f"v_off_{step}",
            score_semantics=OFF_SCORE_SEMANTICS,
        )
    return _receipt(arm="on", step_reports=on_reports), _receipt(
        arm="off", step_reports=off_reports
    )


def _build_corroboration_receipt() -> dict[str, Any]:
    reports: dict[int, dict[str, Any]] = {}
    for step, tau in ((5, 0.081), (10, 0.4), (15, 0.7), (20, 0.886)):
        reports[step] = _step_report(
            step=step,
            loss=1.0,
            applied_indices=list(range(10)),
            q_sha="q_corr",
            votes_sha="v_corr",
            score_semantics=ON_SCORE_SEMANTICS,
            drift_tau=tau,
            drift_sign_agreement=0.5,
        )
    return _receipt(arm="on", step_reports=reports)


def test_classify_why_h1_schedule_mismatch_rejected_flag() -> None:
    on, off = _build_primary_receipts()
    summary = run_classify_why_analysis(on, off)
    assert summary["support_schedule_mismatch_rejected"] is True
    assert summary["verdict"] in CLASSIFY_WHY_PRIMARY_VERDICTS


def test_classify_why_degenerate_jaccard_correlate_forbidden() -> None:
    on, off = _build_primary_receipts()
    summary = run_classify_why_analysis(on, off)
    h2 = summary["h2_cap_churn_geometry"]
    assert h2["degenerate_jaccard_correlate_forbidden"] is True
    assert "1 - cross_arm_jaccard" not in json.dumps(h2)
    assert "degenerate correlate" in " ".join(CLASSIFY_WHY_CANNOT_CLAIMS).lower()


def test_classify_why_sparse_tau_requires_corroboration_for_h3_primary() -> None:
    on, off = _build_classify_h3_primary_receipts()
    without = run_classify_why_analysis(on, off)
    assert without["verdict"] != "classify_proxy_mismatch_primary"
    with_corr = run_classify_why_analysis(
        on,
        off,
        corroboration_on=_build_corroboration_receipt(),
    )
    assert with_corr["verdict"] == "classify_proxy_mismatch_primary"
    assert with_corr["h3_proxy_mismatch"]["stage_c_sparse_tau"]["directionally_consistent"]


def test_classify_why_h2_primary_fixture() -> None:
    on, off = _build_classify_h2_primary_receipts()
    summary = run_classify_why_analysis(on, off)
    assert summary["verdict"] == "classify_cap_churn_primary"
    assert summary["h2_cap_churn_geometry"]["on_rotation_standing"] is True
    assert summary["routing_hint"] == "cap_churn_redesign_or_lane_verdict_class_call"


def test_classify_why_insufficient_surface_on_guard_failure() -> None:
    on, off = _build_primary_receipts()
    off["batch"]["support_order_seed"] = 17
    summary = run_classify_why_analysis(on, off)
    assert summary["verdict"] == "classify_insufficient_surface"
    assert summary["support_schedule_mismatch_rejected"] is False


def test_classify_why_mixed_unresolved_fixture() -> None:
    on, off = _build_classify_h3_primary_receipts()
    for step in range(3, 11):
        on["step_reports"][str(step)]["step_result"]["tensor_stats"][DEFAULT_STATE_KEY][
            "cap_window_jaccard_vs_prior_step"
        ] = 0.5
    summary = run_classify_why_analysis(on, off)
    assert summary["verdict"] == "classify_mixed_unresolved"


def _build_classify_h2_h3_tie_receipts() -> tuple[dict[str, Any], dict[str, Any]]:
    on_reports: dict[int, dict[str, Any]] = {}
    off_reports: dict[int, dict[str, Any]] = {}
    for step in range(1, 11):
        if step <= 2:
            on_applied: list[int] = []
            off_applied = list(range(step * 10, step * 10 + 4096))
            on_loss = off_loss = 1.0
            crossing = None
            candidate = None
        else:
            base = step * 1000
            shared = step * 50
            on_applied = list(range(base, base + 4096))
            off_applied = list(range(base, base + shared)) + list(
                range(base + 5000, base + 9096 - shared)
            )
            on_loss = 6.0 if step in (5, 6, 7) else 1.0 + step * 0.2
            off_loss = 1.0
            crossing = step * 10
            candidate = 4096 + (step % 3) * 500
        drift_tau = 0.081 if step == 5 else (0.886 if step == 10 else None)
        drift_sign = 0.5 if step in (5, 10) else None
        on_reports[step] = _step_report(
            step=step,
            loss=on_loss,
            applied_indices=on_applied,
            q_sha=f"q_on_{step}",
            votes_sha=f"v_on_{step}",
            score_semantics=ON_SCORE_SEMANTICS,
            cap_jaccard=0.0 if step >= 3 else None,
            crossing_count=crossing,
            drift_tau=drift_tau,
            drift_sign_agreement=drift_sign,
        )
        off_reports[step] = _step_report(
            step=step,
            loss=off_loss,
            applied_indices=off_applied,
            q_sha=f"q_off_{step}",
            votes_sha=f"v_off_{step}",
            score_semantics=OFF_SCORE_SEMANTICS,
            candidate_count=candidate,
        )
    return _receipt(arm="on", step_reports=on_reports), _receipt(
        arm="off", step_reports=off_reports
    )


def test_classify_why_h2_h3_tie_returns_mixed_not_cap_churn() -> None:
    on, off = _build_classify_h2_h3_tie_receipts()
    summary = run_classify_why_analysis(
        on,
        off,
        corroboration_on=_build_corroboration_receipt(),
    )
    assert summary["scores"]["h2"] >= 3
    assert summary["scores"]["h3"] >= 3
    assert summary["scores"]["h2"] == summary["scores"]["h3"]
    assert summary["verdict"] == "classify_mixed_unresolved"
    assert summary["routing_hint"] == "confirmatory_seed_if_decision_critical_else_hold"


def test_descriptive_step_delta_association_sparse_tau_handling() -> None:
    assoc = descriptive_step_delta_association([0.1, None, 0.3], [1.0, 2.0, 3.0])
    assert assoc["insufficient_pairs"] is True
    assoc_ok = descriptive_step_delta_association([1, 2, 3, 4], [1.0, 1.5, 2.0, 2.5])
    assert assoc_ok["alignment_fraction"] == 1.0
    assert assoc_ok["descriptive_only"] is True


def test_orchestrator_classify_why_writes_new_artifacts_only(tmp_path: Path) -> None:
    on, off = _build_classify_h2_primary_receipts()
    run_root = _write_paired_run(tmp_path, on, off)
    analysis_dir = run_root / "analysis"
    analysis_dir.mkdir()
    legacy_manifest = analysis_dir / "run_manifest.json"
    legacy_manifest.write_text('{"legacy": true}', encoding="utf-8")
    corroboration = _build_corroboration_receipt()
    corr_path = tmp_path / "attempt6_on_receipt.json"
    corr_path.write_text(json.dumps(corroboration), encoding="utf-8")

    rc = orchestrator_main(
        [
            str(run_root),
            "--mode",
            "classify_why",
            "--corroboration-on-receipt",
            str(corr_path),
            "--repo-head",
            "deadbeef",
        ]
    )
    assert rc == 0
    summary = json.loads(
        (run_root / "analysis" / "stage_c_classify_why_summary.json").read_text(
            encoding="utf-8"
        )
    )
    manifest = json.loads(
        (run_root / "analysis" / "stage_c_classify_why_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["verdict"] == "classify_cap_churn_primary"
    assert (run_root / "analysis" / "stage_c_classify_why_memo.md").exists()
    assert manifest["input_receipt_sha256"]["corroboration_on"] is not None
    assert json.loads(legacy_manifest.read_text(encoding="utf-8")) == {"legacy": True}
    assert not (run_root / "analysis" / "stage_c_identity_summary.json").exists()


def test_outcome_verdict_metric_mismatch_with_indistinguishable_loss() -> None:
    trajectory_rows = [
        {
            "step": 2,
            "delta_on_minus_off": 0.0,
            "exact_accuracy_on": [0, 1],
            "exact_accuracy_off": [1, 1],
            "exact_accuracy_match": False,
        }
    ]
    guards = check_schedule_guards(*_build_primary_receipts())
    payload = outcome_verdict(
        guards=guards,
        trajectory_rows=trajectory_rows,
        mean_delta=0.0,
        cumulative_delta=0.0,
    )
    assert payload["verdict"] == "outcome_diverges_direction_unresolved"
    assert payload["unresolved_reason"] == "metrics_differ_with_indistinguishable_loss"
