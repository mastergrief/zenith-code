"""CPU tests for paired selector-value identity/outcome receipt analysis."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from calm.hrm_text_158.native_full_stack.selector_value_analysis import (
    DEFAULT_STATE_KEY,
    OFF_SCORE_SEMANTICS,
    ON_SCORE_SEMANTICS,
    build_identity_tables,
    build_outcome_tables,
    check_schedule_guards,
    extract_cap_window_steps,
    identity_verdict,
    load_paired_receipts,
    outcome_verdict,
    overlap_band_characterization,
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
) -> dict[str, Any]:
    return {
        DEFAULT_STATE_KEY: {
            "applied_indices": applied_indices,
            "applied_flat_indices_hash16": f"h{len(applied_indices)}",
            "applied_count": len(applied_indices),
            "q_sha256_after": q_sha,
            "votes_sha256": votes_sha,
            "applied_selection_score_p50": score_p50,
            "applied_selection_score_p95": score_p50,
            "applied_selection_score_semantics": score_semantics,
            "cap_window_jaccard_vs_prior_step": 0.0,
        }
    }


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
) -> dict[str, Any]:
    return {
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
            )
        },
    }


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
