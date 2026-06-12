"""CPU tests for offline step_update cost attribution (F1 Slice A)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from calm.hrm_text_158.native_full_stack.grad_proxy_audit import DRIFT_AUDIT_STEP_INTERVAL
from calm.hrm_text_158.native_full_stack.step_update_cost_attribution import (
    DEFAULT_THRESHOLD_S,
    DRIFT_AUDIT_PHASE,
    PHASE_CLASS_DRIFT_AUDIT,
    PHASE_CLASS_SELECTOR_ALLOWLIST,
    PHASE_CLASS_UNKNOWN,
    SELECTOR_ALLOWLIST_PHASES,
    ThresholdConfig,
    analyze_run_log,
    build_derivation_receipt,
    build_step_update_attribution,
    classify_nested_phase,
    parse_run_log_events,
)


def _event(
    *,
    phase: str,
    event: str,
    step: int | None = None,
    duration_seconds: float | None = None,
) -> dict:
    payload = {
        "schema": "hrm_text_158_c2p2_phase_telemetry/v0",
        "phase": phase,
        "event": event,
    }
    if step is not None:
        payload["step"] = step
    if duration_seconds is not None:
        payload["duration_seconds"] = duration_seconds
    return payload


def _write_run_log(path: Path, events: list[dict]) -> None:
    lines = [json.dumps(event) for event in events]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_classify_nested_phase_allowlist_and_drift() -> None:
    assert classify_nested_phase("two_tier_grad_proxy_ingress") == PHASE_CLASS_SELECTOR_ALLOWLIST
    assert classify_nested_phase(DRIFT_AUDIT_PHASE) == PHASE_CLASS_DRIFT_AUDIT
    assert classify_nested_phase("mystery_phase") == PHASE_CLASS_UNKNOWN


def test_build_attribution_decomposes_step_with_drift_audit() -> None:
    events = [
        _event(phase="step_update", event="start", step=10),
        _event(
            phase="two_tier_grad_proxy_ingress",
            event="end",
            step=10,
            duration_seconds=9.213,
        ),
        _event(
            phase=DRIFT_AUDIT_PHASE,
            event="end",
            step=10,
            duration_seconds=9.164,
        ),
        _event(
            phase="step_update",
            event="end",
            step=10,
            duration_seconds=76.076,
        ),
    ]
    artifact = build_step_update_attribution(events, threshold=ThresholdConfig(threshold_s=95.0))
    row = artifact["per_step"][0]
    assert row["step"] == 10
    assert row["selector_allowlist_s"] == pytest.approx(9.213)
    assert row["drift_audit_s"] == pytest.approx(9.164)
    assert row["selector_overhead_s"] == pytest.approx(18.377)
    assert row["unattributed_apply_residual_s"] == pytest.approx(57.699)
    assert artifact["nesting_complete"] is True
    assert artifact["threshold_s"] == 95.0
    assert artifact["drift_audit"]["step_interval"] == DRIFT_AUDIT_STEP_INTERVAL
    assert artifact["discriminators"]["branch_step_update_liveness_fail"] is False
    assert artifact["discriminators"]["step_update_selector_overhead_dominant"] is False


def test_unknown_nested_phase_sets_nesting_complete_false() -> None:
    events = [
        _event(phase="step_update", event="start", step=1),
        _event(phase="unexpected_nested", event="end", step=1, duration_seconds=1.0),
        _event(phase="step_update", event="end", step=1, duration_seconds=5.0),
    ]
    artifact = build_step_update_attribution(events)
    assert artifact["nesting_complete"] is False
    assert "unexpected_nested" in artifact["unknown_nested_phases"]


def test_threshold_parameterization_and_liveness_fail() -> None:
    events = [
        _event(phase="step_update", event="start", step=1),
        _event(phase="step_update", event="end", step=1, duration_seconds=100.0),
    ]
    artifact = build_step_update_attribution(
        events,
        threshold=ThresholdConfig(threshold_s=95.0, lineage_packet_msg_id="1781216817861"),
    )
    assert artifact["discriminators"]["branch_step_update_liveness_fail"] is True
    assert artifact["discriminators"]["step_update_headroom_pct"] == pytest.approx(
        ((95.0 - 100.0) / 95.0) * 100.0
    )


def test_parse_run_log_handles_prog_prefix(tmp_path: Path) -> None:
    run_log = tmp_path / "run.log"
    run_log.write_text(
        '[PROG] {"schema":"hrm_text_158_c2p2_phase_telemetry/v0","phase":"step_update","event":"start","step":1}\n'
        '[PROG] {"schema":"hrm_text_158_c2p2_phase_telemetry/v0","phase":"step_update","event":"end","step":1,"duration_seconds":57.015}\n',
        encoding="utf-8",
    )
    events = parse_run_log_events(run_log)
    assert len(events) == 2
    artifact = analyze_run_log(run_log)
    assert artifact["aggregates"]["step_update_total_s"]["max"] == pytest.approx(57.015)


def test_attempt6_fixture_step1_zero_crossing_flat_residual(tmp_path: Path) -> None:
    run_log = tmp_path / "run.log"
    _write_run_log(
        run_log,
        [
            _event(phase="step_update", event="start", step=1),
            _event(
                phase="two_tier_grad_proxy_ingress",
                event="end",
                step=1,
                duration_seconds=0.019899,
            ),
            _event(phase="step_update", event="end", step=1, duration_seconds=57.015159),
            _event(phase="step_update", event="start", step=20),
            _event(
                phase="two_tier_grad_proxy_ingress",
                event="end",
                step=20,
                duration_seconds=13.724,
            ),
            _event(
                phase=DRIFT_AUDIT_PHASE,
                event="end",
                step=20,
                duration_seconds=13.669,
            ),
            _event(phase="step_update", event="end", step=20, duration_seconds=86.445),
        ],
    )
    artifact = analyze_run_log(run_log)
    step1 = artifact["per_step"][0]
    step20 = artifact["per_step"][1]
    assert step1["unattributed_apply_residual_s"] == pytest.approx(56.995, rel=1e-3)
    assert step20["selector_overhead_s"] == pytest.approx(27.393, rel=1e-3)
    assert artifact["aggregates"]["step_update_total_s"]["max"] == pytest.approx(86.445)
    assert artifact["discriminators"]["step_update_headroom_pct"] == pytest.approx(
        ((DEFAULT_THRESHOLD_S - 86.445) / DEFAULT_THRESHOLD_S) * 100.0,
        rel=1e-3,
    )
    assert all(
        phase in SELECTOR_ALLOWLIST_PHASES or phase == DRIFT_AUDIT_PHASE
        for phase in artifact["nested_phase_classification"]
    )


def test_derivation_receipt_includes_sha_and_cross_run_summary(tmp_path: Path) -> None:
    on_log = tmp_path / "on" / "run.log"
    off_log = tmp_path / "off" / "run.log"
    on_log.parent.mkdir(parents=True)
    off_log.parent.mkdir(parents=True)
    _write_run_log(
        on_log,
        [
            _event(phase="step_update", event="start", step=1),
            _event(phase="step_update", event="end", step=1, duration_seconds=58.0),
        ],
    )
    _write_run_log(
        off_log,
        [
            _event(phase="step_update", event="start", step=1),
            _event(phase="step_update", event="end", step=1, duration_seconds=0.3),
        ],
    )
    receipt = build_derivation_receipt(
        [
            ("attempt6_on", on_log),
            ("attempt6_off", off_log),
        ],
        threshold=ThresholdConfig(threshold_s=95.0),
    )
    assert receipt["nesting_complete_all_runs"] is True
    assert str(on_log.resolve()) in receipt["source_sha256"]
    assert receipt["cross_run_summary"]["on_off_max_ratio"] == pytest.approx(58.0 / 0.3)
    assert receipt["runs"][0]["attribution"]["interpretation"]["enabled_path_cost_ratio_estimate"] is not None
