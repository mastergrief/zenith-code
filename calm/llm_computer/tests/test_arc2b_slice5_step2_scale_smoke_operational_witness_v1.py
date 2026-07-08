"""CPU-static tests for scale_smoke operational witness."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.hrm_text_158_arc2b_slice5_step2_scale_smoke_operational_witness import (
    build_operational_witness,
)

SMOKE_STEPS = 5
CLEARED_LAST_ACTIVE_PHASE = {
    "schema": "hrm_text_158_c2p2_phase_telemetry/v1",
    "event": "active_phase_guard",
    "guard_event": "cleared",
    "phase_status": "completed",
    "phase": "step_update",
    "liveness_failure": False,
}


def _write_success_fixture(run_root: Path) -> None:
    scratch = run_root / "d_recompute_window_diagnostic"
    scratch.mkdir(parents=True, exist_ok=True)
    (scratch / "receipt.json").write_text(
        json.dumps(
            {
                "steps_completed": SMOKE_STEPS,
                "bounded_delta_global_summary": {
                    "sparse_cap_apply_parallel_mode": "serial_cpu",
                },
            }
        ),
        encoding="utf-8",
    )
    (scratch / "last_active_phase.json").write_text(
        json.dumps(CLEARED_LAST_ACTIVE_PHASE),
        encoding="utf-8",
    )
    (scratch / "live_carrier_snapshot.jsonl").write_text(
        '{"live_carrier_bytes_exact": true}\n' * SMOKE_STEPS,
        encoding="utf-8",
    )
    rows = []
    for step in range(1, SMOKE_STEPS + 1):
        rows.append(
            {
                "step": step,
                "replay_constants": {"decay_numerator": 1, "decay_denominator": 2},
            }
        )
    (scratch / "recompute_window_log.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_operational_witness_passes_success_fixture(tmp_path: Path) -> None:
    _write_success_fixture(tmp_path)
    receipt = build_operational_witness(tmp_path, smoke_steps=SMOKE_STEPS)
    assert receipt["pass"] is True
    assert receipt["failures"] == []


def test_operational_witness_fails_missing_smoke_last_active_phase(tmp_path: Path) -> None:
    _write_success_fixture(tmp_path)
    (tmp_path / "d_recompute_window_diagnostic" / "last_active_phase.json").unlink()
    receipt = build_operational_witness(tmp_path, smoke_steps=SMOKE_STEPS)
    assert receipt["pass"] is False
    assert "smoke_last_active_phase_missing" in receipt["failures"]


def test_operational_witness_fails_failure_class_liveness_failure(tmp_path: Path) -> None:
    _write_success_fixture(tmp_path)
    phase_path = tmp_path / "d_recompute_window_diagnostic" / "last_active_phase.json"
    phase = json.loads(phase_path.read_text(encoding="utf-8"))
    phase["failure_class"] = "LIVENESS_FAILURE"
    phase_path.write_text(json.dumps(phase), encoding="utf-8")
    receipt = build_operational_witness(tmp_path, smoke_steps=SMOKE_STEPS)
    assert receipt["pass"] is False
    assert "smoke_failure_class_liveness_failure" in receipt["failures"]


def test_operational_witness_fails_warmup_exhausted_retries(tmp_path: Path) -> None:
    _write_success_fixture(tmp_path)
    prelaunch = tmp_path / "prelaunch"
    prelaunch.mkdir(parents=True, exist_ok=True)
    (prelaunch / "calibration_warmup_retry_witness.json").write_text(
        json.dumps(
            {
                "final_rc": 1,
                "final_reason": "liveness_failure_exhausted_retries",
            }
        ),
        encoding="utf-8",
    )
    receipt = build_operational_witness(tmp_path, smoke_steps=SMOKE_STEPS)
    assert receipt["pass"] is False
    assert "warmup_retry_liveness_failure_exhausted_retries" in receipt["failures"]


def test_operational_witness_fails_warmup_final_rc_missing(tmp_path: Path) -> None:
    _write_success_fixture(tmp_path)
    prelaunch = tmp_path / "prelaunch"
    prelaunch.mkdir(parents=True, exist_ok=True)
    (prelaunch / "calibration_warmup_retry_witness.json").write_text(
        json.dumps({"final_reason": "success"}),
        encoding="utf-8",
    )
    receipt = build_operational_witness(tmp_path, smoke_steps=SMOKE_STEPS)
    assert receipt["pass"] is False
    assert "warmup_retry_final_rc_missing" in receipt["failures"]


def test_operational_witness_fails_warmup_final_reason_missing(tmp_path: Path) -> None:
    _write_success_fixture(tmp_path)
    prelaunch = tmp_path / "prelaunch"
    prelaunch.mkdir(parents=True, exist_ok=True)
    (prelaunch / "calibration_warmup_retry_witness.json").write_text(
        json.dumps({"final_rc": 0}),
        encoding="utf-8",
    )
    receipt = build_operational_witness(tmp_path, smoke_steps=SMOKE_STEPS)
    assert receipt["pass"] is False
    assert "warmup_retry_final_reason_missing" in receipt["failures"]


def test_operational_witness_fails_stale_boolean_only_liveness_failure(tmp_path: Path) -> None:
    _write_success_fixture(tmp_path)
    phase_path = tmp_path / "d_recompute_window_diagnostic" / "last_active_phase.json"
    phase_path.write_text(json.dumps({"liveness_failure": False}), encoding="utf-8")
    receipt = build_operational_witness(tmp_path, smoke_steps=SMOKE_STEPS)
    assert receipt["pass"] is False
    assert "smoke_stale_boolean_only_liveness_failure" in receipt["failures"]


def test_operational_witness_fails_steps_completed_not_five(tmp_path: Path) -> None:
    _write_success_fixture(tmp_path)
    receipt_path = tmp_path / "d_recompute_window_diagnostic" / "receipt.json"
    body = json.loads(receipt_path.read_text(encoding="utf-8"))
    body["steps_completed"] = 4
    receipt_path.write_text(json.dumps(body), encoding="utf-8")
    receipt = build_operational_witness(tmp_path, smoke_steps=SMOKE_STEPS)
    assert receipt["pass"] is False
    assert "steps_completed_4_expected_5" in receipt["failures"]


def test_operational_witness_fails_parallel_mode_not_serial_cpu(tmp_path: Path) -> None:
    _write_success_fixture(tmp_path)
    receipt_path = tmp_path / "d_recompute_window_diagnostic" / "receipt.json"
    body = json.loads(receipt_path.read_text(encoding="utf-8"))
    body["bounded_delta_global_summary"]["sparse_cap_apply_parallel_mode"] = "parallel_cpu"
    receipt_path.write_text(json.dumps(body), encoding="utf-8")
    receipt = build_operational_witness(tmp_path, smoke_steps=SMOKE_STEPS)
    assert receipt["pass"] is False
    assert "sparse_cap_apply_parallel_mode_parallel_cpu_expected_serial_cpu" in receipt["failures"]


def test_operational_witness_fails_doubled_live_carrier_path(tmp_path: Path) -> None:
    _write_success_fixture(tmp_path)
    scratch = tmp_path / "d_recompute_window_diagnostic"
    doubled = scratch / "d_recompute_window_diagnostic"
    doubled.mkdir(parents=True, exist_ok=True)
    (doubled / "live_carrier_snapshot.jsonl").write_text("{}", encoding="utf-8")
    receipt = build_operational_witness(tmp_path, smoke_steps=SMOKE_STEPS)
    assert receipt["pass"] is False
    assert "doubled_live_carrier_snapshot_path_present" in receipt["failures"]


def test_operational_witness_fails_recompute_log_row_count_not_five(tmp_path: Path) -> None:
    _write_success_fixture(tmp_path)
    log_path = tmp_path / "d_recompute_window_diagnostic" / "recompute_window_log.jsonl"
    log_path.write_text(
        json.dumps({"step": 1, "replay_constants": {"decay_numerator": 1, "decay_denominator": 2}})
        + "\n",
        encoding="utf-8",
    )
    receipt = build_operational_witness(tmp_path, smoke_steps=SMOKE_STEPS)
    assert receipt["pass"] is False
    assert "recompute_window_log_rows_1_expected_5" in receipt["failures"]


def test_operational_witness_fails_recompute_log_decay_not_one_half(tmp_path: Path) -> None:
    _write_success_fixture(tmp_path)
    log_path = tmp_path / "d_recompute_window_diagnostic" / "recompute_window_log.jsonl"
    rows = []
    for step in range(1, SMOKE_STEPS + 1):
        rows.append(
            {
                "step": step,
                "replay_constants": {"decay_numerator": 1, "decay_denominator": 1},
            }
        )
    log_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    receipt = build_operational_witness(tmp_path, smoke_steps=SMOKE_STEPS)
    assert receipt["pass"] is False
    assert any(f.startswith("recompute_window_log_row_") for f in receipt["failures"])
