from __future__ import annotations

from pathlib import Path

import torch

from calm.hrm_text_158.native_full_stack.bounded_delta_learner import make_bounded_tensor_state
from calm.hrm_text_158.native_full_stack.s3bb_headroom_telemetry import (
    HEADROOM_WIRING_SIDECAR_FILENAME,
    REQUIRED_HEADROOM_TELEMETRY_FIELDS,
    S3BB_W6_HEADROOM_DIAGNOSTIC_PHASE,
    SNAPSHOT_MODE_AGGREGATE_ONLY,
    W7_DENSE_ACC_IN_VIVO_CONFIRMATION_PHASE,
    W8_DENSE_ACC_IN_VIVO_CONFIRMATION_PHASE,
    WARMUP_STEPS,
    attach_s3bb_headroom_telemetry_to_step_report,
    validate_headroom_telemetry_block,
)


def _tensor_state() -> object:
    return make_bounded_tensor_state(
        "tiny.proj",
        torch.tensor([[0, 1, -1]], dtype=torch.int8),
        1.0,
        torch.tensor([[5, -9, 21]], dtype=torch.int16),
    )


def _attach_with_sidecar(
    *,
    phase: str,
    sidecar_path: Path,
    step: int,
) -> dict:
    report: dict = {"step_id": str(step)}
    attach_s3bb_headroom_telemetry_to_step_report(
        report,
        phase=phase,
        post_update_states={"tiny.proj": _tensor_state()},
        snapshot_mode=SNAPSHOT_MODE_AGGREGATE_ONLY,
        headroom_wiring_sidecar_path=sidecar_path,
        step=step,
    )
    return report


def test_w8_phase_writes_nonempty_sidecar_after_warmup(tmp_path: Path) -> None:
    sidecar_path = tmp_path / HEADROOM_WIRING_SIDECAR_FILENAME
    report = _attach_with_sidecar(
        phase=W8_DENSE_ACC_IN_VIVO_CONFIRMATION_PHASE,
        sidecar_path=sidecar_path,
        step=WARMUP_STEPS + 1,
    )
    telemetry = report["headroom_telemetry"]
    for field in REQUIRED_HEADROOM_TELEMETRY_FIELDS:
        assert field in telemetry
    validate_headroom_telemetry_block(telemetry)
    assert sidecar_path.is_file()
    assert sidecar_path.read_text(encoding="utf-8").strip() != ""


def test_w7_phase_regression_unchanged(tmp_path: Path) -> None:
    sidecar_path = tmp_path / "w7_sidecar.jsonl"
    report = _attach_with_sidecar(
        phase=W7_DENSE_ACC_IN_VIVO_CONFIRMATION_PHASE,
        sidecar_path=sidecar_path,
        step=WARMUP_STEPS + 1,
    )
    assert "headroom_telemetry" in report
    validate_headroom_telemetry_block(report["headroom_telemetry"])
    assert sidecar_path.is_file()
    assert sidecar_path.read_text(encoding="utf-8").strip() != ""


def test_w6_phase_still_emits_headroom_telemetry(tmp_path: Path) -> None:
    sidecar_path = tmp_path / "w6_sidecar.jsonl"
    report = _attach_with_sidecar(
        phase=S3BB_W6_HEADROOM_DIAGNOSTIC_PHASE,
        sidecar_path=sidecar_path,
        step=WARMUP_STEPS + 1,
    )
    assert "headroom_telemetry" in report
    validate_headroom_telemetry_block(report["headroom_telemetry"])
    assert sidecar_path.is_file()
    assert sidecar_path.read_text(encoding="utf-8").strip() != ""


def test_unrelated_phase_does_not_emit_headroom_telemetry(tmp_path: Path) -> None:
    sidecar_path = tmp_path / "blocked_sidecar.jsonl"
    report: dict = {"step_id": "3"}
    attach_s3bb_headroom_telemetry_to_step_report(
        report,
        phase="c2p1-real-model-smoke",
        post_update_states={"tiny.proj": _tensor_state()},
        snapshot_mode=SNAPSHOT_MODE_AGGREGATE_ONLY,
        headroom_wiring_sidecar_path=sidecar_path,
        step=WARMUP_STEPS + 1,
    )
    assert "headroom_telemetry" not in report
    assert not sidecar_path.exists()
