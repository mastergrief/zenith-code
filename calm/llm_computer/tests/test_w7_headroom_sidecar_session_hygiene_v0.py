from __future__ import annotations

from pathlib import Path

from calm.hrm_text_158.native_full_stack.s3bb_headroom_telemetry import (
    WARMUP_STEPS,
    append_headroom_wiring_sidecar_chunk,
    diagnose_sidecar_coverage,
    initialize_headroom_wiring_sidecar_for_probe_session,
)
from calm.hrm_text_158.native_full_stack.w7_dense_acc_in_vivo_confirmation import (
    CLASSIFIER_RUN_HEALTH_FAIL,
    CONFIRMATION_ENVELOPE_CANONICAL_T10_PREREG_V24,
    CPU_PHASE0_STRUCTURAL_FLOOR_WIDTH,
    classify_w7_in_vivo_dual_arm,
    derive_w7_parity_inputs,
    resolve_confirmation_envelope,
)
from scripts.hrm_text_158_bounded_delta_acquisition_probe import (
    HEADROOM_WIRING_SIDECAR_FILENAME,
    RECEIPT_EMIT_PROFILE_SLIM,
)


def _module_keys(module_count: int) -> list[str]:
    return [f"module_{index:02d}.proj" for index in range(module_count)]


def _write_sidecar_session(
    sidecar_path: Path,
    *,
    steps: range,
    module_count: int,
    acc: list[int] | None = None,
) -> None:
    acc = acc or [5, -9, 10]
    q = [0, 1, -1]
    for step in steps:
        for state_key in _module_keys(module_count):
            append_headroom_wiring_sidecar_chunk(
                sidecar_path,
                step=int(step),
                state_key=state_key,
                accumulator_lanes=acc,
                q_lanes=q,
            )


def resolve_probe_slim_headroom_sidecar_path(scratch_root: Path) -> Path:
    """Mirror probe slim sidecar path selection at session start."""

    return scratch_root / HEADROOM_WIRING_SIDECAR_FILENAME


def test_probe_start_init_truncates_stale_sidecar(tmp_path: Path) -> None:
    sidecar_path = resolve_probe_slim_headroom_sidecar_path(tmp_path)
    _write_sidecar_session(sidecar_path, steps=range(3, 7), module_count=32)

    assert sidecar_path.is_file()
    assert sidecar_path.stat().st_size > 0

    initialize_headroom_wiring_sidecar_for_probe_session(sidecar_path)

    assert not sidecar_path.exists()


def test_retry_session_after_probe_init_has_zero_duplicate_keys(tmp_path: Path) -> None:
    sidecar_path = resolve_probe_slim_headroom_sidecar_path(tmp_path)

    # Session 1 partial (mirrors 2189e72006 oracle retry: steps 3-6 only).
    _write_sidecar_session(sidecar_path, steps=range(3, 7), module_count=32)

    # New probe invocation: probe-start init must truncate stale append-only rows.
    initialize_headroom_wiring_sidecar_for_probe_session(sidecar_path)

    # Session 2 full run (steps 3-10, 32 modules).
    _write_sidecar_session(sidecar_path, steps=range(3, 11), module_count=32)

    coverage = diagnose_sidecar_coverage(sidecar_path, sidecar_path)
    expected_rows = (10 - WARMUP_STEPS) * 32
    assert coverage["oracle_row_count"] == expected_rows
    assert coverage["treatment_row_count"] == expected_rows
    assert coverage["oracle_duplicate_key_count"] == 0
    assert coverage["treatment_duplicate_key_count"] == 0
    assert coverage["structural_fail"] is False


def test_append_without_probe_init_reproduces_duplicate_keys(tmp_path: Path) -> None:
    sidecar_path = resolve_probe_slim_headroom_sidecar_path(tmp_path)

    _write_sidecar_session(sidecar_path, steps=range(3, 7), module_count=32)
    _write_sidecar_session(sidecar_path, steps=range(3, 11), module_count=32)

    coverage = diagnose_sidecar_coverage(sidecar_path, sidecar_path)
    assert coverage["oracle_row_count"] == 384
    assert coverage["oracle_duplicate_key_count"] == 128
    assert coverage["structural_fail"] is True
    assert "oracle_duplicate_keys" in coverage["structural_reasons"]


def test_golden_256_shape_invariant_passes_diagnose_sidecar_coverage(tmp_path: Path) -> None:
    oracle_path = tmp_path / "oracle.jsonl"
    treatment_path = tmp_path / "treatment.jsonl"
    initialize_headroom_wiring_sidecar_for_probe_session(oracle_path)
    initialize_headroom_wiring_sidecar_for_probe_session(treatment_path)

    _write_sidecar_session(oracle_path, steps=range(3, 11), module_count=32)
    _write_sidecar_session(treatment_path, steps=range(3, 11), module_count=32)

    coverage = diagnose_sidecar_coverage(oracle_path, treatment_path)
    assert coverage["oracle_row_count"] == 256
    assert coverage["treatment_row_count"] == 256
    assert coverage["oracle_duplicate_key_count"] == 0
    assert coverage["treatment_duplicate_key_count"] == 0
    assert coverage["structural_fail"] is False


def test_contaminated_sidecar_still_classifies_structural_run_health_fail(tmp_path: Path) -> None:
    oracle_path = tmp_path / "oracle_dup.jsonl"
    treatment_path = tmp_path / "treatment_clean.jsonl"
    initialize_headroom_wiring_sidecar_for_probe_session(treatment_path)
    _write_sidecar_session(treatment_path, steps=range(3, 11), module_count=32)

    _write_sidecar_session(oracle_path, steps=range(3, 7), module_count=32)
    _write_sidecar_session(oracle_path, steps=range(3, 11), module_count=32)

    coverage = diagnose_sidecar_coverage(oracle_path, treatment_path)
    assert coverage["structural_fail"] is True
    assert "oracle_duplicate_keys" in coverage["structural_reasons"]

    parity_inputs = derive_w7_parity_inputs(
        "HARNESS_OR_LIVENESS_FAIL",
        {},
        coverage,
    )
    assert parity_inputs["structural_fail"] is True
    assert parity_inputs["parity_break"] is False

    envelope = resolve_confirmation_envelope(CONFIRMATION_ENVELOPE_CANONICAL_T10_PREREG_V24)
    result = classify_w7_in_vivo_dual_arm(
        oracle_receipt={"step_reports": {}},
        treatment_receipt={"step_reports": {}},
        envelope=envelope,
        structural_fail=True,
        structural_reason=str(parity_inputs["structural_reason"]),
        confirmed_vote_acc_floor_width=CPU_PHASE0_STRUCTURAL_FLOOR_WIDTH,
    )
    assert result["primary_classifier"] == CLASSIFIER_RUN_HEALTH_FAIL


def test_slim_emit_profile_constant_matches_probe(tmp_path: Path) -> None:
    assert RECEIPT_EMIT_PROFILE_SLIM == "s3bb_headroom_diagnostic_slim"
    sidecar_path = resolve_probe_slim_headroom_sidecar_path(tmp_path)
    assert sidecar_path.name == HEADROOM_WIRING_SIDECAR_FILENAME
