"""CPU/static tests for HRM recurrent-state buffer accounting."""
from __future__ import annotations

import pytest
import torch

from calm.hrm_text_158.native_full_stack import (
    MODE_LOSSLESS_RECURRENT_STATE_OFFLOAD,
    MODE_LOSSLESS_RECURRENT_STATE_REMATERIALIZATION,
    MODE_LOSSY_RECURRENT_STATE_COMPRESSION,
    MODE_RECURRENT_STATE_OFF,
    RECURRENT_STATE_BUFFER_SCHEMA_VERSION,
    REQUIRED_RECURRENT_STATE_MEASUREMENT_FIELDS,
    TIER1_LOSSLESS_RECURRENT_STATE_RELIEF_DEFERRED,
    TIER2_LOSSY_RECURRENT_STATE_COMPRESSION_DEFERRED,
    RecurrentStateBufferSpec,
    estimate_recurrent_state_buffers,
    recurrent_state_dtype_nbytes,
    recurrent_state_schedule_summary,
    validate_recurrent_state_buffer_measurement,
    validate_recurrent_state_relief_mode,
)


def _complete_receipt(total_bytes: int, z_h_bytes: int, z_l_bytes: int) -> dict:
    return {
        "peak_allocated_bytes": total_bytes + 1024,
        "peak_reserved_bytes": total_bytes + 4096,
        "wall_clock_per_step_seconds": 0.25,
        "max_safe_batch_size": 8,
        "effective_exposure_per_step": 2048,
        "recurrent_state_buffer_schema_version": RECURRENT_STATE_BUFFER_SCHEMA_VERSION,
        "z_H_bytes": z_h_bytes,
        "z_L_bytes": z_l_bytes,
        "persistent_carry_bytes": 0,
        "total_recurrent_state_bytes": total_bytes,
    }


def test_byte_estimator_counts_only_z_h_and_z_l_in_nocarry_mode():
    spec = RecurrentStateBufferSpec(
        batch_size=2,
        seq_len=16,
        hidden_size=32,
        dtype=torch.bfloat16,
        H_cycles=2,
        L_cycles=3,
        bp_steps=2,
    )

    estimate = estimate_recurrent_state_buffers(spec)

    one_state = 2 * 16 * 32 * 2
    assert estimate.schema_version == RECURRENT_STATE_BUFFER_SCHEMA_VERSION
    assert estimate.dtype_name == "bfloat16"
    assert estimate.dtype_bytes == 2
    assert estimate.z_H_bytes == one_state
    assert estimate.z_L_bytes == one_state
    assert estimate.persistent_carry_bytes == 0
    assert estimate.total_recurrent_state_bytes == 2 * one_state


def test_dtype_mapping_accepts_torch_and_string_names_and_rejects_unknowns():
    assert recurrent_state_dtype_nbytes(torch.float32) == 4
    assert recurrent_state_dtype_nbytes("torch.float16") == 2
    assert recurrent_state_dtype_nbytes("bfloat16") == 2

    with pytest.raises(ValueError, match="unsupported recurrent-state dtype"):
        recurrent_state_dtype_nbytes("complex64")


def test_non_nocarry_persistent_state_accounting_is_deferred():
    spec = RecurrentStateBufferSpec(
        batch_size=1,
        seq_len=4,
        hidden_size=8,
        dtype="float32",
        H_cycles=2,
        L_cycles=3,
        bp_steps=2,
        nocarry=False,
    )

    with pytest.raises(NotImplementedError, match="persistent carry accounting is deferred"):
        estimate_recurrent_state_buffers(spec)


def test_schedule_summary_mirrors_hrm_bp_steps_2_and_5():
    bp2 = recurrent_state_schedule_summary(H_cycles=2, L_cycles=3, bp_steps=2)
    assert bp2.total_l_calls == 6
    assert bp2.total_h_calls == 2
    assert bp2.scheduled_l_grad_enabled_calls == 1
    assert bp2.scheduled_h_grad_enabled_calls == 1
    assert bp2.scheduled_grad_enabled_calls == 2
    assert bp2.grad_enabled_call_order == (("L", 5), ("H", 1))

    bp5 = recurrent_state_schedule_summary(H_cycles=2, L_cycles=3, bp_steps=5)
    assert bp5.scheduled_l_grad_enabled_calls == 3
    assert bp5.scheduled_h_grad_enabled_calls == 2
    assert bp5.scheduled_grad_enabled_calls == 5
    assert bp5.grad_enabled_call_order == (
        ("H", 0),
        ("L", 3),
        ("L", 4),
        ("L", 5),
        ("H", 1),
    )


def test_measurement_validator_rejects_memory_only_receipts():
    memory_only = {
        "peak_allocated_bytes": 1024,
        "peak_reserved_bytes": 2048,
    }
    with pytest.raises(ValueError, match="wall_clock_per_step_seconds"):
        validate_recurrent_state_buffer_measurement(memory_only)

    spec = RecurrentStateBufferSpec(
        batch_size=2,
        seq_len=16,
        hidden_size=32,
        dtype="bfloat16",
        H_cycles=2,
        L_cycles=3,
        bp_steps=2,
    )
    estimate = estimate_recurrent_state_buffers(spec)
    receipt = _complete_receipt(
        estimate.total_recurrent_state_bytes,
        estimate.z_H_bytes,
        estimate.z_L_bytes,
    )
    validate_recurrent_state_buffer_measurement(receipt)
    assert set(REQUIRED_RECURRENT_STATE_MEASUREMENT_FIELDS) <= set(receipt)


def test_measurement_validator_rejects_inconsistent_schema_or_byte_totals():
    spec = RecurrentStateBufferSpec(
        batch_size=1,
        seq_len=8,
        hidden_size=16,
        dtype="float32",
        H_cycles=2,
        L_cycles=3,
        bp_steps=5,
    )
    estimate = estimate_recurrent_state_buffers(spec)
    receipt = _complete_receipt(
        estimate.total_recurrent_state_bytes,
        estimate.z_H_bytes,
        estimate.z_L_bytes,
    )

    wrong_schema = dict(receipt, recurrent_state_buffer_schema_version="old")
    with pytest.raises(ValueError, match="recurrent_state_buffer_schema_version"):
        validate_recurrent_state_buffer_measurement(wrong_schema)

    wrong_total = dict(receipt, total_recurrent_state_bytes=receipt["total_recurrent_state_bytes"] + 1)
    with pytest.raises(ValueError, match="total_recurrent_state_bytes"):
        validate_recurrent_state_buffer_measurement(wrong_total)

    stale_carry = dict(receipt, persistent_carry_bytes=1, total_recurrent_state_bytes=receipt["total_recurrent_state_bytes"] + 1)
    with pytest.raises(ValueError, match="persistent_carry_bytes"):
        validate_recurrent_state_buffer_measurement(stale_carry)


def test_relief_modes_are_off_or_deferred_without_claims():
    assert validate_recurrent_state_relief_mode(MODE_RECURRENT_STATE_OFF) == MODE_RECURRENT_STATE_OFF

    for mode in (
        MODE_LOSSLESS_RECURRENT_STATE_OFFLOAD,
        MODE_LOSSLESS_RECURRENT_STATE_REMATERIALIZATION,
    ):
        with pytest.raises(NotImplementedError, match=TIER1_LOSSLESS_RECURRENT_STATE_RELIEF_DEFERRED):
            validate_recurrent_state_relief_mode(mode)

    with pytest.raises(NotImplementedError, match=TIER2_LOSSY_RECURRENT_STATE_COMPRESSION_DEFERRED):
        validate_recurrent_state_relief_mode(MODE_LOSSY_RECURRENT_STATE_COMPRESSION)
