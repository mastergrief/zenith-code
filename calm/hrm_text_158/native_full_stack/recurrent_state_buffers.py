"""CPU/static recurrent-state buffer accounting for HRM-Text-1.58.

This slice is contract/estimator only. It accounts for the live nocarry
recurrent state buffers (`z_H`, `z_L`) and names future relief mechanisms as
deferred; it does not implement offload, rematerialization, compression, or a
runtime memory win.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from calm.hrm_text_158.native_full_stack.activation_relief import (
    MODE_LOSSLESS_RECOMPUTE,
    ActivationReliefPolicy,
    recurrence_checkpoint_decisions,
)


RECURRENT_STATE_BUFFER_SCHEMA_VERSION = (
    "hrm_text_158_recurrent_state_buffers/v0.contract_estimator"
)

MODE_RECURRENT_STATE_OFF = "off"
MODE_LOSSLESS_RECURRENT_STATE_OFFLOAD = "lossless_recurrent_state_offload"
MODE_LOSSLESS_RECURRENT_STATE_REMATERIALIZATION = (
    "lossless_recurrent_state_rematerialization"
)
MODE_LOSSY_RECURRENT_STATE_COMPRESSION = "lossy_recurrent_state_compression"

TIER1_LOSSLESS_RECURRENT_STATE_RELIEF_DEFERRED = (
    "tier1_lossless_recurrent_state_relief_deferred"
)
TIER2_LOSSY_RECURRENT_STATE_COMPRESSION_DEFERRED = (
    "tier2_lossy_recurrent_state_compression_deferred"
)

REQUIRED_RECURRENT_STATE_MEASUREMENT_FIELDS = (
    "peak_allocated_bytes",
    "peak_reserved_bytes",
    "wall_clock_per_step_seconds",
    "max_safe_batch_size",
    "effective_exposure_per_step",
    "recurrent_state_buffer_schema_version",
    "z_H_bytes",
    "z_L_bytes",
    "persistent_carry_bytes",
    "total_recurrent_state_bytes",
)

_DTYPE_BYTE_WIDTHS = {
    torch.float64: 8,
    torch.float32: 4,
    torch.float16: 2,
    torch.bfloat16: 2,
    torch.int64: 8,
    torch.int32: 4,
    torch.int16: 2,
    torch.int8: 1,
    torch.uint8: 1,
    torch.bool: 1,
}
_DTYPE_BY_NAME = {
    "float64": torch.float64,
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "int64": torch.int64,
    "int32": torch.int32,
    "int16": torch.int16,
    "int8": torch.int8,
    "uint8": torch.uint8,
    "bool": torch.bool,
}


def normalize_recurrent_state_dtype(dtype: torch.dtype | str) -> torch.dtype:
    """Return a supported torch dtype for static byte accounting."""

    if isinstance(dtype, str):
        dtype_name = dtype.removeprefix("torch.")
        if dtype_name in _DTYPE_BY_NAME:
            return _DTYPE_BY_NAME[dtype_name]
        valid = ", ".join(sorted(_DTYPE_BY_NAME))
        raise ValueError(f"unsupported recurrent-state dtype {dtype!r}; valid={valid}")
    try:
        if dtype in _DTYPE_BYTE_WIDTHS:
            return dtype
    except TypeError:
        pass
    raise ValueError(f"unsupported recurrent-state dtype {dtype!r}")


def recurrent_state_dtype_nbytes(dtype: torch.dtype | str) -> int:
    """Byte width for dtypes accepted by the recurrent-state estimator."""

    return _DTYPE_BYTE_WIDTHS[normalize_recurrent_state_dtype(dtype)]


@dataclass(frozen=True)
class RecurrentStateBufferSpec:
    batch_size: int
    seq_len: int
    hidden_size: int
    dtype: torch.dtype | str
    H_cycles: int
    L_cycles: int
    bp_steps: int
    nocarry: bool = True

    def validate(self) -> "RecurrentStateBufferSpec":
        for name, value in (
            ("batch_size", self.batch_size),
            ("seq_len", self.seq_len),
            ("hidden_size", self.hidden_size),
            ("H_cycles", self.H_cycles),
            ("L_cycles", self.L_cycles),
            ("bp_steps", self.bp_steps),
        ):
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        normalize_recurrent_state_dtype(self.dtype)
        if not self.nocarry:
            raise NotImplementedError(
                "persistent carry accounting is deferred; HRM-Text-1.58 nocarry "
                "has persistent_carry_bytes=0"
            )
        return self

    @property
    def dtype_bytes(self) -> int:
        return recurrent_state_dtype_nbytes(self.dtype)

    @property
    def one_state_bytes(self) -> int:
        return int(self.batch_size * self.seq_len * self.hidden_size * self.dtype_bytes)


@dataclass(frozen=True)
class RecurrentStateScheduleSummary:
    H_cycles: int
    L_cycles: int
    bp_steps: int
    total_l_calls: int
    total_h_calls: int
    scheduled_l_grad_enabled_calls: int
    scheduled_h_grad_enabled_calls: int
    scheduled_grad_enabled_calls: int
    grad_enabled_call_order: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class RecurrentStateBufferEstimate:
    schema_version: str
    spec: RecurrentStateBufferSpec
    dtype_name: str
    dtype_bytes: int
    z_H_bytes: int
    z_L_bytes: int
    persistent_carry_bytes: int
    total_recurrent_state_bytes: int
    schedule: RecurrentStateScheduleSummary


def recurrent_state_schedule_summary(
    *,
    H_cycles: int,
    L_cycles: int,
    bp_steps: int,
    outer_grad_enabled: bool = True,
) -> RecurrentStateScheduleSummary:
    """Summarize the HRM H/L grad-enabled recurrence schedule."""

    spec = RecurrentStateBufferSpec(
        batch_size=1,
        seq_len=1,
        hidden_size=1,
        dtype=torch.float32,
        H_cycles=H_cycles,
        L_cycles=L_cycles,
        bp_steps=bp_steps,
    ).validate()
    decisions = recurrence_checkpoint_decisions(
        ActivationReliefPolicy(mode=MODE_LOSSLESS_RECOMPUTE),
        H_cycles=spec.H_cycles,
        L_cycles=spec.L_cycles,
        bp_steps=spec.bp_steps,
        outer_grad_enabled=outer_grad_enabled,
    )
    grad_order = tuple(
        (decision.level, decision.rec_idx)
        for decision in decisions
        if decision.scheduled_grad_enabled
    )
    scheduled_l = sum(1 for level, _ in grad_order if level == "L")
    scheduled_h = sum(1 for level, _ in grad_order if level == "H")
    return RecurrentStateScheduleSummary(
        H_cycles=spec.H_cycles,
        L_cycles=spec.L_cycles,
        bp_steps=spec.bp_steps,
        total_l_calls=spec.H_cycles * spec.L_cycles,
        total_h_calls=spec.H_cycles,
        scheduled_l_grad_enabled_calls=scheduled_l,
        scheduled_h_grad_enabled_calls=scheduled_h,
        scheduled_grad_enabled_calls=len(grad_order),
        grad_enabled_call_order=grad_order,
    )


def estimate_recurrent_state_buffers(
    spec: RecurrentStateBufferSpec,
) -> RecurrentStateBufferEstimate:
    """Estimate live nocarry recurrent-state bytes for `z_H` and `z_L`."""

    spec = spec.validate()
    dtype = normalize_recurrent_state_dtype(spec.dtype)
    z_h_bytes = spec.one_state_bytes
    z_l_bytes = spec.one_state_bytes
    persistent_carry_bytes = 0
    return RecurrentStateBufferEstimate(
        schema_version=RECURRENT_STATE_BUFFER_SCHEMA_VERSION,
        spec=spec,
        dtype_name=str(dtype).removeprefix("torch."),
        dtype_bytes=spec.dtype_bytes,
        z_H_bytes=z_h_bytes,
        z_L_bytes=z_l_bytes,
        persistent_carry_bytes=persistent_carry_bytes,
        total_recurrent_state_bytes=z_h_bytes + z_l_bytes + persistent_carry_bytes,
        schedule=recurrent_state_schedule_summary(
            H_cycles=spec.H_cycles,
            L_cycles=spec.L_cycles,
            bp_steps=spec.bp_steps,
        ),
    )


def validate_recurrent_state_relief_mode(mode: str) -> str:
    """Validate named recurrent-state relief modes for this contract slice."""

    if mode == MODE_RECURRENT_STATE_OFF:
        return mode
    if mode in {
        MODE_LOSSLESS_RECURRENT_STATE_OFFLOAD,
        MODE_LOSSLESS_RECURRENT_STATE_REMATERIALIZATION,
    }:
        raise NotImplementedError(
            "lossless recurrent-state offload/rematerialization is "
            f"{TIER1_LOSSLESS_RECURRENT_STATE_RELIEF_DEFERRED}; "
            "this slice is contract/estimator only"
        )
    if mode == MODE_LOSSY_RECURRENT_STATE_COMPRESSION:
        raise NotImplementedError(
            "lossy recurrent-state compression is "
            f"{TIER2_LOSSY_RECURRENT_STATE_COMPRESSION_DEFERRED} and needs "
            "acquisition re-validation before any claim"
        )
    raise ValueError(f"unknown recurrent-state relief mode: {mode!r}")


def _require_numeric(receipt: Mapping[str, object], field: str) -> int | float:
    value = receipt[field]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be numeric, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{field} must be non-negative, got {value!r}")
    return value


def validate_recurrent_state_buffer_measurement(
    receipt: Mapping[str, object],
) -> None:
    """Validate future recurrent-state receipts.

    A valid receipt must include resource metrics and recurrent-state schema/byte
    fields together. Memory-only receipts are intentionally rejected.
    """

    missing = [
        field
        for field in REQUIRED_RECURRENT_STATE_MEASUREMENT_FIELDS
        if field not in receipt
    ]
    if missing:
        raise ValueError(
            "recurrent-state measurement missing required fields: "
            + ", ".join(missing)
        )
    if (
        receipt["recurrent_state_buffer_schema_version"]
        != RECURRENT_STATE_BUFFER_SCHEMA_VERSION
    ):
        raise ValueError(
            "recurrent_state_buffer_schema_version must equal "
            f"{RECURRENT_STATE_BUFFER_SCHEMA_VERSION!r}"
        )

    numeric = {
        field: _require_numeric(receipt, field)
        for field in REQUIRED_RECURRENT_STATE_MEASUREMENT_FIELDS
        if field != "recurrent_state_buffer_schema_version"
    }
    for positive_field in (
        "wall_clock_per_step_seconds",
        "max_safe_batch_size",
        "effective_exposure_per_step",
        "z_H_bytes",
        "z_L_bytes",
        "total_recurrent_state_bytes",
    ):
        if numeric[positive_field] <= 0:
            raise ValueError(f"{positive_field} must be > 0")
    if numeric["peak_reserved_bytes"] < numeric["peak_allocated_bytes"]:
        raise ValueError("peak_reserved_bytes must be >= peak_allocated_bytes")
    if numeric["persistent_carry_bytes"] != 0:
        raise ValueError("persistent_carry_bytes must be 0 for HRM-Text-1.58 nocarry")
    expected_total = (
        numeric["z_H_bytes"]
        + numeric["z_L_bytes"]
        + numeric["persistent_carry_bytes"]
    )
    if numeric["total_recurrent_state_bytes"] != expected_total:
        raise ValueError(
            "total_recurrent_state_bytes must equal "
            "z_H_bytes + z_L_bytes + persistent_carry_bytes"
        )
