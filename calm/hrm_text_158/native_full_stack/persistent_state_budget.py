"""Persistent-state packing and budget accounting for HRM-Text-1.58.

This seam is intentionally a reference/accounting bridge. It proves exact
q:int8 ternary pack/unpack and reports the inclusive q + acc + scale physical
budget without claiming accumulator compression or sub-2-bit persistent state.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import torch

from calm.hrm_text_158.native_full_stack.qscale_linear import (
    QScaleWeightState,
    validate_qscale_weight_state,
)


RUN_GPU_PERSISTENT_STATE_BUDGET_ENV = "HRM_TEXT_158_RUN_GPU_PERSISTENT_STATE_BUDGET"
PERSISTENT_STATE_BUDGET_SCHEMA_VERSION = "hrm_text_158_persistent_state_budget/v0.q_pack_reference"
PERSISTENT_STATE_BUDGET_LABEL = "persistent_state_q_pack_reference_3ledger_accounting_over_target"
PACKED_TERNARY_Q_FORMAT = "packed_2bit_ternary_reference"
INCLUSIVE_3LEDGER_TARGET_BASIS = "inclusive_3ledger_physical"
TRACK_B_DESIGN_ONLY_STATUS = "design_math_only_no_accumulator_compression_implemented"
PHYSICAL_SUB2_NOT_ACHIEVED_STATEMENT = (
    "physical persistent sub-2-bit NOT achieved yet; int16 vote-acc remains the dominant term."
)
EFFECTIVE_FORWARD_TERNARY_BITS = math.log2(3.0)
TARGET_PHYSICAL_BITS_PER_WEIGHT = 2.0

# Reference persisted metadata estimate: uint64 logical_numel, uint64 compact
# schema/format/padding header, plus int64 per logical shape dimension.
PACKED_TERNARY_METADATA_HEADER_BYTES = 16
PACKED_TERNARY_METADATA_BYTES_PER_DIM = 8

R3_PERSISTENT_STATE_BUDGET_SCHEMA_VERSION = "hrm_text_158_persistent_state_budget/v0.r3_w6_byte_packed"
R3_PERSISTENT_STATE_BUDGET_LABEL = "r3_w6_byte_packed_checkpoint_ledger_not_sub2"
R3_ACC_PHYSICAL_BITS_PER_WEIGHT = 6.0
R3_Q_INT8_BITS_PER_WEIGHT = 8.0
R3_LEDGER_PASS_INCLUSIVE_BPW_CEILING = 15.0
R3_ACC_BPW_TOLERANCE = 0.25
R3_W6_BYTE_PACKED_SCHEMA = "w6_lanes_byte_packed/v0"
R3_W6_BYTEPACKED_NOT_SUB2_STATEMENT = (
    "physical persistent sub-2-bit NOT achieved yet; serialized vote accumulator = "
    "W6 / 6 bpw (reduced from int16); NOT sub-2 because q stays int8 = 8 bpw and "
    "inclusive checkpoint ~14 bpw; remaining sub-2 levers = q-pack (rung-4) + "
    "vote-acc width."
)
R3_ARTIFACT_BYTES_SEMANTICS_ACTUAL_PAYLOAD = "actual_packed_payload_bytes"

R4_PERSISTENT_STATE_BUDGET_SCHEMA_VERSION = (
    "hrm_text_158_persistent_state_budget/v0.r4_q_ternary_byte_packed"
)
R4_PERSISTENT_STATE_BUDGET_LABEL = "r4_q_ternary_byte_packed_checkpoint_ledger_not_sub2"
R4_Q_PHYSICAL_BITS_PER_WEIGHT = 2.0
R4_ACC_PHYSICAL_BITS_PER_WEIGHT = 6.0
R4_LEDGER_PASS_INCLUSIVE_BPW_CEILING = 8.5
R4_Q_BPW_TOLERANCE = 0.25
R4_ACC_BPW_TOLERANCE = 0.25
R4_Q_TERNARY_BYTEPACKED_NOT_SUB2_STATEMENT = (
    "R4 q-state/q-term packing: serialized q = ternary 2-bit / ~2 bpw data; "
    "serialized acc = W6 / 6 bpw; inclusive checkpoint ~8 bpw; still NOT sub-2, "
    "NOT readiness, NOT hot-path, NOT mid-run resume."
)

R5_PERSISTENT_STATE_BUDGET_SCHEMA_VERSION = (
    "hrm_text_158_persistent_state_budget/v0.r5_w5_byte_packed_decision_parity"
)
R5_PERSISTENT_STATE_BUDGET_LABEL = (
    "r5_w5_byte_packed_decision_parity_checkpoint_ledger_not_sub2"
)
R5_ACC_PHYSICAL_BITS_PER_WEIGHT = 5.0
R5_LEDGER_PASS_INCLUSIVE_BPW_CEILING = 7.5
R5_ACC_BPW_TOLERANCE = 0.25
R5_W5_BYTE_PACKED_SCHEMA = "w5_lanes_byte_packed/v0"
R5_W5_BYTEPACKED_DECISION_PARITY_NOT_SUB2_STATEMENT = (
    "R5 W5 vote-acc byte-pack: serialized acc = W5 / 5 bpw (lossy clip, decision-parity lane); "
    "with q 2-bit packed inclusive ~7 bpw; NOT lossless, NOT sub-2-inclusive, NOT readiness."
)


@dataclass(frozen=True)
class PackedTernaryQState:
    """Reference 2-bit ternary q payload plus logical metadata."""

    packed: torch.Tensor
    logical_shape: tuple[int, ...]
    logical_numel: int
    padding_values: int
    format: str = PACKED_TERNARY_Q_FORMAT

    @property
    def packed_data_bytes(self) -> int:
        return int(self.packed.numel() * self.packed.element_size())

    @property
    def packed_data_bits(self) -> int:
        return int(self.packed_data_bytes * 8)

    @property
    def ideal_ternary_2bit_bits(self) -> int:
        return int(self.logical_numel * 2)

    @property
    def padding_bits(self) -> int:
        return int(self.packed_data_bits - self.ideal_ternary_2bit_bits)

    @property
    def metadata_bytes(self) -> int:
        return int(PACKED_TERNARY_METADATA_HEADER_BYTES + PACKED_TERNARY_METADATA_BYTES_PER_DIM * len(self.logical_shape))

    @property
    def metadata_bits(self) -> int:
        return int(self.metadata_bytes * 8)


@dataclass(frozen=True)
class PersistentStateBudgetReport:
    """Measured physical persistent-state budget over eligible q entries."""

    schema_version: str
    label: str
    target_basis: str
    target_bits_per_weight: float
    target_achieved: bool
    eligible_weight_count: int
    q_state_count: int
    accumulator_tensor_count: int
    current_q_int8_bits: int
    packed_q_data_bits: int
    packed_q_padding_bits: int
    packed_q_metadata_bits: int
    packed_q_total_bits: int
    acc_int16_bits: int
    frozen_scale_fp32_bits: int
    current_inclusive_physical_bits: int
    packed_inclusive_physical_bits: int
    scale_excluded_diagnostic_bits: int
    acc_scale_excluded_diagnostic_bits: int
    current_inclusive_physical_bits_per_weight: float
    packed_inclusive_physical_bits_per_weight: float
    q_int8_bits_per_weight: float
    q_packed_data_bits_per_weight: float
    q_packed_padding_bits_per_weight: float
    q_packed_metadata_bits_per_weight: float
    q_packed_total_bits_per_weight: float
    acc_int16_bits_per_weight: float
    frozen_scale_fp32_bits_per_weight: float
    scale_excluded_diagnostic_bits_per_weight: float
    acc_scale_excluded_diagnostic_bits_per_weight: float
    q_effective_forward_entropy_bits_per_weight: float
    required_acc_bits_per_weight_for_sub2_physical_q_with_scale_and_metadata: float
    required_acc_bits_per_weight_for_sub2_effective_q_with_scale: float
    dominant_ledger: str
    track_b_status: str
    receipt_statement: str

    def to_dict(self) -> dict[str, int | float | bool | str]:
        return {field.name: getattr(self, field.name) for field in fields(self)}


@dataclass(frozen=True)
class R3PersistentStateBudgetReport:
    """Byte-derived R3 checkpoint ledger for W6-packed accumulator payloads."""

    schema_version: str
    label: str
    eligible_weight_count: int
    q_state_count: int
    accumulator_payload_count: int
    r3_q_int8_bits_per_weight: float
    r3_acc_logical_lane_bits: float
    r3_acc_physical_bits_per_weight: float
    r3_checkpoint_inclusive_physical_bits_per_weight: float
    r3_actual_acc_payload_bytes: int
    r3_frozen_scale_fp32_bits: int
    r3_artifact_bytes_total: int
    r3_artifact_overhead_bytes: int
    r3_packed_payload_content_sha256: str
    r3_ledger_pass: bool
    receipt_statement: str

    def to_dict(self) -> dict[str, int | float | bool | str]:
        return {field.name: getattr(self, field.name) for field in fields(self)}


@dataclass(frozen=True)
class R4PersistentStateBudgetReport:
    """Byte-derived R4 checkpoint ledger for packed-q + W6-packed acc payloads."""

    schema_version: str
    label: str
    eligible_weight_count: int
    q_state_count: int
    accumulator_payload_count: int
    r4_q_physical_bits_per_weight: float
    r4_q_metadata_bits_per_weight: float
    r4_acc_physical_bits_per_weight: float
    r4_checkpoint_inclusive_physical_bits_per_weight: float
    r4_actual_q_payload_bytes: int
    r4_actual_q_metadata_bytes: int
    r4_actual_acc_payload_bytes: int
    r4_frozen_scale_fp32_bits: int
    r4_q_packed_content_sha256: str
    r4_acc_packed_content_sha256: str
    r4_ledger_pass: bool
    receipt_statement: str

    def to_dict(self) -> dict[str, int | float | bool | str]:
        return {field.name: getattr(self, field.name) for field in fields(self)}


@dataclass(frozen=True)
class R5PersistentStateBudgetReport:
    """Byte-derived R5 checkpoint ledger for packed-q + W5-packed acc (decision-parity)."""

    schema_version: str
    label: str
    eligible_weight_count: int
    q_state_count: int
    accumulator_payload_count: int
    r5_q_physical_bits_per_weight: float
    r5_q_metadata_bits_per_weight: float
    r5_acc_physical_bits_per_weight: float
    r5_checkpoint_inclusive_physical_bits_per_weight: float
    r5_actual_q_payload_bytes: int
    r5_actual_q_metadata_bytes: int
    r5_actual_acc_payload_bytes: int
    r5_frozen_scale_fp32_bits: int
    r5_q_packed_content_sha256: str
    r5_acc_packed_content_sha256: str
    r5_ledger_pass: bool
    receipt_statement: str

    def to_dict(self) -> dict[str, int | float | bool | str]:
        return {field.name: getattr(self, field.name) for field in fields(self)}


def _reject_int16_tensor_as_packed_acc(value: torch.Tensor, *, context: str) -> None:
    if value.dtype == torch.int16:
        raise ValueError(
            f"{context} must be a real uint8 byte payload with schema "
            f"{R3_W6_BYTE_PACKED_SCHEMA!r}, not torch.int16"
        )
    if value.element_size() > 1:
        raise ValueError(
            f"{context} physical payload must be 1-byte elements, got "
            f"element_size={value.element_size()}"
        )


def _validate_r3_packed_payloads(
    packed_acc_payloads: Sequence[Any],
    *,
    eligible_weight_count: int,
) -> int:
    if len(packed_acc_payloads) == 0:
        raise ValueError("at least one byte-packed accumulator payload is required for R3 ledger")
    total_lanes = 0
    total_payload_bytes = 0
    for payload in packed_acc_payloads:
        schema = str(getattr(payload, "schema", ""))
        if schema != R3_W6_BYTE_PACKED_SCHEMA:
            raise ValueError(
                f"R3 ledger requires schema {R3_W6_BYTE_PACKED_SCHEMA!r}, got {schema!r}"
            )
        packed = getattr(payload, "packed")
        _reject_int16_tensor_as_packed_acc(packed, context="R3 packed acc payload")
        if packed.dtype != torch.uint8:
            raise ValueError(f"R3 packed acc payload must be torch.uint8, got {packed.dtype}")
        total_lanes += int(getattr(payload, "logical_numel"))
        total_payload_bytes += int(getattr(payload, "packed_data_bytes"))
    if total_lanes != int(eligible_weight_count):
        raise ValueError(
            "sum(packed_acc.logical_numel) must match eligible q entries; "
            f"got packed_lanes={total_lanes}, eligible={eligible_weight_count}"
        )
    return int(total_payload_bytes)


def _validate_r5_packed_acc_payloads(
    packed_acc_payloads: Sequence[Any],
    *,
    eligible_weight_count: int,
) -> int:
    if len(packed_acc_payloads) == 0:
        raise ValueError("at least one byte-packed W5 accumulator payload is required for R5 ledger")
    total_lanes = 0
    total_payload_bytes = 0
    for payload in packed_acc_payloads:
        schema = str(getattr(payload, "schema", ""))
        if schema != R5_W5_BYTE_PACKED_SCHEMA:
            raise ValueError(
                f"R5 ledger requires schema {R5_W5_BYTE_PACKED_SCHEMA!r}, got {schema!r}"
            )
        packed = getattr(payload, "packed")
        _reject_int16_tensor_as_packed_acc(packed, context="R5 packed acc payload")
        if packed.dtype != torch.uint8:
            raise ValueError(f"R5 packed acc payload must be torch.uint8, got {packed.dtype}")
        total_lanes += int(getattr(payload, "logical_numel"))
        total_payload_bytes += int(getattr(payload, "packed_data_bytes"))
    if total_lanes != int(eligible_weight_count):
        raise ValueError(
            "sum(packed_acc.logical_numel) must match eligible q entries; "
            f"got packed_lanes={total_lanes}, eligible={eligible_weight_count}"
        )
    return int(total_payload_bytes)


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_r3_per_module_payload_rows(
    state_keys: Sequence[str],
    packed_acc_payloads: Sequence[Any],
) -> list[dict[str, int | float | str | list[int]]]:
    """Build sorted per-module R3 payload witness rows without raw byte lists."""

    if len(state_keys) != len(packed_acc_payloads):
        raise ValueError("state_keys length must match packed_acc_payloads")
    rows: list[dict[str, int | float | str | list[int]]] = []
    for state_key, payload in sorted(
        zip(state_keys, packed_acc_payloads),
        key=lambda item: item[0],
    ):
        lanes = int(getattr(payload, "logical_numel"))
        payload_bytes = int(getattr(payload, "packed_data_bytes"))
        packed = getattr(payload, "packed").detach().cpu().contiguous()
        rows.append(
            {
                "state_key": str(state_key),
                "logical_shape": [int(dim) for dim in getattr(payload, "logical_shape")],
                "lanes": lanes,
                "payload_bytes": payload_bytes,
                "acc_bpw": _bits_per_weight(payload_bytes * 8, lanes),
                "payload_sha256": _sha256_hex(packed.numpy().tobytes()),
            }
        )
    return rows


def canonical_r3_packed_payload_content_sha256(
    per_module_rows: Sequence[Mapping[str, Any]],
) -> str:
    """Canonical aggregate sha256 over sorted per-module payload witness rows."""

    digest = hashlib.sha256()
    for row in per_module_rows:
        digest.update(
            json.dumps(dict(row), sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def measure_r3_persistent_state_budget(
    qscale_states: Sequence[QScaleWeightState],
    packed_acc_payloads: Sequence[Any],
    *,
    artifact_bytes_total: int | None = None,
    state_keys: Sequence[str] | None = None,
) -> R3PersistentStateBudgetReport:
    """Measure R3 checkpoint ledger from real uint8 accumulator payload bytes."""

    if len(qscale_states) == 0:
        raise ValueError("at least one qscale state is required for R3 persistent-state accounting")

    eligible_weight_count = 0
    scale_bits = 0
    for qscale_state in qscale_states:
        q_levels, _scale, _ = validate_qscale_weight_state(qscale_state)
        eligible_weight_count += int(q_levels.numel())
        scale_bits += 32

    actual_acc_payload_bytes = _validate_r3_packed_payloads(
        packed_acc_payloads,
        eligible_weight_count=eligible_weight_count,
    )
    effective_state_keys = (
        list(state_keys)
        if state_keys is not None
        else [f"payload_{index}" for index in range(len(packed_acc_payloads))]
    )
    if len(effective_state_keys) != len(packed_acc_payloads):
        raise ValueError("state_keys length must match packed_acc_payloads")
    per_module_rows = build_r3_per_module_payload_rows(
        effective_state_keys,
        packed_acc_payloads,
    )
    content_sha256 = canonical_r3_packed_payload_content_sha256(per_module_rows)
    acc_physical_bpw = _bits_per_weight(actual_acc_payload_bytes * 8, eligible_weight_count)
    scale_bpw = _bits_per_weight(scale_bits, eligible_weight_count)
    inclusive_bpw = float(R3_Q_INT8_BITS_PER_WEIGHT) + float(acc_physical_bpw) + float(scale_bpw)
    artifact_total = int(actual_acc_payload_bytes if artifact_bytes_total is None else artifact_bytes_total)
    if artifact_total < actual_acc_payload_bytes:
        raise ValueError("r3_artifact_bytes_total must be >= r3_actual_acc_payload_bytes")
    overhead_bytes = int(artifact_total - actual_acc_payload_bytes)
    ledger_pass = (
        abs(float(acc_physical_bpw) - R3_ACC_PHYSICAL_BITS_PER_WEIGHT) <= R3_ACC_BPW_TOLERANCE
        and float(inclusive_bpw) <= R3_LEDGER_PASS_INCLUSIVE_BPW_CEILING
    )
    return R3PersistentStateBudgetReport(
        schema_version=R3_PERSISTENT_STATE_BUDGET_SCHEMA_VERSION,
        label=R3_PERSISTENT_STATE_BUDGET_LABEL,
        eligible_weight_count=int(eligible_weight_count),
        q_state_count=int(len(qscale_states)),
        accumulator_payload_count=int(len(packed_acc_payloads)),
        r3_q_int8_bits_per_weight=float(R3_Q_INT8_BITS_PER_WEIGHT),
        r3_acc_logical_lane_bits=float(R3_ACC_PHYSICAL_BITS_PER_WEIGHT),
        r3_acc_physical_bits_per_weight=float(acc_physical_bpw),
        r3_checkpoint_inclusive_physical_bits_per_weight=float(inclusive_bpw),
        r3_actual_acc_payload_bytes=int(actual_acc_payload_bytes),
        r3_frozen_scale_fp32_bits=int(scale_bits),
        r3_artifact_bytes_total=int(artifact_total),
        r3_artifact_overhead_bytes=int(overhead_bytes),
        r3_packed_payload_content_sha256=str(content_sha256),
        r3_ledger_pass=bool(ledger_pass),
        receipt_statement=R3_W6_BYTEPACKED_NOT_SUB2_STATEMENT,
    )


def reject_int16_tensors_for_r3_ledger(
    accumulator_tensors: Sequence[torch.Tensor],
) -> None:
    """Fail-closed: R3 ledger must never accept int16 tensors as packed acc."""

    for acc in accumulator_tensors:
        _reject_int16_tensor_as_packed_acc(acc, context="R3 ledger accumulator input")


def _reject_int8_tensor_as_packed_q(value: torch.Tensor, *, context: str) -> None:
    if value.dtype == torch.int8:
        raise ValueError(
            f"{context} must be a real uint8 byte payload with schema "
            f"{PACKED_TERNARY_Q_FORMAT!r}, not torch.int8"
        )
    if value.element_size() > 1:
        raise ValueError(
            f"{context} physical payload must be 1-byte elements, got "
            f"element_size={value.element_size()}"
        )


def _validate_r4_packed_q_payloads(
    packed_q_payloads: Sequence[PackedTernaryQState],
    *,
    eligible_weight_count: int,
) -> tuple[int, int]:
    if len(packed_q_payloads) == 0:
        raise ValueError("at least one byte-packed q payload is required for R4 ledger")
    total_lanes = 0
    total_payload_bytes = 0
    total_metadata_bytes = 0
    for payload in packed_q_payloads:
        _validate_packed_state_metadata(payload)
        _reject_int8_tensor_as_packed_q(payload.packed, context="R4 packed q payload")
        if payload.packed.dtype != torch.uint8:
            raise ValueError(f"R4 packed q payload must be torch.uint8, got {payload.packed.dtype}")
        total_lanes += int(payload.logical_numel)
        total_payload_bytes += int(payload.packed_data_bytes)
        total_metadata_bytes += int(payload.metadata_bytes)
    if total_lanes != int(eligible_weight_count):
        raise ValueError(
            "sum(packed_q.logical_numel) must match eligible q entries; "
            f"got packed_lanes={total_lanes}, eligible={eligible_weight_count}"
        )
    return int(total_payload_bytes), int(total_metadata_bytes)


def build_r4_per_module_q_rows(
    state_keys: Sequence[str],
    packed_q_payloads: Sequence[PackedTernaryQState],
) -> list[dict[str, int | float | str | list[int]]]:
    if len(state_keys) != len(packed_q_payloads):
        raise ValueError("state_keys length must match packed_q_payloads")
    rows: list[dict[str, int | float | str | list[int]]] = []
    for state_key, payload in sorted(zip(state_keys, packed_q_payloads), key=lambda item: item[0]):
        lanes = int(payload.logical_numel)
        payload_bytes = int(payload.packed_data_bytes)
        packed = payload.packed.detach().cpu().contiguous()
        rows.append(
            {
                "state_key": str(state_key),
                "logical_shape": [int(dim) for dim in payload.logical_shape],
                "lanes": lanes,
                "payload_bytes": payload_bytes,
                "metadata_bytes": int(payload.metadata_bytes),
                "padding_values": int(payload.padding_values),
                "q_bpw": _bits_per_weight(payload_bytes * 8, lanes),
                "payload_sha256": _sha256_hex(packed.numpy().tobytes()),
            }
        )
    return rows


def canonical_r4_q_packed_content_sha256(
    per_module_rows: Sequence[Mapping[str, Any]],
) -> str:
    digest = hashlib.sha256()
    for row in per_module_rows:
        digest.update(
            json.dumps(dict(row), sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def reject_int8_tensors_for_r4_ledger(
    q_tensors: Sequence[torch.Tensor],
) -> None:
    """Fail-closed: R4 ledger must never accept int8 tensors as packed q."""

    for q in q_tensors:
        _reject_int8_tensor_as_packed_q(q, context="R4 ledger q input")


def measure_r4_persistent_state_budget(
    qscale_states: Sequence[QScaleWeightState],
    packed_q_payloads: Sequence[PackedTernaryQState],
    packed_acc_payloads: Sequence[Any],
    *,
    state_keys: Sequence[str] | None = None,
) -> R4PersistentStateBudgetReport:
    """Measure R4 checkpoint ledger from real uint8 q + acc payload bytes."""

    if len(qscale_states) == 0:
        raise ValueError("at least one qscale state is required for R4 persistent-state accounting")

    eligible_weight_count = 0
    scale_bits = 0
    for qscale_state in qscale_states:
        q_levels, _scale, _ = validate_qscale_weight_state(qscale_state)
        eligible_weight_count += int(q_levels.numel())
        scale_bits += 32

    actual_q_payload_bytes, actual_q_metadata_bytes = _validate_r4_packed_q_payloads(
        packed_q_payloads,
        eligible_weight_count=eligible_weight_count,
    )
    actual_acc_payload_bytes = _validate_r3_packed_payloads(
        packed_acc_payloads,
        eligible_weight_count=eligible_weight_count,
    )
    effective_state_keys = (
        list(state_keys)
        if state_keys is not None
        else [f"payload_{index}" for index in range(len(packed_q_payloads))]
    )
    if len(effective_state_keys) != len(packed_q_payloads):
        raise ValueError("state_keys length must match packed_q_payloads")
    q_rows = build_r4_per_module_q_rows(effective_state_keys, packed_q_payloads)
    acc_rows = build_r3_per_module_payload_rows(effective_state_keys, packed_acc_payloads)
    q_content_sha256 = canonical_r4_q_packed_content_sha256(q_rows)
    acc_content_sha256 = canonical_r3_packed_payload_content_sha256(acc_rows)
    q_physical_bpw = _bits_per_weight(actual_q_payload_bytes * 8, eligible_weight_count)
    q_metadata_bpw = _bits_per_weight(actual_q_metadata_bytes * 8, eligible_weight_count)
    acc_physical_bpw = _bits_per_weight(actual_acc_payload_bytes * 8, eligible_weight_count)
    scale_bpw = _bits_per_weight(scale_bits, eligible_weight_count)
    inclusive_bpw = float(q_physical_bpw + q_metadata_bpw + acc_physical_bpw + scale_bpw)
    ledger_pass = (
        abs(float(q_physical_bpw) - R4_Q_PHYSICAL_BITS_PER_WEIGHT) <= R4_Q_BPW_TOLERANCE
        and abs(float(acc_physical_bpw) - R4_ACC_PHYSICAL_BITS_PER_WEIGHT) <= R4_ACC_BPW_TOLERANCE
        and float(inclusive_bpw) <= R4_LEDGER_PASS_INCLUSIVE_BPW_CEILING
    )
    return R4PersistentStateBudgetReport(
        schema_version=R4_PERSISTENT_STATE_BUDGET_SCHEMA_VERSION,
        label=R4_PERSISTENT_STATE_BUDGET_LABEL,
        eligible_weight_count=int(eligible_weight_count),
        q_state_count=int(len(qscale_states)),
        accumulator_payload_count=int(len(packed_acc_payloads)),
        r4_q_physical_bits_per_weight=float(q_physical_bpw),
        r4_q_metadata_bits_per_weight=float(q_metadata_bpw),
        r4_acc_physical_bits_per_weight=float(acc_physical_bpw),
        r4_checkpoint_inclusive_physical_bits_per_weight=float(inclusive_bpw),
        r4_actual_q_payload_bytes=int(actual_q_payload_bytes),
        r4_actual_q_metadata_bytes=int(actual_q_metadata_bytes),
        r4_actual_acc_payload_bytes=int(actual_acc_payload_bytes),
        r4_frozen_scale_fp32_bits=int(scale_bits),
        r4_q_packed_content_sha256=str(q_content_sha256),
        r4_acc_packed_content_sha256=str(acc_content_sha256),
        r4_ledger_pass=bool(ledger_pass),
        receipt_statement=R4_Q_TERNARY_BYTEPACKED_NOT_SUB2_STATEMENT,
    )


def measure_r5_persistent_state_budget(
    qscale_states: Sequence[QScaleWeightState],
    packed_q_payloads: Sequence[PackedTernaryQState],
    packed_acc_payloads: Sequence[Any],
    *,
    state_keys: Sequence[str] | None = None,
) -> R5PersistentStateBudgetReport:
    """Measure R5 checkpoint ledger from real uint8 q + W5 acc payload bytes."""

    if len(qscale_states) == 0:
        raise ValueError("at least one qscale state is required for R5 persistent-state accounting")

    eligible_weight_count = 0
    scale_bits = 0
    for qscale_state in qscale_states:
        q_levels, _scale, _ = validate_qscale_weight_state(qscale_state)
        eligible_weight_count += int(q_levels.numel())
        scale_bits += 32

    actual_q_payload_bytes, actual_q_metadata_bytes = _validate_r4_packed_q_payloads(
        packed_q_payloads,
        eligible_weight_count=eligible_weight_count,
    )
    actual_acc_payload_bytes = _validate_r5_packed_acc_payloads(
        packed_acc_payloads,
        eligible_weight_count=eligible_weight_count,
    )
    effective_state_keys = (
        list(state_keys)
        if state_keys is not None
        else [f"payload_{index}" for index in range(len(packed_q_payloads))]
    )
    if len(effective_state_keys) != len(packed_q_payloads):
        raise ValueError("state_keys length must match packed_q_payloads")
    q_rows = build_r4_per_module_q_rows(effective_state_keys, packed_q_payloads)
    acc_rows = build_r3_per_module_payload_rows(effective_state_keys, packed_acc_payloads)
    q_content_sha256 = canonical_r4_q_packed_content_sha256(q_rows)
    acc_content_sha256 = canonical_r3_packed_payload_content_sha256(acc_rows)
    q_physical_bpw = _bits_per_weight(actual_q_payload_bytes * 8, eligible_weight_count)
    q_metadata_bpw = _bits_per_weight(actual_q_metadata_bytes * 8, eligible_weight_count)
    acc_physical_bpw = _bits_per_weight(actual_acc_payload_bytes * 8, eligible_weight_count)
    scale_bpw = _bits_per_weight(scale_bits, eligible_weight_count)
    inclusive_bpw = float(q_physical_bpw + q_metadata_bpw + acc_physical_bpw + scale_bpw)
    ledger_pass = (
        abs(float(q_physical_bpw) - R4_Q_PHYSICAL_BITS_PER_WEIGHT) <= R4_Q_BPW_TOLERANCE
        and abs(float(acc_physical_bpw) - R5_ACC_PHYSICAL_BITS_PER_WEIGHT) <= R5_ACC_BPW_TOLERANCE
        and float(inclusive_bpw) <= R5_LEDGER_PASS_INCLUSIVE_BPW_CEILING
    )
    return R5PersistentStateBudgetReport(
        schema_version=R5_PERSISTENT_STATE_BUDGET_SCHEMA_VERSION,
        label=R5_PERSISTENT_STATE_BUDGET_LABEL,
        eligible_weight_count=int(eligible_weight_count),
        q_state_count=int(len(qscale_states)),
        accumulator_payload_count=int(len(packed_acc_payloads)),
        r5_q_physical_bits_per_weight=float(q_physical_bpw),
        r5_q_metadata_bits_per_weight=float(q_metadata_bpw),
        r5_acc_physical_bits_per_weight=float(acc_physical_bpw),
        r5_checkpoint_inclusive_physical_bits_per_weight=float(inclusive_bpw),
        r5_actual_q_payload_bytes=int(actual_q_payload_bytes),
        r5_actual_q_metadata_bytes=int(actual_q_metadata_bytes),
        r5_actual_acc_payload_bytes=int(actual_acc_payload_bytes),
        r5_frozen_scale_fp32_bits=int(scale_bits),
        r5_q_packed_content_sha256=str(q_content_sha256),
        r5_acc_packed_content_sha256=str(acc_content_sha256),
        r5_ledger_pass=bool(ledger_pass),
        receipt_statement=R5_W5_BYTEPACKED_DECISION_PARITY_NOT_SUB2_STATEMENT,
    )


def _numel_from_shape(shape: Sequence[int]) -> int:
    numel = 1
    for dim in shape:
        if int(dim) < 0:
            raise ValueError(f"logical_shape dims must be non-negative, got {tuple(shape)}")
        numel *= int(dim)
    return int(numel)


def _validate_q_levels(q_levels: torch.Tensor) -> torch.Tensor:
    if q_levels.dtype != torch.int8:
        raise ValueError(f"q_levels must be torch.int8, got {q_levels.dtype}")
    if q_levels.numel() <= 0:
        raise ValueError("q_levels must be non-empty for persistent-state packing")
    invalid = (q_levels != -1) & (q_levels != 0) & (q_levels != 1)
    if bool(invalid.any().item()):
        raise ValueError("q_levels must contain only ternary int8 levels {-1, 0, +1}")
    return q_levels.contiguous()


def pack_ternary_q_2bit_reference(q_levels: torch.Tensor) -> PackedTernaryQState:
    """Pack q:int8 {-1,0,+1} into actual uint8 bytes using 2-bit codes.

    Codebook: -1 -> 0b00, 0 -> 0b01, +1 -> 0b10. 0b11 is unused and invalid
    for active logical lanes; padding lanes are ignored by unpack.
    """

    q_contig = _validate_q_levels(q_levels)
    logical_numel = int(q_contig.numel())
    padding_values = (-logical_numel) % 4
    codes_i16 = q_contig.flatten().to(torch.int16) + 1
    if padding_values:
        pad = torch.zeros(padding_values, dtype=torch.int16, device=q_contig.device)
        codes_i16 = torch.cat((codes_i16, pad), dim=0)
    quartets = codes_i16.view(-1, 4)
    packed_i16 = (
        quartets[:, 0]
        | (quartets[:, 1] << 2)
        | (quartets[:, 2] << 4)
        | (quartets[:, 3] << 6)
    )
    return PackedTernaryQState(
        packed=packed_i16.to(torch.uint8).contiguous(),
        logical_shape=tuple(int(dim) for dim in q_contig.shape),
        logical_numel=logical_numel,
        padding_values=int(padding_values),
    )


def _validate_packed_state_metadata(state: PackedTernaryQState) -> None:
    _reject_int8_tensor_as_packed_q(state.packed, context="packed q payload")
    if state.format != PACKED_TERNARY_Q_FORMAT:
        raise ValueError(f"packed q format must be {PACKED_TERNARY_Q_FORMAT!r}, got {state.format!r}")
    if state.packed.dtype != torch.uint8:
        raise ValueError(f"packed q payload must be torch.uint8, got {state.packed.dtype}")
    if state.packed.ndim != 1:
        raise ValueError(f"packed q payload must be 1-D bytes, got shape {tuple(state.packed.shape)}")
    if int(state.logical_numel) <= 0:
        raise ValueError("logical_numel must be positive")
    if _numel_from_shape(state.logical_shape) != int(state.logical_numel):
        raise ValueError("logical_shape product must match logical_numel")
    expected_padding = (-int(state.logical_numel)) % 4
    if int(state.padding_values) != expected_padding:
        raise ValueError("padding_values must match logical_numel modulo 4")
    expected_bytes = (int(state.logical_numel) + 3) // 4
    if int(state.packed.numel()) != expected_bytes:
        raise ValueError("packed byte length must equal ceil(logical_numel / 4)")


def unpack_ternary_q_2bit_reference(state: PackedTernaryQState) -> torch.Tensor:
    """Unpack a reference 2-bit ternary q payload back to torch.int8 levels."""

    _validate_packed_state_metadata(state)
    bytes_i16 = state.packed.to(torch.int16)
    codes = torch.stack(
        (
            bytes_i16 & 0b00000011,
            (bytes_i16 >> 2) & 0b00000011,
            (bytes_i16 >> 4) & 0b00000011,
            (bytes_i16 >> 6) & 0b00000011,
        ),
        dim=1,
    ).flatten()
    active_codes = codes[: int(state.logical_numel)]
    if bool((active_codes == 0b11).any().item()):
        raise ValueError("packed q payload contains unused code 0b11 in active logical lanes")
    q_levels = (active_codes.to(torch.int16) - 1).to(torch.int8)
    return q_levels.view(state.logical_shape).contiguous()


def _validate_accumulators(accumulators: Sequence[torch.Tensor], *, eligible_weight_count: int) -> int:
    if len(accumulators) == 0:
        raise ValueError("at least one int16 accumulator tensor is required for 3-ledger accounting")
    total = 0
    for acc in accumulators:
        if acc.dtype != torch.int16:
            raise ValueError(f"accumulators must be torch.int16, got {acc.dtype}")
        total += int(acc.numel())
    if total != int(eligible_weight_count):
        raise ValueError(
            "sum(accumulator.numel()) must match eligible q entries; "
            f"got accumulators={total}, eligible={eligible_weight_count}"
        )
    return total


def _bits_per_weight(bits: int | float, denominator: int) -> float:
    return float(bits) / float(denominator)


def measure_persistent_state_budget(
    qscale_states: Sequence[QScaleWeightState],
    accumulator_tensors: Sequence[torch.Tensor],
    *,
    target_bits_per_weight: float = TARGET_PHYSICAL_BITS_PER_WEIGHT,
) -> PersistentStateBudgetReport:
    """Measure current and q-packed persistent-state ledgers.

    Primary target accounting is inclusive physical storage:
    Ledger A q packed data + padding + packed-state metadata,
    Ledger B int16 vote/accumulators, and Ledger C frozen FP32 scales.
    """

    if len(qscale_states) == 0:
        raise ValueError("at least one qscale state is required for persistent-state accounting")
    if target_bits_per_weight <= 0.0:
        raise ValueError("target_bits_per_weight must be > 0")

    eligible_weight_count = 0
    current_q_bits = 0
    packed_q_data_bits = 0
    packed_q_padding_bits = 0
    packed_q_metadata_bits = 0
    scale_bits = 0

    for qscale_state in qscale_states:
        q_levels, scale, _ = validate_qscale_weight_state(qscale_state)
        del scale
        packed = pack_ternary_q_2bit_reference(q_levels)
        eligible_weight_count += int(q_levels.numel())
        current_q_bits += int(q_levels.numel() * 8)
        packed_q_data_bits += packed.packed_data_bits
        packed_q_padding_bits += packed.padding_bits
        packed_q_metadata_bits += packed.metadata_bits
        scale_bits += 32

    _validate_accumulators(accumulator_tensors, eligible_weight_count=eligible_weight_count)
    acc_bits = int(sum(int(acc.numel()) * 16 for acc in accumulator_tensors))
    packed_q_total_bits = int(packed_q_data_bits + packed_q_metadata_bits)
    current_inclusive_bits = int(current_q_bits + acc_bits + scale_bits)
    packed_inclusive_bits = int(packed_q_total_bits + acc_bits + scale_bits)
    scale_excluded_bits = int(packed_q_total_bits + acc_bits)
    acc_scale_excluded_bits = int(packed_q_total_bits)

    q_packed_total_bpw = _bits_per_weight(packed_q_total_bits, eligible_weight_count)
    scale_bpw = _bits_per_weight(scale_bits, eligible_weight_count)
    packed_inclusive_bpw = _bits_per_weight(packed_inclusive_bits, eligible_weight_count)
    target_achieved = packed_inclusive_bpw < float(target_bits_per_weight)
    required_acc_physical = float(target_bits_per_weight) - q_packed_total_bpw - scale_bpw
    required_acc_effective = float(target_bits_per_weight) - EFFECTIVE_FORWARD_TERNARY_BITS - scale_bpw

    ledger_bits = {
        "q_packed_data_plus_metadata": packed_q_total_bits,
        "acc_int16": acc_bits,
        "frozen_scale_fp32": scale_bits,
    }
    dominant_ledger = max(ledger_bits, key=ledger_bits.get)

    report = PersistentStateBudgetReport(
        schema_version=PERSISTENT_STATE_BUDGET_SCHEMA_VERSION,
        label=PERSISTENT_STATE_BUDGET_LABEL,
        target_basis=INCLUSIVE_3LEDGER_TARGET_BASIS,
        target_bits_per_weight=float(target_bits_per_weight),
        target_achieved=target_achieved,
        eligible_weight_count=int(eligible_weight_count),
        q_state_count=int(len(qscale_states)),
        accumulator_tensor_count=int(len(accumulator_tensors)),
        current_q_int8_bits=int(current_q_bits),
        packed_q_data_bits=int(packed_q_data_bits),
        packed_q_padding_bits=int(packed_q_padding_bits),
        packed_q_metadata_bits=int(packed_q_metadata_bits),
        packed_q_total_bits=int(packed_q_total_bits),
        acc_int16_bits=int(acc_bits),
        frozen_scale_fp32_bits=int(scale_bits),
        current_inclusive_physical_bits=int(current_inclusive_bits),
        packed_inclusive_physical_bits=int(packed_inclusive_bits),
        scale_excluded_diagnostic_bits=int(scale_excluded_bits),
        acc_scale_excluded_diagnostic_bits=int(acc_scale_excluded_bits),
        current_inclusive_physical_bits_per_weight=_bits_per_weight(current_inclusive_bits, eligible_weight_count),
        packed_inclusive_physical_bits_per_weight=packed_inclusive_bpw,
        q_int8_bits_per_weight=_bits_per_weight(current_q_bits, eligible_weight_count),
        q_packed_data_bits_per_weight=_bits_per_weight(packed_q_data_bits, eligible_weight_count),
        q_packed_padding_bits_per_weight=_bits_per_weight(packed_q_padding_bits, eligible_weight_count),
        q_packed_metadata_bits_per_weight=_bits_per_weight(packed_q_metadata_bits, eligible_weight_count),
        q_packed_total_bits_per_weight=q_packed_total_bpw,
        acc_int16_bits_per_weight=_bits_per_weight(acc_bits, eligible_weight_count),
        frozen_scale_fp32_bits_per_weight=scale_bpw,
        scale_excluded_diagnostic_bits_per_weight=_bits_per_weight(scale_excluded_bits, eligible_weight_count),
        acc_scale_excluded_diagnostic_bits_per_weight=_bits_per_weight(
            acc_scale_excluded_bits,
            eligible_weight_count,
        ),
        q_effective_forward_entropy_bits_per_weight=EFFECTIVE_FORWARD_TERNARY_BITS,
        required_acc_bits_per_weight_for_sub2_physical_q_with_scale_and_metadata=required_acc_physical,
        required_acc_bits_per_weight_for_sub2_effective_q_with_scale=required_acc_effective,
        dominant_ledger=dominant_ledger,
        track_b_status=TRACK_B_DESIGN_ONLY_STATUS,
        receipt_statement=PHYSICAL_SUB2_NOT_ACHIEVED_STATEMENT,
    )
    validate_persistent_state_budget_report(report)
    return report


def validate_persistent_state_budget_report(report: PersistentStateBudgetReport) -> None:
    """Reject false sub-2 claims derived from q-only or diagnostic ledgers."""

    if report.target_basis != INCLUSIVE_3LEDGER_TARGET_BASIS:
        raise ValueError("target_achieved must use the inclusive 3-ledger physical target basis")
    recomputed_target = report.packed_inclusive_physical_bits_per_weight < report.target_bits_per_weight
    if bool(report.target_achieved) != bool(recomputed_target):
        raise ValueError("target_achieved must be computed from inclusive 3-ledger physical bits/weight")
    if (
        report.acc_int16_bits > 0
        and report.packed_inclusive_physical_bits_per_weight >= report.target_bits_per_weight
        and report.target_achieved
    ):
        raise ValueError("q-only packing cannot be labeled target-achieved while int16 accumulators remain")
    if report.label == PERSISTENT_STATE_BUDGET_LABEL and report.target_achieved:
        raise ValueError("over-target persistent-state budget label cannot carry a target-achieved claim")
    if report.receipt_statement != PHYSICAL_SUB2_NOT_ACHIEVED_STATEMENT:
        raise ValueError("receipt statement must preserve the not-achieved int16-accumulator caveat")
