"""C1.1a base-3 q-entropy packing and physical q-code accounting.

This is a CPU/static reference layer for q storage only. It packs ternary
q:int8 levels into base-3 bytes, measures the q-code ledger from actual byte
counts, and explicitly keeps the result out of accumulator-compression progress.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Sequence

import torch

from calm.hrm_text_158.native_full_stack.accumulator_compression import (
    CandidateAssessment,
    CandidateClassification,
    candidate_assessment,
    required_decision_dimension_names,
    validate_candidate_assessment,
)
from calm.hrm_text_158.native_full_stack.persistent_state_budget import (
    INCLUSIVE_3LEDGER_TARGET_BASIS,
    PACKED_TERNARY_METADATA_BYTES_PER_DIM,
    PACKED_TERNARY_METADATA_HEADER_BYTES,
    PHYSICAL_SUB2_NOT_ACHIEVED_STATEMENT,
    TARGET_PHYSICAL_BITS_PER_WEIGHT,
)
from calm.hrm_text_158.native_full_stack.qscale_linear import (
    QScaleWeightState,
    validate_qscale_weight_state,
)


BASE3_Q_ENTROPY_SCHEMA_VERSION = "hrm_text_158_q_entropy_packing/v0.base3_5perbyte_static"
BASE3_Q_FORMAT = "packed_base3_5ternary_uint8_reference"
BASE3_Q_ENTROPY_LABEL = "q_entropy_base3_5perbyte_storage_only_not_accumulator_progress"
BASE3_Q_STORAGE_ORTHOGONALITY_LABEL = "q_storage_bit_exact_not_accumulator_candidate"
BASE3_GROUP_SIZE = 5
BASE3_TRIT_RADIX = 3
BASE3_FULL_GROUP_CODE_COUNT = BASE3_TRIT_RADIX**BASE3_GROUP_SIZE
BASE3_UNUSED_UINT8_CODES_PER_FULL_GROUP = 256 - BASE3_FULL_GROUP_CODE_COUNT
BASE3_TRIT_WEIGHTS = (1, 3, 9, 27, 81)
BASE3_FIXED_PAYLOAD_BITS_PER_WEIGHT = 8.0 / BASE3_GROUP_SIZE
BASE3_EFFECTIVE_TERNARY_ENTROPY_BITS_PER_WEIGHT = math.log2(3.0)
BASE3_Q_STORAGE_ONLY_STATUS = (
    "q_storage_only_not_accumulator_candidate_not_c2_accumulator_progress"
)
BASE3_Q_METADATA_HEADER_BYTES = PACKED_TERNARY_METADATA_HEADER_BYTES
BASE3_Q_METADATA_BYTES_PER_DIM = PACKED_TERNARY_METADATA_BYTES_PER_DIM
BASE3_Q_METADATA_SIDECAR_FIELDS = (
    "logical_numel_uint64",
    "compact_schema_format_version_group_size_remainder_padding_table_id_uint64",
    "logical_shape_int64_per_dim",
)
BASE3_Q_STORAGE_CANDIDATE_NAME = "base3_5perbyte_q_storage_not_accumulator_candidate"


@dataclass(frozen=True)
class PackedBase3TernaryQState:
    """Reference base-3 ternary q payload plus compact logical metadata."""

    packed: torch.Tensor
    logical_shape: tuple[int, ...]
    logical_numel: int
    padding_values: int
    group_size: int = BASE3_GROUP_SIZE
    format: str = BASE3_Q_FORMAT

    @property
    def packed_data_bytes(self) -> int:
        return int(self.packed.numel() * self.packed.element_size())

    @property
    def packed_data_bits(self) -> int:
        return int(self.packed_data_bytes * 8)

    @property
    def ideal_5perbyte_payload_bits(self) -> float:
        return float(self.logical_numel) * BASE3_FIXED_PAYLOAD_BITS_PER_WEIGHT

    @property
    def payload_padding_bits_over_5perbyte(self) -> float:
        return float(self.packed_data_bits) - self.ideal_5perbyte_payload_bits

    @property
    def ideal_log2_ternary_entropy_bits(self) -> float:
        return float(self.logical_numel) * BASE3_EFFECTIVE_TERNARY_ENTROPY_BITS_PER_WEIGHT

    @property
    def payload_over_log2_3_bits(self) -> float:
        return float(self.packed_data_bits) - self.ideal_log2_ternary_entropy_bits

    @property
    def metadata_bytes(self) -> int:
        return int(BASE3_Q_METADATA_HEADER_BYTES + BASE3_Q_METADATA_BYTES_PER_DIM * len(self.logical_shape))

    @property
    def metadata_bits(self) -> int:
        return int(self.metadata_bytes * 8)


@dataclass(frozen=True)
class Base3QEntropyLedgerRow:
    """Compact physical ledger for one q-code storage regime."""

    schema_version: str
    label: str
    regime_name: str
    format: str
    target_basis: str
    target_bits_per_weight: float
    target_achieved: bool
    claimable_physical_sub2: bool
    storage_only_status: str
    not_accumulator_candidate: bool
    not_c2_accumulator_compression_progress: bool
    eligible_weight_count: int
    q_state_count: int
    accumulator_tensor_count: int
    packed_q_data_bits: int
    packed_q_padding_values: int
    packed_q_padding_bits_over_5perbyte: float
    packed_q_payload_over_log2_3_bits: float
    packed_q_metadata_bits: int
    packed_q_total_bits: int
    accumulator_bits: float
    frozen_scale_fp32_bits: int
    packed_inclusive_physical_bits: float
    q_packed_data_bits_per_weight: float
    q_packed_padding_bits_over_5perbyte_per_weight: float
    q_packed_payload_over_log2_3_bits_per_weight: float
    q_packed_metadata_bits_per_weight: float
    q_packed_total_bits_per_weight: float
    accumulator_bits_per_weight: float
    frozen_scale_fp32_bits_per_weight: float
    packed_inclusive_physical_bits_per_weight: float
    remaining_accumulator_budget_bits_per_weight: float
    base3_fixed_payload_bits_per_weight: float
    effective_ternary_entropy_bits_per_weight: float
    dominant_ledger: str
    receipt_statement: str
    metadata_sidecar_fields: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["metadata_sidecar_fields"] = list(self.metadata_sidecar_fields)
        return out


@dataclass(frozen=True)
class QStorageOrthogonalityReport:
    """Executable classification proving q storage is not accumulator progress."""

    schema_version: str
    label: str
    candidate_assessment: CandidateAssessment
    storage_only_status: str
    bit_exact_q_storage: bool
    not_accumulator_candidate: bool
    not_c2_accumulator_compression_progress: bool
    no_vote_state_compression: bool
    qscale_boundary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "label": self.label,
            "candidate_assessment": self.candidate_assessment.to_dict(),
            "storage_only_status": self.storage_only_status,
            "bit_exact_q_storage": self.bit_exact_q_storage,
            "not_accumulator_candidate": self.not_accumulator_candidate,
            "not_c2_accumulator_compression_progress": self.not_c2_accumulator_compression_progress,
            "no_vote_state_compression": self.no_vote_state_compression,
            "qscale_boundary": self.qscale_boundary,
        }


@dataclass(frozen=True)
class Base3QEntropyCompactReport:
    """Compact C1.1a report: codec, ledger rows, orthogonality, non-claims."""

    schema_version: str
    label: str
    codec_summary: dict[str, Any]
    ledger_rows: tuple[Base3QEntropyLedgerRow, ...]
    orthogonality: QStorageOrthogonalityReport
    non_claims: tuple[str, ...]
    raw_arrays_included: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "label": self.label,
            "codec_summary": dict(self.codec_summary),
            "ledger_rows": [row.to_dict() for row in self.ledger_rows],
            "orthogonality": self.orthogonality.to_dict(),
            "non_claims": list(self.non_claims),
            "raw_arrays_included": bool(self.raw_arrays_included),
        }


def _numel_from_shape(shape: Sequence[int]) -> int:
    numel = 1
    for dim in shape:
        dim_i = int(dim)
        if dim_i < 0:
            raise ValueError(f"logical_shape dims must be non-negative, got {tuple(shape)}")
        numel *= dim_i
    return int(numel)


def _numel_from_budget_shape(shape: Sequence[int]) -> int:
    shape_t = tuple(int(dim) for dim in shape)
    if len(shape_t) == 0:
        raise ValueError("logical_shape must have at least one dimension for q-code budget rows")
    numel = 1
    for dim in shape_t:
        if dim <= 0:
            raise ValueError(
                "zero-length q codec cases are not bits-per-weight budget rows; "
                f"got logical_shape={shape_t}"
            )
        numel *= dim
    return int(numel)


def _bits_per_weight(bits: int | float, denominator: int) -> float:
    if int(denominator) <= 0:
        raise ValueError("zero-length q codec cases are not bits-per-weight budget rows")
    return float(bits) / float(denominator)


def _metadata_bits_for_shape(logical_shape: Sequence[int]) -> int:
    shape_t = tuple(int(dim) for dim in logical_shape)
    _numel_from_budget_shape(shape_t)
    metadata_bytes = BASE3_Q_METADATA_HEADER_BYTES + BASE3_Q_METADATA_BYTES_PER_DIM * len(shape_t)
    return int(metadata_bytes * 8)


def _validate_q_levels(q_levels: torch.Tensor) -> torch.Tensor:
    if q_levels.dtype != torch.int8:
        raise ValueError(f"q_levels must be torch.int8, got {q_levels.dtype}")
    invalid = (q_levels != -1) & (q_levels != 0) & (q_levels != 1)
    if bool(invalid.any().item()):
        raise ValueError("q_levels must contain only ternary int8 levels {-1, 0, +1}")
    return q_levels.contiguous()


def pack_ternary_q_base3_5perbyte_reference(q_levels: torch.Tensor) -> PackedBase3TernaryQState:
    """Pack q:int8 {-1,0,+1} into base-3 uint8 payload bytes.

    Codebook: -1 -> trit 0, 0 -> trit 1, +1 -> trit 2. Five trits are stored
    little-endian in one byte, leaving byte codes 243..255 unused for full
    groups. Final partial-group padding trits are encoded as zero and must
    remain zero when decoded.
    """

    q_contig = _validate_q_levels(q_levels)
    logical_numel = int(q_contig.numel())
    padding_values = (-logical_numel) % BASE3_GROUP_SIZE
    trits_i16 = q_contig.flatten().to(torch.int16) + 1
    if padding_values:
        pad = torch.zeros(padding_values, dtype=torch.int16, device=q_contig.device)
        trits_i16 = torch.cat((trits_i16, pad), dim=0)
    if trits_i16.numel() == 0:
        packed = torch.empty(0, dtype=torch.uint8, device=q_contig.device)
    else:
        groups = trits_i16.view(-1, BASE3_GROUP_SIZE)
        weights = torch.tensor(BASE3_TRIT_WEIGHTS, dtype=torch.int16, device=q_contig.device)
        packed = (groups * weights).sum(dim=1).to(torch.uint8).contiguous()
    return PackedBase3TernaryQState(
        packed=packed,
        logical_shape=tuple(int(dim) for dim in q_contig.shape),
        logical_numel=logical_numel,
        padding_values=int(padding_values),
    )


def _validate_packed_state_metadata(state: PackedBase3TernaryQState) -> None:
    if state.format != BASE3_Q_FORMAT:
        raise ValueError(f"packed q format must be {BASE3_Q_FORMAT!r}, got {state.format!r}")
    if state.group_size != BASE3_GROUP_SIZE:
        raise ValueError(f"base-3 q group_size must be {BASE3_GROUP_SIZE}, got {state.group_size}")
    if state.packed.dtype != torch.uint8:
        raise ValueError(f"packed q payload must be torch.uint8, got {state.packed.dtype}")
    if state.packed.ndim != 1:
        raise ValueError(f"packed q payload must be 1-D bytes, got shape {tuple(state.packed.shape)}")
    if int(state.logical_numel) < 0:
        raise ValueError("logical_numel must be non-negative")
    if _numel_from_shape(state.logical_shape) != int(state.logical_numel):
        raise ValueError("logical_shape product must match logical_numel")
    expected_padding = (-int(state.logical_numel)) % BASE3_GROUP_SIZE
    if int(state.padding_values) != expected_padding:
        raise ValueError("padding_values must match logical_numel modulo 5")
    expected_bytes = (int(state.logical_numel) + BASE3_GROUP_SIZE - 1) // BASE3_GROUP_SIZE
    if int(state.packed.numel()) != expected_bytes:
        raise ValueError("packed byte length must equal ceil(logical_numel / 5)")


def _validate_base3_codes(state: PackedBase3TernaryQState, codes_i16: torch.Tensor) -> None:
    logical_numel = int(state.logical_numel)
    if logical_numel == 0:
        return
    full_group_count = logical_numel // BASE3_GROUP_SIZE
    remainder = logical_numel % BASE3_GROUP_SIZE
    full_codes = codes_i16 if remainder == 0 else codes_i16[:full_group_count]
    if full_codes.numel() and bool((full_codes >= BASE3_FULL_GROUP_CODE_COUNT).any().item()):
        raise ValueError("packed q payload contains unused base-3 byte code >=243 in a full group")
    if remainder:
        final_code = int(codes_i16[-1].item())
        if final_code >= BASE3_FULL_GROUP_CODE_COUNT:
            raise ValueError("packed q payload contains unused base-3 byte code >=243 in final partial group")
        if final_code >= BASE3_TRIT_RADIX**remainder:
            raise ValueError("packed q payload contains non-zero padded trits in final partial group")


def unpack_ternary_q_base3_5perbyte_reference(state: PackedBase3TernaryQState) -> torch.Tensor:
    """Unpack a base-3 ternary q payload back to torch.int8 levels."""

    _validate_packed_state_metadata(state)
    if int(state.logical_numel) == 0:
        return torch.empty(state.logical_shape, dtype=torch.int8, device=state.packed.device)
    codes_i16 = state.packed.to(torch.int16)
    _validate_base3_codes(state, codes_i16)
    trits = torch.stack(
        tuple((codes_i16 // weight) % BASE3_TRIT_RADIX for weight in BASE3_TRIT_WEIGHTS),
        dim=1,
    ).flatten()
    active_trits = trits[: int(state.logical_numel)]
    q_levels = (active_trits.to(torch.int16) - 1).to(torch.int8)
    return q_levels.view(state.logical_shape).contiguous()


def base3_payload_bits_and_padding(logical_numel: int) -> tuple[int, int, float, float]:
    """Return actual base-3 payload bits plus diagnostic padding/entropy overhead."""

    numel = int(logical_numel)
    if numel <= 0:
        raise ValueError(
            "zero-length q codec cases are not bits-per-weight budget rows; "
            f"got logical_numel={logical_numel}"
        )
    payload_bits = int(math.ceil(numel / float(BASE3_GROUP_SIZE)) * 8)
    padding_values = int((-numel) % BASE3_GROUP_SIZE)
    padding_bits_over_5perbyte = float(payload_bits) - (float(numel) * BASE3_FIXED_PAYLOAD_BITS_PER_WEIGHT)
    payload_over_log2_3_bits = (
        float(payload_bits) - (float(numel) * BASE3_EFFECTIVE_TERNARY_ENTROPY_BITS_PER_WEIGHT)
    )
    return payload_bits, padding_values, padding_bits_over_5perbyte, payload_over_log2_3_bits


def _validate_accumulators(accumulators: Sequence[torch.Tensor], *, eligible_weight_count: int) -> None:
    if len(accumulators) == 0:
        raise ValueError("at least one int16 accumulator tensor is required for inclusive accounting")
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


def _build_base3_ledger_row(
    *,
    regime_name: str,
    eligible_weight_count: int,
    q_state_count: int,
    accumulator_tensor_count: int,
    packed_q_data_bits: int,
    packed_q_padding_values: int,
    packed_q_padding_bits_over_5perbyte: float,
    packed_q_payload_over_log2_3_bits: float,
    packed_q_metadata_bits: int,
    frozen_scale_fp32_bits: int,
    accumulator_bits_per_weight: float,
    target_bits_per_weight: float = TARGET_PHYSICAL_BITS_PER_WEIGHT,
) -> Base3QEntropyLedgerRow:
    if not regime_name:
        raise ValueError("regime_name must be non-empty")
    eligible = int(eligible_weight_count)
    if eligible <= 0:
        raise ValueError("zero-length q codec cases are not bits-per-weight budget rows")
    if int(q_state_count) <= 0:
        raise ValueError("q_state_count must be > 0")
    if int(accumulator_tensor_count) <= 0:
        raise ValueError("accumulator_tensor_count must be > 0")
    if target_bits_per_weight <= 0.0:
        raise ValueError("target_bits_per_weight must be > 0")
    if packed_q_data_bits < 0 or packed_q_metadata_bits < 0 or frozen_scale_fp32_bits < 0:
        raise ValueError("ledger bit counts must be non-negative")
    if packed_q_padding_values < 0:
        raise ValueError("packed_q_padding_values must be >= 0")
    if accumulator_bits_per_weight < 0.0:
        raise ValueError("accumulator_bits_per_weight must be >= 0")

    packed_q_total_bits = int(packed_q_data_bits + packed_q_metadata_bits)
    q_total_bpw = _bits_per_weight(packed_q_total_bits, eligible)
    scale_bpw = _bits_per_weight(frozen_scale_fp32_bits, eligible)
    accumulator_bits = float(accumulator_bits_per_weight) * float(eligible)
    inclusive_bits = float(packed_q_total_bits) + float(frozen_scale_fp32_bits) + accumulator_bits
    inclusive_bpw = _bits_per_weight(inclusive_bits, eligible)
    remaining_acc_budget = float(target_bits_per_weight) - q_total_bpw - scale_bpw
    target_achieved = inclusive_bpw < float(target_bits_per_weight)

    ledger_bits = {
        "q_packed_data_plus_metadata": float(packed_q_total_bits),
        "accumulator": accumulator_bits,
        "frozen_scale_fp32": float(frozen_scale_fp32_bits),
    }
    dominant_ledger = max(ledger_bits, key=ledger_bits.get)

    row = Base3QEntropyLedgerRow(
        schema_version=BASE3_Q_ENTROPY_SCHEMA_VERSION,
        label=BASE3_Q_ENTROPY_LABEL,
        regime_name=regime_name,
        format=BASE3_Q_FORMAT,
        target_basis=INCLUSIVE_3LEDGER_TARGET_BASIS,
        target_bits_per_weight=float(target_bits_per_weight),
        target_achieved=bool(target_achieved),
        claimable_physical_sub2=bool(target_achieved),
        storage_only_status=BASE3_Q_STORAGE_ONLY_STATUS,
        not_accumulator_candidate=True,
        not_c2_accumulator_compression_progress=True,
        eligible_weight_count=eligible,
        q_state_count=int(q_state_count),
        accumulator_tensor_count=int(accumulator_tensor_count),
        packed_q_data_bits=int(packed_q_data_bits),
        packed_q_padding_values=int(packed_q_padding_values),
        packed_q_padding_bits_over_5perbyte=float(packed_q_padding_bits_over_5perbyte),
        packed_q_payload_over_log2_3_bits=float(packed_q_payload_over_log2_3_bits),
        packed_q_metadata_bits=int(packed_q_metadata_bits),
        packed_q_total_bits=packed_q_total_bits,
        accumulator_bits=accumulator_bits,
        frozen_scale_fp32_bits=int(frozen_scale_fp32_bits),
        packed_inclusive_physical_bits=inclusive_bits,
        q_packed_data_bits_per_weight=_bits_per_weight(packed_q_data_bits, eligible),
        q_packed_padding_bits_over_5perbyte_per_weight=_bits_per_weight(
            packed_q_padding_bits_over_5perbyte,
            eligible,
        ),
        q_packed_payload_over_log2_3_bits_per_weight=_bits_per_weight(
            packed_q_payload_over_log2_3_bits,
            eligible,
        ),
        q_packed_metadata_bits_per_weight=_bits_per_weight(packed_q_metadata_bits, eligible),
        q_packed_total_bits_per_weight=q_total_bpw,
        accumulator_bits_per_weight=float(accumulator_bits_per_weight),
        frozen_scale_fp32_bits_per_weight=scale_bpw,
        packed_inclusive_physical_bits_per_weight=inclusive_bpw,
        remaining_accumulator_budget_bits_per_weight=remaining_acc_budget,
        base3_fixed_payload_bits_per_weight=BASE3_FIXED_PAYLOAD_BITS_PER_WEIGHT,
        effective_ternary_entropy_bits_per_weight=BASE3_EFFECTIVE_TERNARY_ENTROPY_BITS_PER_WEIGHT,
        dominant_ledger=dominant_ledger,
        receipt_statement=PHYSICAL_SUB2_NOT_ACHIEVED_STATEMENT,
        metadata_sidecar_fields=BASE3_Q_METADATA_SIDECAR_FIELDS,
    )
    validate_base3_q_entropy_ledger(row)
    return row


def base3_q_entropy_ledger_for_shapes(
    *,
    regime_name: str,
    logical_shapes: Sequence[Sequence[int]],
    scale_count: int,
    accumulator_bits_per_weight: float = 16.0,
    target_bits_per_weight: float = TARGET_PHYSICAL_BITS_PER_WEIGHT,
) -> Base3QEntropyLedgerRow:
    """Build a q-code ledger row from shape-derived actual byte counts."""

    if not logical_shapes:
        raise ValueError("logical_shapes must be non-empty")
    if int(scale_count) < 0:
        raise ValueError("scale_count must be >= 0")

    eligible = 0
    data_bits = 0
    padding_values = 0
    padding_bits_over_5perbyte = 0.0
    payload_over_log2_3_bits = 0.0
    metadata_bits = 0
    for shape in logical_shapes:
        shape_t = tuple(int(dim) for dim in shape)
        numel = _numel_from_budget_shape(shape_t)
        payload_bits, tensor_padding_values, tensor_padding_bits, tensor_over_entropy_bits = (
            base3_payload_bits_and_padding(numel)
        )
        eligible += numel
        data_bits += payload_bits
        padding_values += tensor_padding_values
        padding_bits_over_5perbyte += tensor_padding_bits
        payload_over_log2_3_bits += tensor_over_entropy_bits
        metadata_bits += _metadata_bits_for_shape(shape_t)

    return _build_base3_ledger_row(
        regime_name=regime_name,
        eligible_weight_count=eligible,
        q_state_count=len(tuple(logical_shapes)),
        accumulator_tensor_count=len(tuple(logical_shapes)),
        packed_q_data_bits=data_bits,
        packed_q_padding_values=padding_values,
        packed_q_padding_bits_over_5perbyte=padding_bits_over_5perbyte,
        packed_q_payload_over_log2_3_bits=payload_over_log2_3_bits,
        packed_q_metadata_bits=metadata_bits,
        frozen_scale_fp32_bits=int(scale_count) * 32,
        accumulator_bits_per_weight=float(accumulator_bits_per_weight),
        target_bits_per_weight=float(target_bits_per_weight),
    )


def measure_base3_q_entropy_budget(
    qscale_states: Sequence[QScaleWeightState],
    accumulator_tensors: Sequence[torch.Tensor],
    *,
    regime_name: str = "measured_qscale_states_base3_5perbyte",
    target_bits_per_weight: float = TARGET_PHYSICAL_BITS_PER_WEIGHT,
) -> Base3QEntropyLedgerRow:
    """Measure a base-3 q-code ledger from actual packed qscale payload bytes."""

    if len(qscale_states) == 0:
        raise ValueError("at least one qscale state is required for q-entropy accounting")
    eligible = 0
    data_bits = 0
    padding_values = 0
    padding_bits_over_5perbyte = 0.0
    payload_over_log2_3_bits = 0.0
    metadata_bits = 0
    scale_bits = 0
    for state in qscale_states:
        q_levels, scale, _ = validate_qscale_weight_state(state)
        del scale
        packed = pack_ternary_q_base3_5perbyte_reference(q_levels)
        eligible += int(q_levels.numel())
        data_bits += packed.packed_data_bits
        padding_values += packed.padding_values
        padding_bits_over_5perbyte += packed.payload_padding_bits_over_5perbyte
        payload_over_log2_3_bits += packed.payload_over_log2_3_bits
        metadata_bits += packed.metadata_bits
        scale_bits += 32

    if eligible <= 0:
        raise ValueError("zero-length q codec cases are not bits-per-weight budget rows")
    _validate_accumulators(accumulator_tensors, eligible_weight_count=eligible)
    accumulator_bits = sum(int(acc.numel()) * 16 for acc in accumulator_tensors)
    return _build_base3_ledger_row(
        regime_name=regime_name,
        eligible_weight_count=eligible,
        q_state_count=len(qscale_states),
        accumulator_tensor_count=len(accumulator_tensors),
        packed_q_data_bits=data_bits,
        packed_q_padding_values=padding_values,
        packed_q_padding_bits_over_5perbyte=padding_bits_over_5perbyte,
        packed_q_payload_over_log2_3_bits=payload_over_log2_3_bits,
        packed_q_metadata_bits=metadata_bits,
        frozen_scale_fp32_bits=scale_bits,
        accumulator_bits_per_weight=_bits_per_weight(accumulator_bits, eligible),
        target_bits_per_weight=float(target_bits_per_weight),
    )


def default_base3_q_entropy_ledger_table() -> tuple[Base3QEntropyLedgerRow, ...]:
    """Return the named C1.1a q-code ledger regimes."""

    realistic_shape = (4096, 4096)
    return (
        base3_q_entropy_ledger_for_shapes(
            regime_name="tiny_two_projection_fixture_base3_q",
            logical_shapes=((8, 16), (4, 8)),
            scale_count=2,
        ),
        base3_q_entropy_ledger_for_shapes(
            regime_name="non_multiple_of_five_len6_base3_q",
            logical_shapes=((6,),),
            scale_count=1,
        ),
        base3_q_entropy_ledger_for_shapes(
            regime_name="prior_large_fixture_base3_q",
            logical_shapes=((128, 128),),
            scale_count=1,
        ),
        base3_q_entropy_ledger_for_shapes(
            regime_name="illustrative_4096x4096_one_tensor_one_scale_base3_q",
            logical_shapes=(realistic_shape,),
            scale_count=1,
        ),
        base3_q_entropy_ledger_for_shapes(
            regime_name="illustrative_4096x4096_one_tensor_per_row_scale_base3_q",
            logical_shapes=(realistic_shape,),
            scale_count=realistic_shape[0],
        ),
    )


def validate_base3_q_entropy_ledger(
    row: Base3QEntropyLedgerRow,
    *,
    claimed_physical_sub2_achieved: bool = False,
    claimed_q_payload_sub2_achieved: bool = False,
    claimed_scale_acc_excluded_sub2_achieved: bool = False,
) -> None:
    """Reject q-only, scale-excluded, or accumulator-excluded sub-2 claims."""

    eligible = int(row.eligible_weight_count)
    if eligible <= 0:
        raise ValueError("zero-length q codec cases are not bits-per-weight budget rows")
    if row.schema_version != BASE3_Q_ENTROPY_SCHEMA_VERSION:
        raise ValueError("unexpected q-entropy ledger schema version")
    if row.label != BASE3_Q_ENTROPY_LABEL:
        raise ValueError("unexpected q-entropy ledger label")
    if row.format != BASE3_Q_FORMAT:
        raise ValueError("unexpected q-entropy packed format")
    if row.target_basis != INCLUSIVE_3LEDGER_TARGET_BASIS:
        raise ValueError("target_achieved must use inclusive 3-ledger physical target basis")
    if row.storage_only_status != BASE3_Q_STORAGE_ONLY_STATUS:
        raise ValueError("q-entropy ledger must preserve the storage-only status")
    if not row.not_accumulator_candidate or not row.not_c2_accumulator_compression_progress:
        raise ValueError("q-storage ledger cannot be reported as accumulator compression progress")

    recomputed_q_total_bits = int(row.packed_q_data_bits + row.packed_q_metadata_bits)
    if int(row.packed_q_total_bits) != recomputed_q_total_bits:
        raise ValueError("packed_q_total_bits must be packed payload data plus metadata")
    recomputed_data_bpw = _bits_per_weight(row.packed_q_data_bits, eligible)
    if not math.isclose(row.q_packed_data_bits_per_weight, recomputed_data_bpw, abs_tol=1e-12):
        raise ValueError("q data bits/weight must be computed from actual packed bytes")
    recomputed_metadata_bpw = _bits_per_weight(row.packed_q_metadata_bits, eligible)
    if not math.isclose(row.q_packed_metadata_bits_per_weight, recomputed_metadata_bpw, abs_tol=1e-12):
        raise ValueError("q metadata bits/weight must be computed from sidecar metadata bits")
    recomputed_q_total_bpw = _bits_per_weight(row.packed_q_total_bits, eligible)
    if not math.isclose(row.q_packed_total_bits_per_weight, recomputed_q_total_bpw, abs_tol=1e-12):
        raise ValueError("q total bits/weight must be packed payload plus metadata")
    recomputed_scale_bpw = _bits_per_weight(row.frozen_scale_fp32_bits, eligible)
    if not math.isclose(row.frozen_scale_fp32_bits_per_weight, recomputed_scale_bpw, abs_tol=1e-12):
        raise ValueError("scale bits/weight must be computed from frozen FP32 scale bits")
    recomputed_remaining = (
        row.target_bits_per_weight - row.q_packed_total_bits_per_weight - row.frozen_scale_fp32_bits_per_weight
    )
    if not math.isclose(
        row.remaining_accumulator_budget_bits_per_weight,
        recomputed_remaining,
        abs_tol=1e-12,
    ):
        raise ValueError("remaining accumulator budget must match target - q_total - scale")
    recomputed_inclusive = (
        row.q_packed_total_bits_per_weight
        + row.frozen_scale_fp32_bits_per_weight
        + row.accumulator_bits_per_weight
    )
    if not math.isclose(
        row.packed_inclusive_physical_bits_per_weight,
        recomputed_inclusive,
        abs_tol=1e-12,
    ):
        raise ValueError("inclusive physical bits/weight must include q, scale, and accumulator")
    recomputed_target = recomputed_inclusive < row.target_bits_per_weight
    if bool(row.target_achieved) != bool(recomputed_target):
        raise ValueError("target flag must be computed from inclusive physical q+scale+acc ledger")
    if bool(row.claimable_physical_sub2) != bool(recomputed_target):
        raise ValueError("claimable physical sub-2 must match the inclusive physical target flag")
    if row.receipt_statement != PHYSICAL_SUB2_NOT_ACHIEVED_STATEMENT and not row.target_achieved:
        raise ValueError("over-target rows must preserve the int16-accumulator caveat")

    if claimed_physical_sub2_achieved and not row.claimable_physical_sub2:
        raise ValueError("physical sub-2 claim is not allowed while the inclusive ledger remains over target")
    if claimed_q_payload_sub2_achieved and not row.claimable_physical_sub2:
        raise ValueError("q-payload sub-2 is storage-only and cannot claim inclusive physical sub-2")
    if claimed_scale_acc_excluded_sub2_achieved and not row.claimable_physical_sub2:
        raise ValueError("scale/acc-excluded diagnostics cannot claim inclusive physical sub-2")


def base3_q_storage_orthogonality_report() -> QStorageOrthogonalityReport:
    """Classify base-3 q storage as bit-exact but not accumulator compression."""

    assessment = candidate_assessment(
        candidate_name=BASE3_Q_STORAGE_CANDIDATE_NAME,
        classification=CandidateClassification.BIT_EXACT,
        covered_decision_dimensions=required_decision_dimension_names(),
        compressed_representation=False,
        note=(
            "bit-exact q-storage codec over the C1.0 semantic surface; "
            "not an accumulator/vote-state compressed representation and not C2 progress"
        ),
    )
    report = QStorageOrthogonalityReport(
        schema_version=BASE3_Q_ENTROPY_SCHEMA_VERSION,
        label=BASE3_Q_STORAGE_ORTHOGONALITY_LABEL,
        candidate_assessment=assessment,
        storage_only_status=BASE3_Q_STORAGE_ONLY_STATUS,
        bit_exact_q_storage=True,
        not_accumulator_candidate=True,
        not_c2_accumulator_compression_progress=True,
        no_vote_state_compression=True,
        qscale_boundary="unpacked q re-enters QScaleWeightFormat.INT8_LEVELS",
    )
    validate_base3_q_storage_orthogonality(report)
    return report


def validate_base3_q_storage_orthogonality(report: QStorageOrthogonalityReport) -> None:
    """Machine-check that q-storage is not counted as accumulator progress."""

    if report.schema_version != BASE3_Q_ENTROPY_SCHEMA_VERSION:
        raise ValueError("unexpected q-storage orthogonality schema version")
    if report.label != BASE3_Q_STORAGE_ORTHOGONALITY_LABEL:
        raise ValueError("unexpected q-storage orthogonality label")
    if report.storage_only_status != BASE3_Q_STORAGE_ONLY_STATUS:
        raise ValueError("q-storage orthogonality report must preserve storage-only status")
    validate_candidate_assessment(report.candidate_assessment)
    if report.candidate_assessment.normalized_classification != CandidateClassification.BIT_EXACT:
        raise ValueError("q-storage assessment must be bit_exact")
    if report.candidate_assessment.compressed_representation:
        raise ValueError("q-storage must not be an accumulator compressed representation")
    if report.candidate_assessment.c2_eligible_by_default:
        raise ValueError("q-storage must not be C2 accumulator-compression progress")
    if not report.bit_exact_q_storage:
        raise ValueError("q-storage must be marked bit-exact over unpacked q levels")
    if not report.not_accumulator_candidate:
        raise ValueError("q-storage must be marked not_accumulator_candidate")
    if not report.not_c2_accumulator_compression_progress:
        raise ValueError("q-storage must be marked not C2 accumulator-compression progress")
    if not report.no_vote_state_compression:
        raise ValueError("q-storage must not claim vote-state compression")


def base3_q_entropy_codec_summary() -> dict[str, Any]:
    return {
        "format": BASE3_Q_FORMAT,
        "group_size": BASE3_GROUP_SIZE,
        "trit_radix": BASE3_TRIT_RADIX,
        "full_group_code_count": BASE3_FULL_GROUP_CODE_COUNT,
        "unused_uint8_codes_per_full_group": BASE3_UNUSED_UINT8_CODES_PER_FULL_GROUP,
        "trit_mapping": {"-1": 0, "0": 1, "+1": 2},
        "decode_guards": [
            "reject full-group bytes >=243",
            "reject final partial-group bytes >=243",
            "reject non-zero padded trits in final partial group",
        ],
        "metadata_sidecar_fields": list(BASE3_Q_METADATA_SIDECAR_FIELDS),
    }


def base3_q_entropy_compact_report() -> Base3QEntropyCompactReport:
    """Return the compact C1.1a report without raw q/packed/accumulator arrays."""

    return Base3QEntropyCompactReport(
        schema_version=BASE3_Q_ENTROPY_SCHEMA_VERSION,
        label=BASE3_Q_ENTROPY_LABEL,
        codec_summary=base3_q_entropy_codec_summary(),
        ledger_rows=default_base3_q_entropy_ledger_table(),
        orthogonality=base3_q_storage_orthogonality_report(),
        non_claims=(
            "no accumulator encoder",
            "no vote-state compression",
            "no qscale boundary semantics change",
            "no trainer/live-run integration",
            "no acquisition or stability dynamics claim",
            "no physical sub-2 achievement while int16 accumulators remain",
            "no .pt or creditdir mutation",
            "compact report only; no raw per-weight arrays",
        ),
    )


__all__ = [
    "BASE3_FULL_GROUP_CODE_COUNT",
    "BASE3_GROUP_SIZE",
    "BASE3_Q_ENTROPY_LABEL",
    "BASE3_Q_ENTROPY_SCHEMA_VERSION",
    "BASE3_Q_FORMAT",
    "BASE3_Q_STORAGE_ONLY_STATUS",
    "BASE3_UNUSED_UINT8_CODES_PER_FULL_GROUP",
    "Base3QEntropyCompactReport",
    "Base3QEntropyLedgerRow",
    "PackedBase3TernaryQState",
    "QStorageOrthogonalityReport",
    "base3_payload_bits_and_padding",
    "base3_q_entropy_codec_summary",
    "base3_q_entropy_compact_report",
    "base3_q_entropy_ledger_for_shapes",
    "base3_q_storage_orthogonality_report",
    "default_base3_q_entropy_ledger_table",
    "measure_base3_q_entropy_budget",
    "pack_ternary_q_base3_5perbyte_reference",
    "unpack_ternary_q_base3_5perbyte_reference",
    "validate_base3_q_entropy_ledger",
    "validate_base3_q_storage_orthogonality",
]
