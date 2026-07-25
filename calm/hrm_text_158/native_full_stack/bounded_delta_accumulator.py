"""C1.1c bounded-delta accumulator reference/oracle measurement.

This module is intentionally adapter-only: it projects a compact bounded-delta
accumulator ledger, decodes that bounded state back into the existing int16
vote/cap references, and reports decision drift versus the exact int16 oracle.
It does not replace the production vote-update or global-rate-cap paths.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
import numpy as np
from typing import Any, Mapping, Sequence

import torch

from calm.hrm_text_158.native_full_stack.sparse_vote_events import SparseVoteEvents

from calm.hrm_text_158.native_full_stack.accumulator_compression import (
    CandidateAssessment,
    CandidateClassification,
    candidate_assessment,
    required_decision_dimension_names,
)
from calm.hrm_text_158.native_full_stack.global_rate_cap import (
    GlobalRateCapResult,
    GlobalRateCapSpec,
    GlobalRateCapTensorInput,
    apply_global_rate_cap_reference,
    tensor_offsets_for_vote_update_states,
)
from calm.hrm_text_158.native_full_stack.q_entropy_packing import (
    Base3QEntropyLedgerRow,
    validate_base3_q_entropy_ledger,
)
from calm.hrm_text_158.native_full_stack.vote_update import (
    VoteUpdateInputs,
    VoteUpdatePlan,
    VoteUpdateResult,
    VoteUpdateSpec,
    VoteUpdateState,
    apply_integer_vote_update_reference,
    plan_integer_vote_update_reference,
)


BOUNDED_DELTA_ACCUMULATOR_SCHEMA_VERSION = (
    "hrm_text_158_bounded_delta_accumulator/v0.reference_oracle"
)
BOUNDED_DELTA_ACCUMULATOR_LABEL = (
    "c1p1c_bounded_delta_accumulator_reference_measurement"
)
HYBRID_HOT_EXACT_COLD_DEFAULT_CANDIDATE = (
    "budget_capped_hot_exact_cold_default_sparse_exceptions"
)
EVENT_CODED_CROSSING_RESIDUAL_LOG_CANDIDATE = (
    "event_coded_crossing_residual_log"
)
COARSE_SIGNED_CHARGE_SPARSE_FRONTIER_CANDIDATE = (
    "coarse_signed_charge_sparse_exact_frontier"
)
BOUNDED_DELTA_WITH_REPORT = CandidateClassification.BOUNDED_DELTA_WITH_REPORT.value
BOUNDED_DELTA_GUARDRAIL_FAILED = "bounded_delta_guardrail_failed"
BOUNDED_DELTA_LEDGER_FAILED = "bounded_delta_ledger_failed"
BOUNDED_DELTA_ADMISSION_FAILED = "bounded_delta_admission_failed"
ACCUMULATOR_SUBSTITUTE_LOCAL_VOTE_UPDATE_EXECUTABLE = (
    "accumulator_substitute.local_vote_update_executable"
)
ALGORITHMIC_LOCAL_VOTE_UPDATE_EXECUTABLE_NOT_PHYSICAL_SUB2 = (
    "accumulator_substitute.algorithmic_local_vote_update_executable_not_physical_sub2"
)
INTRINSIC_BOUNDED_UPDATE_DOMAIN_GAP = "intrinsic_bounded_update_domain_gap"
BOUNDED_LOCAL_VOTE_UPDATE_PROOF_SCHEMA_VERSION = (
    "hrm_text_158_bounded_delta_local_vote_update_proof/v0"
)


def _bits_per_weight(bits: int | float, eligible_weight_count: int) -> float:
    eligible = int(eligible_weight_count)
    if eligible <= 0:
        raise ValueError("eligible_weight_count must be > 0")
    return float(bits) / float(eligible)


def _index_bits_for_numel(numel: int) -> int:
    numel_i = int(numel)
    if numel_i <= 0:
        raise ValueError("numel must be > 0")
    if numel_i == 1:
        return 1
    return int(math.ceil(math.log2(float(numel_i))))


def _tensor_sha256(tensor: torch.Tensor) -> str:
    cpu = tensor.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str(cpu.dtype).encode("utf-8"))
    h.update(str(tuple(cpu.shape)).encode("utf-8"))
    h.update(cpu.numpy().tobytes())
    return h.hexdigest()


def _ids_from_indices(state_key: str, indices: torch.Tensor) -> set[tuple[str, int]]:
    return {
        (str(state_key), int(idx))
        for idx in indices.detach().cpu().to(torch.int64).flatten().tolist()
    }


def _backlog_key_set(backlog: Mapping[str, Mapping[int, Mapping[str, int]]]) -> set[tuple[str, int]]:
    return {
        (str(state_key), int(flat_index))
        for state_key, by_index in backlog.items()
        for flat_index in by_index
    }


def _backlog_keys_sha256(backlog: Mapping[str, Mapping[int, Mapping[str, int]]]) -> str:
    return hashlib.sha256(str(sorted(_backlog_key_set(backlog))).encode("utf-8")).hexdigest()


def _identity_sha256(identities: set[tuple[str, int]]) -> str:
    h = hashlib.sha256()
    for state_key, flat_index in sorted(identities):
        h.update(state_key.encode("utf-8"))
        h.update(b":")
        h.update(str(int(flat_index)).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _ordered_identity_sha256(state_key: str, indices: Sequence[int]) -> str:
    h = hashlib.sha256()
    for index in indices:
        h.update(str(state_key).encode("utf-8"))
        h.update(b":")
        h.update(str(int(index)).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _ordered_value_sha256(state_key: str, label: str, values: Mapping[int, int]) -> str:
    h = hashlib.sha256()
    for index, value in values.items():
        h.update(str(state_key).encode("utf-8"))
        h.update(b":")
        h.update(str(label).encode("utf-8"))
        h.update(b":")
        h.update(str(int(index)).encode("utf-8"))
        h.update(b"=")
        h.update(str(int(value)).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _symmetric_fraction(
    exact: set[tuple[str, int]],
    bounded: set[tuple[str, int]],
) -> tuple[int, float]:
    changed = len(exact ^ bounded)
    denominator = len(exact | bounded)
    return changed, (float(changed) / float(denominator) if denominator else 0.0)


def _rank_delta(
    exact_order: Sequence[tuple[str, int]],
    bounded_order: Sequence[tuple[str, int]],
) -> int:
    exact_rank = {identity: rank for rank, identity in enumerate(exact_order)}
    bounded_rank = {identity: rank for rank, identity in enumerate(bounded_order)}
    common = set(exact_rank) & set(bounded_rank)
    if not common:
        return 0
    return max(abs(exact_rank[identity] - bounded_rank[identity]) for identity in common)


def _p95(values: torch.Tensor) -> float:
    flat = values.detach().cpu().to(torch.float64).flatten()
    if int(flat.numel()) == 0:
        return 0.0
    sorted_values = torch.sort(flat).values
    idx = int(math.ceil(0.95 * float(sorted_values.numel())) - 1)
    idx = max(0, min(idx, int(sorted_values.numel()) - 1))
    return float(sorted_values[idx].item())


@dataclass(frozen=True)
class BoundedDeltaStorageProjection:
    """Overhead-inclusive storage projection for the bounded-delta acc state."""

    eligible_weight_count: int
    hot_exact_row_count: int
    cold_exception_row_count: int
    event_delta_count: int
    backlog_entry_count: int
    index_bits_per_row: int
    hot_value_bits_per_row: int
    hot_flag_bits_per_row: int
    cold_exception_value_bits_per_row: int
    cold_exception_flag_bits_per_row: int
    event_delta_bits_per_entry: int
    event_delta_flag_bits_per_entry: int
    backlog_age_bits_per_entry: int
    backlog_defer_count_bits_per_entry: int
    tensor_metadata_bits: int
    bucket_metadata_bits: int
    scale_metadata_bits: int
    guardrail_metadata_bits: int
    dense_cold_bits_per_weight: float
    hot_exact_bits: int
    cold_exception_bits: int
    event_delta_bits: int
    backlog_bits: int
    metadata_bits: int
    dense_cold_bits: float
    total_projected_bits: float
    bounded_delta_acc_bits_per_weight: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class BoundedDeltaInclusiveLedger:
    """Conjunctive q+scale+bounded-acc physical ledger."""

    schema_version: str
    label: str
    q_regime_name: str
    target_bits_per_weight: float
    eligible_weight_count: int
    q_packed_total_bits_per_weight: float
    frozen_scale_fp32_bits_per_weight: float
    bounded_delta_acc_bits_per_weight: float
    remaining_accumulator_budget_bits_per_weight: float
    packed_inclusive_physical_bits_per_weight: float
    accumulator_fits_remaining_budget: bool
    inclusive_target_achieved: bool
    claimable_physical_sub2: bool
    ledger_status: str

    def to_dict(self) -> dict[str, int | float | bool | str]:
        return asdict(self)


@dataclass(frozen=True)
class BoundedDeltaGuardSpec:
    """Pre-declared decision-surface drift bounds.

    This is intentionally separate from the measured report so pass/fail cannot
    be moved after measuring drift.
    """

    name: str = "c1p1c_budget_capped_bounded_tolerance_guard"
    max_candidate_changed_fraction: float = 0.0
    max_accepted_changed_fraction: float = 0.0
    max_deferred_changed_fraction: float = 0.0
    max_q_changed_fraction: float = 0.0
    max_backlog_key_changed_fraction: float = 0.0
    max_cap_frontier_rank_delta: int = 0
    hot_risk_rows_require_zero_drift: bool = True

    def validate(self) -> None:
        for name in (
            "max_candidate_changed_fraction",
            "max_accepted_changed_fraction",
            "max_deferred_changed_fraction",
            "max_q_changed_fraction",
            "max_backlog_key_changed_fraction",
        ):
            value = float(getattr(self, name))
            if value < 0.0 or value > 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        if int(self.max_cap_frontier_rank_delta) < 0:
            raise ValueError("max_cap_frontier_rank_delta must be >= 0")

    def to_dict(self) -> dict[str, int | float | bool | str]:
        return asdict(self)


@dataclass(frozen=True)
class BoundedDeltaAdmissionContract:
    """Machine-checkable candidate contract for the cheap exact admission gate."""

    candidate_name: str
    preserved_information: tuple[str, ...]
    capacity_hypothesis: str
    sub2_persistent_strategy: str
    exact_surfaces: tuple[str, ...]
    allowed_divergence_contract: str
    max_cap_frontier_rank_delta: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_name": self.candidate_name,
            "preserved_information": list(self.preserved_information),
            "capacity_hypothesis": self.capacity_hypothesis,
            "sub2_persistent_strategy": self.sub2_persistent_strategy,
            "exact_surfaces": list(self.exact_surfaces),
            "allowed_divergence_contract": self.allowed_divergence_contract,
            "max_cap_frontier_rank_delta": int(self.max_cap_frontier_rank_delta),
        }


@dataclass(frozen=True)
class BoundedDeltaRejectionSurface:
    surface: str
    observed: int | float | bool | str
    threshold: int | float | bool | str
    status: str
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BoundedDeltaRejectionTelemetry:
    candidate_name: str
    admission_passed: bool
    summary: str
    failed_surfaces: tuple[str, ...]
    surfaces: tuple[BoundedDeltaRejectionSurface, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_name": self.candidate_name,
            "admission_passed": bool(self.admission_passed),
            "summary": self.summary,
            "failed_surfaces": list(self.failed_surfaces),
            "surfaces": [item.to_dict() for item in self.surfaces],
        }


@dataclass(frozen=True)
class BoundedDeltaAccumulatorState:
    """Compact reference state: exact hot rows plus cold default/exceptions."""

    logical_shape: tuple[int, ...]
    cold_default_value: int
    hot_exact_indices: tuple[int, ...]
    hot_exact_values: tuple[int, ...]
    cold_exception_indices: tuple[int, ...] = ()
    cold_exception_values: tuple[int, ...] = ()
    candidate_name: str = HYBRID_HOT_EXACT_COLD_DEFAULT_CANDIDATE
    raw_arrays_included: bool = False

    @property
    def logical_numel(self) -> int:
        out = 1
        for dim in self.logical_shape:
            out *= int(dim)
        return int(out)

    def to_dict(self) -> dict[str, Any]:
        return {
            "logical_shape": list(self.logical_shape),
            "logical_numel": self.logical_numel,
            "cold_default_value": int(self.cold_default_value),
            "hot_exact_row_count": len(self.hot_exact_indices),
            "cold_exception_row_count": len(self.cold_exception_indices),
            "candidate_name": self.candidate_name,
            "raw_arrays_included": bool(self.raw_arrays_included),
        }


@dataclass(frozen=True)
class BoundedDirectLocalUpdateCandidateResult:
    next_bounded_accumulator: BoundedDeltaAccumulatorState
    next_q_levels: torch.Tensor
    proof: dict[str, Any]


@dataclass(frozen=True)
class BoundedDeltaOracleInput:
    """One exact state plus the bounded-delta encoder choices for it."""

    state_key: str
    state: VoteUpdateState
    vote_inputs: VoteUpdateInputs
    vote_spec: VoteUpdateSpec
    hot_exact_indices: tuple[int, ...] = ()
    cold_default_value: int = 0
    cold_exception_indices: tuple[int, ...] = ()
    cold_exception_values: tuple[int, ...] | None = None


@dataclass(frozen=True)
class BoundedDeltaMeasuredReport:
    """Compact measured decision drift; no raw per-weight tensors."""

    schema_version: str
    label: str
    candidate_name: str
    candidate_changed_count: int
    candidate_union_count: int
    candidate_changed_fraction: float
    direction_changed_count: int
    accepted_changed_count: int
    accepted_union_count: int
    accepted_changed_fraction: float
    deferred_changed_count: int
    deferred_union_count: int
    deferred_changed_fraction: float
    q_changed_count: int
    q_changed_union_count: int
    q_changed_fraction: float
    backlog_key_changed_count: int
    backlog_key_union_count: int
    backlog_key_changed_fraction: float
    cap_frontier_rank_delta: int
    hot_risk_changed_count: int
    max_abs_acc_error: int
    p95_abs_acc_error: float
    fired_or_accepted_residual_changed_count: int
    fired_or_accepted_residual_identities_sha256: str
    hot_residual_changed_count: int
    hot_residual_identities_sha256: str
    accumulator_residual_hash_match: bool
    exact_accumulator_residuals_sha256: dict[str, str]
    bounded_accumulator_residuals_sha256: dict[str, str]
    exact_candidate_identities_sha256: str
    bounded_candidate_identities_sha256: str
    exact_accepted_identities_sha256: str
    bounded_accepted_identities_sha256: str
    exact_deferred_identities_sha256: str
    bounded_deferred_identities_sha256: str
    oracle_parity: dict[str, bool | str | int]
    raw_arrays_included: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BoundedDeltaGuardEvaluation:
    guard_spec: BoundedDeltaGuardSpec
    measured_report: BoundedDeltaMeasuredReport
    guard_passed: bool
    failed_metrics: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "guard_spec": self.guard_spec.to_dict(),
            "measured_report": self.measured_report.to_dict(),
            "guard_passed": bool(self.guard_passed),
            "failed_metrics": list(self.failed_metrics),
        }


@dataclass(frozen=True)
class BoundedDeltaReferenceReport:
    """C1.1c classification tying ledger + guard + measured drift."""

    schema_version: str
    label: str
    candidate_name: str
    classification: str
    ledger: BoundedDeltaInclusiveLedger
    storage_projection: BoundedDeltaStorageProjection
    guard_spec: BoundedDeltaGuardSpec
    admission_contract: BoundedDeltaAdmissionContract
    measured_report: BoundedDeltaMeasuredReport
    guard_passed: bool
    failed_metrics: tuple[str, ...]
    admission_passed: bool
    admission_failed_surfaces: tuple[str, ...]
    candidate_assessment: CandidateAssessment
    rejection_telemetry: BoundedDeltaRejectionTelemetry
    raw_arrays_included: bool
    non_claims: tuple[str, ...]
    next_candidate_if_failed: str

    @property
    def claimable_physical_sub2_with_guardrail(self) -> bool:
        return (
            self.classification == BOUNDED_DELTA_WITH_REPORT
            and self.ledger.claimable_physical_sub2
            and self.guard_passed
            and self.admission_passed
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "label": self.label,
            "candidate_name": self.candidate_name,
            "classification": self.classification,
            "ledger": self.ledger.to_dict(),
            "storage_projection": self.storage_projection.to_dict(),
            "guard_spec": self.guard_spec.to_dict(),
            "admission_contract": self.admission_contract.to_dict(),
            "measured_report": self.measured_report.to_dict(),
            "guard_passed": bool(self.guard_passed),
            "failed_metrics": list(self.failed_metrics),
            "admission_passed": bool(self.admission_passed),
            "admission_failed_surfaces": list(self.admission_failed_surfaces),
            "candidate_assessment": self.candidate_assessment.to_dict(),
            "rejection_telemetry": self.rejection_telemetry.to_dict(),
            "raw_arrays_included": bool(self.raw_arrays_included),
            "non_claims": list(self.non_claims),
            "next_candidate_if_failed": self.next_candidate_if_failed,
            "claimable_physical_sub2_with_guardrail": (
                self.claimable_physical_sub2_with_guardrail
            ),
        }


def project_bounded_delta_accumulator_bpw(
    *,
    eligible_weight_count: int,
    hot_exact_row_count: int,
    cold_exception_row_count: int = 0,
    event_delta_count: int = 0,
    backlog_entry_count: int = 0,
    index_bits_per_row: int | None = None,
    hot_value_bits_per_row: int = 16,
    hot_flag_bits_per_row: int = 2,
    cold_exception_value_bits_per_row: int = 16,
    cold_exception_flag_bits_per_row: int = 2,
    event_delta_bits_per_entry: int = 16,
    event_delta_flag_bits_per_entry: int = 2,
    backlog_age_bits_per_entry: int = 16,
    backlog_defer_count_bits_per_entry: int = 16,
    tensor_metadata_bits: int = 0,
    bucket_metadata_bits: int = 0,
    scale_metadata_bits: int = 0,
    guardrail_metadata_bits: int = 0,
    dense_cold_bits_per_weight: float = 0.0,
) -> BoundedDeltaStorageProjection:
    """Project all bounded-delta accumulator storage terms in bits/weight."""

    eligible = int(eligible_weight_count)
    if eligible <= 0:
        raise ValueError("eligible_weight_count must be > 0")
    counts = {
        "hot_exact_row_count": hot_exact_row_count,
        "cold_exception_row_count": cold_exception_row_count,
        "event_delta_count": event_delta_count,
        "backlog_entry_count": backlog_entry_count,
    }
    for name, value in counts.items():
        if int(value) < 0:
            raise ValueError(f"{name} must be >= 0")
        if int(value) > eligible:
            raise ValueError(f"{name} cannot exceed eligible_weight_count")
    bit_fields = {
        "hot_value_bits_per_row": hot_value_bits_per_row,
        "hot_flag_bits_per_row": hot_flag_bits_per_row,
        "cold_exception_value_bits_per_row": cold_exception_value_bits_per_row,
        "cold_exception_flag_bits_per_row": cold_exception_flag_bits_per_row,
        "event_delta_bits_per_entry": event_delta_bits_per_entry,
        "event_delta_flag_bits_per_entry": event_delta_flag_bits_per_entry,
        "backlog_age_bits_per_entry": backlog_age_bits_per_entry,
        "backlog_defer_count_bits_per_entry": backlog_defer_count_bits_per_entry,
        "tensor_metadata_bits": tensor_metadata_bits,
        "bucket_metadata_bits": bucket_metadata_bits,
        "scale_metadata_bits": scale_metadata_bits,
        "guardrail_metadata_bits": guardrail_metadata_bits,
    }
    for name, value in bit_fields.items():
        if int(value) < 0:
            raise ValueError(f"{name} must be >= 0")
    if int(hot_value_bits_per_row) <= 0 or int(cold_exception_value_bits_per_row) <= 0:
        raise ValueError("value bit widths must be > 0")
    if int(event_delta_bits_per_entry) <= 0:
        raise ValueError("event_delta_bits_per_entry must be > 0")
    dense_cold_bpw = float(dense_cold_bits_per_weight)
    if dense_cold_bpw < 0.0:
        raise ValueError("dense_cold_bits_per_weight must be >= 0")

    index_bits = _index_bits_for_numel(eligible) if index_bits_per_row is None else int(index_bits_per_row)
    if index_bits <= 0:
        raise ValueError("index_bits_per_row must be > 0")

    hot_bits = int(hot_exact_row_count) * (
        index_bits + int(hot_value_bits_per_row) + int(hot_flag_bits_per_row)
    )
    cold_bits = int(cold_exception_row_count) * (
        index_bits
        + int(cold_exception_value_bits_per_row)
        + int(cold_exception_flag_bits_per_row)
    )
    event_bits = int(event_delta_count) * (
        index_bits + int(event_delta_bits_per_entry) + int(event_delta_flag_bits_per_entry)
    )
    backlog_bits = int(backlog_entry_count) * (
        index_bits
        + int(backlog_age_bits_per_entry)
        + int(backlog_defer_count_bits_per_entry)
    )
    metadata_bits = (
        int(tensor_metadata_bits)
        + int(bucket_metadata_bits)
        + int(scale_metadata_bits)
        + int(guardrail_metadata_bits)
    )
    dense_bits = dense_cold_bpw * float(eligible)
    total_bits = float(hot_bits + cold_bits + event_bits + backlog_bits + metadata_bits) + dense_bits
    return BoundedDeltaStorageProjection(
        eligible_weight_count=eligible,
        hot_exact_row_count=int(hot_exact_row_count),
        cold_exception_row_count=int(cold_exception_row_count),
        event_delta_count=int(event_delta_count),
        backlog_entry_count=int(backlog_entry_count),
        index_bits_per_row=index_bits,
        hot_value_bits_per_row=int(hot_value_bits_per_row),
        hot_flag_bits_per_row=int(hot_flag_bits_per_row),
        cold_exception_value_bits_per_row=int(cold_exception_value_bits_per_row),
        cold_exception_flag_bits_per_row=int(cold_exception_flag_bits_per_row),
        event_delta_bits_per_entry=int(event_delta_bits_per_entry),
        event_delta_flag_bits_per_entry=int(event_delta_flag_bits_per_entry),
        backlog_age_bits_per_entry=int(backlog_age_bits_per_entry),
        backlog_defer_count_bits_per_entry=int(backlog_defer_count_bits_per_entry),
        tensor_metadata_bits=int(tensor_metadata_bits),
        bucket_metadata_bits=int(bucket_metadata_bits),
        scale_metadata_bits=int(scale_metadata_bits),
        guardrail_metadata_bits=int(guardrail_metadata_bits),
        dense_cold_bits_per_weight=dense_cold_bpw,
        hot_exact_bits=hot_bits,
        cold_exception_bits=cold_bits,
        event_delta_bits=event_bits,
        backlog_bits=backlog_bits,
        metadata_bits=metadata_bits,
        dense_cold_bits=dense_bits,
        total_projected_bits=total_bits,
        bounded_delta_acc_bits_per_weight=_bits_per_weight(total_bits, eligible),
    )


def bounded_delta_inclusive_ledger(
    q_ledger_row: Base3QEntropyLedgerRow,
    projection: BoundedDeltaStorageProjection,
) -> BoundedDeltaInclusiveLedger:
    """Fold selected q/scale regime with bounded-delta accumulator storage."""

    validate_base3_q_entropy_ledger(q_ledger_row)
    eligible = int(projection.eligible_weight_count)
    if int(q_ledger_row.eligible_weight_count) != eligible:
        raise ValueError(
            "q ledger eligible weight count must match bounded-delta projection; "
            f"q={q_ledger_row.eligible_weight_count} acc={eligible}"
        )
    q_bpw = float(q_ledger_row.q_packed_total_bits_per_weight)
    scale_bpw = float(q_ledger_row.frozen_scale_fp32_bits_per_weight)
    acc_bpw = float(projection.bounded_delta_acc_bits_per_weight)
    target = float(q_ledger_row.target_bits_per_weight)
    remaining = target - q_bpw - scale_bpw
    inclusive = q_bpw + scale_bpw + acc_bpw
    fits_remaining = acc_bpw <= remaining
    target_achieved = inclusive < target
    claimable = bool(fits_remaining and target_achieved)
    return BoundedDeltaInclusiveLedger(
        schema_version=BOUNDED_DELTA_ACCUMULATOR_SCHEMA_VERSION,
        label=BOUNDED_DELTA_ACCUMULATOR_LABEL,
        q_regime_name=q_ledger_row.regime_name,
        target_bits_per_weight=target,
        eligible_weight_count=eligible,
        q_packed_total_bits_per_weight=q_bpw,
        frozen_scale_fp32_bits_per_weight=scale_bpw,
        bounded_delta_acc_bits_per_weight=acc_bpw,
        remaining_accumulator_budget_bits_per_weight=remaining,
        packed_inclusive_physical_bits_per_weight=inclusive,
        accumulator_fits_remaining_budget=bool(fits_remaining),
        inclusive_target_achieved=bool(target_achieved),
        claimable_physical_sub2=claimable,
        ledger_status=(
            "bounded_delta_inclusive_ledger_pass"
            if claimable else "bounded_delta_inclusive_ledger_failed"
        ),
    )


def validate_bounded_delta_inclusive_ledger(
    ledger: BoundedDeltaInclusiveLedger,
    *,
    claimed_physical_sub2_achieved: bool = False,
) -> None:
    recomputed_remaining = (
        ledger.target_bits_per_weight
        - ledger.q_packed_total_bits_per_weight
        - ledger.frozen_scale_fp32_bits_per_weight
    )
    if not math.isclose(
        ledger.remaining_accumulator_budget_bits_per_weight,
        recomputed_remaining,
        abs_tol=1e-12,
    ):
        raise ValueError("remaining budget must be target - q_total - frozen_scale")
    recomputed_inclusive = (
        ledger.q_packed_total_bits_per_weight
        + ledger.frozen_scale_fp32_bits_per_weight
        + ledger.bounded_delta_acc_bits_per_weight
    )
    if not math.isclose(
        ledger.packed_inclusive_physical_bits_per_weight,
        recomputed_inclusive,
        abs_tol=1e-12,
    ):
        raise ValueError("inclusive ledger must be q_total + frozen_scale + bounded_acc")
    recomputed_fits = (
        ledger.bounded_delta_acc_bits_per_weight
        <= ledger.remaining_accumulator_budget_bits_per_weight
    )
    if bool(ledger.accumulator_fits_remaining_budget) != bool(recomputed_fits):
        raise ValueError("accumulator budget flag must be recomputed from the selected regime")
    recomputed_target = recomputed_inclusive < ledger.target_bits_per_weight
    if bool(ledger.inclusive_target_achieved) != bool(recomputed_target):
        raise ValueError("inclusive target flag must be recomputed from the selected regime")
    if bool(ledger.claimable_physical_sub2) != bool(recomputed_fits and recomputed_target):
        raise ValueError("physical sub-2 claimability must be conjunctive")
    if claimed_physical_sub2_achieved and not ledger.claimable_physical_sub2:
        raise ValueError("physical sub-2 claim is invalid when ledger/remaining-budget gate fails")


def _coerce_index_tuple(
    values: Sequence[int] | torch.Tensor,
    *,
    numel: int,
    name: str,
) -> tuple[int, ...]:
    if isinstance(values, torch.Tensor):
        items = values.detach().cpu().to(torch.int64).flatten().tolist()
    else:
        items = [int(value) for value in values]
    out = tuple(sorted(int(value) for value in items))
    if len(out) != len(set(out)):
        raise ValueError(f"{name} must not contain duplicate flat indices")
    if any(value < 0 or value >= int(numel) for value in out):
        raise ValueError(f"{name} contains out-of-range flat indices")
    return out


def _coerce_value_tuple(
    values: Sequence[int] | torch.Tensor,
    *,
    expected_count: int,
    name: str,
) -> tuple[int, ...]:
    if isinstance(values, torch.Tensor):
        items = values.detach().cpu().to(torch.int64).flatten().tolist()
    else:
        items = [int(value) for value in values]
    if len(items) != int(expected_count):
        raise ValueError(f"{name} count must match its index count")
    out = tuple(int(value) for value in items)
    if any(value < -32768 or value > 32767 for value in out):
        raise ValueError(f"{name} values must fit int16")
    return out


def encode_budget_capped_hybrid_reference(
    state: VoteUpdateState,
    *,
    hot_exact_indices: Sequence[int] | torch.Tensor,
    cold_default_value: int = 0,
    cold_exception_indices: Sequence[int] | torch.Tensor = (),
    cold_exception_values: Sequence[int] | torch.Tensor | None = None,
) -> BoundedDeltaAccumulatorState:
    """Encode exact int16 accumulators as hot exact rows plus cold default/exceptions."""

    acc = state.accumulators
    if acc.dtype != torch.int16:
        raise ValueError(f"accumulators must be torch.int16, got {acc.dtype}")
    numel = int(acc.numel())
    if numel <= 0:
        raise ValueError("bounded-delta encoder requires a non-empty accumulator tensor")
    hot = _coerce_index_tuple(hot_exact_indices, numel=numel, name="hot_exact_indices")
    cold_ex = _coerce_index_tuple(
        cold_exception_indices,
        numel=numel,
        name="cold_exception_indices",
    )
    overlap = set(hot) & set(cold_ex)
    if overlap:
        raise ValueError(f"hot/exact and cold exceptions overlap: {sorted(overlap)[:8]}")
    flat = acc.detach().cpu().flatten().to(torch.int16)
    hot_values = tuple(int(flat[index].item()) for index in hot)
    if cold_exception_values is None:
        cold_values = tuple(int(flat[index].item()) for index in cold_ex)
    else:
        cold_values = _coerce_value_tuple(
            cold_exception_values,
            expected_count=len(cold_ex),
            name="cold_exception_values",
        )
    default = int(cold_default_value)
    if default < -32768 or default > 32767:
        raise ValueError("cold_default_value must fit int16")
    return BoundedDeltaAccumulatorState(
        logical_shape=tuple(int(dim) for dim in acc.shape),
        cold_default_value=default,
        hot_exact_indices=hot,
        hot_exact_values=hot_values,
        cold_exception_indices=cold_ex,
        cold_exception_values=cold_values,
    )


def decode_bounded_accumulator_to_i16(state: BoundedDeltaAccumulatorState) -> torch.Tensor:
    """Decode the bounded reference state to an approximate int16 accumulator."""

    if state.raw_arrays_included:
        raise ValueError("bounded-delta compact state must not be marked raw-array-inclusive")
    numel = state.logical_numel
    out = torch.full((numel,), int(state.cold_default_value), dtype=torch.int16)
    for indices, values, name in (
        (state.cold_exception_indices, state.cold_exception_values, "cold exceptions"),
        (state.hot_exact_indices, state.hot_exact_values, "hot exact rows"),
    ):
        if len(indices) != len(values):
            raise ValueError(f"{name} index/value count mismatch")
        for index, value in zip(indices, values):
            if index < 0 or index >= numel:
                raise ValueError(f"{name} contains out-of-range index")
            out[int(index)] = int(value)
    return out.view(state.logical_shape).contiguous()


def bounded_accumulator_decoded_sha256(state: BoundedDeltaAccumulatorState) -> str:
    """Hash the dense decoded accumulator without materializing the full tensor."""

    if state.raw_arrays_included:
        raise ValueError("bounded-delta compact state must not be marked raw-array-inclusive")
    numel = int(state.logical_numel)
    shape = tuple(state.logical_shape)
    default = int(state.cold_default_value)
    default_bytes = int(default).to_bytes(2, byteorder="little", signed=True)
    cold_indices = state.cold_exception_indices
    cold_values = state.cold_exception_values
    hot_indices = state.hot_exact_indices
    hot_values = state.hot_exact_values
    for indices, values, name in (
        (cold_indices, cold_values, "cold exceptions"),
        (hot_indices, hot_values, "hot exact rows"),
    ):
        if len(indices) != len(values):
            raise ValueError(f"{name} index/value count mismatch")
        for index in indices:
            if index < 0 or index >= numel:
                raise ValueError(f"{name} contains out-of-range index")
    overrides: dict[int, int] = {
        int(index): int(value) for index, value in zip(cold_indices, cold_values)
    }
    overrides.update(
        {int(index): int(value) for index, value in zip(hot_indices, hot_values)}
    )
    h = hashlib.sha256()
    h.update(str(torch.int16).encode("utf-8"))
    h.update(str(shape).encode("utf-8"))
    if numel <= 0:
        return h.hexdigest()

    default_chunk_elems = max(1, (1024 * 1024) // len(default_bytes))

    def _update_default_run(gap_len: int) -> None:
        if gap_len <= 0:
            return
        remaining = gap_len
        while remaining > 0:
            chunk = min(remaining, default_chunk_elems)
            h.update(default_bytes * chunk)
            remaining -= chunk

    pos = 0
    for idx in sorted(overrides):
        if idx > pos:
            _update_default_run(idx - pos)
        h.update(int(overrides[idx]).to_bytes(2, byteorder="little", signed=True))
        pos = idx + 1
    if pos < numel:
        _update_default_run(numel - pos)
    return h.hexdigest()


def _bounded_value_dict(
    state: BoundedDeltaAccumulatorState,
) -> tuple[dict[int, int], dict[int, int]]:
    return (
        {
            int(index): int(value)
            for index, value in zip(state.hot_exact_indices, state.hot_exact_values)
        },
        {
            int(index): int(value)
            for index, value in zip(state.cold_exception_indices, state.cold_exception_values)
        },
    )


def _sparse_value_sha256(
    state_key: str,
    values_by_index: Mapping[int, int],
) -> str:
    h = hashlib.sha256()
    for flat_index, value in sorted(
        (int(index), int(item))
        for index, item in values_by_index.items()
    ):
        h.update(state_key.encode("utf-8"))
        h.update(b":")
        h.update(str(flat_index).encode("utf-8"))
        h.update(b"=")
        h.update(str(value).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _truncate_toward_zero_division(numerator: int, denominator: int) -> int:
    if int(denominator) <= 0:
        raise ValueError("denominator must be > 0")
    n = int(numerator)
    d = int(denominator)
    q = abs(n) // d
    return q if n >= 0 else -q


def _clip_i16(value: int, clip_min: int, clip_max: int) -> int:
    return int(max(int(clip_min), min(int(clip_max), int(value))))


def _sparse_index_set_from_carrier(carrier: SparseVoteEvents) -> set[int]:
    return {int(carrier.indices[i].item()) for i in range(carrier.event_count())}


def _sparse_vote_value_at(carrier: SparseVoteEvents, index: int) -> int:
    matches = (carrier.indices == int(index)).nonzero(as_tuple=True)[0]
    if matches.numel() == 0:
        return 0
    return int(carrier.values[int(matches[0].item())].item())


def _vote_dense_from_carrier(carrier: SparseVoteEvents, *, numel: int) -> torch.Tensor:
    vote_dense = torch.zeros(numel, dtype=torch.int16)
    if carrier.event_count():
        vote_dense[carrier.indices] = carrier.values
    return vote_dense


def _support_mask_from_carrier(
    *,
    numel: int,
    vote_dense: torch.Tensor,
    bounded_accumulator: BoundedDeltaAccumulatorState,
) -> torch.Tensor:
    support_mask = vote_dense != 0
    if bounded_accumulator.hot_exact_indices:
        support_mask[
            torch.tensor(bounded_accumulator.hot_exact_indices, dtype=torch.int64)
        ] = True
    if bounded_accumulator.cold_exception_indices:
        support_mask[
            torch.tensor(bounded_accumulator.cold_exception_indices, dtype=torch.int64)
        ] = True
    return support_mask


def _old_values_tensor(
    *,
    numel: int,
    default_before: int,
    bounded_accumulator: BoundedDeltaAccumulatorState,
) -> torch.Tensor:
    old_full = torch.full((numel,), int(default_before), dtype=torch.int32)
    if bounded_accumulator.hot_exact_indices:
        old_full[
            torch.tensor(bounded_accumulator.hot_exact_indices, dtype=torch.int64)
        ] = torch.tensor(bounded_accumulator.hot_exact_values, dtype=torch.int32)
    if bounded_accumulator.cold_exception_indices:
        old_full[
            torch.tensor(bounded_accumulator.cold_exception_indices, dtype=torch.int64)
        ] = torch.tensor(bounded_accumulator.cold_exception_values, dtype=torch.int32)
    return old_full


def _execute_direct_bounded_local_vote_update_dense_carrier(
    *,
    state_key: str,
    q_levels: torch.Tensor,
    bounded_accumulator: BoundedDeltaAccumulatorState,
    sparse_carrier: SparseVoteEvents,
    vote_spec: VoteUpdateSpec,
) -> BoundedDirectLocalUpdateCandidateResult:
    """Tensor-native dense apply for SparseVoteEvents (r5 STEP 2)."""
    q_flat = q_levels.detach().cpu().to(torch.int8).flatten().contiguous()
    numel = int(q_flat.numel())
    sparse_carrier.validate(numel=numel)
    hot_map, cold_map = _bounded_value_dict(bounded_accumulator)
    default_before = int(bounded_accumulator.cold_default_value)
    dense_vote_authority_used = False
    event_vote_count = sparse_carrier.event_count()

    vote_dense = _vote_dense_from_carrier(sparse_carrier, numel=numel)
    support_mask = _support_mask_from_carrier(
        numel=numel,
        vote_dense=vote_dense,
        bounded_accumulator=bounded_accumulator,
    )
    support_indices = torch.nonzero(support_mask, as_tuple=False).flatten()
    support_row_count = int(support_indices.numel())

    clip_min = int(vote_spec.accumulator_clip_min)
    clip_max = int(vote_spec.accumulator_clip_max)
    threshold = int(vote_spec.threshold_abs)
    decay_num = int(vote_spec.decay_numerator)
    decay_den = int(vote_spec.decay_denominator)
    default_after = _clip_i16(
        _truncate_toward_zero_division(default_before * decay_num, decay_den),
        clip_min,
        clip_max,
    )
    default_direction = 1 if default_after >= threshold else -1 if default_after <= -threshold else 0
    default_mass_crossing_count = 0
    if default_direction != 0:
        if default_direction > 0:
            crossing_mask = q_flat < 1
        else:
            crossing_mask = q_flat > -1
        default_mass_crossing_count = int((~support_mask & crossing_mask).sum().item())

    coverage_domain = {
        "schema": BOUNDED_LOCAL_VOTE_UPDATE_PROOF_SCHEMA_VERSION,
        "state_key": state_key,
        "no_global_cap": True,
        "sparse_vote_events_only": True,
        "supports_replay_ce_veto": False,
        "supports_pc_aux": False,
        "supports_global_backlog": False,
        "supports_default_mass_crossing": False,
        "supports_dense_vote_authority": False,
        "supports_dense_shadow_authority": False,
        "supports_dense_decode_candidate_path": False,
        "supported_decision_dimensions": [
            "local_vote_update",
            "sparse_vote_events",
            "q_changed_identity_count",
            "applied_row_identity",
            "residual_after_threshold",
            "bounded_checkpoint_serialization",
        ],
        "blocked_decision_dimensions": [
            "global_cap",
            "replay_ce_veto",
            "pc_aux",
            "implicit_default_mass_crossing",
        ],
    }
    storage_projection = project_bounded_delta_accumulator_bpw(
        eligible_weight_count=numel,
        hot_exact_row_count=len(bounded_accumulator.hot_exact_indices),
        cold_exception_row_count=len(bounded_accumulator.cold_exception_indices),
        dense_cold_bits_per_weight=0.0,
    )

    if default_mass_crossing_count > 0:
        proof = {
            "schema": BOUNDED_LOCAL_VOTE_UPDATE_PROOF_SCHEMA_VERSION,
            "surface": "accumulator_substitute",
            "scoped_label": None,
            "terminal_classification": INTRINSIC_BOUNDED_UPDATE_DOMAIN_GAP,
            "pass": False,
            "runtime_state_authority_after": "sub2_persistent_hybrid_dense_transient_credit",
            "candidate_dense_decode_used": False,
            "candidate_accumulator_transient_over2_used": False,
            "candidate_vote_transient_over2_used": False,
            "candidate_dense_vote_authority_used": dense_vote_authority_used,
            "dense_oracle_control_used": False,
            "scoped_physical_budget_claim": "not_applicable_domain_gap",
            "q_storage_physical_budget_covered_by_scoped_proof": False,
            "frozen_scale_physical_budget_covered_by_scoped_proof": False,
            "coverage_domain": coverage_domain,
            "domain_gap_dimension": "implicit_default_mass_crossing",
            "domain_gap_detail": (
                "cold default update would create threshold-crossing mass on rows "
                "that are not explicitly enumerated in the bounded support"
            ),
            "default_mass_crossing_count": int(default_mass_crossing_count),
            "event_vote_count": int(event_vote_count),
            "support_row_count": support_row_count,
            "hot_exact_row_count_before": int(len(bounded_accumulator.hot_exact_indices)),
            "cold_exception_row_count_before": int(len(bounded_accumulator.cold_exception_indices)),
            "hot_exact_row_count_after": int(len(bounded_accumulator.hot_exact_indices)),
            "cold_exception_row_count_after": int(len(bounded_accumulator.cold_exception_indices)),
            "storage_projection": storage_projection.to_dict(),
        }
        return BoundedDirectLocalUpdateCandidateResult(
            next_bounded_accumulator=bounded_accumulator,
            next_q_levels=q_flat.view_as(q_levels).clone(),
            proof=proof,
        )

    old_full = _old_values_tensor(
        numel=numel,
        default_before=default_before,
        bounded_accumulator=bounded_accumulator,
    )
    old_at_support = old_full[support_indices]
    vote_at_support = vote_dense[support_indices].to(torch.int32)
    decayed = torch.div(
        old_at_support * decay_num,
        decay_den,
        rounding_mode="trunc",
    )
    new_at_support = (decayed + vote_at_support).clamp(clip_min, clip_max).to(torch.int16)

    q_at_support = q_flat[support_indices].to(torch.int32)
    flip_mask = ((new_at_support >= threshold) & (q_at_support < 1)) | (
        (new_at_support <= -threshold) & (q_at_support > -1)
    )
    flip_idx = support_indices[flip_mask]
    flip_new = new_at_support[flip_mask]
    candidate_count = int(flip_idx.numel())
    max_flips = int(vote_spec.max_flips(numel))
    applied_idx, applied_new = _select_applied_flip_candidates(
        flip_idx,
        flip_new,
        max_flips=max_flips,
        numel=numel,
    )

    q_after = q_flat.clone()
    residual_after_threshold: dict[int, int] = {}
    applied_directions_by_index: dict[int, int] = {}
    applied_thresholds_by_index: dict[int, int] = {}
    acc_at_support = new_at_support.clone()
    if applied_idx.numel() > 0:
        directions = torch.where(
            applied_new >= threshold,
            torch.ones_like(applied_new, dtype=torch.int32),
            -torch.ones_like(applied_new, dtype=torch.int32),
        )
        q_applied = q_after[applied_idx].to(torch.int32) + directions
        q_after[applied_idx] = torch.clamp(q_applied, -1, 1).to(torch.int8)
        residual_tensor = torch.clamp(
            applied_new.to(torch.int32) - directions * threshold,
            -threshold + 1,
            threshold - 1,
        ).to(torch.int16)
        applied_positions = torch.searchsorted(support_indices, applied_idx)
        acc_at_support[applied_positions] = residual_tensor
        applied_indices_list = applied_idx.tolist()
        directions_list = directions.tolist()
        residual_list = residual_tensor.tolist()
        for index, direction, residual in zip(
            applied_indices_list,
            directions_list,
            residual_list,
        ):
            applied_directions_by_index[int(index)] = int(direction)
            applied_thresholds_by_index[int(index)] = int(threshold)
            residual_after_threshold[int(index)] = int(residual)
    else:
        applied_indices_list: list[int] = []

    hot_idx_tensor = (
        torch.tensor(bounded_accumulator.hot_exact_indices, dtype=torch.int64)
        if bounded_accumulator.hot_exact_indices
        else None
    )
    if hot_idx_tensor is not None and hot_idx_tensor.numel() > 0:
        hot_in_support = torch.isin(support_indices, hot_idx_tensor)
    else:
        hot_in_support = torch.zeros(support_indices.numel(), dtype=torch.bool)
    cold_exc_pos = (~hot_in_support) & (acc_at_support != int(default_after))
    next_cold_exception_indices = tuple(support_indices[cold_exc_pos].tolist())
    next_cold_exception_values = tuple(acc_at_support[cold_exc_pos].tolist())
    if hot_idx_tensor is not None and hot_idx_tensor.numel() > 0:
        hot_positions = torch.searchsorted(support_indices, hot_idx_tensor)
        hot_valid = (hot_positions < support_indices.numel()) & (
            support_indices[hot_positions] == hot_idx_tensor
        )
        hot_values = torch.full(
            (hot_idx_tensor.numel(),),
            int(default_after),
            dtype=torch.int16,
        )
        hot_values[hot_valid] = acc_at_support[hot_positions[hot_valid]]
        next_hot_values = tuple(int(value) for value in hot_values.tolist())
    else:
        next_hot_values = ()
    applied_indices = tuple(applied_indices_list)
    next_bounded = BoundedDeltaAccumulatorState(
        logical_shape=bounded_accumulator.logical_shape,
        cold_default_value=int(default_after),
        hot_exact_indices=tuple(int(index) for index in bounded_accumulator.hot_exact_indices),
        hot_exact_values=next_hot_values,
        cold_exception_indices=next_cold_exception_indices,
        cold_exception_values=next_cold_exception_values,
        candidate_name=bounded_accumulator.candidate_name,
        raw_arrays_included=False,
    )
    next_projection = project_bounded_delta_accumulator_bpw(
        eligible_weight_count=numel,
        hot_exact_row_count=len(next_bounded.hot_exact_indices),
        cold_exception_row_count=len(next_bounded.cold_exception_indices),
        dense_cold_bits_per_weight=0.0,
    )
    changed = torch.nonzero(q_flat != q_after, as_tuple=False).flatten()
    q_changed_indices = tuple(sorted(int(index.item()) for index in changed))
    accumulator_physical_sub2_pass = (
        float(next_projection.bounded_delta_acc_bits_per_weight) < 2.0
    )
    scoped_positive_label = (
        ACCUMULATOR_SUBSTITUTE_LOCAL_VOTE_UPDATE_EXECUTABLE
        if accumulator_physical_sub2_pass
        else ALGORITHMIC_LOCAL_VOTE_UPDATE_EXECUTABLE_NOT_PHYSICAL_SUB2
    )
    proof = {
        "schema": BOUNDED_LOCAL_VOTE_UPDATE_PROOF_SCHEMA_VERSION,
        "surface": "accumulator_substitute",
        "scoped_label": scoped_positive_label,
        "terminal_classification": scoped_positive_label,
        "pass": True,
        "runtime_state_authority_after": "sub2_persistent_hybrid_dense_transient_credit",
        "candidate_dense_decode_used": False,
        "candidate_accumulator_transient_over2_used": False,
        "candidate_vote_transient_over2_used": False,
        "candidate_dense_vote_authority_used": dense_vote_authority_used,
        "dense_oracle_control_used": False,
        "scoped_physical_budget_claim": (
            "physical_sub2_budgeted"
            if accumulator_physical_sub2_pass
            else "algorithmic_only_not_physical_sub2"
        ),
        "q_storage_physical_budget_covered_by_scoped_proof": False,
        "frozen_scale_physical_budget_covered_by_scoped_proof": False,
        "coverage_domain": coverage_domain,
        "domain_gap_dimension": None,
        "domain_gap_detail": None,
        "default_mass_crossing_count": 0,
        "event_vote_count": int(event_vote_count),
        "candidate_count": candidate_count,
        "max_flips": int(max_flips),
        "pre_veto_selected_flip_count": int(len(applied_indices)),
        "support_row_count": support_row_count,
        "hot_exact_row_count_before": int(len(bounded_accumulator.hot_exact_indices)),
        "cold_exception_row_count_before": int(len(bounded_accumulator.cold_exception_indices)),
        "hot_exact_row_count_after": int(len(next_bounded.hot_exact_indices)),
        "cold_exception_row_count_after": int(len(next_bounded.cold_exception_indices)),
        "q_changed_count": int(len(q_changed_indices)),
        "q_changed_identities_sha256": _identity_sha256(
            {(state_key, int(index)) for index in q_changed_indices},
        ),
        "applied_row_count": int(len(applied_indices)),
        "applied_row_identities_sha256": _identity_sha256(
            {(state_key, int(index)) for index in applied_indices},
        ),
        "ordered_applied_row_identities_sha256": _ordered_identity_sha256(
            state_key,
            applied_indices,
        ),
        "applied_directions_sha256": _ordered_value_sha256(
            state_key,
            "direction",
            applied_directions_by_index,
        ),
        "applied_thresholds_sha256": _ordered_value_sha256(
            state_key,
            "threshold",
            applied_thresholds_by_index,
        ),
        "residual_after_threshold_sha256": _sparse_value_sha256(
            state_key,
            residual_after_threshold,
        ),
        "candidate_q_sha256_after": _tensor_sha256(q_after.view_as(q_levels)),
        "storage_projection": next_projection.to_dict(),
        "accumulator_physical_sub2_pass": bool(accumulator_physical_sub2_pass),
        "bounded_accumulator_summary_after": next_bounded.to_dict(),
    }
    return BoundedDirectLocalUpdateCandidateResult(
        next_bounded_accumulator=next_bounded,
        next_q_levels=q_after.view_as(q_levels).to(torch.int8).contiguous(),
        proof=proof,
    )


def _select_applied_flip_candidates(
    flip_idx: torch.Tensor,
    flip_new: torch.Tensor,
    *,
    max_flips: int,
    numel: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if flip_idx.numel() == 0:
        return flip_idx, flip_new
    if flip_idx.numel() <= max_flips:
        order_np = np.lexsort(
            (flip_idx.cpu().numpy(), (-flip_new.abs()).cpu().numpy()),
        )
        order = torch.from_numpy(order_np)
        return flip_idx[order], flip_new[order]
    score = flip_new.abs().to(torch.float64) * (float(numel) + 1.0) + (
        float(numel) - flip_idx.to(torch.float64)
    )
    _top_vals, top_pos = torch.topk(score, k=max_flips, largest=True)
    top_pos_order = torch.argsort(score[top_pos], descending=True)
    selected = top_pos[top_pos_order]
    return flip_idx[selected], flip_new[selected]


def _sparse_votes_dict_from_mapping(
    sparse_vote_events: Mapping[int, int],
    *,
    state_key: str,
    numel: int,
) -> dict[int, int]:
    sparse_votes: dict[int, int] = {}
    for raw_index, raw_vote in sparse_vote_events.items():
        index = int(raw_index)
        vote = int(raw_vote)
        if index < 0 or index >= numel:
            raise ValueError(f"sparse vote index {index} out of range for {state_key}")
        if vote < -32768 or vote > 32767:
            raise ValueError(f"sparse vote value {vote} must fit int16")
        if vote != 0:
            sparse_votes[index] = vote
    return sparse_votes


def execute_direct_bounded_local_vote_update_candidate(
    *,
    state_key: str,
    q_levels: torch.Tensor,
    bounded_accumulator: BoundedDeltaAccumulatorState,
    sparse_vote_events: SparseVoteEvents | Mapping[int, int],
    vote_spec: VoteUpdateSpec,
) -> BoundedDirectLocalUpdateCandidateResult:
    if q_levels.dtype != torch.int8:
        raise ValueError(f"q_levels must be torch.int8, got {q_levels.dtype}")
    if tuple(int(dim) for dim in q_levels.shape) != tuple(bounded_accumulator.logical_shape):
        raise ValueError("q_levels shape must match bounded accumulator logical_shape")
    vote_spec.validate()

    if isinstance(sparse_vote_events, SparseVoteEvents):
        return _execute_direct_bounded_local_vote_update_dense_carrier(
            state_key=state_key,
            q_levels=q_levels,
            bounded_accumulator=bounded_accumulator,
            sparse_carrier=sparse_vote_events,
            vote_spec=vote_spec,
        )

    q_flat = q_levels.detach().cpu().to(torch.int8).flatten().contiguous()
    numel = int(q_flat.numel())
    hot_map, cold_map = _bounded_value_dict(bounded_accumulator)
    default_before = int(bounded_accumulator.cold_default_value)
    dense_vote_authority_used = False
    sparse_carrier = None
    sparse_votes = _sparse_votes_dict_from_mapping(
        sparse_vote_events,
        state_key=state_key,
        numel=numel,
    )
    event_vote_count = len(sparse_votes)
    explicit_support = set(hot_map) | set(cold_map) | set(sparse_votes)

    def vote_at(index: int) -> int:
        return sparse_votes.get(index, 0)
    clip_min = int(vote_spec.accumulator_clip_min)
    clip_max = int(vote_spec.accumulator_clip_max)
    threshold = int(vote_spec.threshold_abs)
    default_after = _clip_i16(
        _truncate_toward_zero_division(
            default_before * int(vote_spec.decay_numerator),
            int(vote_spec.decay_denominator),
        ),
        clip_min,
        clip_max,
    )
    default_direction = 1 if default_after >= threshold else -1 if default_after <= -threshold else 0
    default_mass_crossing_count = 0
    if default_direction != 0:
        support_indices = sorted(explicit_support)
        support_mask = torch.zeros(numel, dtype=torch.bool)
        if support_indices:
            support_mask[torch.tensor(support_indices, dtype=torch.int64)] = True
        if default_direction > 0:
            crossing_mask = q_flat < 1
        else:
            crossing_mask = q_flat > -1
        default_mass_crossing_count = int((~support_mask & crossing_mask).sum().item())

    coverage_domain = {
        "schema": BOUNDED_LOCAL_VOTE_UPDATE_PROOF_SCHEMA_VERSION,
        "state_key": state_key,
        "no_global_cap": True,
        "sparse_vote_events_only": True,
        "supports_replay_ce_veto": False,
        "supports_pc_aux": False,
        "supports_global_backlog": False,
        "supports_default_mass_crossing": False,
        "supports_dense_vote_authority": False,
        "supports_dense_shadow_authority": False,
        "supports_dense_decode_candidate_path": False,
        "supported_decision_dimensions": [
            "local_vote_update",
            "sparse_vote_events",
            "q_changed_identity_count",
            "applied_row_identity",
            "residual_after_threshold",
            "bounded_checkpoint_serialization",
        ],
        "blocked_decision_dimensions": [
            "global_cap",
            "replay_ce_veto",
            "pc_aux",
            "implicit_default_mass_crossing",
        ],
    }
    storage_projection = project_bounded_delta_accumulator_bpw(
        eligible_weight_count=numel,
        hot_exact_row_count=len(bounded_accumulator.hot_exact_indices),
        cold_exception_row_count=len(bounded_accumulator.cold_exception_indices),
        dense_cold_bits_per_weight=0.0,
    )

    if default_mass_crossing_count > 0:
        proof = {
            "schema": BOUNDED_LOCAL_VOTE_UPDATE_PROOF_SCHEMA_VERSION,
            "surface": "accumulator_substitute",
            "scoped_label": None,
            "terminal_classification": INTRINSIC_BOUNDED_UPDATE_DOMAIN_GAP,
            "pass": False,
            "runtime_state_authority_after": "sub2_persistent_hybrid_dense_transient_credit",
            "candidate_dense_decode_used": False,
            "candidate_accumulator_transient_over2_used": False,
            "candidate_vote_transient_over2_used": False,
            "candidate_dense_vote_authority_used": dense_vote_authority_used,
            "dense_oracle_control_used": False,
            "scoped_physical_budget_claim": "not_applicable_domain_gap",
            "q_storage_physical_budget_covered_by_scoped_proof": False,
            "frozen_scale_physical_budget_covered_by_scoped_proof": False,
            "coverage_domain": coverage_domain,
            "domain_gap_dimension": "implicit_default_mass_crossing",
            "domain_gap_detail": (
                "cold default update would create threshold-crossing mass on rows "
                "that are not explicitly enumerated in the bounded support"
            ),
            "default_mass_crossing_count": int(default_mass_crossing_count),
            "event_vote_count": int(event_vote_count),
            "support_row_count": int(len(explicit_support)),
            "hot_exact_row_count_before": int(len(bounded_accumulator.hot_exact_indices)),
            "cold_exception_row_count_before": int(len(bounded_accumulator.cold_exception_indices)),
            "hot_exact_row_count_after": int(len(bounded_accumulator.hot_exact_indices)),
            "cold_exception_row_count_after": int(len(bounded_accumulator.cold_exception_indices)),
            "storage_projection": storage_projection.to_dict(),
        }
        return BoundedDirectLocalUpdateCandidateResult(
            next_bounded_accumulator=bounded_accumulator,
            next_q_levels=q_flat.view_as(q_levels).clone(),
            proof=proof,
        )

    support_after = {}
    candidate_indices: list[int] = []
    for index in sorted(explicit_support):
        old_value = hot_map.get(index, cold_map.get(index, default_before))
        vote_value = vote_at(index)
        decayed = _truncate_toward_zero_division(
            old_value * int(vote_spec.decay_numerator),
            int(vote_spec.decay_denominator),
        )
        new_value = _clip_i16(decayed + vote_value, clip_min, clip_max)
        support_after[index] = int(new_value)
        q_value = int(q_flat[int(index)].item())
        if (new_value >= threshold and q_value < 1) or (new_value <= -threshold and q_value > -1):
            candidate_indices.append(index)

    max_flips = int(vote_spec.max_flips(numel))
    ordered_candidates = sorted(
        (int(index) for index in candidate_indices),
        key=lambda index: (-abs(int(support_after[index])), int(index)),
    )
    applied_indices = tuple(ordered_candidates[:max_flips])
    q_after = q_flat.clone()
    residual_after_threshold: dict[int, int] = {}
    applied_directions_by_index: dict[int, int] = {}
    applied_thresholds_by_index: dict[int, int] = {}
    for index in applied_indices:
        direction = 1 if int(support_after[index]) >= threshold else -1
        applied_directions_by_index[index] = int(direction)
        applied_thresholds_by_index[index] = int(threshold)
        q_after[index] = int(max(-1, min(1, int(q_after[index].item()) + direction)))
        residual = int(support_after[index]) - (direction * threshold)
        residual = _clip_i16(residual, -threshold + 1, threshold - 1)
        support_after[index] = residual
        residual_after_threshold[index] = residual

    next_hot_values = tuple(
        int(support_after[int(index)])
        for index in bounded_accumulator.hot_exact_indices
    )
    next_cold_exception_indices = tuple(
        int(index)
        for index in sorted(idx for idx in support_after if idx not in hot_map and int(support_after[idx]) != default_after)
    )
    next_cold_exception_values = tuple(
        int(support_after[index])
        for index in next_cold_exception_indices
    )
    next_bounded = BoundedDeltaAccumulatorState(
        logical_shape=bounded_accumulator.logical_shape,
        cold_default_value=int(default_after),
        hot_exact_indices=tuple(int(index) for index in bounded_accumulator.hot_exact_indices),
        hot_exact_values=next_hot_values,
        cold_exception_indices=next_cold_exception_indices,
        cold_exception_values=next_cold_exception_values,
        candidate_name=bounded_accumulator.candidate_name,
        raw_arrays_included=False,
    )
    next_projection = project_bounded_delta_accumulator_bpw(
        eligible_weight_count=numel,
        hot_exact_row_count=len(next_bounded.hot_exact_indices),
        cold_exception_row_count=len(next_bounded.cold_exception_indices),
        dense_cold_bits_per_weight=0.0,
    )
    changed = torch.nonzero(q_flat != q_after, as_tuple=False).flatten()
    q_changed_indices = tuple(sorted(int(index.item()) for index in changed))
    accumulator_physical_sub2_pass = (
        float(next_projection.bounded_delta_acc_bits_per_weight) < 2.0
    )
    scoped_positive_label = (
        ACCUMULATOR_SUBSTITUTE_LOCAL_VOTE_UPDATE_EXECUTABLE
        if accumulator_physical_sub2_pass
        else ALGORITHMIC_LOCAL_VOTE_UPDATE_EXECUTABLE_NOT_PHYSICAL_SUB2
    )
    proof = {
        "schema": BOUNDED_LOCAL_VOTE_UPDATE_PROOF_SCHEMA_VERSION,
        "surface": "accumulator_substitute",
        "scoped_label": scoped_positive_label,
        "terminal_classification": scoped_positive_label,
        "pass": True,
        "runtime_state_authority_after": "sub2_persistent_hybrid_dense_transient_credit",
        "candidate_dense_decode_used": False,
        "candidate_accumulator_transient_over2_used": False,
        "candidate_vote_transient_over2_used": False,
        "candidate_dense_vote_authority_used": dense_vote_authority_used,
        "dense_oracle_control_used": False,
        "scoped_physical_budget_claim": (
            "physical_sub2_budgeted"
            if accumulator_physical_sub2_pass
            else "algorithmic_only_not_physical_sub2"
        ),
        "q_storage_physical_budget_covered_by_scoped_proof": False,
        "frozen_scale_physical_budget_covered_by_scoped_proof": False,
        "coverage_domain": coverage_domain,
        "domain_gap_dimension": None,
        "domain_gap_detail": None,
        "default_mass_crossing_count": 0,
        "event_vote_count": int(event_vote_count),
        "candidate_count": int(len(candidate_indices)),
        "max_flips": int(max_flips),
        "pre_veto_selected_flip_count": int(len(applied_indices)),
        "support_row_count": int(len(explicit_support)),
        "hot_exact_row_count_before": int(len(bounded_accumulator.hot_exact_indices)),
        "cold_exception_row_count_before": int(len(bounded_accumulator.cold_exception_indices)),
        "hot_exact_row_count_after": int(len(next_bounded.hot_exact_indices)),
        "cold_exception_row_count_after": int(len(next_bounded.cold_exception_indices)),
        "q_changed_count": int(len(q_changed_indices)),
        "q_changed_identities_sha256": _identity_sha256(
            {(state_key, int(index)) for index in q_changed_indices},
        ),
        "applied_row_count": int(len(applied_indices)),
        "applied_row_identities_sha256": _identity_sha256(
            {(state_key, int(index)) for index in applied_indices},
        ),
        "ordered_applied_row_identities_sha256": _ordered_identity_sha256(
            state_key,
            applied_indices,
        ),
        "applied_directions_sha256": _ordered_value_sha256(
            state_key,
            "direction",
            applied_directions_by_index,
        ),
        "applied_thresholds_sha256": _ordered_value_sha256(
            state_key,
            "threshold",
            applied_thresholds_by_index,
        ),
        "residual_after_threshold_sha256": _sparse_value_sha256(
            state_key,
            residual_after_threshold,
        ),
        "candidate_q_sha256_after": _tensor_sha256(q_after.view_as(q_levels)),
        "storage_projection": next_projection.to_dict(),
        "accumulator_physical_sub2_pass": bool(accumulator_physical_sub2_pass),
        "bounded_accumulator_summary_after": next_bounded.to_dict(),
    }
    return BoundedDirectLocalUpdateCandidateResult(
        next_bounded_accumulator=next_bounded,
        next_q_levels=q_after.view_as(q_levels).to(torch.int8).contiguous(),
        proof=proof,
    )


def _execute_direct_bounded_local_vote_update_reference_3936d74(
    *,
    state_key: str,
    q_levels: torch.Tensor,
    bounded_accumulator: BoundedDeltaAccumulatorState,
    sparse_vote_events: Mapping[int, int],
    vote_spec: VoteUpdateSpec,
) -> BoundedDirectLocalUpdateCandidateResult:
    if q_levels.dtype != torch.int8:
        raise ValueError(f"q_levels must be torch.int8, got {q_levels.dtype}")
    if tuple(int(dim) for dim in q_levels.shape) != tuple(bounded_accumulator.logical_shape):
        raise ValueError("q_levels shape must match bounded accumulator logical_shape")
    vote_spec.validate()

    q_flat = q_levels.detach().cpu().to(torch.int8).flatten().contiguous()
    numel = int(q_flat.numel())
    hot_map, cold_map = _bounded_value_dict(bounded_accumulator)
    default_before = int(bounded_accumulator.cold_default_value)
    dense_vote_authority_used = False
    sparse_votes = {}
    for raw_index, raw_vote in sparse_vote_events.items():
        index = int(raw_index)
        vote = int(raw_vote)
        if index < 0 or index >= numel:
            raise ValueError(f"sparse vote index {index} out of range for {state_key}")
        if vote < -32768 or vote > 32767:
            raise ValueError(f"sparse vote value {vote} must fit int16")
        if vote != 0:
            sparse_votes[index] = vote

    explicit_support = set(hot_map) | set(cold_map) | set(sparse_votes)
    clip_min = int(vote_spec.accumulator_clip_min)
    clip_max = int(vote_spec.accumulator_clip_max)
    threshold = int(vote_spec.threshold_abs)
    default_after = _clip_i16(
        _truncate_toward_zero_division(
            default_before * int(vote_spec.decay_numerator),
            int(vote_spec.decay_denominator),
        ),
        clip_min,
        clip_max,
    )
    default_direction = 1 if default_after >= threshold else -1 if default_after <= -threshold else 0
    default_mass_crossing_count = 0
    if default_direction != 0:
        q_list = q_flat.tolist()
        default_mass_crossing_count = sum(
            1
            for flat_index, q_value in enumerate(q_list)
            if flat_index not in explicit_support
            and (
                (default_direction > 0 and int(q_value) < 1)
                or (default_direction < 0 and int(q_value) > -1)
            )
        )

    coverage_domain = {
        "schema": BOUNDED_LOCAL_VOTE_UPDATE_PROOF_SCHEMA_VERSION,
        "state_key": state_key,
        "no_global_cap": True,
        "sparse_vote_events_only": True,
        "supports_replay_ce_veto": False,
        "supports_pc_aux": False,
        "supports_global_backlog": False,
        "supports_default_mass_crossing": False,
        "supports_dense_vote_authority": False,
        "supports_dense_shadow_authority": False,
        "supports_dense_decode_candidate_path": False,
        "supported_decision_dimensions": [
            "local_vote_update",
            "sparse_vote_events",
            "q_changed_identity_count",
            "applied_row_identity",
            "residual_after_threshold",
            "bounded_checkpoint_serialization",
        ],
        "blocked_decision_dimensions": [
            "global_cap",
            "replay_ce_veto",
            "pc_aux",
            "implicit_default_mass_crossing",
        ],
    }
    storage_projection = project_bounded_delta_accumulator_bpw(
        eligible_weight_count=numel,
        hot_exact_row_count=len(bounded_accumulator.hot_exact_indices),
        cold_exception_row_count=len(bounded_accumulator.cold_exception_indices),
        dense_cold_bits_per_weight=0.0,
    )

    if default_mass_crossing_count > 0:
        proof = {
            "schema": BOUNDED_LOCAL_VOTE_UPDATE_PROOF_SCHEMA_VERSION,
            "surface": "accumulator_substitute",
            "scoped_label": None,
            "terminal_classification": INTRINSIC_BOUNDED_UPDATE_DOMAIN_GAP,
            "pass": False,
            "runtime_state_authority_after": "sub2_persistent_hybrid_dense_transient_credit",
            "candidate_dense_decode_used": False,
            "candidate_accumulator_transient_over2_used": False,
            "candidate_vote_transient_over2_used": False,
            "candidate_dense_vote_authority_used": dense_vote_authority_used,
            "dense_oracle_control_used": False,
            "scoped_physical_budget_claim": "not_applicable_domain_gap",
            "q_storage_physical_budget_covered_by_scoped_proof": False,
            "frozen_scale_physical_budget_covered_by_scoped_proof": False,
            "coverage_domain": coverage_domain,
            "domain_gap_dimension": "implicit_default_mass_crossing",
            "domain_gap_detail": (
                "cold default update would create threshold-crossing mass on rows "
                "that are not explicitly enumerated in the bounded support"
            ),
            "default_mass_crossing_count": int(default_mass_crossing_count),
            "event_vote_count": int(len(sparse_votes)),
            "support_row_count": int(len(explicit_support)),
            "hot_exact_row_count_before": int(len(bounded_accumulator.hot_exact_indices)),
            "cold_exception_row_count_before": int(len(bounded_accumulator.cold_exception_indices)),
            "hot_exact_row_count_after": int(len(bounded_accumulator.hot_exact_indices)),
            "cold_exception_row_count_after": int(len(bounded_accumulator.cold_exception_indices)),
            "storage_projection": storage_projection.to_dict(),
        }
        return BoundedDirectLocalUpdateCandidateResult(
            next_bounded_accumulator=bounded_accumulator,
            next_q_levels=q_flat.view_as(q_levels).clone(),
            proof=proof,
        )

    support_after = {}
    candidate_indices: list[int] = []
    q_before = q_flat.tolist()
    for index in sorted(explicit_support):
        old_value = hot_map.get(index, cold_map.get(index, default_before))
        vote_value = sparse_votes.get(index, 0)
        decayed = _truncate_toward_zero_division(
            old_value * int(vote_spec.decay_numerator),
            int(vote_spec.decay_denominator),
        )
        new_value = _clip_i16(decayed + vote_value, clip_min, clip_max)
        support_after[index] = int(new_value)
        q_value = int(q_before[index])
        if (new_value >= threshold and q_value < 1) or (new_value <= -threshold and q_value > -1):
            candidate_indices.append(index)

    max_flips = int(vote_spec.max_flips(numel))
    ordered_candidates = sorted(
        (int(index) for index in candidate_indices),
        key=lambda index: (-abs(int(support_after[index])), int(index)),
    )
    applied_indices = tuple(ordered_candidates[:max_flips])
    q_after = q_flat.clone()
    residual_after_threshold: dict[int, int] = {}
    applied_directions_by_index: dict[int, int] = {}
    applied_thresholds_by_index: dict[int, int] = {}
    for index in applied_indices:
        direction = 1 if int(support_after[index]) >= threshold else -1
        applied_directions_by_index[index] = int(direction)
        applied_thresholds_by_index[index] = int(threshold)
        q_after[index] = int(max(-1, min(1, int(q_after[index].item()) + direction)))
        residual = int(support_after[index]) - (direction * threshold)
        residual = _clip_i16(residual, -threshold + 1, threshold - 1)
        support_after[index] = residual
        residual_after_threshold[index] = residual

    next_hot_values = tuple(
        int(support_after[int(index)])
        for index in bounded_accumulator.hot_exact_indices
    )
    next_cold_exception_indices = tuple(
        int(index)
        for index in sorted(idx for idx in support_after if idx not in hot_map and int(support_after[idx]) != default_after)
    )
    next_cold_exception_values = tuple(
        int(support_after[index])
        for index in next_cold_exception_indices
    )
    next_bounded = BoundedDeltaAccumulatorState(
        logical_shape=bounded_accumulator.logical_shape,
        cold_default_value=int(default_after),
        hot_exact_indices=tuple(int(index) for index in bounded_accumulator.hot_exact_indices),
        hot_exact_values=next_hot_values,
        cold_exception_indices=next_cold_exception_indices,
        cold_exception_values=next_cold_exception_values,
        candidate_name=bounded_accumulator.candidate_name,
        raw_arrays_included=False,
    )
    next_projection = project_bounded_delta_accumulator_bpw(
        eligible_weight_count=numel,
        hot_exact_row_count=len(next_bounded.hot_exact_indices),
        cold_exception_row_count=len(next_bounded.cold_exception_indices),
        dense_cold_bits_per_weight=0.0,
    )
    q_changed_indices = tuple(
        index
        for index, (before, after) in enumerate(zip(q_before, q_after.tolist()))
        if int(before) != int(after)
    )
    accumulator_physical_sub2_pass = (
        float(next_projection.bounded_delta_acc_bits_per_weight) < 2.0
    )
    scoped_positive_label = (
        ACCUMULATOR_SUBSTITUTE_LOCAL_VOTE_UPDATE_EXECUTABLE
        if accumulator_physical_sub2_pass
        else ALGORITHMIC_LOCAL_VOTE_UPDATE_EXECUTABLE_NOT_PHYSICAL_SUB2
    )
    proof = {
        "schema": BOUNDED_LOCAL_VOTE_UPDATE_PROOF_SCHEMA_VERSION,
        "surface": "accumulator_substitute",
        "scoped_label": scoped_positive_label,
        "terminal_classification": scoped_positive_label,
        "pass": True,
        "runtime_state_authority_after": "sub2_persistent_hybrid_dense_transient_credit",
        "candidate_dense_decode_used": False,
        "candidate_accumulator_transient_over2_used": False,
        "candidate_vote_transient_over2_used": False,
        "candidate_dense_vote_authority_used": dense_vote_authority_used,
        "dense_oracle_control_used": False,
        "scoped_physical_budget_claim": (
            "physical_sub2_budgeted"
            if accumulator_physical_sub2_pass
            else "algorithmic_only_not_physical_sub2"
        ),
        "q_storage_physical_budget_covered_by_scoped_proof": False,
        "frozen_scale_physical_budget_covered_by_scoped_proof": False,
        "coverage_domain": coverage_domain,
        "domain_gap_dimension": None,
        "domain_gap_detail": None,
        "default_mass_crossing_count": 0,
        "event_vote_count": int(len(sparse_votes)),
        "candidate_count": int(len(candidate_indices)),
        "max_flips": int(max_flips),
        "pre_veto_selected_flip_count": int(len(applied_indices)),
        "support_row_count": int(len(explicit_support)),
        "hot_exact_row_count_before": int(len(bounded_accumulator.hot_exact_indices)),
        "cold_exception_row_count_before": int(len(bounded_accumulator.cold_exception_indices)),
        "hot_exact_row_count_after": int(len(next_bounded.hot_exact_indices)),
        "cold_exception_row_count_after": int(len(next_bounded.cold_exception_indices)),
        "q_changed_count": int(len(q_changed_indices)),
        "q_changed_identities_sha256": _identity_sha256(
            {(state_key, int(index)) for index in q_changed_indices},
        ),
        "applied_row_count": int(len(applied_indices)),
        "applied_row_identities_sha256": _identity_sha256(
            {(state_key, int(index)) for index in applied_indices},
        ),
        "ordered_applied_row_identities_sha256": _ordered_identity_sha256(
            state_key,
            applied_indices,
        ),
        "applied_directions_sha256": _ordered_value_sha256(
            state_key,
            "direction",
            applied_directions_by_index,
        ),
        "applied_thresholds_sha256": _ordered_value_sha256(
            state_key,
            "threshold",
            applied_thresholds_by_index,
        ),
        "residual_after_threshold_sha256": _sparse_value_sha256(
            state_key,
            residual_after_threshold,
        ),
        "candidate_q_sha256_after": _tensor_sha256(q_after.view_as(q_levels)),
        "storage_projection": next_projection.to_dict(),
        "accumulator_physical_sub2_pass": bool(accumulator_physical_sub2_pass),
        "bounded_accumulator_summary_after": next_bounded.to_dict(),
    }
    return BoundedDirectLocalUpdateCandidateResult(
        next_bounded_accumulator=next_bounded,
        next_q_levels=q_after.view_as(q_levels).to(torch.int8).contiguous(),
        proof=proof,
    )


def bounded_delta_candidate_assessment(
    *,
    candidate_name: str = HYBRID_HOT_EXACT_COLD_DEFAULT_CANDIDATE,
) -> CandidateAssessment:
    contract = bounded_delta_admission_contract(candidate_name=candidate_name)
    return candidate_assessment(
        candidate_name=contract.candidate_name,
        classification=CandidateClassification.BOUNDED_DELTA_WITH_REPORT,
        covered_decision_dimensions=required_decision_dimension_names(),
        compressed_representation=True,
        bounded_delta_hypothesis=contract.capacity_hypothesis,
        guardrail=contract.allowed_divergence_contract,
        preserved_information=contract.preserved_information,
        sub2_persistent_strategy=contract.sub2_persistent_strategy,
        note="adapter/oracle reference only; no production vote/cap replacement",
    )


def bounded_delta_admission_contract(
    *,
    candidate_name: str = HYBRID_HOT_EXACT_COLD_DEFAULT_CANDIDATE,
) -> BoundedDeltaAdmissionContract:
    if candidate_name == HYBRID_HOT_EXACT_COLD_DEFAULT_CANDIDATE:
        return BoundedDeltaAdmissionContract(
            candidate_name=candidate_name,
            preserved_information=(
                "exact residual magnitude/sign on hot frontier rows",
                "accepted/deferred identity carry plus backlog continuity",
                "exact post-threshold residual state on applied and vetoed rows",
            ),
            capacity_hypothesis=(
                "preserve the exact frontier and backlog rows the dense control uses "
                "to decide threshold crossings while collapsing cold rows to a shared "
                "default plus sparse exceptions"
            ),
            sub2_persistent_strategy=(
                "charge exact hot rows, sparse cold exceptions, and backlog metadata "
                "under the inclusive q+scale+acc ledger; no dense cold field"
            ),
            exact_surfaces=(
                "candidate_mask",
                "accepted_rows",
                "deferred_rows",
                "final_q_changes",
                "backlog_carry",
                "cap_frontier_rank_delta",
                "hot_risk_rows",
                "accumulator_residuals",
            ),
            allowed_divergence_contract=(
                "none; this candidate is admitted only if it preserves the dense "
                "control exactly on every preregistered surface"
            ),
            max_cap_frontier_rank_delta=0,
        )
    if candidate_name == EVENT_CODED_CROSSING_RESIDUAL_LOG_CANDIDATE:
        return BoundedDeltaAdmissionContract(
            candidate_name=candidate_name,
            preserved_information=(
                "threshold-crossing event identity and direction",
                "residual state on rows that actually fired",
                "accepted/deferred identity carry plus backlog continuity",
            ),
            capacity_hypothesis=(
                "learning is carried by sparse crossing events and deferred carry, "
                "so non-fired cold residual magnitude may be dropped if decisive "
                "surfaces remain exact"
            ),
            sub2_persistent_strategy=(
                "persist only event rows plus backlog metadata and keep the cold "
                "field implicit under the inclusive q+scale+acc ledger"
            ),
            exact_surfaces=(
                "accepted_rows",
                "deferred_rows",
                "final_q_changes",
                "backlog_carry",
                "cap_frontier_rank_delta",
                "hot_risk_rows",
            ),
            allowed_divergence_contract=(
                "candidate-mask drift and non-fired cold residual drift are allowed "
                "only when accepted/deferred/q/backlog/frontier-hot surfaces stay exact"
            ),
            max_cap_frontier_rank_delta=0,
        )
    if candidate_name == COARSE_SIGNED_CHARGE_SPARSE_FRONTIER_CANDIDATE:
        return BoundedDeltaAdmissionContract(
            candidate_name=candidate_name,
            preserved_information=(
                "coarse ubiquitous signed pressure toward threshold",
                "exact sparse frontier overrides on rows near decision boundaries",
                "accepted/deferred identity carry plus backlog continuity",
            ),
            capacity_hypothesis=(
                "weak evidence must accumulate everywhere, but only the sparse "
                "frontier requires exact residual state to preserve dense decisions"
            ),
            sub2_persistent_strategy=(
                "charge a coarse dense cold field plus sparse frontier overrides and "
                "backlog metadata under the inclusive q+scale+acc ledger"
            ),
            exact_surfaces=(
                "accepted_rows",
                "deferred_rows",
                "final_q_changes",
                "backlog_carry",
                "hot_risk_rows",
                "accumulator_residuals",
            ),
            allowed_divergence_contract=(
                "cold candidate-mask density may drift away from the active frontier "
                "and non-decisive cap-frontier reordering is allowed only up to "
                "cap_frontier_rank_delta<=1"
            ),
            max_cap_frontier_rank_delta=1,
        )
    raise ValueError(f"unsupported bounded-delta candidate_name {candidate_name!r}")


@dataclass(frozen=True)
class _BoundedDeltaAdmissionEvaluation:
    admission_passed: bool
    failed_surfaces: tuple[str, ...]
    rejection_telemetry: BoundedDeltaRejectionTelemetry


def _detail_for_fraction(
    *,
    count: int,
    union_count: int,
    fraction: float,
) -> str:
    return f"changed={int(count)} union={int(union_count)} fraction={float(fraction):.6f}"


def _evaluate_bounded_delta_admission(
    contract: BoundedDeltaAdmissionContract,
    measured_report: BoundedDeltaMeasuredReport,
) -> _BoundedDeltaAdmissionEvaluation:
    decisive_zero = (
        measured_report.accepted_changed_count == 0
        and measured_report.deferred_changed_count == 0
        and measured_report.q_changed_count == 0
        and measured_report.backlog_key_changed_count == 0
        and measured_report.hot_risk_changed_count == 0
    )
    residual_exact = (
        bool(measured_report.accumulator_residual_hash_match)
        and int(measured_report.max_abs_acc_error) == 0
    )
    candidate_mask_allowed = (
        contract.candidate_name
        in {
            EVENT_CODED_CROSSING_RESIDUAL_LOG_CANDIDATE,
            COARSE_SIGNED_CHARGE_SPARSE_FRONTIER_CANDIDATE,
        }
        and decisive_zero
        and int(measured_report.cap_frontier_rank_delta) <= int(contract.max_cap_frontier_rank_delta)
    )
    residual_only_allowed = (
        contract.candidate_name == EVENT_CODED_CROSSING_RESIDUAL_LOG_CANDIDATE
        and decisive_zero
        and int(measured_report.cap_frontier_rank_delta) == 0
        and int(measured_report.fired_or_accepted_residual_changed_count) == 0
        and int(measured_report.hot_residual_changed_count) == 0
    )

    surfaces: list[BoundedDeltaRejectionSurface] = []
    failed_surfaces: list[str] = []

    def add_surface(
        *,
        surface: str,
        observed: int | float | bool | str,
        threshold: int | float | bool | str,
        status: str,
        detail: str,
    ) -> None:
        surfaces.append(
            BoundedDeltaRejectionSurface(
                surface=surface,
                observed=observed,
                threshold=threshold,
                status=status,
                detail=detail,
            )
        )
        if status not in {
            "pass",
            "allowed_non_decisive_divergence",
            "allowed_non_fired_cold_residual_divergence",
        }:
            failed_surfaces.append(surface)

    candidate_fraction = float(measured_report.candidate_changed_fraction)
    if math.isclose(candidate_fraction, 0.0, abs_tol=1e-12):
        add_surface(
            surface="candidate_mask",
            observed=candidate_fraction,
            threshold="exact 0.0",
            status="pass",
            detail=_detail_for_fraction(
                count=measured_report.candidate_changed_count,
                union_count=measured_report.candidate_union_count,
                fraction=candidate_fraction,
            ),
        )
    elif candidate_mask_allowed:
        add_surface(
            surface="candidate_mask",
            observed=candidate_fraction,
            threshold=contract.allowed_divergence_contract,
            status="allowed_non_decisive_divergence",
            detail=(
                "candidate-mask drift stayed off decisive surfaces; "
                + _detail_for_fraction(
                    count=measured_report.candidate_changed_count,
                    union_count=measured_report.candidate_union_count,
                    fraction=candidate_fraction,
                )
            ),
        )
    else:
        add_surface(
            surface="candidate_mask",
            observed=candidate_fraction,
            threshold="exact 0.0",
            status="exact_surface_miss",
            detail=_detail_for_fraction(
                count=measured_report.candidate_changed_count,
                union_count=measured_report.candidate_union_count,
                fraction=candidate_fraction,
            ),
        )

    accepted_fraction = float(measured_report.accepted_changed_fraction)
    if math.isclose(accepted_fraction, 0.0, abs_tol=1e-12):
        add_surface(
            surface="accepted_rows",
            observed=accepted_fraction,
            threshold="exact 0.0",
            status="pass",
            detail=_detail_for_fraction(
                count=measured_report.accepted_changed_count,
                union_count=measured_report.accepted_union_count,
                fraction=accepted_fraction,
            ),
        )
    else:
        status = "destructive_approximation"
        detail = _detail_for_fraction(
            count=measured_report.accepted_changed_count,
            union_count=measured_report.accepted_union_count,
            fraction=accepted_fraction,
        )
        if (
            contract.candidate_name == COARSE_SIGNED_CHARGE_SPARSE_FRONTIER_CANDIDATE
            and measured_report.q_changed_count == 0
            and measured_report.hot_risk_changed_count == 0
        ):
            status = "revisit_divergence_contract"
            detail = (
                f"would have flipped {int(measured_report.accepted_changed_count)} accepted rows "
                "under the dense control without q mutation; revisit whether this is a "
                "capacity signal or a contract-breaking approximation"
            )
        add_surface(
            surface="accepted_rows",
            observed=accepted_fraction,
            threshold="exact 0.0",
            status=status,
            detail=detail,
        )

    deferred_fraction = float(measured_report.deferred_changed_fraction)
    if math.isclose(deferred_fraction, 0.0, abs_tol=1e-12):
        add_surface(
            surface="deferred_rows",
            observed=deferred_fraction,
            threshold="exact 0.0",
            status="pass",
            detail=_detail_for_fraction(
                count=measured_report.deferred_changed_count,
                union_count=measured_report.deferred_union_count,
                fraction=deferred_fraction,
            ),
        )
    else:
        status = "destructive_approximation"
        detail = _detail_for_fraction(
            count=measured_report.deferred_changed_count,
            union_count=measured_report.deferred_union_count,
            fraction=deferred_fraction,
        )
        if (
            contract.candidate_name == COARSE_SIGNED_CHARGE_SPARSE_FRONTIER_CANDIDATE
            and measured_report.q_changed_count == 0
            and measured_report.hot_risk_changed_count == 0
        ):
            status = "revisit_divergence_contract"
            detail = (
                f"would have changed {int(measured_report.deferred_changed_count)} deferred rows "
                "without q mutation; revisit whether this is a capacity signal or drift"
            )
        add_surface(
            surface="deferred_rows",
            observed=deferred_fraction,
            threshold="exact 0.0",
            status=status,
            detail=detail,
        )

    q_fraction = float(measured_report.q_changed_fraction)
    add_surface(
        surface="final_q_changes",
        observed=q_fraction,
        threshold="exact 0.0",
        status="pass" if math.isclose(q_fraction, 0.0, abs_tol=1e-12) else "destructive_approximation",
        detail=_detail_for_fraction(
            count=measured_report.q_changed_count,
            union_count=measured_report.q_changed_union_count,
            fraction=q_fraction,
        ),
    )

    backlog_fraction = float(measured_report.backlog_key_changed_fraction)
    add_surface(
        surface="backlog_carry",
        observed=backlog_fraction,
        threshold="exact 0.0",
        status=(
            "pass"
            if math.isclose(backlog_fraction, 0.0, abs_tol=1e-12)
            else "destructive_approximation"
        ),
        detail=_detail_for_fraction(
            count=measured_report.backlog_key_changed_count,
            union_count=measured_report.backlog_key_union_count,
            fraction=backlog_fraction,
        ),
    )

    cap_rank_delta = int(measured_report.cap_frontier_rank_delta)
    if cap_rank_delta <= int(contract.max_cap_frontier_rank_delta):
        add_surface(
            surface="cap_frontier_rank_delta",
            observed=cap_rank_delta,
            threshold=int(contract.max_cap_frontier_rank_delta),
            status=(
                "pass"
                if cap_rank_delta == 0
                else "allowed_non_decisive_divergence"
            ),
            detail=(
                f"cap_frontier_rank_delta={cap_rank_delta}"
                if cap_rank_delta == 0
                else f"reordered non-decisive ranks within preregistered ceiling {contract.max_cap_frontier_rank_delta}"
            ),
        )
    else:
        add_surface(
            surface="cap_frontier_rank_delta",
            observed=cap_rank_delta,
            threshold=int(contract.max_cap_frontier_rank_delta),
            status=(
                "revisit_divergence_contract"
                if decisive_zero
                else "destructive_approximation"
            ),
            detail=(
                f"reordered non-decisive ranks beyond preregistered ceiling; delta={cap_rank_delta}"
                if decisive_zero
                else f"cap-frontier rank delta exceeded ceiling with decisive drift; delta={cap_rank_delta}"
            ),
        )

    hot_risk_count = int(measured_report.hot_risk_changed_count)
    add_surface(
        surface="hot_risk_rows",
        observed=hot_risk_count,
        threshold=0,
        status="pass" if hot_risk_count == 0 else "destructive_approximation",
        detail=f"hot_risk_changed_count={hot_risk_count}",
    )

    residual_detail = (
        f"hash_match={bool(measured_report.accumulator_residual_hash_match)} "
        f"max_abs_acc_error={int(measured_report.max_abs_acc_error)} "
        f"p95_abs_acc_error={float(measured_report.p95_abs_acc_error):.6f} "
        f"fired_or_accepted_residual_changed_count={int(measured_report.fired_or_accepted_residual_changed_count)} "
        f"hot_residual_changed_count={int(measured_report.hot_residual_changed_count)}"
    )
    if residual_exact:
        add_surface(
            surface="accumulator_residuals",
            observed=True,
            threshold="exact hash match and zero error",
            status="pass",
            detail=residual_detail,
        )
    elif residual_only_allowed:
        add_surface(
            surface="accumulator_residuals",
            observed=False,
            threshold=contract.allowed_divergence_contract,
            status="allowed_non_fired_cold_residual_divergence",
            detail=(
                "residual mismatch stayed off fired/accepted and hot rows; "
                + residual_detail
            ),
        )
    else:
        if int(measured_report.fired_or_accepted_residual_changed_count) > 0:
            status = "fired_or_accepted_residual_drift"
            detail = (
                "residual mismatch touched fired/accepted rows; "
                + residual_detail
            )
        elif int(measured_report.hot_residual_changed_count) > 0:
            status = "hot_residual_drift"
            detail = "residual mismatch touched hot rows; " + residual_detail
        else:
            status = "exact_surface_miss"
            detail = residual_detail
        add_surface(
            surface="accumulator_residuals",
            observed=False,
            threshold="exact hash match and zero error",
            status=status,
            detail=detail,
        )

    failed = tuple(failed_surfaces)
    if not failed:
        summary = "admission_pass"
    elif any(
        item.status == "fired_or_accepted_residual_drift"
        for item in surfaces
        if item.surface in failed
    ):
        summary = "fired_or_accepted_residual_drift"
    elif any(
        item.status == "hot_residual_drift"
        for item in surfaces
        if item.surface in failed
    ):
        summary = "hot_residual_drift"
    elif any(
        item.status == "destructive_approximation"
        for item in surfaces
        if item.surface in failed
    ):
        summary = "destructive_approximation"
    elif any(
        item.status == "revisit_divergence_contract"
        for item in surfaces
        if item.surface in failed
    ):
        summary = "revisit_divergence_contract"
    else:
        summary = "exact_surface_miss"
    telemetry = BoundedDeltaRejectionTelemetry(
        candidate_name=contract.candidate_name,
        admission_passed=not failed,
        summary=summary,
        failed_surfaces=failed,
        surfaces=tuple(surfaces),
    )
    return _BoundedDeltaAdmissionEvaluation(
        admission_passed=not failed,
        failed_surfaces=failed,
        rejection_telemetry=telemetry,
    )


@dataclass(frozen=True)
class _PathResult:
    plans: dict[str, VoteUpdatePlan]
    candidate_ids: set[tuple[str, int]]
    candidate_direction_by_id: dict[tuple[str, int], int]
    accepted_ids: set[tuple[str, int]]
    fired_ids: set[tuple[str, int]]
    deferred_ids: set[tuple[str, int]]
    q_changed_ids: set[tuple[str, int]]
    backlog_ids: set[tuple[str, int]]
    ordered_row_ids: list[tuple[str, int]]
    output_q_by_key: dict[str, torch.Tensor]
    output_acc_by_key: dict[str, torch.Tensor]
    cap_result: GlobalRateCapResult | None
    local_results: dict[str, VoteUpdateResult]


def _candidate_direction_map(
    state_key: str,
    plan: VoteUpdatePlan,
) -> dict[tuple[str, int], int]:
    flat_new_acc = plan.new_acc_i32.flatten().to(torch.int32)
    out: dict[tuple[str, int], int] = {}
    for idx in plan.candidate_indices.detach().cpu().to(torch.int64).tolist():
        value = int(flat_new_acc[int(idx)].item())
        out[(state_key, int(idx))] = 1 if value >= 0 else -1
    return out


def _run_reference_path(
    inputs: Sequence[BoundedDeltaOracleInput],
    *,
    states_by_key: dict[str, VoteUpdateState],
    global_cap_spec: GlobalRateCapSpec | None,
    deferred_backlog: dict[str, dict[int, dict[str, int]]] | None,
    tensor_offsets: dict[str, int] | None,
) -> _PathResult:
    plans: dict[str, VoteUpdatePlan] = {}
    cap_inputs: list[GlobalRateCapTensorInput] = []
    candidate_ids: set[tuple[str, int]] = set()
    candidate_direction_by_id: dict[tuple[str, int], int] = {}
    for item in inputs:
        state = states_by_key[item.state_key]
        plan = plan_integer_vote_update_reference(state, item.vote_inputs, item.vote_spec)
        plans[item.state_key] = plan
        candidate_ids |= _ids_from_indices(item.state_key, plan.candidate_indices)
        candidate_direction_by_id.update(_candidate_direction_map(item.state_key, plan))
        cap_inputs.append(GlobalRateCapTensorInput(item.state_key, state, plan))

    cap_result: GlobalRateCapResult | None = None
    local_results: dict[str, VoteUpdateResult] = {}
    output_q: dict[str, torch.Tensor] = {}
    output_acc: dict[str, torch.Tensor] = {}
    accepted_ids: set[tuple[str, int]] = set()
    fired_ids: set[tuple[str, int]] = set()
    deferred_ids: set[tuple[str, int]] = set()
    q_changed_ids: set[tuple[str, int]] = set()
    ordered_row_ids: list[tuple[str, int]] = []
    if global_cap_spec is not None:
        offsets = tensor_offsets or tensor_offsets_for_vote_update_states(cap_inputs)
        cap_result = apply_global_rate_cap_reference(
            cap_inputs,
            global_cap_spec,
            deferred_backlog=deferred_backlog,
            tensor_offsets=offsets,
        )
        accepted_ids = {(row.state_key, int(row.flat_index)) for row in cap_result.accepted_rows}
        fired_ids = set(accepted_ids)
        for item in inputs:
            fired_ids |= _ids_from_indices(
                item.state_key,
                plans[item.state_key].replay_ce_veto_indices,
            )
        deferred_ids = {(row.state_key, int(row.flat_index)) for row in cap_result.deferred_rows}
        ordered_row_ids = [(row.state_key, int(row.flat_index)) for row in cap_result.rows]
        result_by_key = {result.state_key: result for result in cap_result.tensor_results}
        for item in inputs:
            result = result_by_key[item.state_key]
            output_q[item.state_key] = result.q_levels
            output_acc[item.state_key] = result.accumulators
            changed = torch.nonzero(
                result.q_levels.flatten() != states_by_key[item.state_key].q_levels.flatten(),
                as_tuple=False,
            ).flatten()
            q_changed_ids |= _ids_from_indices(item.state_key, changed)
        backlog_ids = _backlog_key_set(cap_result.deferred_backlog)
    else:
        backlog_ids = _backlog_key_set(deferred_backlog or {})
        for item in inputs:
            result = apply_integer_vote_update_reference(
                states_by_key[item.state_key],
                item.vote_inputs,
                item.vote_spec,
            )
            local_results[item.state_key] = result
            output_q[item.state_key] = result.q_levels
            output_acc[item.state_key] = result.accumulators
            accepted_ids |= _ids_from_indices(item.state_key, result.plan.applied_indices)
            fired_ids |= _ids_from_indices(item.state_key, result.plan.applied_indices)
            fired_ids |= _ids_from_indices(item.state_key, result.plan.replay_ce_veto_indices)
            ordered_row_ids.extend(
                (item.state_key, int(idx))
                for idx in result.plan.applied_indices.detach().cpu().to(torch.int64).tolist()
            )
            changed = torch.nonzero(
                result.q_levels.flatten() != states_by_key[item.state_key].q_levels.flatten(),
                as_tuple=False,
            ).flatten()
            q_changed_ids |= _ids_from_indices(item.state_key, changed)

    return _PathResult(
        plans=plans,
        candidate_ids=candidate_ids,
        candidate_direction_by_id=candidate_direction_by_id,
        accepted_ids=accepted_ids,
        fired_ids=fired_ids,
        deferred_ids=deferred_ids,
        q_changed_ids=q_changed_ids,
        backlog_ids=backlog_ids,
        ordered_row_ids=ordered_row_ids,
        output_q_by_key=output_q,
        output_acc_by_key=output_acc,
        cap_result=cap_result,
        local_results=local_results,
    )


def _evaluate_guardrail(
    guard_spec: BoundedDeltaGuardSpec,
    measured_report: BoundedDeltaMeasuredReport,
) -> BoundedDeltaGuardEvaluation:
    guard_spec.validate()
    failed: list[str] = []
    if measured_report.candidate_changed_fraction > guard_spec.max_candidate_changed_fraction:
        failed.append("candidate_changed_fraction")
    if measured_report.accepted_changed_fraction > guard_spec.max_accepted_changed_fraction:
        failed.append("accepted_changed_fraction")
    if measured_report.deferred_changed_fraction > guard_spec.max_deferred_changed_fraction:
        failed.append("deferred_changed_fraction")
    if measured_report.q_changed_fraction > guard_spec.max_q_changed_fraction:
        failed.append("q_changed_fraction")
    if measured_report.backlog_key_changed_fraction > guard_spec.max_backlog_key_changed_fraction:
        failed.append("backlog_key_changed_fraction")
    if measured_report.cap_frontier_rank_delta > guard_spec.max_cap_frontier_rank_delta:
        failed.append("cap_frontier_rank_delta")
    if (
        guard_spec.hot_risk_rows_require_zero_drift
        and measured_report.hot_risk_changed_count > 0
    ):
        failed.append("hot_risk_changed_count")
    return BoundedDeltaGuardEvaluation(
        guard_spec=guard_spec,
        measured_report=measured_report,
        guard_passed=not failed,
        failed_metrics=tuple(failed),
    )


def _hash_vote_inputs(inputs: Sequence[BoundedDeltaOracleInput]) -> str:
    h = hashlib.sha256()
    for item in inputs:
        h.update(item.state_key.encode("utf-8"))
        h.update(_tensor_sha256(item.vote_inputs.votes).encode("utf-8"))
        if item.vote_inputs.replay_ce_veto_votes is not None:
            h.update(_tensor_sha256(item.vote_inputs.replay_ce_veto_votes).encode("utf-8"))
        if item.vote_inputs.replay_ce_veto_moves is not None:
            h.update(_tensor_sha256(item.vote_inputs.replay_ce_veto_moves).encode("utf-8"))
    return h.hexdigest()


def _hash_cap_spec(spec: GlobalRateCapSpec | None) -> str:
    if spec is None:
        return "none"
    payload = (
        f"{int(spec.cap)}|{int(spec.step)}|{spec.normalized_ordering_mode.value}|"
        f"{int(spec.ordering_seed)}|{bool(spec.mutate_outputs)}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _hot_identity_set(inputs: Sequence[BoundedDeltaOracleInput]) -> set[tuple[str, int]]:
    return {
        (item.state_key, int(index))
        for item in inputs
        for index in item.hot_exact_indices
    }


def _default_next_candidate_if_failed(candidate_name: str) -> str:
    if candidate_name == HYBRID_HOT_EXACT_COLD_DEFAULT_CANDIDATE:
        return EVENT_CODED_CROSSING_RESIDUAL_LOG_CANDIDATE
    if candidate_name == EVENT_CODED_CROSSING_RESIDUAL_LOG_CANDIDATE:
        return COARSE_SIGNED_CHARGE_SPARSE_FRONTIER_CANDIDATE
    return "none_declared_stop_after_coarse_candidate"


def _build_measured_report_from_paths(
    *,
    inputs: Sequence[BoundedDeltaOracleInput],
    candidate_name: str,
    exact_path: _PathResult,
    bounded_path: _PathResult,
    global_cap_spec: GlobalRateCapSpec | None,
    exact_input_backlog: Mapping[str, Mapping[int, Mapping[str, int]]] | None,
    bounded_input_backlog: Mapping[str, Mapping[int, Mapping[str, int]]] | None,
    bounded_stored_backlog: Mapping[str, Mapping[int, Mapping[str, int]]] | None,
    tensor_offsets: Mapping[str, int] | None,
    bounded_backlog_policy_active: bool,
    path_difference: str,
    exact_input_states: Mapping[str, VoteUpdateState] | None = None,
    bounded_input_states: Mapping[str, VoteUpdateState] | None = None,
    oracle_parity_overrides: Mapping[str, bool | str | int] | None = None,
) -> BoundedDeltaMeasuredReport:
    exact_backlog = exact_input_backlog or {}
    bounded_input = exact_backlog if bounded_input_backlog is None else bounded_input_backlog
    bounded_stored = (
        bounded_stored_backlog
        if bounded_stored_backlog is not None
        else bounded_path.cap_result.deferred_backlog
        if bounded_path.cap_result is not None
        else bounded_input
    )
    exact_backlog_ids = _backlog_key_set(
        exact_path.cap_result.deferred_backlog if exact_path.cap_result is not None else exact_backlog
    )
    bounded_backlog_ids = _backlog_key_set(bounded_stored)
    candidate_changed_count, candidate_fraction = _symmetric_fraction(
        exact_path.candidate_ids,
        bounded_path.candidate_ids,
    )
    accepted_changed_count, accepted_fraction = _symmetric_fraction(
        exact_path.accepted_ids,
        bounded_path.accepted_ids,
    )
    deferred_changed_count, deferred_fraction = _symmetric_fraction(
        exact_path.deferred_ids,
        bounded_path.deferred_ids,
    )
    q_changed_count, q_fraction = _symmetric_fraction(
        exact_path.q_changed_ids,
        bounded_path.q_changed_ids,
    )
    backlog_changed_count, backlog_fraction = _symmetric_fraction(
        exact_backlog_ids,
        bounded_backlog_ids,
    )

    exact_direction = exact_path.candidate_direction_by_id
    bounded_direction = bounded_path.candidate_direction_by_id
    direction_keys = set(exact_direction) | set(bounded_direction)
    direction_changed = sum(
        1 for key in direction_keys if exact_direction.get(key) != bounded_direction.get(key)
    )
    rank_delta = _rank_delta(exact_path.ordered_row_ids, bounded_path.ordered_row_ids)

    exact_hashes: dict[str, str] = {}
    bounded_hashes: dict[str, str] = {}
    acc_errors: list[torch.Tensor] = []
    residual_hash_match = True
    residual_error_ids: set[tuple[str, int]] = set()
    for item in inputs:
        exact_acc = exact_path.output_acc_by_key[item.state_key].detach().cpu().to(torch.int32)
        bounded_acc = bounded_path.output_acc_by_key[item.state_key].detach().cpu().to(torch.int32)
        exact_hash = _tensor_sha256(exact_acc)
        bounded_hash = _tensor_sha256(bounded_acc)
        exact_hashes[item.state_key] = exact_hash
        bounded_hashes[item.state_key] = bounded_hash
        residual_hash_match = residual_hash_match and exact_hash == bounded_hash
        acc_errors.append((exact_acc - bounded_acc).abs().flatten())
        changed = torch.nonzero(
            exact_acc.flatten() != bounded_acc.flatten(),
            as_tuple=False,
        ).flatten()
        residual_error_ids |= _ids_from_indices(item.state_key, changed)
    all_errors = torch.cat(acc_errors) if acc_errors else torch.empty(0, dtype=torch.int32)
    max_abs_error = int(all_errors.max().item()) if int(all_errors.numel()) else 0

    hot_ids = _hot_identity_set(inputs)
    decision_symdiff = (
        (exact_path.candidate_ids ^ bounded_path.candidate_ids)
        | (exact_path.accepted_ids ^ bounded_path.accepted_ids)
        | (exact_path.deferred_ids ^ bounded_path.deferred_ids)
        | (exact_path.q_changed_ids ^ bounded_path.q_changed_ids)
    )
    hot_risk_changed = len(decision_symdiff & hot_ids)
    fired_or_accepted_ids = exact_path.fired_ids | bounded_path.fired_ids
    fired_or_accepted_residual_changed = residual_error_ids & fired_or_accepted_ids
    hot_residual_changed = residual_error_ids & hot_ids

    same_initial_q = True
    if exact_input_states is not None and bounded_input_states is not None:
        same_initial_q = all(
            _tensor_sha256(exact_input_states[state_key].q_levels)
            == _tensor_sha256(bounded_input_states[state_key].q_levels)
            for state_key in exact_input_states
        )
    vote_hash = _hash_vote_inputs(inputs)
    cap_hash = _hash_cap_spec(global_cap_spec)
    offsets_hash = hashlib.sha256(str(sorted((tensor_offsets or {}).items())).encode("utf-8")).hexdigest()
    oracle_parity: dict[str, bool | str | int] = {
        "same_initial_q": same_initial_q,
        "same_votes_sha256": True,
        "votes_sha256": vote_hash,
        "same_cap_spec": True,
        "cap_spec_sha256": cap_hash,
        "same_deferred_backlog": not bounded_backlog_policy_active,
        "bounded_backlog_policy_active": bounded_backlog_policy_active,
        "exact_input_deferred_backlog_count": len(_backlog_key_set(exact_backlog)),
        "bounded_input_deferred_backlog_count": len(_backlog_key_set(bounded_input)),
        "exact_output_deferred_backlog_count": len(exact_backlog_ids),
        "bounded_stored_deferred_backlog_count": len(bounded_backlog_ids),
        "deferred_backlog_keys_sha256": _backlog_keys_sha256(exact_backlog),
        "exact_input_deferred_backlog_keys_sha256": _backlog_keys_sha256(exact_backlog),
        "bounded_input_deferred_backlog_keys_sha256": _backlog_keys_sha256(bounded_input),
        "exact_output_deferred_backlog_keys_sha256": _identity_sha256(exact_backlog_ids),
        "bounded_stored_deferred_backlog_keys_sha256": _identity_sha256(bounded_backlog_ids),
        "same_tensor_offsets": True,
        "tensor_offsets_sha256": offsets_hash,
        "path_difference": path_difference,
    }
    if oracle_parity_overrides:
        oracle_parity.update(dict(oracle_parity_overrides))
    return BoundedDeltaMeasuredReport(
        schema_version=BOUNDED_DELTA_ACCUMULATOR_SCHEMA_VERSION,
        label=BOUNDED_DELTA_ACCUMULATOR_LABEL,
        candidate_name=candidate_name,
        candidate_changed_count=candidate_changed_count,
        candidate_union_count=len(exact_path.candidate_ids | bounded_path.candidate_ids),
        candidate_changed_fraction=candidate_fraction,
        direction_changed_count=direction_changed,
        accepted_changed_count=accepted_changed_count,
        accepted_union_count=len(exact_path.accepted_ids | bounded_path.accepted_ids),
        accepted_changed_fraction=accepted_fraction,
        deferred_changed_count=deferred_changed_count,
        deferred_union_count=len(exact_path.deferred_ids | bounded_path.deferred_ids),
        deferred_changed_fraction=deferred_fraction,
        q_changed_count=q_changed_count,
        q_changed_union_count=len(exact_path.q_changed_ids | bounded_path.q_changed_ids),
        q_changed_fraction=q_fraction,
        backlog_key_changed_count=backlog_changed_count,
        backlog_key_union_count=len(exact_backlog_ids | bounded_backlog_ids),
        backlog_key_changed_fraction=backlog_fraction,
        cap_frontier_rank_delta=rank_delta,
        hot_risk_changed_count=hot_risk_changed,
        max_abs_acc_error=max_abs_error,
        p95_abs_acc_error=_p95(all_errors),
        fired_or_accepted_residual_changed_count=len(fired_or_accepted_residual_changed),
        fired_or_accepted_residual_identities_sha256=_identity_sha256(
            fired_or_accepted_residual_changed
        ),
        hot_residual_changed_count=len(hot_residual_changed),
        hot_residual_identities_sha256=_identity_sha256(hot_residual_changed),
        accumulator_residual_hash_match=residual_hash_match,
        exact_accumulator_residuals_sha256=exact_hashes,
        bounded_accumulator_residuals_sha256=bounded_hashes,
        exact_candidate_identities_sha256=_identity_sha256(exact_path.candidate_ids),
        bounded_candidate_identities_sha256=_identity_sha256(bounded_path.candidate_ids),
        exact_accepted_identities_sha256=_identity_sha256(exact_path.accepted_ids),
        bounded_accepted_identities_sha256=_identity_sha256(bounded_path.accepted_ids),
        exact_deferred_identities_sha256=_identity_sha256(exact_path.deferred_ids),
        bounded_deferred_identities_sha256=_identity_sha256(bounded_path.deferred_ids),
        oracle_parity=oracle_parity,
    )


def compare_bounded_delta_paths_to_int16_oracle(
    *,
    inputs: Sequence[BoundedDeltaOracleInput],
    q_ledger_row: Base3QEntropyLedgerRow,
    exact_path: _PathResult,
    bounded_path: _PathResult,
    storage_projection: BoundedDeltaStorageProjection,
    guard_spec: BoundedDeltaGuardSpec | None = None,
    candidate_name: str = HYBRID_HOT_EXACT_COLD_DEFAULT_CANDIDATE,
    global_cap_spec: GlobalRateCapSpec | None = None,
    exact_input_states: Mapping[str, VoteUpdateState] | None = None,
    bounded_input_states: Mapping[str, VoteUpdateState] | None = None,
    exact_input_backlog: Mapping[str, Mapping[int, Mapping[str, int]]] | None = None,
    bounded_input_backlog: Mapping[str, Mapping[int, Mapping[str, int]]] | None = None,
    bounded_stored_backlog: Mapping[str, Mapping[int, Mapping[str, int]]] | None = None,
    tensor_offsets: Mapping[str, int] | None = None,
    bounded_backlog_policy_active: bool = False,
    path_difference: str = "bounded path differs only by encode_decode_accumulator_loss",
    oracle_parity_overrides: Mapping[str, bool | str | int] | None = None,
    next_candidate_if_failed: str | None = None,
    non_claims: Sequence[str] = (),
) -> BoundedDeltaReferenceReport:
    guard = guard_spec or BoundedDeltaGuardSpec()
    guard.validate()
    admission_contract = bounded_delta_admission_contract(candidate_name=candidate_name)
    next_candidate = (
        _default_next_candidate_if_failed(admission_contract.candidate_name)
        if next_candidate_if_failed is None
        else str(next_candidate_if_failed)
    )
    validate_base3_q_entropy_ledger(q_ledger_row)
    ledger = bounded_delta_inclusive_ledger(q_ledger_row, storage_projection)
    validate_bounded_delta_inclusive_ledger(ledger)
    measured = _build_measured_report_from_paths(
        inputs=inputs,
        candidate_name=admission_contract.candidate_name,
        exact_path=exact_path,
        bounded_path=bounded_path,
        global_cap_spec=global_cap_spec,
        exact_input_backlog=exact_input_backlog,
        bounded_input_backlog=bounded_input_backlog,
        bounded_stored_backlog=bounded_stored_backlog,
        tensor_offsets=tensor_offsets,
        bounded_backlog_policy_active=bounded_backlog_policy_active,
        path_difference=path_difference,
        exact_input_states=exact_input_states,
        bounded_input_states=bounded_input_states,
        oracle_parity_overrides=oracle_parity_overrides,
    )
    guard_eval = _evaluate_guardrail(guard, measured)
    admission_eval = _evaluate_bounded_delta_admission(admission_contract, measured)
    if not guard_eval.guard_passed:
        classification = BOUNDED_DELTA_GUARDRAIL_FAILED
    elif not ledger.claimable_physical_sub2:
        classification = BOUNDED_DELTA_LEDGER_FAILED
    elif not admission_eval.admission_passed:
        classification = BOUNDED_DELTA_ADMISSION_FAILED
    else:
        classification = BOUNDED_DELTA_WITH_REPORT
    return BoundedDeltaReferenceReport(
        schema_version=BOUNDED_DELTA_ACCUMULATOR_SCHEMA_VERSION,
        label=BOUNDED_DELTA_ACCUMULATOR_LABEL,
        candidate_name=admission_contract.candidate_name,
        classification=classification,
        ledger=ledger,
        storage_projection=storage_projection,
        guard_spec=guard,
        admission_contract=admission_contract,
        measured_report=measured,
        guard_passed=guard_eval.guard_passed,
        failed_metrics=guard_eval.failed_metrics,
        admission_passed=admission_eval.admission_passed,
        admission_failed_surfaces=admission_eval.failed_surfaces,
        candidate_assessment=bounded_delta_candidate_assessment(
            candidate_name=admission_contract.candidate_name
        ),
        rejection_telemetry=admission_eval.rejection_telemetry,
        raw_arrays_included=False,
        non_claims=tuple(non_claims),
        next_candidate_if_failed=next_candidate,
    )


def compare_bounded_delta_step_to_int16_oracle(
    inputs: Sequence[BoundedDeltaOracleInput],
    *,
    q_ledger_row: Base3QEntropyLedgerRow,
    guard_spec: BoundedDeltaGuardSpec | None = None,
    candidate_name: str = HYBRID_HOT_EXACT_COLD_DEFAULT_CANDIDATE,
    global_cap_spec: GlobalRateCapSpec | None = None,
    deferred_backlog: dict[str, dict[int, dict[str, int]]] | None = None,
    bounded_deferred_backlog: dict[str, dict[int, dict[str, int]]] | None = None,
    bounded_stored_deferred_backlog: dict[str, dict[int, dict[str, int]]] | None = None,
    tensor_offsets: dict[str, int] | None = None,
    storage_projection: BoundedDeltaStorageProjection | None = None,
    dense_cold_bits_per_weight: float = 0.0,
    event_delta_count: int = 0,
    tensor_metadata_bits: int | None = None,
    bucket_metadata_bits: int = 64,
    scale_metadata_bits: int = 0,
    guardrail_metadata_bits: int = 64,
    next_candidate_if_failed: str | None = None,
) -> BoundedDeltaReferenceReport:
    """Compare exact int16 dynamics against bounded encode/decode loss.

    By default this preserves the original same-backlog strict control: exact and
    bounded paths receive the same deferred backlog, and the projection charges the
    max exact/bounded output backlog. Supplying ``bounded_deferred_backlog`` and/or
    ``bounded_stored_deferred_backlog`` opts into an explicit degraded-backlog
    measurement: the bounded path receives/carries the provided bounded backlog,
    and the ledger charges the actual bounded stored backlog rather than hiding an
    exact backlog behind a budget-fitting override.
    """

    if not inputs:
        raise ValueError("at least one bounded-delta oracle input is required")
    guard = guard_spec or BoundedDeltaGuardSpec()
    guard.validate()
    admission_contract = bounded_delta_admission_contract(candidate_name=candidate_name)
    next_candidate = (
        _default_next_candidate_if_failed(admission_contract.candidate_name)
        if next_candidate_if_failed is None
        else str(next_candidate_if_failed)
    )
    seen: set[str] = set()
    exact_states: dict[str, VoteUpdateState] = {}
    bounded_states: dict[str, VoteUpdateState] = {}
    encoded_states: dict[str, BoundedDeltaAccumulatorState] = {}
    eligible = 0
    hot_count = 0
    cold_exception_count = 0
    for item in inputs:
        if not item.state_key:
            raise ValueError("state_key must be non-empty")
        if item.state_key in seen:
            raise ValueError(f"duplicate state_key {item.state_key!r}")
        seen.add(item.state_key)
        encoded = encode_budget_capped_hybrid_reference(
            item.state,
            hot_exact_indices=item.hot_exact_indices,
            cold_default_value=item.cold_default_value,
            cold_exception_indices=item.cold_exception_indices,
            cold_exception_values=item.cold_exception_values,
        )
        decoded_acc = decode_bounded_accumulator_to_i16(encoded)
        exact_states[item.state_key] = item.state
        bounded_states[item.state_key] = VoteUpdateState(
            q_levels=item.state.q_levels.detach().clone().contiguous(),
            accumulators=decoded_acc.to(device=item.state.accumulators.device),
        )
        encoded_states[item.state_key] = encoded
        eligible += int(item.state.q_levels.numel())
        hot_count += len(encoded.hot_exact_indices)
        cold_exception_count += len(encoded.cold_exception_indices)

    exact_cap_inputs = [
        GlobalRateCapTensorInput(
            item.state_key,
            exact_states[item.state_key],
            plan_integer_vote_update_reference(
                exact_states[item.state_key],
                item.vote_inputs,
                item.vote_spec,
            ),
        )
        for item in inputs
    ]
    offsets = tensor_offsets
    if global_cap_spec is not None and offsets is None:
        offsets = tensor_offsets_for_vote_update_states(exact_cap_inputs)

    exact = _run_reference_path(
        inputs,
        states_by_key=exact_states,
        global_cap_spec=global_cap_spec,
        deferred_backlog=deferred_backlog,
        tensor_offsets=offsets,
    )
    bounded_backlog_policy_active = (
        bounded_deferred_backlog is not None
        or bounded_stored_deferred_backlog is not None
    )
    bounded_input_backlog = (
        deferred_backlog if bounded_deferred_backlog is None else bounded_deferred_backlog
    )
    bounded = _run_reference_path(
        inputs,
        states_by_key=bounded_states,
        global_cap_spec=global_cap_spec,
        deferred_backlog=bounded_input_backlog,
        tensor_offsets=offsets,
    )
    bounded_output_backlog = (
        bounded_stored_deferred_backlog
        if bounded_stored_deferred_backlog is not None
        else bounded.cap_result.deferred_backlog
        if bounded.cap_result is not None
        else bounded_input_backlog
    ) or {}
    exact_backlog_ids = exact.backlog_ids
    bounded_backlog_ids = _backlog_key_set(bounded_output_backlog)
    if bounded_stored_deferred_backlog is not None and bounded.cap_result is not None:
        bounded_actual_backlog_ids = _backlog_key_set(bounded.cap_result.deferred_backlog)
        if not bounded_backlog_ids <= bounded_actual_backlog_ids:
            invented_ids = sorted(bounded_backlog_ids - bounded_actual_backlog_ids)
            raise ValueError(
                "bounded_stored_deferred_backlog must be a subset of the bounded "
                "path's actual output backlog; invented_or_stale_ids="
                f"{invented_ids[:8]}"
            )

    backlog_entry_count = (
        len(bounded_backlog_ids)
        if bounded_backlog_policy_active
        else max(len(exact_backlog_ids), len(bounded.backlog_ids))
    )
    if (
        bounded_backlog_policy_active
        and storage_projection is not None
        and int(storage_projection.backlog_entry_count) != backlog_entry_count
    ):
        raise ValueError(
            "bounded-backlog policy storage_projection must charge the actual "
            "bounded stored backlog entry count; got "
            f"{storage_projection.backlog_entry_count} != {backlog_entry_count}"
        )
    projection = storage_projection or project_bounded_delta_accumulator_bpw(
        eligible_weight_count=eligible,
        hot_exact_row_count=hot_count,
        cold_exception_row_count=cold_exception_count,
        event_delta_count=event_delta_count,
        backlog_entry_count=backlog_entry_count,
        tensor_metadata_bits=len(inputs) * 64 if tensor_metadata_bits is None else tensor_metadata_bits,
        bucket_metadata_bits=bucket_metadata_bits,
        scale_metadata_bits=scale_metadata_bits,
        guardrail_metadata_bits=guardrail_metadata_bits,
        dense_cold_bits_per_weight=dense_cold_bits_per_weight,
    )
    path_difference = (
        "bounded path differs only by encode_decode_accumulator_loss"
        if not bounded_backlog_policy_active
        else (
            "bounded path differs by encode_decode_accumulator_loss_and_"
            "bounded_backlog_encode_drop"
        )
    )
    return compare_bounded_delta_paths_to_int16_oracle(
        inputs=inputs,
        q_ledger_row=q_ledger_row,
        exact_path=exact,
        bounded_path=bounded,
        storage_projection=projection,
        guard_spec=guard,
        candidate_name=admission_contract.candidate_name,
        global_cap_spec=global_cap_spec,
        exact_input_states=exact_states,
        bounded_input_states=bounded_states,
        exact_input_backlog=deferred_backlog or {},
        bounded_input_backlog=bounded_input_backlog or {},
        bounded_stored_backlog=bounded_output_backlog,
        tensor_offsets=offsets,
        bounded_backlog_policy_active=bounded_backlog_policy_active,
        path_difference=path_difference,
        next_candidate_if_failed=next_candidate,
        non_claims=(
            "no production vote_update/global_rate_cap replacement",
            "no GPU lane",
            "no trainer/live-run/checkpoint/creditdir mutation",
            "no acquisition or stability claim",
            "no decision_exact claim",
            "compact counts/hashes only; no raw per-weight arrays",
        ),
    )


__all__ = [
    "ACCUMULATOR_SUBSTITUTE_LOCAL_VOTE_UPDATE_EXECUTABLE",
    "ALGORITHMIC_LOCAL_VOTE_UPDATE_EXECUTABLE_NOT_PHYSICAL_SUB2",
    "BOUNDED_DELTA_ADMISSION_FAILED",
    "BOUNDED_DELTA_GUARDRAIL_FAILED",
    "BOUNDED_DELTA_LEDGER_FAILED",
    "BOUNDED_LOCAL_VOTE_UPDATE_PROOF_SCHEMA_VERSION",
    "BOUNDED_DELTA_WITH_REPORT",
    "COARSE_SIGNED_CHARGE_SPARSE_FRONTIER_CANDIDATE",
    "EVENT_CODED_CROSSING_RESIDUAL_LOG_CANDIDATE",
    "HYBRID_HOT_EXACT_COLD_DEFAULT_CANDIDATE",
    "BoundedDeltaAccumulatorState",
    "BoundedDeltaAdmissionContract",
    "BoundedDirectLocalUpdateCandidateResult",
    "BoundedDeltaGuardSpec",
    "BoundedDeltaInclusiveLedger",
    "BoundedDeltaMeasuredReport",
    "BoundedDeltaOracleInput",
    "BoundedDeltaRejectionTelemetry",
    "BoundedDeltaRejectionSurface",
    "BoundedDeltaReferenceReport",
    "BoundedDeltaStorageProjection",
    "INTRINSIC_BOUNDED_UPDATE_DOMAIN_GAP",
    "bounded_delta_admission_contract",
    "bounded_delta_candidate_assessment",
    "bounded_delta_inclusive_ledger",
    "compare_bounded_delta_paths_to_int16_oracle",
    "compare_bounded_delta_step_to_int16_oracle",
    "bounded_accumulator_decoded_sha256",
    "decode_bounded_accumulator_to_i16",
    "encode_budget_capped_hybrid_reference",
    "execute_direct_bounded_local_vote_update_candidate",
    "_execute_direct_bounded_local_vote_update_reference_3936d74",
    "project_bounded_delta_accumulator_bpw",
    "validate_bounded_delta_inclusive_ledger",
]
