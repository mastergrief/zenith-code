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
from typing import Any, Mapping, Sequence

import torch

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
BOUNDED_DELTA_WITH_REPORT = CandidateClassification.BOUNDED_DELTA_WITH_REPORT.value
BOUNDED_DELTA_GUARDRAIL_FAILED = "bounded_delta_guardrail_failed"
BOUNDED_DELTA_LEDGER_FAILED = "bounded_delta_ledger_failed"


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


def _identity_sha256(identities: set[tuple[str, int]]) -> str:
    h = hashlib.sha256()
    for state_key, flat_index in sorted(identities):
        h.update(state_key.encode("utf-8"))
        h.update(b":")
        h.update(str(int(flat_index)).encode("utf-8"))
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
    measured_report: BoundedDeltaMeasuredReport
    guard_passed: bool
    failed_metrics: tuple[str, ...]
    candidate_assessment: CandidateAssessment
    raw_arrays_included: bool
    non_claims: tuple[str, ...]
    next_candidate_if_failed: str

    @property
    def claimable_physical_sub2_with_guardrail(self) -> bool:
        return (
            self.classification == BOUNDED_DELTA_WITH_REPORT
            and self.ledger.claimable_physical_sub2
            and self.guard_passed
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
            "measured_report": self.measured_report.to_dict(),
            "guard_passed": bool(self.guard_passed),
            "failed_metrics": list(self.failed_metrics),
            "candidate_assessment": self.candidate_assessment.to_dict(),
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


def bounded_delta_candidate_assessment(
    *,
    candidate_name: str = HYBRID_HOT_EXACT_COLD_DEFAULT_CANDIDATE,
) -> CandidateAssessment:
    return candidate_assessment(
        candidate_name=candidate_name,
        classification=CandidateClassification.BOUNDED_DELTA_WITH_REPORT,
        covered_decision_dimensions=required_decision_dimension_names(),
        compressed_representation=True,
        bounded_delta_hypothesis=(
            "budget-capped hot exact rows plus cold default/sparse exceptions fit "
            "the q+scale remaining budget while allowing bounded nonzero decision drift"
        ),
        guardrail=(
            "pre-declared decision-surface drift bounds over candidate/order/"
            "accepted/deferred/q_changed/backlog/rank metrics"
        ),
        note="adapter/oracle reference only; no production vote/cap replacement",
    )


@dataclass(frozen=True)
class _PathResult:
    plans: dict[str, VoteUpdatePlan]
    candidate_ids: set[tuple[str, int]]
    candidate_direction_by_id: dict[tuple[str, int], int]
    accepted_ids: set[tuple[str, int]]
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


def compare_bounded_delta_step_to_int16_oracle(
    inputs: Sequence[BoundedDeltaOracleInput],
    *,
    q_ledger_row: Base3QEntropyLedgerRow,
    guard_spec: BoundedDeltaGuardSpec | None = None,
    global_cap_spec: GlobalRateCapSpec | None = None,
    deferred_backlog: dict[str, dict[int, dict[str, int]]] | None = None,
    tensor_offsets: dict[str, int] | None = None,
    storage_projection: BoundedDeltaStorageProjection | None = None,
    dense_cold_bits_per_weight: float = 0.0,
    event_delta_count: int = 0,
    tensor_metadata_bits: int | None = None,
    bucket_metadata_bits: int = 64,
    scale_metadata_bits: int = 0,
    guardrail_metadata_bits: int = 64,
    next_candidate_if_failed: str = "event_coded_crossing_residual_log",
) -> BoundedDeltaReferenceReport:
    """Compare exact int16 dynamics against bounded encode/decode loss only."""

    if not inputs:
        raise ValueError("at least one bounded-delta oracle input is required")
    guard = guard_spec or BoundedDeltaGuardSpec()
    guard.validate()
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
    bounded = _run_reference_path(
        inputs,
        states_by_key=bounded_states,
        global_cap_spec=global_cap_spec,
        deferred_backlog=deferred_backlog,
        tensor_offsets=offsets,
    )

    backlog_entry_count = max(len(exact.backlog_ids), len(bounded.backlog_ids))
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
    ledger = bounded_delta_inclusive_ledger(q_ledger_row, projection)
    validate_bounded_delta_inclusive_ledger(ledger)

    candidate_changed_count, candidate_fraction = _symmetric_fraction(
        exact.candidate_ids,
        bounded.candidate_ids,
    )
    accepted_changed_count, accepted_fraction = _symmetric_fraction(
        exact.accepted_ids,
        bounded.accepted_ids,
    )
    deferred_changed_count, deferred_fraction = _symmetric_fraction(
        exact.deferred_ids,
        bounded.deferred_ids,
    )
    q_changed_count, q_fraction = _symmetric_fraction(exact.q_changed_ids, bounded.q_changed_ids)
    backlog_changed_count, backlog_fraction = _symmetric_fraction(
        exact.backlog_ids,
        bounded.backlog_ids,
    )

    exact_direction = exact.candidate_direction_by_id
    bounded_direction = bounded.candidate_direction_by_id
    direction_keys = set(exact_direction) | set(bounded_direction)
    direction_changed = sum(
        1 for key in direction_keys if exact_direction.get(key) != bounded_direction.get(key)
    )
    rank_delta = _rank_delta(exact.ordered_row_ids, bounded.ordered_row_ids)

    exact_hashes: dict[str, str] = {}
    bounded_hashes: dict[str, str] = {}
    acc_errors: list[torch.Tensor] = []
    residual_hash_match = True
    for item in inputs:
        exact_acc = exact.output_acc_by_key[item.state_key].detach().cpu().to(torch.int32)
        bounded_acc = bounded.output_acc_by_key[item.state_key].detach().cpu().to(torch.int32)
        exact_hash = _tensor_sha256(exact_acc)
        bounded_hash = _tensor_sha256(bounded_acc)
        exact_hashes[item.state_key] = exact_hash
        bounded_hashes[item.state_key] = bounded_hash
        residual_hash_match = residual_hash_match and exact_hash == bounded_hash
        acc_errors.append((exact_acc - bounded_acc).abs().flatten())
    all_errors = torch.cat(acc_errors) if acc_errors else torch.empty(0, dtype=torch.int32)
    max_abs_error = int(all_errors.max().item()) if int(all_errors.numel()) else 0

    hot_ids = _hot_identity_set(inputs)
    decision_symdiff = (
        (exact.candidate_ids ^ bounded.candidate_ids)
        | (exact.accepted_ids ^ bounded.accepted_ids)
        | (exact.deferred_ids ^ bounded.deferred_ids)
        | (exact.q_changed_ids ^ bounded.q_changed_ids)
    )
    hot_risk_changed = len(decision_symdiff & hot_ids)

    vote_hash = _hash_vote_inputs(inputs)
    cap_hash = _hash_cap_spec(global_cap_spec)
    offsets_hash = hashlib.sha256(str(sorted((offsets or {}).items())).encode("utf-8")).hexdigest()
    backlog_hash = hashlib.sha256(str(sorted(_backlog_key_set(deferred_backlog or {}))).encode("utf-8")).hexdigest()
    measured = BoundedDeltaMeasuredReport(
        schema_version=BOUNDED_DELTA_ACCUMULATOR_SCHEMA_VERSION,
        label=BOUNDED_DELTA_ACCUMULATOR_LABEL,
        candidate_name=HYBRID_HOT_EXACT_COLD_DEFAULT_CANDIDATE,
        candidate_changed_count=candidate_changed_count,
        candidate_union_count=len(exact.candidate_ids | bounded.candidate_ids),
        candidate_changed_fraction=candidate_fraction,
        direction_changed_count=direction_changed,
        accepted_changed_count=accepted_changed_count,
        accepted_union_count=len(exact.accepted_ids | bounded.accepted_ids),
        accepted_changed_fraction=accepted_fraction,
        deferred_changed_count=deferred_changed_count,
        deferred_union_count=len(exact.deferred_ids | bounded.deferred_ids),
        deferred_changed_fraction=deferred_fraction,
        q_changed_count=q_changed_count,
        q_changed_union_count=len(exact.q_changed_ids | bounded.q_changed_ids),
        q_changed_fraction=q_fraction,
        backlog_key_changed_count=backlog_changed_count,
        backlog_key_union_count=len(exact.backlog_ids | bounded.backlog_ids),
        backlog_key_changed_fraction=backlog_fraction,
        cap_frontier_rank_delta=rank_delta,
        hot_risk_changed_count=hot_risk_changed,
        max_abs_acc_error=max_abs_error,
        p95_abs_acc_error=_p95(all_errors),
        accumulator_residual_hash_match=residual_hash_match,
        exact_accumulator_residuals_sha256=exact_hashes,
        bounded_accumulator_residuals_sha256=bounded_hashes,
        exact_candidate_identities_sha256=_identity_sha256(exact.candidate_ids),
        bounded_candidate_identities_sha256=_identity_sha256(bounded.candidate_ids),
        exact_accepted_identities_sha256=_identity_sha256(exact.accepted_ids),
        bounded_accepted_identities_sha256=_identity_sha256(bounded.accepted_ids),
        exact_deferred_identities_sha256=_identity_sha256(exact.deferred_ids),
        bounded_deferred_identities_sha256=_identity_sha256(bounded.deferred_ids),
        oracle_parity={
            "same_initial_q": True,
            "same_votes_sha256": True,
            "votes_sha256": vote_hash,
            "same_cap_spec": True,
            "cap_spec_sha256": cap_hash,
            "same_deferred_backlog": True,
            "deferred_backlog_keys_sha256": backlog_hash,
            "same_tensor_offsets": True,
            "tensor_offsets_sha256": offsets_hash,
            "path_difference": "bounded path differs only by encode_decode_accumulator_loss",
        },
    )
    guard_eval = _evaluate_guardrail(guard, measured)
    if not guard_eval.guard_passed:
        classification = BOUNDED_DELTA_GUARDRAIL_FAILED
    elif not ledger.claimable_physical_sub2:
        classification = BOUNDED_DELTA_LEDGER_FAILED
    else:
        classification = BOUNDED_DELTA_WITH_REPORT

    return BoundedDeltaReferenceReport(
        schema_version=BOUNDED_DELTA_ACCUMULATOR_SCHEMA_VERSION,
        label=BOUNDED_DELTA_ACCUMULATOR_LABEL,
        candidate_name=HYBRID_HOT_EXACT_COLD_DEFAULT_CANDIDATE,
        classification=classification,
        ledger=ledger,
        storage_projection=projection,
        guard_spec=guard,
        measured_report=measured,
        guard_passed=guard_eval.guard_passed,
        failed_metrics=guard_eval.failed_metrics,
        candidate_assessment=bounded_delta_candidate_assessment(),
        raw_arrays_included=False,
        non_claims=(
            "no production vote_update/global_rate_cap replacement",
            "no GPU lane",
            "no trainer/live-run/checkpoint/creditdir mutation",
            "no acquisition or stability claim",
            "no decision_exact claim",
            "compact counts/hashes only; no raw per-weight arrays",
        ),
        next_candidate_if_failed=next_candidate_if_failed,
    )


__all__ = [
    "BOUNDED_DELTA_GUARDRAIL_FAILED",
    "BOUNDED_DELTA_LEDGER_FAILED",
    "BOUNDED_DELTA_WITH_REPORT",
    "HYBRID_HOT_EXACT_COLD_DEFAULT_CANDIDATE",
    "BoundedDeltaAccumulatorState",
    "BoundedDeltaGuardSpec",
    "BoundedDeltaInclusiveLedger",
    "BoundedDeltaMeasuredReport",
    "BoundedDeltaOracleInput",
    "BoundedDeltaReferenceReport",
    "BoundedDeltaStorageProjection",
    "bounded_delta_candidate_assessment",
    "bounded_delta_inclusive_ledger",
    "compare_bounded_delta_step_to_int16_oracle",
    "decode_bounded_accumulator_to_i16",
    "encode_budget_capped_hybrid_reference",
    "project_bounded_delta_accumulator_bpw",
    "validate_bounded_delta_inclusive_ledger",
]
