"""C1.1b accumulator decision-density feasibility/classification.

This module is deliberately measurement-only. It asks whether exact int16
accumulator information is sparse enough to fit the C1.1a q-entropy remaining
budget before any accumulator encoder is built.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from typing import Any, Iterable, Sequence

import torch

from calm.hrm_text_158.native_full_stack.accumulator_compression import (
    CandidateAssessment,
    CandidateClassification,
    candidate_assessment,
    required_decision_dimension_names,
)
from calm.hrm_text_158.native_full_stack.global_rate_cap import (
    GlobalRateCapSpec,
    GlobalRateCapTensorInput,
    apply_global_rate_cap_reference,
    tensor_offsets_for_vote_update_states,
)
from calm.hrm_text_158.native_full_stack.q_entropy_packing import (
    default_base3_q_entropy_ledger_table,
)
from calm.hrm_text_158.native_full_stack.vote_update import (
    VoteUpdateInputs,
    VoteUpdateSpec,
    VoteUpdateState,
    plan_integer_vote_update_reference,
)


ACCUMULATOR_DECISION_DENSITY_SCHEMA_VERSION = (
    "hrm_text_158_accumulator_decision_density/v0.feasibility_classification"
)
ACCUMULATOR_DECISION_DENSITY_LABEL = (
    "c1p1b_accumulator_decision_density_feasibility_no_encoder"
)
DECISION_EXACT_INFEASIBLE = "decision_exact_infeasible"
VALUE_ENTROPY_IS_NOT_DECISION_EXACT = (
    "far-row value entropy/run-length is reported separately and cannot by "
    "itself unlock decision_exact"
)
ACTIVE_DEFINITION_NEXT_STEP_LAW = (
    "trunc(acc * decay_num / decay_den) + fixture_vote -> clamp; candidate mask "
    "after q saturation gate"
)
RANKING_EXACT_DENSITY_DEFINITION = (
    "actual reference-plan candidate rows plus global-cap row surface"
)
CAP_FRONTIER_DIAGNOSTIC_DEFINITION = (
    "accepted tail plus deferred head; bounded-delta evidence only unless a "
    "guard proves accepted/deferred identity unchanged"
)


def c1_1a_prior_large_accumulator_budget_bits_per_weight() -> float:
    """Return the C1.1a-pinned realistic remaining accumulator budget."""

    for row in default_base3_q_entropy_ledger_table():
        if row.regime_name == "prior_large_fixture_base3_q":
            return float(row.remaining_accumulator_budget_bits_per_weight)
    raise RuntimeError("prior_large_fixture_base3_q ledger row is missing")


def _bits_per_weight(bits: int | float, eligible_weight_count: int) -> float:
    eligible = int(eligible_weight_count)
    if eligible <= 0:
        raise ValueError("eligible_weight_count must be > 0")
    return float(bits) / float(eligible)


def index_bits_for_numel(numel: int) -> int:
    """Bits required to address one row in a tensor/flat state of size ``numel``."""

    numel_i = int(numel)
    if numel_i <= 0:
        raise ValueError("numel must be > 0")
    if numel_i == 1:
        return 1
    return int(math.ceil(math.log2(float(numel_i))))


def dense_fixed_width_bits_per_weight(width_bits: int | float) -> float:
    """Dense fixed-width accumulators cost exactly their width per weight."""

    width = float(width_bits)
    if width <= 0:
        raise ValueError("width_bits must be > 0")
    return width


def _tensor_sha256(tensor: torch.Tensor) -> str:
    cpu = tensor.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str(cpu.dtype).encode("utf-8"))
    h.update(str(tuple(cpu.shape)).encode("utf-8"))
    h.update(cpu.numpy().tobytes())
    return h.hexdigest()


def _identity_hash(identities: Iterable[tuple[str, int]]) -> str:
    h = hashlib.sha256()
    for state_key, flat_index in sorted((str(k), int(i)) for k, i in identities):
        h.update(state_key.encode("utf-8"))
        h.update(b":")
        h.update(str(flat_index).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _safe_density(count: int, eligible_weight_count: int) -> float:
    return float(count) / float(eligible_weight_count) if int(eligible_weight_count) else 0.0


@dataclass(frozen=True)
class SparseAccumulatorBpwProjection:
    """Overhead-inclusive sparse accumulator storage projection."""

    eligible_weight_count: int
    stored_row_count: int
    index_bits_per_row: int
    value_bits_per_row: int
    flag_bits_per_row: int
    tensor_metadata_bits: int
    backlog_entry_count: int
    backlog_index_bits_per_entry: int
    backlog_age_bits_per_entry: int
    backlog_defer_count_bits_per_entry: int
    entropy_model_bits: int
    row_storage_bits: int
    backlog_storage_bits: int
    total_projected_bits: int
    projected_bits_per_weight: float
    payload_only_bits_per_weight: float
    target_bits_per_weight: float
    fits_target: bool
    payload_only_would_fit: bool

    def to_dict(self) -> dict[str, int | float | bool]:
        return asdict(self)


@dataclass(frozen=True)
class FarValueEntropyReport:
    """Compact entropy/run-length summary for non-decision far rows."""

    far_row_count: int
    unique_value_count: int
    shannon_entropy_bits_per_value: float
    most_common_value_count: int
    most_common_value_fraction: float
    longest_equal_run: int
    density_note: str = VALUE_ENTROPY_IS_NOT_DECISION_EXACT

    def to_dict(self) -> dict[str, int | float | str]:
        return asdict(self)


@dataclass(frozen=True)
class AccumulatorDecisionDensityInput:
    """One tensor's state and votes for a C1.1b compact measurement."""

    state_key: str
    state: VoteUpdateState
    vote_inputs: VoteUpdateInputs
    spec: VoteUpdateSpec


@dataclass(frozen=True)
class AccumulatorCandidateClassification:
    """Classification or infeasibility diagnostic for a candidate representation."""

    candidate_name: str
    classification: str
    projected_bits_per_weight: float
    target_bits_per_weight: float
    decision_exact_feasible: bool
    candidate_assessment: CandidateAssessment | None
    missing_decision_dimensions: tuple[str, ...]
    infeasibility_reason: str

    @property
    def c2_eligible_by_default(self) -> bool:
        return bool(self.candidate_assessment and self.candidate_assessment.c2_eligible_by_default)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_name": self.candidate_name,
            "classification": self.classification,
            "projected_bits_per_weight": self.projected_bits_per_weight,
            "target_bits_per_weight": self.target_bits_per_weight,
            "decision_exact_feasible": self.decision_exact_feasible,
            "candidate_assessment": (
                self.candidate_assessment.to_dict() if self.candidate_assessment is not None else None
            ),
            "missing_decision_dimensions": list(self.missing_decision_dimensions),
            "infeasibility_reason": self.infeasibility_reason,
            "c2_eligible_by_default": self.c2_eligible_by_default,
        }


@dataclass(frozen=True)
class AccumulatorDecisionDensityReport:
    """Compact decision-density measurement; no raw per-weight arrays."""

    schema_version: str
    label: str
    eligible_weight_count: int
    tensor_count: int
    target_bits_per_weight: float
    active_definition: str
    ranking_exact_density_definition: str
    cap_frontier_diagnostic_definition: str
    fixture_vote_record: str
    fixture_vote_abs_max: int
    fixture_vote_nonzero_count: int
    fixture_vote_sha256: str
    current_magnitude_threshold_count: int
    active_next_step_count: int
    ranking_sensitive_exact_count: int
    cap_frontier_diagnostic_count: int
    backlog_state_carry_count: int
    replay_veto_residual_count: int
    decision_relevant_exact_count: int
    far_row_count: int
    current_magnitude_threshold_density: float
    active_next_step_density: float
    ranking_sensitive_exact_density: float
    cap_frontier_diagnostic_density: float
    backlog_state_carry_density: float
    decision_relevant_exact_density: float
    current_magnitude_threshold_indices_sha256: str
    active_next_step_indices_sha256: str
    ranking_sensitive_exact_indices_sha256: str
    cap_frontier_diagnostic_indices_sha256: str
    backlog_state_carry_indices_sha256: str
    decision_relevant_exact_indices_sha256: str
    sparse_exact_projection: SparseAccumulatorBpwProjection
    far_value_entropy: FarValueEntropyReport
    global_cap_used: bool
    global_cap_row_count: int
    global_cap_accepted_count: int
    global_cap_deferred_count: int
    global_cap_saturated: bool
    backlog_max_age_steps: int
    backlog_max_defer_count: int
    raw_arrays_included: bool
    non_claims: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["sparse_exact_projection"] = self.sparse_exact_projection.to_dict()
        payload["far_value_entropy"] = self.far_value_entropy.to_dict()
        payload["non_claims"] = list(self.non_claims)
        return payload


def project_sparse_accumulator_bpw(
    *,
    eligible_weight_count: int,
    stored_row_count: int,
    target_bits_per_weight: float | None = None,
    index_bits_per_row: int | None = None,
    value_bits_per_row: int = 16,
    flag_bits_per_row: int = 2,
    tensor_metadata_bits: int = 0,
    backlog_entry_count: int = 0,
    backlog_age_bits_per_entry: int = 16,
    backlog_defer_count_bits_per_entry: int = 16,
    entropy_model_bits: int = 0,
) -> SparseAccumulatorBpwProjection:
    """Project sparse accumulator cost, charging index/payload/metadata overhead."""

    eligible = int(eligible_weight_count)
    stored = int(stored_row_count)
    if eligible <= 0:
        raise ValueError("eligible_weight_count must be > 0")
    if stored < 0:
        raise ValueError("stored_row_count must be >= 0")
    if int(backlog_entry_count) < 0:
        raise ValueError("backlog_entry_count must be >= 0")
    if stored > eligible:
        raise ValueError("stored_row_count cannot exceed eligible_weight_count")
    target = (
        c1_1a_prior_large_accumulator_budget_bits_per_weight()
        if target_bits_per_weight is None
        else float(target_bits_per_weight)
    )
    index_bits = index_bits_for_numel(eligible) if index_bits_per_row is None else int(index_bits_per_row)
    value_bits = int(value_bits_per_row)
    flag_bits = int(flag_bits_per_row)
    if index_bits <= 0 or value_bits <= 0 or flag_bits < 0:
        raise ValueError("index/value/flag bits must be positive (flags may be zero)")
    metadata_bits = int(tensor_metadata_bits)
    entropy_bits = int(entropy_model_bits)
    if metadata_bits < 0 or entropy_bits < 0:
        raise ValueError("metadata and entropy model bits must be >= 0")

    row_storage_bits = stored * (index_bits + value_bits + flag_bits)
    backlog_bits = int(backlog_entry_count) * (
        index_bits + int(backlog_age_bits_per_entry) + int(backlog_defer_count_bits_per_entry)
    )
    total_bits = row_storage_bits + backlog_bits + metadata_bits + entropy_bits
    payload_only_bits = stored * value_bits
    projected_bpw = _bits_per_weight(total_bits, eligible)
    payload_only_bpw = _bits_per_weight(payload_only_bits, eligible)
    return SparseAccumulatorBpwProjection(
        eligible_weight_count=eligible,
        stored_row_count=stored,
        index_bits_per_row=index_bits,
        value_bits_per_row=value_bits,
        flag_bits_per_row=flag_bits,
        tensor_metadata_bits=metadata_bits,
        backlog_entry_count=int(backlog_entry_count),
        backlog_index_bits_per_entry=index_bits,
        backlog_age_bits_per_entry=int(backlog_age_bits_per_entry),
        backlog_defer_count_bits_per_entry=int(backlog_defer_count_bits_per_entry),
        entropy_model_bits=entropy_bits,
        row_storage_bits=row_storage_bits,
        backlog_storage_bits=backlog_bits,
        total_projected_bits=total_bits,
        projected_bits_per_weight=projected_bpw,
        payload_only_bits_per_weight=payload_only_bpw,
        target_bits_per_weight=target,
        fits_target=projected_bpw <= target,
        payload_only_would_fit=payload_only_bpw <= target,
    )


def _rows_from_tensor(indices: torch.Tensor, state_key: str) -> set[tuple[str, int]]:
    return {(state_key, int(idx)) for idx in indices.detach().cpu().to(torch.int64).flatten().tolist()}


def _backlog_identities(backlog: dict[str, dict[int, dict[str, int]]]) -> set[tuple[str, int]]:
    return {
        (state_key, int(flat_index))
        for state_key, by_index in backlog.items()
        for flat_index in by_index
    }


def _backlog_age_summary(
    backlog: dict[str, dict[int, dict[str, int]]],
    *,
    step: int,
) -> tuple[int, int]:
    entries = [entry for by_index in backlog.values() for entry in by_index.values()]
    if not entries:
        return 0, 0
    max_age = max(int(step) - int(entry.get("first_step", step)) for entry in entries)
    max_defer_count = max(int(entry.get("defer_count", 0)) for entry in entries)
    return max_age, max_defer_count


def _far_entropy(values: torch.Tensor) -> FarValueEntropyReport:
    flat = values.detach().cpu().to(torch.int32).flatten()
    count = int(flat.numel())
    if count == 0:
        return FarValueEntropyReport(
            far_row_count=0,
            unique_value_count=0,
            shannon_entropy_bits_per_value=0.0,
            most_common_value_count=0,
            most_common_value_fraction=0.0,
            longest_equal_run=0,
        )
    _, counts = torch.unique(flat, return_counts=True)
    counts_f = counts.to(torch.float64)
    probs = counts_f / float(count)
    entropy = float(-(probs * torch.log2(probs)).sum().item())
    most_common = int(counts.max().item())
    if count == 1:
        longest_run = 1
    else:
        changes = torch.nonzero(flat[1:] != flat[:-1], as_tuple=False).flatten() + 1
        boundaries = torch.cat(
            [
                torch.tensor([0], dtype=torch.int64),
                changes.to(torch.int64),
                torch.tensor([count], dtype=torch.int64),
            ],
        )
        longest_run = int((boundaries[1:] - boundaries[:-1]).max().item())
    return FarValueEntropyReport(
        far_row_count=count,
        unique_value_count=int(counts.numel()),
        shannon_entropy_bits_per_value=entropy,
        most_common_value_count=most_common,
        most_common_value_fraction=float(most_common) / float(count),
        longest_equal_run=longest_run,
    )


def measure_accumulator_decision_density(
    inputs: Sequence[AccumulatorDecisionDensityInput],
    *,
    global_cap_spec: GlobalRateCapSpec | None = None,
    deferred_backlog: dict[str, dict[int, dict[str, int]]] | None = None,
    tensor_offsets: dict[str, int] | None = None,
    target_bits_per_weight: float | None = None,
    cap_frontier_width: int = 1,
    value_bits_per_row: int = 16,
    flag_bits_per_row: int = 2,
    tensor_metadata_bits: int | None = None,
    entropy_model_bits: int = 0,
) -> AccumulatorDecisionDensityReport:
    """Measure decision-relevant accumulator density using existing references."""

    if not inputs:
        raise ValueError("at least one accumulator decision-density input is required")
    if int(cap_frontier_width) <= 0:
        raise ValueError("cap_frontier_width must be > 0")
    target = (
        c1_1a_prior_large_accumulator_budget_bits_per_weight()
        if target_bits_per_weight is None
        else float(target_bits_per_weight)
    )
    plans: dict[str, Any] = {}
    cap_inputs: list[GlobalRateCapTensorInput] = []
    eligible = 0
    vote_abs_max = 0
    vote_nonzero_count = 0
    vote_hash_parts: list[str] = []
    current_threshold_ids: set[tuple[str, int]] = set()
    active_next_step_ids: set[tuple[str, int]] = set()
    ranking_exact_ids: set[tuple[str, int]] = set()
    replay_veto_ids: set[tuple[str, int]] = set()

    seen: set[str] = set()
    for item in inputs:
        if not item.state_key:
            raise ValueError("state_key must be non-empty")
        if item.state_key in seen:
            raise ValueError(f"duplicate state_key {item.state_key!r}")
        seen.add(item.state_key)
        plan = plan_integer_vote_update_reference(item.state, item.vote_inputs, item.spec)
        plans[item.state_key] = plan
        cap_inputs.append(
            GlobalRateCapTensorInput(
                state_key=item.state_key,
                state=item.state,
                plan=plan,
            ),
        )
        flat_q = item.state.q_levels.flatten().to(torch.int16)
        flat_acc = item.state.accumulators.flatten().to(torch.int32)
        threshold = int(item.spec.threshold_abs)
        current_candidates = ((flat_acc >= threshold) & (flat_q < 1)) | (
            (flat_acc <= -threshold) & (flat_q > -1)
        )
        current_threshold_ids |= _rows_from_tensor(
            torch.nonzero(current_candidates, as_tuple=False).flatten(),
            item.state_key,
        )
        active_next_step_ids |= _rows_from_tensor(plan.candidate_indices, item.state_key)
        ranking_exact_ids |= _rows_from_tensor(plan.candidate_indices, item.state_key)
        replay_veto_ids |= _rows_from_tensor(plan.replay_ce_veto_indices, item.state_key)
        votes = item.vote_inputs.votes.detach()
        vote_abs_max = max(vote_abs_max, int(votes.abs().max().item()) if votes.numel() else 0)
        vote_nonzero_count += int((votes != 0).sum().item())
        vote_hash_parts.append(f"{item.state_key}:{_tensor_sha256(votes)}")
        eligible += int(item.state.q_levels.numel())

    if eligible <= 0:
        raise ValueError("zero eligible weights are not decision-density rows")

    global_cap_used = global_cap_spec is not None
    global_cap_row_ids: set[tuple[str, int]] = set()
    cap_frontier_ids: set[tuple[str, int]] = set()
    backlog_ids: set[tuple[str, int]] = set()
    global_cap_row_count = 0
    accepted_count = 0
    deferred_count = 0
    global_cap_saturated = False
    backlog: dict[str, dict[int, dict[str, int]]] = {}
    cap_step = int(global_cap_spec.step) if global_cap_spec is not None else 0
    if global_cap_spec is not None:
        offsets = tensor_offsets or tensor_offsets_for_vote_update_states(cap_inputs)
        cap_result = apply_global_rate_cap_reference(
            cap_inputs,
            global_cap_spec,
            deferred_backlog=deferred_backlog,
            tensor_offsets=offsets,
        )
        global_cap_row_ids = {(row.state_key, int(row.flat_index)) for row in cap_result.rows}
        ranking_exact_ids |= global_cap_row_ids
        accepted_count = len(cap_result.accepted_rows)
        deferred_count = len(cap_result.deferred_rows)
        global_cap_row_count = len(cap_result.rows)
        global_cap_saturated = bool(cap_result.step_summary["global_rate_cap_saturated"])
        width = int(cap_frontier_width)
        frontier_rows = cap_result.accepted_rows[-width:] + cap_result.deferred_rows[:width]
        cap_frontier_ids = {(row.state_key, int(row.flat_index)) for row in frontier_rows}
        backlog = cap_result.deferred_backlog
        backlog_ids = _backlog_identities(backlog)
    else:
        cap_frontier_ids = {
            identity
            for item in inputs
            for identity in _rows_from_tensor(
                plans[item.state_key].pre_veto_selected_indices[: int(cap_frontier_width)],
                item.state_key,
            )
        }
        backlog = deferred_backlog or {}
        backlog_ids = _backlog_identities(backlog)

    decision_relevant_ids = set()
    decision_relevant_ids |= active_next_step_ids
    decision_relevant_ids |= ranking_exact_ids
    decision_relevant_ids |= replay_veto_ids
    decision_relevant_ids |= backlog_ids

    far_values: list[torch.Tensor] = []
    relevant_by_key: dict[str, set[int]] = {}
    for state_key, flat_index in decision_relevant_ids:
        relevant_by_key.setdefault(state_key, set()).add(int(flat_index))
    for item in inputs:
        plan = plans[item.state_key]
        flat = plan.new_acc_i32.flatten().to(torch.int32)
        mask = torch.ones(int(flat.numel()), dtype=torch.bool)
        for flat_index in relevant_by_key.get(item.state_key, set()):
            if 0 <= flat_index < int(flat.numel()):
                mask[flat_index] = False
        far_values.append(flat[mask])
    far_tensor = torch.cat(far_values) if far_values else torch.empty(0, dtype=torch.int32)
    far_entropy = _far_entropy(far_tensor)

    metadata_bits = int(tensor_metadata_bits) if tensor_metadata_bits is not None else len(inputs) * 64
    projection = project_sparse_accumulator_bpw(
        eligible_weight_count=eligible,
        stored_row_count=len(decision_relevant_ids),
        target_bits_per_weight=target,
        value_bits_per_row=value_bits_per_row,
        flag_bits_per_row=flag_bits_per_row,
        tensor_metadata_bits=metadata_bits,
        backlog_entry_count=len(backlog_ids),
        entropy_model_bits=entropy_model_bits,
    )
    max_age, max_defer = _backlog_age_summary(backlog, step=cap_step)
    vote_hash = hashlib.sha256("|".join(sorted(vote_hash_parts)).encode("utf-8")).hexdigest()

    return AccumulatorDecisionDensityReport(
        schema_version=ACCUMULATOR_DECISION_DENSITY_SCHEMA_VERSION,
        label=ACCUMULATOR_DECISION_DENSITY_LABEL,
        eligible_weight_count=eligible,
        tensor_count=len(inputs),
        target_bits_per_weight=target,
        active_definition=ACTIVE_DEFINITION_NEXT_STEP_LAW,
        ranking_exact_density_definition=RANKING_EXACT_DENSITY_DEFINITION,
        cap_frontier_diagnostic_definition=CAP_FRONTIER_DIAGNOSTIC_DEFINITION,
        fixture_vote_record="actual_fixture_votes",
        fixture_vote_abs_max=vote_abs_max,
        fixture_vote_nonzero_count=vote_nonzero_count,
        fixture_vote_sha256=vote_hash,
        current_magnitude_threshold_count=len(current_threshold_ids),
        active_next_step_count=len(active_next_step_ids),
        ranking_sensitive_exact_count=len(ranking_exact_ids),
        cap_frontier_diagnostic_count=len(cap_frontier_ids),
        backlog_state_carry_count=len(backlog_ids),
        replay_veto_residual_count=len(replay_veto_ids),
        decision_relevant_exact_count=len(decision_relevant_ids),
        far_row_count=int(far_entropy.far_row_count),
        current_magnitude_threshold_density=_safe_density(len(current_threshold_ids), eligible),
        active_next_step_density=_safe_density(len(active_next_step_ids), eligible),
        ranking_sensitive_exact_density=_safe_density(len(ranking_exact_ids), eligible),
        cap_frontier_diagnostic_density=_safe_density(len(cap_frontier_ids), eligible),
        backlog_state_carry_density=_safe_density(len(backlog_ids), eligible),
        decision_relevant_exact_density=_safe_density(len(decision_relevant_ids), eligible),
        current_magnitude_threshold_indices_sha256=_identity_hash(current_threshold_ids),
        active_next_step_indices_sha256=_identity_hash(active_next_step_ids),
        ranking_sensitive_exact_indices_sha256=_identity_hash(ranking_exact_ids),
        cap_frontier_diagnostic_indices_sha256=_identity_hash(cap_frontier_ids),
        backlog_state_carry_indices_sha256=_identity_hash(backlog_ids),
        decision_relevant_exact_indices_sha256=_identity_hash(decision_relevant_ids),
        sparse_exact_projection=projection,
        far_value_entropy=far_entropy,
        global_cap_used=global_cap_used,
        global_cap_row_count=global_cap_row_count,
        global_cap_accepted_count=accepted_count,
        global_cap_deferred_count=deferred_count,
        global_cap_saturated=global_cap_saturated,
        backlog_max_age_steps=max_age,
        backlog_max_defer_count=max_defer,
        raw_arrays_included=False,
        non_claims=(
            "no accumulator encoder",
            "no q/selection/cap/apply logic change",
            "no GPU lane",
            "no trainer/live-run/checkpoint/creditdir mutation",
            "no acquisition or stability claim",
            "no inclusive physical sub-2 claim",
            "compact counts/hashes only; no raw per-weight arrays",
        ),
    )


def classify_accumulator_candidate(
    *,
    candidate_name: str,
    classification: CandidateClassification | str,
    projection: SparseAccumulatorBpwProjection,
    covered_decision_dimensions: Iterable[str] | None = None,
    compressed_representation: bool = True,
    bounded_delta_hypothesis: str | None = None,
    guardrail: str | None = None,
    note: str = "",
) -> AccumulatorCandidateClassification:
    """Classify a C1.1b candidate without allowing over-budget decision_exact."""

    requested = CandidateClassification(classification)
    covered = tuple(
        required_decision_dimension_names()
        if covered_decision_dimensions is None
        else covered_decision_dimensions
    )
    missing = tuple(name for name in required_decision_dimension_names() if name not in set(covered))
    if requested == CandidateClassification.DECISION_EXACT and (
        not projection.fits_target or missing
    ):
        reasons: list[str] = []
        if not projection.fits_target:
            reasons.append(
                "overhead-inclusive projected bpw exceeds C1.1a accumulator budget"
            )
        if missing:
            reasons.append(f"missing required decision dimensions: {missing}")
        return AccumulatorCandidateClassification(
            candidate_name=candidate_name,
            classification=DECISION_EXACT_INFEASIBLE,
            projected_bits_per_weight=projection.projected_bits_per_weight,
            target_bits_per_weight=projection.target_bits_per_weight,
            decision_exact_feasible=False,
            candidate_assessment=None,
            missing_decision_dimensions=missing,
            infeasibility_reason="; ".join(reasons),
        )
    assessment = candidate_assessment(
        candidate_name=candidate_name,
        classification=requested,
        covered_decision_dimensions=covered,
        compressed_representation=compressed_representation,
        bounded_delta_hypothesis=bounded_delta_hypothesis,
        guardrail=guardrail,
        note=note,
    )
    return AccumulatorCandidateClassification(
        candidate_name=candidate_name,
        classification=assessment.normalized_classification.value,
        projected_bits_per_weight=projection.projected_bits_per_weight,
        target_bits_per_weight=projection.target_bits_per_weight,
        decision_exact_feasible=(
            assessment.normalized_classification == CandidateClassification.DECISION_EXACT
            and projection.fits_target
        ),
        candidate_assessment=assessment,
        missing_decision_dimensions=assessment.missing_decision_dimensions,
        infeasibility_reason="",
    )


def assess_default_accumulator_candidates(
    report: AccumulatorDecisionDensityReport,
) -> tuple[AccumulatorCandidateClassification, ...]:
    """Assess the four C1.1b representation families without encoding them."""

    dims = required_decision_dimension_names()
    projection = report.sparse_exact_projection
    return (
        classify_accumulator_candidate(
            candidate_name="sparse_exact_decision_set",
            classification=CandidateClassification.DECISION_EXACT,
            projection=projection,
            covered_decision_dimensions=dims,
            note="exact int16 only for measured decision-relevant rows",
        ),
        classify_accumulator_candidate(
            candidate_name="event_coded_crossing_residual_log",
            classification=CandidateClassification.BOUNDED_DELTA_WITH_REPORT,
            projection=projection,
            covered_decision_dimensions=dims,
            bounded_delta_hypothesis=(
                "bounded event queue/checkpoint schedule preserves decisions only "
                "when event and backlog growth remain under the measured budget"
            ),
            guardrail=(
                "report candidate/order/accepted/deferred/backlog/hash drift before C2"
            ),
            note="assessment only; no event encoder implemented",
        ),
        classify_accumulator_candidate(
            candidate_name="bucketed_residual_with_exact_guard_band",
            classification=CandidateClassification.BOUNDED_DELTA_WITH_REPORT,
            projection=projection,
            covered_decision_dimensions=dims,
            bounded_delta_hypothesis=(
                "coarse far-row residual buckets are safe only when exact guard "
                "band catches every decision-changing row"
            ),
            guardrail=(
                "report changed candidate/order/accepted/deferred/q_changed/residual hash counts"
            ),
            note="assessment only; no bucket encoder implemented",
        ),
        classify_accumulator_candidate(
            candidate_name="hybrid_hot_exact_cold_bucket",
            classification=CandidateClassification.BOUNDED_DELTA_WITH_REPORT,
            projection=projection,
            covered_decision_dimensions=dims,
            bounded_delta_hypothesis=(
                "hot rows stay exact while cold buckets remain decision-inert under the guard"
            ),
            guardrail="prove accepted/deferred identity unchanged or report bounded drift",
            note="assessment only; no hybrid encoder implemented",
        ),
    )


def validate_accumulator_decision_density_report(
    report: AccumulatorDecisionDensityReport,
    *,
    claimed_decision_exact_physical_sub2: bool = False,
) -> None:
    """False-claim and compactness guard for C1.1b measurement reports."""

    if report.raw_arrays_included:
        raise ValueError("compact C1.1b report must not include raw per-weight arrays")
    if report.eligible_weight_count <= 0:
        raise ValueError("decision-density report requires positive eligible weights")
    if report.decision_relevant_exact_count > report.eligible_weight_count:
        raise ValueError("decision-relevant count cannot exceed eligible weights")
    if report.far_row_count + report.decision_relevant_exact_count != report.eligible_weight_count:
        raise ValueError("far rows plus decision-relevant rows must cover the measured surface")
    if report.far_value_entropy.density_note != VALUE_ENTROPY_IS_NOT_DECISION_EXACT:
        raise ValueError("far-row entropy must be labeled separate from decision_exact")
    if claimed_decision_exact_physical_sub2 and not report.sparse_exact_projection.fits_target:
        raise ValueError(
            "physical sub-2 decision_exact claim is invalid when overhead-inclusive "
            "sparse projection exceeds the C1.1a accumulator budget"
        )
