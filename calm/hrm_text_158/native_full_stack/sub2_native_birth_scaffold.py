"""Strict sub-2 native-birth scaffold reporting for HRM-Text-1.58.

This slice is intentionally scaffold-first and fail-closed. It does NOT claim
an executable sub-2 learner. Instead it emits:

- the binding persistent candidate-path ledger subtotal,
- explicit off-path controls/blockers,
- adjacent transient/runtime ledgers kept separate from persistent authority,
- the dense-baseline parity contract for the first executable sub-2 proof.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from typing import Any, Mapping, Sequence

from calm.hrm_text_158.native_full_stack.bounded_delta_accumulator import (
    ACCUMULATOR_SUBSTITUTE_LOCAL_VOTE_UPDATE_EXECUTABLE,
    ALGORITHMIC_LOCAL_VOTE_UPDATE_EXECUTABLE_NOT_PHYSICAL_SUB2,
    HYBRID_HOT_EXACT_COLD_DEFAULT_CANDIDATE,
    INTRINSIC_BOUNDED_UPDATE_DOMAIN_GAP,
    bounded_delta_admission_contract,
    bounded_delta_candidate_assessment,
)
from calm.hrm_text_158.native_full_stack.fp_exceptions import (
    HIDDEN_FP_LEARNER_FAIL_STATE,
)
from calm.hrm_text_158.native_full_stack.q_entropy_packing import (
    base3_q_entropy_ledger_for_shapes,
    base3_q_storage_orthogonality_report,
)


STRICT_SUB2_CANDIDATE_RUNTIME_SCAFFOLD_SCHEMA_VERSION = (
    "hrm_text_158_strict_sub2_candidate_runtime_scaffold/v0"
)
STRICT_SUB2_CANDIDATE_RUNTIME_SCAFFOLD_LABEL = (
    "strict_sub2_candidate_runtime_scaffold"
)
STRICT_SUB2_CANDIDATE_RUNTIME_TARGET_NAME = (
    "strict_sub2_candidate_runtime_scaffold"
)

RUNTIME_STATE_AUTHORITY_DENSE_CONTROL = "dense_control"
RUNTIME_STATE_AUTHORITY_SUB2_SCAFFOLD_ONLY = "sub2_scaffold_only"
RUNTIME_STATE_AUTHORITY_SUB2_PERSISTENT_HYBRID_DENSE_TRANSIENT_CREDIT = (
    "sub2_persistent_hybrid_dense_transient_credit"
)
RUNTIME_STATE_AUTHORITY_SUB2_CANDIDATE_EXECUTABLE = "sub2_candidate_executable"

LEDGER_CLASS_LEQ2 = "<=2_bits"
LEDGER_CLASS_EXECUTABLE = "sub2_candidate_executable"
LEDGER_CLASS_NOT_YET = "not_yet_in_candidate_runtime"

PERSISTENT_CANDIDATE_SECTION = "persistent_candidate"
OFF_PATH_CONTROL_SECTION = "off_path_control"
ADJACENT_RUNTIME_SECTION = "adjacent_runtime"

ACQUISITION_GATE_DEFERRED = "deferred_until_after_parity_non_regression"
ACQUISITION_GATE_UNBLOCKED_NOT_RUN = "unblocked_not_run"

HYBRID_SCOPE_DECISION_SOURCE_MSG_ID = "1780806189949-f8a44a15"
HYBRID_SCOPE_DECISION_LOCKED_OPTION = "Option A"
HYBRID_SCOPE_DECISION_LOCKED_ANSWER = "Pragmatic hybrid"
DENSE_TRANSIENT_CREDIT_ROLE_TRAINING_COMPUTE_CONTROL_ONLY = (
    "training_compute_control_only"
)
STRICT_SUB2_HYBRID_PERSISTENT_SIDECAR_LEDGER_SCHEMA_VERSION = (
    "hrm_text_158_strict_sub2_hybrid_persistent_sidecar_ledger/v0"
)
STRICT_SUB2_HYBRID_RUNTIME_MOVEMENT_OVERLAY_SCHEMA_VERSION = (
    "hrm_text_158_strict_sub2_hybrid_runtime_movement_overlay/v0"
)
STRICT_SUB2_HYBRID_RUNTIME_MOVEMENT_OVERLAY_LABEL = (
    "strict_sub2_hybrid_runtime_movement_overlay"
)
HYBRID_PERSISTENT_MODE_APPLIED_CROSSING_DIRECTION_PLUS_4BIT_RESIDUAL = (
    "applied_crossing_direction_plus_4bit_residual"
)
HYBRID_PERSISTENT_MODE_ZERO_PERSISTENT_ACCUMULATOR = (
    "zero_persistent_accumulator"
)
HYBRID_MOVEMENT_CONTRACT_SCOPE = "native_hybrid_persistent_sub2_movement_smoke"
HYBRID_MOVEMENT_METRIC_NAME = "support_wide_strict_exact_best_delta"
ACQUISITION_GATE_RUNNING = "running"
ACQUISITION_GATE_RESULT = "result"


@dataclass(frozen=True)
class StrictSub2ScaffoldRow:
    name: str
    section: str
    classification: str
    in_candidate_authority: bool
    counted_in_physical_persistent_bpw: bool
    bits_per_weight: float | None
    blocker: bool
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StrictSub2CandidateRuntimeScaffoldReport:
    schema_version: str
    label: str
    target_name: str
    runtime_state_authority: str
    pass_report: bool
    eligible_module_count: int
    eligible_weight_count: int
    physical_persistent_bpw: float
    physical_persistent_target_bpw: float
    physical_persistent_target_pass: bool
    physical_persistent_interpretation: str
    persistent_sub2_hybrid_only: bool
    dense_transient_credit_allowed: bool
    dense_transient_credit_role: str
    dense_transient_credit_counted_in_physical_persistent_bpw: bool
    transient_debt_present: bool
    transient_debt_non_blocking: bool
    transient_debt_row_names: tuple[str, ...]
    full_runtime_sub2_achieved: bool
    native_transient_sub2_achieved: bool
    fully_fp_free_achieved: bool
    scope_decision_source_msg_id: str
    scope_decision_locked_option: str
    scope_decision_locked_answer: str
    acquisition_science_status: str
    acquisition_achieved: bool
    candidate_runtime_complete: bool
    candidate_authority_row_names: tuple[str, ...]
    blocker_names: tuple[str, ...]
    persistent_candidate_rows: tuple[StrictSub2ScaffoldRow, ...]
    off_path_control_rows: tuple[StrictSub2ScaffoldRow, ...]
    adjacent_runtime_rows: tuple[StrictSub2ScaffoldRow, ...]
    q_storage_orthogonality: dict[str, Any]
    parity_contract: dict[str, Any]
    acquisition_gate: dict[str, Any]
    hot_loop_residency: dict[str, Any]
    hidden_fp_learner_fail_state: str
    scoped_candidate_proof: dict[str, Any] | None
    non_claims: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "label": self.label,
            "target_name": self.target_name,
            "runtime_state_authority": self.runtime_state_authority,
            "pass": bool(self.pass_report),
            "eligible_module_count": int(self.eligible_module_count),
            "eligible_weight_count": int(self.eligible_weight_count),
            "physical_persistent_bpw": float(self.physical_persistent_bpw),
            "physical_persistent_target_bpw": float(self.physical_persistent_target_bpw),
            "physical_persistent_target_pass": bool(self.physical_persistent_target_pass),
            "physical_persistent_interpretation": self.physical_persistent_interpretation,
            "persistent_sub2_hybrid_only": bool(self.persistent_sub2_hybrid_only),
            "dense_transient_credit_allowed": bool(self.dense_transient_credit_allowed),
            "dense_transient_credit_role": self.dense_transient_credit_role,
            "dense_transient_credit_counted_in_physical_persistent_bpw": bool(
                self.dense_transient_credit_counted_in_physical_persistent_bpw
            ),
            "transient_debt_present": bool(self.transient_debt_present),
            "transient_debt_non_blocking": bool(self.transient_debt_non_blocking),
            "transient_debt_row_names": list(self.transient_debt_row_names),
            "full_runtime_sub2_achieved": bool(self.full_runtime_sub2_achieved),
            "native_transient_sub2_achieved": bool(self.native_transient_sub2_achieved),
            "fully_fp_free_achieved": bool(self.fully_fp_free_achieved),
            "scope_decision_source_msg_id": self.scope_decision_source_msg_id,
            "scope_decision_locked_option": self.scope_decision_locked_option,
            "scope_decision_locked_answer": self.scope_decision_locked_answer,
            "acquisition_science_status": self.acquisition_science_status,
            "acquisition_achieved": bool(self.acquisition_achieved),
            "candidate_runtime_complete": bool(self.candidate_runtime_complete),
            "candidate_authority_row_names": list(self.candidate_authority_row_names),
            "blocker_names": list(self.blocker_names),
            "persistent_candidate_rows": [row.to_dict() for row in self.persistent_candidate_rows],
            "off_path_control_rows": [row.to_dict() for row in self.off_path_control_rows],
            "adjacent_runtime_rows": [row.to_dict() for row in self.adjacent_runtime_rows],
            "q_storage_orthogonality": dict(self.q_storage_orthogonality),
            "parity_contract": dict(self.parity_contract),
            "acquisition_gate": dict(self.acquisition_gate),
            "hot_loop_residency": dict(self.hot_loop_residency),
            "hidden_fp_learner_fail_state": self.hidden_fp_learner_fail_state,
            "scoped_candidate_proof": (
                None if self.scoped_candidate_proof is None else dict(self.scoped_candidate_proof)
            ),
            "non_claims": list(self.non_claims),
        }


@dataclass(frozen=True)
class HybridPersistentSidecarLedger:
    schema_version: str
    label: str
    persistent_mode: str
    eligible_module_count: int
    eligible_weight_count: int
    total_event_count: int
    index_bits_kind: str
    direction_bits_per_event: int
    residual_bits_per_event: int
    total_metadata_bits: int
    packet_count_bits_formula: str
    q_bits_per_weight: float
    frozen_scale_bits_per_weight: float
    sidecar_bits_per_weight: float
    inclusive_bits_per_weight: float
    row_only_lt2: bool
    inclusive_lt2: bool
    shape_breakdown: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "label": self.label,
            "persistent_mode": self.persistent_mode,
            "eligible_module_count": int(self.eligible_module_count),
            "eligible_weight_count": int(self.eligible_weight_count),
            "total_event_count": int(self.total_event_count),
            "index_bits_kind": self.index_bits_kind,
            "direction_bits_per_event": int(self.direction_bits_per_event),
            "residual_bits_per_event": int(self.residual_bits_per_event),
            "total_metadata_bits": int(self.total_metadata_bits),
            "packet_count_bits_formula": self.packet_count_bits_formula,
            "q_bits_per_weight": float(self.q_bits_per_weight),
            "frozen_scale_bits_per_weight": float(self.frozen_scale_bits_per_weight),
            "sidecar_bits_per_weight": float(self.sidecar_bits_per_weight),
            "inclusive_bits_per_weight": float(self.inclusive_bits_per_weight),
            "row_only_lt2": bool(self.row_only_lt2),
            "inclusive_lt2": bool(self.inclusive_lt2),
            "shape_breakdown": [dict(row) for row in self.shape_breakdown],
        }


@dataclass(frozen=True)
class StrictSub2HybridRuntimeMovementOverlay:
    schema_version: str
    label: str
    base_scaffold_schema_version: str
    base_scaffold_label: str
    runtime_state_authority: str
    persistent_mode: str
    pass_report: bool
    persistent_dense_shadow_present: bool
    persistent_dense_shadow_bytes: int
    bounded_only_collapse: bool
    local_update_law_label: str
    local_update_law_reused: bool
    second_update_law_required: bool
    dense_transient_credit_allowed: bool
    dense_transient_credit_role: str
    transient_debt_present: bool
    persistent_sub2_hybrid_only: bool
    full_runtime_sub2_achieved: bool
    candidate_runtime_complete: bool
    acquisition_science_status: str
    acquisition_achieved: bool
    movement_contract_scope: str
    movement_metric_name: str
    movement_metric_min_delta: int
    q_changed_must_be_positive: bool
    hard_fail_required_false: bool
    live_dense_transient_selection_role: str
    persistent_authority_row_names: tuple[str, ...]
    blocked_row_names: tuple[str, ...]
    persistent_sidecar_ledger: dict[str, Any]
    claim_boundary: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "label": self.label,
            "base_scaffold_schema_version": self.base_scaffold_schema_version,
            "base_scaffold_label": self.base_scaffold_label,
            "runtime_state_authority": self.runtime_state_authority,
            "persistent_mode": self.persistent_mode,
            "pass": bool(self.pass_report),
            "persistent_dense_shadow_present": bool(self.persistent_dense_shadow_present),
            "persistent_dense_shadow_bytes": int(self.persistent_dense_shadow_bytes),
            "bounded_only_collapse": bool(self.bounded_only_collapse),
            "local_update_law_label": self.local_update_law_label,
            "local_update_law_reused": bool(self.local_update_law_reused),
            "second_update_law_required": bool(self.second_update_law_required),
            "dense_transient_credit_allowed": bool(self.dense_transient_credit_allowed),
            "dense_transient_credit_role": self.dense_transient_credit_role,
            "transient_debt_present": bool(self.transient_debt_present),
            "persistent_sub2_hybrid_only": bool(self.persistent_sub2_hybrid_only),
            "full_runtime_sub2_achieved": bool(self.full_runtime_sub2_achieved),
            "candidate_runtime_complete": bool(self.candidate_runtime_complete),
            "acquisition_science_status": self.acquisition_science_status,
            "acquisition_achieved": bool(self.acquisition_achieved),
            "movement_contract_scope": self.movement_contract_scope,
            "movement_metric_name": self.movement_metric_name,
            "movement_metric_min_delta": int(self.movement_metric_min_delta),
            "q_changed_must_be_positive": bool(self.q_changed_must_be_positive),
            "hard_fail_required_false": bool(self.hard_fail_required_false),
            "live_dense_transient_selection_role": self.live_dense_transient_selection_role,
            "persistent_authority_row_names": list(self.persistent_authority_row_names),
            "blocked_row_names": list(self.blocked_row_names),
            "persistent_sidecar_ledger": dict(self.persistent_sidecar_ledger),
            "claim_boundary": list(self.claim_boundary),
        }


def _shape_tuple(shape: Sequence[int]) -> tuple[int, ...]:
    out = tuple(int(dim) for dim in shape)
    if not out or any(dim <= 0 for dim in out):
        raise ValueError(f"eligible module shapes must be non-empty positive tuples, got {shape!r}")
    return out


def _numel(shape: Sequence[int]) -> int:
    out = 1
    for dim in shape:
        out *= int(dim)
    return int(out)


def _index_bits_for_numel(numel: int) -> int:
    numel = int(numel)
    if numel <= 0:
        raise ValueError("numel must be > 0")
    return int(math.ceil(math.log2(numel))) if numel > 1 else 1


def _scale_bits_per_weight(scale_count: int, eligible_weight_count: int) -> float:
    return float(scale_count * 32) / float(eligible_weight_count)


def _max_activation_bits_per_value(activation_paid_bits_ledger: Mapping[str, Any]) -> float | None:
    surfaces = activation_paid_bits_ledger.get("surfaces")
    if not isinstance(surfaces, Sequence) or not surfaces:
        return None
    values = []
    for row in surfaces:
        if not isinstance(row, Mapping):
            continue
        paid = row.get("paid_bits_per_value")
        if paid is None:
            continue
        values.append(float(paid))
    return max(values) if values else None


def _all_rows(
    report: StrictSub2CandidateRuntimeScaffoldReport,
) -> tuple[StrictSub2ScaffoldRow, ...]:
    return (
        report.persistent_candidate_rows
        + report.off_path_control_rows
        + report.adjacent_runtime_rows
    )


def validate_hybrid_persistent_sidecar_ledger(
    ledger: HybridPersistentSidecarLedger,
) -> None:
    if (
        ledger.schema_version
        != STRICT_SUB2_HYBRID_PERSISTENT_SIDECAR_LEDGER_SCHEMA_VERSION
    ):
        raise ValueError("hybrid persistent sidecar ledger schema version mismatch")
    if ledger.label != "strict_sub2_hybrid_persistent_sidecar_ledger":
        raise ValueError("hybrid persistent sidecar ledger label mismatch")
    if ledger.persistent_mode not in {
        HYBRID_PERSISTENT_MODE_APPLIED_CROSSING_DIRECTION_PLUS_4BIT_RESIDUAL,
        HYBRID_PERSISTENT_MODE_ZERO_PERSISTENT_ACCUMULATOR,
    }:
        raise ValueError("unknown hybrid persistent sidecar mode")
    if int(ledger.eligible_module_count) <= 0:
        raise ValueError("eligible_module_count must be > 0")
    if int(ledger.eligible_weight_count) <= 0:
        raise ValueError("eligible_weight_count must be > 0")
    if int(ledger.total_event_count) < 0:
        raise ValueError("total_event_count must be >= 0")
    if ledger.index_bits_kind != "per_tensor_local":
        raise ValueError("index_bits_kind must disclose per_tensor_local accounting")
    if int(ledger.direction_bits_per_event) != 1:
        raise ValueError("direction_bits_per_event must be 1")
    if ledger.persistent_mode == HYBRID_PERSISTENT_MODE_APPLIED_CROSSING_DIRECTION_PLUS_4BIT_RESIDUAL:
        if int(ledger.residual_bits_per_event) != 4:
            raise ValueError("4-bit sidecar mode must disclose residual_bits_per_event=4")
    if int(ledger.residual_bits_per_event) < 0:
        raise ValueError("residual_bits_per_event must be >= 0")
    if int(ledger.total_metadata_bits) < 0:
        raise ValueError("total_metadata_bits must be >= 0")
    if ledger.packet_count_bits_formula != "ceil(log2(numel + 1)) per tensor":
        raise ValueError("packet_count_bits_formula must stay explicit")
    if float(ledger.q_bits_per_weight) <= 0.0:
        raise ValueError("q_bits_per_weight must be > 0")
    if float(ledger.frozen_scale_bits_per_weight) < 0.0:
        raise ValueError("frozen_scale_bits_per_weight must be >= 0")
    if float(ledger.sidecar_bits_per_weight) < 0.0:
        raise ValueError("sidecar_bits_per_weight must be >= 0")
    if abs(
        float(ledger.inclusive_bits_per_weight)
        - (
            float(ledger.q_bits_per_weight)
            + float(ledger.frozen_scale_bits_per_weight)
            + float(ledger.sidecar_bits_per_weight)
        )
    ) > 1e-12:
        raise ValueError("inclusive_bits_per_weight must equal q + scale + sidecar")
    if bool(ledger.row_only_lt2) != (float(ledger.sidecar_bits_per_weight) < 2.0):
        raise ValueError("row_only_lt2 must be computed from sidecar_bits_per_weight")
    if bool(ledger.inclusive_lt2) != (float(ledger.inclusive_bits_per_weight) < 2.0):
        raise ValueError("inclusive_lt2 must be computed from inclusive_bits_per_weight")
    breakdown = tuple(ledger.shape_breakdown)
    if len(breakdown) != int(ledger.eligible_module_count):
        raise ValueError("shape_breakdown length must equal eligible_module_count")
    event_total = 0
    metadata_total = 0
    eligible_total = 0
    sidecar_total_bits = 0
    for row in breakdown:
        shape = tuple(int(dim) for dim in row.get("shape", ()))
        if not shape or any(dim <= 0 for dim in shape):
            raise ValueError("shape_breakdown rows must disclose positive shapes")
        eligible = int(row.get("eligible_weight_count", 0))
        if eligible != _numel(shape):
            raise ValueError("shape_breakdown eligible_weight_count must match shape numel")
        event_count = int(row.get("event_count", -1))
        if event_count < 0 or event_count > eligible:
            raise ValueError("shape_breakdown event_count must be in [0, eligible_weight_count]")
        index_bits = int(row.get("index_bits", 0))
        if index_bits != _index_bits_for_numel(eligible):
            raise ValueError("shape_breakdown index_bits must be the local per-tensor index width")
        packet_count_bits = int(row.get("packet_count_bits", -1))
        expected_packet_bits = int(math.ceil(math.log2(eligible + 1)))
        if packet_count_bits != expected_packet_bits:
            raise ValueError("shape_breakdown packet_count_bits must equal ceil(log2(numel + 1))")
        metadata_bits = int(row.get("metadata_bits", -1))
        if metadata_bits < packet_count_bits:
            raise ValueError("shape_breakdown metadata_bits must include packet_count_bits")
        total_bits = int(row.get("total_bits", -1))
        expected_total_bits = metadata_bits + event_count * (
            index_bits
            + int(ledger.direction_bits_per_event)
            + int(ledger.residual_bits_per_event)
        )
        if total_bits != expected_total_bits:
            raise ValueError("shape_breakdown total_bits must match the explicit codec formula")
        event_total += event_count
        metadata_total += metadata_bits
        eligible_total += eligible
        sidecar_total_bits += total_bits
    if event_total != int(ledger.total_event_count):
        raise ValueError("total_event_count must equal the sum of shape_breakdown event counts")
    if metadata_total != int(ledger.total_metadata_bits):
        raise ValueError("total_metadata_bits must equal the sum of shape_breakdown metadata bits")
    if eligible_total != int(ledger.eligible_weight_count):
        raise ValueError("eligible_weight_count must equal the sum of shape_breakdown eligible weights")
    if abs(float(ledger.sidecar_bits_per_weight) - (float(sidecar_total_bits) / float(eligible_total))) > 1e-12:
        raise ValueError("sidecar_bits_per_weight must equal aggregate sidecar bits / eligible_weight_count")


def build_hybrid_persistent_sidecar_ledger(
    *,
    logical_shapes: Sequence[Sequence[int]],
    event_counts: Sequence[int],
    persistent_mode: str,
    residual_bits_per_event: int,
    direction_bits_per_event: int = 1,
    extra_metadata_bits_per_tensor: int = 0,
    target_bits_per_weight: float = 2.0,
) -> HybridPersistentSidecarLedger:
    if not logical_shapes:
        raise ValueError("logical_shapes must be non-empty")
    if len(logical_shapes) != len(event_counts):
        raise ValueError("logical_shapes and event_counts must have identical length")
    ordered_shapes = tuple(_shape_tuple(shape) for shape in logical_shapes)
    if int(direction_bits_per_event) != 1:
        raise ValueError("direction_bits_per_event must stay 1 for this codec family")
    if int(residual_bits_per_event) < 0:
        raise ValueError("residual_bits_per_event must be >= 0")
    if int(extra_metadata_bits_per_tensor) < 0:
        raise ValueError("extra_metadata_bits_per_tensor must be >= 0")
    eligible_weight_count = sum(_numel(shape) for shape in ordered_shapes)
    q_row = base3_q_entropy_ledger_for_shapes(
        regime_name="strict_sub2_hybrid_sidecar_q_shape_ledger",
        logical_shapes=ordered_shapes,
        scale_count=0,
        accumulator_bits_per_weight=0.0,
    )
    scale_bpw = _scale_bits_per_weight(len(ordered_shapes), eligible_weight_count)
    shape_breakdown: list[dict[str, Any]] = []
    total_event_count = 0
    total_metadata_bits = 0
    total_sidecar_bits = 0
    for shape, raw_event_count in zip(ordered_shapes, event_counts):
        eligible = _numel(shape)
        event_count = int(raw_event_count)
        if event_count < 0 or event_count > eligible:
            raise ValueError("event_counts entries must be in [0, tensor_numel]")
        index_bits = _index_bits_for_numel(eligible)
        packet_count_bits = int(math.ceil(math.log2(eligible + 1)))
        metadata_bits = packet_count_bits + int(extra_metadata_bits_per_tensor)
        total_bits = metadata_bits + event_count * (
            index_bits
            + int(direction_bits_per_event)
            + int(residual_bits_per_event)
        )
        shape_breakdown.append(
            {
                "shape": list(shape),
                "eligible_weight_count": int(eligible),
                "event_count": int(event_count),
                "index_bits": int(index_bits),
                "packet_count_bits": int(packet_count_bits),
                "metadata_bits": int(metadata_bits),
                "total_bits": int(total_bits),
                "bits_per_weight": float(total_bits) / float(eligible),
            }
        )
        total_event_count += event_count
        total_metadata_bits += metadata_bits
        total_sidecar_bits += total_bits
    sidecar_bpw = float(total_sidecar_bits) / float(eligible_weight_count)
    inclusive_bpw = (
        float(q_row.q_packed_total_bits_per_weight)
        + float(scale_bpw)
        + float(sidecar_bpw)
    )
    ledger = HybridPersistentSidecarLedger(
        schema_version=STRICT_SUB2_HYBRID_PERSISTENT_SIDECAR_LEDGER_SCHEMA_VERSION,
        label="strict_sub2_hybrid_persistent_sidecar_ledger",
        persistent_mode=persistent_mode,
        eligible_module_count=len(ordered_shapes),
        eligible_weight_count=eligible_weight_count,
        total_event_count=int(total_event_count),
        index_bits_kind="per_tensor_local",
        direction_bits_per_event=int(direction_bits_per_event),
        residual_bits_per_event=int(residual_bits_per_event),
        total_metadata_bits=int(total_metadata_bits),
        packet_count_bits_formula="ceil(log2(numel + 1)) per tensor",
        q_bits_per_weight=float(q_row.q_packed_total_bits_per_weight),
        frozen_scale_bits_per_weight=float(scale_bpw),
        sidecar_bits_per_weight=float(sidecar_bpw),
        inclusive_bits_per_weight=float(inclusive_bpw),
        row_only_lt2=bool(sidecar_bpw < float(target_bits_per_weight)),
        inclusive_lt2=bool(inclusive_bpw < float(target_bits_per_weight)),
        shape_breakdown=tuple(shape_breakdown),
    )
    validate_hybrid_persistent_sidecar_ledger(ledger)
    return ledger


def validate_strict_sub2_hybrid_runtime_movement_overlay(
    report: StrictSub2HybridRuntimeMovementOverlay,
) -> None:
    if (
        report.schema_version
        != STRICT_SUB2_HYBRID_RUNTIME_MOVEMENT_OVERLAY_SCHEMA_VERSION
    ):
        raise ValueError("hybrid runtime movement overlay schema version mismatch")
    if report.label != STRICT_SUB2_HYBRID_RUNTIME_MOVEMENT_OVERLAY_LABEL:
        raise ValueError("hybrid runtime movement overlay label mismatch")
    if (
        report.base_scaffold_schema_version
        != STRICT_SUB2_CANDIDATE_RUNTIME_SCAFFOLD_SCHEMA_VERSION
    ):
        raise ValueError("base scaffold schema version mismatch")
    if report.base_scaffold_label != STRICT_SUB2_CANDIDATE_RUNTIME_SCAFFOLD_LABEL:
        raise ValueError("base scaffold label mismatch")
    if (
        report.runtime_state_authority
        != RUNTIME_STATE_AUTHORITY_SUB2_PERSISTENT_HYBRID_DENSE_TRANSIENT_CREDIT
    ):
        raise ValueError("hybrid runtime overlay must stay on the Option A hybrid authority")
    if report.persistent_mode not in {
        HYBRID_PERSISTENT_MODE_APPLIED_CROSSING_DIRECTION_PLUS_4BIT_RESIDUAL,
        HYBRID_PERSISTENT_MODE_ZERO_PERSISTENT_ACCUMULATOR,
    }:
        raise ValueError("unknown hybrid runtime overlay persistent_mode")
    if bool(report.persistent_dense_shadow_present):
        raise ValueError("hybrid runtime overlay cannot persist dense shadow state")
    if int(report.persistent_dense_shadow_bytes) != 0:
        raise ValueError("persistent_dense_shadow_bytes must be 0 on the candidate path")
    if not bool(report.bounded_only_collapse):
        raise ValueError("hybrid runtime overlay must disclose bounded_only_collapse=true")
    if not bool(report.local_update_law_reused):
        raise ValueError("hybrid runtime overlay must reuse the already-proved local update law")
    if bool(report.second_update_law_required):
        raise ValueError("hybrid runtime overlay must fail closed on second_update_law_required")
    if not bool(report.dense_transient_credit_allowed):
        raise ValueError("hybrid runtime overlay must disclose dense_transient_credit_allowed=true")
    if (
        report.dense_transient_credit_role
        != DENSE_TRANSIENT_CREDIT_ROLE_TRAINING_COMPUTE_CONTROL_ONLY
    ):
        raise ValueError("hybrid runtime overlay must preserve the training_compute_control_only role")
    if not bool(report.transient_debt_present):
        raise ValueError("hybrid runtime overlay must disclose transient_debt_present=true")
    if not bool(report.persistent_sub2_hybrid_only):
        raise ValueError("hybrid runtime overlay must preserve persistent_sub2_hybrid_only=true")
    if bool(report.full_runtime_sub2_achieved):
        raise ValueError("hybrid runtime overlay cannot claim full_runtime_sub2_achieved")
    if bool(report.candidate_runtime_complete):
        raise ValueError("hybrid runtime overlay cannot claim candidate_runtime_complete")
    if report.acquisition_science_status not in {
        ACQUISITION_GATE_UNBLOCKED_NOT_RUN,
        ACQUISITION_GATE_RUNNING,
        ACQUISITION_GATE_RESULT,
    }:
        raise ValueError("hybrid runtime overlay acquisition_science_status is unknown")
    if bool(report.acquisition_achieved):
        raise ValueError("hybrid runtime overlay cannot claim acquisition_achieved under this slice")
    if report.movement_contract_scope != HYBRID_MOVEMENT_CONTRACT_SCOPE:
        raise ValueError("hybrid runtime overlay must use the movement-first sibling contract scope")
    if report.movement_metric_name != HYBRID_MOVEMENT_METRIC_NAME:
        raise ValueError("hybrid runtime overlay must lock the movement metric name")
    if int(report.movement_metric_min_delta) != 1:
        raise ValueError("hybrid runtime overlay must require best strict-exact delta >= 1")
    if not bool(report.q_changed_must_be_positive):
        raise ValueError("hybrid runtime overlay must require q_changed_count > 0")
    if not bool(report.hard_fail_required_false):
        raise ValueError("hybrid runtime overlay must keep the no-hard-fail bar explicit")
    if (
        report.live_dense_transient_selection_role
        != DENSE_TRANSIENT_CREDIT_ROLE_TRAINING_COMPUTE_CONTROL_ONLY
    ):
        raise ValueError("hybrid runtime overlay must disclose the dense-transient selector role exactly")
    if tuple(report.persistent_authority_row_names) != (
        "q_storage",
        "frozen_scales_fp32_metadata",
        "accumulator_sidecar",
    ):
        raise ValueError("hybrid runtime overlay must name q + frozen scales + accumulator_sidecar as the persistent authority rows")
    if "accumulator_substitute" not in tuple(report.blocked_row_names):
        raise ValueError("hybrid runtime overlay must keep accumulator_substitute blocked")
    if "attention_kv_append_update" not in tuple(report.blocked_row_names):
        raise ValueError("hybrid runtime overlay must keep attention_kv_append_update blocked")
    if "qacc_hot_loop_residency" not in tuple(report.blocked_row_names):
        raise ValueError("hybrid runtime overlay must keep qacc_hot_loop_residency blocked")
    ledger = HybridPersistentSidecarLedger(
        schema_version=str(report.persistent_sidecar_ledger.get("schema_version")),
        label=str(report.persistent_sidecar_ledger.get("label")),
        persistent_mode=str(report.persistent_sidecar_ledger.get("persistent_mode")),
        eligible_module_count=int(report.persistent_sidecar_ledger.get("eligible_module_count", 0)),
        eligible_weight_count=int(report.persistent_sidecar_ledger.get("eligible_weight_count", 0)),
        total_event_count=int(report.persistent_sidecar_ledger.get("total_event_count", 0)),
        index_bits_kind=str(report.persistent_sidecar_ledger.get("index_bits_kind")),
        direction_bits_per_event=int(report.persistent_sidecar_ledger.get("direction_bits_per_event", 0)),
        residual_bits_per_event=int(report.persistent_sidecar_ledger.get("residual_bits_per_event", 0)),
        total_metadata_bits=int(report.persistent_sidecar_ledger.get("total_metadata_bits", 0)),
        packet_count_bits_formula=str(report.persistent_sidecar_ledger.get("packet_count_bits_formula")),
        q_bits_per_weight=float(report.persistent_sidecar_ledger.get("q_bits_per_weight", 0.0)),
        frozen_scale_bits_per_weight=float(report.persistent_sidecar_ledger.get("frozen_scale_bits_per_weight", 0.0)),
        sidecar_bits_per_weight=float(report.persistent_sidecar_ledger.get("sidecar_bits_per_weight", 0.0)),
        inclusive_bits_per_weight=float(report.persistent_sidecar_ledger.get("inclusive_bits_per_weight", 0.0)),
        row_only_lt2=bool(report.persistent_sidecar_ledger.get("row_only_lt2")),
        inclusive_lt2=bool(report.persistent_sidecar_ledger.get("inclusive_lt2")),
        shape_breakdown=tuple(
            dict(row)
            for row in report.persistent_sidecar_ledger.get("shape_breakdown", ())
        ),
    )
    validate_hybrid_persistent_sidecar_ledger(ledger)
    if ledger.persistent_mode != report.persistent_mode:
        raise ValueError("overlay persistent_mode must match persistent_sidecar_ledger persistent_mode")
    if bool(report.pass_report) != bool(
        ledger.inclusive_lt2
        and not report.persistent_dense_shadow_present
        and int(report.persistent_dense_shadow_bytes) == 0
        and report.bounded_only_collapse
        and report.local_update_law_reused
        and not report.second_update_law_required
        and report.dense_transient_credit_allowed
        and report.transient_debt_present
        and report.persistent_sub2_hybrid_only
        and not report.full_runtime_sub2_achieved
        and not report.candidate_runtime_complete
    ):
        raise ValueError("hybrid runtime overlay pass flag must be computed from the explicit non-overclaim gates")


def build_strict_sub2_hybrid_runtime_movement_overlay(
    *,
    logical_shapes: Sequence[Sequence[int]],
    event_counts: Sequence[int],
    persistent_mode: str,
    residual_bits_per_event: int,
    persistent_dense_shadow_present: bool,
    persistent_dense_shadow_bytes: int,
    local_update_law_label: str,
    acquisition_science_status: str = ACQUISITION_GATE_UNBLOCKED_NOT_RUN,
    acquisition_achieved: bool = False,
) -> StrictSub2HybridRuntimeMovementOverlay:
    ledger = build_hybrid_persistent_sidecar_ledger(
        logical_shapes=logical_shapes,
        event_counts=event_counts,
        persistent_mode=persistent_mode,
        residual_bits_per_event=residual_bits_per_event,
    )
    report = StrictSub2HybridRuntimeMovementOverlay(
        schema_version=STRICT_SUB2_HYBRID_RUNTIME_MOVEMENT_OVERLAY_SCHEMA_VERSION,
        label=STRICT_SUB2_HYBRID_RUNTIME_MOVEMENT_OVERLAY_LABEL,
        base_scaffold_schema_version=STRICT_SUB2_CANDIDATE_RUNTIME_SCAFFOLD_SCHEMA_VERSION,
        base_scaffold_label=STRICT_SUB2_CANDIDATE_RUNTIME_SCAFFOLD_LABEL,
        runtime_state_authority=RUNTIME_STATE_AUTHORITY_SUB2_PERSISTENT_HYBRID_DENSE_TRANSIENT_CREDIT,
        persistent_mode=persistent_mode,
        pass_report=bool(
            ledger.inclusive_lt2
            and not bool(persistent_dense_shadow_present)
            and int(persistent_dense_shadow_bytes) == 0
        ),
        persistent_dense_shadow_present=bool(persistent_dense_shadow_present),
        persistent_dense_shadow_bytes=int(persistent_dense_shadow_bytes),
        bounded_only_collapse=True,
        local_update_law_label=local_update_law_label,
        local_update_law_reused=True,
        second_update_law_required=False,
        dense_transient_credit_allowed=True,
        dense_transient_credit_role=DENSE_TRANSIENT_CREDIT_ROLE_TRAINING_COMPUTE_CONTROL_ONLY,
        transient_debt_present=True,
        persistent_sub2_hybrid_only=True,
        full_runtime_sub2_achieved=False,
        candidate_runtime_complete=False,
        acquisition_science_status=acquisition_science_status,
        acquisition_achieved=bool(acquisition_achieved),
        movement_contract_scope=HYBRID_MOVEMENT_CONTRACT_SCOPE,
        movement_metric_name=HYBRID_MOVEMENT_METRIC_NAME,
        movement_metric_min_delta=1,
        q_changed_must_be_positive=True,
        hard_fail_required_false=True,
        live_dense_transient_selection_role=DENSE_TRANSIENT_CREDIT_ROLE_TRAINING_COMPUTE_CONTROL_ONLY,
        persistent_authority_row_names=(
            "q_storage",
            "frozen_scales_fp32_metadata",
            "accumulator_sidecar",
        ),
        blocked_row_names=(
            "accumulator_substitute",
            "attention_kv_append_update",
            "qacc_hot_loop_residency",
            "dense_int16_accumulator_control",
            "fp_shell_and_noneligible_fp_controls",
        ),
        persistent_sidecar_ledger=ledger.to_dict(),
        claim_boundary=(
            "persistent-sub2 hybrid movement smoke only",
            "dense transient selection is training-control only",
            "not full-runtime sub-2",
            "not native transient sub-2",
            "not fp-free runtime",
            "not 120/120 acquisition",
        ),
    )
    validate_strict_sub2_hybrid_runtime_movement_overlay(report)
    return report


def validate_strict_sub2_candidate_runtime_scaffold_report(
    report: StrictSub2CandidateRuntimeScaffoldReport,
) -> None:
    if report.schema_version != STRICT_SUB2_CANDIDATE_RUNTIME_SCAFFOLD_SCHEMA_VERSION:
        raise ValueError("strict sub-2 scaffold schema version mismatch")
    if report.label != STRICT_SUB2_CANDIDATE_RUNTIME_SCAFFOLD_LABEL:
        raise ValueError("strict sub-2 scaffold label mismatch")
    if report.target_name != STRICT_SUB2_CANDIDATE_RUNTIME_TARGET_NAME:
        raise ValueError("strict sub-2 scaffold target name mismatch")
    if report.runtime_state_authority not in {
        RUNTIME_STATE_AUTHORITY_DENSE_CONTROL,
        RUNTIME_STATE_AUTHORITY_SUB2_SCAFFOLD_ONLY,
        RUNTIME_STATE_AUTHORITY_SUB2_PERSISTENT_HYBRID_DENSE_TRANSIENT_CREDIT,
        RUNTIME_STATE_AUTHORITY_SUB2_CANDIDATE_EXECUTABLE,
    }:
        raise ValueError("unknown runtime_state_authority")
    if report.runtime_state_authority not in {
        RUNTIME_STATE_AUTHORITY_SUB2_SCAFFOLD_ONLY,
        RUNTIME_STATE_AUTHORITY_SUB2_PERSISTENT_HYBRID_DENSE_TRANSIENT_CREDIT,
    }:
        raise ValueError(
            "this slice must emit runtime_state_authority=sub2_scaffold_only or "
            "sub2_persistent_hybrid_dense_transient_credit"
        )
    hybrid_authority = (
        report.runtime_state_authority
        == RUNTIME_STATE_AUTHORITY_SUB2_PERSISTENT_HYBRID_DENSE_TRANSIENT_CREDIT
    )
    allowed = {LEDGER_CLASS_LEQ2, LEDGER_CLASS_EXECUTABLE, LEDGER_CLASS_NOT_YET}
    for row in _all_rows(report):
        if row.classification not in allowed:
            raise ValueError(f"unknown ledger row classification for {row.name!r}")
        if row.counted_in_physical_persistent_bpw and not row.in_candidate_authority:
            raise ValueError(f"{row.name!r} cannot count toward persistent bpw outside candidate authority")
        if row.section != PERSISTENT_CANDIDATE_SECTION and row.counted_in_physical_persistent_bpw:
            raise ValueError(
                f"{row.name!r} cannot count toward persistent bpw from off-path or adjacent runtime sections"
            )
        if row.in_candidate_authority and row.classification == LEDGER_CLASS_NOT_YET:
            raise ValueError(f"{row.name!r} cannot be in candidate authority while blocked")
        if row.counted_in_physical_persistent_bpw:
            if row.bits_per_weight is None:
                raise ValueError(f"{row.name!r} counted in persistent bpw must disclose bits_per_weight")
            if float(row.bits_per_weight) >= 2.0:
                raise ValueError(f"{row.name!r} candidate-authority row exceeds the strict sub-2 limit")
    recomputed_bpw = sum(
        float(row.bits_per_weight)
        for row in report.persistent_candidate_rows
        if row.counted_in_physical_persistent_bpw
    )
    if abs(recomputed_bpw - float(report.physical_persistent_bpw)) > 1e-12:
        raise ValueError("physical_persistent_bpw must equal the counted candidate-authority subtotal")
    if bool(report.physical_persistent_target_pass) != (float(report.physical_persistent_bpw) < float(report.physical_persistent_target_bpw)):
        raise ValueError("physical_persistent_target_pass must be computed from the binding subtotal")
    if report.candidate_runtime_complete:
        raise ValueError("this scaffold slice must stay non-executable/candidate_runtime_complete=false")
    acquisition_status = report.acquisition_gate.get("status")
    if acquisition_status not in {
        ACQUISITION_GATE_DEFERRED,
        ACQUISITION_GATE_UNBLOCKED_NOT_RUN,
    }:
        raise ValueError("acquisition gate status is unknown")
    adjacent_names = tuple(row.name for row in report.adjacent_runtime_rows)
    if hybrid_authority:
        if not bool(report.persistent_sub2_hybrid_only):
            raise ValueError("hybrid authority must set persistent_sub2_hybrid_only=true")
        if not bool(report.dense_transient_credit_allowed):
            raise ValueError("hybrid authority must set dense_transient_credit_allowed=true")
        if (
            report.dense_transient_credit_role
            != DENSE_TRANSIENT_CREDIT_ROLE_TRAINING_COMPUTE_CONTROL_ONLY
        ):
            raise ValueError("hybrid authority must disclose dense_transient_credit_role=training_compute_control_only")
        if bool(report.dense_transient_credit_counted_in_physical_persistent_bpw):
            raise ValueError("hybrid authority cannot count dense transient credit in physical_persistent_bpw")
        if not bool(report.transient_debt_present):
            raise ValueError("hybrid authority must set transient_debt_present=true")
        if not bool(report.transient_debt_non_blocking):
            raise ValueError("hybrid authority must set transient_debt_non_blocking=true")
        if tuple(report.transient_debt_row_names) != adjacent_names or not adjacent_names:
            raise ValueError("hybrid authority must disclose a non-empty exact transient_debt_row_names list")
        if bool(report.full_runtime_sub2_achieved):
            raise ValueError("hybrid authority cannot claim full_runtime_sub2_achieved")
        if bool(report.native_transient_sub2_achieved):
            raise ValueError("hybrid authority cannot claim native_transient_sub2_achieved")
        if bool(report.fully_fp_free_achieved):
            raise ValueError("hybrid authority cannot claim fully_fp_free_achieved")
        if report.scope_decision_source_msg_id != HYBRID_SCOPE_DECISION_SOURCE_MSG_ID:
            raise ValueError("hybrid authority must disclose the locked scope decision source msg id")
        if report.scope_decision_locked_option != HYBRID_SCOPE_DECISION_LOCKED_OPTION:
            raise ValueError("hybrid authority must disclose the locked scope decision option")
        if report.scope_decision_locked_answer != HYBRID_SCOPE_DECISION_LOCKED_ANSWER:
            raise ValueError("hybrid authority must disclose the locked scope decision answer")
        if report.acquisition_science_status != ACQUISITION_GATE_UNBLOCKED_NOT_RUN:
            raise ValueError("hybrid authority must disclose acquisition_science_status=unblocked_not_run")
        if bool(report.acquisition_achieved):
            raise ValueError("hybrid authority cannot claim acquisition_achieved=true")
        if acquisition_status != ACQUISITION_GATE_UNBLOCKED_NOT_RUN:
            raise ValueError("hybrid authority must leave acquisition gate at unblocked_not_run")
    serialized = str(report.to_dict())
    if "justified_fp_exception" in serialized:
        raise ValueError("justified_fp_exception labels are forbidden in the candidate path")
    if report.hot_loop_residency.get("qacc_kernelized") is not False:
        raise ValueError("this slice must still disclose qacc_kernelized=false")
    hot = report.hot_loop_residency.get("hot_loop_residency", {})
    if hot.get("qacc_update_over_64") != "cpu_reference":
        raise ValueError("this slice must still disclose qacc_update_over_64=cpu_reference")
    if report.hidden_fp_learner_fail_state != HIDDEN_FP_LEARNER_FAIL_STATE:
        raise ValueError("hidden FP learner fail state must be preserved verbatim")
    if not bool(report.pass_report):
        raise ValueError("scaffold report pass flag must reflect a valid fail-closed scaffold, not executable success")
    if report.scoped_candidate_proof is not None:
        proof = dict(report.scoped_candidate_proof)
        if proof.get("surface") != "accumulator_substitute":
            raise ValueError("scoped candidate proof must stay on accumulator_substitute only")
        if proof.get("runtime_state_authority_after") != report.runtime_state_authority:
            raise ValueError("scoped candidate proof must leave runtime_state_authority aligned with the report authority")
        if bool(proof.get("candidate_dense_decode_used")):
            raise ValueError("scoped candidate proof cannot use dense decode on the candidate path")
        if bool(proof.get("candidate_accumulator_transient_over2_used")):
            raise ValueError("scoped candidate proof cannot use >2-bit accumulator transients")
        if bool(proof.get("candidate_vote_transient_over2_used")):
            raise ValueError("scoped candidate proof cannot use dense vote transients")
        if bool(proof.get("candidate_dense_vote_authority_used")):
            raise ValueError("scoped candidate proof cannot use dense vote authority")
        if proof.get("q_storage_physical_budget_covered_by_scoped_proof") is not False:
            raise ValueError("scoped candidate proof must not claim q-storage physical budget coverage")
        if proof.get("frozen_scale_physical_budget_covered_by_scoped_proof") is not False:
            raise ValueError("scoped candidate proof must not claim frozen-scale physical budget coverage")
        if not isinstance(proof.get("coverage_domain"), Mapping):
            raise ValueError("scoped candidate proof must disclose its coverage domain")
        terminal = proof.get("terminal_classification")
        if not bool(proof.get("pass")) and terminal != INTRINSIC_BOUNDED_UPDATE_DOMAIN_GAP:
            raise ValueError("negative scoped candidate proof must land as the intrinsic domain-gap null")
        if terminal not in {
            ACCUMULATOR_SUBSTITUTE_LOCAL_VOTE_UPDATE_EXECUTABLE,
            ALGORITHMIC_LOCAL_VOTE_UPDATE_EXECUTABLE_NOT_PHYSICAL_SUB2,
            INTRINSIC_BOUNDED_UPDATE_DOMAIN_GAP,
        }:
            raise ValueError("scoped candidate proof terminal classification is unknown")
        if bool(proof.get("pass")):
            if not isinstance(proof.get("storage_projection"), Mapping):
                raise ValueError("positive scoped candidate proof must disclose storage_projection")
            accumulator_bpw = float(
                proof["storage_projection"].get("bounded_delta_acc_bits_per_weight")
            )
            if proof.get("scoped_label") == ACCUMULATOR_SUBSTITUTE_LOCAL_VOTE_UPDATE_EXECUTABLE:
                if accumulator_bpw >= 2.0:
                    raise ValueError(
                        "positive scoped candidate proof with the physical local-vote label "
                        "must validate storage_projection bounded_delta_acc_bits_per_weight < 2"
                    )
                if proof.get("scoped_physical_budget_claim") != "physical_sub2_budgeted":
                    raise ValueError(
                        "physical local-vote label must explicitly claim physical_sub2_budgeted"
                    )
            elif proof.get("scoped_label") == ALGORITHMIC_LOCAL_VOTE_UPDATE_EXECUTABLE_NOT_PHYSICAL_SUB2:
                if proof.get("scoped_physical_budget_claim") != "algorithmic_only_not_physical_sub2":
                    raise ValueError(
                        "algorithmic-only local-vote label must explicitly reject physical-sub2 interpretation"
                    )
            else:
                raise ValueError("positive scoped candidate proof uses an unknown positive label")
        persistent_rows = {
            row.name: row
            for row in report.persistent_candidate_rows
        }
        accumulator_row = persistent_rows["accumulator_substitute"]
        if accumulator_row.in_candidate_authority or accumulator_row.classification != LEDGER_CLASS_NOT_YET:
            raise ValueError(
                "scoped candidate proof cannot silently promote the full accumulator row; "
                "it stays blocked/not-yet until broader decision dimensions are covered"
            )


def build_strict_sub2_candidate_runtime_scaffold(
    *,
    eligible_module_shapes: Mapping[str, Sequence[int]],
    activation_paid_bits_ledger: Mapping[str, Any],
    live_both_gate: Mapping[str, Any],
    hot_loop_residency: Mapping[str, Any],
    candidate_name: str = HYBRID_HOT_EXACT_COLD_DEFAULT_CANDIDATE,
) -> StrictSub2CandidateRuntimeScaffoldReport:
    if not eligible_module_shapes:
        raise ValueError("eligible_module_shapes must be non-empty")
    ordered_shapes = tuple(
        _shape_tuple(eligible_module_shapes[name])
        for name in sorted(eligible_module_shapes)
    )
    eligible_weight_count = sum(_numel(shape) for shape in ordered_shapes)
    scale_count = len(ordered_shapes)
    q_storage = base3_q_entropy_ledger_for_shapes(
        regime_name="strict_sub2_candidate_q_storage_shape_ledger",
        logical_shapes=ordered_shapes,
        scale_count=0,
        accumulator_bits_per_weight=0.0,
    )
    q_orthogonality = base3_q_storage_orthogonality_report().to_dict()
    admission_contract = bounded_delta_admission_contract(candidate_name=candidate_name)
    candidate_assessment = bounded_delta_candidate_assessment(candidate_name=candidate_name)
    scale_bpw = _scale_bits_per_weight(scale_count, eligible_weight_count)
    activation_bits = _max_activation_bits_per_value(activation_paid_bits_ledger)
    activation_packable = bool(activation_paid_bits_ledger.get("pass"))
    kv_uncovered = "kv_cache.append_update" in tuple(live_both_gate.get("not_covered", ()))
    hot_loop = dict(hot_loop_residency)
    qacc_kernelized = bool(hot_loop.get("qacc_kernelized"))
    qacc_hot_loop = dict(hot_loop.get("hot_loop_residency", {}))
    qacc_cpu_reference = (
        qacc_hot_loop.get("qacc_update_over_64") == "cpu_reference"
        or qacc_hot_loop.get("qacc_vote_selection") == "cpu_reference"
        or qacc_hot_loop.get("qacc_apply_vote_step") == "cpu_reference"
    )

    persistent_candidate_rows = (
        StrictSub2ScaffoldRow(
            name="q_storage",
            section=PERSISTENT_CANDIDATE_SECTION,
            classification=LEDGER_CLASS_LEQ2,
            in_candidate_authority=True,
            counted_in_physical_persistent_bpw=True,
            bits_per_weight=float(q_storage.q_packed_total_bits_per_weight),
            blocker=False,
            rationale=(
                "Base-3 q storage is the current strict candidate-authority row. "
                "Its ledger is shape-derived and orthogonal to accumulator progress."
            ),
        ),
        StrictSub2ScaffoldRow(
            name="accumulator_substitute",
            section=PERSISTENT_CANDIDATE_SECTION,
            classification=LEDGER_CLASS_NOT_YET,
            in_candidate_authority=False,
            counted_in_physical_persistent_bpw=False,
            bits_per_weight=None,
            blocker=True,
            rationale=(
                "Default slot is the bounded-delta candidate "
                f"{candidate_name!r}, but this slice keeps it scaffold/reference-only: "
                "the adapter/oracle path has a real admission contract and capacity "
                "hypothesis, yet no executable materialize/update/collapse runtime."
            ),
        ),
    )
    off_path_control_rows = (
        StrictSub2ScaffoldRow(
            name="frozen_scales_fp32_metadata",
            section=OFF_PATH_CONTROL_SECTION,
            classification=LEDGER_CLASS_NOT_YET,
            in_candidate_authority=False,
            counted_in_physical_persistent_bpw=False,
            bits_per_weight=float(scale_bpw),
            blocker=True,
            rationale=(
                "Frozen scales are >2 physical bits per component and therefore "
                "must remain off-path controls/blockers until a strict <=2-bit "
                "replacement exists."
            ),
        ),
        StrictSub2ScaffoldRow(
            name="dense_int16_accumulator_control",
            section=OFF_PATH_CONTROL_SECTION,
            classification=LEDGER_CLASS_NOT_YET,
            in_candidate_authority=False,
            counted_in_physical_persistent_bpw=False,
            bits_per_weight=16.0,
            blocker=True,
            rationale="Dense int16 accumulator is the banked control/baseline only, never candidate authority.",
        ),
        StrictSub2ScaffoldRow(
            name="fp_shell_and_noneligible_fp_controls",
            section=OFF_PATH_CONTROL_SECTION,
            classification=LEDGER_CLASS_NOT_YET,
            in_candidate_authority=False,
            counted_in_physical_persistent_bpw=False,
            bits_per_weight=None,
            blocker=True,
            rationale=(
                "FP shell, lm_head/embeddings/norms, and other >2-bit or non-eligible "
                "tensors remain off-path controls only; hidden FP learning stays a hard fail state."
            ),
        ),
    )
    adjacent_runtime_rows = (
        StrictSub2ScaffoldRow(
            name="activations_and_residual_runtime_packability",
            section=ADJACENT_RUNTIME_SECTION,
            classification=LEDGER_CLASS_LEQ2 if activation_packable else LEDGER_CLASS_NOT_YET,
            in_candidate_authority=False,
            counted_in_physical_persistent_bpw=False,
            bits_per_weight=activation_bits,
            blocker=not activation_packable,
            rationale=(
                "Adjacent runtime activation surfaces stay outside persistent authority, "
                "but must be tracked beside it. This row is sourced from the harness "
                "activation paid-bits ledger."
            ),
        ),
        StrictSub2ScaffoldRow(
            name="attention_kv_append_update",
            section=ADJACENT_RUNTIME_SECTION,
            classification=LEDGER_CLASS_NOT_YET if kv_uncovered else LEDGER_CLASS_EXECUTABLE,
            in_candidate_authority=False,
            counted_in_physical_persistent_bpw=False,
            bits_per_weight=None,
            blocker=kv_uncovered,
            rationale=(
                "KV append/update remains explicit uncovered/estimator-only in the current "
                "runtime gate and therefore blocks a complete candidate runtime."
            ),
        ),
        StrictSub2ScaffoldRow(
            name="qacc_hot_loop_residency",
            section=ADJACENT_RUNTIME_SECTION,
            classification=LEDGER_CLASS_NOT_YET if (qacc_cpu_reference or not qacc_kernelized) else LEDGER_CLASS_EXECUTABLE,
            in_candidate_authority=False,
            counted_in_physical_persistent_bpw=False,
            bits_per_weight=None,
            blocker=bool(qacc_cpu_reference or not qacc_kernelized),
            rationale=(
                "The standing speed blocker remains the qacc hot loop: vote selection/apply/"
                "update are still CPU-reference today, so hot-loop GPU residency is not yet in the candidate runtime."
            ),
        ),
    )

    blocker_names = tuple(
        row.name for row in (
            persistent_candidate_rows + off_path_control_rows + adjacent_runtime_rows
        ) if row.blocker
    )
    candidate_authority_row_names = tuple(
        row.name for row in persistent_candidate_rows if row.in_candidate_authority
    )
    physical_persistent_bpw = sum(
        float(row.bits_per_weight)
        for row in persistent_candidate_rows
        if row.counted_in_physical_persistent_bpw
    )
    parity_contract = {
        "candidate_name": candidate_name,
        "candidate_assessment": candidate_assessment.to_dict(),
        "preserved_information": list(admission_contract.preserved_information),
        "capacity_hypothesis": admission_contract.capacity_hypothesis,
        "sub2_persistent_strategy": admission_contract.sub2_persistent_strategy,
        "exact_surfaces": list(admission_contract.exact_surfaces),
        "allowed_divergence_contract": admission_contract.allowed_divergence_contract,
        "dense_baseline_non_regression_required": True,
        "executable_proof_before_acquisition": [
            "physical_persistent_bpw < 2.0",
            "dense int16 accumulator absent from candidate persistent authority",
            "hidden FP learner absent",
            "q storage orthogonality preserved",
            "q_changed identities/counts parity contract present",
            "accepted/deferred/backlog/frontier guard surfaces declared",
        ],
        "adapter_oracle_only": True,
        "acquisition_not_used_as_first_gate": True,
    }
    acquisition_gate = {
        "status": ACQUISITION_GATE_UNBLOCKED_NOT_RUN,
        "support_name": "L0c2-K2-addition-120",
        "reason": (
            "Option A boundary encoded: persistent learner/runtime state is the hard <2 gate, "
            "dense transient credit stays allowed training-compute/control, and the next step is "
            "an acquisition smoke that remains not-yet-run in this slice"
        ),
    }

    transient_debt_row_names = tuple(row.name for row in adjacent_runtime_rows)
    report = StrictSub2CandidateRuntimeScaffoldReport(
        schema_version=STRICT_SUB2_CANDIDATE_RUNTIME_SCAFFOLD_SCHEMA_VERSION,
        label=STRICT_SUB2_CANDIDATE_RUNTIME_SCAFFOLD_LABEL,
        target_name=STRICT_SUB2_CANDIDATE_RUNTIME_TARGET_NAME,
        runtime_state_authority=RUNTIME_STATE_AUTHORITY_SUB2_PERSISTENT_HYBRID_DENSE_TRANSIENT_CREDIT,
        pass_report=True,
        eligible_module_count=len(ordered_shapes),
        eligible_weight_count=eligible_weight_count,
        physical_persistent_bpw=float(physical_persistent_bpw),
        physical_persistent_target_bpw=2.0,
        physical_persistent_target_pass=bool(physical_persistent_bpw < 2.0),
        physical_persistent_interpretation=(
            "binding subtotal covers persistent learner/runtime state only; dense transient "
            "credit and adjacent runtime debt are explicit, non-blocking, and excluded from "
            "the physical_persistent_bpw gate"
        ),
        persistent_sub2_hybrid_only=True,
        dense_transient_credit_allowed=True,
        dense_transient_credit_role=DENSE_TRANSIENT_CREDIT_ROLE_TRAINING_COMPUTE_CONTROL_ONLY,
        dense_transient_credit_counted_in_physical_persistent_bpw=False,
        transient_debt_present=True,
        transient_debt_non_blocking=True,
        transient_debt_row_names=transient_debt_row_names,
        full_runtime_sub2_achieved=False,
        native_transient_sub2_achieved=False,
        fully_fp_free_achieved=False,
        scope_decision_source_msg_id=HYBRID_SCOPE_DECISION_SOURCE_MSG_ID,
        scope_decision_locked_option=HYBRID_SCOPE_DECISION_LOCKED_OPTION,
        scope_decision_locked_answer=HYBRID_SCOPE_DECISION_LOCKED_ANSWER,
        acquisition_science_status=ACQUISITION_GATE_UNBLOCKED_NOT_RUN,
        acquisition_achieved=False,
        candidate_runtime_complete=False,
        candidate_authority_row_names=candidate_authority_row_names,
        blocker_names=blocker_names,
        persistent_candidate_rows=persistent_candidate_rows,
        off_path_control_rows=off_path_control_rows,
        adjacent_runtime_rows=adjacent_runtime_rows,
        q_storage_orthogonality=q_orthogonality,
        parity_contract=parity_contract,
        acquisition_gate=acquisition_gate,
        hot_loop_residency=hot_loop,
        hidden_fp_learner_fail_state=HIDDEN_FP_LEARNER_FAIL_STATE,
        scoped_candidate_proof=None,
        non_claims=(
            "persistent-sub2 hybrid only; no full-runtime sub-2 claim",
            "dense transient credit is allowed training-compute/control and remains non-persistent debt",
            "native transient sub-2 is not achieved in this slice",
            "fully fp-free runtime is not achieved in this slice",
            "no executable bounded-delta authority until a real materialize/update/collapse path exists",
            "acquisition science is unblocked but not yet run in this slice",
        ),
    )
    validate_strict_sub2_candidate_runtime_scaffold_report(report)
    return report


def attach_strict_sub2_scoped_candidate_proof(
    report: StrictSub2CandidateRuntimeScaffoldReport,
    *,
    scoped_candidate_proof: Mapping[str, Any],
) -> StrictSub2CandidateRuntimeScaffoldReport:
    proof = dict(scoped_candidate_proof)
    proof["runtime_state_authority_after"] = report.runtime_state_authority
    updated = replace(
        report,
        scoped_candidate_proof=proof,
    )
    validate_strict_sub2_candidate_runtime_scaffold_report(updated)
    return updated


__all__ = [
    "ACQUISITION_GATE_DEFERRED",
    "ACQUISITION_GATE_RESULT",
    "ACQUISITION_GATE_RUNNING",
    "ACQUISITION_GATE_UNBLOCKED_NOT_RUN",
    "LEDGER_CLASS_EXECUTABLE",
    "LEDGER_CLASS_LEQ2",
    "LEDGER_CLASS_NOT_YET",
    "HYBRID_MOVEMENT_CONTRACT_SCOPE",
    "HYBRID_MOVEMENT_METRIC_NAME",
    "HYBRID_PERSISTENT_MODE_APPLIED_CROSSING_DIRECTION_PLUS_4BIT_RESIDUAL",
    "HYBRID_PERSISTENT_MODE_ZERO_PERSISTENT_ACCUMULATOR",
    "RUNTIME_STATE_AUTHORITY_DENSE_CONTROL",
    "RUNTIME_STATE_AUTHORITY_SUB2_CANDIDATE_EXECUTABLE",
    "RUNTIME_STATE_AUTHORITY_SUB2_PERSISTENT_HYBRID_DENSE_TRANSIENT_CREDIT",
    "RUNTIME_STATE_AUTHORITY_SUB2_SCAFFOLD_ONLY",
    "STRICT_SUB2_CANDIDATE_RUNTIME_SCAFFOLD_LABEL",
    "STRICT_SUB2_CANDIDATE_RUNTIME_SCAFFOLD_SCHEMA_VERSION",
    "STRICT_SUB2_CANDIDATE_RUNTIME_TARGET_NAME",
    "STRICT_SUB2_HYBRID_PERSISTENT_SIDECAR_LEDGER_SCHEMA_VERSION",
    "STRICT_SUB2_HYBRID_RUNTIME_MOVEMENT_OVERLAY_LABEL",
    "STRICT_SUB2_HYBRID_RUNTIME_MOVEMENT_OVERLAY_SCHEMA_VERSION",
    "HybridPersistentSidecarLedger",
    "StrictSub2HybridRuntimeMovementOverlay",
    "StrictSub2CandidateRuntimeScaffoldReport",
    "StrictSub2ScaffoldRow",
    "attach_strict_sub2_scoped_candidate_proof",
    "build_hybrid_persistent_sidecar_ledger",
    "build_strict_sub2_candidate_runtime_scaffold",
    "build_strict_sub2_hybrid_runtime_movement_overlay",
    "validate_strict_sub2_candidate_runtime_scaffold_report",
    "validate_hybrid_persistent_sidecar_ledger",
    "validate_strict_sub2_hybrid_runtime_movement_overlay",
    "DENSE_TRANSIENT_CREDIT_ROLE_TRAINING_COMPUTE_CONTROL_ONLY",
    "HYBRID_SCOPE_DECISION_LOCKED_ANSWER",
    "HYBRID_SCOPE_DECISION_LOCKED_OPTION",
    "HYBRID_SCOPE_DECISION_SOURCE_MSG_ID",
]
