"""CPU design contract for HRM-Text-1.58 native-integer optimizer credit candidate path.

Implements the frozen v3 design packet (BR-3C-D): lane-isolated alloc-guard,
branch classifier, capture transient discriminator, and design receipt validator.
Does NOT authorize row flip, GPU runtime receipt, or real_native_integer_* present.

Forward-carry (future ranking slice — NOT resolved here): integer_sparse_rank_votes.py
casts sparse credit_q31 magnitudes to float32 for grouped_bisect_right ranking
(:144-148). A later real-native-integer-ranking proof must decide sparse-FP rank
math vs strict integer-only ranking mode. This slice documents only; it is not the
dense [O,I] leak and branch-8 lacks real-native/GPU authority.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
from typing import Any, Iterator, Mapping, Sequence
from unittest import mock

import torch

from calm.hrm_text_158.native_full_stack.integer_marginal_attribution import (
    INTEGER_MARGINAL_ATTRIBUTION_PRODUCTION_LAW_ID,
    IntegerMarginalAttributionEvents,
    integer_marginal_attribution_from_captures,
)
from calm.hrm_text_158.native_full_stack.integer_optimizer_credit_path import (
    emit_integer_sparse_vote_events_from_trainer_handle,
)
from calm.hrm_text_158.native_full_stack.optimizer_credit_state import (
    OPTIMIZER_CREDIT_STATE_FP_EXCEPTION_CAVEAT,
    OPTIMIZER_CREDIT_STATE_FAIL_CLOSED_NON_CLAIMS,
)

INTEGER_NATIVE_OPTIMIZER_CREDIT_PATH_DESIGN_SCHEMA_VERSION = (
    "hrm_text_158_integer_native_optimizer_credit_path_design/v3"
)
INTEGER_NATIVE_OPTIMIZER_CREDIT_PATH_DESIGN_TARGET_NAME = (
    "optimizer_credit_state_native_integer_candidate_path_design"
)

EXECUTION_LANE_REFERENCE_ORACLE = "reference_oracle"
EXECUTION_LANE_CANDIDATE = "candidate"
REGISTERED_EXECUTION_LANES = frozenset(
    {EXECUTION_LANE_REFERENCE_ORACLE, EXECUTION_LANE_CANDIDATE}
)

ATTRIBUTION_SUBCONTRACT_MODE_REFERENCE_DENSE_INTERNAL = "reference_dense_internal"
ATTRIBUTION_SUBCONTRACT_MODE_STREAMING_SPARSE = "streaming_sparse"
REGISTERED_ATTRIBUTION_SUBCONTRACT_MODES = frozenset(
    {
        ATTRIBUTION_SUBCONTRACT_MODE_REFERENCE_DENSE_INTERNAL,
        ATTRIBUTION_SUBCONTRACT_MODE_STREAMING_SPARSE,
    }
)

BRANCH_D_MEASUREMENT_INVALID = "BR-D-MEASUREMENT-INVALID"
BRANCH_D_DENSE_LEAK = "BR-D-DENSE-LEAK"
BRANCH_D_HIDDEN_FP_STRUCTURAL = "BR-D-HIDDEN-FP-STRUCTURAL"
BRANCH_D_WIRE_ONLY_REFERENCE_GAP = "BR-D-WIRE-ONLY-REFERENCE-GAP"
BRANCH_D_REPRESENTATION_LIMIT = "BR-D-REPRESENTATION-LIMIT"
BRANCH_D_RANKING_GAP = "BR-D-RANKING-GAP"
BRANCH_D_PARTIAL_COVERAGE = "BR-D-PARTIAL-COVERAGE"
BRANCH_D_INTEGER_VIABLE = "BR-D-INTEGER-VIABLE"

REGISTERED_DESIGN_BRANCH_IDS = frozenset(
    {
        BRANCH_D_MEASUREMENT_INVALID,
        BRANCH_D_DENSE_LEAK,
        BRANCH_D_HIDDEN_FP_STRUCTURAL,
        BRANCH_D_WIRE_ONLY_REFERENCE_GAP,
        BRANCH_D_REPRESENTATION_LIMIT,
        BRANCH_D_RANKING_GAP,
        BRANCH_D_PARTIAL_COVERAGE,
        BRANCH_D_INTEGER_VIABLE,
    }
)

AUDIT_NO_DENSE_WG = "AUDIT-NO-DENSE-WG"
AUDIT_NO_DENSE_CREDIT = "AUDIT-NO-DENSE-CREDIT"
AUDIT_NO_DENSE_VOTES = "AUDIT-NO-DENSE-VOTES"
AUDIT_NO_DENSE_INT_ACCUM = "AUDIT-NO-DENSE-INT-ACCUM"
AUDIT_NO_DENSE_INT_ATTR = "AUDIT-NO-DENSE-INT-ATTR"

CANDIDATE_DENSE_INTEGER_SCRATCH_SURFACES = frozenset(
    {AUDIT_NO_DENSE_INT_ACCUM, AUDIT_NO_DENSE_INT_ATTR}
)
CANDIDATE_DENSE_FP_SURFACES = frozenset(
    {AUDIT_NO_DENSE_WG, AUDIT_NO_DENSE_CREDIT, AUDIT_NO_DENSE_VOTES}
)
CANDIDATE_GUARDED_DENSE_SURFACES = (
    CANDIDATE_DENSE_FP_SURFACES | CANDIDATE_DENSE_INTEGER_SCRATCH_SURFACES
)

STATIC_COMMITTED_SUB2_SURFACE_COUNT = 4
STATIC_COMMITTED_REMAINING_BLOCKERS = (
    "optimizer_credit_state",
    "activations_residuals",
    "attention_kv_attention_buffers",
    "native_kernelized_hot_path",
)
BANKED_LIVE_SUB2_SURFACE_COUNT = 5
BANKED_LIVE_R2A_L_RECEIPT_PREFIX = "9559f16b"
BANKED_LIVE_REMAINING_BLOCKERS = (
    "optimizer_credit_state",
    "attention_kv_attention_buffers",
    "native_kernelized_hot_path",
)

INTEGER_NATIVE_OPTIMIZER_CREDIT_PATH_DESIGN_NON_CLAIMS = (
    *OPTIMIZER_CREDIT_STATE_FAIL_CLOSED_NON_CLAIMS,
    "integer native candidate-path design packet only; not a GPU runtime receipt",
    "does not classify BR-D-INTEGER-VIABLE as a readiness row flip",
    "wire SPARSE_EVENT_SHAPE_ONLY parity does not imply BR-D-INTEGER-VIABLE",
    (
        "transient_fp_debt credit_capture_tensors may remain at hook seam; "
        "draining dense intermediates is the candidate-path goal"
    ),
    (
        "dense int32/int64 [O,I] scratch is reference-oracle lane only until "
        "streaming_sparse candidate attribution exists"
    ),
)

FORBIDDEN_DESIGN_RECEIPT_FIELDS = (
    "br_3c_c_audit_pass_cpu",
    "optimizer_state_eligible_exclusion_proven",
    "optimizer_credit_state_sub2_claim",
    "readiness_row_flip_authorized",
    "ready_to_flip",
    "gpu_runtime_receipt_present",
    "real_native_integer_attribution_present",
    "real_native_integer_credit_ranking_present",
    "persistent_carrier_width_claim",
    "fp_exception_laundering_claim",
)


@dataclass(frozen=True)
class DenseIntegerScratchObservation:
    weight_shape: tuple[int, int]
    int64_accum_observed: bool
    int32_attr_observed: bool

    @property
    def candidate_dense_integer_scratch_observed(self) -> bool:
        return self.int64_accum_observed or self.int32_attr_observed

    @property
    def candidate_dense_integer_scratch_surfaces(self) -> tuple[str, ...]:
        surfaces: list[str] = []
        if self.int64_accum_observed:
            surfaces.append(AUDIT_NO_DENSE_INT_ACCUM)
        if self.int32_attr_observed:
            surfaces.append(AUDIT_NO_DENSE_INT_ATTR)
        return tuple(surfaces)


@dataclass(frozen=True)
class AttributionLaneResult:
    execution_lane: str
    attribution_subcontract_mode: str
    events: IntegerMarginalAttributionEvents | None
    dense_scratch_observation: DenseIntegerScratchObservation
    reference_oracle_run_id: str | None
    candidate_run_id: str | None

    @property
    def candidate_alloc_guard_pass(self) -> bool:
        if self.execution_lane == EXECUTION_LANE_REFERENCE_ORACLE:
            return True
        return not self.dense_scratch_observation.candidate_dense_integer_scratch_observed


@dataclass(frozen=True)
class IntegerNativeOptimizerCreditPathDesignReceipt:
    schema_version: str
    target_name: str
    branch_id: str
    execution_lane: str
    attribution_subcontract_mode: str
    candidate_run_id: str
    reference_oracle_run_id: str | None
    candidate_alloc_guard_pass: bool
    candidate_dense_surfaces_observed: tuple[str, ...]
    candidate_dense_integer_scratch_observed: bool
    candidate_dense_integer_scratch_surfaces: tuple[str, ...]
    capture_transient_discriminator_pass: bool
    capture_retained_fp_tensor_count: int
    capture_stashed_in_closure_or_registry_count: int
    attribution_subcontract_pass: bool
    ranking_subcontract_pass: bool
    comparable_set_complete: bool
    fp_exception_caveat: str
    non_claims: tuple[str, ...]
    br_3c_c_audit_pass_cpu: bool = False
    optimizer_state_eligible_exclusion_proven: bool = False
    optimizer_credit_state_sub2_claim: bool = False
    readiness_row_flip_authorized: bool = False
    ready_to_flip: bool = False
    gpu_runtime_receipt_present: bool = False
    real_native_integer_attribution_present: bool = False
    real_native_integer_credit_ranking_present: bool = False
    persistent_carrier_width_claim: bool = False
    fp_exception_laundering_claim: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "target_name": self.target_name,
            "branch_id": self.branch_id,
            "execution_lane": self.execution_lane,
            "attribution_subcontract_mode": self.attribution_subcontract_mode,
            "candidate_run_id": self.candidate_run_id,
            "reference_oracle_run_id": self.reference_oracle_run_id,
            "candidate_alloc_guard_pass": self.candidate_alloc_guard_pass,
            "candidate_dense_surfaces_observed": list(self.candidate_dense_surfaces_observed),
            "candidate_dense_integer_scratch_observed": (
                self.candidate_dense_integer_scratch_observed
            ),
            "candidate_dense_integer_scratch_surfaces": list(
                self.candidate_dense_integer_scratch_surfaces
            ),
            "capture_transient_discriminator_pass": (
                self.capture_transient_discriminator_pass
            ),
            "capture_retained_fp_tensor_count": self.capture_retained_fp_tensor_count,
            "capture_stashed_in_closure_or_registry_count": (
                self.capture_stashed_in_closure_or_registry_count
            ),
            "attribution_subcontract_pass": self.attribution_subcontract_pass,
            "ranking_subcontract_pass": self.ranking_subcontract_pass,
            "comparable_set_complete": self.comparable_set_complete,
            "fp_exception_caveat": self.fp_exception_caveat,
            "non_claims": list(self.non_claims),
            **{
                field: getattr(self, field)
                for field in FORBIDDEN_DESIGN_RECEIPT_FIELDS
            },
        }


def integer_native_optimizer_credit_path_design_hard_false_snapshot() -> dict[str, bool]:
    return {field: False for field in FORBIDDEN_DESIGN_RECEIPT_FIELDS}


def evaluate_capture_transient_discriminator(
    *,
    capture_retained_fp_tensor_count: int,
    capture_stashed_in_closure_or_registry_count: int,
) -> bool:
    return (
        int(capture_retained_fp_tensor_count) == 0
        and int(capture_stashed_in_closure_or_registry_count) == 0
    )


class _DenseIntegerScratchRecorder:
    def __init__(self, weight_shape: tuple[int, int]) -> None:
        self.weight_shape = weight_shape
        self.int64_accum_observed = False
        self.int32_attr_observed = False

    def record_full_shape_int32_tensor(self, tensor: torch.Tensor) -> None:
        if tuple(int(dim) for dim in tensor.shape) == self.weight_shape:
            if tensor.dtype == torch.int32:
                self.int32_attr_observed = True

    def observation(self) -> DenseIntegerScratchObservation:
        # integer_marginal_attribution_from_captures unconditionally materializes
        # attribution_dense [O,I] int32 from the int64 accumulator (:215-218).
        int32_observed = self.int32_attr_observed or self.int64_accum_observed
        return DenseIntegerScratchObservation(
            weight_shape=self.weight_shape,
            int64_accum_observed=self.int64_accum_observed,
            int32_attr_observed=int32_observed,
        )


def _shape_from_size_args(size_args: tuple[Any, ...]) -> tuple[int, ...]:
    if len(size_args) == 1 and isinstance(size_args[0], (tuple, list)):
        return tuple(int(dim) for dim in size_args[0])
    return tuple(int(dim) for dim in size_args)


@contextmanager
def recording_dense_integer_scratch(
    weight_shape: tuple[int, int],
) -> Iterator[_DenseIntegerScratchRecorder]:
    recorder = _DenseIntegerScratchRecorder(weight_shape)
    expected_shape = weight_shape
    original_zeros = torch.zeros
    original_empty = torch.empty
    original_tensor_to = torch.Tensor.to

    def guarded_zeros(*size, **kwargs):
        dtype = kwargs.get("dtype", torch.get_default_dtype())
        shape = _shape_from_size_args(size)
        if shape == expected_shape:
            if dtype == torch.int64:
                recorder.int64_accum_observed = True
            elif dtype == torch.int32:
                recorder.int32_attr_observed = True
        return original_zeros(*size, **kwargs)

    def guarded_empty(*size, **kwargs):
        dtype = kwargs.get("dtype", torch.get_default_dtype())
        shape = _shape_from_size_args(size)
        if shape == expected_shape and dtype == torch.int32:
            recorder.int32_attr_observed = True
        return original_empty(*size, **kwargs)

    def guarded_tensor_to(self, *args, **kwargs):
        result = original_tensor_to(self, *args, **kwargs)
        dtype = kwargs.get("dtype")
        if dtype is None and args:
            first = args[0]
            if isinstance(first, torch.dtype):
                dtype = first
        recorder.record_full_shape_int32_tensor(result)
        return result

    attribution_module = (
        "calm.hrm_text_158.native_full_stack.integer_marginal_attribution.torch"
    )
    with (
        mock.patch("torch.zeros", side_effect=guarded_zeros),
        mock.patch("torch.empty", side_effect=guarded_empty),
        mock.patch("torch.Tensor.to", guarded_tensor_to),
        mock.patch(f"{attribution_module}.zeros", side_effect=guarded_zeros),
        mock.patch(f"{attribution_module}.empty", side_effect=guarded_empty),
        mock.patch(f"{attribution_module}.Tensor.to", guarded_tensor_to),
    ):
        yield recorder


def run_attribution_with_execution_lane(
    inputs: Sequence[torch.Tensor],
    grad_outputs: Sequence[torch.Tensor],
    *,
    weight_shape: Sequence[int],
    execution_lane: str,
    reference_oracle_run_id: str | None = None,
    candidate_run_id: str | None = None,
    law_id: str = INTEGER_MARGINAL_ATTRIBUTION_PRODUCTION_LAW_ID,
) -> AttributionLaneResult:
    if execution_lane not in REGISTERED_EXECUTION_LANES:
        raise ValueError(f"unsupported execution_lane: {execution_lane!r}")
    weight_dims = tuple(int(dim) for dim in weight_shape)
    with recording_dense_integer_scratch(weight_dims) as recorder:
        events = integer_marginal_attribution_from_captures(
            inputs,
            grad_outputs,
            weight_shape=weight_dims,
            law_id=law_id,
        )
    return AttributionLaneResult(
        execution_lane=execution_lane,
        attribution_subcontract_mode=ATTRIBUTION_SUBCONTRACT_MODE_REFERENCE_DENSE_INTERNAL,
        events=events,
        dense_scratch_observation=recorder.observation(),
        reference_oracle_run_id=reference_oracle_run_id,
        candidate_run_id=candidate_run_id,
    )


def observe_current_wire_candidate_dense_integer_scratch(
    handle: Any,
    states: Mapping[str, Any],
    rank_spec: Any,
) -> DenseIntegerScratchObservation:
    """Observe dense integer scratch when the Option-A wire runs on candidate lane."""

    state_key = next(iter(sorted(states.keys())))
    state = states[state_key]
    weight_shape = tuple(int(dim) for dim in state.q_levels.shape)
    with recording_dense_integer_scratch(weight_shape) as recorder:
        emit_integer_sparse_vote_events_from_trainer_handle(
            handle,
            states,
            rank_spec,
        )
    return recorder.observation()


def classify_integer_native_optimizer_credit_path_branch(
    *,
    candidate_alloc_guard_pass: bool,
    candidate_dense_surfaces_observed: Sequence[str],
    candidate_dense_integer_scratch_observed: bool,
    capture_transient_discriminator_pass: bool,
    attribution_subcontract_mode: str,
    attribution_subcontract_pass: bool,
    ranking_subcontract_pass: bool,
    comparable_set_complete: bool,
    measurement_complete: bool = True,
    partial_coverage_only: bool = False,
    wire_shape_only_pass: bool = False,
) -> str:
    if not measurement_complete:
        return BRANCH_D_MEASUREMENT_INVALID
    dense_surfaces = tuple(str(surface) for surface in candidate_dense_surfaces_observed)
    if (
        not candidate_alloc_guard_pass
        or dense_surfaces
        or candidate_dense_integer_scratch_observed
    ):
        return BRANCH_D_DENSE_LEAK
    if not capture_transient_discriminator_pass:
        return BRANCH_D_HIDDEN_FP_STRUCTURAL
    if (
        wire_shape_only_pass
        or attribution_subcontract_mode == ATTRIBUTION_SUBCONTRACT_MODE_REFERENCE_DENSE_INTERNAL
    ):
        return BRANCH_D_WIRE_ONLY_REFERENCE_GAP
    if not attribution_subcontract_pass:
        return BRANCH_D_REPRESENTATION_LIMIT
    if not ranking_subcontract_pass:
        return BRANCH_D_RANKING_GAP
    if partial_coverage_only:
        return BRANCH_D_PARTIAL_COVERAGE
    if (
        attribution_subcontract_mode == ATTRIBUTION_SUBCONTRACT_MODE_STREAMING_SPARSE
        and candidate_alloc_guard_pass
        and not candidate_dense_integer_scratch_observed
        and attribution_subcontract_pass
        and ranking_subcontract_pass
        and capture_transient_discriminator_pass
        and comparable_set_complete
    ):
        return BRANCH_D_INTEGER_VIABLE
    return BRANCH_D_WIRE_ONLY_REFERENCE_GAP


def build_integer_native_optimizer_credit_path_design_receipt(
    *,
    execution_lane: str,
    attribution_subcontract_mode: str,
    candidate_run_id: str,
    reference_oracle_run_id: str | None,
    candidate_alloc_guard_pass: bool,
    candidate_dense_surfaces_observed: Sequence[str] = (),
    candidate_dense_integer_scratch_observed: bool = False,
    candidate_dense_integer_scratch_surfaces: Sequence[str] = (),
    capture_retained_fp_tensor_count: int = 0,
    capture_stashed_in_closure_or_registry_count: int = 0,
    attribution_subcontract_pass: bool = False,
    ranking_subcontract_pass: bool = False,
    comparable_set_complete: bool = False,
    measurement_complete: bool = True,
    partial_coverage_only: bool = False,
    wire_shape_only_pass: bool = False,
) -> IntegerNativeOptimizerCreditPathDesignReceipt:
    capture_transient_discriminator_pass = evaluate_capture_transient_discriminator(
        capture_retained_fp_tensor_count=capture_retained_fp_tensor_count,
        capture_stashed_in_closure_or_registry_count=(
            capture_stashed_in_closure_or_registry_count
        ),
    )
    dense_surfaces = tuple(str(surface) for surface in candidate_dense_surfaces_observed)
    int_surfaces = tuple(
        str(surface) for surface in candidate_dense_integer_scratch_surfaces
    )
    branch_id = classify_integer_native_optimizer_credit_path_branch(
        candidate_alloc_guard_pass=candidate_alloc_guard_pass,
        candidate_dense_surfaces_observed=dense_surfaces,
        candidate_dense_integer_scratch_observed=candidate_dense_integer_scratch_observed,
        capture_transient_discriminator_pass=capture_transient_discriminator_pass,
        attribution_subcontract_mode=attribution_subcontract_mode,
        attribution_subcontract_pass=attribution_subcontract_pass,
        ranking_subcontract_pass=ranking_subcontract_pass,
        comparable_set_complete=comparable_set_complete,
        measurement_complete=measurement_complete,
        partial_coverage_only=partial_coverage_only,
        wire_shape_only_pass=wire_shape_only_pass,
    )
    receipt = IntegerNativeOptimizerCreditPathDesignReceipt(
        schema_version=INTEGER_NATIVE_OPTIMIZER_CREDIT_PATH_DESIGN_SCHEMA_VERSION,
        target_name=INTEGER_NATIVE_OPTIMIZER_CREDIT_PATH_DESIGN_TARGET_NAME,
        branch_id=branch_id,
        execution_lane=execution_lane,
        attribution_subcontract_mode=attribution_subcontract_mode,
        candidate_run_id=candidate_run_id,
        reference_oracle_run_id=reference_oracle_run_id,
        candidate_alloc_guard_pass=candidate_alloc_guard_pass,
        candidate_dense_surfaces_observed=dense_surfaces,
        candidate_dense_integer_scratch_observed=candidate_dense_integer_scratch_observed,
        candidate_dense_integer_scratch_surfaces=int_surfaces,
        capture_transient_discriminator_pass=capture_transient_discriminator_pass,
        capture_retained_fp_tensor_count=int(capture_retained_fp_tensor_count),
        capture_stashed_in_closure_or_registry_count=int(
            capture_stashed_in_closure_or_registry_count
        ),
        attribution_subcontract_pass=attribution_subcontract_pass,
        ranking_subcontract_pass=ranking_subcontract_pass,
        comparable_set_complete=comparable_set_complete,
        fp_exception_caveat=OPTIMIZER_CREDIT_STATE_FP_EXCEPTION_CAVEAT,
        non_claims=INTEGER_NATIVE_OPTIMIZER_CREDIT_PATH_DESIGN_NON_CLAIMS,
    )
    validate_integer_native_optimizer_credit_path_design_receipt(receipt)
    return receipt


def build_design_receipt_from_attribution_lane_result(
    lane_result: AttributionLaneResult,
    *,
    candidate_run_id: str,
    capture_retained_fp_tensor_count: int = 0,
    capture_stashed_in_closure_or_registry_count: int = 0,
    attribution_subcontract_pass: bool | None = None,
    ranking_subcontract_pass: bool = False,
    comparable_set_complete: bool = False,
    wire_shape_only_pass: bool = False,
) -> IntegerNativeOptimizerCreditPathDesignReceipt:
    observation = lane_result.dense_scratch_observation
    on_candidate_lane = lane_result.execution_lane == EXECUTION_LANE_CANDIDATE
    int_scratch_observed = (
        observation.candidate_dense_integer_scratch_observed if on_candidate_lane else False
    )
    int_scratch_surfaces = (
        observation.candidate_dense_integer_scratch_surfaces if on_candidate_lane else ()
    )
    if attribution_subcontract_pass is None:
        attribution_subcontract_pass = lane_result.events is not None
    return build_integer_native_optimizer_credit_path_design_receipt(
        execution_lane=lane_result.execution_lane,
        attribution_subcontract_mode=lane_result.attribution_subcontract_mode,
        candidate_run_id=candidate_run_id,
        reference_oracle_run_id=lane_result.reference_oracle_run_id,
        candidate_alloc_guard_pass=lane_result.candidate_alloc_guard_pass,
        candidate_dense_integer_scratch_observed=int_scratch_observed,
        candidate_dense_integer_scratch_surfaces=int_scratch_surfaces,
        capture_retained_fp_tensor_count=capture_retained_fp_tensor_count,
        capture_stashed_in_closure_or_registry_count=(
            capture_stashed_in_closure_or_registry_count
        ),
        attribution_subcontract_pass=attribution_subcontract_pass,
        ranking_subcontract_pass=ranking_subcontract_pass,
        comparable_set_complete=comparable_set_complete,
        wire_shape_only_pass=wire_shape_only_pass,
    )


def validate_integer_native_optimizer_credit_path_design_receipt(
    receipt: IntegerNativeOptimizerCreditPathDesignReceipt,
) -> None:
    if receipt.schema_version != INTEGER_NATIVE_OPTIMIZER_CREDIT_PATH_DESIGN_SCHEMA_VERSION:
        raise ValueError("integer native optimizer credit path design schema mismatch")
    if receipt.target_name != INTEGER_NATIVE_OPTIMIZER_CREDIT_PATH_DESIGN_TARGET_NAME:
        raise ValueError("integer native optimizer credit path design target mismatch")
    if receipt.branch_id not in REGISTERED_DESIGN_BRANCH_IDS:
        raise ValueError("integer native optimizer credit path design branch unknown")
    if receipt.execution_lane not in REGISTERED_EXECUTION_LANES:
        raise ValueError("integer native optimizer credit path execution lane unknown")
    if receipt.attribution_subcontract_mode not in REGISTERED_ATTRIBUTION_SUBCONTRACT_MODES:
        raise ValueError(
            "integer native optimizer credit path attribution subcontract mode unknown"
        )
    if receipt.fp_exception_caveat != OPTIMIZER_CREDIT_STATE_FP_EXCEPTION_CAVEAT:
        raise ValueError(
            "integer native optimizer credit path must keep credit_capture_tensors "
            "attribution-only"
        )
    if receipt.non_claims != INTEGER_NATIVE_OPTIMIZER_CREDIT_PATH_DESIGN_NON_CLAIMS:
        raise ValueError(
            "integer native optimizer credit path design non-claims must be exact"
        )
    for field in FORBIDDEN_DESIGN_RECEIPT_FIELDS:
        if bool(getattr(receipt, field)):
            raise ValueError(f"{field} is forbidden on design receipt")

    int_surfaces = tuple(receipt.candidate_dense_integer_scratch_surfaces)
    if receipt.candidate_dense_integer_scratch_observed != bool(int_surfaces):
        raise ValueError(
            "candidate_dense_integer_scratch_observed must match whether "
            "candidate_dense_integer_scratch_surfaces is non-empty"
        )
    unknown_int_surfaces = [
        surface
        for surface in int_surfaces
        if surface not in CANDIDATE_DENSE_INTEGER_SCRATCH_SURFACES
    ]
    if unknown_int_surfaces:
        raise ValueError(
            "candidate_dense_integer_scratch_surfaces must be a subset of "
            f"{sorted(CANDIDATE_DENSE_INTEGER_SCRATCH_SURFACES)!r}"
        )
    if receipt.candidate_dense_surfaces_observed and receipt.candidate_alloc_guard_pass:
        raise ValueError(
            "candidate_alloc_guard_pass cannot be true when "
            "candidate_dense_surfaces_observed is non-empty"
        )
    if int_surfaces and receipt.candidate_alloc_guard_pass:
        raise ValueError(
            "candidate_alloc_guard_pass cannot be true when "
            "candidate_dense_integer_scratch_surfaces is non-empty"
        )
    if (
        receipt.candidate_dense_surfaces_observed
        or int_surfaces
        or receipt.candidate_dense_integer_scratch_observed
    ) and receipt.branch_id != BRANCH_D_DENSE_LEAK:
        raise ValueError(
            "non-empty dense surface evidence requires branch_id=BR-D-DENSE-LEAK"
        )

    if receipt.branch_id == BRANCH_D_INTEGER_VIABLE:
        if receipt.attribution_subcontract_mode != ATTRIBUTION_SUBCONTRACT_MODE_STREAMING_SPARSE:
            raise ValueError(
                "BR-D-INTEGER-VIABLE requires attribution_subcontract_mode=streaming_sparse"
            )
        if receipt.candidate_dense_integer_scratch_observed:
            raise ValueError(
                "BR-D-INTEGER-VIABLE requires candidate_dense_integer_scratch_observed=false"
            )
        if receipt.candidate_dense_surfaces_observed:
            raise ValueError(
                "BR-D-INTEGER-VIABLE requires candidate_dense_surfaces_observed=()"
            )
        if receipt.candidate_dense_integer_scratch_surfaces:
            raise ValueError(
                "BR-D-INTEGER-VIABLE requires candidate_dense_integer_scratch_surfaces=()"
            )
        if not receipt.comparable_set_complete:
            raise ValueError("BR-D-INTEGER-VIABLE requires comparable_set_complete=true")
        if not (
            receipt.candidate_alloc_guard_pass
            and receipt.capture_transient_discriminator_pass
            and receipt.attribution_subcontract_pass
            and receipt.ranking_subcontract_pass
        ):
            raise ValueError(
                "BR-D-INTEGER-VIABLE requires alloc-guard, capture discriminator, "
                "and both subcontracts to pass"
            )

    if receipt.candidate_dense_integer_scratch_observed and receipt.candidate_alloc_guard_pass:
        raise ValueError(
            "candidate_alloc_guard_pass cannot be true when dense integer scratch observed"
        )


# --- BR-3C-E streaming-sparse attribution subcontract (Step-0 observer + receipt) ---

from calm.hrm_text_158.native_full_stack.integer_marginal_attribution import (  # noqa: E402
    StreamingSparseAttributionMetrics,
    streaming_sparse_attribution_from_captures,
)
from torch.utils._python_dispatch import TorchDispatchMode  # noqa: E402

STREAMING_SPARSE_ATTRIBUTION_SUBCONTRACT_SCHEMA_VERSION = (
    "hrm_text_158_streaming_sparse_attribution_subcontract/v1"
)
STREAMING_SPARSE_ATTRIBUTION_SUBCONTRACT_TARGET_NAME = (
    "optimizer_credit_state_streaming_sparse_attribution_subcontract"
)

STREAMING_SPARSE_ATTRIBUTION_SUBCONTRACT_NON_CLAIMS = (
    *OPTIMIZER_CREDIT_STATE_FAIL_CLOSED_NON_CLAIMS,
    "attribution subcontract receipt only; ranking subcontract NOT evaluated here",
    (
        "strict-integer-only ranking is a separate deferred slice, REQUIRED before "
        "any real_native_integer_credit_ranking / BR-D-INTEGER-VIABLE / GPU-flip claim"
    ),
    "does NOT invoke branch-8 classifier or claim BR-D-INTEGER-VIABLE",
    "does NOT prove GPU memory relief or subquadratic compute",
    "does NOT flip optimizer_credit_state sub2 row or set readiness flags",
    "1-D event carrier approaching O*I bytes is a RECORDED caveat, NOT a green",
    "CPU win = no 2-D [O,I] materialization + exact full-support parity only",
    "tile-peak reduction is separate from event-carrier density; dense carrier is not a miss",
)

STREAMING_SPARSE_FULL_DENSE_BASELINE_ITEMSIZE = 8
STREAMING_SPARSE_EVENT_CARRIER_INDEX_BYTES = 8
STREAMING_SPARSE_EVENT_CARRIER_ATTR_BYTES = 4
STREAMING_SPARSE_EVENT_CARRIER_BYTES_PER_EVENT = (
    STREAMING_SPARSE_EVENT_CARRIER_INDEX_BYTES + STREAMING_SPARSE_EVENT_CARRIER_ATTR_BYTES
)
STREAMING_SPARSE_DENSITY_RATIO_TOLERANCE = 1e-9
STREAMING_SPARSE_INT_TILE_ITEMSIZE_OPTIONS = (4, 8)

FORBIDDEN_STREAMING_SPARSE_SUBCONTRACT_FIELDS = (
    "ready_to_flip",
    "optimizer_credit_state_sub2_claim",
    "readiness_row_flip_authorized",
    "real_native_integer_attribution_present",
    "real_native_integer_credit_ranking_present",
    "gpu_runtime_receipt_present",
    "fp_exception_laundering_claim",
    "branch_d_integer_viable_claimed",
    "optimizer_state_eligible_exclusion_proven",
    "br_3c_c_audit_pass_cpu",
)


@dataclass(frozen=True)
class CandidateDenseIntegerDispatchObservation:
    weight_shape: tuple[int, int]
    full_dense_numel: int
    int64_accum_observed: bool
    int32_attr_observed: bool
    max_candidate_tile_shape: tuple[int, ...]
    max_candidate_tile_numel: int
    max_candidate_tile_bytes: int

    @property
    def candidate_dense_integer_scratch_observed(self) -> bool:
        return self.int64_accum_observed or self.int32_attr_observed

    @property
    def candidate_dense_integer_scratch_surfaces(self) -> tuple[str, ...]:
        surfaces: list[str] = []
        if self.int64_accum_observed:
            surfaces.append(AUDIT_NO_DENSE_INT_ACCUM)
        if self.int32_attr_observed:
            surfaces.append(AUDIT_NO_DENSE_INT_ATTR)
        return tuple(surfaces)


class CandidateDenseIntegerDispatchObserver(TorchDispatchMode):
    """Inspect all ATen op outputs; flag full-size 2-D int32/int64 tensors (FOLD-1/2)."""

    def __init__(self, weight_shape: tuple[int, int]) -> None:
        super().__init__()
        self.weight_shape = tuple(int(dim) for dim in weight_shape)
        self.full_dense_numel = int(self.weight_shape[0] * self.weight_shape[1])
        self.int64_accum_observed = False
        self.int32_attr_observed = False
        self.max_candidate_tile_shape: tuple[int, ...] = (0,)
        self.max_candidate_tile_numel = 0
        self.max_candidate_tile_bytes = 0

    def _is_full_dense_integer_leak(self, tensor: torch.Tensor) -> bool:
        if tensor.dtype not in (torch.int32, torch.int64):
            return False
        if int(tensor.ndim) != 2:
            return False
        return int(tensor.numel()) == self.full_dense_numel

    def _record_tensor(self, tensor: torch.Tensor) -> None:
        if not isinstance(tensor, torch.Tensor):
            return
        if self._is_full_dense_integer_leak(tensor):
            if tensor.dtype == torch.int64:
                self.int64_accum_observed = True
            elif tensor.dtype == torch.int32:
                self.int32_attr_observed = True
            return
        numel = int(tensor.numel())
        if numel <= 0:
            return
        if tensor.dtype in (torch.int32, torch.int64):
            shape = tuple(int(dim) for dim in tensor.shape)
            bytes_ = numel * int(tensor.element_size())
            if numel > self.max_candidate_tile_numel:
                self.max_candidate_tile_numel = numel
                self.max_candidate_tile_shape = shape
                self.max_candidate_tile_bytes = bytes_

    def _inspect_nested(self, obj: Any) -> None:
        if isinstance(obj, torch.Tensor):
            self._record_tensor(obj)
        elif isinstance(obj, (tuple, list)):
            for item in obj:
                self._inspect_nested(item)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):  # type: ignore[no-untyped-def]
        kwargs = kwargs or {}
        result = func(*args, **kwargs)
        self._inspect_nested(result)
        return result

    def observation(self) -> CandidateDenseIntegerDispatchObservation:
        return CandidateDenseIntegerDispatchObservation(
            weight_shape=self.weight_shape,
            full_dense_numel=self.full_dense_numel,
            int64_accum_observed=self.int64_accum_observed,
            int32_attr_observed=self.int32_attr_observed,
            max_candidate_tile_shape=self.max_candidate_tile_shape,
            max_candidate_tile_numel=self.max_candidate_tile_numel,
            max_candidate_tile_bytes=self.max_candidate_tile_bytes,
        )


@contextmanager
def candidate_dense_integer_dispatch_observation(
    weight_shape: tuple[int, int],
) -> Iterator[CandidateDenseIntegerDispatchObserver]:
    observer = CandidateDenseIntegerDispatchObserver(weight_shape)
    with observer:
        yield observer


@dataclass(frozen=True)
class StreamingSparseAttributionSubcontractReceipt:
    schema_version: str
    target_name: str
    attribution_subcontract_mode: str
    max_candidate_tile_shape: tuple[int, ...]
    max_candidate_tile_numel: int
    max_candidate_tile_bytes: int
    full_dense_shape: tuple[int, int]
    full_dense_numel: int
    full_dense_baseline_bytes: int
    candidate_event_count: int
    candidate_event_carrier_peak_bytes: int
    event_carrier_density_ratio: float
    candidate_dense_integer_scratch_observed: bool
    candidate_dense_integer_scratch_surfaces: tuple[str, ...]
    full_support_parity_pass: bool
    comparable_set_id: str
    reference_oracle_run_id: str
    candidate_run_id: str
    fp_exception_caveat: str
    non_claims: tuple[str, ...]
    ready_to_flip: bool = False
    optimizer_credit_state_sub2_claim: bool = False
    readiness_row_flip_authorized: bool = False
    real_native_integer_attribution_present: bool = False
    real_native_integer_credit_ranking_present: bool = False
    gpu_runtime_receipt_present: bool = False
    fp_exception_laundering_claim: bool = False
    branch_d_integer_viable_claimed: bool = False
    optimizer_state_eligible_exclusion_proven: bool = False
    br_3c_c_audit_pass_cpu: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "target_name": self.target_name,
            "attribution_subcontract_mode": self.attribution_subcontract_mode,
            "max_candidate_tile_shape": list(self.max_candidate_tile_shape),
            "max_candidate_tile_numel": self.max_candidate_tile_numel,
            "max_candidate_tile_bytes": self.max_candidate_tile_bytes,
            "full_dense_shape": list(self.full_dense_shape),
            "full_dense_numel": self.full_dense_numel,
            "full_dense_baseline_bytes": self.full_dense_baseline_bytes,
            "candidate_event_count": self.candidate_event_count,
            "candidate_event_carrier_peak_bytes": self.candidate_event_carrier_peak_bytes,
            "event_carrier_density_ratio": self.event_carrier_density_ratio,
            "candidate_dense_integer_scratch_observed": (
                self.candidate_dense_integer_scratch_observed
            ),
            "candidate_dense_integer_scratch_surfaces": list(
                self.candidate_dense_integer_scratch_surfaces
            ),
            "full_support_parity_pass": self.full_support_parity_pass,
            "comparable_set_id": self.comparable_set_id,
            "reference_oracle_run_id": self.reference_oracle_run_id,
            "candidate_run_id": self.candidate_run_id,
            "fp_exception_caveat": self.fp_exception_caveat,
            "non_claims": list(self.non_claims),
            **{
                field: getattr(self, field)
                for field in FORBIDDEN_STREAMING_SPARSE_SUBCONTRACT_FIELDS
            },
        }


def streaming_sparse_attribution_subcontract_hard_false_snapshot() -> dict[str, bool]:
    return {field: False for field in FORBIDDEN_STREAMING_SPARSE_SUBCONTRACT_FIELDS}


def _validate_dispatch_observation_matches_metrics(
    metrics: StreamingSparseAttributionMetrics,
    dispatch_observation: CandidateDenseIntegerDispatchObservation,
) -> None:
    if dispatch_observation.weight_shape != metrics.full_dense_shape:
        raise ValueError(
            "dispatch_observation.weight_shape must match metrics.full_dense_shape"
        )
    if dispatch_observation.full_dense_numel != metrics.full_dense_numel:
        raise ValueError(
            "dispatch_observation.full_dense_numel must match metrics.full_dense_numel"
        )


def _validate_streaming_sparse_attribution_metric_invariants(
    *,
    max_candidate_tile_shape: tuple[int, ...],
    max_candidate_tile_numel: int,
    max_candidate_tile_bytes: int,
    full_dense_shape: tuple[int, int],
    full_dense_numel: int,
    full_dense_baseline_bytes: int,
    candidate_event_count: int,
    candidate_event_carrier_peak_bytes: int,
    event_carrier_density_ratio: float,
) -> None:
    if full_dense_numel <= 0:
        raise ValueError("full_dense_numel must be > 0")
    if math.prod(full_dense_shape) != full_dense_numel:
        raise ValueError("prod(full_dense_shape) must equal full_dense_numel")
    expected_baseline = full_dense_numel * STREAMING_SPARSE_FULL_DENSE_BASELINE_ITEMSIZE
    if full_dense_baseline_bytes != expected_baseline:
        raise ValueError(
            "full_dense_baseline_bytes must equal full_dense_numel * baseline itemsize"
        )
    if candidate_event_count < 0 or candidate_event_count > full_dense_numel:
        raise ValueError("candidate_event_count must be in [0, full_dense_numel]")
    expected_carrier = (
        candidate_event_count * STREAMING_SPARSE_EVENT_CARRIER_BYTES_PER_EVENT
    )
    if candidate_event_carrier_peak_bytes != expected_carrier:
        raise ValueError(
            "candidate_event_carrier_peak_bytes must equal event_count * carrier bytes"
        )
    expected_density = candidate_event_count / float(full_dense_numel)
    if (
        abs(event_carrier_density_ratio - expected_density)
        > STREAMING_SPARSE_DENSITY_RATIO_TOLERANCE
    ):
        raise ValueError(
            "event_carrier_density_ratio must equal event_count / full_dense_numel"
        )
    if max_candidate_tile_numel < 0:
        raise ValueError("max_candidate_tile_numel must be >= 0")
    if max_candidate_tile_numel == 0:
        if max_candidate_tile_bytes != 0:
            raise ValueError(
                "max_candidate_tile_bytes must be 0 when max_candidate_tile_numel is 0"
            )
        return
    if math.prod(max_candidate_tile_shape) != max_candidate_tile_numel:
        raise ValueError(
            "prod(max_candidate_tile_shape) must equal max_candidate_tile_numel"
        )
    if max_candidate_tile_bytes % max_candidate_tile_numel != 0:
        raise ValueError(
            "max_candidate_tile_bytes must be divisible by max_candidate_tile_numel"
        )
    tile_itemsize = max_candidate_tile_bytes // max_candidate_tile_numel
    if tile_itemsize not in STREAMING_SPARSE_INT_TILE_ITEMSIZE_OPTIONS:
        raise ValueError(
            "max_candidate_tile_bytes must equal max_candidate_tile_numel * itemsize"
        )


def events_bit_identical(
    oracle: IntegerMarginalAttributionEvents,
    candidate: IntegerMarginalAttributionEvents,
) -> bool:
    return (
        oracle.law_id == candidate.law_id
        and oracle.numel == candidate.numel
        and oracle.flat_indices.equal(candidate.flat_indices)
        and oracle.attribution_q31.equal(candidate.attribution_q31)
    )


def prove_streaming_sparse_attribution_subcontract(
    inputs: Sequence[torch.Tensor],
    grad_outputs: Sequence[torch.Tensor],
    *,
    weight_shape: Sequence[int],
    comparable_set_id: str,
    reference_oracle_run_id: str,
    candidate_run_id: str,
    law_id: str = INTEGER_MARGINAL_ATTRIBUTION_PRODUCTION_LAW_ID,
) -> StreamingSparseAttributionSubcontractReceipt:
    weight_dims = tuple(int(dim) for dim in weight_shape)
    oracle = integer_marginal_attribution_from_captures(
        inputs,
        grad_outputs,
        weight_shape=weight_dims,
        law_id=law_id,
    )
    with candidate_dense_integer_dispatch_observation(weight_dims) as observer:
        candidate, metrics = streaming_sparse_attribution_from_captures(
            inputs,
            grad_outputs,
            weight_shape=weight_dims,
            law_id=law_id,
        )
    dispatch_obs = observer.observation()
    parity_pass = events_bit_identical(oracle, candidate)
    return build_streaming_sparse_attribution_subcontract_receipt(
        metrics=metrics,
        dispatch_observation=dispatch_obs,
        full_support_parity_pass=parity_pass,
        comparable_set_id=comparable_set_id,
        reference_oracle_run_id=reference_oracle_run_id,
        candidate_run_id=candidate_run_id,
    )


def build_streaming_sparse_attribution_subcontract_receipt(
    *,
    metrics: StreamingSparseAttributionMetrics,
    dispatch_observation: CandidateDenseIntegerDispatchObservation,
    full_support_parity_pass: bool,
    comparable_set_id: str,
    reference_oracle_run_id: str,
    candidate_run_id: str,
) -> StreamingSparseAttributionSubcontractReceipt:
    _validate_dispatch_observation_matches_metrics(metrics, dispatch_observation)
    _validate_streaming_sparse_attribution_metric_invariants(
        max_candidate_tile_shape=metrics.max_candidate_tile_shape,
        max_candidate_tile_numel=metrics.max_candidate_tile_numel,
        max_candidate_tile_bytes=metrics.max_candidate_tile_bytes,
        full_dense_shape=metrics.full_dense_shape,
        full_dense_numel=metrics.full_dense_numel,
        full_dense_baseline_bytes=metrics.full_dense_baseline_bytes,
        candidate_event_count=metrics.candidate_event_count,
        candidate_event_carrier_peak_bytes=metrics.candidate_event_carrier_peak_bytes,
        event_carrier_density_ratio=metrics.event_carrier_density_ratio,
    )
    receipt = StreamingSparseAttributionSubcontractReceipt(
        schema_version=STREAMING_SPARSE_ATTRIBUTION_SUBCONTRACT_SCHEMA_VERSION,
        target_name=STREAMING_SPARSE_ATTRIBUTION_SUBCONTRACT_TARGET_NAME,
        attribution_subcontract_mode=ATTRIBUTION_SUBCONTRACT_MODE_STREAMING_SPARSE,
        max_candidate_tile_shape=metrics.max_candidate_tile_shape,
        max_candidate_tile_numel=metrics.max_candidate_tile_numel,
        max_candidate_tile_bytes=metrics.max_candidate_tile_bytes,
        full_dense_shape=metrics.full_dense_shape,
        full_dense_numel=metrics.full_dense_numel,
        full_dense_baseline_bytes=metrics.full_dense_baseline_bytes,
        candidate_event_count=metrics.candidate_event_count,
        candidate_event_carrier_peak_bytes=metrics.candidate_event_carrier_peak_bytes,
        event_carrier_density_ratio=metrics.event_carrier_density_ratio,
        candidate_dense_integer_scratch_observed=(
            dispatch_observation.candidate_dense_integer_scratch_observed
        ),
        candidate_dense_integer_scratch_surfaces=(
            dispatch_observation.candidate_dense_integer_scratch_surfaces
        ),
        full_support_parity_pass=full_support_parity_pass,
        comparable_set_id=comparable_set_id,
        reference_oracle_run_id=reference_oracle_run_id,
        candidate_run_id=candidate_run_id,
        fp_exception_caveat=OPTIMIZER_CREDIT_STATE_FP_EXCEPTION_CAVEAT,
        non_claims=STREAMING_SPARSE_ATTRIBUTION_SUBCONTRACT_NON_CLAIMS,
    )
    validate_streaming_sparse_attribution_subcontract_receipt(receipt)
    return receipt


def validate_streaming_sparse_attribution_subcontract_receipt(
    receipt: StreamingSparseAttributionSubcontractReceipt,
) -> None:
    if receipt.schema_version != STREAMING_SPARSE_ATTRIBUTION_SUBCONTRACT_SCHEMA_VERSION:
        raise ValueError("streaming sparse attribution subcontract schema mismatch")
    if receipt.target_name != STREAMING_SPARSE_ATTRIBUTION_SUBCONTRACT_TARGET_NAME:
        raise ValueError("streaming sparse attribution subcontract target mismatch")
    if receipt.attribution_subcontract_mode != ATTRIBUTION_SUBCONTRACT_MODE_STREAMING_SPARSE:
        raise ValueError(
            "streaming sparse attribution subcontract mode must be streaming_sparse"
        )
    if receipt.fp_exception_caveat != OPTIMIZER_CREDIT_STATE_FP_EXCEPTION_CAVEAT:
        raise ValueError(
            "streaming sparse attribution subcontract must keep exact FP-exception caveat"
        )
    if receipt.non_claims != STREAMING_SPARSE_ATTRIBUTION_SUBCONTRACT_NON_CLAIMS:
        raise ValueError(
            "streaming sparse attribution subcontract non_claims must be exact"
        )
    for field in FORBIDDEN_STREAMING_SPARSE_SUBCONTRACT_FIELDS:
        if bool(getattr(receipt, field)):
            raise ValueError(f"{field} is forbidden on streaming sparse subcontract receipt")
    if receipt.reference_oracle_run_id == receipt.candidate_run_id:
        raise ValueError("reference_oracle_run_id must differ from candidate_run_id")
    int_surfaces = tuple(receipt.candidate_dense_integer_scratch_surfaces)
    if receipt.candidate_dense_integer_scratch_observed != bool(int_surfaces):
        raise ValueError(
            "candidate_dense_integer_scratch_observed must match surfaces tuple"
        )
    if not receipt.full_support_parity_pass:
        raise ValueError("full_support_parity_pass must be true for a valid subcontract receipt")
    if receipt.candidate_dense_integer_scratch_observed:
        raise ValueError(
            "candidate_dense_integer_scratch_observed must be false for streaming_sparse pass"
        )
    if receipt.max_candidate_tile_bytes > receipt.full_dense_baseline_bytes:
        raise ValueError(
            "max_candidate_tile_bytes must not exceed full_dense_baseline_bytes"
        )
    _validate_streaming_sparse_attribution_metric_invariants(
        max_candidate_tile_shape=receipt.max_candidate_tile_shape,
        max_candidate_tile_numel=receipt.max_candidate_tile_numel,
        max_candidate_tile_bytes=receipt.max_candidate_tile_bytes,
        full_dense_shape=receipt.full_dense_shape,
        full_dense_numel=receipt.full_dense_numel,
        full_dense_baseline_bytes=receipt.full_dense_baseline_bytes,
        candidate_event_count=receipt.candidate_event_count,
        candidate_event_carrier_peak_bytes=receipt.candidate_event_carrier_peak_bytes,
        event_carrier_density_ratio=receipt.event_carrier_density_ratio,
    )
