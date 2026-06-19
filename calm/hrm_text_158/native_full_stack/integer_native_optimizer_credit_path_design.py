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
import hashlib
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


# --- BR-3C-F strict-integer ranking subcontract receipt ---

from calm.hrm_text_158.native_full_stack.integer_sparse_rank_votes import (  # noqa: E402
    BR_F_RANKING_INTEGER_EXACT,
    INTEGER_SPARSE_RANK_PRODUCTION_CREDIT_LAW_ID,
    PRODUCTION_STRICT_INTEGER_CREDIT_LAW_IDS,
    StrictIntegerRankingComparisonResult,
    compare_strict_integer_ranking_to_float32_reference,
)
from calm.hrm_text_158.native_full_stack.bounded_delta_learner import RankVoteSpec  # noqa: E402

RANKING_SUBCONTRACT_SCHEMA_VERSION = "hrm_text_158_strict_integer_ranking_subcontract/v1"
RANKING_SUBCONTRACT_TARGET_NAME = "optimizer_credit_state_strict_integer_ranking_subcontract"
RANKING_SUBCONTRACT_MODE_STRICT_INTEGER = "strict_integer"

RANKING_SUBCONTRACT_NON_CLAIMS = (
    *OPTIMIZER_CREDIT_STATE_FAIL_CLOSED_NON_CLAIMS,
    "ranking subcontract receipt only; attribution subcontract evaluated separately",
    "does NOT invoke branch-8 classifier or claim BR-D-INTEGER-VIABLE",
    "does NOT prove GPU memory relief or subquadratic compute",
    "does NOT flip optimizer_credit_state sub2 row or set readiness flags",
    "pow2 credit law strict-integer path is OUT OF SCOPE for this subcontract",
    "divergence branches are terminal science; drop-in float32 parity may be false",
    "CPU strict-integer ranking candidate only; not real_native_integer_credit_ranking_present",
)

FORBIDDEN_RANKING_SUBCONTRACT_FIELDS = (
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
class RankingSubcontractReceipt:
    schema_version: str
    target_name: str
    ranking_subcontract_mode: str
    credit_law_id: str
    rank_method: str
    rank_bin_spec_canonical_tuple: tuple[tuple[int, int, int, int, int, bool], ...]
    rank_bin_spec_sha256: str
    candidate_count: int
    credit_q31_count: int
    projected_move_count: int
    flat_index_count: int
    emitted_event_count: int
    integer_vs_float_rank_mismatch_count: int
    vote_mismatch_count: int
    measurement_invalid_count: int
    representation_limit_count: int
    partial_coverage_count: int
    bin_boundary_divergence_count: int
    precision_divergence_count: int
    tie_group_divergence_count: int
    drop_in_float32_parity_pass: bool
    strict_integer_self_consistency_pass: bool
    branch_id: str
    comparable_set_id: str
    reference_float32_run_id: str
    candidate_strict_run_id: str
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


def ranking_subcontract_hard_false_snapshot() -> dict[str, bool]:
    return {field: False for field in FORBIDDEN_RANKING_SUBCONTRACT_FIELDS}


def _validate_ranking_dual_pass_coupling(receipt: RankingSubcontractReceipt) -> None:
    if receipt.branch_id == BR_F_RANKING_INTEGER_EXACT:
        if not receipt.drop_in_float32_parity_pass:
            raise ValueError("INTEGER-EXACT requires drop_in_float32_parity_pass=true")
        if not receipt.strict_integer_self_consistency_pass:
            raise ValueError("INTEGER-EXACT requires strict_integer_self_consistency_pass=true")
        return
    if receipt.branch_id in {
        "BR-F-RANKING-BIN-BOUNDARY-DIVERGENCE",
        "BR-F-RANKING-TIE-GROUP-DIVERGENCE",
        "BR-F-RANKING-PRECISION-DIVERGENCE",
    }:
        if receipt.drop_in_float32_parity_pass:
            raise ValueError("divergence branch requires drop_in_float32_parity_pass=false")
        return
    if receipt.drop_in_float32_parity_pass or receipt.strict_integer_self_consistency_pass:
        raise ValueError("measurement/representation/partial branches require both pass booleans false")


def _validate_ranking_branch_id_coupling(receipt: RankingSubcontractReceipt) -> None:
    divergence_total = (
        receipt.bin_boundary_divergence_count
        + receipt.precision_divergence_count
        + receipt.tie_group_divergence_count
        + receipt.measurement_invalid_count
        + receipt.representation_limit_count
        + receipt.partial_coverage_count
    )
    if receipt.branch_id == BR_F_RANKING_INTEGER_EXACT:
        if divergence_total != 0:
            raise ValueError("INTEGER-EXACT requires all divergence counts zero")
        if receipt.integer_vs_float_rank_mismatch_count != 0:
            raise ValueError("INTEGER-EXACT requires zero rank mismatches")
        if receipt.vote_mismatch_count != 0:
            raise ValueError("INTEGER-EXACT requires zero vote mismatches")
        return
    if divergence_total != 1:
        raise ValueError("non-INTEGER-EXACT branch requires exactly one divergence count")


def _validate_ranking_metric_invariants(receipt: RankingSubcontractReceipt) -> None:
    if receipt.candidate_count < 0:
        raise ValueError("candidate_count must be >= 0")
    if receipt.emitted_event_count > receipt.candidate_count:
        raise ValueError("emitted_event_count must be <= candidate_count")
    if receipt.credit_q31_count != receipt.candidate_count:
        raise ValueError("credit_q31_count must equal candidate_count")
    if receipt.projected_move_count != receipt.candidate_count:
        raise ValueError("projected_move_count must equal candidate_count")
    if receipt.flat_index_count != receipt.candidate_count:
        raise ValueError("flat_index_count must equal candidate_count")
    expected_sha = canonical_rank_bin_spec_sha256_from_tuple(receipt.rank_bin_spec_canonical_tuple)
    if receipt.rank_bin_spec_sha256 != expected_sha:
        raise ValueError("rank_bin_spec_sha256 must match canonical tuple")


def canonical_rank_bin_spec_sha256_from_tuple(
    canonical_tuple: tuple[tuple[int, int, int, int, int, bool], ...],
) -> str:
    return hashlib.sha256(repr(canonical_tuple).encode("utf-8")).hexdigest()


def build_ranking_subcontract_receipt(
    comparison: StrictIntegerRankingComparisonResult,
    *,
    comparable_set_id: str,
    reference_float32_run_id: str,
    candidate_strict_run_id: str,
) -> RankingSubcontractReceipt:
    if comparison.credit_law_id not in PRODUCTION_STRICT_INTEGER_CREDIT_LAW_IDS:
        raise ValueError("ranking subcontract requires production neg-attribution credit law")
    if comparison.rank_bin_spec_sha256 != canonical_rank_bin_spec_sha256_from_tuple(
        comparison.rank_bin_spec_canonical_tuple
    ):
        raise ValueError("comparison rank_bin_spec_sha256 must match canonical tuple")
    receipt = RankingSubcontractReceipt(
        schema_version=RANKING_SUBCONTRACT_SCHEMA_VERSION,
        target_name=RANKING_SUBCONTRACT_TARGET_NAME,
        ranking_subcontract_mode=RANKING_SUBCONTRACT_MODE_STRICT_INTEGER,
        credit_law_id=comparison.credit_law_id,
        rank_method=comparison.rank_method,
        rank_bin_spec_canonical_tuple=comparison.rank_bin_spec_canonical_tuple,
        rank_bin_spec_sha256=comparison.rank_bin_spec_sha256,
        candidate_count=comparison.candidate_count,
        credit_q31_count=comparison.credit_q31_count,
        projected_move_count=comparison.projected_move_count,
        flat_index_count=comparison.flat_index_count,
        emitted_event_count=comparison.emitted_event_count,
        integer_vs_float_rank_mismatch_count=comparison.integer_vs_float_rank_mismatch_count,
        vote_mismatch_count=comparison.vote_mismatch_count,
        measurement_invalid_count=comparison.measurement_invalid_count,
        representation_limit_count=comparison.representation_limit_count,
        partial_coverage_count=comparison.partial_coverage_count,
        bin_boundary_divergence_count=comparison.bin_boundary_divergence_count,
        precision_divergence_count=comparison.precision_divergence_count,
        tie_group_divergence_count=comparison.tie_group_divergence_count,
        drop_in_float32_parity_pass=comparison.drop_in_float32_parity_pass,
        strict_integer_self_consistency_pass=comparison.strict_integer_self_consistency_pass,
        branch_id=comparison.branch_id,
        comparable_set_id=comparable_set_id,
        reference_float32_run_id=reference_float32_run_id,
        candidate_strict_run_id=candidate_strict_run_id,
        fp_exception_caveat=OPTIMIZER_CREDIT_STATE_FP_EXCEPTION_CAVEAT,
        non_claims=RANKING_SUBCONTRACT_NON_CLAIMS,
    )
    validate_ranking_subcontract_receipt(receipt)
    return receipt


def prove_strict_integer_ranking_subcontract(
    credit_q31: torch.Tensor,
    projected_moves: torch.Tensor,
    flat_indices: torch.Tensor,
    spec: RankVoteSpec,
    *,
    comparable_set_id: str,
    reference_float32_run_id: str,
    candidate_strict_run_id: str,
    credit_law_id: str = INTEGER_SPARSE_RANK_PRODUCTION_CREDIT_LAW_ID,
) -> RankingSubcontractReceipt:
    comparison = compare_strict_integer_ranking_to_float32_reference(
        credit_q31,
        projected_moves,
        flat_indices,
        spec,
        credit_law_id=credit_law_id,
    )
    return build_ranking_subcontract_receipt(
        comparison,
        comparable_set_id=comparable_set_id,
        reference_float32_run_id=reference_float32_run_id,
        candidate_strict_run_id=candidate_strict_run_id,
    )


def validate_ranking_subcontract_receipt(receipt: RankingSubcontractReceipt) -> None:
    if receipt.schema_version != RANKING_SUBCONTRACT_SCHEMA_VERSION:
        raise ValueError("ranking subcontract schema mismatch")
    if receipt.target_name != RANKING_SUBCONTRACT_TARGET_NAME:
        raise ValueError("ranking subcontract target mismatch")
    if receipt.ranking_subcontract_mode != RANKING_SUBCONTRACT_MODE_STRICT_INTEGER:
        raise ValueError("ranking subcontract mode must be strict_integer")
    if receipt.credit_law_id not in PRODUCTION_STRICT_INTEGER_CREDIT_LAW_IDS:
        raise ValueError("ranking subcontract credit_law_id must be production neg-attribution")
    if receipt.fp_exception_caveat != OPTIMIZER_CREDIT_STATE_FP_EXCEPTION_CAVEAT:
        raise ValueError("ranking subcontract must keep exact FP-exception caveat")
    if receipt.non_claims != RANKING_SUBCONTRACT_NON_CLAIMS:
        raise ValueError("ranking subcontract non_claims must be exact")
    for field in FORBIDDEN_RANKING_SUBCONTRACT_FIELDS:
        if bool(getattr(receipt, field)):
            raise ValueError(f"{field} is forbidden on ranking subcontract receipt")
    if receipt.reference_float32_run_id == receipt.candidate_strict_run_id:
        raise ValueError("reference_float32_run_id must differ from candidate_strict_run_id")
    _validate_ranking_metric_invariants(receipt)
    _validate_ranking_branch_id_coupling(receipt)
    _validate_ranking_dual_pass_coupling(receipt)


# --- BR-3C-G integer credit-axis CPU integration (frozen PLAN v2) ---

import numpy as np  # noqa: E402

from calm.hrm_text_158.native_full_stack.integer_marginal_attribution import (  # noqa: E402
    projected_moves_from_integer_attribution,
)
from calm.hrm_text_158.native_full_stack.integer_sparse_rank_votes import (  # noqa: E402
    credit_q31_from_attribution,
)

INTEGER_CREDIT_AXIS_INTEGRATION_SCHEMA_VERSION = (
    "hrm_text_158_integer_credit_axis_integration/v2"
)
INTEGER_CREDIT_AXIS_INTEGRATION_TARGET_NAME = (
    "optimizer_credit_state_native_integer_candidate_integration"
)
INTEGRATION_AUTHORITY_CPU_EVIDENCE_ONLY = "cpu_evidence_only"
INTEGRATION_HASH_BYTE_ORDER = "little_endian"

INTEGER_CREDIT_AXIS_INTEGRATION_NON_CLAIMS = (
    *OPTIMIZER_CREDIT_STATE_FAIL_CLOSED_NON_CLAIMS,
    "CPU integration evidence only; optimizer_credit_state row flip deferred to GPU runtime receipt",
    "BR-D-INTEGER-VIABLE on integration receipt is branch-8 CPU evidence, not readiness flip authority",
    "ranking subcontract evaluated under BR-F INTEGER-EXACT gate; attribution under streaming_sparse gate",
)

FORBIDDEN_INTEGRATION_RECEIPT_FIELDS = (
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
    "persistent_carrier_width_claim",
)


@dataclass(frozen=True)
class BoundCandidateAttributionEvents:
    law_id: str
    numel: int
    flat_indices: torch.Tensor
    attribution_q31: torch.Tensor

    def validate(self) -> None:
        if int(self.flat_indices.numel()) != int(self.attribution_q31.numel()):
            raise ValueError("bound attribution events length mismatch")

    def as_integer_marginal_attribution_events(self) -> IntegerMarginalAttributionEvents:
        return IntegerMarginalAttributionEvents(
            law_id=self.law_id,
            numel=self.numel,
            flat_indices=self.flat_indices,
            attribution_q31=self.attribution_q31,
        )


@dataclass(frozen=True)
class IntegerCreditAxisIntegrationReceipt:
    schema_version: str
    target_name: str
    branch_id: str
    integration_authority_level: str
    attribution_subcontract_pass: bool
    ranking_subcontract_pass: bool
    attribution_subcontract_snapshot: StreamingSparseAttributionSubcontractReceipt
    ranking_subcontract_snapshot: RankingSubcontractReceipt
    bound_candidate_attribution_events: BoundCandidateAttributionEvents
    bound_q_levels_flat: torch.Tensor
    bound_projected_move_indices: torch.Tensor
    bound_projected_moves: torch.Tensor
    bound_credit_q31: torch.Tensor
    candidate_alloc_guard_pass: bool
    candidate_dense_surfaces_observed: tuple[str, ...]
    candidate_dense_integer_scratch_observed: bool
    candidate_dense_integer_scratch_surfaces: tuple[str, ...]
    capture_transient_discriminator_pass: bool
    capture_retained_fp_tensor_count: int
    capture_stashed_in_closure_or_registry_count: int
    comparable_set_complete: bool
    partial_coverage_only: bool
    attribution_events_hash: str
    projected_move_indices_hash: str
    projected_moves_hash: str
    credit_q31_hash: str
    q_levels_hash: str
    rank_bin_spec_hash: str
    comparable_set_id_hash: str
    candidate_run_id_hash: str
    reference_oracle_run_id_hash: str
    integration_data_digest_sha256: str
    hash_byte_order: str
    comparable_set_id: str
    candidate_run_id: str
    reference_oracle_run_id: str
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
    persistent_carrier_width_claim: bool = False


def integer_credit_axis_integration_hard_false_snapshot() -> dict[str, bool]:
    return {field: False for field in FORBIDDEN_INTEGRATION_RECEIPT_FIELDS}


def _numpy_little_endian_array(tensor: torch.Tensor) -> np.ndarray:
    t = tensor.detach().cpu().contiguous()
    arr = t.numpy()
    if arr.dtype.byteorder == ">":
        le_dtype = np.dtype(arr.dtype.str).newbyteorder("<")
        return arr.astype(le_dtype, copy=False)
    if arr.dtype.byteorder == "=" and not np.little_endian:
        le_dtype = np.dtype(arr.dtype.str).newbyteorder("<")
        return arr.astype(le_dtype, copy=False)
    return arr


def canonical_tensor_payload_sha256(tensor: torch.Tensor) -> str:
    t = tensor.detach().cpu().contiguous()
    meta = (
        f"{str(t.dtype)}|{tuple(int(x) for x in t.shape)}|{INTEGRATION_HASH_BYTE_ORDER}|"
    ).encode("utf-8")
    return hashlib.sha256(meta + _numpy_little_endian_array(t).tobytes()).hexdigest()


def canonical_attribution_events_payload_sha256(
    events: BoundCandidateAttributionEvents,
) -> str:
    flat_hash = canonical_tensor_payload_sha256(events.flat_indices)
    attr_hash = canonical_tensor_payload_sha256(events.attribution_q31)
    return hashlib.sha256((flat_hash + attr_hash).encode("utf-8")).hexdigest()


def canonical_utf8_payload_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def integration_data_digest_sha256_from_payload_hashes(
    *,
    attribution_events_hash: str,
    projected_move_indices_hash: str,
    projected_moves_hash: str,
    credit_q31_hash: str,
    q_levels_hash: str,
    rank_bin_spec_hash: str,
    comparable_set_id_hash: str,
    candidate_run_id_hash: str,
    reference_oracle_run_id_hash: str,
) -> str:
    joined = (
        attribution_events_hash
        + projected_move_indices_hash
        + projected_moves_hash
        + credit_q31_hash
        + q_levels_hash
        + rank_bin_spec_hash
        + comparable_set_id_hash
        + candidate_run_id_hash
        + reference_oracle_run_id_hash
    )
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def _attribution_selected_for_moves(
    events: IntegerMarginalAttributionEvents | BoundCandidateAttributionEvents,
    move_indices: torch.Tensor,
) -> torch.Tensor:
    index_to_pos = {
        int(index): pos for pos, index in enumerate(events.flat_indices.tolist())
    }
    return torch.tensor(
        [
            int(events.attribution_q31[index_to_pos[int(index)]].item())
            for index in move_indices.tolist()
        ],
        dtype=torch.int32,
    )


def _bound_attribution_events_from_candidate(
    candidate_events: IntegerMarginalAttributionEvents,
) -> BoundCandidateAttributionEvents:
    return BoundCandidateAttributionEvents(
        law_id=candidate_events.law_id,
        numel=candidate_events.numel,
        flat_indices=candidate_events.flat_indices.detach().cpu().contiguous().clone(),
        attribution_q31=candidate_events.attribution_q31.detach().cpu().contiguous().clone(),
    )


def _cross_bind_ranking_tensors_from_bound_events(
    bound_events: BoundCandidateAttributionEvents,
    q_levels_flat: torch.Tensor,
    *,
    credit_law_id: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    events = bound_events.as_integer_marginal_attribution_events()
    move_indices, moves = projected_moves_from_integer_attribution(events, q_levels_flat)
    attribution_selected = _attribution_selected_for_moves(events, move_indices)
    credit_q31 = credit_q31_from_attribution(
        attribution_selected,
        credit_law_id=credit_law_id,
    )
    return (
        move_indices.contiguous(),
        moves.contiguous(),
        credit_q31.contiguous(),
    )


def _compute_attribution_subcontract_pass(
    snapshot: StreamingSparseAttributionSubcontractReceipt,
    *,
    candidate_run_id: str,
    comparable_set_id: str,
) -> bool:
    try:
        validate_streaming_sparse_attribution_subcontract_receipt(snapshot)
    except ValueError:
        return False
    return (
        snapshot.full_support_parity_pass is True
        and snapshot.attribution_subcontract_mode == ATTRIBUTION_SUBCONTRACT_MODE_STREAMING_SPARSE
        and snapshot.candidate_dense_integer_scratch_observed is False
        and snapshot.candidate_run_id == candidate_run_id
        and snapshot.comparable_set_id == comparable_set_id
    )


def _compute_ranking_subcontract_pass(
    snapshot: RankingSubcontractReceipt,
    *,
    candidate_run_id: str,
    comparable_set_id: str,
) -> bool:
    try:
        validate_ranking_subcontract_receipt(snapshot)
    except ValueError:
        return False
    return (
        snapshot.branch_id == BR_F_RANKING_INTEGER_EXACT
        and snapshot.drop_in_float32_parity_pass is True
        and snapshot.strict_integer_self_consistency_pass is True
        and snapshot.integer_vs_float_rank_mismatch_count == 0
        and snapshot.vote_mismatch_count == 0
        and snapshot.candidate_strict_run_id == candidate_run_id
        and snapshot.comparable_set_id == comparable_set_id
    )


def _recompute_integration_branch_id(
    *,
    attribution_subcontract_snapshot: StreamingSparseAttributionSubcontractReceipt,
    attribution_subcontract_pass: bool,
    ranking_subcontract_pass: bool,
    capture_retained_fp_tensor_count: int,
    capture_stashed_in_closure_or_registry_count: int,
    comparable_set_complete: bool,
    measurement_complete: bool = True,
    partial_coverage_only: bool = False,
    wire_shape_only_pass: bool = False,
) -> str:
    capture_transient_discriminator_pass = evaluate_capture_transient_discriminator(
        capture_retained_fp_tensor_count=capture_retained_fp_tensor_count,
        capture_stashed_in_closure_or_registry_count=capture_stashed_in_closure_or_registry_count,
    )
    scratch_observed = attribution_subcontract_snapshot.candidate_dense_integer_scratch_observed
    scratch_surfaces = tuple(
        attribution_subcontract_snapshot.candidate_dense_integer_scratch_surfaces
    )
    return classify_integer_native_optimizer_credit_path_branch(
        candidate_alloc_guard_pass=not scratch_observed,
        candidate_dense_surfaces_observed=scratch_surfaces,
        candidate_dense_integer_scratch_observed=scratch_observed,
        capture_transient_discriminator_pass=capture_transient_discriminator_pass,
        attribution_subcontract_mode=attribution_subcontract_snapshot.attribution_subcontract_mode,
        attribution_subcontract_pass=attribution_subcontract_pass,
        ranking_subcontract_pass=ranking_subcontract_pass,
        comparable_set_complete=comparable_set_complete,
        measurement_complete=measurement_complete,
        partial_coverage_only=partial_coverage_only,
        wire_shape_only_pass=wire_shape_only_pass,
    )


def _validate_integration_hash_bindings(
    receipt: IntegerCreditAxisIntegrationReceipt,
    *,
    credit_law_id: str,
) -> None:
    if receipt.hash_byte_order != INTEGRATION_HASH_BYTE_ORDER:
        raise ValueError("integration receipt hash_byte_order must be little_endian")
    bound_events = receipt.bound_candidate_attribution_events
    bound_events.validate()
    expected_attribution_hash = canonical_attribution_events_payload_sha256(bound_events)
    if receipt.attribution_events_hash != expected_attribution_hash:
        raise ValueError("attribution_events_hash mismatch")
    q_levels = receipt.bound_q_levels_flat.detach().cpu().contiguous()
    if receipt.q_levels_hash != canonical_tensor_payload_sha256(q_levels):
        raise ValueError("q_levels_hash mismatch")
    move_indices, moves, credit_q31 = _cross_bind_ranking_tensors_from_bound_events(
        bound_events,
        q_levels,
        credit_law_id=credit_law_id,
    )
    if not torch.equal(move_indices, receipt.bound_projected_move_indices):
        raise ValueError("bound projected_move_indices cross-bind mismatch")
    if not torch.equal(moves, receipt.bound_projected_moves):
        raise ValueError("bound projected_moves cross-bind mismatch")
    if not torch.equal(credit_q31, receipt.bound_credit_q31):
        raise ValueError("bound credit_q31 cross-bind mismatch")
    if receipt.projected_move_indices_hash != canonical_tensor_payload_sha256(move_indices):
        raise ValueError("projected_move_indices_hash mismatch")
    if receipt.projected_moves_hash != canonical_tensor_payload_sha256(moves):
        raise ValueError("projected_moves_hash mismatch")
    if receipt.credit_q31_hash != canonical_tensor_payload_sha256(credit_q31):
        raise ValueError("credit_q31_hash mismatch")
    expected_rank_spec_hash = canonical_rank_bin_spec_sha256_from_tuple(
        receipt.ranking_subcontract_snapshot.rank_bin_spec_canonical_tuple
    )
    if receipt.rank_bin_spec_hash != expected_rank_spec_hash:
        raise ValueError("rank_bin_spec_hash mismatch")
    if (
        receipt.ranking_subcontract_snapshot.rank_bin_spec_sha256
        != receipt.rank_bin_spec_hash
    ):
        raise ValueError("ranking snapshot rank_bin_spec_sha256 bind mismatch")
    if receipt.comparable_set_id_hash != canonical_utf8_payload_sha256(
        receipt.comparable_set_id
    ):
        raise ValueError("comparable_set_id_hash mismatch")
    if receipt.candidate_run_id_hash != canonical_utf8_payload_sha256(
        receipt.candidate_run_id
    ):
        raise ValueError("candidate_run_id_hash mismatch")
    if receipt.reference_oracle_run_id_hash != canonical_utf8_payload_sha256(
        receipt.reference_oracle_run_id
    ):
        raise ValueError("reference_oracle_run_id_hash mismatch")
    expected_digest = integration_data_digest_sha256_from_payload_hashes(
        attribution_events_hash=receipt.attribution_events_hash,
        projected_move_indices_hash=receipt.projected_move_indices_hash,
        projected_moves_hash=receipt.projected_moves_hash,
        credit_q31_hash=receipt.credit_q31_hash,
        q_levels_hash=receipt.q_levels_hash,
        rank_bin_spec_hash=receipt.rank_bin_spec_hash,
        comparable_set_id_hash=receipt.comparable_set_id_hash,
        candidate_run_id_hash=receipt.candidate_run_id_hash,
        reference_oracle_run_id_hash=receipt.reference_oracle_run_id_hash,
    )
    if receipt.integration_data_digest_sha256 != expected_digest:
        raise ValueError("integration_data_digest_sha256 mismatch")


def build_integer_credit_axis_integration_receipt(
    *,
    candidate_events: IntegerMarginalAttributionEvents,
    q_levels_flat: torch.Tensor,
    bound_projected_move_indices: torch.Tensor,
    bound_projected_moves: torch.Tensor,
    bound_credit_q31: torch.Tensor,
    attribution_subcontract_snapshot: StreamingSparseAttributionSubcontractReceipt,
    ranking_subcontract_snapshot: RankingSubcontractReceipt,
    rank_spec: RankVoteSpec,
    comparable_set_id: str,
    reference_oracle_run_id: str,
    candidate_run_id: str,
    credit_law_id: str = INTEGER_SPARSE_RANK_PRODUCTION_CREDIT_LAW_ID,
    capture_retained_fp_tensor_count: int = 0,
    capture_stashed_in_closure_or_registry_count: int = 0,
    comparable_set_complete: bool | None = None,
    partial_coverage_only: bool = False,
    wire_shape_only_pass: bool = False,
    measurement_complete: bool = True,
) -> IntegerCreditAxisIntegrationReceipt:
    if reference_oracle_run_id == candidate_run_id:
        raise ValueError("reference_oracle_run_id must differ from candidate_run_id")
    bound_events = _bound_attribution_events_from_candidate(candidate_events)
    if not events_bit_identical(
        candidate_events,
        bound_events.as_integer_marginal_attribution_events(),
    ):
        raise ValueError("bound candidate attribution events must be bit-identical to source")
    validate_streaming_sparse_attribution_subcontract_receipt(attribution_subcontract_snapshot)
    validate_ranking_subcontract_receipt(ranking_subcontract_snapshot)
    attribution_subcontract_pass = _compute_attribution_subcontract_pass(
        attribution_subcontract_snapshot,
        candidate_run_id=candidate_run_id,
        comparable_set_id=comparable_set_id,
    )
    ranking_subcontract_pass = _compute_ranking_subcontract_pass(
        ranking_subcontract_snapshot,
        candidate_run_id=candidate_run_id,
        comparable_set_id=comparable_set_id,
    )
    if attribution_subcontract_snapshot.candidate_run_id != candidate_run_id:
        raise ValueError("attribution snapshot candidate_run_id bind mismatch")
    if attribution_subcontract_snapshot.comparable_set_id != comparable_set_id:
        raise ValueError("attribution snapshot comparable_set_id bind mismatch")
    if ranking_subcontract_snapshot.candidate_strict_run_id != candidate_run_id:
        raise ValueError("ranking snapshot candidate_strict_run_id bind mismatch")
    if ranking_subcontract_snapshot.comparable_set_id != comparable_set_id:
        raise ValueError("ranking snapshot comparable_set_id bind mismatch")
    if ranking_subcontract_snapshot.reference_float32_run_id != reference_oracle_run_id:
        raise ValueError("ranking snapshot reference_float32_run_id bind mismatch")
    if (
        attribution_subcontract_snapshot.candidate_event_count
        != int(bound_events.flat_indices.numel())
    ):
        raise ValueError("attribution snapshot event count bind mismatch")
    bound_q_levels = q_levels_flat.detach().cpu().contiguous().reshape(-1)
    if int(bound_q_levels.numel()) != int(bound_events.numel):
        raise ValueError("bound q_levels_flat numel mismatch")
    re_move_indices, re_moves, re_credit_q31 = _cross_bind_ranking_tensors_from_bound_events(
        bound_events,
        bound_q_levels,
        credit_law_id=credit_law_id,
    )
    if not torch.equal(re_move_indices, bound_projected_move_indices.contiguous()):
        raise ValueError("builder projected_move_indices must match bound re-derivation")
    if not torch.equal(re_moves, bound_projected_moves.contiguous()):
        raise ValueError("builder projected_moves must match bound re-derivation")
    if not torch.equal(re_credit_q31, bound_credit_q31.contiguous()):
        raise ValueError("builder credit_q31 must match bound re-derivation")
    if comparable_set_complete is None:
        comparable_set_complete = (
            attribution_subcontract_pass
            and ranking_subcontract_pass
            and int(ranking_subcontract_snapshot.candidate_count)
            == int(bound_projected_move_indices.numel())
        )
    branch_id = _recompute_integration_branch_id(
        attribution_subcontract_snapshot=attribution_subcontract_snapshot,
        attribution_subcontract_pass=attribution_subcontract_pass,
        ranking_subcontract_pass=ranking_subcontract_pass,
        capture_retained_fp_tensor_count=capture_retained_fp_tensor_count,
        capture_stashed_in_closure_or_registry_count=(
            capture_stashed_in_closure_or_registry_count
        ),
        comparable_set_complete=comparable_set_complete,
        measurement_complete=measurement_complete,
        partial_coverage_only=partial_coverage_only,
        wire_shape_only_pass=wire_shape_only_pass,
    )
    attribution_events_hash = canonical_attribution_events_payload_sha256(bound_events)
    projected_move_indices_hash = canonical_tensor_payload_sha256(re_move_indices)
    projected_moves_hash = canonical_tensor_payload_sha256(re_moves)
    credit_q31_hash = canonical_tensor_payload_sha256(re_credit_q31)
    q_levels_hash = canonical_tensor_payload_sha256(bound_q_levels)
    rank_bin_spec_hash = canonical_rank_bin_spec_sha256_from_tuple(
        ranking_subcontract_snapshot.rank_bin_spec_canonical_tuple
    )
    comparable_set_id_hash = canonical_utf8_payload_sha256(comparable_set_id)
    candidate_run_id_hash = canonical_utf8_payload_sha256(candidate_run_id)
    reference_oracle_run_id_hash = canonical_utf8_payload_sha256(reference_oracle_run_id)
    integration_digest = integration_data_digest_sha256_from_payload_hashes(
        attribution_events_hash=attribution_events_hash,
        projected_move_indices_hash=projected_move_indices_hash,
        projected_moves_hash=projected_moves_hash,
        credit_q31_hash=credit_q31_hash,
        q_levels_hash=q_levels_hash,
        rank_bin_spec_hash=rank_bin_spec_hash,
        comparable_set_id_hash=comparable_set_id_hash,
        candidate_run_id_hash=candidate_run_id_hash,
        reference_oracle_run_id_hash=reference_oracle_run_id_hash,
    )
    scratch_observed = attribution_subcontract_snapshot.candidate_dense_integer_scratch_observed
    scratch_surfaces = tuple(
        attribution_subcontract_snapshot.candidate_dense_integer_scratch_surfaces
    )
    capture_transient_discriminator_pass = evaluate_capture_transient_discriminator(
        capture_retained_fp_tensor_count=capture_retained_fp_tensor_count,
        capture_stashed_in_closure_or_registry_count=(
            capture_stashed_in_closure_or_registry_count
        ),
    )
    receipt = IntegerCreditAxisIntegrationReceipt(
        schema_version=INTEGER_CREDIT_AXIS_INTEGRATION_SCHEMA_VERSION,
        target_name=INTEGER_CREDIT_AXIS_INTEGRATION_TARGET_NAME,
        branch_id=branch_id,
        integration_authority_level=INTEGRATION_AUTHORITY_CPU_EVIDENCE_ONLY,
        attribution_subcontract_pass=attribution_subcontract_pass,
        ranking_subcontract_pass=ranking_subcontract_pass,
        attribution_subcontract_snapshot=attribution_subcontract_snapshot,
        ranking_subcontract_snapshot=ranking_subcontract_snapshot,
        bound_candidate_attribution_events=bound_events,
        bound_q_levels_flat=bound_q_levels,
        bound_projected_move_indices=re_move_indices,
        bound_projected_moves=re_moves,
        bound_credit_q31=re_credit_q31,
        candidate_alloc_guard_pass=not scratch_observed,
        candidate_dense_surfaces_observed=scratch_surfaces,
        candidate_dense_integer_scratch_observed=scratch_observed,
        candidate_dense_integer_scratch_surfaces=scratch_surfaces,
        capture_transient_discriminator_pass=capture_transient_discriminator_pass,
        capture_retained_fp_tensor_count=int(capture_retained_fp_tensor_count),
        capture_stashed_in_closure_or_registry_count=int(
            capture_stashed_in_closure_or_registry_count
        ),
        comparable_set_complete=comparable_set_complete,
        partial_coverage_only=partial_coverage_only,
        attribution_events_hash=attribution_events_hash,
        projected_move_indices_hash=projected_move_indices_hash,
        projected_moves_hash=projected_moves_hash,
        credit_q31_hash=credit_q31_hash,
        q_levels_hash=q_levels_hash,
        rank_bin_spec_hash=rank_bin_spec_hash,
        comparable_set_id_hash=comparable_set_id_hash,
        candidate_run_id_hash=candidate_run_id_hash,
        reference_oracle_run_id_hash=reference_oracle_run_id_hash,
        integration_data_digest_sha256=integration_digest,
        hash_byte_order=INTEGRATION_HASH_BYTE_ORDER,
        comparable_set_id=comparable_set_id,
        candidate_run_id=candidate_run_id,
        reference_oracle_run_id=reference_oracle_run_id,
        fp_exception_caveat=OPTIMIZER_CREDIT_STATE_FP_EXCEPTION_CAVEAT,
        non_claims=INTEGER_CREDIT_AXIS_INTEGRATION_NON_CLAIMS,
    )
    validate_integer_credit_axis_integration_receipt(receipt, credit_law_id=credit_law_id)
    return receipt


def prove_integer_credit_axis_integration(
    inputs: Sequence[torch.Tensor],
    grad_outputs: Sequence[torch.Tensor],
    *,
    weight_shape: Sequence[int],
    q_levels_flat: torch.Tensor,
    rank_spec: RankVoteSpec,
    comparable_set_id: str,
    reference_oracle_run_id: str,
    candidate_run_id: str,
    law_id: str = INTEGER_MARGINAL_ATTRIBUTION_PRODUCTION_LAW_ID,
    credit_law_id: str = INTEGER_SPARSE_RANK_PRODUCTION_CREDIT_LAW_ID,
    capture_retained_fp_tensor_count: int = 0,
    capture_stashed_in_closure_or_registry_count: int = 0,
    comparable_set_complete: bool | None = None,
    partial_coverage_only: bool = False,
) -> IntegerCreditAxisIntegrationReceipt:
    weight_dims = tuple(int(dim) for dim in weight_shape)
    oracle_events = integer_marginal_attribution_from_captures(
        inputs,
        grad_outputs,
        weight_shape=weight_dims,
        law_id=law_id,
    )
    with candidate_dense_integer_dispatch_observation(weight_dims) as observer:
        candidate_events, sparse_metrics = streaming_sparse_attribution_from_captures(
            inputs,
            grad_outputs,
            weight_shape=weight_dims,
            law_id=law_id,
        )
    dispatch_obs = observer.observation()
    parity_pass = events_bit_identical(oracle_events, candidate_events)
    attribution_snapshot = build_streaming_sparse_attribution_subcontract_receipt(
        metrics=sparse_metrics,
        dispatch_observation=dispatch_obs,
        full_support_parity_pass=parity_pass,
        comparable_set_id=comparable_set_id,
        reference_oracle_run_id=reference_oracle_run_id,
        candidate_run_id=candidate_run_id,
    )
    move_indices, moves = projected_moves_from_integer_attribution(
        candidate_events,
        q_levels_flat,
    )
    attribution_selected = _attribution_selected_for_moves(candidate_events, move_indices)
    credit_q31 = credit_q31_from_attribution(
        attribution_selected,
        credit_law_id=credit_law_id,
    )
    ranking_snapshot = prove_strict_integer_ranking_subcontract(
        credit_q31,
        moves,
        move_indices,
        rank_spec,
        comparable_set_id=comparable_set_id,
        reference_float32_run_id=reference_oracle_run_id,
        candidate_strict_run_id=candidate_run_id,
        credit_law_id=credit_law_id,
    )
    return build_integer_credit_axis_integration_receipt(
        candidate_events=candidate_events,
        q_levels_flat=q_levels_flat,
        bound_projected_move_indices=move_indices,
        bound_projected_moves=moves,
        bound_credit_q31=credit_q31,
        attribution_subcontract_snapshot=attribution_snapshot,
        ranking_subcontract_snapshot=ranking_snapshot,
        rank_spec=rank_spec,
        comparable_set_id=comparable_set_id,
        reference_oracle_run_id=reference_oracle_run_id,
        candidate_run_id=candidate_run_id,
        credit_law_id=credit_law_id,
        capture_retained_fp_tensor_count=capture_retained_fp_tensor_count,
        capture_stashed_in_closure_or_registry_count=(
            capture_stashed_in_closure_or_registry_count
        ),
        comparable_set_complete=comparable_set_complete,
        partial_coverage_only=partial_coverage_only,
    )


def validate_integer_credit_axis_integration_receipt(
    receipt: IntegerCreditAxisIntegrationReceipt,
    *,
    credit_law_id: str = INTEGER_SPARSE_RANK_PRODUCTION_CREDIT_LAW_ID,
) -> None:
    if receipt.schema_version != INTEGER_CREDIT_AXIS_INTEGRATION_SCHEMA_VERSION:
        raise ValueError("integer credit axis integration schema mismatch")
    if receipt.target_name != INTEGER_CREDIT_AXIS_INTEGRATION_TARGET_NAME:
        raise ValueError("integer credit axis integration target mismatch")
    if receipt.integration_authority_level != INTEGRATION_AUTHORITY_CPU_EVIDENCE_ONLY:
        raise ValueError("integration_authority_level must be cpu_evidence_only")
    if receipt.fp_exception_caveat != OPTIMIZER_CREDIT_STATE_FP_EXCEPTION_CAVEAT:
        raise ValueError("integration receipt must keep exact FP-exception caveat")
    if receipt.non_claims != INTEGER_CREDIT_AXIS_INTEGRATION_NON_CLAIMS:
        raise ValueError("integration receipt non_claims must be exact")
    for field in FORBIDDEN_INTEGRATION_RECEIPT_FIELDS:
        if bool(getattr(receipt, field)):
            raise ValueError(f"{field} is forbidden on integration receipt")
    validate_streaming_sparse_attribution_subcontract_receipt(
        receipt.attribution_subcontract_snapshot
    )
    validate_ranking_subcontract_receipt(receipt.ranking_subcontract_snapshot)
    if receipt.attribution_subcontract_snapshot.comparable_set_id != receipt.comparable_set_id:
        raise ValueError("attribution snapshot comparable_set_id bind mismatch")
    if receipt.attribution_subcontract_snapshot.candidate_run_id != receipt.candidate_run_id:
        raise ValueError("attribution snapshot candidate_run_id bind mismatch")
    if receipt.ranking_subcontract_snapshot.comparable_set_id != receipt.comparable_set_id:
        raise ValueError("ranking snapshot comparable_set_id bind mismatch")
    if receipt.ranking_subcontract_snapshot.candidate_strict_run_id != receipt.candidate_run_id:
        raise ValueError("ranking snapshot candidate_strict_run_id bind mismatch")
    if (
        receipt.ranking_subcontract_snapshot.reference_float32_run_id
        != receipt.reference_oracle_run_id
    ):
        raise ValueError("ranking snapshot reference_float32_run_id bind mismatch")
    recomputed_attribution_pass = _compute_attribution_subcontract_pass(
        receipt.attribution_subcontract_snapshot,
        candidate_run_id=receipt.candidate_run_id,
        comparable_set_id=receipt.comparable_set_id,
    )
    if receipt.attribution_subcontract_pass != recomputed_attribution_pass:
        raise ValueError("attribution_subcontract_pass mismatch vs carried snapshot")
    recomputed_ranking_pass = _compute_ranking_subcontract_pass(
        receipt.ranking_subcontract_snapshot,
        candidate_run_id=receipt.candidate_run_id,
        comparable_set_id=receipt.comparable_set_id,
    )
    if receipt.ranking_subcontract_pass != recomputed_ranking_pass:
        raise ValueError("ranking_subcontract_pass mismatch vs carried snapshot")
    _validate_integration_hash_bindings(receipt, credit_law_id=credit_law_id)
    scratch_observed = receipt.attribution_subcontract_snapshot.candidate_dense_integer_scratch_observed
    scratch_surfaces = tuple(
        receipt.attribution_subcontract_snapshot.candidate_dense_integer_scratch_surfaces
    )
    if receipt.candidate_dense_integer_scratch_observed != scratch_observed:
        raise ValueError("candidate_dense_integer_scratch_observed snapshot mismatch")
    if receipt.candidate_dense_integer_scratch_surfaces != scratch_surfaces:
        raise ValueError("candidate_dense_integer_scratch_surfaces snapshot mismatch")
    if receipt.candidate_alloc_guard_pass != (not scratch_observed):
        raise ValueError("candidate_alloc_guard_pass snapshot mismatch")
    if receipt.candidate_dense_surfaces_observed != scratch_surfaces:
        raise ValueError("candidate_dense_surfaces_observed snapshot mismatch")
    recomputed_branch = _recompute_integration_branch_id(
        attribution_subcontract_snapshot=receipt.attribution_subcontract_snapshot,
        attribution_subcontract_pass=recomputed_attribution_pass,
        ranking_subcontract_pass=recomputed_ranking_pass,
        capture_retained_fp_tensor_count=receipt.capture_retained_fp_tensor_count,
        capture_stashed_in_closure_or_registry_count=(
            receipt.capture_stashed_in_closure_or_registry_count
        ),
        comparable_set_complete=receipt.comparable_set_complete,
        partial_coverage_only=receipt.partial_coverage_only,
    )
    if receipt.branch_id != recomputed_branch:
        raise ValueError("integration branch_id mismatch vs recomputed evidence")
    if receipt.branch_id == BRANCH_D_INTEGER_VIABLE:
        if not recomputed_attribution_pass or not recomputed_ranking_pass:
            raise ValueError("INTEGER-VIABLE requires both subcontract passes")
        if not receipt.comparable_set_complete:
            raise ValueError("INTEGER-VIABLE requires comparable_set_complete")
        if receipt.integration_authority_level != INTEGRATION_AUTHORITY_CPU_EVIDENCE_ONLY:
            raise ValueError("INTEGER-VIABLE requires cpu_evidence_only authority")
