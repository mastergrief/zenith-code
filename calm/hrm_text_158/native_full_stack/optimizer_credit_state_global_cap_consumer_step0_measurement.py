"""Read-only demand projection for B2-5b Step-0 consumer measurement.

Projects global-cap demand from representative tensor inputs without mutation,
hot-loop entry, or native selector dispatch.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import torch
import torch.nn.functional as F

from calm.hrm_text_158.bit_linear import BitLinear
from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
    ACCUMULATOR_SUBSTITUTE_LOCAL_VOTE_UPDATE_EXECUTABLE,
    authoritative_forward_context,
    default_dry_run_rank_vote_spec,
    make_bounded_tensor_state,
)
from calm.hrm_text_158.native_full_stack.global_rate_cap import (
    GlobalRateCapOrderingMode,
    GlobalRateCapSpec,
    GlobalRateCapTensorInput,
    resolve_named_global_cap_spec,
    select_global_rate_cap_rows,
    tensor_offsets_for_vote_update_states,
)
from calm.hrm_text_158.native_full_stack.global_rate_cap_margin_selection_step0_budget import (
    build_upper_bound_fixture_inputs,
    expected_row_count_upper_bound,
)
from calm.hrm_text_158.native_full_stack.integer_optimizer_credit_path import (
    INTEGER_OPTIMIZER_CREDIT_PATH_WIRE_HARD_FALSE_FIELDS,
    IntegerOptimizerCreditPathWireReceipt,
    apply_integer_optimizer_credit_path_step,
    build_integer_optimizer_credit_path_wire_receipt_from_step,
    default_integer_optimizer_credit_path_vote_update_spec,
    emit_integer_sparse_vote_events_from_trainer_handle,
)
from calm.hrm_text_158.native_full_stack.optimizer_credit_state_global_cap_consumer_step0_receipt import (
    PINNED_CALL_SITE_BDL_CANDIDATE_GLOBAL_CAP_REJECT,
    PINNED_CALL_SITE_BDL_GLOBAL_CAP_REFERENCE,
    PINNED_CALL_SITE_GRC_SPEC_DEFAULT_MARGIN,
    PINNED_CALL_SITE_GRC_STEP_SUMMARY_SURFACE,
    PINNED_CALL_SITE_IOCP_DEFAULT_OFF_FLAG,
    PINNED_CALL_SITE_IOCP_RECEIPT_FLAGS,
    PINNED_CALL_SITE_IOCP_SPARSE_EMIT_STEP,
    ConsumerStep0FixtureMeasurement,
)

C1_BANKED_FAITHFUL_LONG_RUN_GLOBAL_CAP_CONTRACT_NAME = (
    "c1_banked_faithful_long_run_global_cap"
)

IOCP_DEFAULT_OFF_SOURCE = "integer_optimizer_credit_path.py:45-67"
IOCP_SPARSE_EMIT_SOURCE = "integer_optimizer_credit_path.py:366-380"
IOCP_RECEIPT_FLAGS_SOURCE = "integer_optimizer_credit_path.py:241-263"
BDL_GLOBAL_CAP_REF_SOURCE = "bounded_delta_learner.py:1900-1914"
BDL_CANDIDATE_REJECT_SOURCE = "bounded_delta_learner.py:1639-1649"
GRC_STEP_SUMMARY_SOURCE = "global_rate_cap.py:803-850"
GRC_SPEC_DEFAULT_MARGIN_SOURCE = "global_rate_cap.py:50-60,194-207"


@dataclass(frozen=True)
class _MinimalIntegerWireBundle:
    total_sparse_event_count: int
    wire_receipt: IntegerOptimizerCreditPathWireReceipt


class _MinimalIntegerWireModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = BitLinear(3, 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


@lru_cache(maxsize=1)
def _minimal_integer_wire_bundle() -> _MinimalIntegerWireBundle:
    """CPU-only synthetic fixture mirroring integer_optimizer_credit_path wire tests."""

    torch.manual_seed(158)
    model = _MinimalIntegerWireModel()
    with torch.no_grad():
        model.proj.weight.zero_()
    eligible = {"proj": model.proj}
    q = torch.zeros_like(model.proj.weight.detach(), dtype=torch.int8)
    state = make_bounded_tensor_state(
        "proj",
        q,
        torch.tensor(1.0, dtype=torch.float32),
        hot_exact_indices=tuple(range(int(q.numel()))),
    )
    states = {"proj": state}
    x = torch.tensor([[1.0, -2.0, 3.0]], dtype=torch.float32)
    target = torch.tensor([[2.0, -1.0]], dtype=torch.float32)
    model.zero_grad(set_to_none=True)
    with authoritative_forward_context(
        eligible,
        states,
        device="cpu",
        requires_grad=True,
    ) as handle:
        out = model(x)
        loss = F.mse_loss(out, target)
        loss.backward()
        sparse_events_by_key = emit_integer_sparse_vote_events_from_trainer_handle(
            handle,
            states,
            default_dry_run_rank_vote_spec(),
        )
    vote_spec = default_integer_optimizer_credit_path_vote_update_spec()
    step_result = apply_integer_optimizer_credit_path_step(
        states,
        sparse_events_by_key,
        {"proj": vote_spec},
    )
    wire_receipt = build_integer_optimizer_credit_path_wire_receipt_from_step(
        step_result=step_result,
        sparse_events_by_key=sparse_events_by_key,
    )
    total_sparse_event_count = int(wire_receipt.total_sparse_event_count)
    if total_sparse_event_count <= 0:
        raise RuntimeError("minimal integer wire fixture must emit non-zero sparse events")
    for field in INTEGER_OPTIMIZER_CREDIT_PATH_WIRE_HARD_FALSE_FIELDS:
        if bool(getattr(wire_receipt, field)):
            raise RuntimeError(f"wire receipt hard-false field {field!r} must be False")
    return _MinimalIntegerWireBundle(
        total_sparse_event_count=total_sparse_event_count,
        wire_receipt=wire_receipt,
    )


def _path_a_integer_wire_measurement_row(
    *,
    fixture_name: str,
    pinned_call_site_id: str,
    source_anchor: str,
    ordering_mode_source: str,
    total_sparse_event_count: int,
) -> ConsumerStep0FixtureMeasurement:
    return ConsumerStep0FixtureMeasurement(
        fixture_name=fixture_name,
        fixture_role="representative_consumer",
        pinned_call_site_id=pinned_call_site_id,
        source_anchor=source_anchor,
        consumer_path_class="PATH_A_INTEGER_WIRE",
        candidate_mode_class="CANDIDATE_ONLY",
        total_sparse_event_count=total_sparse_event_count,
        projected_full_demand_count=0,
        projected_global_pre_cap_would_apply_count=0,
        max_row_count=0,
        ordering_mode="margin",
        ordering_mode_source=ordering_mode_source,
        cap=0,
        deferred_count=0,
        saturation_observed=False,
        candidate_rejects_global_cap=True,
        seam_resolved=False,
    )


@dataclass(frozen=True)
class ConsumerStep0DemandProjection:
    projected_full_demand_count: int
    projected_global_pre_cap_would_apply_count: int
    max_row_count: int
    ordering_mode: str
    ordering_mode_source: str
    cap: int
    deferred_count: int
    saturation_observed: bool


def build_path_b_representative_inputs(*, target_row_count: int) -> list[GlobalRateCapTensorInput]:
    if target_row_count <= 0 or target_row_count % 256 != 0:
        raise ValueError("target_row_count must be a positive multiple of 256")
    per_state = target_row_count // 256
    inputs = build_upper_bound_fixture_inputs(
        numel=256,
        max_abs_per_tensor=256,
        num_states=per_state,
    )
    expected = expected_row_count_upper_bound(
        numel=256,
        max_abs_per_tensor=256,
        num_states=per_state,
    )
    if expected != target_row_count:
        raise ValueError(
            f"fixture builder mismatch: expected {target_row_count}, got {expected}"
        )
    return inputs


def project_global_cap_demand(
    inputs: list[GlobalRateCapTensorInput],
    spec: GlobalRateCapSpec,
) -> ConsumerStep0DemandProjection:
    spec.validate()
    offsets = tensor_offsets_for_vote_update_states(inputs)
    rows, accepted, deferred = select_global_rate_cap_rows(
        inputs,
        spec,
        tensor_offsets=offsets,
    )
    demand = len(rows)
    cap = int(spec.cap)
    return ConsumerStep0DemandProjection(
        projected_full_demand_count=demand,
        projected_global_pre_cap_would_apply_count=demand,
        max_row_count=demand,
        ordering_mode=spec.normalized_ordering_mode.value,
        ordering_mode_source="GlobalRateCapSpec",
        cap=cap,
        deferred_count=len(deferred),
        saturation_observed=demand > cap,
    )


def _path_b_measurement_row(
    *,
    fixture_name: str,
    fixture_role: str,
    pinned_call_site_id: str,
    source_anchor: str,
    target_row_count: int,
    cap: int,
    ordering_mode: GlobalRateCapOrderingMode = GlobalRateCapOrderingMode.MARGIN,
    step: int = 1,
) -> ConsumerStep0FixtureMeasurement:
    inputs = build_path_b_representative_inputs(target_row_count=target_row_count)
    spec = GlobalRateCapSpec(cap=cap, step=step, ordering_mode=ordering_mode)
    projection = project_global_cap_demand(inputs, spec)
    return ConsumerStep0FixtureMeasurement(
        fixture_name=fixture_name,
        fixture_role=fixture_role,  # type: ignore[arg-type]
        pinned_call_site_id=pinned_call_site_id,
        source_anchor=source_anchor,
        consumer_path_class="PATH_B_GLOBAL_CAP_REFERENCE",
        candidate_mode_class="NON_CANDIDATE_GLOBAL_CAP_REFERENCE",
        total_sparse_event_count=0,
        projected_full_demand_count=projection.projected_full_demand_count,
        projected_global_pre_cap_would_apply_count=projection.projected_global_pre_cap_would_apply_count,
        max_row_count=projection.max_row_count,
        ordering_mode=projection.ordering_mode,
        ordering_mode_source=projection.ordering_mode_source,
        cap=projection.cap,
        deferred_count=projection.deferred_count,
        saturation_observed=projection.saturation_observed,
        candidate_rejects_global_cap=False,
        seam_resolved=True,
    )


def measure_iocp_default_off() -> ConsumerStep0FixtureMeasurement:
    return ConsumerStep0FixtureMeasurement(
        fixture_name="F_IOCP_DEFAULT_OFF",
        fixture_role="representative_consumer",
        pinned_call_site_id=PINNED_CALL_SITE_IOCP_DEFAULT_OFF_FLAG,
        source_anchor=IOCP_DEFAULT_OFF_SOURCE,
        consumer_path_class="PATH_A_INTEGER_WIRE",
        candidate_mode_class="CANDIDATE_ONLY",
        total_sparse_event_count=0,
        projected_full_demand_count=0,
        projected_global_pre_cap_would_apply_count=0,
        max_row_count=0,
        ordering_mode="margin",
        ordering_mode_source="structural_default_off_wire",
        cap=0,
        deferred_count=0,
        saturation_observed=False,
        candidate_rejects_global_cap=True,
        seam_resolved=False,
    )


def measure_iocp_sparse_emit_step() -> ConsumerStep0FixtureMeasurement:
    bundle = _minimal_integer_wire_bundle()
    return _path_a_integer_wire_measurement_row(
        fixture_name="F_IOCP_SPARSE_STEP",
        pinned_call_site_id=PINNED_CALL_SITE_IOCP_SPARSE_EMIT_STEP,
        source_anchor=IOCP_SPARSE_EMIT_SOURCE,
        ordering_mode_source="integer_wire_sparse_emit_path",
        total_sparse_event_count=bundle.total_sparse_event_count,
    )


def measure_iocp_receipt_flags() -> ConsumerStep0FixtureMeasurement:
    bundle = _minimal_integer_wire_bundle()
    return _path_a_integer_wire_measurement_row(
        fixture_name="F_IOCP_RECEIPT_FLAGS",
        pinned_call_site_id=PINNED_CALL_SITE_IOCP_RECEIPT_FLAGS,
        source_anchor=IOCP_RECEIPT_FLAGS_SOURCE,
        ordering_mode_source="integer_wire_receipt_surface",
        total_sparse_event_count=bundle.wire_receipt.total_sparse_event_count,
    )


def measure_bdl_candidate_global_cap_reject() -> ConsumerStep0FixtureMeasurement:
    return ConsumerStep0FixtureMeasurement(
        fixture_name="F_BDL_CANDIDATE_REJECT",
        fixture_role="representative_consumer",
        pinned_call_site_id=PINNED_CALL_SITE_BDL_CANDIDATE_GLOBAL_CAP_REJECT,
        source_anchor=BDL_CANDIDATE_REJECT_SOURCE,
        consumer_path_class="STRUCTURAL_SEAM_FACT",
        candidate_mode_class="STRUCTURAL_REJECTION",
        total_sparse_event_count=0,
        projected_full_demand_count=0,
        projected_global_pre_cap_would_apply_count=0,
        max_row_count=0,
        ordering_mode="margin",
        ordering_mode_source="bounded_delta_learner_candidate_guard",
        cap=0,
        deferred_count=0,
        saturation_observed=False,
        candidate_rejects_global_cap=True,
        seam_resolved=False,
    )


def measure_bdl_global_cap_reference(*, target_row_count: int = 512, cap: int = 256) -> ConsumerStep0FixtureMeasurement:
    return _path_b_measurement_row(
        fixture_name="F_BDL_GLOBAL_CAP_REF",
        fixture_role="representative_consumer",
        pinned_call_site_id=PINNED_CALL_SITE_BDL_GLOBAL_CAP_REFERENCE,
        source_anchor=BDL_GLOBAL_CAP_REF_SOURCE,
        target_row_count=target_row_count,
        cap=cap,
    )


def measure_grc_step_summary_surface(*, target_row_count: int = 512, cap: int = 256) -> ConsumerStep0FixtureMeasurement:
    return _path_b_measurement_row(
        fixture_name="F_GRC_STEP_SUMMARY",
        fixture_role="representative_consumer",
        pinned_call_site_id=PINNED_CALL_SITE_GRC_STEP_SUMMARY_SURFACE,
        source_anchor=GRC_STEP_SUMMARY_SOURCE,
        target_row_count=target_row_count,
        cap=cap,
    )


def measure_grc_spec_default_margin() -> ConsumerStep0FixtureMeasurement:
    spec = resolve_named_global_cap_spec(
        C1_BANKED_FAITHFUL_LONG_RUN_GLOBAL_CAP_CONTRACT_NAME,
        step=1,
    )
    if spec is None:
        raise RuntimeError("expected non-null global cap spec for faithful long-run contract")
    return ConsumerStep0FixtureMeasurement(
        fixture_name="F_GRC_MARGIN_DEFAULT",
        fixture_role="representative_consumer",
        pinned_call_site_id=PINNED_CALL_SITE_GRC_SPEC_DEFAULT_MARGIN,
        source_anchor=GRC_SPEC_DEFAULT_MARGIN_SOURCE,
        consumer_path_class="GRC_SPEC_SURFACE",
        candidate_mode_class="NOT_APPLICABLE",
        total_sparse_event_count=0,
        projected_full_demand_count=0,
        projected_global_pre_cap_would_apply_count=0,
        max_row_count=0,
        ordering_mode=spec.normalized_ordering_mode.value,
        ordering_mode_source="resolve_named_global_cap_spec",
        cap=int(spec.cap),
        deferred_count=0,
        saturation_observed=False,
        candidate_rejects_global_cap=False,
        seam_resolved=True,
    )


def measure_margin_at_1280() -> ConsumerStep0FixtureMeasurement:
    return _path_b_measurement_row(
        fixture_name="F_MARGIN_AT_1280",
        fixture_role="representative_consumer",
        pinned_call_site_id=PINNED_CALL_SITE_BDL_GLOBAL_CAP_REFERENCE,
        source_anchor=BDL_GLOBAL_CAP_REF_SOURCE,
        target_row_count=1280,
        cap=512,
    )


def measure_margin_at_2048() -> ConsumerStep0FixtureMeasurement:
    return _path_b_measurement_row(
        fixture_name="F_MARGIN_AT_2048",
        fixture_role="representative_consumer",
        pinned_call_site_id=PINNED_CALL_SITE_BDL_GLOBAL_CAP_REFERENCE,
        source_anchor=BDL_GLOBAL_CAP_REF_SOURCE,
        target_row_count=2048,
        cap=1024,
    )


def measure_classifier_negative_over_ceiling() -> ConsumerStep0FixtureMeasurement:
    return ConsumerStep0FixtureMeasurement(
        fixture_name="F_NEG_OVER_CEILING",
        fixture_role="classifier_negative",
        pinned_call_site_id=PINNED_CALL_SITE_BDL_GLOBAL_CAP_REFERENCE,
        source_anchor=BDL_GLOBAL_CAP_REF_SOURCE,
        consumer_path_class="CLASSIFIER_NEGATIVE_PROBE",
        candidate_mode_class="NON_CANDIDATE_GLOBAL_CAP_REFERENCE",
        total_sparse_event_count=0,
        projected_full_demand_count=3000,
        projected_global_pre_cap_would_apply_count=3000,
        max_row_count=3000,
        ordering_mode="margin",
        ordering_mode_source="classifier_negative_probe",
        cap=512,
        deferred_count=2488,
        saturation_observed=True,
        candidate_rejects_global_cap=False,
        seam_resolved=True,
    )


def measure_classifier_negative_non_margin() -> ConsumerStep0FixtureMeasurement:
    return _path_b_measurement_row(
        fixture_name="F_NEG_NON_MARGIN",
        fixture_role="classifier_negative",
        pinned_call_site_id=PINNED_CALL_SITE_GRC_STEP_SUMMARY_SURFACE,
        source_anchor=GRC_STEP_SUMMARY_SOURCE,
        target_row_count=512,
        cap=256,
        ordering_mode=GlobalRateCapOrderingMode.HASH_SHUFFLE,
    )


def build_representative_consumer_measurements() -> tuple[ConsumerStep0FixtureMeasurement, ...]:
    return (
        measure_iocp_default_off(),
        measure_iocp_sparse_emit_step(),
        measure_iocp_receipt_flags(),
        measure_bdl_global_cap_reference(),
        measure_bdl_candidate_global_cap_reject(),
        measure_grc_step_summary_surface(),
        measure_grc_spec_default_margin(),
        measure_margin_at_1280(),
        measure_margin_at_2048(),
    )


def build_classifier_negative_measurements() -> tuple[ConsumerStep0FixtureMeasurement, ...]:
    return (
        measure_classifier_negative_over_ceiling(),
        measure_classifier_negative_non_margin(),
    )


def candidate_mode_would_reject_global_cap() -> bool:
    """Structural fact from bounded_delta_learner candidate guard."""

    _ = ACCUMULATOR_SUBSTITUTE_LOCAL_VOTE_UPDATE_EXECUTABLE
    return True
