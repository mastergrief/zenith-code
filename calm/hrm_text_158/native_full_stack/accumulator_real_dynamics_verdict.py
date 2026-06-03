"""Assumption-bound accumulator verdict for the C1.1b density harness.

This module is measurement/reporting glue only. It builds a deterministic,
large-N in-tree native-loop distribution and labels its bindingness before
looking at the density numbers, so a favorable result cannot be upgraded into a
live-regime claim.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from calm.hrm_text_158.native_full_stack.accumulator_compression import (
    CandidateClassification,
    required_decision_dimension_names,
)
from calm.hrm_text_158.native_full_stack.accumulator_decision_density import (
    DECISION_EXACT_INFEASIBLE,
    AccumulatorCandidateClassification,
    AccumulatorDecisionDensityInput,
    AccumulatorDecisionDensityReport,
    classify_accumulator_candidate,
    measure_accumulator_decision_density,
    validate_accumulator_decision_density_report,
)
from calm.hrm_text_158.native_full_stack.global_rate_cap import (
    GlobalRateCapSpec,
    GlobalRateCapTensorInput,
    apply_global_rate_cap_reference,
    tensor_offsets_for_vote_update_states,
)
from calm.hrm_text_158.native_full_stack.vote_update import (
    VoteUpdateInputs,
    VoteUpdateSpec,
    VoteUpdateState,
    plan_integer_vote_update_reference,
)


ACCUMULATOR_ASSUMPTION_BOUND_VERDICT_SCHEMA_VERSION = (
    "hrm_text_158_accumulator_assumption_bound_engineering_verdict/v0"
)
ASSUMPTION_BOUND_ENGINEERING_VERDICT = "assumption_bound_engineering_verdict"
BINDING_FOR_IN_TREE_NATIVE_LOOP_DISTRIBUTION = (
    "binding_for_in_tree_native_loop_distribution"
)
PARTIAL_FOR_S1_REAL_DYNAMICS = "partial_for_s1_real_dynamics"
PARTIAL_EVIDENCE_ONLY = "partial_evidence_only"
BINDING_S1_REGIME_VERDICT = "binding_s1_regime_verdict"
RAW_STATE_CAPTURE_SCHEMA_VERSION = "hrm_text_158_native_live_regime_raw_capture/v0"

SOURCE_KIND_GENERATED_NATIVE_LOOP = "generated_in_tree_native_loop"
SOURCE_KIND_COMPACT_FULL_LOOP_ARTIFACT = "compact_full_loop_artifact"
SOURCE_KIND_RAW_S1_TELEMETRY = "raw_s1_telemetry"

C1_1C_ROUTE_SPARSE_EXACT = "sparse_exact_decision_set"
C1_1C_ROUTE_BOUNDED_DELTA_WITH_REPORT = "bounded_delta_with_report"

PRIMARY_ELIGIBLE_WEIGHT_COUNT = 16_384
PRIMARY_TENSOR_NUMEL = PRIMARY_ELIGIBLE_WEIGHT_COUNT // 2
PRIMARY_STATE_KEYS = ("proj_in", "proj_out")
FORBIDDEN_GENERATED_SERIALIZATION_TERMS = (
    "real-dynamics verdict",
    BINDING_S1_REGIME_VERDICT,
)

CLASSIFIER_COVERAGE_FIELDS = (
    "active_near_threshold",
    "ranking_under_cap",
    "backlog_state_carry",
    "vote_threshold_decay",
    "n_scale",
)
RAW_CAPTURE_COVERAGE_FIELDS = (
    "raw_q_acc_state",
    "raw_votes",
    "cap_selected_deferred_rows",
    "deferred_backlog",
    "source_hashes_timestamps",
)


@dataclass(frozen=True)
class SourceFieldCoverage:
    """Dimension coverage map used before a verdict label is assigned."""

    active_near_threshold: bool
    ranking_under_cap: bool
    backlog_state_carry: bool
    vote_threshold_decay: bool
    n_scale: bool
    raw_q_acc_state: bool = False
    raw_votes: bool = False
    cap_selected_deferred_rows: bool = False
    deferred_backlog: bool = False
    source_hashes_timestamps: bool = False

    @classmethod
    def full_generated_native_loop(cls) -> "SourceFieldCoverage":
        return cls(
            active_near_threshold=True,
            ranking_under_cap=True,
            backlog_state_carry=True,
            vote_threshold_decay=True,
            n_scale=True,
            raw_q_acc_state=True,
            raw_votes=True,
            cap_selected_deferred_rows=True,
            deferred_backlog=True,
            source_hashes_timestamps=False,
        )

    @classmethod
    def full_raw_s1_telemetry(cls) -> "SourceFieldCoverage":
        return cls(
            active_near_threshold=True,
            ranking_under_cap=True,
            backlog_state_carry=True,
            vote_threshold_decay=True,
            n_scale=True,
            raw_q_acc_state=True,
            raw_votes=True,
            cap_selected_deferred_rows=True,
            deferred_backlog=True,
            source_hashes_timestamps=True,
        )

    @classmethod
    def compact_artifact_only(cls) -> "SourceFieldCoverage":
        return cls(
            active_near_threshold=False,
            ranking_under_cap=False,
            backlog_state_carry=False,
            vote_threshold_decay=False,
            n_scale=False,
        )

    @property
    def covers_classifier_dimensions(self) -> bool:
        return all(bool(getattr(self, field)) for field in CLASSIFIER_COVERAGE_FIELDS)

    @property
    def covers_raw_capture_dimensions(self) -> bool:
        return all(bool(getattr(self, field)) for field in RAW_CAPTURE_COVERAGE_FIELDS)

    @property
    def can_claim_binding_live_regime(self) -> bool:
        return self.covers_classifier_dimensions and self.covers_raw_capture_dimensions

    def missing_classifier_dimensions(self) -> tuple[str, ...]:
        return tuple(field for field in CLASSIFIER_COVERAGE_FIELDS if not getattr(self, field))

    def missing_raw_capture_dimensions(self) -> tuple[str, ...]:
        return tuple(field for field in RAW_CAPTURE_COVERAGE_FIELDS if not getattr(self, field))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["covers_classifier_dimensions"] = self.covers_classifier_dimensions
        payload["covers_raw_capture_dimensions"] = self.covers_raw_capture_dimensions
        payload["can_claim_binding_live_regime"] = self.can_claim_binding_live_regime
        payload["missing_classifier_dimensions"] = list(self.missing_classifier_dimensions())
        payload["missing_raw_capture_dimensions"] = list(self.missing_raw_capture_dimensions())
        return payload


@dataclass(frozen=True)
class SourceBindingness:
    """Pre-registered evidence label derived from source kind and fields."""

    source_kind: str
    primary_bindingness: str
    s1_bindingness: str
    evidence_class: str
    can_claim_binding_live_regime: bool
    missing_fields: tuple[str, ...]
    note: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["missing_fields"] = list(self.missing_fields)
        return payload


@dataclass(frozen=True)
class VotePressureStepSpec:
    """One fixed step in the anti-rigging vote-pressure schedule."""

    name: str
    step: int
    rows_per_tensor: int
    start_index: int
    vote_abs: int
    cap: int
    expected_regime: str

    def to_dict(self) -> dict[str, int | str]:
        return asdict(self)


PRE_REGISTERED_VOTE_PRESSURE_SCHEDULE: tuple[VotePressureStepSpec, ...] = (
    VotePressureStepSpec(
        name="sparse_unsaturated",
        step=1,
        rows_per_tensor=32,
        start_index=0,
        vote_abs=12,
        cap=512,
        expected_regime="sparse below floor; cap not saturated",
    ),
    VotePressureStepSpec(
        name="moderate_unsaturated",
        step=2,
        rows_per_tensor=160,
        start_index=512,
        vote_abs=16,
        cap=512,
        expected_regime="moderate pressure; cap not saturated but overhead may exceed budget",
    ),
    VotePressureStepSpec(
        name="cap_saturated",
        step=3,
        rows_per_tensor=768,
        start_index=2_048,
        vote_abs=24,
        cap=256,
        expected_regime="cap saturated; deferred backlog is created",
    ),
    VotePressureStepSpec(
        name="backlog_growth",
        step=4,
        rows_per_tensor=768,
        start_index=4_096,
        vote_abs=24,
        cap=256,
        expected_regime="cap saturated again; prior deferred backlog carries forward",
    ),
)


@dataclass(frozen=True)
class FileIntegritySnapshot:
    """Read-only source integrity proof for optional cross-tree telemetry."""

    path: str
    sha256: str
    size_bytes: int
    mtime_ns: int

    def to_dict(self) -> dict[str, str | int]:
        return asdict(self)


@dataclass(frozen=True)
class NativeLoopVerdictStep:
    """Compact per-step verdict row; the density report contains no raw arrays."""

    schedule_name: str
    step: int
    expected_regime: str
    cap: int
    rows_per_tensor: int
    vote_abs: int
    vote_nonzero_count: int
    vote_abs_max: int
    threshold_abs: int
    decay_numerator: int
    decay_denominator: int
    clip_min: int
    clip_max: int
    eligible_weight_count: int
    global_pre_cap_would_apply_count: int
    global_cap_saturated: bool
    global_cap_accepted_count: int
    global_cap_deferred_count: int
    backlog_state_carry_count: int
    decision_relevant_exact_density: float
    projected_bits_per_weight: float
    target_bits_per_weight: float
    sparse_classification: AccumulatorCandidateClassification
    c1_1c_step_route: str
    density_report: AccumulatorDecisionDensityReport

    def to_dict(self) -> dict[str, Any]:
        return {
            "schedule_name": self.schedule_name,
            "step": self.step,
            "expected_regime": self.expected_regime,
            "cap": self.cap,
            "rows_per_tensor": self.rows_per_tensor,
            "vote_abs": self.vote_abs,
            "vote_nonzero_count": self.vote_nonzero_count,
            "vote_abs_max": self.vote_abs_max,
            "threshold_abs": self.threshold_abs,
            "decay_numerator": self.decay_numerator,
            "decay_denominator": self.decay_denominator,
            "clip_min": self.clip_min,
            "clip_max": self.clip_max,
            "eligible_weight_count": self.eligible_weight_count,
            "global_pre_cap_would_apply_count": self.global_pre_cap_would_apply_count,
            "global_cap_saturated": self.global_cap_saturated,
            "global_cap_accepted_count": self.global_cap_accepted_count,
            "global_cap_deferred_count": self.global_cap_deferred_count,
            "backlog_state_carry_count": self.backlog_state_carry_count,
            "decision_relevant_exact_density": self.decision_relevant_exact_density,
            "projected_bits_per_weight": self.projected_bits_per_weight,
            "target_bits_per_weight": self.target_bits_per_weight,
            "sparse_classification": self.sparse_classification.to_dict(),
            "c1_1c_step_route": self.c1_1c_step_route,
            "density_report": self.density_report.to_dict(),
        }


@dataclass(frozen=True)
class AccumulatorAssumptionBoundVerdictReport:
    """Compact terminal verdict for the generated native-loop distribution."""

    schema_version: str
    label: str
    source_name: str
    source_kind: str
    source_bindingness: SourceBindingness
    field_coverage: SourceFieldCoverage
    raw_state_capture_schema: dict[str, Any]
    vote_pressure_summary: dict[str, Any]
    representativeness_argument: str
    representativeness_limits: tuple[str, ...]
    pre_registered_schedule: tuple[VotePressureStepSpec, ...]
    per_step_reports: tuple[NativeLoopVerdictStep, ...]
    terminal_decision: str
    c1_1c_route: str
    raw_arrays_included: bool
    non_claims: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "label": self.label,
            "source_name": self.source_name,
            "source_kind": self.source_kind,
            "source_bindingness": self.source_bindingness.to_dict(),
            "field_coverage": self.field_coverage.to_dict(),
            "raw_state_capture_schema": self.raw_state_capture_schema,
            "vote_pressure_summary": self.vote_pressure_summary,
            "representativeness_argument": self.representativeness_argument,
            "representativeness_limits": list(self.representativeness_limits),
            "pre_registered_schedule": [
                step.to_dict() for step in self.pre_registered_schedule
            ],
            "per_step_reports": [step.to_dict() for step in self.per_step_reports],
            "terminal_decision": self.terminal_decision,
            "c1_1c_route": self.c1_1c_route,
            "raw_arrays_included": self.raw_arrays_included,
            "non_claims": list(self.non_claims),
        }


def raw_state_capture_schema() -> dict[str, Any]:
    """Return the future capture fields needed for a live S1/native verdict."""

    return {
        "schema_version": RAW_STATE_CAPTURE_SCHEMA_VERSION,
        "purpose": (
            "upgrade this assumption-bound engineering evidence into a true "
            "S1/native live-regime measurement"
        ),
        "required_fields": [
            {
                "name": "source_identity_integrity",
                "description": "path, sha256, size_bytes, mtime_ns before and after read",
            },
            {
                "name": "q_acc_state",
                "description": "raw or exactly reconstructible q:int8 levels and acc:int16 tensors",
            },
            {
                "name": "votes",
                "description": "raw or exactly reconstructible vote:int16 tensors per step",
            },
            {
                "name": "vote_update_spec",
                "description": "threshold_abs, decay numerator/denominator, clip, tensor cap settings",
            },
            {
                "name": "global_cap_rows_or_inputs",
                "description": (
                    "selected/deferred rows or the complete inputs needed to reproduce "
                    "global-cap ordering"
                ),
            },
            {
                "name": "deferred_backlog",
                "description": "per-state deferred rows with first/last step and defer counts",
            },
            {
                "name": "n_scale",
                "description": "eligible weight count and tensor shapes for the measured surface",
            },
        ],
        "compact_report_policy": (
            "receipts may emit counts, hashes, and schema metadata; raw per-weight "
            "arrays stay in the capture source"
        ),
    }


def pre_register_source_bindingness(
    *,
    source_kind: str,
    coverage: SourceFieldCoverage,
) -> SourceBindingness:
    """Assign bindingness from source coverage before density numbers exist."""

    classifier_missing = coverage.missing_classifier_dimensions()
    raw_missing = coverage.missing_raw_capture_dimensions()

    if source_kind == SOURCE_KIND_GENERATED_NATIVE_LOOP:
        if classifier_missing:
            return SourceBindingness(
                source_kind=source_kind,
                primary_bindingness=PARTIAL_EVIDENCE_ONLY,
                s1_bindingness=PARTIAL_EVIDENCE_ONLY,
                evidence_class=PARTIAL_EVIDENCE_ONLY,
                can_claim_binding_live_regime=False,
                missing_fields=classifier_missing,
                note="generated source is incomplete for the C1.1b classifier dimensions",
            )
        return SourceBindingness(
            source_kind=source_kind,
            primary_bindingness=BINDING_FOR_IN_TREE_NATIVE_LOOP_DISTRIBUTION,
            s1_bindingness=PARTIAL_FOR_S1_REAL_DYNAMICS,
            evidence_class=ASSUMPTION_BOUND_ENGINEERING_VERDICT,
            can_claim_binding_live_regime=False,
            missing_fields=raw_missing,
            note=(
                "generated distribution covers the classifier dimensions in memory, "
                "but it is assumption-bound and not a raw S1 telemetry source"
            ),
        )

    if source_kind == SOURCE_KIND_RAW_S1_TELEMETRY:
        if coverage.can_claim_binding_live_regime:
            return SourceBindingness(
                source_kind=source_kind,
                primary_bindingness=BINDING_S1_REGIME_VERDICT,
                s1_bindingness=BINDING_S1_REGIME_VERDICT,
                evidence_class=BINDING_S1_REGIME_VERDICT,
                can_claim_binding_live_regime=True,
                missing_fields=(),
                note="raw S1 telemetry covers classifier and integrity dimensions",
            )
        return SourceBindingness(
            source_kind=source_kind,
            primary_bindingness=PARTIAL_EVIDENCE_ONLY,
            s1_bindingness=PARTIAL_EVIDENCE_ONLY,
            evidence_class=PARTIAL_EVIDENCE_ONLY,
            can_claim_binding_live_regime=False,
            missing_fields=classifier_missing + raw_missing,
            note="raw S1 telemetry is incomplete for a binding live-regime claim",
        )

    if source_kind == SOURCE_KIND_COMPACT_FULL_LOOP_ARTIFACT:
        return SourceBindingness(
            source_kind=source_kind,
            primary_bindingness=PARTIAL_EVIDENCE_ONLY,
            s1_bindingness=PARTIAL_EVIDENCE_ONLY,
            evidence_class=PARTIAL_EVIDENCE_ONLY,
            can_claim_binding_live_regime=False,
            missing_fields=classifier_missing + raw_missing,
            note=(
                "compact full-loop receipts expose counts/hashes but not raw "
                "q/acc/vote/cap/backlog state"
            ),
        )

    raise ValueError(f"unknown source_kind {source_kind!r}")


def _contains_key(payload: Any, names: set[str]) -> bool:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            if str(key) in names or _contains_key(value, names):
                return True
    elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        return any(_contains_key(item, names) for item in payload)
    return False


def classify_compact_full_loop_artifact_payload(payload: Mapping[str, Any]) -> SourceBindingness:
    """Classify compact receipts as partial unless raw reconstructible state exists."""

    coverage = SourceFieldCoverage(
        active_near_threshold=_contains_key(payload, {"active_next_step_count", "candidate_count"}),
        ranking_under_cap=_contains_key(
            payload,
            {
                "global_rate_cap_accepted_count",
                "global_rate_cap_deferred_count",
                "global_cap_counts_match_cpu_oracle",
            },
        ),
        backlog_state_carry=_contains_key(
            payload,
            {"deferred_backlog", "global_cap_deferred_backlog_size"},
        ),
        vote_threshold_decay=_contains_key(payload, {"threshold_abs", "decay", "votes"}),
        n_scale=_contains_key(payload, {"eligible_weight_count", "tensor_shapes"}),
        raw_q_acc_state=_contains_key(payload, {"q_levels", "accumulators", "q_acc_state"}),
        raw_votes=_contains_key(payload, {"votes", "vote_tensors"}),
        cap_selected_deferred_rows=_contains_key(
            payload,
            {"global_cap_accepted_indices", "global_cap_deferred_indices"},
        ),
        deferred_backlog=_contains_key(payload, {"deferred_backlog"}),
        source_hashes_timestamps=_contains_key(payload, {"sha256", "mtime_ns", "size_bytes"}),
    )
    return pre_register_source_bindingness(
        source_kind=SOURCE_KIND_COMPACT_FULL_LOOP_ARTIFACT,
        coverage=coverage,
    )


def capture_file_integrity(path: str | Path) -> FileIntegritySnapshot:
    """Read a file without mutation and capture sha256/mtime/size metadata."""

    target = Path(path)
    stat = target.stat()
    h = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return FileIntegritySnapshot(
        path=str(target),
        sha256=h.hexdigest(),
        size_bytes=int(stat.st_size),
        mtime_ns=int(stat.st_mtime_ns),
    )


def assert_file_integrity_unchanged(
    before: FileIntegritySnapshot,
    after: FileIntegritySnapshot,
) -> None:
    """Raise if a read-only telemetry path changed during inspection."""

    if before != after:
        raise ValueError(
            "read-only telemetry integrity changed: "
            f"before={before.to_dict()} after={after.to_dict()}"
        )


def default_vote_update_spec() -> VoteUpdateSpec:
    return VoteUpdateSpec(
        threshold_abs=10,
        accumulator_clip_min=-127,
        accumulator_clip_max=127,
        decay_numerator=1,
        decay_denominator=1,
        max_abs_per_tensor=PRIMARY_TENSOR_NUMEL,
        fraction_per_tensor=1.0,
    )


def _initial_states() -> dict[str, VoteUpdateState]:
    return {
        key: VoteUpdateState(
            q_levels=torch.zeros(PRIMARY_TENSOR_NUMEL, dtype=torch.int8),
            accumulators=torch.zeros(PRIMARY_TENSOR_NUMEL, dtype=torch.int16),
        )
        for key in PRIMARY_STATE_KEYS
    }


def _votes_for_schedule_step(
    schedule_step: VotePressureStepSpec,
    *,
    tensor_index: int,
    tensor_numel: int = PRIMARY_TENSOR_NUMEL,
) -> torch.Tensor:
    count = int(schedule_step.rows_per_tensor)
    start = int(schedule_step.start_index)
    if count <= 0:
        raise ValueError("rows_per_tensor must be > 0")
    if start < 0 or start + count > int(tensor_numel):
        raise ValueError(
            f"vote range {start}:{start + count} exceeds tensor_numel={tensor_numel}"
        )
    indices = torch.arange(start, start + count, dtype=torch.int64)
    signs = torch.where(
        ((indices + int(schedule_step.step) + int(tensor_index)) % 2) == 0,
        1,
        -1,
    ).to(torch.int16)
    votes = torch.zeros(int(tensor_numel), dtype=torch.int16)
    votes[indices] = signs * int(schedule_step.vote_abs)
    return votes


def _density_inputs_for_step(
    states: Mapping[str, VoteUpdateState],
    schedule_step: VotePressureStepSpec,
    spec: VoteUpdateSpec,
) -> tuple[AccumulatorDecisionDensityInput, ...]:
    inputs: list[AccumulatorDecisionDensityInput] = []
    for tensor_index, state_key in enumerate(PRIMARY_STATE_KEYS):
        inputs.append(
            AccumulatorDecisionDensityInput(
                state_key=state_key,
                state=states[state_key],
                vote_inputs=VoteUpdateInputs(
                    votes=_votes_for_schedule_step(
                        schedule_step,
                        tensor_index=tensor_index,
                        tensor_numel=states[state_key].q_levels.numel(),
                    ),
                ),
                spec=spec,
            )
        )
    return tuple(inputs)


def _cap_inputs_for_density_inputs(
    inputs: Iterable[AccumulatorDecisionDensityInput],
) -> list[GlobalRateCapTensorInput]:
    return [
        GlobalRateCapTensorInput(
            state_key=item.state_key,
            state=item.state,
            plan=plan_integer_vote_update_reference(item.state, item.vote_inputs, item.spec),
        )
        for item in inputs
    ]


def _classify_sparse_step(
    report: AccumulatorDecisionDensityReport,
) -> AccumulatorCandidateClassification:
    return classify_accumulator_candidate(
        candidate_name=C1_1C_ROUTE_SPARSE_EXACT,
        classification=CandidateClassification.DECISION_EXACT,
        projection=report.sparse_exact_projection,
        covered_decision_dimensions=required_decision_dimension_names(),
        note="exact int16 for the generated decision-relevant rows only",
    )


def _step_route(classification: AccumulatorCandidateClassification) -> str:
    return (
        C1_1C_ROUTE_SPARSE_EXACT
        if classification.decision_exact_feasible
        else C1_1C_ROUTE_BOUNDED_DELTA_WITH_REPORT
    )


def _vote_pressure_summary(spec: VoteUpdateSpec) -> dict[str, Any]:
    return {
        "eligible_weight_count": PRIMARY_ELIGIBLE_WEIGHT_COUNT,
        "tensor_count": len(PRIMARY_STATE_KEYS),
        "tensor_numel": PRIMARY_TENSOR_NUMEL,
        "state_keys": list(PRIMARY_STATE_KEYS),
        "threshold_abs": int(spec.threshold_abs),
        "decay_numerator": int(spec.decay_numerator),
        "decay_denominator": int(spec.decay_denominator),
        "clip": [int(spec.accumulator_clip_min), int(spec.accumulator_clip_max)],
        "cap_schedule": [
            {"step": step.step, "name": step.name, "cap": step.cap}
            for step in PRE_REGISTERED_VOTE_PRESSURE_SCHEDULE
        ],
        "vote_abs_values": sorted(
            {int(step.vote_abs) for step in PRE_REGISTERED_VOTE_PRESSURE_SCHEDULE}
        ),
        "schedule_fixed_before_measurement": True,
        "post_hoc_tuning_allowed": False,
    }


def run_pre_registered_native_loop_verdict() -> AccumulatorAssumptionBoundVerdictReport:
    """Run the fixed large-N CPU/native-loop distribution through C1.1b."""

    spec = default_vote_update_spec()
    coverage = SourceFieldCoverage.full_generated_native_loop()
    bindingness = pre_register_source_bindingness(
        source_kind=SOURCE_KIND_GENERATED_NATIVE_LOOP,
        coverage=coverage,
    )
    states = _initial_states()
    backlog: dict[str, dict[int, dict[str, int]]] = {}
    step_reports: list[NativeLoopVerdictStep] = []

    for schedule_step in PRE_REGISTERED_VOTE_PRESSURE_SCHEDULE:
        density_inputs = _density_inputs_for_step(states, schedule_step, spec)
        cap_inputs = _cap_inputs_for_density_inputs(density_inputs)
        offsets = tensor_offsets_for_vote_update_states(cap_inputs)
        cap_spec = GlobalRateCapSpec(cap=int(schedule_step.cap), step=int(schedule_step.step))
        density_report = measure_accumulator_decision_density(
            density_inputs,
            global_cap_spec=cap_spec,
            deferred_backlog=backlog,
            tensor_offsets=offsets,
        )
        validate_accumulator_decision_density_report(density_report)
        sparse_classification = _classify_sparse_step(density_report)
        cap_result = apply_global_rate_cap_reference(
            cap_inputs,
            cap_spec,
            deferred_backlog=backlog,
            tensor_offsets=offsets,
        )
        step_route = _step_route(sparse_classification)
        step_reports.append(
            NativeLoopVerdictStep(
                schedule_name=schedule_step.name,
                step=int(schedule_step.step),
                expected_regime=schedule_step.expected_regime,
                cap=int(schedule_step.cap),
                rows_per_tensor=int(schedule_step.rows_per_tensor),
                vote_abs=int(schedule_step.vote_abs),
                vote_nonzero_count=int(density_report.fixture_vote_nonzero_count),
                vote_abs_max=int(density_report.fixture_vote_abs_max),
                threshold_abs=int(spec.threshold_abs),
                decay_numerator=int(spec.decay_numerator),
                decay_denominator=int(spec.decay_denominator),
                clip_min=int(spec.accumulator_clip_min),
                clip_max=int(spec.accumulator_clip_max),
                eligible_weight_count=int(density_report.eligible_weight_count),
                global_pre_cap_would_apply_count=int(
                    cap_result.step_summary["global_pre_cap_would_apply_count"]
                ),
                global_cap_saturated=bool(density_report.global_cap_saturated),
                global_cap_accepted_count=int(density_report.global_cap_accepted_count),
                global_cap_deferred_count=int(density_report.global_cap_deferred_count),
                backlog_state_carry_count=int(density_report.backlog_state_carry_count),
                decision_relevant_exact_density=float(
                    density_report.decision_relevant_exact_density
                ),
                projected_bits_per_weight=float(
                    density_report.sparse_exact_projection.projected_bits_per_weight
                ),
                target_bits_per_weight=float(
                    density_report.sparse_exact_projection.target_bits_per_weight
                ),
                sparse_classification=sparse_classification,
                c1_1c_step_route=step_route,
                density_report=density_report,
            )
        )
        states = {
            result.state_key: VoteUpdateState(
                q_levels=result.q_levels,
                accumulators=result.accumulators,
            )
            for result in cap_result.tensor_results
        }
        backlog = cap_result.deferred_backlog

    all_sparse_exact = all(
        step.sparse_classification.decision_exact_feasible for step in step_reports
    )
    terminal_decision = (
        CandidateClassification.DECISION_EXACT.value
        if all_sparse_exact
        else DECISION_EXACT_INFEASIBLE
    )
    c1_1c_route = (
        C1_1C_ROUTE_SPARSE_EXACT
        if all_sparse_exact
        else C1_1C_ROUTE_BOUNDED_DELTA_WITH_REPORT
    )
    report = AccumulatorAssumptionBoundVerdictReport(
        schema_version=ACCUMULATOR_ASSUMPTION_BOUND_VERDICT_SCHEMA_VERSION,
        label=ASSUMPTION_BOUND_ENGINEERING_VERDICT,
        source_name="pre_registered_large_n_in_tree_native_loop_distribution",
        source_kind=SOURCE_KIND_GENERATED_NATIVE_LOOP,
        source_bindingness=bindingness,
        field_coverage=coverage,
        raw_state_capture_schema=raw_state_capture_schema(),
        vote_pressure_summary=_vote_pressure_summary(spec),
        representativeness_argument=(
            "The generated source uses the same q:int8/acc:int16 next-step law, "
            "threshold_abs=10, decay=1/1, clip bounds, global cap selection, "
            "N=16384 scale, and deferred-backlog accounting as the C1.1b native "
            "references. Its fixed schedule deliberately includes sparse, "
            "moderate, cap-saturated, and backlog-growth pressure."
        ),
        representativeness_limits=(
            "Vote locations and magnitudes are deterministic stress inputs, not raw S1 telemetry.",
            "A true live-regime measurement still needs the emitted raw-state capture schema.",
            "The result may route C1.1c engineering, but it is partial for S1 dynamics.",
        ),
        pre_registered_schedule=PRE_REGISTERED_VOTE_PRESSURE_SCHEDULE,
        per_step_reports=tuple(step_reports),
        terminal_decision=terminal_decision,
        c1_1c_route=c1_1c_route,
        raw_arrays_included=False,
        non_claims=(
            "no accumulator encoder or runtime integration",
            "no GPU lane, trainer, live run, checkpoint, creditdir, or .pt mutation",
            "no acquisition, retention, or stability claim",
            "no inclusive physical sub-2 claim",
            "no generated-source upgrade to raw S1 authority",
            "compact counts/hashes only; no raw per-weight arrays in this report",
        ),
    )
    validate_accumulator_assumption_bound_verdict_report(report)
    return report


def validate_accumulator_assumption_bound_verdict_report(
    report: AccumulatorAssumptionBoundVerdictReport,
) -> None:
    """Guard compactness, source labels, and terminal route consistency."""

    if report.raw_arrays_included:
        raise ValueError("assumption-bound verdict report must not include raw arrays")
    if report.label != ASSUMPTION_BOUND_ENGINEERING_VERDICT:
        raise ValueError("generated verdict must use the assumption-bound label")
    if not report.per_step_reports:
        raise ValueError("verdict report requires at least one measured step")
    if report.source_kind == SOURCE_KIND_GENERATED_NATIVE_LOOP:
        payload_text = json.dumps(report.to_dict(), sort_keys=True)
        for term in FORBIDDEN_GENERATED_SERIALIZATION_TERMS:
            if term in payload_text:
                raise ValueError(
                    f"generated source report must not serialize forbidden claim {term!r}"
                )
        if report.source_bindingness.primary_bindingness != (
            BINDING_FOR_IN_TREE_NATIVE_LOOP_DISTRIBUTION
        ):
            raise ValueError("generated source bindingness must stay in-tree only")
        if report.source_bindingness.s1_bindingness != PARTIAL_FOR_S1_REAL_DYNAMICS:
            raise ValueError("generated source must stay partial for S1 dynamics")
        if report.source_bindingness.can_claim_binding_live_regime:
            raise ValueError("generated source cannot claim binding live-regime coverage")

    for step in report.per_step_reports:
        validate_accumulator_decision_density_report(step.density_report)
        if step.density_report.raw_arrays_included:
            raise ValueError("nested density report must stay compact")
        expected_route = _step_route(step.sparse_classification)
        if step.c1_1c_step_route != expected_route:
            raise ValueError("per-step route does not match sparse classification")

    all_sparse_exact = all(
        step.sparse_classification.decision_exact_feasible for step in report.per_step_reports
    )
    expected_terminal_decision = (
        CandidateClassification.DECISION_EXACT.value
        if all_sparse_exact
        else DECISION_EXACT_INFEASIBLE
    )
    expected_route = (
        C1_1C_ROUTE_SPARSE_EXACT
        if all_sparse_exact
        else C1_1C_ROUTE_BOUNDED_DELTA_WITH_REPORT
    )
    if report.terminal_decision != expected_terminal_decision:
        raise ValueError("terminal decision does not match pre-registered step results")
    if report.c1_1c_route != expected_route:
        raise ValueError("terminal C1.1c route does not match step verdicts")


def assert_compact_payload_has_no_tensors(value: Any) -> None:
    """Test helper: recursively reject raw torch tensors in compact payloads."""

    if isinstance(value, torch.Tensor):
        raise AssertionError("compact verdict payload must not contain tensors")
    if isinstance(value, Mapping):
        for child in value.values():
            assert_compact_payload_has_no_tensors(child)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            assert_compact_payload_has_no_tensors(child)
