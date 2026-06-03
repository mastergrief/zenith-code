"""Acceptance metric schemas for the native-full-stack scaffold."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AcceptanceMetric:
    name: str
    primary_gate: bool
    formula_or_fields: tuple[str, ...]
    pass_semantics: str
    failure_semantics: str
    required_for_first_addition: bool


ACCEPTANCE_METRICS = (
    AcceptanceMetric(
        name="wall_clock_per_step",
        primary_gate=True,
        formula_or_fields=("step_duration_seconds", "profile_phase_seconds"),
        pass_semantics="iteration loop gets faster or more diagnosable under native constraints",
        failure_semantics="native path is slower without a proof or diagnostic benefit",
        required_for_first_addition=True,
    ),
    AcceptanceMetric(
        name="max_safe_batch",
        primary_gate=True,
        formula_or_fields=("batch_size", "oom_free", "diagnostic_fail_free"),
        pass_semantics="larger safe batch or clear headroom under the same smoke guard",
        failure_semantics="OOM/diagnostic failure before useful exposure",
        required_for_first_addition=True,
    ),
    AcceptanceMetric(
        name="effective_exposure_per_step",
        primary_gate=True,
        formula_or_fields=("valid_labels", "support_rows", "updated_tensor_count"),
        pass_semantics="useful supervised exposure per step is recorded and comparable",
        failure_semantics="no comparable exposure accounting",
        required_for_first_addition=True,
    ),
    AcceptanceMetric(
        name="time_to_diagnosable_failure",
        primary_gate=True,
        formula_or_fields=("first_failed_phase", "first_failed_step", "failure_reason"),
        pass_semantics="bad additions fail quickly with an attributable reason",
        failure_semantics="slow or silent failure",
        required_for_first_addition=True,
    ),
    AcceptanceMetric(
        name="resource_headroom",
        primary_gate=True,
        formula_or_fields=("peak_allocated", "peak_reserved", "free_memory", "cap_pressure"),
        pass_semantics="resource headroom is measured without hidden GPU use in Phase-0",
        failure_semantics="untracked resource pressure or GPU entry during CPU/static phase",
        required_for_first_addition=True,
    ),
    AcceptanceMetric(
        name="attribution_integrity",
        primary_gate=True,
        formula_or_fields=("source_hash", "state_hash", "hook_counts", "decode_eos_floor"),
        pass_semantics="ternary q/vote path is shown to be the learner before banking",
        failure_semantics="fast hidden-FP learner or broken attribution",
        required_for_first_addition=True,
    ),
    AcceptanceMetric(
        name="acquisition_quality",
        primary_gate=False,
        formula_or_fields=("strict_exact_rate", "parsed_rate", "surface_family_counts"),
        pass_semantics="tracked trend only for first native addition",
        failure_semantics="trend regression reported but not a 90/90 bank gate for first addition",
        required_for_first_addition=True,
    ),
)


def first_class_metric_names() -> tuple[str, ...]:
    return tuple(metric.name for metric in ACCEPTANCE_METRICS if metric.primary_gate)
