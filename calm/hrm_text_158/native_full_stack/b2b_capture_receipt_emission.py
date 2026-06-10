"""Receipt-write-only emission for B2b capture §2C threshold semantics and warmup tags.

All enrichment runs at receipt-write time from already-recorded trace rows.
No step-loop / forward / backward / vote / apply instrumentation.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from calm.hrm_text_158.native_full_stack.acc_width_recorded_row_sweep import (
    build_teacher_forced_applied_candidate_ids,
    load_acc_width_trace_steps,
)
from calm.hrm_text_158.native_full_stack.accumulator_policy_shadow_screen import (
    B2B_SEQUENTIAL_TRACE_SCHEMA,
)

# Frozen §2C from transient_selection_interface.md (R2 1781122431786).
CROSSING_THRESHOLD_ABS = 10
FROZEN_THRESHOLD_SEMANTICS: dict[str, Any] = {
    "crossing_threshold_abs": 10,
    "crossing_threshold_source": "canonical_default_spec_accumulator_real_dynamics_verdict",
    "crossing_authority": "vote_update_spec",
    "residual_band_encoding": "threshold_minus_one",
    "row_fields_authority": "telemetry_not_crossing",
    "row_crosscheck_policy": "informational",
}

WARMUP_APPLY_CLASS_CANONICAL = "canonical"
WARMUP_APPLY_CLASS_SUBTHRESHOLD_BOOTSTRAP = "subthreshold_bootstrap"


def frozen_threshold_semantics_block() -> dict[str, Any]:
    """Return the verbatim §2C threshold_semantics block."""

    return dict(FROZEN_THRESHOLD_SEMANTICS)


def _applied_flat_indices_for_step(
    step: Mapping[str, Any],
    *,
    applied_candidate_ids_by_step: Mapping[int, str],
) -> list[int]:
    telemetry = dict(step.get("post_update_telemetry") or {})
    override = telemetry.get("applied_flip_flat_indices")
    if isinstance(override, list):
        return [int(value) for value in override]
    if int(telemetry.get("q_changed_count", 0)) <= 0:
        return []
    step_index = int(step["optimizer_step_index"])
    candidate_id = applied_candidate_ids_by_step.get(step_index)
    if not candidate_id:
        return []
    for row in step.get("sampled_candidate_table") or ():
        if not isinstance(row, Mapping):
            continue
        if str(row.get("candidate_id")) == str(candidate_id):
            return [int(row["flat_index"])]
    return []


def derive_step_warmup_apply_tags(
    step: Mapping[str, Any],
    *,
    applied_candidate_ids_by_step: Mapping[int, str],
) -> dict[str, Any]:
    """Derive per-step warmup tags from recorded applied-row new_acc values."""

    applied_indices = _applied_flat_indices_for_step(
        step,
        applied_candidate_ids_by_step=applied_candidate_ids_by_step,
    )
    applied_abs_new_acc: list[int] = []
    for flat_index in applied_indices:
        row = next(
            (
                candidate_row
                for candidate_row in step.get("sampled_candidate_table") or ()
                if isinstance(candidate_row, Mapping)
                and int(candidate_row.get("flat_index", -1)) == int(flat_index)
            ),
            None,
        )
        if row is None:
            continue
        applied_abs_new_acc.append(abs(int(row["new_acc_i32_signed"])))

    if any(value < CROSSING_THRESHOLD_ABS for value in applied_abs_new_acc):
        return {
            "warmup_apply_class": WARMUP_APPLY_CLASS_SUBTHRESHOLD_BOOTSTRAP,
            "effective_apply_threshold_abs": (
                max(applied_abs_new_acc) if applied_abs_new_acc else None
            ),
        }
    return {
        "warmup_apply_class": WARMUP_APPLY_CLASS_CANONICAL,
        "effective_apply_threshold_abs": None,
    }


def enrich_b2b_trace_steps_at_receipt_write(
    steps: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Attach warmup tags to each step dict without touching loop capture paths."""

    applied_candidate_ids_by_step = build_teacher_forced_applied_candidate_ids(steps)
    enriched: list[dict[str, Any]] = []
    for step in steps:
        record = dict(step)
        record.update(
            derive_step_warmup_apply_tags(
                step,
                applied_candidate_ids_by_step=applied_candidate_ids_by_step,
            )
        )
        enriched.append(record)
    return enriched


def rewrite_b2b_trace_with_receipt_emissions(trace_path: Path) -> list[dict[str, Any]]:
    """Load trace, enrich per-step warmup tags, and rewrite at receipt-write."""

    steps, load_failures = load_acc_width_trace_steps(trace_path)
    if load_failures:
        raise ValueError(
            "b2b trace load failed at receipt-write: " + ",".join(load_failures)
        )
    enriched = enrich_b2b_trace_steps_at_receipt_write(steps)
    lines = [json.dumps({"schema": B2B_SEQUENTIAL_TRACE_SCHEMA}, sort_keys=True)]
    lines.extend(json.dumps(step, sort_keys=True) for step in enriched)
    trace_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return enriched


def finalize_b2b_capture_receipt(base_receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Attach frozen §2C threshold_semantics to the capture receipt payload."""

    receipt = dict(base_receipt)
    receipt["threshold_semantics"] = frozen_threshold_semantics_block()
    return receipt
