"""Default-off R7 cap/defer pressure instrumentation chunk emitter."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from calm.hrm_text_158.native_full_stack.acc_width_recorded_row_sweep import (
    CANONICAL_VOTE_UPDATE_THRESHOLD_ABS,
)
from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
    BoundedDeltaTensorState,
)
from calm.hrm_text_158.native_full_stack.vote_update import VoteUpdateState

R7_STEP_CHUNK_SCHEMA_VERSION = "hrm_text_158_r7_cap_defer_pressure_step_chunk/v1"
R7_SIDECAR_FILENAME = "r7_cap_defer_pressure_sidecar.jsonl"

HIGH_PRESSURE_ABS = int(CANONICAL_VOTE_UPDATE_THRESHOLD_ABS)

REQUIRED_CAP_FIELDS: tuple[str, ...] = (
    "global_rate_cap_enabled",
    "global_pre_cap_would_apply_count",
    "global_rate_cap_accepted_count",
    "global_rate_cap_deferred_count",
    "global_rate_cap_cap",
    "global_rate_cap_saturated",
    "q_changed_count",
    "deferred_backlog_size",
    "deferred_backlog_max_age_steps",
    "deferred_backlog_max_defer_count",
)


def _accumulator_i16_flat(state: Any) -> torch.Tensor:
    if isinstance(state, BoundedDeltaTensorState):
        return (
            state.decoded_accumulators(rebuild_if_stale=True)
            .detach()
            .cpu()
            .flatten()
            .to(torch.int16)
        )
    if isinstance(state, VoteUpdateState):
        return state.accumulators.detach().cpu().flatten().to(torch.int16)
    raise TypeError(
        "pressure_mass_from_tensor_states: unsupported tensor state type "
        f"{type(state)!r}; expected BoundedDeltaTensorState or VoteUpdateState"
    )


def pressure_mass_from_tensor_states(tensor_states: Mapping[str, Any]) -> int:
    total = 0
    for state in tensor_states.values():
        acc = _accumulator_i16_flat(state)
        total += int(torch.sum(acc.abs() >= HIGH_PRESSURE_ABS).item())
    return total


def _step_content_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def validate_accounting_invariant(summary: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    if not bool(summary.get("global_rate_cap_enabled")):
        failures.append("global_rate_cap_disabled")
        return failures
    for field in REQUIRED_CAP_FIELDS:
        if field not in summary:
            failures.append(f"missing_field:{field}")
    if failures:
        return failures
    candidate = int(summary["global_pre_cap_would_apply_count"])
    accepted = int(summary["global_rate_cap_accepted_count"])
    deferred = int(summary["global_rate_cap_deferred_count"])
    if candidate != accepted + deferred:
        failures.append("candidate_partition_violation")
    accepted_from_prior = int(summary.get("accepted_from_prior_deferred_count", 0))
    if not (0 <= accepted_from_prior <= accepted):
        failures.append("accepted_from_prior_deferred_bounds")
    accepted_fresh = int(summary.get("accepted_fresh_count", accepted - accepted_from_prior))
    if accepted_fresh != accepted - accepted_from_prior:
        failures.append("accepted_fresh_mismatch")
    return failures


def build_step_chunk(
    *,
    step: int,
    global_summary: Mapping[str, Any],
    pressure_mass: int,
    pressure_mass_delta: int | None,
    optional_selection_scores: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    summary = dict(global_summary)
    invariant_failures = validate_accounting_invariant(summary)
    chunk: dict[str, Any] = {
        "schema_version": R7_STEP_CHUNK_SCHEMA_VERSION,
        "step": int(step),
        "candidate_count": int(summary.get("global_pre_cap_would_apply_count", 0)),
        "accepted_count": int(summary.get("global_rate_cap_accepted_count", 0)),
        "deferred_count": int(summary.get("global_rate_cap_deferred_count", 0)),
        "accepted_from_prior_deferred_count": int(
            summary.get("accepted_from_prior_deferred_count", 0)
        ),
        "accepted_fresh_count": int(summary.get("accepted_fresh_count", 0)),
        "q_apply_count": int(summary.get("q_changed_count", 0)),
        "global_rate_cap_cap": int(summary.get("global_rate_cap_cap", 0)),
        "global_rate_cap_saturated": bool(summary.get("global_rate_cap_saturated", False)),
        "deferred_backlog_size": int(summary.get("deferred_backlog_size", 0)),
        "deferred_backlog_max_age_steps": int(
            summary.get("deferred_backlog_max_age_steps", 0)
        ),
        "deferred_backlog_max_defer_count": int(
            summary.get("deferred_backlog_max_defer_count", 0)
        ),
        "pressure_mass": int(pressure_mass),
        "pressure_mass_delta": pressure_mass_delta,
        "accounting_invariant_failures": list(invariant_failures),
        "raw_arrays_included": False,
    }
    if optional_selection_scores:
        for key in ("applied_selection_score_p50", "applied_selection_score_p95"):
            if key in optional_selection_scores:
                chunk[key] = optional_selection_scores[key]
    chunk["step_content_hash"] = _step_content_hash(
        {key: value for key, value in chunk.items() if key != "step_content_hash"}
    )
    return chunk


def append_step_chunk(sidecar_path: Path, chunk: Mapping[str, Any]) -> None:
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    with sidecar_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(chunk), sort_keys=True) + "\n")


def iter_sidecar_chunks(sidecar_path: Path) -> list[dict[str, Any]]:
    if not sidecar_path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line in sidecar_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        records.append(json.loads(line))
    return records


def optional_selection_scores_from_step_result_compact(
    step_result_compact: Mapping[str, Any],
) -> dict[str, Any]:
    tensor_stats = step_result_compact.get("tensor_stats")
    if not isinstance(tensor_stats, dict):
        return {}
    for stats in tensor_stats.values():
        if not isinstance(stats, dict):
            continue
        scores: dict[str, Any] = {}
        for key in ("applied_selection_score_p50", "applied_selection_score_p95"):
            if key in stats:
                scores[key] = stats[key]
        if scores:
            return scores
    return {}


def field_presence_from_step_summary(step_summary: Mapping[str, Any]) -> dict[str, Any]:
    missing = [field for field in REQUIRED_CAP_FIELDS if field not in step_summary]
    return {
        "schema_version": "hrm_text_158_r7_cap_seam_field_presence_witness/v1",
        "field_presence_pass": not missing,
        "required_fields": list(REQUIRED_CAP_FIELDS),
        "missing_fields": missing,
        "observed_fields": {field: step_summary.get(field) for field in REQUIRED_CAP_FIELDS},
    }
