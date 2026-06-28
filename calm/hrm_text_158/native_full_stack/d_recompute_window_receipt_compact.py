"""Compact D-feasibility probe receipt step_result tensor_stats for long horizons."""
from __future__ import annotations

import json
from typing import Any, Mapping

from calm.hrm_text_158.native_full_stack.s3bb_headroom_telemetry import (
    RECEIPT_EMIT_PROFILE_SLIM,
)

D_RECOMPUTE_WINDOW_FEASIBILITY_PHASE = "d-recompute-window-feasibility"

COMPACT_SCHEMA_VERSION = "hrm_text_158_d_recompute_window_receipt_compact/v0"

# Raw index/score arrays and other bulky per-module surfaces dropped before receipt write.
D_DIAGNOSTIC_COMPACT_TENSOR_STATS_DROP_KEYS: frozenset[str] = frozenset(
    {
        "pre_veto_selected_indices",
        "applied_indices",
        "post_veto_would_apply_pre_cap_indices",
        "post_veto_applied_indices",
        "replay_ce_veto_indices",
        "top4096_flat_indices_hash16",
        "post_veto_applied_flip_count_lanes",
        "applied_selection_scores",
        "pre_veto_selection_scores",
        "optional_selection_scores",
    }
)

D_DIAGNOSTIC_COMPACT_TENSOR_STATS_KEEP_KEYS: frozenset[str] = frozenset(
    {
        "q_changed_count",
        "replay_ce_veto_count",
        "post_veto_applied_flip_count",
        "applied_flat_indices_hash16",
        "top8_flat_indices_hash16",
        "top64_flat_indices_hash16",
        "applied_selection_score_p50",
        "applied_selection_score_p95",
        "applied_selection_score_semantics",
        "cap_window_jaccard_vs_prior_step",
        "cap_window_audit_non_authoritative",
    }
)

DEFAULT_EXTRAPOLATED_H100_RECEIPT_BYTES_MAX = 52_428_800
DEFAULT_EXTRAPOLATED_H100_RECOMPUTE_LOG_BYTES_MAX = 33_554_432
DEFAULT_RECEIPT_BYTES_PER_STEP_MAX = 524_288
DEFAULT_RECOMPUTE_LOG_BYTES_PER_STEP_MAX = 327_680
DEFAULT_TARGET_TENSOR_STATS_BYTES_PER_STEP = 50 * 1024
DEFAULT_H100_CONFIRMATION_STEPS = 100


def should_apply_d_diagnostic_receipt_compaction(
    *,
    phase: str,
    receipt_emit_profile: str,
    d_diagnostic_compact_step_reports: bool,
) -> bool:
    return (
        bool(d_diagnostic_compact_step_reports)
        and str(phase) == D_RECOMPUTE_WINDOW_FEASIBILITY_PHASE
        and str(receipt_emit_profile) == RECEIPT_EMIT_PROFILE_SLIM
    )


def _compact_tensor_stats_entry(stats: Mapping[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key, value in dict(stats).items():
        if key in D_DIAGNOSTIC_COMPACT_TENSOR_STATS_DROP_KEYS:
            continue
        if key in D_DIAGNOSTIC_COMPACT_TENSOR_STATS_KEEP_KEYS:
            compact[key] = value
            continue
        if isinstance(value, list) and (
            key.endswith("_indices")
            or key.endswith("_scores")
            or len(value) > 32
        ):
            continue
        if isinstance(value, (int, float, bool, str)) or value is None:
            compact[key] = value
    return compact


def compact_d_diagnostic_step_result(step_result_compact: Mapping[str, Any]) -> dict[str, Any]:
    compact = dict(step_result_compact)
    tensor_stats = {
        state_key: _compact_tensor_stats_entry(stats)
        for state_key, stats in dict(compact.get("tensor_stats", {})).items()
    }
    compact["tensor_stats"] = tensor_stats
    compact["d_diagnostic_receipt_compact"] = {
        "schema_version": COMPACT_SCHEMA_VERSION,
        "drop_keys": sorted(D_DIAGNOSTIC_COMPACT_TENSOR_STATS_DROP_KEYS),
        "keep_keys": sorted(D_DIAGNOSTIC_COMPACT_TENSOR_STATS_KEEP_KEYS),
    }
    return compact


def estimate_step_reports_tensor_stats_bytes(step_reports: Mapping[str, Any]) -> int:
    total = 0
    for report in step_reports.values():
        step_result = dict(report).get("step_result", {})
        tensor_stats = dict(step_result).get("tensor_stats", {})
        total += len(
            json.dumps(tensor_stats, separators=(",", ":"), sort_keys=True).encode("utf-8")
        )
    return int(total)


def extrapolate_h100_byte_projections(
    *,
    receipt_bytes: int,
    smoke_steps: int,
    recompute_log_bytes: int = 0,
    confirmation_steps: int = DEFAULT_H100_CONFIRMATION_STEPS,
    receipt_bytes_per_step_max: int = DEFAULT_RECEIPT_BYTES_PER_STEP_MAX,
    recompute_log_bytes_per_step_max: int = DEFAULT_RECOMPUTE_LOG_BYTES_PER_STEP_MAX,
    extrapolated_h100_receipt_bytes_max: int = DEFAULT_EXTRAPOLATED_H100_RECEIPT_BYTES_MAX,
    extrapolated_h100_recompute_log_bytes_max: int = DEFAULT_EXTRAPOLATED_H100_RECOMPUTE_LOG_BYTES_MAX,
) -> dict[str, Any]:
    if int(smoke_steps) <= 0:
        raise ValueError("smoke_steps must be positive")
    receipt_bytes_per_step = float(receipt_bytes) / float(smoke_steps)
    recompute_log_bytes_per_step = float(recompute_log_bytes) / float(smoke_steps)
    extrapolated_receipt_bytes = int(receipt_bytes_per_step * float(confirmation_steps))
    extrapolated_recompute_log_bytes = int(
        recompute_log_bytes_per_step * float(confirmation_steps)
    )
    return {
        "schema_version": COMPACT_SCHEMA_VERSION,
        "smoke_steps": int(smoke_steps),
        "confirmation_steps": int(confirmation_steps),
        "receipt_bytes": int(receipt_bytes),
        "recompute_log_bytes": int(recompute_log_bytes),
        "receipt_bytes_per_step": receipt_bytes_per_step,
        "recompute_log_bytes_per_step": recompute_log_bytes_per_step,
        "extrapolated_h100_receipt_bytes": extrapolated_receipt_bytes,
        "extrapolated_h100_recompute_log_bytes": extrapolated_recompute_log_bytes,
        "caps": {
            "receipt_bytes_per_step_max": int(receipt_bytes_per_step_max),
            "recompute_log_bytes_per_step_max": int(recompute_log_bytes_per_step_max),
            "extrapolated_h100_receipt_bytes_max": int(extrapolated_h100_receipt_bytes_max),
            "extrapolated_h100_recompute_log_bytes_max": int(
                extrapolated_h100_recompute_log_bytes_max
            ),
        },
        "pass": {
            "receipt_bytes_per_step": receipt_bytes_per_step <= float(receipt_bytes_per_step_max),
            "recompute_log_bytes_per_step": recompute_log_bytes_per_step
            <= float(recompute_log_bytes_per_step_max),
            "extrapolated_h100_receipt_bytes": extrapolated_receipt_bytes
            <= int(extrapolated_h100_receipt_bytes_max),
            "extrapolated_h100_recompute_log_bytes": extrapolated_recompute_log_bytes
            <= int(extrapolated_h100_recompute_log_bytes_max),
        },
        "launch_allowed": (
            receipt_bytes_per_step <= float(receipt_bytes_per_step_max)
            and recompute_log_bytes_per_step <= float(recompute_log_bytes_per_step_max)
            and extrapolated_receipt_bytes <= int(extrapolated_h100_receipt_bytes_max)
            and extrapolated_recompute_log_bytes <= int(extrapolated_h100_recompute_log_bytes_max)
        ),
    }
