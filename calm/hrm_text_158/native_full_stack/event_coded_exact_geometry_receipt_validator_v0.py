"""Pure event-coded exact-geometry receipt validator (no IO/GPU).

Behavior-preserving extraction from receipt_compactness_guard.py (round-4 seam).
Census / compaction remain on the bankable-receipt guard; this module owns only
per-step geometry authority for the exact-geometry smoke.
"""
from __future__ import annotations

from typing import Any, Mapping

# Exact-geometry failure classes for event-coded nondense smokes (pure, no IO/GPU).
EXACT_GEOMETRY_EXPECTED_STEPS = 20
EXACT_GEOMETRY_LIVE_AUTHORITY = "event_coded_live_carrier"
EXACT_GEOMETRY_FAILURE_CLASSES = frozenset(
    {
        "steps_requested",
        "steps_completed",
        "step_reports_coverage",
        "toplevel_event_coded_live",
        "toplevel_sparse_vote_authority",
        "per_step_global_rate_cap",
        "per_step_event_coded_live",
        "live_authority",
        "gpu_execution_evidence",
        "bdgs_corroboration",
    }
)


def validate_event_coded_exact_geometry_receipt(
    receipt: Mapping[str, Any],
    *,
    expected_steps: int = EXACT_GEOMETRY_EXPECTED_STEPS,
) -> list[dict[str, str]]:
    """Pure receipt-geometry validator for the exact-geometry event-coded smoke.

    Authority is **per-step** ``step_reports[i].step_result.global_summary`` and
    every ``tensor_stats`` row's ``live_authority`` — NOT the top-level
    ``bounded_delta_global_summary`` (producer copies only the last step).
    BDGS is checked only as optional corroboration after per-step authority.

    Returns a list of ``{"class": <failure_class>, "detail": <msg>}``; empty = ok.
    Pure: no IO, no GPU imports, no side effects.
    """

    failures: list[dict[str, str]] = []

    def fail(cls: str, detail: str) -> None:
        failures.append({"class": cls, "detail": detail})

    if not isinstance(receipt, Mapping):
        fail("step_reports_coverage", f"receipt_type={type(receipt).__name__}")
        return failures

    if receipt.get("steps_requested") != expected_steps:
        fail(
            "steps_requested",
            f"steps_requested={receipt.get('steps_requested')!r} expected={expected_steps}",
        )
    if receipt.get("steps_completed") != expected_steps:
        fail(
            "steps_completed",
            f"steps_completed={receipt.get('steps_completed')!r} expected={expected_steps}",
        )

    step_reports = receipt.get("step_reports")
    expected_keys = {str(i) for i in range(1, int(expected_steps) + 1)}
    if not isinstance(step_reports, dict):
        fail(
            "step_reports_coverage",
            f"step_reports_type={type(step_reports).__name__}",
        )
        step_reports = {}
    else:
        observed = set(step_reports.keys())
        if observed != expected_keys:
            missing = sorted(expected_keys - observed, key=lambda s: int(s) if s.isdigit() else s)
            extra = sorted(observed - expected_keys, key=lambda s: int(s) if s.isdigit() else s)
            fail(
                "step_reports_coverage",
                f"keys_mismatch missing={missing!r} extra={extra!r} count={len(observed)}",
            )

    if receipt.get("persistent_accumulator_event_coded_live") is not True:
        fail(
            "toplevel_event_coded_live",
            "persistent_accumulator_event_coded_live="
            f"{receipt.get('persistent_accumulator_event_coded_live')!r}",
        )
    if receipt.get("event_coded_sparse_vote_authority") is not True:
        fail(
            "toplevel_sparse_vote_authority",
            "event_coded_sparse_vote_authority="
            f"{receipt.get('event_coded_sparse_vote_authority')!r}",
        )

    # Per-step authority (not last-step-only BDGS).
    for step_key in sorted(expected_keys, key=lambda s: int(s)):
        step = step_reports.get(step_key) if isinstance(step_reports, dict) else None
        if not isinstance(step, dict):
            # Coverage already flagged missing keys; skip deeper if absent.
            continue
        step_result = step.get("step_result")
        if not isinstance(step_result, dict):
            fail(
                "per_step_event_coded_live",
                f"step={step_key} missing_step_result",
            )
            continue
        global_summary = step_result.get("global_summary")
        if not isinstance(global_summary, dict):
            fail(
                "per_step_event_coded_live",
                f"step={step_key} missing_global_summary",
            )
            global_summary = {}
        if global_summary.get("global_rate_cap_enabled") is not False:
            fail(
                "per_step_global_rate_cap",
                f"step={step_key} global_rate_cap_enabled="
                f"{global_summary.get('global_rate_cap_enabled')!r}",
            )
        if global_summary.get("event_coded_live_carrier_enabled") is not True:
            fail(
                "per_step_event_coded_live",
                f"step={step_key} event_coded_live_carrier_enabled="
                f"{global_summary.get('event_coded_live_carrier_enabled')!r}",
            )
        tensor_stats = step_result.get("tensor_stats")
        if not isinstance(tensor_stats, dict) or not tensor_stats:
            fail(
                "live_authority",
                f"step={step_key} missing_or_empty_tensor_stats",
            )
        else:
            for mod_name, row in tensor_stats.items():
                if not isinstance(row, dict):
                    fail(
                        "live_authority",
                        f"step={step_key} mod={mod_name!r} non_dict_row",
                    )
                    continue
                if row.get("live_authority") != EXACT_GEOMETRY_LIVE_AUTHORITY:
                    fail(
                        "live_authority",
                        f"step={step_key} mod={mod_name!r} live_authority="
                        f"{row.get('live_authority')!r} expected="
                        f"{EXACT_GEOMETRY_LIVE_AUTHORITY!r}",
                    )

    device = str(receipt.get("device") or "")
    device_guard = (
        receipt.get("device_guard")
        if isinstance(receipt.get("device_guard"), Mapping)
        else {}
    )
    gpu_ok = (
        device.startswith("cuda")
        and receipt.get("gpu_launched") is True
        and receipt.get("gpu_launch_authorized") is True
        and receipt.get("forward_backward_update_executed") is True
        and device_guard.get("cuda_available") is True
        and device_guard.get("pass") is True
    )
    if not gpu_ok:
        fail(
            "gpu_execution_evidence",
            f"device={device!r} gpu_launched={receipt.get('gpu_launched')!r} "
            f"gpu_launch_authorized={receipt.get('gpu_launch_authorized')!r} "
            f"forward_backward_update_executed="
            f"{receipt.get('forward_backward_update_executed')!r} "
            f"device_guard={dict(device_guard)!r}",
        )

    # Corroboration only — last-step-only BDGS must not contradict per-step authority.
    bdgs = receipt.get("bounded_delta_global_summary")
    if isinstance(bdgs, Mapping):
        if bdgs.get("global_rate_cap_enabled") is not False:
            fail(
                "bdgs_corroboration",
                "bounded_delta_global_summary.global_rate_cap_enabled="
                f"{bdgs.get('global_rate_cap_enabled')!r}",
            )
        if bdgs.get("event_coded_live_carrier_enabled") is not True:
            fail(
                "bdgs_corroboration",
                "bounded_delta_global_summary.event_coded_live_carrier_enabled="
                f"{bdgs.get('event_coded_live_carrier_enabled')!r}",
            )

    return failures
