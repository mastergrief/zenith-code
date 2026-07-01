#!/usr/bin/env python3
"""v6i OOM profile/attribution: extract aborted-run artifacts and attribute RSS owners."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.hrm_text_158_bounded_delta_acquisition_probe import (  # noqa: E402
    HOST_RSS_PROFILE_JSONL_NAME,
    PROFILE_ALLOCATOR_HOST_CACHE_DIAG_ENV,
    PROFILE_ALLOCATOR_NATIVE_ENV,
    PROFILE_HOST_RSS_ALLOCATOR_SCHEMA,
    PROFILE_HOST_RSS_ALLOCATOR_SITE_SCHEMA,
    PROFILE_HOST_RSS_CENSUS_SCHEMA,
    PROFILE_HOST_RSS_ENV,
    PROFILE_HOST_RSS_LIVE_RESIDENT_DROP_GIB,
    PROFILE_HOST_RSS_LIVE_RESIDENT_ENV,
    PROFILE_HOST_RSS_SUBPHASE_IDS,
    PROFILE_TORCH_CPU_CENSUS_ENV,
)

ATTRIBUTION_SCHEMA = "hrm_text_158_v6i_oom_profile_attribution_receipt/v5"
EXTRACT_SCHEMA = "hrm_text_158_v6i_oom_profile_extract_readonly/v1"

SUBPHASE_RESOLVE_FRACTION = 0.80
SUBPHASE_UNMAPPED_FRACTION = 0.50
DIMENSIONAL_OVERHEAD_FACTOR = 8.0
LIVE_RESIDENT_DIAGNOSTIC_SUBPHASE = "C4_gpu_cap_apply_sync"

TARGET_PHASES = (
    "two_tier_grad_proxy_ingress",
    "activation_credit_forward_backward",
    "activation_credit_gather",
    "delta_weight_scatter",
    "coverage",
    "sparse_cap_apply",
    "step_forward_backward",
    "sparse_vote_construction",
    "step_update",
    "step",
    "live_carrier_snapshot_emit",
    "receipt_write",
    "bounded_steps",
)

RSS_ATTRIBUTION_LEAF_PHASES = frozenset({
    "step_forward_backward",
    "sparse_vote_construction",
    "sparse_cap_apply",
    "live_carrier_snapshot_emit",
    "receipt_write",
})

CULPRIT_CLASSES = {
    "A": "sparse_cap_gpu_seam_host_mirrors",
    "B": "live_carrier_snapshot_emit",
    "C": "per_step_in_memory_accumulation",
    "D": "step_forward_backward_host_tensors",
    "E": "two_tier_grad_proxy_oracle_captures",
    "F": "receipt_checkpoint_materialization",
}

PHASE_CLASS_CANDIDATE_HINTS: dict[str, str] = {
    "sparse_cap_apply": "A",
    "live_carrier_snapshot_emit": "B",
    "step_update": "C",
    "step": "C",
    "step_forward_backward": "D",
    "receipt_write": "F",
}

SUBPHASE_MECHANISM_HINTS: dict[str, str] = {
    "C1_vote_plan_build": "vote_plan_construction",
    "C2_cap_input_assembly": "cap_input_shape_stub_dense_int16",
    "C3_gpu_cap_selection": "gpu_cap_selection_host_mirrors",
    "C4_gpu_cap_apply_sync": "per_state_q_carrier_residency",
    "C5_next_state_materialize": "next_state_q_materialization",
    "C6_deferred_backlog_telemetry": "deferred_backlog_copy",
}

ALLOCATION_SITE_ORIGINS: dict[str, tuple[str, str]] = {
    "C3_exit": (
        "sparse_cap_gpu_seam_adapter.py:391-401",
        "tensor_results_all_states_materialized",
    ),
    "C4_enter": (
        "bounded_delta_learner.py:2367-2405",
        "handoff_after_c6_before_apply_loop",
    ),
    "C4_after_state": (
        "bounded_delta_learner.py:2407-2408",
        "q_by_key_carriers_accumulation",
    ),
    "C4_exit": (
        "bounded_delta_learner.py:2409-2413",
        "post_apply_loop_residency",
    ),
}

CENSUS_RECONCILE_RATIO_MIN = 0.8
CENSUS_RECONCILE_RATIO_MAX = 1.2
ALLOCATOR_TIER_A_DOMINANCE = 0.80
ALLOCATOR_PER_STATE_SLOPE_MIN = 0.75
ALLOCATOR_PER_STATE_SLOPE_MAX = 1.25
ALLOCATOR_TIER_B_SITE_FRACTION = 0.50
HOST_CACHE_CONFIRM_DROP_GIB = 1.0


def _is_allocator_mark(row: Mapping[str, Any]) -> bool:
    return str(row.get("schema", "")) == PROFILE_HOST_RSS_ALLOCATOR_SCHEMA


def _is_allocator_site_mark(row: Mapping[str, Any]) -> bool:
    return str(row.get("schema", "")) == PROFILE_HOST_RSS_ALLOCATOR_SITE_SCHEMA


def _probe_nested(probe: Mapping[str, Any], *keys: str) -> Any:
    current: Any = probe
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _allocator_source_scalars(probe: Mapping[str, Any]) -> dict[str, Any]:
    cuda = dict(probe.get("cuda_allocator") or {})
    mallinfo = dict(probe.get("mallinfo2") or {})
    rollup = dict(probe.get("smaps_rollup") or {})
    categories = dict(probe.get("smaps_categories") or {})
    host_active = None
    for key, value in cuda.items():
        if key.startswith("cuda_host_active_bytes"):
            host_active = int(value)
            break
    if host_active is None:
        for key, value in cuda.items():
            if key.startswith("cuda_host_allocated_bytes"):
                host_active = int(value)
                break
    return {
        "anonymous_kb": rollup.get("anonymous_kb"),
        "private_dirty_kb": rollup.get("private_dirty_kb"),
        "heap_kb": categories.get("heap_kb"),
        "glibc_uordblks_bytes": mallinfo.get("uordblks_bytes"),
        "cuda_host_active_bytes": host_active,
        "host_memory_stats_available": bool(cuda.get("host_memory_stats_available")),
        "cuda_gpu_allocated_bytes": cuda.get("cuda_gpu_allocated_bytes"),
        "cuda_gpu_stats_role": cuda.get("cuda_gpu_stats_role"),
    }


def _mark_rss_kib(mark: Mapping[str, Any]) -> int | None:
    snap = dict(mark.get("resource_snapshot") or {})
    value = snap.get("rss_kib")
    return int(value) if value is not None else None


def _delta_gib(curr: int | None, prev: int | None, *, kib: bool = False) -> float | None:
    if curr is None or prev is None:
        return None
    if kib:
        return (float(curr) - float(prev)) / (1024.0 * 1024.0)
    return (float(curr) - float(prev)) / (1024.0**3)


def _mark_state_index(row: Mapping[str, Any], *, default: int | None = None) -> int | None:
    if "state_index" not in row:
        return default
    return int(row["state_index"])


def attribute_allocator_native_profile(
    allocator_marks: list[dict[str, Any]],
    *,
    c4_delta_rss_gib: float | None,
) -> dict[str, Any]:
    marks = [row for row in allocator_marks if _is_allocator_mark(row)]
    site_marks = [row for row in allocator_marks if _is_allocator_site_mark(row)]
    by_event: dict[str, dict[str, Any]] = {}
    for row in marks:
        by_event[str(row.get("event"))] = row

    bucket_marks = sorted(
        [row for row in marks if str(row.get("event")) == "allocator_C4_after_state"],
        key=lambda row: _mark_state_index(row, default=-1) or 0,
    )
    c4_enter = by_event.get("allocator_C4_enter")
    series: list[dict[str, Any]] = []
    prev = c4_enter
    for row in bucket_marks:
        if prev is None:
            prev = row
            continue
        prev_rss = _mark_rss_kib(prev)
        curr_rss = _mark_rss_kib(row)
        prev_sources = _allocator_source_scalars(dict(prev.get("allocator_probe") or {}))
        curr_sources = _allocator_source_scalars(dict(row.get("allocator_probe") or {}))
        prev_idx = _mark_state_index(prev, default=-1)
        curr_idx = _mark_state_index(row)
        if curr_idx is None:
            continue
        states = int(curr_idx) - int(prev_idx if prev_idx is not None else -1)
        if states <= 0:
            states = 4
        delta_rss_gib = _delta_gib(curr_rss, prev_rss, kib=True)
        source_deltas: dict[str, float | None] = {}
        for name, curr_val, prev_val, kib in (
            ("anonymous", curr_sources.get("anonymous_kb"), prev_sources.get("anonymous_kb"), True),
            (
                "private_dirty",
                curr_sources.get("private_dirty_kb"),
                prev_sources.get("private_dirty_kb"),
                True,
            ),
            (
                "glibc_uordblks",
                curr_sources.get("glibc_uordblks_bytes"),
                prev_sources.get("glibc_uordblks_bytes"),
                False,
            ),
            (
                "cuda_host_active",
                curr_sources.get("cuda_host_active_bytes"),
                prev_sources.get("cuda_host_active_bytes"),
                False,
            ),
        ):
            source_deltas[name] = _delta_gib(
                int(curr_val) if curr_val is not None else None,
                int(prev_val) if prev_val is not None else None,
                kib=kib,
            )
        dominance: dict[str, float | None] = {}
        if delta_rss_gib is not None and abs(delta_rss_gib) > 1e-9:
            for name, delta in source_deltas.items():
                dominance[name] = (
                    abs(float(delta)) / abs(float(delta_rss_gib))
                    if delta is not None
                    else None
                )
        per_state: dict[str, float | None] = {}
        if delta_rss_gib is not None:
            per_state["rss_gib_per_state"] = float(delta_rss_gib) / float(states)
            for name, delta in source_deltas.items():
                if delta is not None:
                    per_state[f"{name}_gib_per_state"] = float(delta) / float(states)
        series.append(
            {
                "state_index_end": int(curr_idx),
                "state_bucket": int(row.get("state_bucket") or 0),
                "states_in_bucket": int(states),
                "delta_rss_gib": delta_rss_gib,
                "source_deltas_gib": source_deltas,
                "dominance_ratios_vs_rss": dominance,
                "per_state_slopes_gib": per_state,
            }
        )
        prev = row

    avg_rss_per_state = None
    if series:
        slopes = [
            float(row["per_state_slopes_gib"]["rss_gib_per_state"])
            for row in series
            if row.get("per_state_slopes_gib", {}).get("rss_gib_per_state") is not None
        ]
        if slopes:
            avg_rss_per_state = sum(slopes) / len(slopes)

    source_avg_dominance: dict[str, float] = {}
    for source_name in ("anonymous", "private_dirty", "glibc_uordblks", "cuda_host_active"):
        ratios = [
            float(row["dominance_ratios_vs_rss"][source_name])
            for row in series
            if row.get("dominance_ratios_vs_rss", {}).get(source_name) is not None
        ]
        if ratios:
            source_avg_dominance[source_name] = sum(ratios) / len(ratios)

    dominant_source = None
    dominant_dominance = 0.0
    for name, avg in source_avg_dominance.items():
        if avg > dominant_dominance:
            dominant_dominance = avg
            dominant_source = name

    mechanism_owner_status = "UNMAPPED_OR_UNRESOLVED"
    allocation_source: str | None = None
    call_site_status = "UNRESOLVED"
    call_site_id: str | None = None
    call_site_origin: str | None = None
    remainder_status = "overlap_unknown"
    next_probe_route: str | None = "alloc_hook_proc_maps_diff"

    if dominant_source is not None and dominant_dominance >= ALLOCATOR_TIER_A_DOMINANCE:
        mechanism_owner_status = "RESOLVED"
        allocation_source = str(dominant_source)
        if dominant_source == "cuda_host_active":
            allocation_source = "cuda_host_caching_allocator"
        elif dominant_source == "glibc_uordblks":
            allocation_source = "glibc_malloc_arena"
        elif dominant_source in {"anonymous", "private_dirty"}:
            allocation_source = f"smaps_{dominant_source}_symptom_only"
            mechanism_owner_status = "UNMAPPED_OR_UNRESOLVED"
            next_probe_route = "alloc_hook_proc_maps_diff"

    site_deltas: list[dict[str, Any]] = []
    sampled = [
        row
        for row in site_marks
        if _mark_state_index(row, default=-1) == 0
    ]
    by_site: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sampled:
        by_site[str(row.get("site_id"))].append(row)
    for site_id, rows in sorted(by_site.items()):
        pre = next((row for row in rows if str(row.get("event", "")).endswith("_pre")), None)
        post = next((row for row in rows if str(row.get("event", "")).endswith("_post")), None)
        if pre is None or post is None:
            continue
        delta = _delta_gib(_mark_rss_kib(post), _mark_rss_kib(pre), kib=True)
        site_deltas.append(
            {
                "site_id": site_id,
                "origin_file": post.get("origin_file"),
                "origin_line": post.get("origin_line"),
                "delta_rss_gib": delta,
            }
        )
    positive_site = [row for row in site_deltas if row.get("delta_rss_gib") is not None and row["delta_rss_gib"] > 0]
    if positive_site:
        top_site = max(positive_site, key=lambda row: float(row["delta_rss_gib"]))
        total_positive = sum(float(row["delta_rss_gib"]) for row in positive_site)
        if total_positive > 0 and float(top_site["delta_rss_gib"]) / total_positive >= ALLOCATOR_TIER_B_SITE_FRACTION:
            call_site_status = "RESOLVED"
            call_site_id = str(top_site["site_id"])
            call_site_origin = f"{top_site.get('origin_file')}:{top_site.get('origin_line')}"

    host_cache_diag = classify_host_cache_empty_diagnostic(marks)

    if mechanism_owner_status == "UNMAPPED_OR_UNRESOLVED":
        call_site_status = "UNRESOLVED"
        call_site_id = None
        call_site_origin = None

    return {
        "allocator_bucket_series": series,
        "avg_rss_per_state_gib": avg_rss_per_state,
        "source_avg_dominance_ratios": source_avg_dominance,
        "dominant_allocator_source": dominant_source,
        "dominant_allocator_dominance_ratio": round(dominant_dominance, 4),
        "mechanism_owner_status": mechanism_owner_status,
        "allocation_source": allocation_source,
        "call_site_status": call_site_status,
        "call_site_id": call_site_id,
        "call_site_origin_file_line": call_site_origin,
        "intra_state_site_deltas": site_deltas,
        "overlap_accounting": {
            "remainder_status": remainder_status,
            "exclusive_host_source_model": False,
            "note": "individual source deltas reported; NOT naively summed (FOLD D)",
            "cuda_gpu_allocated_role": "negative_control_not_host_rss_contributor",
        },
        "host_cache_empty_diagnostic": host_cache_diag,
        "next_probe_route": next_probe_route,
        "c4_delta_rss_gib_reference": c4_delta_rss_gib,
    }


def classify_host_cache_empty_diagnostic(
    marks: list[dict[str, Any]],
) -> dict[str, Any]:
    pre = next(
        (row for row in marks if str(row.get("event")) == "allocator_host_cache_pre_empty"),
        None,
    )
    post = next(
        (row for row in marks if str(row.get("event")) == "allocator_host_cache_post_empty"),
        None,
    )
    if pre is None or post is None:
        return {
            "classification": "NOT_RUN",
            "measurement_perturbed": True,
        }
    pre_rss = _rss_gib(dict(pre.get("resource_snapshot") or {}))
    post_rss = _rss_gib(dict(post.get("resource_snapshot") or {}))
    trim_delta = None
    if pre_rss is not None and post_rss is not None:
        trim_delta = float(pre_rss) - float(post_rss)
    post_dims = dict(post.get("allocation_dims") or {}).get("host_cache_diag") or {}
    empty_cache_status = post_dims.get("status")
    status_ok = str(empty_cache_status or "").lower() in {"ok", "success"}

    classification = "INCONCLUSIVE"
    if empty_cache_status is None:
        classification = "API_UNAVAILABLE"
    elif not status_ok:
        classification = "INCONCLUSIVE"
    elif trim_delta is not None:
        if trim_delta >= HOST_CACHE_CONFIRM_DROP_GIB:
            classification = "CUDA_HOST_CACHE_CONFIRMED"
        elif trim_delta <= 0.25:
            classification = "LIVE_RESIDENT"
    return {
        "classification": classification,
        "measurement_perturbed": True,
        "pre_rss_gib": pre_rss,
        "post_rss_gib": post_rss,
        "trim_delta_rss_gib": round(trim_delta, 4) if trim_delta is not None else None,
        "empty_cache_status": empty_cache_status,
        "cache_falsified": bool(
            status_ok and trim_delta is not None and trim_delta <= 0.25
        ),
    }


def _is_census_mark(row: Mapping[str, Any]) -> bool:
    return str(row.get("schema", "")) == PROFILE_HOST_RSS_CENSUS_SCHEMA


def _census_unique_bytes(row: Mapping[str, Any] | None) -> int | None:
    if row is None:
        return None
    census = row.get("torch_census") or {}
    value = census.get("unique_storage_bytes")
    return int(value) if value is not None else None


def _group_key_from_row(group: Mapping[str, Any]) -> tuple[str, str, tuple[int, ...]]:
    return (
        str(group.get("device")),
        str(group.get("dtype")),
        tuple(int(dim) for dim in (group.get("shape") or [])),
    )


def _top_group_map(row: Mapping[str, Any] | None) -> dict[tuple[str, str, tuple[int, ...]], int]:
    if row is None:
        return {}
    census = row.get("torch_census") or {}
    out: dict[tuple[str, str, tuple[int, ...]], int] = {}
    for group in census.get("top_groups") or []:
        key = _group_key_from_row(group)
        out[key] = int(group.get("unique_storage_bytes") or 0)
    return out


def attribute_torch_census_profile(
    census_marks: list[dict[str, Any]],
    *,
    c4_delta_rss_gib: float | None,
) -> dict[str, Any]:
    marks = [row for row in census_marks if _is_census_mark(row)]
    by_event: dict[str, dict[str, Any]] = {}
    for row in marks:
        by_event[str(row.get("event"))] = row

    c3_exit = by_event.get("census_C3_exit")
    c4_enter = by_event.get("census_C4_enter")
    c4_exit = by_event.get("census_C4_exit")

    c3_bytes = _census_unique_bytes(c3_exit)
    c4_enter_bytes = _census_unique_bytes(c4_enter)
    c4_exit_bytes = _census_unique_bytes(c4_exit)

    handoff_unique_bytes = None
    loop_unique_bytes = None
    total_c4_unique_bytes = None
    if c3_bytes is not None and c4_enter_bytes is not None:
        handoff_unique_bytes = int(c4_enter_bytes) - int(c3_bytes)
    if c4_enter_bytes is not None and c4_exit_bytes is not None:
        loop_unique_bytes = int(c4_exit_bytes) - int(c4_enter_bytes)
    if c3_bytes is not None and c4_exit_bytes is not None:
        total_c4_unique_bytes = int(c4_exit_bytes) - int(c3_bytes)

    enter_groups = _top_group_map(c4_enter)
    exit_groups = _top_group_map(c4_exit)
    group_growth: list[dict[str, Any]] = []
    for key, exit_unique in sorted(exit_groups.items(), key=lambda item: item[1], reverse=True):
        enter_unique = int(enter_groups.get(key, 0))
        growth = int(exit_unique) - enter_unique
        device, dtype, shape = key
        group_growth.append(
            {
                "device": device,
                "dtype": dtype,
                "shape": list(shape),
                "unique_storage_bytes_at_enter": enter_unique,
                "unique_storage_bytes_at_exit": int(exit_unique),
                "unique_storage_growth_bytes": growth,
            }
        )

    dominant_group = group_growth[0] if group_growth else None
    dominant_growth = (
        int(dominant_group["unique_storage_growth_bytes"]) if dominant_group is not None else None
    )

    observed_rss_bytes = (
        int(float(c4_delta_rss_gib) * (1024.0**3))
        if c4_delta_rss_gib is not None
        else None
    )
    reconcile_bytes = loop_unique_bytes
    reconcile_source = "loop_unique_storage_c4_enter_to_exit"
    if reconcile_bytes is None:
        reconcile_bytes = total_c4_unique_bytes
        reconcile_source = "total_unique_storage_c3_exit_to_c4_exit"

    ratio = None
    status = "UNMAPPED_OR_UNRESOLVED"
    if observed_rss_bytes is not None and reconcile_bytes is not None and observed_rss_bytes > 0:
        ratio = float(reconcile_bytes) / float(observed_rss_bytes)
        if CENSUS_RECONCILE_RATIO_MIN <= ratio <= CENSUS_RECONCILE_RATIO_MAX:
            status = "PASS"
        else:
            status = "FAIL"

    mechanism_owner_status = "UNMAPPED_OR_UNRESOLVED"
    culprit_class: str | None = None
    culprit_class_status = "UNRESOLVED"
    dominant_allocation: dict[str, Any] | None = None
    allocation_site_id: str | None = None
    next_probe_route: str | None = "allocator_native_smaps_anonymous"

    if status == "PASS" and dominant_group is not None:
        if handoff_unique_bytes is not None and loop_unique_bytes is not None:
            if abs(int(handoff_unique_bytes)) > abs(int(loop_unique_bytes)):
                allocation_site_id = "C4_enter"
            else:
                allocation_site_id = "C4_after_state"
        else:
            allocation_site_id = "C4_exit"
        origin_file_line, origin_label = ALLOCATION_SITE_ORIGINS.get(
            str(allocation_site_id),
            ("unknown", "unknown"),
        )
        shape = dominant_group["shape"]
        element_count = 1
        for dim in shape:
            element_count *= int(dim)
        dtype = str(dominant_group["dtype"])
        bytes_per_element = 4 if "float32" in dtype else 2 if "int16" in dtype else 1
        dominant_allocation = {
            "site_id": allocation_site_id,
            "origin_file_line": origin_file_line,
            "origin_label": origin_label,
            "dtype": dtype,
            "shape": shape,
            "element_count": int(element_count),
            "bytes_per_storage_instance": int(dominant_group["unique_storage_bytes_at_exit"]),
            "instance_count": int(dominant_group.get("unique_storage_count") or 1),
            "unique_storage_growth_bytes": int(dominant_growth or 0),
            "unique_storage_bytes_at_exit": int(dominant_group["unique_storage_bytes_at_exit"]),
        }
        mechanism_owner_status = "RESOLVED"
        culprit_class = "C"
        culprit_class_status = "RESOLVED"
        next_probe_route = None

    return {
        "census_mark_count": len(marks),
        "census_events_seen": sorted(by_event.keys()),
        "census_boundaries": {
            "c3_exit_unique_storage_bytes": c3_bytes,
            "c4_enter_unique_storage_bytes": c4_enter_bytes,
            "c4_exit_unique_storage_bytes": c4_exit_bytes,
            "handoff_unique_storage_bytes": handoff_unique_bytes,
            "loop_unique_storage_bytes": loop_unique_bytes,
            "total_c4_unique_storage_bytes": total_c4_unique_bytes,
        },
        "group_growth_top": group_growth[:10],
        "dominant_group_growth": dominant_group,
        "dimensional_reconciliation_unique_storage": {
            "observed_c4_delta_rss_bytes": observed_rss_bytes,
            "reconcile_unique_storage_bytes": reconcile_bytes,
            "reconcile_source": reconcile_source,
            "ratio": round(ratio, 4) if ratio is not None else None,
            "status": status,
        },
        "dominant_allocation": dominant_allocation,
        "mechanism_owner_status": mechanism_owner_status,
        "mechanism_allocation_id": allocation_site_id,
        "culprit_class": culprit_class,
        "culprit_class_status": culprit_class_status,
        "next_probe_route": next_probe_route,
        "measurement_perturbed": True,
    }


def _phase_class_candidate_hint(phase_name: str) -> str | None:
    if phase_name in PHASE_CLASS_CANDIDATE_HINTS:
        return PHASE_CLASS_CANDIDATE_HINTS[phase_name]
    if phase_name.startswith("activation_credit") or phase_name == "two_tier_grad_proxy_ingress":
        return "E"
    return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _parse_run_log_phases(run_log: Path) -> tuple[dict[tuple[str, Any], float], set[str]]:
    totals: dict[tuple[str, Any], float] = defaultdict(float)
    seen: set[str] = set()
    if not run_log.is_file():
        return totals, seen
    for line in run_log.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        phase = row.get("phase")
        if phase:
            seen.add(str(phase))
        if row.get("event") == "end" and "duration_seconds" in row and phase:
            totals[(str(phase), row.get("step"))] += float(row["duration_seconds"])
    return totals, seen


def _sum_phase_wall(totals: Mapping[tuple[str, Any], float], phase: str) -> float:
    return sum(value for (name, _), value in totals.items() if name == phase)


def extract_run_root(run_root: Path) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema": EXTRACT_SCHEMA,
        "run_root": str(run_root),
        "arms": [],
    }
    argv_path = run_root / "prelaunch" / "argv_witness_receipt.json"
    if argv_path.is_file():
        report["argv_witness"] = json.loads(argv_path.read_text(encoding="utf-8"))
    for arm in ("baseline_snapshot_off", "instrumented_snapshot_on"):
        arm_dir = run_root / arm
        arm_report: dict[str, Any] = {"arm": arm, "exists": arm_dir.is_dir()}
        if not arm_dir.is_dir():
            report["arms"].append(arm_report)
            continue
        totals, seen = _parse_run_log_phases(arm_dir / "run.log")
        arm_report["top_wall"] = [
            {"phase": phase, "step": step, "seconds": round(seconds, 3)}
            for (phase, step), seconds in sorted(
                totals.items(), key=lambda item: -item[1]
            )[:12]
        ]
        arm_report["phase_wall_totals"] = {
            phase: round(_sum_phase_wall(totals, phase), 3)
            for phase in sorted({name for name, _ in totals})
        }
        arm_report["target_phase_present"] = {
            phase: phase in seen for phase in TARGET_PHASES
        }
        lap = arm_dir / "last_active_phase.json"
        if lap.is_file():
            arm_report["last_active"] = json.loads(lap.read_text(encoding="utf-8"))
        cuda_path = arm_dir / "cuda_memory_snapshots.jsonl"
        cuda_rows = _read_jsonl(cuda_path)
        arm_report["cuda_snapshot_count"] = len(cuda_rows)
        if cuda_rows:
            arm_report["cuda_max_allocated_gib"] = round(
                max(row.get("cuda_max_allocated_bytes", 0) for row in cuda_rows)
                / (1024**3),
                4,
            )
        receipt = arm_dir / "receipt.json"
        if receipt.is_file():
            arm_report["receipt_bytes"] = receipt.stat().st_size
            arm_report["receipt_rss_count"] = len(
                re.findall(r'"rss_kib"\s*:\s*(\d+)', receipt.read_text(encoding="utf-8"))
            )
        profile_path = arm_dir / HOST_RSS_PROFILE_JSONL_NAME
        arm_report["host_rss_profile_path"] = str(profile_path)
        arm_report["host_rss_profile_mark_count"] = len(_read_jsonl(profile_path))
        report["arms"].append(arm_report)
    return report


def _phase_key(row: Mapping[str, Any]) -> tuple[str, Any]:
    return (str(row.get("phase")), row.get("step"))


def _subphase_key(row: Mapping[str, Any]) -> tuple[str, Any]:
    return (str(row.get("sub_phase")), row.get("step"))


def _rss_gib(snapshot: Mapping[str, Any]) -> float | None:
    rss_kib = snapshot.get("rss_kib")
    if rss_kib is None:
        return None
    return float(rss_kib) / (1024.0 * 1024.0)


def _is_parent_phase_mark(row: Mapping[str, Any]) -> bool:
    return row.get("sub_phase") is None


def _is_subphase_mark(row: Mapping[str, Any]) -> bool:
    return row.get("sub_phase") is not None


def _compute_phase_deltas(
    marks: list[dict[str, Any]],
    *,
    key_fn: Any,
    phase_filter: Any | None = None,
) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, Any], list[dict[str, Any]]] = defaultdict(list)
    for row in marks:
        if phase_filter is not None and not phase_filter(row):
            continue
        by_key[key_fn(row)].append(row)

    phase_deltas: list[dict[str, Any]] = []
    for key, rows in sorted(by_key.items()):
        phase, step = key
        enter = next((row for row in rows if row.get("event") == "enter"), None)
        exit_row = next((row for row in rows if row.get("event") == "exit"), None)
        if enter is None or exit_row is None:
            continue
        enter_snap = dict(enter.get("resource_snapshot") or {})
        exit_snap = dict(exit_row.get("resource_snapshot") or {})
        delta_rss_gib = None
        if enter_snap.get("rss_kib") is not None and exit_snap.get("rss_kib") is not None:
            delta_rss_gib = (float(exit_snap["rss_kib"]) - float(enter_snap["rss_kib"])) / (
                1024.0 * 1024.0
            )
        exit_allocation_dims = dict(exit_row.get("allocation_dims") or {})
        phase_deltas.append(
            {
                "phase": phase,
                "step": step,
                "delta_rss_gib": delta_rss_gib,
                "enter_rss_gib": _rss_gib(enter_snap),
                "exit_rss_gib": _rss_gib(exit_snap),
                "exit_pss_gib": (
                    float(exit_snap["pss_kib"]) / (1024.0 * 1024.0)
                    if exit_snap.get("pss_kib") is not None
                    else None
                ),
                "exit_uss_gib": (
                    float(exit_snap["uss_kib"]) / (1024.0 * 1024.0)
                    if exit_snap.get("uss_kib") is not None
                    else None
                ),
                "allocation_dims": exit_allocation_dims or None,
                "measurement_perturbed": bool(
                    exit_row.get("measurement_perturbed", False)
                ),
            }
        )
    return phase_deltas


def _expected_raw_bytes(allocation_dims: Mapping[str, Any]) -> int:
    candidates = (
        allocation_dims.get("expected_raw_bytes_shape_stub"),
        allocation_dims.get("expected_raw_bytes_q_tensors"),
        allocation_dims.get("expected_raw_bytes_float32_q"),
    )
    values = [int(v) for v in candidates if v is not None]
    return max(values) if values else 0


def _dimensional_reconciliation(
    *,
    delta_rss_gib: float | None,
    allocation_dims: Mapping[str, Any] | None,
) -> dict[str, Any]:
    dims = dict(allocation_dims or {})
    expected_raw_bytes = _expected_raw_bytes(dims)
    observed_bytes = (
        int(float(delta_rss_gib) * (1024.0**3)) if delta_rss_gib is not None else None
    )
    plausible = False
    if expected_raw_bytes > 0 and observed_bytes is not None:
        plausible = observed_bytes <= int(expected_raw_bytes * DIMENSIONAL_OVERHEAD_FACTOR)
    return {
        "dtype": dims.get("dtype_q_levels") or dims.get("dtype_q") or dims.get("shape_stub_dtype"),
        "shape": None,
        "element_count": dims.get("shape_stub_element_count") or dims.get("total_q_numel"),
        "instance_count": (
            dims.get("shape_stub_instance_count")
            or dims.get("n_q_held")
            or dims.get("n_next_states")
            or dims.get("n_cap_inputs")
        ),
        "expected_raw_bytes": expected_raw_bytes,
        "observed_rss_bytes": observed_bytes,
        "rss_plausible_for_mechanism": plausible,
        "allocation_site": dims.get("allocation_site"),
    }


def attribute_subphase_rss_profile(
    marks: list[dict[str, Any]],
) -> dict[str, Any]:
    unperturbed = [
        row
        for row in marks
        if _is_subphase_mark(row) and not bool(row.get("measurement_perturbed", False))
    ]
    subphase_deltas = _compute_phase_deltas(
        unperturbed,
        key_fn=_subphase_key,
        phase_filter=_is_subphase_mark,
    )
    for row in subphase_deltas:
        row["mechanism_hint"] = SUBPHASE_MECHANISM_HINTS.get(str(row["phase"]))
        row["dimensional_reconciliation"] = _dimensional_reconciliation(
            delta_rss_gib=row.get("delta_rss_gib"),
            allocation_dims=row.get("allocation_dims") or {},
        )

    positive = [
        row
        for row in subphase_deltas
        if row.get("delta_rss_gib") is not None and float(row["delta_rss_gib"]) > 0
    ]
    dominant = max(positive, key=lambda row: float(row["delta_rss_gib"])) if positive else None

    signed_sum = sum(
        float(row["delta_rss_gib"])
        for row in subphase_deltas
        if row.get("delta_rss_gib") is not None
    )
    positive_sum = sum(
        float(row["delta_rss_gib"])
        for row in positive
    )

    return {
        "subphase_deltas": subphase_deltas,
        "dominant_subphase_owner": (
            str(dominant["phase"]) if dominant is not None else None
        ),
        "dominant_subphase_delta_rss_gib": (
            float(dominant["delta_rss_gib"]) if dominant is not None else None
        ),
        "signed_subphase_sum_rss_gib": signed_sum,
        "positive_subphase_sum_rss_gib": positive_sum,
        "largest_positive_subphase": (
            {
                "sub_phase": dominant["phase"],
                "delta_rss_gib": dominant["delta_rss_gib"],
                "dimensional_reconciliation": dominant["dimensional_reconciliation"],
            }
            if dominant is not None
            else None
        ),
    }


def classify_live_vs_resident_diagnostic(
    marks: list[dict[str, Any]],
) -> dict[str, Any]:
    perturbed = [
        row
        for row in marks
        if bool(row.get("measurement_perturbed", False))
        and str(row.get("sub_phase")) == LIVE_RESIDENT_DIAGNOSTIC_SUBPHASE
    ]
    pre = next(
        (row for row in perturbed if row.get("event") == "live_resident_pre_trim"),
        None,
    )
    post = next(
        (row for row in perturbed if row.get("event") == "live_resident_post_trim"),
        None,
    )
    if pre is None or post is None:
        return {
            "live_vs_resident_classification": "INCONCLUSIVE",
            "live_vs_resident_verdict_source": "missing_diagnostic_marks",
            "measurement_perturbed": True,
            "pre_trim_rss_gib": _rss_gib(dict(pre.get("resource_snapshot") or {}))
            if pre is not None
            else None,
            "post_trim_rss_gib": _rss_gib(dict(post.get("resource_snapshot") or {}))
            if post is not None
            else None,
            "trim_delta_rss_gib": None,
            "next_fix_type": None,
        }

    pre_rss = _rss_gib(dict(pre.get("resource_snapshot") or {}))
    post_rss = _rss_gib(dict(post.get("resource_snapshot") or {}))
    if pre_rss is None or post_rss is None:
        return {
            "live_vs_resident_classification": "INCONCLUSIVE",
            "live_vs_resident_verdict_source": "missing_rss_snapshot",
            "measurement_perturbed": True,
            "pre_trim_rss_gib": pre_rss,
            "post_trim_rss_gib": post_rss,
            "trim_delta_rss_gib": None,
            "next_fix_type": None,
        }

    trim_delta = float(pre_rss) - float(post_rss)
    classification = "INCONCLUSIVE"
    next_fix_type: str | None = None
    if trim_delta >= float(PROFILE_HOST_RSS_LIVE_RESIDENT_DROP_GIB):
        classification = "ALLOCATOR_RETENTION"
        next_fix_type = "allocator_trim"
    elif trim_delta <= 0.25:
        classification = "LIVE_ALLOCATION"
        next_fix_type = "materialization_shape"

    return {
        "live_vs_resident_classification": classification,
        "live_vs_resident_verdict_source": "c4_post_apply_malloc_trim_diagnostic",
        "measurement_perturbed": True,
        "pre_trim_rss_gib": round(pre_rss, 4),
        "post_trim_rss_gib": round(post_rss, 4),
        "trim_delta_rss_gib": round(trim_delta, 4),
        "next_fix_type": next_fix_type,
    }


def reconcile_parent_subphases(
    *,
    parent_delta_rss_gib: float | None,
    subphase_attribution: Mapping[str, Any],
) -> dict[str, Any]:
    signed_sum = float(subphase_attribution.get("signed_subphase_sum_rss_gib") or 0.0)
    positive_sum = float(subphase_attribution.get("positive_subphase_sum_rss_gib") or 0.0)
    parent = float(parent_delta_rss_gib) if parent_delta_rss_gib is not None else None
    unmapped = None
    status = "UNRESOLVED_SUBPHASE_REQUIRED"
    if parent is not None:
        unmapped = parent - signed_sum
        if not subphase_attribution.get("subphase_deltas"):
            status = "UNRESOLVED_SUBPHASE_REQUIRED"
        elif positive_sum / parent < SUBPHASE_UNMAPPED_FRACTION:
            status = "UNMAPPED_OR_UNRESOLVED"
        else:
            status = "RECONCILED_PARTIAL"
    return {
        "parent_delta_rss_gib": parent,
        "parent_delta_pss_gib": None,
        "parent_delta_uss_gib": None,
        "signed_subphase_sum_rss_gib": signed_sum,
        "positive_subphase_sum_rss_gib": positive_sum,
        "unmapped_remainder_rss_gib": unmapped,
        "reconciliation_status": status,
    }


def resolve_mechanism_owner(
    *,
    parent_delta_rss_gib: float | None,
    subphase_attribution: Mapping[str, Any],
    reconciliation: Mapping[str, Any],
) -> dict[str, Any]:
    parent = parent_delta_rss_gib
    dominant_subphase = subphase_attribution.get("dominant_subphase_owner")
    dominant_delta = subphase_attribution.get("dominant_subphase_delta_rss_gib")
    largest = subphase_attribution.get("largest_positive_subphase")

    mechanism_owner_status = "UNRESOLVED_SUBPHASE_REQUIRED"
    mechanism_allocation_id: str | None = None
    culprit_class: str | None = None
    culprit_class_status = "UNRESOLVED"
    live_vs_resident = "unperturbed_primary"
    live_vs_resident_diagnostic: dict[str, Any] | None = None

    if parent is None or parent <= 0 or dominant_subphase is None or dominant_delta is None:
        return {
            "mechanism_owner_status": mechanism_owner_status,
            "mechanism_allocation_id": mechanism_allocation_id,
            "dominant_subphase_owner": dominant_subphase,
            "culprit_class": culprit_class,
            "culprit_class_status": culprit_class_status,
            "live_vs_resident_classification": live_vs_resident,
            "live_vs_resident_diagnostic": live_vs_resident_diagnostic,
        }

    recon_status = str(reconciliation.get("reconciliation_status"))
    positive_sum = float(subphase_attribution.get("positive_subphase_sum_rss_gib") or 0.0)
    fraction_of_parent = float(dominant_delta) / float(parent)

    dim = (largest or {}).get("dimensional_reconciliation") or {}
    plausible = bool(dim.get("rss_plausible_for_mechanism"))

    if recon_status == "UNMAPPED_OR_UNRESOLVED":
        mechanism_owner_status = "UNMAPPED_OR_UNRESOLVED"
    elif fraction_of_parent >= SUBPHASE_RESOLVE_FRACTION and plausible:
        mechanism_owner_status = "RESOLVED"
        mechanism_allocation_id = str(dominant_subphase)
        culprit_class = "C"
        culprit_class_status = "RESOLVED"
    elif positive_sum / float(parent) >= SUBPHASE_UNMAPPED_FRACTION:
        mechanism_owner_status = "UNMAPPED_OR_UNRESOLVED"
    else:
        mechanism_owner_status = "UNRESOLVED_SUBPHASE_REQUIRED"

    return {
        "mechanism_owner_status": mechanism_owner_status,
        "mechanism_allocation_id": mechanism_allocation_id,
        "dominant_subphase_owner": dominant_subphase,
        "dominant_subphase_delta_rss_gib": dominant_delta,
        "dominant_subphase_fraction_of_parent": round(fraction_of_parent, 4),
        "culprit_class": culprit_class,
        "culprit_class_status": culprit_class_status,
        "live_vs_resident_classification": live_vs_resident,
        "live_vs_resident_diagnostic": live_vs_resident_diagnostic,
        "dimensional_reconciliation": dim,
    }


def attribute_host_rss_profile(
    marks: list[dict[str, Any]],
    *,
    wall_totals: Mapping[str, float] | None = None,
    diagnostic_marks: list[dict[str, Any]] | None = None,
    census_marks: list[dict[str, Any]] | None = None,
    allocator_marks: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    parent_marks = [row for row in marks if _is_parent_phase_mark(row)]
    phase_deltas = _compute_phase_deltas(
        parent_marks,
        key_fn=_phase_key,
        phase_filter=_is_parent_phase_mark,
    )

    positive = [
        row
        for row in phase_deltas
        if row.get("delta_rss_gib") is not None
        and row["delta_rss_gib"] > 0
        and str(row["phase"]) in RSS_ATTRIBUTION_LEAF_PHASES
    ]
    dominant_rss = None
    if positive:
        dominant_rss = max(positive, key=lambda row: float(row["delta_rss_gib"]))

    wall_owner = None
    if wall_totals:
        if wall_totals:
            top_phase = max(wall_totals.items(), key=lambda item: item[1])[0]
            wall_owner = {
                "phase": top_phase,
                "seconds": wall_totals[top_phase],
            }

    dominant_phase_owner: str | None = None
    phase_class_candidate_hint: str | None = None
    falsified_mechanism: str | None = None
    next_candidate_class: str | None = None
    if dominant_rss is not None:
        dominant_phase_owner = str(dominant_rss["phase"])
        phase_class_candidate_hint = _phase_class_candidate_hint(dominant_phase_owner)
        if dominant_phase_owner == "sparse_cap_apply":
            falsified_mechanism = "A"
            next_candidate_class = "C"

    parent_sparse_cap_delta = None
    for row in phase_deltas:
        if str(row["phase"]) == "sparse_cap_apply" and row.get("delta_rss_gib") is not None:
            parent_sparse_cap_delta = float(row["delta_rss_gib"])

    subphase_attribution = attribute_subphase_rss_profile(marks)
    reconciliation = reconcile_parent_subphases(
        parent_delta_rss_gib=parent_sparse_cap_delta,
        subphase_attribution=subphase_attribution,
    )
    mechanism = resolve_mechanism_owner(
        parent_delta_rss_gib=parent_sparse_cap_delta,
        subphase_attribution=subphase_attribution,
        reconciliation=reconciliation,
    )

    if diagnostic_marks:
        diagnostic = classify_live_vs_resident_diagnostic(diagnostic_marks)
        mechanism["live_vs_resident_diagnostic"] = diagnostic
        verdict = str(diagnostic.get("live_vs_resident_classification"))
        if verdict in {"ALLOCATOR_RETENTION", "LIVE_ALLOCATION", "INCONCLUSIVE"}:
            mechanism["live_vs_resident_classification"] = verdict

    c4_delta = None
    for row in subphase_attribution.get("subphase_deltas") or []:
        if str(row.get("phase")) == "C4_gpu_cap_apply_sync":
            c4_delta = row.get("delta_rss_gib")
            break

    census_attribution: dict[str, Any] | None = None
    if census_marks:
        census_attribution = attribute_torch_census_profile(
            census_marks,
            c4_delta_rss_gib=float(c4_delta) if c4_delta is not None else parent_sparse_cap_delta,
        )
        if census_attribution.get("mechanism_owner_status") == "RESOLVED":
            mechanism["mechanism_owner_status"] = "RESOLVED"
            mechanism["mechanism_allocation_id"] = census_attribution.get("mechanism_allocation_id")
            mechanism["culprit_class"] = census_attribution.get("culprit_class")
            mechanism["culprit_class_status"] = census_attribution.get("culprit_class_status")
            mechanism["dominant_allocation"] = census_attribution.get("dominant_allocation")
        elif census_attribution.get("mechanism_owner_status") == "UNMAPPED_OR_UNRESOLVED":
            mechanism["mechanism_owner_status"] = "UNMAPPED_OR_UNRESOLVED"
            mechanism["culprit_class"] = None
            mechanism["culprit_class_status"] = "UNRESOLVED"
            mechanism["next_probe_route"] = census_attribution.get("next_probe_route")

    allocator_attribution: dict[str, Any] | None = None
    if allocator_marks:
        allocator_attribution = attribute_allocator_native_profile(
            allocator_marks,
            c4_delta_rss_gib=float(c4_delta) if c4_delta is not None else parent_sparse_cap_delta,
        )
        if allocator_attribution.get("mechanism_owner_status") == "RESOLVED":
            mechanism["mechanism_owner_status"] = "RESOLVED"
            mechanism["allocation_source"] = allocator_attribution.get("allocation_source")
            mechanism["culprit_class"] = "C"
            mechanism["culprit_class_status"] = "RESOLVED"
            mechanism["call_site_status"] = allocator_attribution.get("call_site_status")
            mechanism["call_site_id"] = allocator_attribution.get("call_site_id")
            mechanism["call_site_origin_file_line"] = allocator_attribution.get(
                "call_site_origin_file_line"
            )
        elif allocator_attribution.get("mechanism_owner_status") == "UNMAPPED_OR_UNRESOLVED":
            mechanism["mechanism_owner_status"] = "UNMAPPED_OR_UNRESOLVED"
            mechanism["next_probe_route"] = allocator_attribution.get("next_probe_route")
        mechanism["host_cache_empty_diagnostic"] = allocator_attribution.get(
            "host_cache_empty_diagnostic"
        )

    has_subphase_marks = any(_is_subphase_mark(row) for row in marks)
    if not has_subphase_marks:
        mechanism["mechanism_owner_status"] = "UNRESOLVED_SUBPHASE_REQUIRED"
        mechanism["mechanism_allocation_id"] = None
        mechanism["culprit_class"] = None
        mechanism["culprit_class_status"] = "UNRESOLVED"

    return {
        "phase_deltas": phase_deltas,
        "dominant_rss_owner": dominant_rss,
        "dominant_phase_owner": dominant_phase_owner,
        "dominant_wall_owner": wall_owner,
        "culprit_class": mechanism.get("culprit_class"),
        "culprit_class_name": CULPRIT_CLASSES.get(str(mechanism.get("culprit_class") or "")),
        "culprit_class_status": mechanism.get("culprit_class_status", "UNRESOLVED"),
        "phase_class_candidate_hint": phase_class_candidate_hint,
        "phase_class_candidate_hint_name": CULPRIT_CLASSES.get(
            str(phase_class_candidate_hint or ""),
            None,
        ),
        "falsified_mechanism": falsified_mechanism,
        "next_candidate_class": next_candidate_class,
        "parent_sparse_cap_delta_rss_gib": parent_sparse_cap_delta,
        "subphase_attribution": subphase_attribution,
        "reconciliation": reconciliation,
        "census_attribution": census_attribution,
        "allocator_attribution": allocator_attribution,
        **mechanism,
    }


def build_attribution_receipt(
    *,
    run_root: Path,
    profile_path: Path,
    extract_report: Mapping[str, Any] | None = None,
    diagnostic_profile_path: Path | None = None,
    census_profile_path: Path | None = None,
    allocator_profile_path: Path | None = None,
) -> dict[str, Any]:
    marks = _read_jsonl(profile_path)
    diagnostic_marks = (
        _read_jsonl(diagnostic_profile_path) if diagnostic_profile_path is not None else None
    )
    census_marks = (
        _read_jsonl(census_profile_path) if census_profile_path is not None else None
    )
    allocator_marks = (
        _read_jsonl(allocator_profile_path) if allocator_profile_path is not None else None
    )
    wall_totals: dict[str, float] = {}
    if extract_report is None and run_root.is_dir():
        extract_report = extract_run_root(run_root)
    if extract_report is not None:
        for arm in extract_report.get("arms", []):
            if arm.get("arm") != "baseline_snapshot_off":
                continue
            for phase, seconds in (arm.get("phase_wall_totals") or {}).items():
                wall_totals[phase] = float(seconds)
    attribution = attribute_host_rss_profile(
        marks,
        wall_totals=wall_totals or None,
        diagnostic_marks=diagnostic_marks,
        census_marks=census_marks,
        allocator_marks=allocator_marks,
    )
    receipt: dict[str, Any] = {
        "schema": ATTRIBUTION_SCHEMA,
        "run_root": str(run_root),
        "profile_path": str(profile_path),
        "profile_mark_count": len(marks),
        "subphase_mark_count": sum(1 for row in marks if _is_subphase_mark(row)),
        "extract_report": extract_report,
        **attribution,
    }
    if diagnostic_profile_path is not None:
        receipt["diagnostic_profile_path"] = str(diagnostic_profile_path)
        receipt["diagnostic_mark_count"] = len(diagnostic_marks or [])
    if census_profile_path is not None:
        receipt["census_profile_path"] = str(census_profile_path)
        receipt["census_mark_count"] = len(census_marks or [])
    if allocator_profile_path is not None:
        receipt["allocator_profile_path"] = str(allocator_profile_path)
        receipt["allocator_mark_count"] = len(allocator_marks or [])
    if attribution["dominant_phase_owner"] is None:
        receipt["rss_phase_owner_status"] = "UNRESOLVED"
    else:
        receipt["rss_phase_owner_status"] = "RESOLVED"
    if "mechanism_owner_status" not in receipt:
        receipt["mechanism_owner_status"] = "UNRESOLVED_SUBPHASE_REQUIRED"
    receipt["rss_owner_status"] = receipt["rss_phase_owner_status"]
    return receipt


def _fixture_probe_argv(scratch_root: Path) -> list[str]:
    parent = (
        "calm/hrm/checkpoints/hrm_text_158_phase3_L0c1_seed0017_replay83_n12k_lr7p5e5_pc1p0_"
        "rsL0b1math1r1b2_1_anchorsv1r3_from_L0b_final_step01500.pt"
    )
    return [
        sys.executable,
        "-u",
        "scripts/hrm_text_158_bounded_delta_acquisition_probe.py",
        "--allow-gpu-launch",
        "--enable-bounded-delta-probe",
        "--device",
        "cuda",
        "--parent",
        parent,
        "--parent-sha256",
        "9b4e311a22787e7d4808bde7bc2953568d767a2ee8ac648942a3f5dbb7b4d5ec",
        "--curriculum-seed",
        "43",
        "--support-order-seed",
        "43",
        "--eligible-scope",
        "all-bitlinear",
        "--batch-size",
        "1",
        "--science-arm",
        "A0_rank_bucket_current_ordering",
        "--global-cap-contract",
        "c1_banked_faithful_long_run_global_cap",
        "--confirmation-envelope",
        "canonical_t10_prereg_v24",
        "--phase",
        "d-recompute-window-feasibility",
        "--emit-progress",
        "--phase-heartbeat-seconds",
        "30",
        "--persistent-q-ternary-base3-codec",
        "--persistent-accumulator-event-coded-live",
        "--event-coded-live-demotion-band",
        "1",
        "--receipt-emit-profile",
        "s3bb_headroom_diagnostic_slim",
        "--d-diagnostic-compact-step-reports",
        "--steps",
        "1",
        "--max-steps-hard",
        "1",
        "--phase-timeout-seconds",
        "2280",
        "--total-timeout-seconds",
        "5400",
        "--event-coded-sparse-vote-authority",
        "--scratch-root",
        str(scratch_root),
    ]


def run_fixture_live_resident_diagnostic(out_root: Path) -> dict[str, Any]:
    out_root.mkdir(parents=True, exist_ok=True)
    scratch = out_root / "live_resident_fixture_n1"
    scratch.mkdir(parents=True, exist_ok=True)
    lane_holding: dict[str, Any] | None = None
    lane_release: dict[str, Any] | None = None
    try:
        from scripts.hrm_text_158_r7_resource_lane_acquire import acquire_resource_lane

        lane_holding = acquire_resource_lane(out_root)
    except Exception as exc:
        lane_holding = {"acquire_error": f"{type(exc).__name__}: {exc}"}

    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    env["HRM_TEXT_158_RUN_GPU_GLOBAL_RATE_CAP"] = "1"
    env["HRM_TEXT_158_RUN_GPU_Q_ACC_APPLY"] = "1"
    env["HRM_TEXT_158_ALLOW_C2_GPU_LAUNCH"] = "1"
    env["HRM_TEXT_158_RUN_C2_ACQUISITION_PROBE"] = "1"
    env[PROFILE_HOST_RSS_ENV] = "1"
    env[PROFILE_HOST_RSS_LIVE_RESIDENT_ENV] = "1"
    cmd = _fixture_probe_argv(scratch)
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )
    profile_path = scratch / HOST_RSS_PROFILE_JSONL_NAME
    diagnostic = classify_live_vs_resident_diagnostic(_read_jsonl(profile_path))
    try:
        from scripts.hrm_text_158_r7_resource_lane_release import release_resource_lane

        lane_release = release_resource_lane(out_root)
    except Exception as exc:
        lane_release = {"release_error": f"{type(exc).__name__}: {exc}"}

    return {
        "scratch_root": str(scratch),
        "profile_path": str(profile_path),
        "command": cmd,
        "exit_code": int(proc.returncode),
        "stdout_tail": "\n".join(proc.stdout.splitlines()[-20:]),
        "stderr_tail": "\n".join(proc.stderr.splitlines()[-20:]),
        "resource_lane_holding": lane_holding,
        "resource_lane_release": lane_release,
        **diagnostic,
    }


def run_fixture_torch_census(out_root: Path) -> dict[str, Any]:
    out_root.mkdir(parents=True, exist_ok=True)
    scratch = out_root / "torch_census_fixture_n1"
    scratch.mkdir(parents=True, exist_ok=True)
    lane_holding: dict[str, Any] | None = None
    lane_release: dict[str, Any] | None = None
    try:
        from scripts.hrm_text_158_r7_resource_lane_acquire import acquire_resource_lane

        lane_holding = acquire_resource_lane(out_root)
    except Exception as exc:
        lane_holding = {"acquire_error": f"{type(exc).__name__}: {exc}"}

    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    env["HRM_TEXT_158_RUN_GPU_GLOBAL_RATE_CAP"] = "1"
    env["HRM_TEXT_158_RUN_GPU_Q_ACC_APPLY"] = "1"
    env["HRM_TEXT_158_ALLOW_C2_GPU_LAUNCH"] = "1"
    env["HRM_TEXT_158_RUN_C2_ACQUISITION_PROBE"] = "1"
    env[PROFILE_HOST_RSS_ENV] = "1"
    env[PROFILE_TORCH_CPU_CENSUS_ENV] = "1"
    cmd = _fixture_probe_argv(scratch)
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )
    profile_path = scratch / HOST_RSS_PROFILE_JSONL_NAME
    census_marks = [row for row in _read_jsonl(profile_path) if _is_census_mark(row)]
    try:
        from scripts.hrm_text_158_r7_resource_lane_release import release_resource_lane

        lane_release = release_resource_lane(out_root)
    except Exception as exc:
        lane_release = {"release_error": f"{type(exc).__name__}: {exc}"}

    return {
        "scratch_root": str(scratch),
        "profile_path": str(profile_path),
        "command": cmd,
        "exit_code": int(proc.returncode),
        "stdout_tail": "\n".join(proc.stdout.splitlines()[-20:]),
        "stderr_tail": "\n".join(proc.stderr.splitlines()[-20:]),
        "resource_lane_holding": lane_holding,
        "resource_lane_release": lane_release,
        "census_mark_count": len(census_marks),
        "census_events_seen": sorted({str(row.get("event")) for row in census_marks}),
        "measurement_perturbed": True,
    }


def run_fixture_allocator_native(out_root: Path) -> dict[str, Any]:
    out_root.mkdir(parents=True, exist_ok=True)
    scratch = out_root / "allocator_native_fixture_n1"
    scratch.mkdir(parents=True, exist_ok=True)
    lane_holding: dict[str, Any] | None = None
    lane_release: dict[str, Any] | None = None
    try:
        from scripts.hrm_text_158_r7_resource_lane_acquire import acquire_resource_lane

        lane_holding = acquire_resource_lane(out_root)
    except Exception as exc:
        lane_holding = {"acquire_error": f"{type(exc).__name__}: {exc}"}

    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    env["HRM_TEXT_158_RUN_GPU_GLOBAL_RATE_CAP"] = "1"
    env["HRM_TEXT_158_RUN_GPU_Q_ACC_APPLY"] = "1"
    env["HRM_TEXT_158_ALLOW_C2_GPU_LAUNCH"] = "1"
    env["HRM_TEXT_158_RUN_C2_ACQUISITION_PROBE"] = "1"
    env[PROFILE_HOST_RSS_ENV] = "1"
    env[PROFILE_ALLOCATOR_NATIVE_ENV] = "1"
    env[PROFILE_ALLOCATOR_HOST_CACHE_DIAG_ENV] = "1"
    cmd = _fixture_probe_argv(scratch)
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )
    profile_path = scratch / HOST_RSS_PROFILE_JSONL_NAME
    allocator_marks = [
        row
        for row in _read_jsonl(profile_path)
        if _is_allocator_mark(row) or _is_allocator_site_mark(row)
    ]
    try:
        from scripts.hrm_text_158_r7_resource_lane_release import release_resource_lane

        lane_release = release_resource_lane(out_root)
    except Exception as exc:
        lane_release = {"release_error": f"{type(exc).__name__}: {exc}"}

    return {
        "scratch_root": str(scratch),
        "profile_path": str(profile_path),
        "command": cmd,
        "exit_code": int(proc.returncode),
        "stdout_tail": "\n".join(proc.stdout.splitlines()[-20:]),
        "stderr_tail": "\n".join(proc.stderr.splitlines()[-20:]),
        "resource_lane_holding": lane_holding,
        "resource_lane_release": lane_release,
        "allocator_mark_count": len(allocator_marks),
        "allocator_events_seen": sorted({str(row.get("event")) for row in allocator_marks}),
        "measurement_perturbed": True,
    }


def run_fixture(out_root: Path) -> dict[str, Any]:
    out_root.mkdir(parents=True, exist_ok=True)
    scratch = out_root / "baseline_fixture_n1"
    scratch.mkdir(parents=True, exist_ok=True)
    lane_holding: dict[str, Any] | None = None
    lane_release: dict[str, Any] | None = None
    try:
        from scripts.hrm_text_158_r7_resource_lane_acquire import acquire_resource_lane
        from scripts.hrm_text_158_r7_resource_lane_release import release_resource_lane

        lane_holding = acquire_resource_lane(out_root)
    except Exception as exc:
        lane_holding = {"acquire_error": f"{type(exc).__name__}: {exc}"}

    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    env["HRM_TEXT_158_RUN_GPU_GLOBAL_RATE_CAP"] = "1"
    env["HRM_TEXT_158_RUN_GPU_Q_ACC_APPLY"] = "1"
    env["HRM_TEXT_158_ALLOW_C2_GPU_LAUNCH"] = "1"
    env["HRM_TEXT_158_RUN_C2_ACQUISITION_PROBE"] = "1"
    env[PROFILE_HOST_RSS_ENV] = "1"
    cmd = _fixture_probe_argv(scratch)
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )
    profile_path = scratch / HOST_RSS_PROFILE_JSONL_NAME
    receipt = build_attribution_receipt(run_root=out_root, profile_path=profile_path)
    try:
        from scripts.hrm_text_158_r7_resource_lane_release import release_resource_lane

        lane_release = release_resource_lane(out_root)
    except Exception as exc:
        lane_release = {"release_error": f"{type(exc).__name__}: {exc}"}

    receipt["fixture"] = {
        "scratch_root": str(scratch),
        "command": cmd,
        "exit_code": int(proc.returncode),
        "stdout_tail": "\n".join(proc.stdout.splitlines()[-20:]),
        "stderr_tail": "\n".join(proc.stderr.splitlines()[-20:]),
        "resource_lane_holding": lane_holding,
        "resource_lane_release": lane_release,
    }
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=(
            "extract",
            "attribute",
            "fixture",
            "fixture_live_resident",
            "fixture_torch_census",
            "fixture_allocator_native",
        ),
        required=True,
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--aborted-run-root",
        type=Path,
        default=None,
        help="Optional aborted-run extract root for combined D3 attribution",
    )
    parser.add_argument(
        "--profile-path",
        type=Path,
        default=None,
        help="host_rss_profile.jsonl for attribute mode",
    )
    parser.add_argument(
        "--diagnostic-profile-path",
        type=Path,
        default=None,
        help="Optional live-resident diagnostic host_rss_profile.jsonl for attribute mode",
    )
    parser.add_argument(
        "--allocator-profile-path",
        type=Path,
        default=None,
        help="Optional allocator-native host_rss_profile.jsonl for attribute mode",
    )
    parser.add_argument(
        "--census-profile-path",
        type=Path,
        default=None,
        help="Optional torch CPU census host_rss_profile.jsonl for attribute mode",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.mode == "extract":
        payload = extract_run_root(args.run_root)
    elif args.mode == "attribute":
        profile_path = args.profile_path or (
            args.run_root / "baseline_fixture_n1" / HOST_RSS_PROFILE_JSONL_NAME
        )
        extract_report = None
        if args.aborted_run_root is not None:
            aborted_extract = args.aborted_run_root / (
                "v6i_oom_profile_attribution_extract_readonly.json"
            )
            if aborted_extract.is_file():
                extract_report = json.loads(aborted_extract.read_text(encoding="utf-8"))
            else:
                extract_report = extract_run_root(args.aborted_run_root)
        payload = build_attribution_receipt(
            run_root=args.run_root,
            profile_path=profile_path,
            extract_report=extract_report,
            diagnostic_profile_path=args.diagnostic_profile_path,
            census_profile_path=args.census_profile_path,
            allocator_profile_path=args.allocator_profile_path,
        )
        if args.aborted_run_root is not None:
            payload["aborted_run_root"] = str(args.aborted_run_root)
    elif args.mode == "fixture_live_resident":
        payload = run_fixture_live_resident_diagnostic(args.run_root)
    elif args.mode == "fixture_torch_census":
        payload = run_fixture_torch_census(args.run_root)
    elif args.mode == "fixture_allocator_native":
        payload = run_fixture_allocator_native(args.run_root)
    else:
        payload = run_fixture(args.run_root)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "mode": args.mode}, indent=2))
    if args.mode in {
        "fixture",
        "fixture_live_resident",
        "fixture_torch_census",
        "fixture_allocator_native",
    } and int(payload.get("exit_code", payload.get("fixture", {}).get("exit_code", 1))) != 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
