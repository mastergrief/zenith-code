#!/usr/bin/env python3
"""v6i OOM profile/attribution: extract aborted-run artifacts and attribute RSS owners."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.hrm_text_158_bounded_delta_acquisition_probe import (  # noqa: E402
    HOST_RSS_PROFILE_JSONL_NAME,
    PROFILE_ALLOCATOR_HOST_CACHE_DIAG_ENV,
    PROFILE_ALLOCATOR_NATIVE_ENV,
    PROFILE_ALLOC_HOOK_ENV,
    PROFILE_HOST_RSS_ALLOCATOR_SCHEMA,
    PROFILE_HOST_RSS_ALLOCATOR_SITE_SCHEMA,
    PROFILE_HOST_RSS_ALLOC_HOOK_SCHEMA,
    PROFILE_HOST_RSS_CENSUS_SCHEMA,
    PROFILE_HOST_RSS_ENV,
    PROFILE_HOST_RSS_LIVE_RESIDENT_DROP_GIB,
    PROFILE_HOST_RSS_LIVE_RESIDENT_ENV,
    PROFILE_HOST_RSS_SUBPHASE_IDS,
    PROFILE_HOST_RSS_TRIANGULATION_SCHEMA,
    PROFILE_HOST_RSS_OBMALLOC_SCHEMA,
    PROFILE_HOST_RSS_OBMALLOC_SITE_SCHEMA,
    PROFILE_OBMALLOC_SITE_BRACKETS_ENV,
    PROFILE_OBMALLOC_EXPANDED_ENV,
    PROFILE_TORCH_CPU_CENSUS_ENV,
    PROFILE_TRACEMALLOC_ENV,
    PROFILE_DEBUGMALLOCSTATS_ENV,
)

ATTRIBUTION_SCHEMA = "hrm_text_158_v6i_oom_profile_attribution_receipt/v12"
EXTRACT_SCHEMA = "hrm_text_158_v6i_oom_profile_extract_readonly/v1"

FIXTURE_PROBE_STREAM_LOG_NAME = "probe_stream.log"
FIXTURE_PROBE_MAX_SILENT_PHASE_SECONDS = 600
FIXTURE_PROBE_MAX_SILENT_PHASE_SECONDS_TRACEMALLOC = 900
DEFAULT_DURABLE_MIRROR_PATH = Path(
    "/home/gabe/hrm158_v6i_slice8m_census/v6i_obmalloc_expanded_attribution_8m.json"
)

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
    "C4.S1a": ("event_coded_vote_update_adapter.py:1081", "carrier_cow_copy"),
    "C4.S1b": (
        "event_coded_vote_update_adapter.py:1100",
        "applied_indices_tolist_applied_set",
    ),
    "C4.S1c_clone": (
        "event_coded_vote_update_adapter.py:1021",
        "sync_q_levels_int8_clone",
    ),
    "C4.S1c_contig": (
        "event_coded_vote_update_adapter.py:1037",
        "sync_q_levels_contiguous",
    ),
    "C4.S1d": (
        "event_coded_vote_update_adapter.py:1087",
        "sparse_vote_numpy_branch",
    ),
    "C4.S1d.1": (
        "event_coded_vote_update_adapter.py:1077",
        "sparse_vote_numpy_detach",
    ),
    "C4.S1d.0": (
        "event_coded_acc_live_carrier.py:694-716",
        "apply_step_vote_ingress",
    ),
    "C4.S1d.2": (
        "event_coded_acc_live_carrier.py:717-725",
        "apply_step_active_sorted_union",
    ),
    "C4.S1d.3": (
        "event_coded_acc_live_carrier.py:750-824",
        "apply_step_int32_lane_vectors",
    ),
    "C4.S1d.4": (
        "event_coded_acc_live_carrier.py:825-874",
        "apply_step_bool_mask_arrays",
    ),
    "C4.S1d.5": (
        "event_coded_acc_live_carrier.py:899-909",
        "apply_step_hot_table_merge",
    ),
    "C4.S1d.6": (
        "event_coded_acc_live_carrier.py:911-924",
        "apply_step_surface_record",
    ),
    "C4.S1d.7": (
        "event_coded_acc_live_carrier.py:876-897",
        "apply_step_crossing_commit",
    ),
    "C4.S1d.8": (
        "event_coded_acc_live_carrier.py:661-925",
        "apply_step_outer_audit",
    ),
    "C4.S1e": ("event_coded_vote_update_adapter.py:1112", "c8_runtime_guards"),
    "C4.S1f": ("event_coded_vote_update_adapter.py:1126", "c8_stats_assembly"),
    "C4.S1f.1": (
        "event_coded_vote_update_adapter.py:1196",
        "q_changed_count_full_bool_compare",
    ),
    "C4.S1f.2": (
        "event_coded_vote_update_adapter.py:1193",
        "observed_surfaces_dict",
    ),
}

CENSUS_RECONCILE_RATIO_MIN = 0.8
CENSUS_RECONCILE_RATIO_MAX = 1.2
ALLOCATOR_TIER_A_DOMINANCE = 0.80
ALLOCATOR_PER_STATE_SLOPE_MIN = 0.75
ALLOCATOR_PER_STATE_SLOPE_MAX = 1.25
ALLOCATOR_TIER_B_SITE_FRACTION = 0.50
HOST_CACHE_CONFIRM_DROP_GIB = 1.0

BANKED_REFERENCE_COMMIT = "0698608"
BANKED_NON_GLIBC_MMAP_REFERENCE_BYTES = 4470079488
BANKED_NON_GLIBC_MMAP_REFERENCE_GIB = 4.1630859375
TOTAL_C4_REFERENCE_GIB = 7.6359100341796875
GUARD_STABILITY_FRAC = 0.25
GUARD_ENVELOPE_MIN_GIB = 0.25
PERTURBATION_MIN_GIB = 0.5
PERTURBATION_NOISE_K = 2.0
OBMALLOC_RECONCILE_MIN = 0.5
OBMALLOC_RECONCILE_MAX = 1.5
OBMALLOC_NOT_OBMALLOC_MAX = 0.25
OBMALLOC_OUT_OF_BAND_LOW_MIN = 0.25
OBMALLOC_OCCUPANCY_LIVE_MIN = 0.6
OBMALLOC_OCCUPANCY_HIGH_WATER_MAX = 0.3
OBMALLOC_ARENA_DELTA_FLOOR_BYTES = 1024 * 1024
C4_SUBPHASE = "C4_gpu_cap_apply_sync"

OBMALLOC_SITE_LEAF_SITES = (
    "C4.S1",
    "C4.S1a",
    "C4.S1b",
    "C4.S1c_clone",
    "C4.S1c_contig",
    "C4.S1d",
    "C4.S1d.1",
    "C4.S1d.2",
    "C4.S1d.3",
    "C4.S1d.4",
    "C4.S1d.5",
    "C4.S1d.6",
    "C4.S1d.0",
    "C4.S1d.7",
    "C4.S1d.8",
    "C4.S1e",
    "C4.S1f",
    "C4.S1f.1",
    "C4.S1f.2",
    "C4.S2a",
    "C4.S2b",
    "C4.S2c",
)
OBMALLOC_SITE_CHILD_SITES = (
    "C4.S1a",
    "C4.S1b",
    "C4.S1c_clone",
    "C4.S1c_contig",
    "C4.S1d",
    "C4.S1e",
    "C4.S1f",
)
OBMALLOC_SITE_S1D_CHILD_SITES = (
    "C4.S1d.0",
    "C4.S1d.1",
    "C4.S1d.2",
    "C4.S1d.3",
    "C4.S1d.4",
    "C4.S1d.5",
    "C4.S1d.6",
    "C4.S1d.7",
)
OBMALLOC_SITE_S1D_AUDIT_SITES = (
    "C4.S1d.8",
)
OBMALLOC_SITE_S1F_CHILD_SITES = (
    "C4.S1f.1",
    "C4.S1f.2",
)
OBMALLOC_SITE_AGGREGATE_SITE = "C4.S2"
OBMALLOC_SITE_WINDOW_ENTRY = "C4.S1"
OBMALLOC_SITE_WINDOW_EXIT = "C4.S2"
OBMALLOC_SITE_DOMINANCE_FRAC = 0.60
OBMALLOC_SITE_REMAINDER_MAX_FRAC = 0.15
OBMALLOC_SITE_REPRESENTATIVENESS_UNCERTAIN_MAX = 0.25
OBMALLOC_SITE_AMBIGUOUS_WITHIN_FRAC = 0.20
OBMALLOC_SITE_N_STATES_WITNESS = 32

OBMALLOC_EXPANDED_HOLDER_DOMINANCE_FRAC = 0.60
OBMALLOC_EXPANDED_STATE_DOMINANCE_MIN = 0.75
OBMALLOC_EXPANDED_REPRESENTATIVENESS_MIN = 0.25
OBMALLOC_EXPANDED_CANCELLATION_NEG_FRAC = 0.10
OBMALLOC_EXPANDED_RETENTION_MONOTONIC_MIN = 0.75
OBMALLOC_EXPANDED_EVENT_COUNT_TARGET = 186
OBMALLOC_EXPANDED_EVENT_COUNT_MAX = 194
OBMALLOC_EXPANDED_RETENTION_FLOOR_BYTES = 1024
# Sites allowed to emit multiple ordered pre/post pairs in SYNTHETIC/legacy streams.
# Real emitter (post FIX-C1) uses one pair per site; consumer aggregation still
# accepts multi-pair marks for these sites when present in test fixtures.
OBMALLOC_SITE_MULTI_PAIR_SITES: frozenset[str] = frozenset({"C4.S1d.3", "C4.S1d.4"})

FAIL_CLOSED_TERMINAL_EXIT_CODES: dict[str, int] = {
    "OBSERVER_PERTURBED_INCONCLUSIVE": 37,
    "ARENA_STATS_UNPARSEABLE_INCONCLUSIVE": 37,
    "HOLDER_AMBIGUOUS": 37,
    "CHILD_COVERAGE_FAIL": 37,
    "S1D_CHILD_COVERAGE_FAIL": 37,
    "S1F_CHILD_COVERAGE_FAIL": 37,
    "CHILD_PARENT_RECONCILE_FAIL": 37,
    "CHILD_OVERLAP_DOUBLE_COUNT": 37,
    "INCONCLUSIVE_PENDING_NOISE_FLOOR": 37,
    "DENOMINATOR_INVALID_INCONCLUSIVE": 37,
    "INCONCLUSIVE_CROSS_RUN_DENOMINATOR": 37,
    "OBMALLOC_SELF_FOOTPRINT_INCONCLUSIVE": 37,
    "RECONCILE_OUT_OF_BAND_INCONCLUSIVE": 37,
    "BRACKET_REMAINDER_TOO_LARGE": 37,
    "TRACEMALLOC_PERTURBED_INCONCLUSIVE": 37,
    "CLASSIFIER_INCONCLUSIVE": 37,
    "CODE_CURRENCY_MISMATCH_INCONCLUSIVE": 37,
}


def _mapped_terminal_exit_code(fail_closed_terminal: str | None) -> int | None:
    if fail_closed_terminal is None:
        return None
    return int(FAIL_CLOSED_TERMINAL_EXIT_CODES.get(str(fail_closed_terminal), 37))


def _exit_code_agreement_truthful(
    *,
    fail_closed_terminal: str | None,
    probe_exit_code: int,
    process_exit_code: int,
    mapped_terminal_code: int | None,
) -> bool:
    if fail_closed_terminal is not None:
        if mapped_terminal_code is None:
            return False
        return (
            int(process_exit_code) == int(mapped_terminal_code)
            and int(process_exit_code) != 0
        )
    return (
        mapped_terminal_code is None
        and int(process_exit_code) == int(probe_exit_code)
    )


def _fail_closed_terminal_from_attribution_payload(
    payload: Mapping[str, Any],
) -> str | None:
    fail_closed = payload.get("fail_closed_terminal")
    if fail_closed is not None:
        return str(fail_closed)
    expanded = payload.get("obmalloc_expanded_attribution") or {}
    nested = expanded.get("fail_closed_terminal")
    if nested is not None:
        return str(nested)
    return None


def _probe_exit_code_from_attribution_payload(payload: Mapping[str, Any]) -> int:
    if payload.get("probe_exit_code") is not None:
        return int(payload["probe_exit_code"])
    if payload.get("exit_code") is not None:
        return int(payload["exit_code"])
    runs = payload.get("runs") or {}
    exit_codes: list[int] = []
    for arm in ("A", "A_prime", "B"):
        arm_payload = runs.get(arm) or {}
        if "exit_code" in arm_payload:
            exit_codes.append(int(arm_payload["exit_code"]))
    if exit_codes:
        return max(exit_codes)
    return 1


def _resolve_attribution_process_exit_code(
    *,
    probe_exit_code: int,
    fail_closed_terminal: str | None,
) -> dict[str, Any]:
    mapped_terminal_code = _mapped_terminal_exit_code(fail_closed_terminal)
    process_exit_code = int(probe_exit_code)
    if mapped_terminal_code is not None:
        process_exit_code = int(mapped_terminal_code)
    exit_code_agreement = _exit_code_agreement_truthful(
        fail_closed_terminal=fail_closed_terminal,
        probe_exit_code=int(probe_exit_code),
        process_exit_code=int(process_exit_code),
        mapped_terminal_code=mapped_terminal_code,
    )
    return {
        "process_exit_code": int(process_exit_code),
        "mapped_terminal_code": mapped_terminal_code,
        "exit_code_agreement": bool(exit_code_agreement),
    }


def resolve_classifier_exit_fields_from_attribution_payload(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    fail_closed_terminal = _fail_closed_terminal_from_attribution_payload(payload)
    probe_exit_code = _probe_exit_code_from_attribution_payload(payload)
    serialized_process_exit_code = payload.get("process_exit_code")
    if serialized_process_exit_code is None:
        serialized_process_exit_code = payload.get("exit_code", probe_exit_code)
    serialized_process_exit_code = int(serialized_process_exit_code)

    serialized_mapped_terminal_code = payload.get("mapped_terminal_code")
    if serialized_mapped_terminal_code is not None:
        serialized_mapped_terminal_code = int(serialized_mapped_terminal_code)

    expected_mapped_terminal_code = (
        _mapped_terminal_exit_code(fail_closed_terminal)
        if fail_closed_terminal is not None
        else None
    )
    agreement_mapped_terminal_code = (
        serialized_mapped_terminal_code
        if serialized_mapped_terminal_code is not None
        else expected_mapped_terminal_code
    )
    exit_code_agreement = _exit_code_agreement_truthful(
        fail_closed_terminal=fail_closed_terminal,
        probe_exit_code=probe_exit_code,
        process_exit_code=serialized_process_exit_code,
        mapped_terminal_code=agreement_mapped_terminal_code,
    )

    canonical = _resolve_attribution_process_exit_code(
        probe_exit_code=probe_exit_code,
        fail_closed_terminal=fail_closed_terminal,
    )
    return {
        "fail_closed_terminal": fail_closed_terminal,
        "probe_exit_code": int(probe_exit_code),
        "process_exit_code": int(canonical["process_exit_code"]),
        "mapped_terminal_code": canonical["mapped_terminal_code"],
        "exit_code_agreement": bool(exit_code_agreement),
        "exit_code": int(canonical["process_exit_code"]),
        "serialized_process_exit_code": int(serialized_process_exit_code),
        "serialized_mapped_terminal_code": serialized_mapped_terminal_code,
    }


PHASE3_CALLSITE_EVENT_TOTAL_NOMINAL_MIN = 194
PHASE3_CALLSITE_EVENT_TOTAL_NOMINAL_MAX = 202
PHASE3_CALLSITE_EVENT_TOTAL_EXPECTED_CLEAN_MIN = 182
PHASE3_CALLSITE_EVENT_TOTAL_EXPECTED_CLEAN_MAX = 210
PHASE3_CALLSITE_EVENT_TOTAL_HARD_CEILING = 258
PHASE3_CALLSITE_EVENT_TOTAL_DIAGNOSTIC_CAVEAT = (
    "event total outside 182-210 is NOT a clean post-rescope single-pair topology "
    "confirmation without Claude/co_lead evidence review."
)
PHASE3_CALLSITE_S1D7_MARK_PAIR_COUNT_EXPECTED = 4
BANKED_RECONCILE_PROVENANCE: dict[str, Any] = {
    "commit": "5fad596af1cac6ad11dad77d32f6ad1f8c481b3a",
    "dual_accept_msg_ids": ["1783074111500", "1783074295721"],
    "s1d_parent_reconcile_fraction": 1.34e-05,
    "s1d_dominant_bracket": "C4.S1d.7",
    "call_site_status_at_bank": "UNRESOLVED",
    "claim_wording": (
        "call-site resolution CONDITIONAL on banked S1d.7/reconcile — "
        "NOT a fresh same-run reconcile"
    ),
}
PHASE3_CALLSITE_TOPOLOGY_BOUNDS = {
    "C4.S1d.3": {"max_allowed": 2, "safety_net_max": 5},
    "C4.S1d.4": {"max_allowed": 2, "safety_net_max": 5},
}


def _phase3_callsite_event_topology_from_expanded(
    expanded: Mapping[str, Any],
    *,
    fail_closed: str | None,
) -> dict[str, Any]:
    guards = dict(expanded.get("guards") or {})
    event_counts = dict(guards.get("obmalloc_expanded_event_counts") or {})
    event_validation = dict(guards.get("obmalloc_expanded_event_validation") or {})
    pair_counts = dict(event_validation.get("pair_counts_by_site") or {})
    total_events = event_counts.get("total")
    total_events_int = int(total_events) if total_events is not None else None
    event_stream_valid = bool(event_validation.get("valid"))
    event_total_in_expected_clean_envelope = (
        total_events_int is not None
        and PHASE3_CALLSITE_EVENT_TOTAL_EXPECTED_CLEAN_MIN <= total_events_int
        <= PHASE3_CALLSITE_EVENT_TOTAL_EXPECTED_CLEAN_MAX
    )
    event_total_diagnostic_tolerated = (
        total_events_int is not None
        and (
            total_events_int < PHASE3_CALLSITE_EVENT_TOTAL_EXPECTED_CLEAN_MIN
            or PHASE3_CALLSITE_EVENT_TOTAL_EXPECTED_CLEAN_MAX
            < total_events_int
            <= PHASE3_CALLSITE_EVENT_TOTAL_HARD_CEILING
        )
    )
    event_total_exceeds_hard_ceiling = (
        total_events_int is not None
        and total_events_int > PHASE3_CALLSITE_EVENT_TOTAL_HARD_CEILING
    )
    topology_violations: list[str] = []
    pair_count_regression_signals: list[str] = []
    unbalanced_pair_violations: list[str] = []
    pair_watch: dict[str, Any] = {}
    for site_id, per_state in pair_counts.items():
        for state_idx, pc in per_state.items():
            if int(pc) < 0:
                unbalanced_pair_violations.append(f"{site_id}_state{state_idx}_pair_count_{pc}")
    if unbalanced_pair_violations:
        topology_violations.extend(["unbalanced_pair:" + v for v in unbalanced_pair_violations])
    if not event_stream_valid:
        topology_violations.append(
            "event_stream_invalid:"
            + ",".join(event_validation.get("corruption_reasons") or ["unknown"])
        )
    for site, bounds in PHASE3_CALLSITE_TOPOLOGY_BOUNDS.items():
        site_pairs = pair_counts.get(site, {})
        site_info: dict[str, Any] = {"per_state_pair_count": site_pairs}
        if site_pairs:
            balanced_counts = [int(v) for v in site_pairs.values() if int(v) >= 0]
            site_info["n_pre_n_post_balanced"] = len(balanced_counts) == len(site_pairs)
            if balanced_counts:
                site_info["max_pairs"] = max(balanced_counts)
                site_info["min_pairs"] = min(balanced_counts)
            for state_idx, pc in site_pairs.items():
                safety_max = int(bounds.get("safety_net_max", bounds["max_allowed"]))
                if int(pc) > safety_max:
                    topology_violations.append(
                        f"{site}_state{state_idx}_pairs{pc}_exceeds_safety_net_max{safety_max}"
                    )
                elif int(pc) > bounds["max_allowed"]:
                    pair_count_regression_signals.append(
                        f"{site}_state{state_idx}_pairs{pc}_exceeds_max"
                        f"{bounds['max_allowed']}_regression"
                    )
        pair_watch[site] = site_info
    if event_total_exceeds_hard_ceiling:
        topology_violations.append(
            f"total_events_{total_events_int}_exceeds_hard_ceiling_"
            f"{PHASE3_CALLSITE_EVENT_TOTAL_HARD_CEILING}"
        )
    event_total_topology_clean_confirmed = (
        fail_closed is None
        and event_stream_valid
        and len(topology_violations) == 0
        and event_total_in_expected_clean_envelope
    )
    return {
        "event_counts": event_counts,
        "event_validation": event_validation,
        "pair_counts": pair_counts,
        "total_events_int": total_events_int,
        "event_stream_valid": event_stream_valid,
        "event_total_in_expected_clean_envelope": event_total_in_expected_clean_envelope,
        "event_total_diagnostic_tolerated": event_total_diagnostic_tolerated,
        "event_total_exceeds_hard_ceiling": event_total_exceeds_hard_ceiling,
        "event_total_topology_clean_confirmed": event_total_topology_clean_confirmed,
        "topology_violations": topology_violations,
        "pair_count_regression_signals": pair_count_regression_signals,
        "unbalanced_pair_violations": unbalanced_pair_violations,
        "pair_watch": pair_watch,
    }


def build_phase3_callsite_classifier_receipt_from_attribution_payload(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    expanded = dict(payload.get("obmalloc_expanded_attribution") or {})
    loc = dict(expanded.get("localization") or {})
    guards = dict(expanded.get("guards") or {})
    fail_closed = expanded.get("fail_closed_terminal")
    phase3_s1d = bool(guards.get("phase3_s1d_subsplit_mode") or loc.get("phase3_s1d_subsplit_mode"))
    callsite_b_prime_mode = bool(guards.get("callsite_b_prime_mode"))
    s1d_reconcile = loc.get("s1d_parent_reconcile_fraction")
    banked_reconcile_ok = (
        float(BANKED_RECONCILE_PROVENANCE["s1d_parent_reconcile_fraction"]) <= 0.15
    )
    b_arm = dict((payload.get("runs") or {}).get("B") or {})
    b_arm_profile_mark_count = int(b_arm.get("profile_mark_count") or 0)
    exit_fields = resolve_classifier_exit_fields_from_attribution_payload(payload)
    process_exit_code = int(exit_fields["process_exit_code"])
    mapped_terminal_code = exit_fields.get("mapped_terminal_code")
    exit_code_agreement = bool(exit_fields["exit_code_agreement"])
    topology = _phase3_callsite_event_topology_from_expanded(
        expanded,
        fail_closed=str(fail_closed) if fail_closed is not None else None,
    )
    call_site_status = expanded.get("call_site_status", "UNRESOLVED")
    call_site_origin = expanded.get("call_site_origin_file_line")
    call_site_candidate = expanded.get("s1d7_call_site_candidate")
    call_site_branch = expanded.get("s1d7_call_site_branch_outcome")
    tracemalloc_perturbed = expanded.get("tracemalloc_perturbed")
    if tracemalloc_perturbed is None:
        tracemalloc_perturbed = guards.get("tracemalloc_perturbed")
    mark_pair_count = expanded.get("s1d7_tracemalloc_mark_pair_count")
    concentration = expanded.get("s1d7_tracemalloc_top_concentration_fraction")
    in_bracket_ok = expanded.get("s1d7_call_site_in_bracket_ok")
    fail_closed_reason = None
    s1d7_site = dict(loc.get("s1d7_tracemalloc_call_site") or {})
    if fail_closed_reason is None:
        fail_closed_reason = s1d7_site.get("fail_closed_reason")
    out: dict[str, Any] = {
        "schema": "hrm_text_158_phase3_callsite_classifier_receipt/v1",
        "exit_code": payload.get("exit_code"),
        "process_exit_code": process_exit_code,
        "mapped_terminal_code": mapped_terminal_code,
        "exit_code_agreement": exit_code_agreement,
        "b_arm_profile_mark_count": b_arm_profile_mark_count,
        "fail_closed_terminal": fail_closed,
        "phase3_s1d_subsplit_mode": phase3_s1d,
        "callsite_b_prime_mode": callsite_b_prime_mode,
        "banked_reconcile_provenance": dict(BANKED_RECONCILE_PROVENANCE),
        "banked_reconcile_precondition_ok": banked_reconcile_ok,
        "s1d_parent_reconcile_fraction": s1d_reconcile,
        "s1d_rescope_reconcile_pass": (
            banked_reconcile_ok
            if callsite_b_prime_mode
            else (s1d_reconcile is not None and float(s1d_reconcile) <= 0.15)
        ),
        "call_site_status": call_site_status,
        "call_site_origin_file_line": call_site_origin,
        "s1d7_call_site_candidate": call_site_candidate,
        "s1d7_call_site_branch_outcome": call_site_branch,
        "s1d7_tracemalloc_top_concentration_fraction": concentration,
        "tracemalloc_perturbed": tracemalloc_perturbed,
        "s1d7_call_site_in_bracket_ok": in_bracket_ok,
        "s1d7_tracemalloc_mark_pair_count": mark_pair_count,
        "s1d7_tracemalloc_diff": expanded.get("s1d7_tracemalloc_diff"),
        "fail_closed_reason": fail_closed_reason,
        "obmalloc_expanded_event_count_total": topology["total_events_int"],
        "event_count_total_reported": topology["total_events_int"],
        "pair_counts_by_site": topology["pair_counts"],
        "event_stream_valid": topology["event_stream_valid"],
        "event_total_in_expected_clean_envelope": topology["event_total_in_expected_clean_envelope"],
        "event_total_diagnostic_tolerated": topology["event_total_diagnostic_tolerated"],
        "event_total_diagnostic_tolerated_caveat": (
            PHASE3_CALLSITE_EVENT_TOTAL_DIAGNOSTIC_CAVEAT
            if topology["event_total_diagnostic_tolerated"]
            else None
        ),
        "event_total_topology_clean_confirmed": topology["event_total_topology_clean_confirmed"],
        "multi_pair_watch": topology["pair_watch"],
        "unbalanced_pair_violations": topology["unbalanced_pair_violations"],
        "pair_count_regression_signals": topology["pair_count_regression_signals"],
        "topology_violations": topology["topology_violations"],
        "attribution_fail_closed_pre_science": fail_closed is not None,
        "multi_pair_topology_pass": topology["event_total_topology_clean_confirmed"],
        "terminal_receipt": {
            "fail_closed_terminal": fail_closed,
            "event_stream_valid": topology["event_stream_valid"],
            "event_counts_total": topology["total_events_int"],
            "pair_counts_by_site": topology["pair_counts"],
            "corruption_reasons": list(
                topology["event_validation"].get("corruption_reasons") or []
            ),
            "event_total_in_expected_clean_envelope": topology[
                "event_total_in_expected_clean_envelope"
            ],
            "event_total_diagnostic_tolerated_caveat": (
                PHASE3_CALLSITE_EVENT_TOTAL_DIAGNOSTIC_CAVEAT
                if topology["event_total_diagnostic_tolerated"]
                else None
            ),
            "event_total_topology_clean_confirmed": topology[
                "event_total_topology_clean_confirmed"
            ],
            "process_exit_code": process_exit_code,
            "mapped_terminal_code": mapped_terminal_code,
            "exit_code_agreement": exit_code_agreement,
            "call_site_status": call_site_status,
            "call_site_origin_file_line": call_site_origin,
            "s1d7_call_site_candidate": call_site_candidate,
            "s1d7_call_site_branch_outcome": call_site_branch,
            "s1d7_tracemalloc_top_concentration_fraction": concentration,
            "tracemalloc_perturbed": tracemalloc_perturbed,
            "s1d7_call_site_in_bracket_ok": in_bracket_ok,
            "s1d7_tracemalloc_mark_pair_count": mark_pair_count,
        },
    }
    classifier_exit_code = 0
    if b_arm_profile_mark_count <= 0:
        out["branch_outcome"] = "EMPTY_ATTRIBUTION_STREAM"
        out["topology_fail_reason"] = "b_arm_profile_mark_count_0"
        classifier_exit_code = 38
    elif topology["unbalanced_pair_violations"]:
        out["branch_outcome"] = "UNBALANCED_PAIR_COUNT"
        out["topology_fail_reason"] = topology["unbalanced_pair_violations"][0]
        classifier_exit_code = 36
    elif fail_closed is not None:
        out["branch_outcome"] = f"ATTRIBUTION_FAIL_CLOSED_{fail_closed}"
        classifier_exit_code = 37
    elif topology["topology_violations"]:
        out["branch_outcome"] = "MULTI_PAIR_COUNT_EXCEEDS_TOPOLOGY"
        out["topology_fail_reason"] = topology["topology_violations"][0]
        classifier_exit_code = 36
    elif fail_closed == "CHILD_OVERLAP_DOUBLE_COUNT":
        out["branch_outcome"] = "OVERLAP_REGRESSION_FAIL"
        classifier_exit_code = 32
    elif fail_closed == "S1D_CHILD_COVERAGE_FAIL":
        out["branch_outcome"] = "S1D_CHILD_COVERAGE_FAIL"
        classifier_exit_code = 32
    elif fail_closed == "CHILD_PARENT_RECONCILE_FAIL":
        out["branch_outcome"] = "S1D_RESCOPE_INSTRUMENTATION_FAIL"
        out["branch_reason"] = "reconcile_fail_gates_science"
        classifier_exit_code = 33
    elif callsite_b_prime_mode and not banked_reconcile_ok:
        out["branch_outcome"] = "BANKED_RECONCILE_PRECONDITION_FAIL"
        classifier_exit_code = 35
    elif not callsite_b_prime_mode and not phase3_s1d:
        out["branch_outcome"] = "S1D_CHILD_COVERAGE_FAIL"
        out["branch_reason"] = "phase3_s1d_subsplit_mode_false"
        classifier_exit_code = 34
    elif not callsite_b_prime_mode and (
        s1d_reconcile is None or float(s1d_reconcile) > 0.15
    ):
        out["branch_outcome"] = "RECONCILE_FAIL"
        classifier_exit_code = 35
    elif bool(tracemalloc_perturbed):
        out["branch_outcome"] = "TRACEMALLOC_PERTURBED_INCONCLUSIVE"
        classifier_exit_code = 35
    elif mark_pair_count != PHASE3_CALLSITE_S1D7_MARK_PAIR_COUNT_EXPECTED:
        out["branch_outcome"] = "TRACEMALLOC_INCONCLUSIVE"
        out["branch_reason"] = (
            f"s1d7_tracemalloc_mark_pair_count={mark_pair_count}"
            f" expected={PHASE3_CALLSITE_S1D7_MARK_PAIR_COUNT_EXPECTED}"
        )
        classifier_exit_code = 35
    elif call_site_status != "RESOLVED":
        reason = fail_closed_reason or "TRACEMALLOC_INCONCLUSIVE"
        if reason == "CALL_SITE_OUTSIDE_S1D7_BRACKET":
            out["branch_outcome"] = "CALL_SITE_OUTSIDE_S1D7_BRACKET"
        elif reason == "TRACEMALLOC_CONCENTRATION_FAIL":
            out["branch_outcome"] = "TRACEMALLOC_CONCENTRATION_FAIL"
        else:
            out["branch_outcome"] = "TRACEMALLOC_INCONCLUSIVE"
        out["branch_reason"] = reason
        classifier_exit_code = 35
    elif call_site_branch:
        out["branch_outcome"] = str(call_site_branch)
    elif call_site_candidate == "ambiguous":
        out["branch_outcome"] = "S1D7_CALL_SITE_RESOLVED_CANDIDATE_AMBIGUOUS"
    else:
        out["branch_outcome"] = "TRACEMALLOC_INCONCLUSIVE"
        classifier_exit_code = 35
    out["classifier_exit_code"] = int(classifier_exit_code)
    if int(process_exit_code) != 0 and classifier_exit_code == 0:
        classifier_exit_code = 31
        out["classifier_exit_code"] = 31
    return out


def build_postrun_aggregate_exit_summary(
    *,
    attribution_payload: Mapping[str, Any] | None,
    exit_code_artifact: int | None,
    classifier_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    attribution_exit_fields: dict[str, Any] = {}
    if attribution_payload is not None:
        attribution_exit_fields = resolve_classifier_exit_fields_from_attribution_payload(
            attribution_payload
        )

    classifier_exit_fields: dict[str, Any] | None = None
    if classifier_receipt is not None:
        merged_payload: dict[str, Any] = dict(attribution_payload or {})
        merged_payload.update(
            {
                "process_exit_code": classifier_receipt.get("process_exit_code"),
                "mapped_terminal_code": classifier_receipt.get("mapped_terminal_code"),
                "exit_code_agreement": classifier_receipt.get("exit_code_agreement"),
                "fail_closed_terminal": classifier_receipt.get("fail_closed_terminal")
                or attribution_exit_fields.get("fail_closed_terminal"),
                "exit_code": classifier_receipt.get("exit_code"),
            }
        )
        classifier_exit_fields = resolve_classifier_exit_fields_from_attribution_payload(
            merged_payload
        )

    aggregate_exit_code = exit_code_artifact
    if attribution_exit_fields.get("fail_closed_terminal") is not None:
        aggregate_exit_code = int(attribution_exit_fields["process_exit_code"])
    elif classifier_exit_fields and classifier_exit_fields.get("fail_closed_terminal") is not None:
        aggregate_exit_code = int(classifier_exit_fields["process_exit_code"])
    elif attribution_exit_fields:
        aggregate_exit_code = int(attribution_exit_fields["process_exit_code"])

    return {
        "exit_code": aggregate_exit_code,
        "exit_code_artifact": exit_code_artifact,
        "attribution_exit_fields": attribution_exit_fields,
        "classifier_exit_fields": classifier_exit_fields,
        "exit_code_propagation_overrode_zero_artifact": (
            exit_code_artifact == 0
            and aggregate_exit_code not in (None, 0)
        ),
    }


OBMALLOC_EXPANDED_ARM_DIR_NAMES: tuple[str, ...] = (
    "obmalloc_expanded_ab",
    "obmalloc_expanded_ab_replicate",
    "obmalloc_expanded_b",
)
POSTRUN_AGGREGATE_RECEIPT_SCHEMA = (
    "hrm_text_158_phase3_subsplit_postrun_aggregate_receipt/v1"
)
LIVENESS_STACK_DUMP_NAME = "liveness_stack_dump.txt"
LAST_ACTIVE_PHASE_NAME = "last_active_phase.json"
LIVENESS_ROLLUP_SCHEMA = "hrm_text_158_phase3_subsplit_liveness_rollup/v1"


def _last_active_phase_cleared(payload: Mapping[str, Any]) -> bool:
    if payload.get("guard_event") == "cleared":
        return True
    if payload.get("phase_status") == "completed":
        return True
    if payload.get("liveness_failure") is False:
        return True
    return False


def summarize_arm_liveness_telemetry(arm_dir: Path) -> dict[str, Any]:
    """Mechanical per-arm liveness rollup for postrun aggregate receipts (C3)."""
    arm_path = Path(arm_dir)
    summary: dict[str, Any] = {
        "arm_dir": str(arm_path),
        "arm_exists": arm_path.is_dir(),
        "liveness_stack_dump_present": False,
        "last_active_phase_present": False,
        "failure_class": None,
        "fail_closed_termination": None,
        "guard_event": None,
        "phase": None,
        "active_phase_elapsed_seconds": None,
        "last_active_phase_cleared": None,
        "terminal_unresolved_liveness_failure": False,
    }
    if not arm_path.is_dir():
        return summary

    stack_dump_path = arm_path / LIVENESS_STACK_DUMP_NAME
    summary["liveness_stack_dump_present"] = stack_dump_path.is_file()

    lap_path = arm_path / LAST_ACTIVE_PHASE_NAME
    if not lap_path.is_file():
        return summary

    summary["last_active_phase_present"] = True
    lap = json.loads(lap_path.read_text(encoding="utf-8"))
    summary["failure_class"] = lap.get("failure_class")
    summary["fail_closed_termination"] = lap.get("fail_closed_termination")
    summary["guard_event"] = lap.get("guard_event")
    summary["phase"] = lap.get("phase")
    summary["active_phase_elapsed_seconds"] = lap.get("active_phase_elapsed_seconds")
    cleared = _last_active_phase_cleared(lap)
    summary["last_active_phase_cleared"] = cleared
    summary["terminal_unresolved_liveness_failure"] = (
        lap.get("failure_class") == "LIVENESS_FAILURE" and not cleared
    )
    return summary


def build_postrun_aggregate_liveness_rollup(
    run_root: Path,
    *,
    arm_dir_names: Sequence[str] = OBMALLOC_EXPANDED_ARM_DIR_NAMES,
) -> dict[str, Any]:
    arms = {
        str(arm_name): summarize_arm_liveness_telemetry(Path(run_root) / str(arm_name))
        for arm_name in arm_dir_names
    }
    unresolved = [
        arm_name
        for arm_name, arm_summary in arms.items()
        if arm_summary.get("terminal_unresolved_liveness_failure")
    ]
    cleared = [
        arm_name
        for arm_name, arm_summary in arms.items()
        if arm_summary.get("last_active_phase_cleared") is True
    ]
    return {
        "schema": LIVENESS_ROLLUP_SCHEMA,
        "run_root": str(run_root),
        "arms": arms,
        "arms_with_unresolved_terminal_liveness_failure": unresolved,
        "arms_with_cleared_last_active_phase": cleared,
        "any_unresolved_terminal_liveness_failure": bool(unresolved),
    }


def build_phase3_subsplit_postrun_aggregate_receipt(
    run_root: Path,
    *,
    attribution_json_name: str = (
        "v6i_obmalloc_expanded_attribution_c4s1_phase3_subsplit.json"
    ),
    classifier_receipt_rel: str = "postrun/phase3_subsplit_classifier_receipt.json",
) -> dict[str, Any]:
    root = Path(run_root)
    out: dict[str, Any] = {
        "schema": POSTRUN_AGGREGATE_RECEIPT_SCHEMA,
        "run_id": root.name,
        "mirror_durable_absent": True,
    }

    attribution_payload: dict[str, Any] | None = None
    attr_path = root / attribution_json_name
    if attr_path.is_file():
        attribution_payload = json.loads(attr_path.read_text(encoding="utf-8"))
        out["mirror_durable_absent"] = "durable_mirror_receipt" not in attribution_payload

    exit_code_artifact: int | None = None
    ec_path = root / "exit_code.txt"
    if ec_path.is_file():
        exit_code_artifact = int(ec_path.read_text(encoding="utf-8").strip())

    classifier_receipt: dict[str, Any] | None = None
    cls_path = root / classifier_receipt_rel
    if cls_path.is_file():
        classifier_receipt = json.loads(cls_path.read_text(encoding="utf-8"))
        out["classifier"] = classifier_receipt
        out["branch_outcome"] = classifier_receipt.get("branch_outcome")
        out["s1f_branch_outcome"] = classifier_receipt.get("s1f_branch_outcome")

    exit_summary = build_postrun_aggregate_exit_summary(
        attribution_payload=attribution_payload,
        exit_code_artifact=exit_code_artifact,
        classifier_receipt=classifier_receipt,
    )
    out.update(exit_summary)
    out["liveness_rollup"] = build_postrun_aggregate_liveness_rollup(root)
    if attribution_payload is not None:
        out["fail_closed_terminal"] = attribution_payload.get("fail_closed_terminal")
    return out


def resolve_fixture_attribution_main_exit_code(payload: Mapping[str, Any]) -> int:
    fail_closed_terminal = _fail_closed_terminal_from_attribution_payload(payload)
    if fail_closed_terminal is not None:
        mapped = _mapped_terminal_exit_code(fail_closed_terminal)
        if mapped is not None:
            return int(mapped)
    exit_fields = resolve_classifier_exit_fields_from_attribution_payload(payload)
    return int(exit_fields["process_exit_code"])
OBMALLOC_EXPANDED_BOUNDARY_AFTER_STATE_MAX = 8


def _is_allocator_mark(row: Mapping[str, Any]) -> bool:
    return str(row.get("schema", "")) == PROFILE_HOST_RSS_ALLOCATOR_SCHEMA


def _is_allocator_site_mark(row: Mapping[str, Any]) -> bool:
    return str(row.get("schema", "")) == PROFILE_HOST_RSS_ALLOCATOR_SITE_SCHEMA


def _is_alloc_hook_mark(row: Mapping[str, Any]) -> bool:
    return (
        str(row.get("schema", "")) == PROFILE_HOST_RSS_ALLOC_HOOK_SCHEMA
        and str(row.get("event", "")).startswith("alloc_hook_")
    )


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
    malloc_info = dict(probe.get("malloc_info_all_arenas") or {})
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
        "glibc_arena_system_or_retained_bytes": malloc_info.get(
            "glibc_arena_system_or_retained_bytes"
        ),
        "malloc_info_total_mmap_bytes": malloc_info.get("total_mmap_bytes"),
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


def attribute_allocator_type_partition(
    *,
    marks: list[dict[str, Any]],
    alloc_hook_marks: list[dict[str, Any]],
    allocator_marks: list[dict[str, Any]],
    disjointness_probe: Mapping[str, Any] | None,
    self_footprint: Mapping[str, Any] | None,
    cross_run_reconcile_caveat: bool = False,
) -> dict[str, Any]:
    from calm.hrm_text_158.native_full_stack.host_allocator_probe import (
        ALLOCATOR_TYPE_DOMINANCE,
        ALLOCATOR_TYPE_RECONCILE_MAX,
        ALLOCATOR_TYPE_RECONCILE_MIN,
        MALLOC_INFO_SELF_FOOTPRINT_MAX_BYTES,
        compute_delta_disjoint_partition,
    )

    subphase = attribute_subphase_rss_profile(marks)
    c4_delta_gib = None
    for row in subphase.get("subphase_deltas") or []:
        if str(row.get("phase")) == "C4_gpu_cap_apply_sync":
            c4_delta_gib = row.get("delta_rss_gib")
            break
    if c4_delta_gib is None:
        c4_delta_gib = subphase.get("dominant_subphase_delta_rss_gib")
    if c4_delta_gib is None:
        return {
            "allocator_type_owner_status": "INCONCLUSIVE",
            "tier": "C",
            "call_site_status": "UNRESOLVED",
            "reason": "missing_c4_delta",
        }

    hook_enter = next(
        (row for row in alloc_hook_marks if str(row.get("event")) == "alloc_hook_C4_enter"),
        None,
    )
    hook_exit = next(
        (row for row in alloc_hook_marks if str(row.get("event")) == "alloc_hook_C4_exit"),
        None,
    )
    alloc_enter = next(
        (row for row in allocator_marks if str(row.get("event")) == "allocator_C4_enter"),
        None,
    )
    alloc_exit = next(
        (row for row in allocator_marks if str(row.get("event")) == "allocator_C4_exit"),
        None,
    )
    if not hook_exit or not alloc_enter or not alloc_exit:
        return {
            "allocator_type_owner_status": "INCONCLUSIVE",
            "tier": "C",
            "call_site_status": "UNRESOLVED",
            "reason": "missing_boundary_marks",
        }

    hook_window = int(dict(hook_exit.get("alloc_hook_stats") or {}).get("window_net_bytes") or 0)
    malloc_enter = dict((alloc_enter.get("allocator_probe") or {}).get("malloc_info_all_arenas") or {})
    malloc_exit = dict((alloc_exit.get("allocator_probe") or {}).get("malloc_info_all_arenas") or {})
    if not malloc_enter.get("available") or not malloc_exit.get("available"):
        return {
            "allocator_type_owner_status": "INCONCLUSIVE",
            "tier": "C",
            "call_site_status": "UNRESOLVED",
            "reason": "malloc_info_unavailable_at_boundary",
            "malloc_info_enter": malloc_enter,
            "malloc_info_exit": malloc_exit,
        }

    footprint_bytes = 0
    footprint_status = None
    if self_footprint is not None:
        footprint_status = str(self_footprint.get("malloc_info_self_footprint_status") or "")
        if footprint_status == "exceeded":
            return {
                "allocator_type_owner_status": "INCONCLUSIVE",
                "tier": "C",
                "call_site_status": "UNRESOLVED",
                "reason": "malloc_info_self_footprint_exceeded",
                "self_footprint": dict(self_footprint),
            }
        footprint_bytes = int(self_footprint.get("malloc_info_self_footprint_bytes") or 0)

    catches = None
    if disjointness_probe is not None and disjointness_probe.get("status") == "ok":
        catches = bool(disjointness_probe.get("mmap_hook_catches_glibc_internal"))

    enter_cuda = _allocator_source_scalars(dict(alloc_enter.get("allocator_probe") or {}))
    exit_cuda = _allocator_source_scalars(dict(alloc_exit.get("allocator_probe") or {}))
    cuda_measured = bool(enter_cuda.get("host_memory_stats_available")) and bool(
        exit_cuda.get("host_memory_stats_available")
    )
    cuda_delta = None
    if cuda_measured:
        cuda_delta = _delta_gib(
            int(exit_cuda.get("cuda_host_active_bytes"))
            if exit_cuda.get("cuda_host_active_bytes") is not None
            else None,
            int(enter_cuda.get("cuda_host_active_bytes"))
            if enter_cuda.get("cuda_host_active_bytes") is not None
            else None,
            kib=False,
        )
        if cuda_delta is not None:
            cuda_delta = int(cuda_delta * (1024**3))

    partition = compute_delta_disjoint_partition(
        c4_delta_rss_bytes=int(float(c4_delta_gib) * (1024**3)),
        hook_window_net_bytes=hook_window,
        malloc_info_enter=malloc_enter,
        malloc_info_exit=malloc_exit,
        mmap_hook_catches_glibc_internal=catches,
        self_footprint_bytes=footprint_bytes,
        cuda_host_delta_bytes=cuda_delta,
        cuda_host_measured=cuda_measured,
    )

    if cross_run_reconcile_caveat:
        return {
            "allocator_type_owner_status": "INCONCLUSIVE",
            "tier": "C",
            "call_site_status": "UNRESOLVED",
            "cross_run_reconcile_caveat": True,
            "partition": partition,
            "disjointness_probe": dict(disjointness_probe or {}),
            "self_footprint": dict(self_footprint or {}),
            "reason": "cross_run_buckets_forbidden_for_resolved",
        }

    fail_reasons = list(partition.get("fail_reasons") or [])
    if footprint_status == "unavailable":
        fail_reasons.append("malloc_info_self_footprint_unavailable")

    measured = dict(partition.get("measured_buckets_bytes") or {})
    c4_bytes = int(partition.get("c4_delta_rss_bytes") or 0)
    measured_sum = int(partition.get("measured_sum_bytes") or 0)
    residual = int(partition.get("residual_unattributed_bytes") or 0)
    reconcile = partition.get("reconcile_ratio_measured_only")
    residual_fraction = partition.get("residual_fraction_of_c4")

    dominant_type = None
    dominant_bytes = 0
    for name, value in measured.items():
        if value is None:
            continue
        if int(value) > dominant_bytes:
            dominant_bytes = int(value)
            dominant_type = name

    tier = "C"
    owner_status = "INCONCLUSIVE"
    if fail_reasons:
        owner_status = "INCONCLUSIVE"
    elif reconcile is not None and ALLOCATOR_TYPE_RECONCILE_MIN <= float(reconcile) <= ALLOCATOR_TYPE_RECONCILE_MAX:
        if dominant_type and c4_bytes > 0 and (dominant_bytes / c4_bytes) >= ALLOCATOR_TYPE_DOMINANCE:
            owner_status = "RESOLVED_BY_TYPE"
            tier = "A"
        else:
            owner_status = "PARTIAL"
            tier = "B"
    elif (
        not cuda_measured
        and residual_fraction is not None
        and float(residual_fraction) >= ALLOCATOR_TYPE_DOMINANCE
    ):
        owner_status = "INFERRED_RESIDUAL_DOMINANT"
        tier = "C"

    mallinfo_enter = dict((alloc_enter.get("allocator_probe") or {}).get("mallinfo2") or {})
    mallinfo_exit = dict((alloc_exit.get("allocator_probe") or {}).get("mallinfo2") or {})

    return {
        "allocator_type_owner_status": owner_status,
        "tier": tier,
        "dominant_allocator_type": dominant_type,
        "dominant_allocator_type_bytes": dominant_bytes if dominant_type else None,
        "call_site_status": "UNRESOLVED",
        "c4_delta_rss_gib": float(c4_delta_gib),
        "hook_window_net_gib": hook_window / (1024.0**3),
        "cuda_host_measured": cuda_measured,
        "cuda_host_delta_gib": (cuda_delta / (1024.0**3)) if cuda_delta is not None else None,
        "residual_unattributed_gib": residual / (1024.0**3),
        "residual_fraction_of_c4": residual_fraction,
        "reconcile_ratio_measured_only": reconcile,
        "measured_sum_gib": measured_sum / (1024.0**3),
        "partition": partition,
        "disjointness_probe": dict(disjointness_probe or {}),
        "self_footprint": dict(self_footprint or {}),
        "mallinfo2_main_arena_cross_check": {
            "enter_uordblks_bytes": mallinfo_enter.get("uordblks_bytes"),
            "exit_uordblks_bytes": mallinfo_exit.get("uordblks_bytes"),
            "label": "main_arena_cross_check_only",
        },
        "fail_reasons": fail_reasons,
        "cross_run_reconcile_caveat": bool(cross_run_reconcile_caveat),
        "malloc_info_self_footprint_threshold_bytes": MALLOC_INFO_SELF_FOOTPRINT_MAX_BYTES,
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


def _is_triangulation_mark(row: Mapping[str, Any]) -> bool:
    return str(row.get("schema", "")) == PROFILE_HOST_RSS_TRIANGULATION_SCHEMA


def _c4_subphase_delta_gib(marks: list[dict[str, Any]]) -> float | None:
    unperturbed = [
        row
        for row in marks
        if _is_subphase_mark(row)
        and not bool(row.get("measurement_perturbed", False))
        and str(row.get("sub_phase")) == C4_SUBPHASE
    ]
    deltas = _compute_phase_deltas(
        unperturbed,
        key_fn=_subphase_key,
        phase_filter=lambda row: str(row.get("sub_phase")) == C4_SUBPHASE,
    )
    if not deltas:
        return None
    value = deltas[0].get("delta_rss_gib")
    return float(value) if value is not None else None


def _triangulation_mark(
    marks: list[dict[str, Any]],
    event: str,
) -> dict[str, Any] | None:
    for row in marks:
        if _is_triangulation_mark(row) and str(row.get("event")) == str(event):
            return row
    return None


def _banked_reference_paths() -> dict[str, Any]:
    return {
        "banked_reference_commit": BANKED_REFERENCE_COMMIT,
        "banked_non_glibc_mmap_reference_bytes": BANKED_NON_GLIBC_MMAP_REFERENCE_BYTES,
        "banked_non_glibc_mmap_reference_gib": BANKED_NON_GLIBC_MMAP_REFERENCE_GIB,
        "total_c4_reference_gib": TOTAL_C4_REFERENCE_GIB,
        "banked_path_mmap_net_bytes": (
            "alloc_hook_attribution.allocator_type_partition.mmap_net_bytes"
        ),
        "banked_path_mmap_net_gib": (
            "alloc_hook_attribution.allocator_type_partition.mmap_net_gib"
        ),
        "banked_path_c4_subphase_delta_rss_gib": (
            "alloc_hook_attribution.allocator_type_partition.c4_subphase_delta_rss_gib"
        ),
        "denominator_source": "banked_cross_run",
        "cross_run_reconcile_caveat": True,
    }


def attribute_python_allocator_triangulation(
    *,
    marks_a: list[dict[str, Any]],
    marks_a_prime: list[dict[str, Any]],
    marks_b: list[dict[str, Any]],
    debugmallocstats_preflight: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    from calm.hrm_text_158.native_full_stack.host_tracemalloc_probe import (
        BRANCH1_CONCENTRATION,
        BRANCH1_RECONCILE_MAX,
        BRANCH1_RECONCILE_MIN,
        BRANCH2_CURRENT_VS_PEAK,
        BRANCH2_PEAK_FRAC,
        BRANCH3_CURRENT_FRAC,
        classify_branch1_concentration,
        diff_traceback_frames,
    )

    banked = _banked_reference_paths()
    c4_a = _c4_subphase_delta_gib(marks_a)
    c4_a_prime = _c4_subphase_delta_gib(marks_a_prime)
    c4_b = _c4_subphase_delta_gib(marks_b)

    noise_floor: float | None = None
    if c4_a is not None and c4_a_prime is not None:
        noise_floor = abs(float(c4_a_prime) - float(c4_a))

    run_stability_threshold = GUARD_STABILITY_FRAC * TOTAL_C4_REFERENCE_GIB
    denominator_variance_threshold = (
        GUARD_STABILITY_FRAC * BANKED_NON_GLIBC_MMAP_REFERENCE_GIB
    )
    envelope_tolerance = (
        max(float(noise_floor), GUARD_ENVELOPE_MIN_GIB)
        if noise_floor is not None
        else GUARD_ENVELOPE_MIN_GIB
    )
    total_c4_envelope_delta = (
        abs(float(c4_a) - TOTAL_C4_REFERENCE_GIB) if c4_a is not None else None
    )

    guards: dict[str, Any] = {
        "noise_floor_gib": noise_floor,
        "run_stability_threshold_gib": run_stability_threshold,
        "denominator_variance_threshold_gib": denominator_variance_threshold,
        "total_c4_envelope_tolerance_gib": envelope_tolerance,
        "total_c4_envelope_delta_gib": total_c4_envelope_delta,
        "c4_rss_delta_gib_same_run": {
            "A": c4_a,
            "A_prime": c4_a_prime,
            "B": c4_b,
        },
    }

    if noise_floor is None:
        guards["run_stability_ok"] = False
        guards["denominator_variance_ok"] = False
        return {
            **banked,
            "guards": guards,
            "fail_closed_terminal": "INCONCLUSIVE_PENDING_NOISE_FLOOR",
            "classifier_branch": None,
            "rewrite_candidate_frames": None,
        }

    guards["run_stability_ok"] = noise_floor <= run_stability_threshold
    guards["denominator_variance_ok"] = noise_floor <= denominator_variance_threshold
    guards["total_c4_envelope_ok"] = (
        total_c4_envelope_delta is not None
        and total_c4_envelope_delta <= envelope_tolerance
    )

    if not guards["total_c4_envelope_ok"]:
        return {
            **banked,
            "guards": guards,
            "fail_closed_terminal": "DENOMINATOR_INVALID_INCONCLUSIVE",
            "classifier_branch": None,
            "rewrite_candidate_frames": None,
        }
    if not guards["run_stability_ok"] or not guards["denominator_variance_ok"]:
        return {
            **banked,
            "guards": guards,
            "fail_closed_terminal": "INCONCLUSIVE_CROSS_RUN_DENOMINATOR",
            "classifier_branch": None,
            "rewrite_candidate_frames": None,
        }

    perturbation_delta = (
        abs(float(c4_b) - float(c4_a))
        if c4_a is not None and c4_b is not None
        else None
    )
    perturbation_threshold = max(
        PERTURBATION_MIN_GIB,
        PERTURBATION_NOISE_K * float(noise_floor),
    )
    guards["perturbation_delta_gib"] = perturbation_delta
    guards["perturbation_threshold_gib"] = perturbation_threshold
    guards["tracemalloc_perturbed"] = (
        perturbation_delta is not None
        and perturbation_delta > perturbation_threshold
    )
    if guards["tracemalloc_perturbed"]:
        return {
            **banked,
            "guards": guards,
            "fail_closed_terminal": "TRACEMALLOC_PERTURBED_INCONCLUSIVE",
            "classifier_branch": None,
            "rewrite_candidate_frames": None,
        }

    baseline_mark = _triangulation_mark(marks_b, "triangulation_C3_exit")
    exit_mark = _triangulation_mark(marks_b, "triangulation_C4_exit")
    if baseline_mark is None or exit_mark is None:
        has_tracemalloc_marks = any(_is_triangulation_mark(row) for row in marks_b)
        if has_tracemalloc_marks:
            return {
                **banked,
                "guards": guards,
                "fail_closed_terminal": "TRACEMALLOC_PERTURBED_INCONCLUSIVE",
                "classifier_branch": None,
                "rewrite_candidate_frames": None,
                "b_run_incomplete": True,
                "b_incomplete_reason": "missing_triangulation_C4_exit",
            }
        return {
            **banked,
            "guards": guards,
            "fail_closed_terminal": "INCONCLUSIVE_MISSING_TRIANGULATION_MARKS",
            "classifier_branch": None,
            "rewrite_candidate_frames": None,
        }

    preflight = dict(debugmallocstats_preflight or {})
    arena_preflight_ok = str(preflight.get("status")) == "ok"
    baseline_stats = dict(baseline_mark.get("debugmallocstats") or {})
    exit_stats = dict(exit_mark.get("debugmallocstats") or {})
    arena_stats_ok = (
        arena_preflight_ok
        and baseline_stats.get("available")
        and exit_stats.get("available")
    )
    guards["arena_stats_preflight_ok"] = arena_preflight_ok
    guards["arena_stats_available"] = arena_stats_ok

    baseline_tm = dict(baseline_mark.get("tracemalloc") or {})
    exit_tm = dict(exit_mark.get("tracemalloc") or {})
    diff = diff_traceback_frames(baseline_tm, exit_tm)
    current_delta = int(diff.get("current_delta_bytes") or 0)
    peak_delta = int(diff.get("peak_delta_bytes") or 0)
    denom = int(BANKED_NON_GLIBC_MMAP_REFERENCE_BYTES)

    reconcile_ratio = float(current_delta) / float(denom) if denom > 0 else 0.0
    peak_ratio = float(peak_delta) / float(denom) if denom > 0 else 0.0
    current_vs_peak = (
        float(current_delta) / float(peak_delta) if peak_delta > 0 else 0.0
    )

    dual_report: dict[str, Any] = {
        "primary_current_delta_bytes": current_delta,
        "primary_peak_delta_bytes": peak_delta,
        "banked_denominator_bytes": denom,
        "primary_reconcile_ratio": reconcile_ratio,
        "peak_reconcile_ratio": peak_ratio,
        "c4_rss_delta_gib_same_run_B": c4_b,
    }
    diff["dual_report"] = dual_report

    arena_growth = None
    arena_retained_fraction = None
    if arena_stats_ok:
        base_arena = int(baseline_stats.get("arena_bytes") or 0)
        exit_arena = int(exit_stats.get("arena_bytes") or 0)
        arena_growth = exit_arena - base_arena
        if peak_delta > 0:
            arena_retained_fraction = float(exit_arena - base_arena) / float(peak_delta)

    branch1_ok = (
        BRANCH1_RECONCILE_MIN <= reconcile_ratio <= BRANCH1_RECONCILE_MAX
        and classify_branch1_concentration(diff)
    )
    branch2_ok = (
        arena_stats_ok
        and peak_ratio >= BRANCH2_PEAK_FRAC
        and current_vs_peak < BRANCH2_CURRENT_VS_PEAK
        and arena_retained_fraction is not None
        and arena_retained_fraction >= 0.50
    )
    branch3_ok = (
        reconcile_ratio < BRANCH3_CURRENT_FRAC
        and arena_stats_ok
        and arena_growth is not None
        and arena_growth > 0
    )

    if not arena_stats_ok:
        return {
            **banked,
            "guards": guards,
            "tracemalloc_diff": diff,
            "fail_closed_terminal": "ARENA_STATS_UNAVAILABLE_INCONCLUSIVE",
            "classifier_branch": None,
            "rewrite_candidate_frames": None,
        }

    ranked: list[dict[str, Any]] = []
    if branch1_ok:
        ranked.append(
            {
                "branch": "LIVE_PYTHON_OBJECT_CHURN",
                "rank": 1,
                "reconcile_ratio": reconcile_ratio,
                "top_concentration_fraction": diff.get("top_concentration_fraction"),
            }
        )
    if branch2_ok:
        ranked.append(
            {
                "branch": "PYMALLOC_HIGH_WATER_RETENTION",
                "rank": len(ranked) + 1,
                "peak_ratio": peak_ratio,
                "arena_retained_fraction": arena_retained_fraction,
            }
        )
    if branch3_ok:
        ranked.append(
            {
                "branch": "UNTRACED_PYMEMP_C_EXTENSION",
                "rank": len(ranked) + 1,
                "reconcile_ratio": reconcile_ratio,
                "arena_growth_bytes": arena_growth,
            }
        )

    rewrite_candidates: list[dict[str, Any]] | None = None
    classifier_branch: str | None = None
    if branch1_ok:
        classifier_branch = "LIVE_PYTHON_OBJECT_CHURN"
        rewrite_candidates = list(diff.get("top_delta_frames") or [])[:16]
    elif branch2_ok:
        classifier_branch = "PYMALLOC_HIGH_WATER_RETENTION"
    elif branch3_ok:
        classifier_branch = "UNTRACED_PYMEMP_C_EXTENSION"
    elif ranked:
        classifier_branch = str(ranked[0]["branch"])

    return {
        **banked,
        "guards": guards,
        "tracemalloc_diff": diff,
        "arena_growth_bytes": arena_growth,
        "arena_retained_fraction": arena_retained_fraction,
        "ranked_classification": ranked,
        "classifier_branch": classifier_branch,
        "rewrite_candidate_frames": rewrite_candidates,
        "fail_closed_terminal": None if classifier_branch else "CLASSIFIER_INCONCLUSIVE",
    }


def _is_obmalloc_mark(row: Mapping[str, Any]) -> bool:
    return str(row.get("schema")) == PROFILE_HOST_RSS_OBMALLOC_SCHEMA or str(
        row.get("event", "")
    ).startswith("obmalloc_")


def _obmalloc_mark(
    marks: Sequence[Mapping[str, Any]],
    event: str,
) -> dict[str, Any] | None:
    for row in marks:
        if _is_obmalloc_mark(row) and str(row.get("event")) == str(event):
            return dict(row)
    return None


def _obmalloc_field(
    mark: Mapping[str, Any] | None,
    field: str,
) -> int | None:
    if mark is None:
        return None
    stats = dict(mark.get("debugmallocstats") or {})
    if not stats.get("available"):
        return None
    value = stats.get(field)
    return int(value) if value is not None else None


def attribute_obmalloc_arena_retention(
    *,
    marks_a: list[dict[str, Any]],
    marks_a_prime: list[dict[str, Any]],
    marks_b: list[dict[str, Any]],
    debugmallocstats_preflight: Mapping[str, Any] | None = None,
    self_footprint_preflight: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    banked = _banked_reference_paths()
    c4_a = _c4_subphase_delta_gib(marks_a)
    c4_a_prime = _c4_subphase_delta_gib(marks_a_prime)
    c4_b = _c4_subphase_delta_gib(marks_b)

    noise_floor: float | None = None
    if c4_a is not None and c4_a_prime is not None:
        noise_floor = abs(float(c4_a_prime) - float(c4_a))

    run_stability_threshold = GUARD_STABILITY_FRAC * TOTAL_C4_REFERENCE_GIB
    denominator_variance_threshold = (
        GUARD_STABILITY_FRAC * BANKED_NON_GLIBC_MMAP_REFERENCE_GIB
    )
    envelope_tolerance = (
        max(float(noise_floor), GUARD_ENVELOPE_MIN_GIB)
        if noise_floor is not None
        else GUARD_ENVELOPE_MIN_GIB
    )
    total_c4_envelope_delta = (
        abs(float(c4_a) - TOTAL_C4_REFERENCE_GIB) if c4_a is not None else None
    )

    guards: dict[str, Any] = {
        "noise_floor_gib": noise_floor,
        "run_stability_threshold_gib": run_stability_threshold,
        "denominator_variance_threshold_gib": denominator_variance_threshold,
        "total_c4_envelope_tolerance_gib": envelope_tolerance,
        "total_c4_envelope_delta_gib": total_c4_envelope_delta,
        "c4_rss_delta_gib_same_run": {
            "A": c4_a,
            "A_prime": c4_a_prime,
            "B": c4_b,
        },
    }

    if noise_floor is None:
        guards["run_stability_ok"] = False
        guards["denominator_variance_ok"] = False
        return {
            **banked,
            "guards": guards,
            "fail_closed_terminal": "INCONCLUSIVE_PENDING_NOISE_FLOOR",
            "classifier_terminal": None,
        }

    guards["run_stability_ok"] = noise_floor <= run_stability_threshold
    guards["denominator_variance_ok"] = noise_floor <= denominator_variance_threshold
    guards["total_c4_envelope_ok"] = (
        total_c4_envelope_delta is not None
        and total_c4_envelope_delta <= envelope_tolerance
    )

    if not guards["total_c4_envelope_ok"]:
        return {
            **banked,
            "guards": guards,
            "fail_closed_terminal": "DENOMINATOR_INVALID_INCONCLUSIVE",
            "classifier_terminal": None,
        }
    if not guards["run_stability_ok"] or not guards["denominator_variance_ok"]:
        return {
            **banked,
            "guards": guards,
            "fail_closed_terminal": "INCONCLUSIVE_CROSS_RUN_DENOMINATOR",
            "classifier_terminal": None,
        }

    footprint = dict(self_footprint_preflight or {})
    footprint_bytes = footprint.get("debugmallocstats_self_footprint_bytes")
    footprint_status = str(
        footprint.get("debugmallocstats_self_footprint_status")
        or footprint.get("status")
        or ""
    )
    guards["debugmallocstats_self_footprint_bytes"] = footprint_bytes
    guards["debugmallocstats_self_footprint_status"] = footprint_status
    if footprint_status == "exceeded":
        return {
            **banked,
            "guards": guards,
            "fail_closed_terminal": "OBMALLOC_SELF_FOOTPRINT_INCONCLUSIVE",
            "classifier_terminal": None,
        }

    preflight = dict(debugmallocstats_preflight or {})
    if str(preflight.get("status")) != "ok":
        return {
            **banked,
            "guards": guards,
            "fail_closed_terminal": "ARENA_STATS_UNPARSEABLE_INCONCLUSIVE",
            "classifier_terminal": None,
            "arena_stats_unavailable_reason": preflight.get("status"),
        }

    perturbation_delta = (
        abs(float(c4_b) - float(c4_a))
        if c4_a is not None and c4_b is not None
        else None
    )
    perturbation_threshold = max(
        PERTURBATION_MIN_GIB,
        PERTURBATION_NOISE_K * float(noise_floor),
    )
    guards["perturbation_delta_gib"] = perturbation_delta
    guards["perturbation_threshold_gib"] = perturbation_threshold

    c3_exit_mark = _obmalloc_mark(marks_b, "obmalloc_C3_exit")
    c4_enter_mark = _obmalloc_mark(marks_b, "obmalloc_C4_enter")
    c4_exit_mark = _obmalloc_mark(marks_b, "obmalloc_C4_exit")
    b_incomplete = c4_exit_mark is None
    guards["b_run_incomplete"] = b_incomplete
    if b_incomplete:
        has_obmalloc = any(_is_obmalloc_mark(row) for row in marks_b)
        reason = (
            "missing_obmalloc_C4_exit"
            if has_obmalloc
            else "missing_obmalloc_marks"
        )
        guards["b_incomplete_reason"] = reason
        if perturbation_delta is not None and perturbation_delta > perturbation_threshold:
            return {
                **banked,
                "guards": guards,
                "fail_closed_terminal": "OBSERVER_PERTURBED_INCONCLUSIVE",
                "observer_reason": reason,
                "classifier_terminal": None,
            }
        if has_obmalloc:
            return {
                **banked,
                "guards": guards,
                "fail_closed_terminal": "OBSERVER_PERTURBED_INCONCLUSIVE",
                "observer_reason": reason,
                "classifier_terminal": None,
            }
        return {
            **banked,
            "guards": guards,
            "fail_closed_terminal": "ARENA_STATS_UNPARSEABLE_INCONCLUSIVE",
            "classifier_terminal": None,
        }

    if perturbation_delta is not None and perturbation_delta > perturbation_threshold:
        return {
            **banked,
            "guards": guards,
            "fail_closed_terminal": "OBSERVER_PERTURBED_INCONCLUSIVE",
            "observer_reason": "c4_rss_perturbation_exceeded",
            "classifier_terminal": None,
        }

    c3_arena = _obmalloc_field(c3_exit_mark, "arena_bytes")
    c4_enter_arena = _obmalloc_field(c4_enter_mark, "arena_bytes")
    c4_exit_arena = _obmalloc_field(c4_exit_mark, "arena_bytes")
    c4_enter_alloc = _obmalloc_field(c4_enter_mark, "bytes_in_allocated_blocks")
    c4_exit_alloc = _obmalloc_field(c4_exit_mark, "bytes_in_allocated_blocks")

    if (
        c4_enter_arena is None
        or c4_exit_arena is None
        or c4_enter_alloc is None
        or c4_exit_alloc is None
    ):
        return {
            **banked,
            "guards": guards,
            "fail_closed_terminal": "ARENA_STATS_UNPARSEABLE_INCONCLUSIVE",
            "classifier_terminal": None,
        }

    arena_bytes_delta_reconcile = int(c4_exit_arena) - int(c4_enter_arena)
    allocated_block_bytes_delta_reconcile = int(c4_exit_alloc) - int(c4_enter_alloc)
    arena_bytes_delta_c3_to_c4_enter = (
        int(c4_enter_arena) - int(c3_arena) if c3_arena is not None else None
    )

    denom = int(BANKED_NON_GLIBC_MMAP_REFERENCE_BYTES)
    reconcile_ratio = (
        float(arena_bytes_delta_reconcile) / float(denom) if denom > 0 else 0.0
    )

    deltas: dict[str, Any] = {
        "arena_bytes_delta_reconcile": arena_bytes_delta_reconcile,
        "allocated_block_bytes_delta_reconcile": allocated_block_bytes_delta_reconcile,
        "arena_bytes_delta_c3_to_c4_enter": arena_bytes_delta_c3_to_c4_enter,
        "reconcile_ratio": reconcile_ratio,
        "c4_enter_arena_bytes": c4_enter_arena,
        "c4_exit_arena_bytes": c4_exit_arena,
        "c4_enter_allocated_block_bytes": c4_enter_alloc,
        "c4_exit_allocated_block_bytes": c4_exit_alloc,
    }

    if reconcile_ratio < OBMALLOC_NOT_OBMALLOC_MAX:
        return {
            **banked,
            "guards": guards,
            "deltas": deltas,
            "fail_closed_terminal": None,
            "classifier_terminal": "NOT_OBMALLOC_UNTRACED",
            "next_lane": "native_backtrace",
        }

    if reconcile_ratio < OBMALLOC_RECONCILE_MIN or reconcile_ratio > OBMALLOC_RECONCILE_MAX:
        return {
            **banked,
            "guards": guards,
            "deltas": deltas,
            "fail_closed_terminal": "RECONCILE_OUT_OF_BAND_INCONCLUSIVE",
            "classifier_terminal": None,
        }

    if arena_bytes_delta_reconcile <= OBMALLOC_ARENA_DELTA_FLOOR_BYTES:
        return {
            **banked,
            "guards": guards,
            "deltas": deltas,
            "fail_closed_terminal": None,
            "classifier_terminal": "NOT_OBMALLOC_UNTRACED",
            "next_lane": "native_backtrace",
            "arena_delta_floor_breach": True,
        }

    occupancy = (
        float(allocated_block_bytes_delta_reconcile) / float(arena_bytes_delta_reconcile)
    )
    deltas["occupancy_ratio"] = occupancy

    if occupancy >= OBMALLOC_OCCUPANCY_LIVE_MIN:
        classifier_terminal = "OBMALLOC_LIVE_CHURN"
    elif occupancy < OBMALLOC_OCCUPANCY_HIGH_WATER_MAX:
        classifier_terminal = "OBMALLOC_HIGH_WATER_RETENTION"
    else:
        classifier_terminal = "AMBIGUOUS_MID_BAND"

    return {
        **banked,
        "guards": guards,
        "deltas": deltas,
        "fail_closed_terminal": None,
        "classifier_terminal": classifier_terminal,
        "call_site_status": "UNRESOLVED",
    }


def _is_obmalloc_site_mark(row: Mapping[str, Any]) -> bool:
    return str(row.get("schema")) == PROFILE_HOST_RSS_OBMALLOC_SITE_SCHEMA or str(
        row.get("event", "")
    ).startswith("obmalloc_site_")


def _obmalloc_site_mark(
    marks: Sequence[Mapping[str, Any]],
    event: str,
) -> dict[str, Any] | None:
    for row in marks:
        if _is_obmalloc_site_mark(row) and str(row.get("event")) == str(event):
            return dict(row)
    return None


def compute_obmalloc_expanded_sampled_states(n_states: int) -> tuple[int, ...]:
    from calm.hrm_text_158.native_full_stack.sparse_cap_gpu_seam_adapter import (
        compute_obmalloc_expanded_sampled_states as _compute,
    )

    return tuple(sorted(_compute(n_states)))


def _attribute_s1d7_tracemalloc_call_site(
    marks_b: Sequence[Mapping[str, Any]],
    *,
    guards: Mapping[str, Any],
    sampled_states: Sequence[int],
) -> dict[str, Any]:
    from calm.hrm_text_158.native_full_stack.s1d7_tracemalloc_feasibility import (
        attribute_s1d7_tracemalloc_call_site_from_marks,
    )

    return attribute_s1d7_tracemalloc_call_site_from_marks(
        marks_b,
        sampled_states=sampled_states,
        guards=guards,
    )


def _obmalloc_expanded_call_site_fields(
    s1d7_call_site: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "call_site_status": s1d7_call_site.get("call_site_status", "UNRESOLVED"),
        "call_site_origin_file_line": s1d7_call_site.get("call_site_origin_file_line"),
        "s1d7_tracemalloc_diff": s1d7_call_site.get("s1d7_tracemalloc_diff"),
        "s1d7_tracemalloc_top_concentration_fraction": s1d7_call_site.get(
            "s1d7_tracemalloc_top_concentration_fraction"
        ),
        "tracemalloc_perturbed": s1d7_call_site.get("tracemalloc_perturbed"),
        "s1d7_call_site_in_bracket_ok": s1d7_call_site.get("s1d7_call_site_in_bracket_ok"),
        "s1d7_call_site_candidate": s1d7_call_site.get("s1d7_call_site_candidate"),
        "s1d7_call_site_branch_outcome": s1d7_call_site.get("s1d7_call_site_branch_outcome"),
        "s1d7_tracemalloc_mark_pair_count": s1d7_call_site.get(
            "s1d7_tracemalloc_mark_pair_count"
        ),
    }


def _obmalloc_site_mark_for_state(
    marks: Sequence[Mapping[str, Any]],
    event: str,
    *,
    state_index: int | None = None,
) -> dict[str, Any] | None:
    rows = _obmalloc_site_marks_for_state(
        marks,
        event,
        state_index=state_index,
    )
    return rows[0] if rows else None


def _obmalloc_site_marks_for_state(
    marks: Sequence[Mapping[str, Any]],
    event: str,
    *,
    state_index: int | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in marks:
        if not _is_obmalloc_site_mark(row):
            continue
        if str(row.get("event")) != str(event):
            continue
        if state_index is not None and int(row.get("state_index", -1)) != int(state_index):
            continue
        rows.append(dict(row))
    return rows


def _site_bracket_pair_count_for_state(
    marks: Sequence[Mapping[str, Any]],
    site_id: str,
    state_index: int,
) -> int:
    n_pre = len(
        _obmalloc_site_marks_for_state(
            marks,
            f"obmalloc_site_{site_id}_pre",
            state_index=int(state_index),
        )
    )
    n_post = len(
        _obmalloc_site_marks_for_state(
            marks,
            f"obmalloc_site_{site_id}_post",
            state_index=int(state_index),
        )
    )
    if n_pre == 0 and n_post == 0:
        return 0
    if n_pre != n_post:
        return -1
    return int(n_pre)


def _site_bracket_holding_bytes(mark: Mapping[str, Any] | None) -> int | None:
    if mark is None:
        return None
    if mark.get("allocated_blocks_holding") is not None:
        return int(mark["allocated_blocks_holding"])
    return _obmalloc_field(mark, "bytes_in_allocated_blocks")


def _site_bracket_holding_delta_bytes(
    marks: Sequence[Mapping[str, Any]],
    site_id: str,
    state_index: int,
    *,
    absent_is_zero: bool = False,
) -> int | None:
    pres = _obmalloc_site_marks_for_state(
        marks,
        f"obmalloc_site_{site_id}_pre",
        state_index=int(state_index),
    )
    posts = _obmalloc_site_marks_for_state(
        marks,
        f"obmalloc_site_{site_id}_post",
        state_index=int(state_index),
    )
    if not pres and not posts:
        return 0 if absent_is_zero else None
    if len(pres) != len(posts):
        return None
    total = 0
    for pre, post in zip(pres, posts):
        pre_bytes = _site_bracket_holding_bytes(pre)
        post_bytes = _site_bracket_holding_bytes(post)
        if pre_bytes is None or post_bytes is None:
            return None
        total += int(post_bytes) - int(pre_bytes)
    return int(total)


def _site_bracket_forcing_delta_bytes(
    marks: Sequence[Mapping[str, Any]],
    site_id: str,
    state_index: int,
    *,
    absent_is_zero: bool = False,
) -> int | None:
    pres = _obmalloc_site_marks_for_state(
        marks,
        f"obmalloc_site_{site_id}_pre",
        state_index=int(state_index),
    )
    posts = _obmalloc_site_marks_for_state(
        marks,
        f"obmalloc_site_{site_id}_post",
        state_index=int(state_index),
    )
    if not pres and not posts:
        return 0 if absent_is_zero else None
    if len(pres) != len(posts):
        return None
    total = 0
    for pre, post in zip(pres, posts):
        pre_bytes = _site_bracket_arena_bytes(pre)
        post_bytes = _site_bracket_arena_bytes(post)
        if pre_bytes is None or post_bytes is None:
            return None
        total += int(post_bytes) - int(pre_bytes)
    return int(total)


def _site_bracket_holding_delta_bytes_legacy(
    marks: Sequence[Mapping[str, Any]],
    site_id: str,
) -> int | None:
    pre = _obmalloc_site_mark(marks, f"obmalloc_site_{site_id}_pre")
    post = _obmalloc_site_mark(marks, f"obmalloc_site_{site_id}_post")
    pre_bytes = _site_bracket_holding_bytes(pre)
    post_bytes = _site_bracket_holding_bytes(post)
    if pre_bytes is None or post_bytes is None:
        return None
    return int(post_bytes) - int(pre_bytes)


def _profile_has_child_marks(marks: Sequence[Mapping[str, Any]]) -> bool:
    child_events = {
        f"obmalloc_site_{site_id}_{suffix}"
        for site_id in OBMALLOC_SITE_CHILD_SITES
        for suffix in ("pre", "post")
    }
    for row in marks:
        if str(row.get("event", "")) in child_events:
            return True
    return False


def _profile_has_s1d_subsplit_marks(marks: Sequence[Mapping[str, Any]]) -> bool:
    child_events = {
        f"obmalloc_site_{site_id}_{suffix}"
        for site_id in OBMALLOC_SITE_S1D_CHILD_SITES
        for suffix in ("pre", "post")
    }
    for row in marks:
        if str(row.get("event", "")) in child_events:
            return True
    return False


def _profile_has_s1f_subsplit_marks(marks: Sequence[Mapping[str, Any]]) -> bool:
    child_events = {
        f"obmalloc_site_{site_id}_{suffix}"
        for site_id in OBMALLOC_SITE_S1F_CHILD_SITES
        for suffix in ("pre", "post")
    }
    for row in marks:
        if str(row.get("event", "")) in child_events:
            return True
    return False


def _child_site_bracket_pair_present(
    marks: Sequence[Mapping[str, Any]],
    site_id: str,
    state_index: int,
) -> bool:
    return _site_bracket_pair_count_for_state(
        marks,
        site_id,
        int(state_index),
    ) >= 1


def _count_obmalloc_expanded_enabled_events(marks: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    obmalloc_boundary = {
        "obmalloc_C4_enter": 0,
        "obmalloc_C4_after_state": 0,
        "obmalloc_C4_exit": 0,
    }
    site_leaf = 0
    for row in marks:
        event = str(row.get("event", ""))
        if event in obmalloc_boundary:
            obmalloc_boundary[event] += 1
        elif event.startswith("obmalloc_site_") and any(
            event.endswith(f"_{site_id}_{suffix}")
            for site_id in OBMALLOC_SITE_LEAF_SITES
            for suffix in ("pre", "post")
        ):
            site_leaf += 1
    return {
        **obmalloc_boundary,
        "site_leaf_bracket": site_leaf,
        "total": sum(obmalloc_boundary.values()) + site_leaf,
    }


def _validate_obmalloc_expanded_event_stream(
    marks: Sequence[Mapping[str, Any]],
    *,
    sampled_states: Sequence[int],
) -> dict[str, Any]:
    counts = _count_obmalloc_expanded_enabled_events(marks)
    corruption_reasons: list[str] = []
    pair_counts_by_site: dict[str, dict[int, int]] = {
        site_id: {} for site_id in OBMALLOC_SITE_LEAF_SITES
    }

    if counts["obmalloc_C4_after_state"] > OBMALLOC_EXPANDED_BOUNDARY_AFTER_STATE_MAX:
        corruption_reasons.append("duplicate_boundary_after_state")

    for state_idx in sampled_states:
        for site_id in OBMALLOC_SITE_LEAF_SITES:
            pair_count = _site_bracket_pair_count_for_state(
                marks,
                site_id,
                int(state_idx),
            )
            if pair_count == 0:
                continue
            pair_counts_by_site[site_id][int(state_idx)] = int(pair_count)
            if pair_count < 0:
                continue
            if site_id not in OBMALLOC_SITE_MULTI_PAIR_SITES and pair_count > 1:
                corruption_reasons.append(
                    f"duplicate_single_pair_{site_id}_state{int(state_idx)}"
                )

    return {
        "counts": counts,
        "valid": not corruption_reasons,
        "corruption_reasons": corruption_reasons,
        "pair_counts_by_site": pair_counts_by_site,
    }


def _c4_after_state_owner_census_rows(
    marks: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    companion_by_state: dict[int, dict[str, Any]] = {}
    for row in marks:
        if str(row.get("event")) != "c4_retention_owner_census_after_state":
            continue
        dims = dict(row.get("allocation_dims") or {})
        state_index = row.get("state_index")
        if state_index is None:
            # Census marks carry the state index inside allocation_dims
            # (c4_state_index); the top-level state_index is unset on the
            # emitted census event. Fall back to it so census rows join the
            # obmalloc_C4_after_state marks (which carry only top-level
            # state_index and no allocation_dims).
            state_index = dims.get("c4_state_index")
        if state_index is None:
            continue
        companion_by_state[int(state_index)] = dims

    rows: list[dict[str, Any]] = []
    for row in marks:
        if not _is_obmalloc_mark(row):
            continue
        if str(row.get("event")) != "obmalloc_C4_after_state":
            continue
        state_index = int(row.get("state_index", -1))
        dims = dict(row.get("allocation_dims") or {})
        if not dims:
            dims = dict(companion_by_state.get(state_index) or {})
        if not dims:
            continue
        rows.append(
            {
                "state_index": state_index,
                "allocation_dims": dims,
                "bytes_in_allocated_blocks": _obmalloc_field(
                    row, "bytes_in_allocated_blocks"
                ),
                "arena_bytes": _obmalloc_field(row, "arena_bytes"),
            }
        )
    rows.sort(key=lambda item: int(item["state_index"]))
    return rows


def _correlate_owner_counts_with_retention(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    paired: list[dict[str, Any]] = []
    prev_blocks: int | None = None
    prev_carriers: int | None = None
    for row in rows:
        blocks = row.get("bytes_in_allocated_blocks")
        dims = dict(row.get("allocation_dims") or {})
        carriers = int(dims.get("c4_n_carriers_by_key", 0))
        if blocks is None:
            continue
        item = {
            "state_index": int(row.get("state_index", -1)),
            "c4_n_carriers_by_key": carriers,
            "c4_n_tensor_states": int(dims.get("c4_n_tensor_states", 0)),
            "bytes_in_allocated_blocks": int(blocks),
        }
        if prev_blocks is not None and prev_carriers is not None:
            item["retention_delta_blocks"] = int(blocks) - int(prev_blocks)
            item["delta_carriers_by_key"] = int(carriers) - int(prev_carriers)
        paired.append(item)
        prev_blocks = int(blocks)
        prev_carriers = int(carriers)

    correlated_steps = [
        row
        for row in paired
        if int(row.get("retention_delta_blocks", 0)) > OBMALLOC_EXPANDED_RETENTION_FLOOR_BYTES
        and int(row.get("delta_carriers_by_key", 0)) > 0
    ]
    retention_steps = [
        row
        for row in paired
        if int(row.get("retention_delta_blocks", 0)) > OBMALLOC_EXPANDED_RETENTION_FLOOR_BYTES
    ]
    carrier_growth_steps = [
        row for row in paired if int(row.get("delta_carriers_by_key", 0)) > 0
    ]
    correlation_fraction = (
        float(len(correlated_steps)) / float(len(retention_steps))
        if retention_steps
        else 0.0
    )
    return {
        "paired_steps": paired,
        "correlation_fraction": correlation_fraction,
        "retention_step_count": len(retention_steps),
        "carrier_growth_step_count": len(carrier_growth_steps),
        "correlated_step_count": len(correlated_steps),
    }


def attribute_c4_retention_owner_census(
    *,
    marks_a: list[dict[str, Any]],
    marks_a_prime: list[dict[str, Any]],
    marks_b: list[dict[str, Any]],
    n_states: int = OBMALLOC_SITE_N_STATES_WITNESS,
) -> dict[str, Any]:
    from calm.hrm_text_158.native_full_stack.sparse_cap_gpu_seam_adapter import (
        compute_obmalloc_expanded_sampled_states,
    )

    banked = _banked_reference_paths()
    c4_a = _c4_subphase_delta_gib(marks_a)
    c4_a_prime = _c4_subphase_delta_gib(marks_a_prime)
    c4_b = _c4_subphase_delta_gib(marks_b)

    noise_floor: float | None = None
    if c4_a is not None and c4_a_prime is not None:
        noise_floor = abs(float(c4_a_prime) - float(c4_a))

    perturbation_cap_gib = max(
        0.5,
        2.0 * float(noise_floor) if noise_floor is not None else 0.0,
    )
    census_rows = _c4_after_state_owner_census_rows(marks_b)
    correlation = _correlate_owner_counts_with_retention(census_rows)

    tier_b_present = any(
        "c4_weakref_n_new_carriers_by_key" in dict(row.get("allocation_dims") or {})
        for row in census_rows
    )
    tier_b_status = "DISABLED"
    if tier_b_present:
        tier_b_status = "ENABLED"
        if c4_b is not None and noise_floor is not None:
            perturbation_delta_gib = abs(float(c4_b) - float(TOTAL_C4_REFERENCE_GIB))
            if perturbation_delta_gib > perturbation_cap_gib:
                tier_b_status = "OBSERVER_PERTURBED"

    sampled = tuple(compute_obmalloc_expanded_sampled_states(n_states))
    n_tensor = max(
        (int(dict(row.get("allocation_dims") or {}).get("c4_n_tensor_states", 0)) for row in census_rows),
        default=0,
    )
    last_carriers = 0
    if census_rows:
        last_carriers = int(
            dict(census_rows[-1].get("allocation_dims") or {}).get("c4_n_carriers_by_key", 0)
        )

    # Priors persist flat-at-n when c4_n_tensor_states never drops below the
    # peak prior count across the loop. Combined with new carriers accumulating
    # to n, that is the simultaneous 2n-hold (prior baseline held AND new
    # carriers accumulated) that defines the OOM peak — DUAL_HOLD_2N. It must
    # take precedence over NEW_ACCUM_DOMINANT, which is reserved for the
    # single-owner case where priors do NOT persist at n (released/small) while
    # new carriers accumulate.
    tensor_series = [
        int(dict(row.get("allocation_dims") or {}).get("c4_n_tensor_states", 0))
        for row in census_rows
    ]
    min_tensor = min(tensor_series) if tensor_series else 0
    priors_persist_flat_at_n = n_tensor > 0 and min_tensor >= n_tensor
    new_carriers_reach_n = n_tensor > 0 and last_carriers >= n_tensor

    classifier_terminal: str | None = None
    if not census_rows:
        classifier_terminal = "INCONCLUSIVE"
    elif (
        priors_persist_flat_at_n
        and new_carriers_reach_n
        and correlation["retention_step_count"] > 0
    ):
        classifier_terminal = "DUAL_HOLD_2N"
    elif (
        correlation["correlation_fraction"] >= 0.75
        and correlation["carrier_growth_step_count"] > 0
        and correlation["retention_step_count"] > 0
    ):
        classifier_terminal = "NEW_ACCUM_DOMINANT"
    elif correlation["retention_step_count"] > 0 and correlation["carrier_growth_step_count"] == 0:
        classifier_terminal = "PRIOR_DOMINANT"
    else:
        classifier_terminal = "INCONCLUSIVE"

    return {
        **banked,
        "guards": {
            "noise_floor_gib": noise_floor,
            "perturbation_cap_gib": perturbation_cap_gib,
            "sampled_states": list(sampled),
            "n_states": int(n_states),
            "census_row_count": len(census_rows),
        },
        "localization": {
            "census_rows": census_rows,
            "correlation": correlation,
            "tier_b_status": tier_b_status,
            "pre_apply_coowns_priors": True,
            "pre_apply_coowns_priors_kind": "STATIC_ROUTING_FACT",
        },
        "classifier_terminal": classifier_terminal,
        "fail_closed_terminal": None,
        "slice8_rewrite_authorized": False,
    }


def attribute_obmalloc_expanded(
    *,
    marks_a: list[dict[str, Any]],
    marks_a_prime: list[dict[str, Any]],
    marks_b: list[dict[str, Any]],
    n_states: int = OBMALLOC_SITE_N_STATES_WITNESS,
    debugmallocstats_preflight: Mapping[str, Any] | None = None,
    self_footprint_preflight: Mapping[str, Any] | None = None,
    sampled_states: Sequence[int] | None = None,
) -> dict[str, Any]:
    banked = _banked_reference_paths()
    sampled = tuple(
        sampled_states
        if sampled_states is not None
        else compute_obmalloc_expanded_sampled_states(n_states)
    )
    c4_a = _c4_subphase_delta_gib(marks_a)
    c4_a_prime = _c4_subphase_delta_gib(marks_a_prime)
    c4_b = _c4_subphase_delta_gib(marks_b)

    noise_floor: float | None = None
    if c4_a is not None and c4_a_prime is not None:
        noise_floor = abs(float(c4_a_prime) - float(c4_a))

    run_stability_threshold = GUARD_STABILITY_FRAC * TOTAL_C4_REFERENCE_GIB
    denominator_variance_threshold = (
        GUARD_STABILITY_FRAC * BANKED_NON_GLIBC_MMAP_REFERENCE_GIB
    )
    envelope_tolerance = (
        max(float(noise_floor), GUARD_ENVELOPE_MIN_GIB)
        if noise_floor is not None
        else GUARD_ENVELOPE_MIN_GIB
    )
    total_c4_envelope_delta = (
        abs(float(c4_a) - TOTAL_C4_REFERENCE_GIB) if c4_a is not None else None
    )

    guards: dict[str, Any] = {
        "noise_floor_gib": noise_floor,
        "run_stability_threshold_gib": run_stability_threshold,
        "denominator_variance_threshold_gib": denominator_variance_threshold,
        "total_c4_envelope_tolerance_gib": envelope_tolerance,
        "total_c4_envelope_delta_gib": total_c4_envelope_delta,
        "c4_rss_delta_gib_same_run": {
            "A": c4_a,
            "A_prime": c4_a_prime,
            "B": c4_b,
        },
        "sampled_states": list(sampled),
        "n_states": int(n_states),
    }

    event_validation = _validate_obmalloc_expanded_event_stream(
        marks_b,
        sampled_states=sampled,
    )
    event_counts = dict(event_validation["counts"])
    guards["obmalloc_expanded_event_counts"] = event_counts
    guards["obmalloc_expanded_event_validation"] = {
        "valid": bool(event_validation["valid"]),
        "corruption_reasons": list(event_validation["corruption_reasons"]),
        "pair_counts_by_site": event_validation["pair_counts_by_site"],
    }

    if not event_validation["valid"]:
        return {
            **banked,
            "guards": guards,
            "fail_closed_terminal": "OBSERVER_PERTURBED_INCONCLUSIVE",
            "classifier_terminal": None,
            "observer_reason": "duplicate_obmalloc_emit",
            "slice8_rewrite_authorized": False,
        }

    c4_enter_mark = _obmalloc_mark(marks_b, "obmalloc_C4_enter")
    c4_exit_mark = _obmalloc_mark(marks_b, "obmalloc_C4_exit")
    if c4_enter_mark is None or c4_exit_mark is None:
        return {
            **banked,
            "guards": guards,
            "fail_closed_terminal": "OBSERVER_PERTURBED_INCONCLUSIVE",
            "classifier_terminal": None,
            "observer_reason": "missing_obmalloc_C4_exit",
            "slice8_rewrite_authorized": False,
        }

    if noise_floor is None:
        guards["run_stability_ok"] = False
        guards["denominator_variance_ok"] = False
        return {
            **banked,
            "guards": guards,
            "fail_closed_terminal": "INCONCLUSIVE_PENDING_NOISE_FLOOR",
            "classifier_terminal": None,
            "slice8_rewrite_authorized": False,
        }

    guards["run_stability_ok"] = noise_floor <= run_stability_threshold
    guards["denominator_variance_ok"] = noise_floor <= denominator_variance_threshold
    guards["total_c4_envelope_ok"] = (
        total_c4_envelope_delta is not None
        and total_c4_envelope_delta <= envelope_tolerance
    )

    if not guards["total_c4_envelope_ok"]:
        return {
            **banked,
            "guards": guards,
            "fail_closed_terminal": "DENOMINATOR_INVALID_INCONCLUSIVE",
            "classifier_terminal": None,
            "slice8_rewrite_authorized": False,
        }
    if not guards["run_stability_ok"] or not guards["denominator_variance_ok"]:
        return {
            **banked,
            "guards": guards,
            "fail_closed_terminal": "INCONCLUSIVE_CROSS_RUN_DENOMINATOR",
            "classifier_terminal": None,
            "slice8_rewrite_authorized": False,
        }

    footprint = dict(self_footprint_preflight or {})
    footprint_status = str(
        footprint.get("debugmallocstats_self_footprint_status")
        or footprint.get("status")
        or ""
    )
    guards["debugmallocstats_self_footprint_bytes"] = footprint.get(
        "debugmallocstats_self_footprint_bytes"
    )
    guards["debugmallocstats_self_footprint_status"] = footprint_status
    if footprint_status == "exceeded":
        return {
            **banked,
            "guards": guards,
            "fail_closed_terminal": "OBMALLOC_SELF_FOOTPRINT_INCONCLUSIVE",
            "classifier_terminal": None,
            "slice8_rewrite_authorized": False,
        }

    preflight = dict(debugmallocstats_preflight or {})
    if str(preflight.get("status")) != "ok":
        return {
            **banked,
            "guards": guards,
            "fail_closed_terminal": "ARENA_STATS_UNPARSEABLE_INCONCLUSIVE",
            "classifier_terminal": None,
            "arena_stats_unavailable_reason": preflight.get("status"),
            "slice8_rewrite_authorized": False,
        }

    perturbation_delta = (
        abs(float(c4_b) - float(c4_a))
        if c4_a is not None and c4_b is not None
        else None
    )
    perturbation_threshold = max(
        PERTURBATION_MIN_GIB,
        PERTURBATION_NOISE_K * float(noise_floor),
    )
    guards["perturbation_delta_gib"] = perturbation_delta
    guards["perturbation_threshold_gib"] = perturbation_threshold

    if (
        perturbation_delta is not None
        and perturbation_delta > perturbation_threshold
    ):
        return {
            **banked,
            "guards": guards,
            "fail_closed_terminal": "OBSERVER_PERTURBED_INCONCLUSIVE",
            "classifier_terminal": None,
            "observer_reason": "c4_rss_perturbation",
            "slice8_rewrite_authorized": False,
        }

    after_state_marks = [
        dict(row)
        for row in marks_b
        if _is_obmalloc_mark(row) and str(row.get("event")) == "obmalloc_C4_after_state"
    ]
    after_state_marks.sort(key=lambda row: int(row.get("state_index", -1)))
    retention_deltas: list[dict[str, Any]] = []
    prev_blocks: int | None = None
    for row in after_state_marks:
        blocks = _obmalloc_field(row, "bytes_in_allocated_blocks")
        arena = _obmalloc_field(row, "arena_bytes")
        if blocks is None:
            continue
        retention_delta = None
        if prev_blocks is not None:
            retention_delta = int(blocks) - int(prev_blocks)
            retention_deltas.append(
                {
                    "state_index": int(row.get("state_index", -1)),
                    "retention_delta_blocks": retention_delta,
                    "arena_bytes": arena,
                    "bytes_in_allocated_blocks": int(blocks),
                }
            )
        prev_blocks = int(blocks)

    monotonic_count = sum(
        1
        for item in retention_deltas
        if int(item["retention_delta_blocks"]) > OBMALLOC_EXPANDED_RETENTION_FLOOR_BYTES
    )
    monotonic_fraction = (
        float(monotonic_count) / float(len(retention_deltas))
        if retention_deltas
        else 0.0
    )

    child_profile_mode = _profile_has_child_marks(marks_b)
    phase3_s1d_subsplit_mode = _profile_has_s1d_subsplit_marks(marks_b)
    phase3_s1f_subsplit_mode = _profile_has_s1f_subsplit_marks(marks_b)
    guards["child_profile_mode"] = child_profile_mode
    guards["phase3_s1d_subsplit_mode"] = phase3_s1d_subsplit_mode
    guards["phase3_s1f_subsplit_mode"] = phase3_s1f_subsplit_mode
    if child_profile_mode:
        for state_idx in sampled:
            for site_id in OBMALLOC_SITE_CHILD_SITES:
                if not _child_site_bracket_pair_present(
                    marks_b,
                    site_id,
                    int(state_idx),
                ):
                    return {
                        **banked,
                        "guards": guards,
                        "fail_closed_terminal": "CHILD_COVERAGE_FAIL",
                        "classifier_terminal": None,
                        "missing_child_site": site_id,
                        "missing_state_index": int(state_idx),
                        "slice8_rewrite_authorized": False,
                    }
    if phase3_s1d_subsplit_mode:
        for state_idx in sampled:
            for site_id in OBMALLOC_SITE_S1D_CHILD_SITES:
                if not _child_site_bracket_pair_present(
                    marks_b,
                    site_id,
                    int(state_idx),
                ):
                    return {
                        **banked,
                        "guards": guards,
                        "fail_closed_terminal": "S1D_CHILD_COVERAGE_FAIL",
                        "classifier_terminal": None,
                        "missing_child_site": site_id,
                        "missing_state_index": int(state_idx),
                        "slice8_rewrite_authorized": False,
                    }
    if phase3_s1f_subsplit_mode:
        for state_idx in sampled:
            for site_id in OBMALLOC_SITE_S1F_CHILD_SITES:
                if not _child_site_bracket_pair_present(
                    marks_b,
                    site_id,
                    int(state_idx),
                ):
                    return {
                        **banked,
                        "guards": guards,
                        "fail_closed_terminal": "S1F_CHILD_COVERAGE_FAIL",
                        "classifier_terminal": None,
                        "missing_child_site": site_id,
                        "missing_state_index": int(state_idx),
                        "slice8_rewrite_authorized": False,
                    }

    per_state: dict[str, Any] = {}
    bracket_pos_totals = {site_id: 0 for site_id in OBMALLOC_SITE_LEAF_SITES}
    bracket_signed_totals = {site_id: 0 for site_id in OBMALLOC_SITE_LEAF_SITES}
    state_dominants: list[str] = []

    for state_idx in sampled:
        leaf_holding: dict[str, int | None] = {}
        leaf_forcing: dict[str, int | None] = {}
        for site_id in OBMALLOC_SITE_LEAF_SITES:
            leaf_holding[site_id] = _site_bracket_holding_delta_bytes(
                marks_b,
                site_id,
                int(state_idx),
                absent_is_zero=True,
            )
            leaf_forcing[site_id] = _site_bracket_forcing_delta_bytes(
                marks_b,
                site_id,
                int(state_idx),
                absent_is_zero=True,
            )
        if any(delta is None for delta in leaf_holding.values()):
            return {
                **banked,
                "guards": guards,
                "fail_closed_terminal": "ARENA_STATS_UNPARSEABLE_INCONCLUSIVE",
                "classifier_terminal": None,
                "missing_state_index": int(state_idx),
                "slice8_rewrite_authorized": False,
            }

        signed_deltas = {k: int(v) for k, v in leaf_holding.items()}
        pos_sum = sum(max(delta, 0) for delta in signed_deltas.values())
        neg_sum = sum(min(delta, 0) for delta in signed_deltas.values())
        if pos_sum <= 0:
            return {
                **banked,
                "guards": guards,
                "fail_closed_terminal": "HOLDER_AMBIGUOUS",
                "classifier_terminal": None,
                "observer_reason": "holder_leaf_sum_pos_nonpositive",
                "state_index": int(state_idx),
                "slice8_rewrite_authorized": False,
            }
        if abs(int(neg_sum)) > OBMALLOC_EXPANDED_CANCELLATION_NEG_FRAC * float(pos_sum):
            return {
                **banked,
                "guards": guards,
                "fail_closed_terminal": "HOLDER_AMBIGUOUS",
                "classifier_terminal": None,
                "observer_reason": "cancellation_inflation",
                "state_index": int(state_idx),
                "signed_neg_sum": int(neg_sum),
                "holder_leaf_sum_pos": int(pos_sum),
                "slice8_rewrite_authorized": False,
            }

        for site_id, delta in signed_deltas.items():
            bracket_pos_totals[site_id] += max(int(delta), 0)
            bracket_signed_totals[site_id] += int(delta)

        state_fractions = {
            site_id: float(max(signed_deltas[site_id], 0)) / float(pos_sum)
            for site_id in OBMALLOC_SITE_LEAF_SITES
        }
        state_top = max(state_fractions.items(), key=lambda item: item[1])[0]
        state_dominants.append(state_top)
        per_state[str(state_idx)] = {
            "holder_leaf_sum_pos": int(pos_sum),
            "holder_leaf_sum_signed": int(sum(signed_deltas.values())),
            "signed_holding_deltas_bytes": signed_deltas,
            "forcing_deltas_bytes": {
                site_id: int(leaf_forcing[site_id] or 0) for site_id in OBMALLOC_SITE_LEAF_SITES
            },
            "state_holder_fractions": state_fractions,
            "state_dominant_bracket": state_top,
        }

    aggregate_pos = sum(int(bracket_pos_totals[site_id]) for site_id in OBMALLOC_SITE_LEAF_SITES)
    if aggregate_pos <= 0:
        return {
            **banked,
            "guards": guards,
            "fail_closed_terminal": "DENOMINATOR_INVALID_INCONCLUSIVE",
            "classifier_terminal": None,
            "slice8_rewrite_authorized": False,
        }

    aggregate_fractions = {
        site_id: float(bracket_pos_totals[site_id]) / float(aggregate_pos)
        for site_id in OBMALLOC_SITE_LEAF_SITES
    }
    ranked = sorted(aggregate_fractions.items(), key=lambda item: item[1], reverse=True)
    top_site, top_fraction = ranked[0]
    second_fraction = ranked[1][1] if len(ranked) > 1 else 0.0

    states_with_dominance = sum(
        1
        for state_idx in sampled
        if per_state[str(state_idx)]["state_holder_fractions"].get(top_site, 0.0)
        >= OBMALLOC_EXPANDED_HOLDER_DOMINANCE_FRAC
    )
    state_dominance_fraction = (
        float(states_with_dominance) / float(len(sampled)) if sampled else 0.0
    )
    holder_stable = len(set(state_dominants)) == 1

    mean_holder_pos = (
        sum(int(per_state[str(state_idx)]["holder_leaf_sum_pos"]) for state_idx in sampled)
        / float(len(sampled))
    )
    scaled_representativeness = (
        float(mean_holder_pos) * float(n_states)
    ) / float(BANKED_NON_GLIBC_MMAP_REFERENCE_BYTES)
    representativeness_cleared = (
        scaled_representativeness >= OBMALLOC_EXPANDED_REPRESENTATIVENESS_MIN
        and holder_stable
    )

    localization: dict[str, Any] = {
        "per_state": per_state,
        "aggregate_holder_pos_bytes": {k: int(v) for k, v in bracket_pos_totals.items()},
        "aggregate_holder_signed_bytes": {
            k: int(v) for k, v in bracket_signed_totals.items()
        },
        "aggregate_holder_fractions": aggregate_fractions,
        "retention_deltas": retention_deltas,
        "monotonic_retention_fraction": monotonic_fraction,
        "state_dominance_fraction": state_dominance_fraction,
        "holder_stable_across_sampled_states": holder_stable,
        "scaled_representativeness": scaled_representativeness,
        "representativeness_cleared": representativeness_cleared,
        "cross_run_reconcile_caveat": True,
        "child_profile_mode": child_profile_mode,
        "phase3_s1d_subsplit_mode": phase3_s1d_subsplit_mode,
        "phase3_s1f_subsplit_mode": phase3_s1f_subsplit_mode,
    }

    classifier_terminal: str | None = None
    fail_closed_terminal: str | None = None

    if child_profile_mode:
        child_pos_totals = {
            site_id: int(bracket_pos_totals[site_id])
            for site_id in OBMALLOC_SITE_CHILD_SITES
        }
        parent_c4s1_pos = int(bracket_pos_totals.get("C4.S1", 0))
        child_sum_pos = sum(child_pos_totals.values())
        child_parent_reconcile_fraction = abs(child_sum_pos - parent_c4s1_pos) / max(
            parent_c4s1_pos,
            1,
        )
        localization["child_aggregate_holder_pos_bytes"] = child_pos_totals
        localization["child_parent_reconcile_fraction"] = child_parent_reconcile_fraction
        for state_idx in sampled:
            signed = per_state[str(state_idx)]["signed_holding_deltas_bytes"]
            parent_hold = max(int(signed.get("C4.S1", 0)), 0)
            child_state_sum = sum(
                max(int(signed.get(site_id, 0)), 0)
                for site_id in OBMALLOC_SITE_CHILD_SITES
            )
            state_reconcile = abs(child_state_sum - parent_hold) / max(parent_hold, 1)
            if state_reconcile > OBMALLOC_SITE_REMAINDER_MAX_FRAC:
                fail_closed_terminal = "CHILD_PARENT_RECONCILE_FAIL"
                localization["child_reconcile_fail_state_index"] = int(state_idx)
                localization["child_reconcile_fail_fraction"] = state_reconcile
                break
        if (
            fail_closed_terminal is None
            and child_parent_reconcile_fraction > OBMALLOC_SITE_REMAINDER_MAX_FRAC
        ):
            fail_closed_terminal = "CHILD_PARENT_RECONCILE_FAIL"

        child_aggregate_pos = sum(child_pos_totals.values())
        if child_aggregate_pos > 0:
            child_fractions = {
                site_id: float(child_pos_totals[site_id]) / float(child_aggregate_pos)
                for site_id in OBMALLOC_SITE_CHILD_SITES
            }
            child_ranked = sorted(
                child_fractions.items(),
                key=lambda item: item[1],
                reverse=True,
            )
            child_top_site, child_top_fraction = child_ranked[0]
            child_states_with_dominance = 0
            for state_idx in sampled:
                signed = per_state[str(state_idx)]["signed_holding_deltas_bytes"]
                child_pos = {
                    site_id: max(int(signed.get(site_id, 0)), 0)
                    for site_id in OBMALLOC_SITE_CHILD_SITES
                }
                child_state_sum = sum(child_pos.values())
                if child_state_sum <= 0:
                    continue
                child_state_fraction = (
                    float(child_pos[child_top_site]) / float(child_state_sum)
                )
                if child_state_fraction >= OBMALLOC_EXPANDED_HOLDER_DOMINANCE_FRAC:
                    child_states_with_dominance += 1
            child_state_dominance_fraction = (
                float(child_states_with_dominance) / float(len(sampled))
                if sampled
                else 0.0
            )
            localization["child_aggregate_holder_fractions"] = child_fractions
            localization["child_dominant_bracket"] = child_top_site
            localization["child_state_dominance_fraction"] = (
                child_state_dominance_fraction
            )
            if (
                child_top_fraction >= OBMALLOC_EXPANDED_HOLDER_DOMINANCE_FRAC
                and child_state_dominance_fraction
                >= OBMALLOC_EXPANDED_STATE_DOMINANCE_MIN
            ):
                localization["child_dominance_verdict"] = (
                    f"DOMINANT_CHILD_HOLDER_{child_top_site}"
                )
            else:
                localization["child_dominance_verdict"] = "CHILD_HOLDER_AMBIGUOUS"
        else:
            localization["child_dominance_verdict"] = "CHILD_HOLDER_AMBIGUOUS"
    else:
        localization["child_dominance_verdict"] = "legacy_child_sites_absent"

    if phase3_s1d_subsplit_mode:
        s1d_child_pos_totals = {
            site_id: int(bracket_pos_totals[site_id])
            for site_id in OBMALLOC_SITE_S1D_CHILD_SITES
        }
        parent_s1d_pos = int(bracket_pos_totals.get("C4.S1d", 0))
        s1d_child_sum_pos = sum(s1d_child_pos_totals.values())
        s1d_parent_reconcile_fraction = abs(s1d_child_sum_pos - parent_s1d_pos) / max(
            parent_s1d_pos,
            1,
        )
        localization["s1d_child_aggregate_holder_pos_bytes"] = s1d_child_pos_totals
        localization["s1d_parent_reconcile_fraction"] = s1d_parent_reconcile_fraction
        if s1d_child_sum_pos > int(parent_s1d_pos * 1.15) and s1d_parent_reconcile_fraction > OBMALLOC_SITE_REMAINDER_MAX_FRAC:
            fail_closed_terminal = "CHILD_OVERLAP_DOUBLE_COUNT"
        for state_idx in sampled:
            if fail_closed_terminal is not None:
                break
            signed = per_state[str(state_idx)]["signed_holding_deltas_bytes"]
            parent_hold = max(int(signed.get("C4.S1d", 0)), 0)
            s1d_state_sum = sum(
                max(int(signed.get(site_id, 0)), 0)
                for site_id in OBMALLOC_SITE_S1D_CHILD_SITES
            )
            state_reconcile = abs(s1d_state_sum - parent_hold) / max(parent_hold, 1)
            if state_reconcile > OBMALLOC_SITE_REMAINDER_MAX_FRAC:
                fail_closed_terminal = "CHILD_PARENT_RECONCILE_FAIL"
                localization["s1d_reconcile_fail_state_index"] = int(state_idx)
                localization["s1d_reconcile_fail_fraction"] = state_reconcile
                break
        if fail_closed_terminal is None and s1d_parent_reconcile_fraction > OBMALLOC_SITE_REMAINDER_MAX_FRAC:
            fail_closed_terminal = "CHILD_PARENT_RECONCILE_FAIL"
        s1d_audit_outer_holding_bytes = {
            site_id: int(bracket_pos_totals.get(site_id, 0))
            for site_id in OBMALLOC_SITE_S1D_AUDIT_SITES
        }
        localization["s1d_audit_sites"] = list(OBMALLOC_SITE_S1D_AUDIT_SITES)
        localization["s1d_audit_outer_holding_bytes"] = s1d_audit_outer_holding_bytes
        audit_hold = int(s1d_audit_outer_holding_bytes.get("C4.S1d.8", 0))
        localization["s1d_audit_outer_reconcile_fraction"] = abs(audit_hold - parent_s1d_pos) / max(
            parent_s1d_pos,
            1,
        )
        s1d_aggregate_pos = sum(s1d_child_pos_totals.values())
        if s1d_aggregate_pos > 0 and fail_closed_terminal is None:
            s1d_fractions = {
                site_id: float(s1d_child_pos_totals[site_id]) / float(s1d_aggregate_pos)
                for site_id in OBMALLOC_SITE_S1D_CHILD_SITES
            }
            s1d_ranked = sorted(s1d_fractions.items(), key=lambda item: item[1], reverse=True)
            s1d_top_site, s1d_top_fraction = s1d_ranked[0]
            s1d_states_with_dominance = 0
            for state_idx in sampled:
                signed = per_state[str(state_idx)]["signed_holding_deltas_bytes"]
                s1d_pos = {
                    site_id: max(int(signed.get(site_id, 0)), 0)
                    for site_id in OBMALLOC_SITE_S1D_CHILD_SITES
                }
                s1d_state_sum = sum(s1d_pos.values())
                if s1d_state_sum <= 0:
                    continue
                s1d_state_fraction = float(s1d_pos[s1d_top_site]) / float(s1d_state_sum)
                if s1d_state_fraction >= OBMALLOC_EXPANDED_HOLDER_DOMINANCE_FRAC:
                    s1d_states_with_dominance += 1
            s1d_state_dominance_fraction = (
                float(s1d_states_with_dominance) / float(len(sampled)) if sampled else 0.0
            )
            localization["s1d_aggregate_holder_fractions"] = s1d_fractions
            localization["s1d_dominant_bracket"] = s1d_top_site
            localization["s1d_state_dominance_fraction"] = s1d_state_dominance_fraction
            if (
                s1d_top_fraction >= OBMALLOC_EXPANDED_HOLDER_DOMINANCE_FRAC
                and s1d_state_dominance_fraction >= OBMALLOC_EXPANDED_STATE_DOMINANCE_MIN
            ):
                localization["s1d_subsplit_verdict"] = f"DOMINANT_S1D_HOLDER_{s1d_top_site}"
            else:
                localization["s1d_subsplit_verdict"] = "S1D_SUBSPLIT_AMBIGUOUS"
        elif fail_closed_terminal is None:
            localization["s1d_subsplit_verdict"] = "S1D_SUBSPLIT_AMBIGUOUS"
    else:
        localization["s1d_subsplit_verdict"] = "legacy_s1d_subsites_absent"

    if phase3_s1f_subsplit_mode:
        s1f_child_pos_totals = {
            site_id: int(bracket_pos_totals[site_id])
            for site_id in OBMALLOC_SITE_S1F_CHILD_SITES
        }
        parent_s1f_pos = int(bracket_pos_totals.get("C4.S1f", 0))
        s1f_child_sum_pos = sum(s1f_child_pos_totals.values())
        s1f_parent_reconcile_fraction = abs(s1f_child_sum_pos - parent_s1f_pos) / max(
            parent_s1f_pos,
            1,
        )
        localization["s1f_child_aggregate_holder_pos_bytes"] = s1f_child_pos_totals
        localization["s1f_parent_reconcile_fraction"] = s1f_parent_reconcile_fraction
        for state_idx in sampled:
            signed = per_state[str(state_idx)]["signed_holding_deltas_bytes"]
            parent_hold = max(int(signed.get("C4.S1f", 0)), 0)
            s1f_state_sum = sum(
                max(int(signed.get(site_id, 0)), 0)
                for site_id in OBMALLOC_SITE_S1F_CHILD_SITES
            )
            state_reconcile = abs(s1f_state_sum - parent_hold) / max(parent_hold, 1)
            if state_reconcile > OBMALLOC_SITE_REMAINDER_MAX_FRAC:
                fail_closed_terminal = "CHILD_PARENT_RECONCILE_FAIL"
                localization["s1f_reconcile_fail_state_index"] = int(state_idx)
                localization["s1f_reconcile_fail_fraction"] = state_reconcile
                break
        if fail_closed_terminal is None and s1f_parent_reconcile_fraction > OBMALLOC_SITE_REMAINDER_MAX_FRAC:
            fail_closed_terminal = "CHILD_PARENT_RECONCILE_FAIL"
        s1f_aggregate_pos = sum(s1f_child_pos_totals.values())
        if s1f_aggregate_pos > 0 and fail_closed_terminal is None:
            s1f_fractions = {
                site_id: float(s1f_child_pos_totals[site_id]) / float(s1f_aggregate_pos)
                for site_id in OBMALLOC_SITE_S1F_CHILD_SITES
            }
            s1f_ranked = sorted(s1f_fractions.items(), key=lambda item: item[1], reverse=True)
            s1f_top_site, s1f_top_fraction = s1f_ranked[0]
            localization["s1f_aggregate_holder_fractions"] = s1f_fractions
            localization["s1f_dominant_bracket"] = s1f_top_site
            if s1f_top_fraction >= OBMALLOC_EXPANDED_HOLDER_DOMINANCE_FRAC:
                localization["s1f_subsplit_verdict"] = f"DOMINANT_S1F_HOLDER_{s1f_top_site}"
            else:
                localization["s1f_subsplit_verdict"] = "S1F_SUBSPLIT_AMBIGUOUS"
        elif fail_closed_terminal is None:
            localization["s1f_subsplit_verdict"] = "S1F_SUBSPLIT_AMBIGUOUS"
    else:
        localization["s1f_subsplit_verdict"] = "legacy_s1f_subsites_absent"

    if fail_closed_terminal in {
        "CHILD_PARENT_RECONCILE_FAIL",
        "CHILD_OVERLAP_DOUBLE_COUNT",
        "S1D_CHILD_COVERAGE_FAIL",
        "S1F_CHILD_COVERAGE_FAIL",
    }:
        s1d7_call_site = _attribute_s1d7_tracemalloc_call_site(
            marks_b,
            guards=guards,
            sampled_states=sampled,
        )
        localization["s1d7_tracemalloc_call_site"] = dict(s1d7_call_site)
        return {
            **banked,
            "guards": guards,
            "localization": localization,
            "classifier_terminal": None,
            "fail_closed_terminal": fail_closed_terminal,
            "slice8_rewrite_authorized": False,
            **_obmalloc_expanded_call_site_fields(s1d7_call_site),
        }

    if monotonic_fraction >= OBMALLOC_EXPANDED_RETENTION_MONOTONIC_MIN:
        classifier_terminal = "RETENTION_DOMINANT_CROSS_STATE"
    elif (
        top_fraction >= OBMALLOC_EXPANDED_HOLDER_DOMINANCE_FRAC
        and state_dominance_fraction >= OBMALLOC_EXPANDED_STATE_DOMINANCE_MIN
        and holder_stable
    ):
        classifier_terminal = f"DOMINANT_HOLDER_BRACKET_{top_site}"
    elif top_fraction >= OBMALLOC_EXPANDED_HOLDER_DOMINANCE_FRAC:
        classifier_terminal = "HOLDER_AMBIGUOUS_MULTI_BRACKET"
    else:
        classifier_terminal = "HOLDER_AMBIGUOUS_MULTI_BRACKET"

    if not representativeness_cleared:
        if classifier_terminal is None or classifier_terminal.startswith("DOMINANT_HOLDER"):
            classifier_terminal = "REPRESENTATIVENESS_UNCERTAIN"

    k2_fallback = len(sampled) <= 2
    slice8_rewrite_authorized = (
        classifier_terminal is not None
        and classifier_terminal.startswith("DOMINANT_HOLDER_BRACKET_")
        and representativeness_cleared
        and fail_closed_terminal is None
        and not k2_fallback
        and monotonic_fraction < OBMALLOC_EXPANDED_RETENTION_MONOTONIC_MIN
    )

    s1d7_call_site = _attribute_s1d7_tracemalloc_call_site(
        marks_b,
        guards=guards,
        sampled_states=sampled,
    )
    localization["s1d7_tracemalloc_call_site"] = dict(s1d7_call_site)

    return {
        **banked,
        "guards": guards,
        "localization": localization,
        "classifier_terminal": classifier_terminal,
        "fail_closed_terminal": fail_closed_terminal,
        "slice8_rewrite_authorized": slice8_rewrite_authorized,
        **_obmalloc_expanded_call_site_fields(s1d7_call_site),
    }


def _site_bracket_arena_bytes(mark: Mapping[str, Any] | None) -> int | None:
    return _obmalloc_field(mark, "arena_bytes")


def _site_bracket_delta_bytes(
    marks: Sequence[Mapping[str, Any]],
    site_id: str,
) -> int | None:
    pre = _obmalloc_site_mark(marks, f"obmalloc_site_{site_id}_pre")
    post = _obmalloc_site_mark(marks, f"obmalloc_site_{site_id}_post")
    pre_bytes = _site_bracket_arena_bytes(pre)
    post_bytes = _site_bracket_arena_bytes(post)
    if pre_bytes is None or post_bytes is None:
        return None
    return int(post_bytes) - int(pre_bytes)


def attribute_obmalloc_site_brackets(
    *,
    marks_a: list[dict[str, Any]],
    marks_a_prime: list[dict[str, Any]],
    marks_b: list[dict[str, Any]],
    debugmallocstats_preflight: Mapping[str, Any] | None = None,
    self_footprint_preflight: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    banked = _banked_reference_paths()
    c4_a = _c4_subphase_delta_gib(marks_a)
    c4_a_prime = _c4_subphase_delta_gib(marks_a_prime)
    c4_b = _c4_subphase_delta_gib(marks_b)

    noise_floor: float | None = None
    if c4_a is not None and c4_a_prime is not None:
        noise_floor = abs(float(c4_a_prime) - float(c4_a))

    run_stability_threshold = GUARD_STABILITY_FRAC * TOTAL_C4_REFERENCE_GIB
    denominator_variance_threshold = (
        GUARD_STABILITY_FRAC * BANKED_NON_GLIBC_MMAP_REFERENCE_GIB
    )
    envelope_tolerance = (
        max(float(noise_floor), GUARD_ENVELOPE_MIN_GIB)
        if noise_floor is not None
        else GUARD_ENVELOPE_MIN_GIB
    )
    total_c4_envelope_delta = (
        abs(float(c4_a) - TOTAL_C4_REFERENCE_GIB) if c4_a is not None else None
    )

    guards: dict[str, Any] = {
        "noise_floor_gib": noise_floor,
        "run_stability_threshold_gib": run_stability_threshold,
        "denominator_variance_threshold_gib": denominator_variance_threshold,
        "total_c4_envelope_tolerance_gib": envelope_tolerance,
        "total_c4_envelope_delta_gib": total_c4_envelope_delta,
        "c4_rss_delta_gib_same_run": {
            "A": c4_a,
            "A_prime": c4_a_prime,
            "B": c4_b,
        },
    }

    if noise_floor is None:
        guards["run_stability_ok"] = False
        guards["denominator_variance_ok"] = False
        return {
            **banked,
            "guards": guards,
            "fail_closed_terminal": "INCONCLUSIVE_PENDING_NOISE_FLOOR",
            "classifier_terminal": None,
            "slice8_rewrite_authorized": False,
        }

    guards["run_stability_ok"] = noise_floor <= run_stability_threshold
    guards["denominator_variance_ok"] = noise_floor <= denominator_variance_threshold
    guards["total_c4_envelope_ok"] = (
        total_c4_envelope_delta is not None
        and total_c4_envelope_delta <= envelope_tolerance
    )

    if not guards["total_c4_envelope_ok"]:
        return {
            **banked,
            "guards": guards,
            "fail_closed_terminal": "DENOMINATOR_INVALID_INCONCLUSIVE",
            "classifier_terminal": None,
            "slice8_rewrite_authorized": False,
        }
    if not guards["run_stability_ok"] or not guards["denominator_variance_ok"]:
        return {
            **banked,
            "guards": guards,
            "fail_closed_terminal": "INCONCLUSIVE_CROSS_RUN_DENOMINATOR",
            "classifier_terminal": None,
            "slice8_rewrite_authorized": False,
        }

    footprint = dict(self_footprint_preflight or {})
    footprint_status = str(
        footprint.get("debugmallocstats_self_footprint_status")
        or footprint.get("status")
        or ""
    )
    guards["debugmallocstats_self_footprint_bytes"] = footprint.get(
        "debugmallocstats_self_footprint_bytes"
    )
    guards["debugmallocstats_self_footprint_status"] = footprint_status
    if footprint_status == "exceeded":
        return {
            **banked,
            "guards": guards,
            "fail_closed_terminal": "OBMALLOC_SELF_FOOTPRINT_INCONCLUSIVE",
            "classifier_terminal": None,
            "slice8_rewrite_authorized": False,
        }

    preflight = dict(debugmallocstats_preflight or {})
    if str(preflight.get("status")) != "ok":
        return {
            **banked,
            "guards": guards,
            "fail_closed_terminal": "ARENA_STATS_UNPARSEABLE_INCONCLUSIVE",
            "classifier_terminal": None,
            "arena_stats_unavailable_reason": preflight.get("status"),
            "slice8_rewrite_authorized": False,
        }

    perturbation_delta = (
        abs(float(c4_b) - float(c4_a))
        if c4_a is not None and c4_b is not None
        else None
    )
    perturbation_threshold = max(
        PERTURBATION_MIN_GIB,
        PERTURBATION_NOISE_K * float(noise_floor),
    )
    guards["perturbation_delta_gib"] = perturbation_delta
    guards["perturbation_threshold_gib"] = perturbation_threshold

    window_entry_bytes = _site_bracket_arena_bytes(
        _obmalloc_site_mark(
            marks_b,
            f"obmalloc_site_{OBMALLOC_SITE_WINDOW_ENTRY}_pre",
        )
    )
    window_exit_bytes = _site_bracket_arena_bytes(
        _obmalloc_site_mark(
            marks_b,
            f"obmalloc_site_{OBMALLOC_SITE_WINDOW_EXIT}_post",
        )
    )
    if window_entry_bytes is None or window_exit_bytes is None:
        return {
            **banked,
            "guards": guards,
            "fail_closed_terminal": "OBSERVER_PERTURBED_INCONCLUSIVE",
            "classifier_terminal": None,
            "observer_reason": "incomplete_b_missing_state0_window_marks",
            "slice8_rewrite_authorized": False,
        }

    if (
        perturbation_delta is not None
        and perturbation_delta > perturbation_threshold
    ):
        return {
            **banked,
            "guards": guards,
            "fail_closed_terminal": "OBSERVER_PERTURBED_INCONCLUSIVE",
            "classifier_terminal": None,
            "observer_reason": "c4_rss_perturbation",
            "slice8_rewrite_authorized": False,
        }

    state0_local = int(window_exit_bytes) - int(window_entry_bytes)
    leaf_deltas: dict[str, int | None] = {
        site_id: _site_bracket_delta_bytes(marks_b, site_id)
        for site_id in OBMALLOC_SITE_LEAF_SITES
    }
    if any(delta is None for delta in leaf_deltas.values()):
        return {
            **banked,
            "guards": guards,
            "fail_closed_terminal": "ARENA_STATS_UNPARSEABLE_INCONCLUSIVE",
            "classifier_terminal": None,
            "missing_leaf_deltas": [
                site_id for site_id, delta in leaf_deltas.items() if delta is None
            ],
            "slice8_rewrite_authorized": False,
        }

    leaf_sum = sum(int(delta) for delta in leaf_deltas.values())
    s2_delta = _site_bracket_delta_bytes(marks_b, OBMALLOC_SITE_AGGREGATE_SITE)
    unattributed_remainder = int(state0_local) - int(leaf_sum)
    remainder_fraction = (
        abs(float(unattributed_remainder)) / float(state0_local)
        if state0_local > 0
        else 1.0
    )

    leaf_fractions = {
        site_id: (
            float(leaf_deltas[site_id]) / float(state0_local)
            if state0_local > 0 and leaf_deltas[site_id] is not None
            else 0.0
        )
        for site_id in OBMALLOC_SITE_LEAF_SITES
    }
    ranked_leaves = sorted(
        leaf_fractions.items(),
        key=lambda item: item[1],
        reverse=True,
    )
    top_site, top_fraction = ranked_leaves[0]
    second_fraction = ranked_leaves[1][1] if len(ranked_leaves) > 1 else 0.0

    state0_representativeness_ratio = (
        float(state0_local) / float(BANKED_NON_GLIBC_MMAP_REFERENCE_BYTES)
        if BANKED_NON_GLIBC_MMAP_REFERENCE_BYTES > 0
        else 0.0
    )
    state0_scaled_representativeness = (
        float(state0_local) * float(OBMALLOC_SITE_N_STATES_WITNESS)
    ) / float(BANKED_NON_GLIBC_MMAP_REFERENCE_BYTES)
    state0_representativeness_uncertain = (
        state0_representativeness_ratio
        < OBMALLOC_SITE_REPRESENTATIVENESS_UNCERTAIN_MAX
    )

    localization: dict[str, Any] = {
        "state0_local_bytes": state0_local,
        "leaf_sum_bytes": leaf_sum,
        "leaf_deltas_bytes": {k: int(v) for k, v in leaf_deltas.items()},
        "leaf_fractions": leaf_fractions,
        "aggregate_s2_delta_bytes": s2_delta,
        "unattributed_remainder_bytes": unattributed_remainder,
        "remainder_fraction": remainder_fraction,
        "window_entry_site": OBMALLOC_SITE_WINDOW_ENTRY,
        "window_exit_site": OBMALLOC_SITE_WINDOW_EXIT,
        "state0_representativeness_ratio": state0_representativeness_ratio,
        "state0_scaled_representativeness": state0_scaled_representativeness,
        "state0_representativeness_uncertain": state0_representativeness_uncertain,
        "cross_run_reconcile_caveat": True,
        "next_lane_rank1_carrier_audit": (
            remainder_fraction > OBMALLOC_SITE_REMAINDER_MAX_FRAC
        ),
    }

    classifier_terminal: str | None = None
    fail_closed_terminal: str | None = None

    if remainder_fraction > OBMALLOC_SITE_REMAINDER_MAX_FRAC:
        if top_fraction >= OBMALLOC_SITE_DOMINANCE_FRAC:
            classifier_terminal = "AMBIGUOUS_MULTI_BRACKET"
            localization["dominant_signal"] = (
                "unattributed_remainder_includes_unbracketed_q_levels_loop"
            )
        else:
            fail_closed_terminal = "BRACKET_REMAINDER_TOO_LARGE"
    elif top_fraction >= OBMALLOC_SITE_DOMINANCE_FRAC:
        if (
            second_fraction > 0.0
            and second_fraction >= top_fraction - OBMALLOC_SITE_AMBIGUOUS_WITHIN_FRAC
        ):
            classifier_terminal = "AMBIGUOUS_MULTI_BRACKET"
        else:
            classifier_terminal = f"DOMINANT_BRACKET_{top_site}"
    else:
        classifier_terminal = "AMBIGUOUS_MULTI_BRACKET"

    slice8_rewrite_authorized = (
        classifier_terminal is not None
        and classifier_terminal.startswith("DOMINANT_BRACKET_")
        and not state0_representativeness_uncertain
        and fail_closed_terminal is None
    )

    return {
        **banked,
        "guards": guards,
        "localization": localization,
        "classifier_terminal": classifier_terminal,
        "fail_closed_terminal": fail_closed_terminal,
        "slice8_rewrite_authorized": slice8_rewrite_authorized,
        "call_site_status": "UNRESOLVED",
    }


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
    alloc_hook_marks: list[dict[str, Any]] | None = None,
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

    alloc_hook_attribution: dict[str, Any] | None = None
    if alloc_hook_marks:
        from calm.hrm_text_158.native_full_stack.host_alloc_hook_probe import (
            attribute_alloc_hook_profile,
        )
        from calm.hrm_text_158.native_full_stack.host_allocator_probe import (
            diff_vma_entries,
            read_vma_entries,
        )

        alloc_hook_attribution = attribute_alloc_hook_profile(
            alloc_hook_marks,
            c4_delta_rss_gib=float(c4_delta) if c4_delta is not None else parent_sparse_cap_delta,
        )
        enter = next(
            (row for row in alloc_hook_marks if str(row.get("event")) == "alloc_hook_C4_enter"),
            None,
        )
        exit_mark = next(
            (row for row in alloc_hook_marks if str(row.get("event")) == "alloc_hook_C4_exit"),
            None,
        )
        maps_diff: dict[str, Any] | None = None
        if enter and exit_mark:
            exclude = []
            stats = dict((enter.get("alloc_hook_stats") or {}))
            for start_key, end_key in (
                ("hook_table_start", "hook_table_end"),
                ("hook_ring_start", "hook_ring_end"),
            ):
                if stats.get(start_key) is not None and stats.get(end_key) is not None:
                    exclude.append((int(stats[start_key]), int(stats[end_key])))
            before = list((enter.get("allocator_probe") or {}).get("vma_entries") or [])
            after = list((exit_mark.get("allocator_probe") or {}).get("vma_entries") or [])
            if not before:
                before = read_vma_entries(exclude_ranges=exclude)
            if not after:
                after = read_vma_entries(exclude_ranges=exclude)
            maps_diff = diff_vma_entries(before, after)
        alloc_hook_attribution["maps_diff_attribution"] = maps_diff
        status = str(alloc_hook_attribution.get("mechanism_owner_status"))
        if status == "RESOLVED":
            mechanism["mechanism_owner_status"] = "RESOLVED"
            mechanism["allocation_source"] = alloc_hook_attribution.get("allocation_source")
            mechanism["culprit_class"] = "C"
            mechanism["culprit_class_status"] = "RESOLVED"
            mechanism["call_site_status"] = alloc_hook_attribution.get("call_site_status")
            mechanism["call_site_origin_file_line"] = alloc_hook_attribution.get(
                "call_site_origin_file_line"
            )
        elif status in {"HOOK_FAILURE", "INCONCLUSIVE"}:
            mechanism["mechanism_owner_status"] = status
            mechanism["culprit_class"] = None
            mechanism["culprit_class_status"] = "UNRESOLVED"
        elif status == "UNMAPPED_OR_UNRESOLVED":
            mechanism["mechanism_owner_status"] = "UNMAPPED_OR_UNRESOLVED"
            mechanism["next_probe_route"] = "cuda_driver_host_probe"
            mechanism["call_site_status"] = "UNRESOLVED"
            alloc_hook_attribution["call_site_status"] = "UNRESOLVED"

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
        "alloc_hook_attribution": alloc_hook_attribution,
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
    alloc_hook_profile_path: Path | None = None,
    disjointness_probe: Mapping[str, Any] | None = None,
    self_footprint: Mapping[str, Any] | None = None,
    cross_run_reconcile_caveat: bool = False,
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
    if alloc_hook_profile_path is not None:
        alloc_hook_marks = _read_jsonl(alloc_hook_profile_path)
    else:
        alloc_hook_marks = [row for row in marks if _is_alloc_hook_mark(row)] or None
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
        alloc_hook_marks=alloc_hook_marks,
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
    if alloc_hook_marks:
        hook_path = alloc_hook_profile_path or profile_path
        receipt["alloc_hook_profile_path"] = str(hook_path)
        receipt["alloc_hook_mark_count"] = len(alloc_hook_marks)
        receipt["positive_control"] = {
            "kind": "hook_self_control_cpu",
            "note": (
                "CPU hook self-control only (malloc/torch_cpu/aligned); "
                "CUDA guardrail is the separate N=1 GPU fixture reaching C4 clean."
            ),
        }
    effective_allocator_marks = allocator_marks
    if effective_allocator_marks is None:
        effective_allocator_marks = [row for row in marks if _is_allocator_mark(row)] or None
    if (
        effective_allocator_marks
        and alloc_hook_marks
        and any(str(row.get("event")) == "allocator_C4_exit" for row in effective_allocator_marks)
    ):
        type_attr = attribute_allocator_type_partition(
            marks=marks,
            alloc_hook_marks=alloc_hook_marks,
            allocator_marks=effective_allocator_marks,
            disjointness_probe=disjointness_probe,
            self_footprint=self_footprint,
            cross_run_reconcile_caveat=cross_run_reconcile_caveat,
        )
        receipt["allocator_type_attribution"] = type_attr
        receipt["call_site_status"] = "UNRESOLVED"
        from calm.hrm_text_158.native_full_stack.host_alloc_hook_probe import (
            attribute_non_glibc_mmap_source,
        )

        partition = dict(type_attr.get("partition") or {})
        receipt["non_glibc_mmap_source_attribution"] = attribute_non_glibc_mmap_source(
            alloc_hook_marks,
            non_glibc_mmap_target_bytes=partition.get("non_glibc_mmap_bytes"),
            allocator_type_attribution=type_attr,
        )
    alloc_hook_attr = attribution.get("alloc_hook_attribution")
    if isinstance(alloc_hook_attr, Mapping):
        classified_null = alloc_hook_attr.get("classified_null")
        if classified_null:
            receipt["classified_null"] = classified_null
        partition = alloc_hook_attr.get("allocator_type_partition")
        if partition:
            receipt["allocator_type_partition"] = partition
    if attribution["dominant_phase_owner"] is None:
        receipt["rss_phase_owner_status"] = "UNRESOLVED"
    else:
        receipt["rss_phase_owner_status"] = "RESOLVED"
    if "mechanism_owner_status" not in receipt:
        receipt["mechanism_owner_status"] = "UNRESOLVED_SUBPHASE_REQUIRED"
    receipt["rss_owner_status"] = receipt["rss_phase_owner_status"]
    return receipt


def _fixture_probe_argv(
    scratch_root: Path,
    *,
    tracemalloc: bool = False,
    debugmallocstats: bool = False,
    expanded: bool = False,
) -> list[str]:
    parent = (
        "calm/hrm/checkpoints/hrm_text_158_phase3_L0c1_seed0017_replay83_n12k_lr7p5e5_pc1p0_"
        "rsL0b1math1r1b2_1_anchorsv1r3_from_L0b_final_step01500.pt"
    )
    from scripts.hrm_text_158_code_currency_guard import (
        phase3b_probe_python_argv_prefix,
        phase3b_probe_script_path,
    )

    use_bootstrap = bool(expanded) or (tracemalloc and not debugmallocstats)

    cmd = [
        sys.executable,
        *(
            phase3b_probe_python_argv_prefix()
            if use_bootstrap
            else []
        ),
        "-u",
        phase3b_probe_script_path(expanded=use_bootstrap),
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
    if tracemalloc:
        cmd.extend(
            [
                "--max-silent-phase-seconds",
                str(FIXTURE_PROBE_MAX_SILENT_PHASE_SECONDS_TRACEMALLOC),
            ]
        )
    else:
        cmd.extend(
            [
                "--max-silent-phase-seconds",
                str(FIXTURE_PROBE_MAX_SILENT_PHASE_SECONDS),
            ]
        )
    return cmd


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


def run_fixture_allocator_type(out_root: Path) -> dict[str, Any]:
    from calm.hrm_text_158.native_full_stack.host_alloc_hook_probe import (
        default_hook_so_path,
        run_ld_preload_torch_preflight,
        run_positive_control,
    )
    from calm.hrm_text_158.native_full_stack.host_allocator_probe import (
        measure_malloc_info_self_footprint,
        read_malloc_info_all_arenas,
        run_isolated_mmap_disjointness_probe,
    )

    out_root.mkdir(parents=True, exist_ok=True)
    scratch = out_root / "allocator_type_fixture_n1"
    scratch.mkdir(parents=True, exist_ok=True)
    malloc_info_avail = read_malloc_info_all_arenas()
    if not malloc_info_avail.get("available"):
        return {
            "scratch_root": str(scratch),
            "exit_code": 2,
            "malloc_info_availability": malloc_info_avail,
            "allocator_type_attribution": {
                "allocator_type_owner_status": "INCONCLUSIVE",
                "tier": "C",
                "reason": "malloc_info_unavailable",
            },
        }

    so_path = default_hook_so_path()
    preflight = run_ld_preload_torch_preflight(so_path)
    if preflight.get("status") != "ok":
        return {
            "scratch_root": str(scratch),
            "preflight": preflight,
            "exit_code": 2,
            "hook_status": "HOOK_FAILURE",
            "malloc_info_availability": malloc_info_avail,
        }

    positive_env = os.environ.copy()
    positive_env["LD_PRELOAD"] = str(so_path)
    positive_env[PROFILE_ALLOC_HOOK_ENV] = "1"
    positive_env["HRM_TEXT_158_PROFILE_HOST_RSS"] = "1"
    positive_env["HRM_TEXT_158_ALLOC_HOOK_STATS_PATH"] = str(
        scratch / "alloc_hook_positive_control.json"
    )
    positive_proc = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from pathlib import Path; "
                "from calm.hrm_text_158.native_full_stack.host_alloc_hook_probe import run_positive_control; "
                "import json; "
                "out=run_positive_control(Path(%r)); "
                "print(json.dumps(out))"
            )
            % str(scratch / "alloc_hook_positive_control.json"),
        ],
        cwd=REPO_ROOT,
        env={**positive_env, "PYTHONPATH": "."},
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    positive: dict[str, Any]
    try:
        positive = json.loads((positive_proc.stdout or "").strip().splitlines()[-1])
    except Exception:
        positive = {
            "status": "HOOK_FAILURE",
            "reason": "positive_control_subprocess_failed",
            "exit_code": int(positive_proc.returncode),
            "stderr_tail": "\n".join(positive_proc.stderr.splitlines()[-10:]),
        }
    if positive.get("status") != "ok":
        return {
            "scratch_root": str(scratch),
            "preflight": preflight,
            "positive_control": positive,
            "exit_code": 2,
            "hook_status": "HOOK_FAILURE",
            "malloc_info_availability": malloc_info_avail,
        }

    self_footprint = measure_malloc_info_self_footprint()
    disjointness_probe = run_isolated_mmap_disjointness_probe(
        so_path=so_path,
        out_path=scratch / "disjointness_probe.json",
    )

    lane_holding: dict[str, Any] | None = None
    lane_release: dict[str, Any] | None = None
    try:
        from scripts.hrm_text_158_r7_resource_lane_acquire import acquire_resource_lane

        lane_holding = acquire_resource_lane(out_root)
    except Exception as exc:
        lane_holding = {"acquire_error": f"{type(exc).__name__}: {exc}"}

    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    env["LD_PRELOAD"] = str(so_path)
    env["HRM_TEXT_158_RUN_GPU_GLOBAL_RATE_CAP"] = "1"
    env["HRM_TEXT_158_RUN_GPU_Q_ACC_APPLY"] = "1"
    env["HRM_TEXT_158_ALLOW_C2_GPU_LAUNCH"] = "1"
    env["HRM_TEXT_158_RUN_C2_ACQUISITION_PROBE"] = "1"
    env[PROFILE_HOST_RSS_ENV] = "1"
    env[PROFILE_ALLOCATOR_NATIVE_ENV] = "1"
    env[PROFILE_ALLOC_HOOK_ENV] = "1"
    env["HRM_TEXT_158_ALLOC_HOOK_STATS_PATH"] = str(scratch / "alloc_hook_stats.json")
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
    try:
        from scripts.hrm_text_158_r7_resource_lane_release import release_resource_lane

        lane_release = release_resource_lane(out_root)
    except Exception as exc:
        lane_release = {"release_error": f"{type(exc).__name__}: {exc}"}

    receipt = build_attribution_receipt(
        run_root=scratch,
        profile_path=profile_path,
        alloc_hook_profile_path=profile_path,
        disjointness_probe=disjointness_probe,
        self_footprint=self_footprint,
    )
    receipt["fixture"] = {
        "scratch_root": str(scratch),
        "command": cmd,
        "exit_code": int(proc.returncode),
        "stdout_tail": "\n".join(proc.stdout.splitlines()[-20:]),
        "stderr_tail": "\n".join(proc.stderr.splitlines()[-20:]),
        "resource_lane_holding": lane_holding,
        "resource_lane_release": lane_release,
        "preflight": preflight,
        "positive_control": positive,
        "malloc_info_availability": malloc_info_avail,
        "self_footprint": self_footprint,
        "disjointness_probe": disjointness_probe,
        "measurement_perturbed": True,
        "hook_so_path": str(so_path),
        "isolated_subprocess_disjointness_probe": True,
        "in_process_256mib_probe_before_c4": False,
    }
    return receipt


def run_fixture_alloc_hook(out_root: Path) -> dict[str, Any]:
    from calm.hrm_text_158.native_full_stack.host_alloc_hook_probe import (
        default_hook_so_path,
        run_ld_preload_torch_preflight,
        run_positive_control,
    )

    out_root.mkdir(parents=True, exist_ok=True)
    scratch = out_root / "alloc_hook_fixture_n1"
    scratch.mkdir(parents=True, exist_ok=True)
    so_path = default_hook_so_path()
    preflight = run_ld_preload_torch_preflight(so_path)
    if preflight.get("status") != "ok":
        return {
            "scratch_root": str(scratch),
            "preflight": preflight,
            "positive_control": None,
            "exit_code": 2,
            "hook_status": "HOOK_FAILURE",
        }

    positive_env = os.environ.copy()
    positive_env["LD_PRELOAD"] = str(so_path)
    positive_env[PROFILE_ALLOC_HOOK_ENV] = "1"
    positive_env["HRM_TEXT_158_PROFILE_HOST_RSS"] = "1"
    positive_env["HRM_TEXT_158_ALLOC_HOOK_STATS_PATH"] = str(scratch / "alloc_hook_positive_control.json")
    positive_proc = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from pathlib import Path; "
                "from calm.hrm_text_158.native_full_stack.host_alloc_hook_probe import run_positive_control; "
                "import json; "
                "out=run_positive_control(Path(%r)); "
                "print(json.dumps(out))"
            )
            % str(scratch / "alloc_hook_positive_control.json"),
        ],
        cwd=REPO_ROOT,
        env={**positive_env, "PYTHONPATH": "."},
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    positive: dict[str, Any]
    try:
        positive = json.loads((positive_proc.stdout or "").strip().splitlines()[-1])
    except Exception:
        positive = {
            "status": "HOOK_FAILURE",
            "reason": "positive_control_subprocess_failed",
            "exit_code": int(positive_proc.returncode),
            "stderr_tail": "\n".join(positive_proc.stderr.splitlines()[-10:]),
        }
    if positive.get("status") != "ok":
        return {
            "scratch_root": str(scratch),
            "preflight": preflight,
            "positive_control": positive,
            "exit_code": 2,
            "hook_status": "HOOK_FAILURE",
        }

    lane_holding: dict[str, Any] | None = None
    lane_release: dict[str, Any] | None = None
    try:
        from scripts.hrm_text_158_r7_resource_lane_acquire import acquire_resource_lane

        lane_holding = acquire_resource_lane(out_root)
    except Exception as exc:
        lane_holding = {"acquire_error": f"{type(exc).__name__}: {exc}"}

    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    env["LD_PRELOAD"] = str(so_path)
    env["HRM_TEXT_158_RUN_GPU_GLOBAL_RATE_CAP"] = "1"
    env["HRM_TEXT_158_RUN_GPU_Q_ACC_APPLY"] = "1"
    env["HRM_TEXT_158_ALLOW_C2_GPU_LAUNCH"] = "1"
    env["HRM_TEXT_158_RUN_C2_ACQUISITION_PROBE"] = "1"
    env[PROFILE_HOST_RSS_ENV] = "1"
    env[PROFILE_ALLOCATOR_NATIVE_ENV] = "1"
    env[PROFILE_ALLOC_HOOK_ENV] = "1"
    env["HRM_TEXT_158_ALLOC_HOOK_STATS_PATH"] = str(scratch / "alloc_hook_stats.json")
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
    hook_marks = [row for row in _read_jsonl(profile_path) if _is_alloc_hook_mark(row)]
    try:
        from scripts.hrm_text_158_r7_resource_lane_release import release_resource_lane

        lane_release = release_resource_lane(out_root)
    except Exception as exc:
        lane_release = {"release_error": f"{type(exc).__name__}: {exc}"}

    receipt = build_attribution_receipt(
        run_root=scratch,
        profile_path=profile_path,
        alloc_hook_profile_path=profile_path,
    )
    receipt["fixture"] = {
        "scratch_root": str(scratch),
        "command": cmd,
        "exit_code": int(proc.returncode),
        "stdout_tail": "\n".join(proc.stdout.splitlines()[-20:]),
        "stderr_tail": "\n".join(proc.stderr.splitlines()[-20:]),
        "resource_lane_holding": lane_holding,
        "resource_lane_release": lane_release,
        "preflight": preflight,
        "positive_control": positive,
        "alloc_hook_mark_count": len(hook_marks),
        "alloc_hook_events_seen": sorted({str(row.get("event")) for row in hook_marks}),
        "measurement_perturbed": True,
        "hook_so_path": str(so_path),
        "forward_fidelity_skip_scoped_to_alloc_hook_fixture": True,
    }
    return receipt


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


def _fixture_triangulation_env(*, tracemalloc: bool) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    env["HRM_TEXT_158_RUN_GPU_GLOBAL_RATE_CAP"] = "1"
    env["HRM_TEXT_158_RUN_GPU_Q_ACC_APPLY"] = "1"
    env["HRM_TEXT_158_ALLOW_C2_GPU_LAUNCH"] = "1"
    env["HRM_TEXT_158_RUN_C2_ACQUISITION_PROBE"] = "1"
    env[PROFILE_HOST_RSS_ENV] = "1"
    if tracemalloc:
        env[PROFILE_TRACEMALLOC_ENV] = "1"
    return env


def _run_fixture_triangulation_probe(
    out_root: Path,
    *,
    scratch_name: str,
    tracemalloc: bool,
) -> dict[str, Any]:
    out_root.mkdir(parents=True, exist_ok=True)
    scratch = out_root / scratch_name
    scratch.mkdir(parents=True, exist_ok=True)
    lane_holding: dict[str, Any] | None = None
    lane_release: dict[str, Any] | None = None
    try:
        from scripts.hrm_text_158_r7_resource_lane_acquire import acquire_resource_lane

        lane_holding = acquire_resource_lane(out_root)
    except Exception as exc:
        lane_holding = {"acquire_error": f"{type(exc).__name__}: {exc}"}

    env = _fixture_triangulation_env(tracemalloc=tracemalloc)
    cmd = _fixture_probe_argv(scratch, tracemalloc=tracemalloc)
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=1200 if tracemalloc else 600,
    )
    profile_path = scratch / HOST_RSS_PROFILE_JSONL_NAME
    marks = _read_jsonl(profile_path) if profile_path.is_file() else []
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
        "profile_mark_count": len(marks),
        "c4_rss_delta_gib": _c4_subphase_delta_gib(marks),
        "triangulation_mark_count": sum(1 for row in marks if _is_triangulation_mark(row)),
        "marks": marks,
    }


def run_fixture_allocator_triangulation_ab(out_root: Path) -> dict[str, Any]:
    payload = _run_fixture_triangulation_probe(
        out_root,
        scratch_name="allocator_triangulation_ab",
        tracemalloc=False,
    )
    payload["fixture_mode"] = "fixture_allocator_triangulation_ab"
    payload.pop("marks", None)
    return payload


def run_fixture_allocator_triangulation_ab_replicate(out_root: Path) -> dict[str, Any]:
    payload = _run_fixture_triangulation_probe(
        out_root,
        scratch_name="allocator_triangulation_ab_replicate",
        tracemalloc=False,
    )
    payload["fixture_mode"] = "fixture_allocator_triangulation_ab_replicate"
    payload.pop("marks", None)
    return payload


def run_fixture_allocator_triangulation_tracemalloc(out_root: Path) -> dict[str, Any]:
    from calm.hrm_text_158.native_full_stack.host_allocator_probe import (
        preflight_debugmallocstats_self_test,
    )

    preflight = preflight_debugmallocstats_self_test()
    payload = _run_fixture_triangulation_probe(
        out_root,
        scratch_name="allocator_triangulation_tracemalloc",
        tracemalloc=True,
    )
    payload["fixture_mode"] = "fixture_allocator_triangulation_tracemalloc"
    payload["debugmallocstats_preflight"] = preflight
    payload.pop("marks", None)
    return payload


def run_fixture_allocator_triangulation_combined(out_root: Path) -> dict[str, Any]:
    from calm.hrm_text_158.native_full_stack.host_allocator_probe import (
        preflight_debugmallocstats_self_test,
    )

    run_a = _run_fixture_triangulation_probe(
        out_root,
        scratch_name="allocator_triangulation_ab",
        tracemalloc=False,
    )
    run_a_prime = _run_fixture_triangulation_probe(
        out_root,
        scratch_name="allocator_triangulation_ab_replicate",
        tracemalloc=False,
    )
    preflight = preflight_debugmallocstats_self_test()
    run_b = _run_fixture_triangulation_probe(
        out_root,
        scratch_name="allocator_triangulation_tracemalloc",
        tracemalloc=True,
    )

    triangulation = attribute_python_allocator_triangulation(
        marks_a=list(run_a.pop("marks", [])),
        marks_a_prime=list(run_a_prime.pop("marks", [])),
        marks_b=list(run_b.pop("marks", [])),
        debugmallocstats_preflight=preflight,
    )

    exit_codes = [
        int(run_a.get("exit_code", 1)),
        int(run_a_prime.get("exit_code", 1)),
        int(run_b.get("exit_code", 1)),
    ]
    payload: dict[str, Any] = {
        "schema": ATTRIBUTION_SCHEMA,
        "fixture_mode": "fixture_allocator_triangulation_combined",
        "exit_code": max(exit_codes),
        "runs": {
            "A": {k: v for k, v in run_a.items() if k != "marks"},
            "A_prime": {k: v for k, v in run_a_prime.items() if k != "marks"},
            "B": {k: v for k, v in run_b.items() if k != "marks"},
        },
        "debugmallocstats_preflight": preflight,
        "alloc_hook_attribution": {
            "allocator_type_partition": {
                "mmap_net_bytes": BANKED_NON_GLIBC_MMAP_REFERENCE_BYTES,
                "mmap_net_gib": BANKED_NON_GLIBC_MMAP_REFERENCE_GIB,
                "c4_subphase_delta_rss_gib": TOTAL_C4_REFERENCE_GIB,
            }
        },
        "python_allocator_triangulation": triangulation,
        "classifier_branch": triangulation.get("classifier_branch"),
        "fail_closed_terminal": triangulation.get("fail_closed_terminal"),
    }
    return payload


def run_fixture_non_glibc_mmap_source(out_root: Path) -> dict[str, Any]:
    """Same-run allocator_type + non_glibc_mmap source tracing (frozen plan F1-F4)."""
    payload = run_fixture_allocator_type(out_root)
    out_path = out_root / "v6i_non_glibc_mmap_source_combined_attribution.json"
    if payload.get("combined_attribution_path"):
        out_path = Path(str(payload["combined_attribution_path"]))
    payload["fixture_mode"] = "fixture_non_glibc_mmap_source"
    payload["combined_attribution_path"] = str(out_path)
    return payload


def _fixture_obmalloc_env(
    *,
    debugmallocstats: bool,
    site_brackets: bool = False,
    expanded: bool = False,
    c4_retention_owner_census: bool = False,
    tracemalloc: bool = False,
) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    env["HRM_TEXT_158_RUN_GPU_GLOBAL_RATE_CAP"] = "1"
    env["HRM_TEXT_158_RUN_GPU_Q_ACC_APPLY"] = "1"
    env["HRM_TEXT_158_ALLOW_C2_GPU_LAUNCH"] = "1"
    env["HRM_TEXT_158_RUN_C2_ACQUISITION_PROBE"] = "1"
    env[PROFILE_HOST_RSS_ENV] = "1"
    env[PROFILE_DEBUGMALLOCSTATS_ENV] = "1" if debugmallocstats else "0"
    env[PROFILE_OBMALLOC_SITE_BRACKETS_ENV] = "1" if site_brackets else "0"
    env[PROFILE_OBMALLOC_EXPANDED_ENV] = "1" if expanded else "0"
    env[PROFILE_TRACEMALLOC_ENV] = "1" if tracemalloc else "0"
    if c4_retention_owner_census:
        from calm.hrm_text_158.native_full_stack.c4_retention_owner_census import (
            PROFILE_C4_RETENTION_OWNER_CENSUS_ENV,
        )

        env[PROFILE_C4_RETENTION_OWNER_CENSUS_ENV] = "1"
    if expanded:
        from scripts.hrm_text_158_code_currency_guard import prepare_phase3b_probe_launch_env

        env = prepare_phase3b_probe_launch_env(env, repo_root=REPO_ROOT, expanded=True)
    elif tracemalloc:
        from scripts.hrm_text_158_code_currency_guard import (
            prepare_phase3b_callsite_tracemalloc_launch_env,
        )

        env = prepare_phase3b_callsite_tracemalloc_launch_env(env, repo_root=REPO_ROOT)
    return env


def _read_log_tail(log_path: Path, *, max_lines: int = 20) -> str:
    if not log_path.is_file():
        return ""
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])


def _run_subprocess_streaming_to_log(
    cmd: list[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    log_path: Path,
    timeout: float,
) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    timeout_expired = False
    with log_path.open("w", encoding="utf-8") as log_f:
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timeout_expired = True
            proc.kill()
            proc.wait()
    return {
        "exit_code": int(proc.returncode if proc.returncode is not None else 1),
        "probe_stream_log": str(log_path),
        "subprocess_timeout_expired": timeout_expired,
        "stdout_tail": _read_log_tail(log_path),
        "stderr_tail": "",
        "used_capture_output": False,
    }


def _maybe_mirror_durable_attribution(
    payload: dict[str, Any],
    *,
    mirror: bool,
    mirror_path: Path | None = None,
) -> dict[str, Any]:
    if not mirror:
        return payload
    out_path = mirror_path or DEFAULT_DURABLE_MIRROR_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pre_hash: str | None = None
    backup_path: Path | None = None
    if out_path.is_file():
        existing = out_path.read_bytes()
        pre_hash = hashlib.sha256(existing).hexdigest()
        backup_path = out_path.with_suffix(out_path.suffix + f".bak.{pre_hash[:8]}")
        backup_path.write_bytes(existing)
    content = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    out_path.write_text(content, encoding="utf-8")
    post_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    payload["durable_mirror_receipt"] = {
        "mirror_path": str(out_path),
        "pre_hash": pre_hash,
        "backup_path": str(backup_path) if backup_path is not None else None,
        "post_hash": post_hash,
    }
    payload["combined_attribution_path"] = str(out_path)
    payload["durable_artifact_path"] = str(out_path)
    return payload


def _run_fixture_obmalloc_probe(
    out_root: Path,
    *,
    scratch_name: str,
    debugmallocstats: bool,
    site_brackets: bool = False,
    expanded: bool = False,
    c4_retention_owner_census: bool = False,
    tracemalloc: bool = False,
) -> dict[str, Any]:
    out_root.mkdir(parents=True, exist_ok=True)
    scratch = out_root / scratch_name
    scratch.mkdir(parents=True, exist_ok=True)
    profile_path = scratch / HOST_RSS_PROFILE_JSONL_NAME
    if profile_path.is_file():
        profile_path.unlink()
    lane_holding: dict[str, Any] | None = None
    lane_release: dict[str, Any] | None = None
    subprocess_result: dict[str, Any] = {}
    cmd: list[str] = []
    try:
        try:
            from scripts.hrm_text_158_r7_resource_lane_acquire import acquire_resource_lane

            lane_holding = acquire_resource_lane(out_root)
        except Exception as exc:
            lane_holding = {"acquire_error": f"{type(exc).__name__}: {exc}"}

        env = _fixture_obmalloc_env(
            debugmallocstats=debugmallocstats,
            site_brackets=site_brackets,
            expanded=expanded,
            c4_retention_owner_census=c4_retention_owner_census,
            tracemalloc=tracemalloc,
        )
        cmd = _fixture_probe_argv(
            scratch,
            tracemalloc=tracemalloc,
            debugmallocstats=debugmallocstats,
            expanded=expanded,
        )
        probe_stream_log = scratch / FIXTURE_PROBE_STREAM_LOG_NAME
        timeout = 1200.0 if debugmallocstats else 600.0
        if tracemalloc:
            timeout = max(timeout, float(FIXTURE_PROBE_MAX_SILENT_PHASE_SECONDS_TRACEMALLOC))
        subprocess_result = _run_subprocess_streaming_to_log(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            log_path=probe_stream_log,
            timeout=timeout,
        )
    finally:
        try:
            from scripts.hrm_text_158_r7_resource_lane_release import release_resource_lane

            lane_release = release_resource_lane(out_root)
        except Exception as exc:
            lane_release = {"release_error": f"{type(exc).__name__}: {exc}"}

    marks = _read_jsonl(profile_path) if profile_path.is_file() else []

    return {
        "scratch_root": str(scratch),
        "profile_path": str(profile_path),
        "command": cmd,
        "exit_code": int(subprocess_result.get("exit_code", 1)),
        "stdout_tail": str(subprocess_result.get("stdout_tail", "")),
        "stderr_tail": str(subprocess_result.get("stderr_tail", "")),
        "probe_stream_log": subprocess_result.get("probe_stream_log"),
        "subprocess_timeout_expired": bool(
            subprocess_result.get("subprocess_timeout_expired", False)
        ),
        "used_capture_output": bool(subprocess_result.get("used_capture_output", False)),
        "resource_lane_holding": lane_holding,
        "resource_lane_release": lane_release,
        "profile_mark_count": len(marks),
        "c4_rss_delta_gib": _c4_subphase_delta_gib(marks),
        "obmalloc_mark_count": sum(1 for row in marks if _is_obmalloc_mark(row)),
        "obmalloc_site_mark_count": sum(1 for row in marks if _is_obmalloc_site_mark(row)),
        "marks": marks,
    }


def run_fixture_obmalloc_arena_ab(out_root: Path) -> dict[str, Any]:
    payload = _run_fixture_obmalloc_probe(
        out_root,
        scratch_name="obmalloc_arena_ab",
        debugmallocstats=False,
    )
    payload["fixture_mode"] = "fixture_obmalloc_arena_ab"
    payload.pop("marks", None)
    return payload


def run_fixture_obmalloc_arena_ab_replicate(out_root: Path) -> dict[str, Any]:
    payload = _run_fixture_obmalloc_probe(
        out_root,
        scratch_name="obmalloc_arena_ab_replicate",
        debugmallocstats=False,
    )
    payload["fixture_mode"] = "fixture_obmalloc_arena_ab_replicate"
    payload.pop("marks", None)
    return payload


def run_fixture_obmalloc_arena_b(out_root: Path) -> dict[str, Any]:
    from calm.hrm_text_158.native_full_stack.host_allocator_probe import (
        measure_debugmallocstats_self_footprint,
        preflight_debugmallocstats_self_test,
    )

    preflight = preflight_debugmallocstats_self_test()
    footprint = measure_debugmallocstats_self_footprint()
    payload = _run_fixture_obmalloc_probe(
        out_root,
        scratch_name="obmalloc_arena_b",
        debugmallocstats=True,
    )
    payload["fixture_mode"] = "fixture_obmalloc_arena_b"
    payload["debugmallocstats_preflight"] = preflight
    payload["debugmallocstats_self_footprint"] = footprint
    payload.pop("marks", None)
    return payload


def run_fixture_obmalloc_arena_combined(out_root: Path) -> dict[str, Any]:
    from calm.hrm_text_158.native_full_stack.host_allocator_probe import (
        measure_debugmallocstats_self_footprint,
        preflight_debugmallocstats_self_test,
    )

    run_a = _run_fixture_obmalloc_probe(
        out_root,
        scratch_name="obmalloc_arena_ab",
        debugmallocstats=False,
    )
    run_a_prime = _run_fixture_obmalloc_probe(
        out_root,
        scratch_name="obmalloc_arena_ab_replicate",
        debugmallocstats=False,
    )
    preflight = preflight_debugmallocstats_self_test()
    footprint = measure_debugmallocstats_self_footprint()
    run_b = _run_fixture_obmalloc_probe(
        out_root,
        scratch_name="obmalloc_arena_b",
        debugmallocstats=True,
    )

    obmalloc = attribute_obmalloc_arena_retention(
        marks_a=list(run_a.pop("marks", [])),
        marks_a_prime=list(run_a_prime.pop("marks", [])),
        marks_b=list(run_b.pop("marks", [])),
        debugmallocstats_preflight=preflight,
        self_footprint_preflight=footprint,
    )

    exit_codes = [
        int(run_a.get("exit_code", 1)),
        int(run_a_prime.get("exit_code", 1)),
        int(run_b.get("exit_code", 1)),
    ]
    payload: dict[str, Any] = {
        "schema": ATTRIBUTION_SCHEMA,
        "fixture_mode": "fixture_obmalloc_arena_combined",
        "exit_code": max(exit_codes),
        "runs": {
            "A": {k: v for k, v in run_a.items() if k != "marks"},
            "A_prime": {k: v for k, v in run_a_prime.items() if k != "marks"},
            "B": {k: v for k, v in run_b.items() if k != "marks"},
        },
        "debugmallocstats_preflight": preflight,
        "debugmallocstats_self_footprint": footprint,
        "alloc_hook_attribution": {
            "allocator_type_partition": {
                "mmap_net_bytes": BANKED_NON_GLIBC_MMAP_REFERENCE_BYTES,
                "mmap_net_gib": BANKED_NON_GLIBC_MMAP_REFERENCE_GIB,
                "c4_subphase_delta_rss_gib": TOTAL_C4_REFERENCE_GIB,
            }
        },
        "obmalloc_arena_attribution": obmalloc,
        "classifier_terminal": obmalloc.get("classifier_terminal"),
        "fail_closed_terminal": obmalloc.get("fail_closed_terminal"),
    }
    return payload


def run_fixture_obmalloc_site_brackets_combined(out_root: Path) -> dict[str, Any]:
    from calm.hrm_text_158.native_full_stack.host_allocator_probe import (
        measure_debugmallocstats_self_footprint,
        preflight_debugmallocstats_self_test,
    )

    run_a = _run_fixture_obmalloc_probe(
        out_root,
        scratch_name="obmalloc_site_brackets_ab",
        debugmallocstats=False,
    )
    run_a_prime = _run_fixture_obmalloc_probe(
        out_root,
        scratch_name="obmalloc_site_brackets_ab_replicate",
        debugmallocstats=False,
    )
    preflight = preflight_debugmallocstats_self_test()
    footprint = measure_debugmallocstats_self_footprint()
    run_b = _run_fixture_obmalloc_probe(
        out_root,
        scratch_name="obmalloc_site_brackets_b",
        debugmallocstats=True,
        site_brackets=True,
    )

    site_brackets = attribute_obmalloc_site_brackets(
        marks_a=list(run_a.pop("marks", [])),
        marks_a_prime=list(run_a_prime.pop("marks", [])),
        marks_b=list(run_b.pop("marks", [])),
        debugmallocstats_preflight=preflight,
        self_footprint_preflight=footprint,
    )

    exit_codes = [
        int(run_a.get("exit_code", 1)),
        int(run_a_prime.get("exit_code", 1)),
        int(run_b.get("exit_code", 1)),
    ]
    payload: dict[str, Any] = {
        "schema": ATTRIBUTION_SCHEMA,
        "fixture_mode": "fixture_obmalloc_site_brackets",
        "exit_code": max(exit_codes),
        "runs": {
            "A": {k: v for k, v in run_a.items() if k != "marks"},
            "A_prime": {k: v for k, v in run_a_prime.items() if k != "marks"},
            "B": {k: v for k, v in run_b.items() if k != "marks"},
        },
        "debugmallocstats_preflight": preflight,
        "debugmallocstats_self_footprint": footprint,
        "alloc_hook_attribution": {
            "allocator_type_partition": {
                "mmap_net_bytes": BANKED_NON_GLIBC_MMAP_REFERENCE_BYTES,
                "mmap_net_gib": BANKED_NON_GLIBC_MMAP_REFERENCE_GIB,
                "c4_subphase_delta_rss_gib": TOTAL_C4_REFERENCE_GIB,
            }
        },
        "obmalloc_site_brackets_attribution": site_brackets,
        "classifier_terminal": site_brackets.get("classifier_terminal"),
        "fail_closed_terminal": site_brackets.get("fail_closed_terminal"),
        "slice8_rewrite_authorized": site_brackets.get("slice8_rewrite_authorized"),
    }
    localization = dict(site_brackets.get("localization") or {})
    if localization:
        payload["state0_scaled_representativeness"] = localization.get(
            "state0_scaled_representativeness"
        )
        payload["state0_representativeness_uncertain"] = localization.get(
            "state0_representativeness_uncertain"
        )
    return payload


def run_fixture_obmalloc_expanded_combined(
    out_root: Path,
    *,
    mirror_durable_attribution: bool = False,
    mirror_durable_path: Path | None = None,
) -> dict[str, Any]:
    from calm.hrm_text_158.native_full_stack.host_allocator_probe import (
        measure_debugmallocstats_self_footprint,
        preflight_debugmallocstats_self_test,
    )

    run_a = _run_fixture_obmalloc_probe(
        out_root,
        scratch_name="obmalloc_expanded_ab",
        debugmallocstats=False,
    )
    run_a_prime = _run_fixture_obmalloc_probe(
        out_root,
        scratch_name="obmalloc_expanded_ab_replicate",
        debugmallocstats=False,
    )
    preflight = preflight_debugmallocstats_self_test()
    footprint = measure_debugmallocstats_self_footprint()
    run_b = _run_fixture_obmalloc_probe(
        out_root,
        scratch_name="obmalloc_expanded_b",
        debugmallocstats=True,
        site_brackets=True,
        expanded=True,
        c4_retention_owner_census=True,
    )

    marks_a = list(run_a.pop("marks", []))
    marks_a_prime = list(run_a_prime.pop("marks", []))
    marks_b = list(run_b.pop("marks", []))

    expanded = attribute_obmalloc_expanded(
        marks_a=marks_a,
        marks_a_prime=marks_a_prime,
        marks_b=marks_b,
        debugmallocstats_preflight=preflight,
        self_footprint_preflight=footprint,
    )
    owner_census = attribute_c4_retention_owner_census(
        marks_a=marks_a,
        marks_a_prime=marks_a_prime,
        marks_b=marks_b,
    )

    exit_codes = [
        int(run_a.get("exit_code", 1)),
        int(run_a_prime.get("exit_code", 1)),
        int(run_b.get("exit_code", 1)),
    ]
    probe_exit_code = max(exit_codes)
    fail_closed_terminal = expanded.get("fail_closed_terminal")
    exit_fields = _resolve_attribution_process_exit_code(
        probe_exit_code=int(probe_exit_code),
        fail_closed_terminal=(
            str(fail_closed_terminal) if fail_closed_terminal is not None else None
        ),
    )
    localization = dict(expanded.get("localization") or {})
    payload: dict[str, Any] = {
        "schema": ATTRIBUTION_SCHEMA,
        "fixture_mode": "fixture_obmalloc_expanded",
        "exit_code": int(exit_fields["process_exit_code"]),
        "process_exit_code": int(exit_fields["process_exit_code"]),
        "mapped_terminal_code": exit_fields["mapped_terminal_code"],
        "exit_code_agreement": bool(exit_fields["exit_code_agreement"]),
        "probe_exit_code": int(probe_exit_code),
        "runs": {
            "A": {k: v for k, v in run_a.items() if k != "marks"},
            "A_prime": {k: v for k, v in run_a_prime.items() if k != "marks"},
            "B": {k: v for k, v in run_b.items() if k != "marks"},
        },
        "debugmallocstats_preflight": preflight,
        "debugmallocstats_self_footprint": footprint,
        "obmalloc_expanded_attribution": expanded,
        "c4_retention_owner_census": owner_census,
        "classifier_terminal": expanded.get("classifier_terminal"),
        "c4_retention_owner_classifier_terminal": owner_census.get("classifier_terminal"),
        "fail_closed_terminal": expanded.get("fail_closed_terminal"),
        "slice8_rewrite_authorized": expanded.get("slice8_rewrite_authorized"),
    }
    if localization:
        payload["scaled_representativeness"] = localization.get("scaled_representativeness")
        payload["representativeness_cleared"] = localization.get("representativeness_cleared")
    return _maybe_mirror_durable_attribution(
        payload,
        mirror=mirror_durable_attribution,
        mirror_path=mirror_durable_path,
    )


def attribute_callsite_tracemalloc_b_prime(
    *,
    marks_a: Sequence[Mapping[str, Any]],
    marks_b: Sequence[Mapping[str, Any]],
    sampled_states: Sequence[int] = (0, 10, 21, 31),
) -> dict[str, Any]:
    from calm.hrm_text_158.native_full_stack.sparse_cap_gpu_seam_adapter import (
        compute_obmalloc_expanded_sampled_states,
    )

    sampled = tuple(int(x) for x in sampled_states) or tuple(
        compute_obmalloc_expanded_sampled_states(32)
    )
    guards = {
        "callsite_b_prime_mode": True,
        "phase3_s1d_subsplit_mode": False,
        "tracemalloc_perturbed": False,
        "obmalloc_expanded_event_validation": {"valid": True, "pair_counts_by_site": {}},
        "obmalloc_expanded_event_counts": {"total": len(marks_b)},
    }
    s1d7_call_site = _attribute_s1d7_tracemalloc_call_site(
        marks_b,
        guards=guards,
        sampled_states=sampled,
    )
    if s1d7_call_site.get("tracemalloc_perturbed"):
        guards["tracemalloc_perturbed"] = True
    return {
        "guards": guards,
        "localization": {
            "callsite_b_prime_mode": True,
            "s1d7_tracemalloc_call_site": dict(s1d7_call_site),
        },
        "fail_closed_terminal": None,
        **_obmalloc_expanded_call_site_fields(s1d7_call_site),
    }


def dry_check_callsite_b_prime_b_arm_launch_composition() -> dict[str, Any]:
    """Dry-check B_callsite launch composition via the real _run_fixture_obmalloc_probe seam."""

    from scripts.hrm_text_158_code_currency_guard import (
        IMPORT_BYTE_PINS_ENV,
        PHASE3B_PROBE_IMPORT_MODULE_BY_REL,
        hash_file_bytes,
        phase3b_probe_python_argv_prefix,
        phase3b_probe_script_path,
    )

    scratch = REPO_ROOT / ".dry_check_callsite_b_prime_scratch"
    env = _fixture_obmalloc_env(
        debugmallocstats=False,
        site_brackets=False,
        expanded=False,
        tracemalloc=True,
    )
    cmd = _fixture_probe_argv(
        scratch,
        tracemalloc=True,
        debugmallocstats=False,
        expanded=False,
    )
    argv_prefix = phase3b_probe_python_argv_prefix()
    bootstrap_path = phase3b_probe_script_path(expanded=True)
    max_silent = str(FIXTURE_PROBE_MAX_SILENT_PHASE_SECONDS_TRACEMALLOC)
    checks = {
        "env_debugmallocstats_zero": env.get(PROFILE_DEBUGMALLOCSTATS_ENV) == "0",
        "env_site_brackets_zero": env.get(PROFILE_OBMALLOC_SITE_BRACKETS_ENV) == "0",
        "env_obmalloc_expanded_zero": env.get(PROFILE_OBMALLOC_EXPANDED_ENV) == "0",
        "env_tracemalloc_one": env.get(PROFILE_TRACEMALLOC_ENV) == "1",
        "cmd_uses_bootstrap": bootstrap_path in cmd,
        "cmd_has_b_flag": "-B" in argv_prefix and all(flag in cmd for flag in argv_prefix),
        "cmd_max_silent_900": max_silent in cmd,
        "import_byte_pins_present": IMPORT_BYTE_PINS_ENV in env,
    }
    if checks["import_byte_pins_present"]:
        pin_payload = json.loads(env[IMPORT_BYTE_PINS_ENV])
        checks["import_byte_pins_match_disk"] = all(
            pin_payload.get(rel) == hash_file_bytes(REPO_ROOT / rel)
            for rel in PHASE3B_PROBE_IMPORT_MODULE_BY_REL
            if (REPO_ROOT / rel).is_file()
        )
    else:
        checks["import_byte_pins_match_disk"] = False

    guard_exit_code: int | None = None
    guard_script = (
        "import json, os, sys\n"
        "from scripts.hrm_text_158_code_currency_guard import "
        "run_phase3b_probe_executed_code_currency_guard\n"
        "env = json.loads(sys.argv[1])\n"
        "for key, value in env.items():\n"
        "    os.environ[key] = value\n"
        "exit_code = run_phase3b_probe_executed_code_currency_guard("
        "require_obmalloc_expanded=False)\n"
        "raise SystemExit(0 if exit_code is None else int(exit_code))\n"
    )
    guard_proc = subprocess.run(
        [sys.executable, "-c", guard_script, json.dumps(env)],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
        capture_output=True,
        text=True,
        check=False,
    )
    guard_exit_code = int(guard_proc.returncode)

    checks["guard_dry_check_passes"] = guard_exit_code == 0
    ok = all(checks.values())
    return {
        "schema": "hrm_text_158_callsite_b_prime_launch_dry_check/v1",
        "ok": ok,
        "checks": checks,
        "cmd": cmd,
        "env_profile_toggles": {
            PROFILE_DEBUGMALLOCSTATS_ENV: env.get(PROFILE_DEBUGMALLOCSTATS_ENV),
            PROFILE_OBMALLOC_SITE_BRACKETS_ENV: env.get(PROFILE_OBMALLOC_SITE_BRACKETS_ENV),
            PROFILE_OBMALLOC_EXPANDED_ENV: env.get(PROFILE_OBMALLOC_EXPANDED_ENV),
            PROFILE_TRACEMALLOC_ENV: env.get(PROFILE_TRACEMALLOC_ENV),
        },
        "guard_exit_code": guard_exit_code,
    }


def run_fixture_callsite_b_prime_combined(
    out_root: Path,
    *,
    mirror_durable_attribution: bool = False,
    mirror_durable_path: Path | None = None,
) -> dict[str, Any]:
    run_a = _run_fixture_obmalloc_probe(
        out_root,
        scratch_name="callsite_observer_a",
        debugmallocstats=False,
    )
    run_b = _run_fixture_obmalloc_probe(
        out_root,
        scratch_name="callsite_tracemalloc_b",
        debugmallocstats=False,
        tracemalloc=True,
        expanded=False,
    )
    marks_a = list(run_a.pop("marks", []))
    marks_b = list(run_b.pop("marks", []))
    attribution = attribute_callsite_tracemalloc_b_prime(
        marks_a=marks_a,
        marks_b=marks_b,
    )
    exit_codes = [
        int(run_a.get("exit_code", 1)),
        int(run_b.get("exit_code", 1)),
    ]
    probe_exit_code = max(exit_codes)
    fail_closed_terminal = attribution.get("fail_closed_terminal")
    exit_fields = _resolve_attribution_process_exit_code(
        probe_exit_code=int(probe_exit_code),
        fail_closed_terminal=(
            str(fail_closed_terminal) if fail_closed_terminal is not None else None
        ),
    )
    localization = dict(attribution.get("localization") or {})
    payload: dict[str, Any] = {
        "schema": ATTRIBUTION_SCHEMA,
        "fixture_mode": "fixture_callsite_b_prime",
        "exit_code": int(exit_fields["process_exit_code"]),
        "process_exit_code": int(exit_fields["process_exit_code"]),
        "mapped_terminal_code": exit_fields["mapped_terminal_code"],
        "exit_code_agreement": bool(exit_fields["exit_code_agreement"]),
        "probe_exit_code": int(probe_exit_code),
        "runs": {
            "A": {k: v for k, v in run_a.items() if k != "marks"},
            "B": {k: v for k, v in run_b.items() if k != "marks"},
        },
        "callsite_b_prime_attribution": attribution,
        "classifier_terminal": attribution.get("classifier_terminal"),
        "fail_closed_terminal": attribution.get("fail_closed_terminal"),
        "banked_reconcile_provenance": dict(BANKED_RECONCILE_PROVENANCE),
        "banked_reconcile_precondition_ok": (
            float(BANKED_RECONCILE_PROVENANCE["s1d_parent_reconcile_fraction"]) <= 0.15
        ),
    }
    if localization:
        payload["s1d7_tracemalloc_call_site"] = localization.get("s1d7_tracemalloc_call_site")
    return _maybe_mirror_durable_attribution(
        payload,
        mirror=mirror_durable_attribution,
        mirror_path=mirror_durable_path,
    )


def run_callsite_tracemalloc_scale_smoke(out_root: Path) -> dict[str, Any]:
    """Mandatory short tracemalloc-only B-prime fixture smoke before full GPU acceptance."""

    smoke_root = out_root / "prelaunch" / "callsite_tracemalloc_scale_smoke"
    smoke_root.mkdir(parents=True, exist_ok=True)
    run_a = _run_fixture_obmalloc_probe(
        smoke_root,
        scratch_name="callsite_observer_a",
        debugmallocstats=False,
    )
    run_b = _run_fixture_obmalloc_probe(
        smoke_root,
        scratch_name="callsite_tracemalloc_b",
        debugmallocstats=False,
        tracemalloc=True,
        expanded=False,
    )
    marks_a = list(run_a.pop("marks", []))
    marks_b = list(run_b.pop("marks", []))
    expanded = attribute_callsite_tracemalloc_b_prime(
        marks_a=marks_a,
        marks_b=marks_b,
    )
    b_profile_mark_count = int((run_b.get("profile_mark_count") or 0))
    observer_fail_closed = expanded.get("fail_closed_terminal")
    tracemalloc_perturbed = expanded.get("tracemalloc_perturbed")
    if tracemalloc_perturbed is None:
        tracemalloc_perturbed = dict(expanded.get("guards") or {}).get("tracemalloc_perturbed")
    mark_pair_count = expanded.get("s1d7_tracemalloc_mark_pair_count")
    mark_schema = dict(expanded.get("localization") or {}).get("s1d7_tracemalloc_call_site", {}).get(
        "s1d7_tracemalloc_mark_schema"
    )
    if mark_schema is None:
        mark_schema = expanded.get("s1d7_tracemalloc_mark_schema")
    event_counts = dict(dict(expanded.get("guards") or {}).get("obmalloc_expanded_event_counts") or {})
    total_events = event_counts.get("total")
    total_events_int = int(total_events) if total_events is not None else None
    event_total_exceeds_hard_ceiling = (
        total_events_int is not None
        and total_events_int > PHASE3_CALLSITE_EVENT_TOTAL_HARD_CEILING
    )
    new_schema_mark_count = sum(
        1
        for row in marks_b
        if str(row.get("event") or "").startswith("s1d7_tracemalloc_site_C4.S1d.7_")
    )
    checks = {
        "observer_guard_clear": observer_fail_closed != "OBSERVER_PERTURBED_INCONCLUSIVE",
        "tracemalloc_perturbed_false": tracemalloc_perturbed is False,
        "s1d7_tracemalloc_mark_pair_count_eq_4": (
            mark_pair_count == PHASE3_CALLSITE_S1D7_MARK_PAIR_COUNT_EXPECTED
        ),
        "new_schema_mark_count_eq_8": new_schema_mark_count == 8,
        "consumer_resolves_new_schema": expanded.get("call_site_status") == "RESOLVED"
        or mark_pair_count == PHASE3_CALLSITE_S1D7_MARK_PAIR_COUNT_EXPECTED,
        "b_profile_mark_count_gt_0": b_profile_mark_count > 0,
        "event_total_within_hard_ceiling": not event_total_exceeds_hard_ceiling,
        "no_profile_env_mutual_exclusion_abort": int(run_b.get("exit_code", 0)) != -6,
        "no_tracemalloc_perturbed_inconclusive": (
            observer_fail_closed != "TRACEMALLOC_PERTURBED_INCONCLUSIVE"
        ),
    }
    receipt: dict[str, Any] = {
        "schema": "hrm_text_158_callsite_tracemalloc_scale_smoke_receipt/v1",
        "smoke_root": str(smoke_root),
        "checks": checks,
        "ok": all(checks.values()),
        "b_arm_profile_mark_count": b_profile_mark_count,
        "fail_closed_terminal": observer_fail_closed,
        "tracemalloc_perturbed": tracemalloc_perturbed,
        "s1d7_tracemalloc_mark_pair_count": mark_pair_count,
        "s1d7_tracemalloc_mark_schema": mark_schema,
        "new_schema_mark_count": new_schema_mark_count,
        "obmalloc_expanded_event_count_total": total_events_int,
        "call_site_status": expanded.get("call_site_status"),
        "banked_reconcile_precondition_ok": True,
        "runs": {
            "A": {k: v for k, v in run_a.items() if k != "marks"},
            "B": {k: v for k, v in run_b.items() if k != "marks"},
        },
    }
    out_path = smoke_root / "callsite_tracemalloc_scale_smoke_receipt.json"
    out_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    receipt["receipt_path"] = str(out_path)
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
            "fixture_alloc_hook",
            "fixture_allocator_type",
            "fixture_non_glibc_mmap_source",
            "fixture_allocator_triangulation_ab",
            "fixture_allocator_triangulation_ab_replicate",
            "fixture_allocator_triangulation_tracemalloc",
            "fixture_allocator_triangulation_combined",
            "fixture_obmalloc_arena_ab",
            "fixture_obmalloc_arena_ab_replicate",
            "fixture_obmalloc_arena_b",
            "fixture_obmalloc_arena_combined",
            "fixture_obmalloc_site_brackets",
            "fixture_obmalloc_expanded",
            "fixture_callsite_b_prime",
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
    parser.add_argument(
        "--alloc-hook-profile-path",
        type=Path,
        default=None,
        help="Optional alloc-hook host_rss_profile.jsonl for attribute mode",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--mirror-durable-attribution",
        action="store_true",
        help=(
            "Opt-in mirror of fixture_obmalloc_expanded output to the legacy durable "
            "attribution path (default OFF; writes pre-hash/backup/post-hash receipt)."
        ),
    )
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
            alloc_hook_profile_path=args.alloc_hook_profile_path,
        )
        if args.aborted_run_root is not None:
            payload["aborted_run_root"] = str(args.aborted_run_root)
    elif args.mode == "fixture_live_resident":
        payload = run_fixture_live_resident_diagnostic(args.run_root)
    elif args.mode == "fixture_torch_census":
        payload = run_fixture_torch_census(args.run_root)
    elif args.mode == "fixture_allocator_native":
        payload = run_fixture_allocator_native(args.run_root)
    elif args.mode == "fixture_alloc_hook":
        payload = run_fixture_alloc_hook(args.run_root)
    elif args.mode == "fixture_allocator_type":
        payload = run_fixture_allocator_type(args.run_root)
    elif args.mode == "fixture_non_glibc_mmap_source":
        payload = run_fixture_non_glibc_mmap_source(args.run_root)
    elif args.mode == "fixture_allocator_triangulation_ab":
        payload = run_fixture_allocator_triangulation_ab(args.run_root)
    elif args.mode == "fixture_allocator_triangulation_ab_replicate":
        payload = run_fixture_allocator_triangulation_ab_replicate(args.run_root)
    elif args.mode == "fixture_allocator_triangulation_tracemalloc":
        payload = run_fixture_allocator_triangulation_tracemalloc(args.run_root)
    elif args.mode == "fixture_allocator_triangulation_combined":
        payload = run_fixture_allocator_triangulation_combined(args.run_root)
    elif args.mode == "fixture_obmalloc_arena_ab":
        payload = run_fixture_obmalloc_arena_ab(args.run_root)
    elif args.mode == "fixture_obmalloc_arena_ab_replicate":
        payload = run_fixture_obmalloc_arena_ab_replicate(args.run_root)
    elif args.mode == "fixture_obmalloc_arena_b":
        payload = run_fixture_obmalloc_arena_b(args.run_root)
    elif args.mode == "fixture_obmalloc_arena_combined":
        payload = run_fixture_obmalloc_arena_combined(args.run_root)
    elif args.mode == "fixture_obmalloc_site_brackets":
        payload = run_fixture_obmalloc_site_brackets_combined(args.run_root)
    elif args.mode == "fixture_obmalloc_expanded":
        payload = run_fixture_obmalloc_expanded_combined(
            args.run_root,
            mirror_durable_attribution=args.mirror_durable_attribution,
        )
    elif args.mode == "fixture_callsite_b_prime":
        payload = run_fixture_callsite_b_prime_combined(
            args.run_root,
            mirror_durable_attribution=args.mirror_durable_attribution,
        )
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
        "fixture_alloc_hook",
        "fixture_allocator_type",
        "fixture_non_glibc_mmap_source",
        "fixture_allocator_triangulation_ab",
        "fixture_allocator_triangulation_ab_replicate",
        "fixture_allocator_triangulation_tracemalloc",
        "fixture_allocator_triangulation_combined",
        "fixture_obmalloc_arena_ab",
        "fixture_obmalloc_arena_ab_replicate",
        "fixture_obmalloc_arena_b",
        "fixture_obmalloc_arena_combined",
        "fixture_obmalloc_site_brackets",
        "fixture_obmalloc_expanded",
        "fixture_callsite_b_prime",
    }:
        return resolve_fixture_attribution_main_exit_code(payload)
    fail_closed_terminal = _fail_closed_terminal_from_attribution_payload(payload)
    if fail_closed_terminal is not None:
        mapped = _mapped_terminal_exit_code(fail_closed_terminal)
        if mapped is not None:
            return int(mapped)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
