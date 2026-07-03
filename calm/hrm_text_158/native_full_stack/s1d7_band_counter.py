"""Cheap candidate-band allocation counters for C4.S1d.7 (no tracemalloc)."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from calm.hrm_text_158.native_full_stack.s1d7_tracemalloc_feasibility import (
    S1D7_BRANCH_CANDIDATE_A,
    S1D7_BRANCH_CANDIDATE_C,
    S1D7_BRANCH_CANDIDATE_E,
    S1D7_SITE_ID,
)

S1D7_BAND_COUNTER_SITE_SCHEMA = "hrm_text_158_s1d7_band_counter_site/v1"
S1D7_BAND_COUNTER_EVENT = f"s1d7_band_counter_site_{S1D7_SITE_ID}_post"

# Documented allocation-byte model constants (CPython 3.11+ amd64 assumptions).
PY_LIST_HEADER_BYTES = 56
PY_POINTER_SLOT_BYTES = 8
PYLONG_OBJECT_BYTES = 28
EVENT_OBJECT_BYTES = 56
NUMPY_ARRAY_HEADER_BYTES = 96

DOMINANCE_C_SHARE_MIN = 0.80
DOMINANCE_C_MULTIPLIER_MIN = 3.0


def estimate_band_a_allocation_bytes(
    *,
    crossing_indices_len: int,
    applied_indices_len: int,
) -> int:
    """Band A: crossing_indices allocates PyLongs; applied_indices is a shallow copy."""
    n_cross = max(int(crossing_indices_len), 0)
    n_applied = max(int(applied_indices_len), 0)
    crossing_bytes = (
        PY_LIST_HEADER_BYTES
        + n_cross * PY_POINTER_SLOT_BYTES
        + n_cross * PYLONG_OBJECT_BYTES
    )
    applied_bytes = PY_LIST_HEADER_BYTES + n_applied * PY_POINTER_SLOT_BYTES
    return int(crossing_bytes + applied_bytes)


def estimate_band_c_allocation_bytes(
    *,
    append_event_count: int,
    event_encoded_bytes_delta: int,
    q_level_writes: int,
) -> int:
    """Band C: EventCodedAccEvent object growth + list slots + compact encode delta."""
    n = max(int(append_event_count), 0)
    event_objects = n * EVENT_OBJECT_BYTES
    list_slots = n * PY_POINTER_SLOT_BYTES
    q_writes = max(int(q_level_writes), 0) * PYLONG_OBJECT_BYTES
    return int(event_objects + list_slots + max(int(event_encoded_bytes_delta), 0) + q_writes)


def estimate_band_e_allocation_bytes(
    *,
    remove_idx_nbytes: int,
    upd_idx_nbytes: int,
    upd_val_nbytes: int,
    remove_idx_count: int,
    upd_idx_count: int,
    upd_val_count: int,
) -> int:
    """Band E: numpy payload nbytes plus documented per-array header policy."""
    headers = 0
    if int(remove_idx_count) > 0:
        headers += NUMPY_ARRAY_HEADER_BYTES
    if int(upd_idx_count) > 0:
        headers += NUMPY_ARRAY_HEADER_BYTES
    if int(upd_val_count) > 0:
        headers += NUMPY_ARRAY_HEADER_BYTES
    return int(
        max(int(remove_idx_nbytes), 0)
        + max(int(upd_idx_nbytes), 0)
        + max(int(upd_val_nbytes), 0)
        + headers
    )


def collect_s1d7_band_counters(
    *,
    crossing_indices_len: int,
    applied_indices_len: int,
    append_event_count: int,
    event_encoded_bytes_delta: int,
    q_level_writes: int,
    remove_idx: np.ndarray,
    upd_idx: np.ndarray,
    upd_val: np.ndarray,
) -> dict[str, Any]:
    remove_arr = np.asarray(remove_idx, dtype=np.int32)
    upd_idx_arr = np.asarray(upd_idx, dtype=np.int32)
    upd_val_arr = np.asarray(upd_val, dtype=np.int16)
    counts = {
        "crossing_indices_len": int(crossing_indices_len),
        "applied_indices_len": int(applied_indices_len),
        "append_event_count": int(append_event_count),
        "event_encoded_bytes_delta": int(event_encoded_bytes_delta),
        "q_level_writes": int(q_level_writes),
        "remove_idx_count": int(remove_arr.size),
        "upd_idx_count": int(upd_idx_arr.size),
        "upd_val_count": int(upd_val_arr.size),
    }
    byte_proxies = {
        "band_a_bytes": estimate_band_a_allocation_bytes(
            crossing_indices_len=counts["crossing_indices_len"],
            applied_indices_len=counts["applied_indices_len"],
        ),
        "band_c_bytes": estimate_band_c_allocation_bytes(
            append_event_count=counts["append_event_count"],
            event_encoded_bytes_delta=counts["event_encoded_bytes_delta"],
            q_level_writes=counts["q_level_writes"],
        ),
        "band_e_bytes": estimate_band_e_allocation_bytes(
            remove_idx_nbytes=int(remove_arr.nbytes),
            upd_idx_nbytes=int(upd_idx_arr.nbytes),
            upd_val_nbytes=int(upd_val_arr.nbytes),
            remove_idx_count=counts["remove_idx_count"],
            upd_idx_count=counts["upd_idx_count"],
            upd_val_count=counts["upd_val_count"],
        ),
    }
    return {
        "counts": counts,
        "byte_proxies": byte_proxies,
        "byte_model": {
            "py_list_header_bytes": PY_LIST_HEADER_BYTES,
            "py_pointer_slot_bytes": PY_POINTER_SLOT_BYTES,
            "pylong_object_bytes": PYLONG_OBJECT_BYTES,
            "event_object_bytes": EVENT_OBJECT_BYTES,
            "numpy_array_header_bytes": NUMPY_ARRAY_HEADER_BYTES,
            "applied_indices_shallow_copy": True,
        },
    }


def has_s1d7_band_counter_marks(marks: Sequence[Mapping[str, Any]]) -> bool:
    return any(str(row.get("event") or "") == S1D7_BAND_COUNTER_EVENT for row in marks)


def evaluate_band_dominance(
    counter_rows: Sequence[Mapping[str, Any]],
    *,
    sampled_states: Sequence[int],
) -> dict[str, Any]:
    if not counter_rows:
        return {
            "band_counter_dominance_ok": False,
            "fail_closed_reason": "BAND_COUNTER_MISSING_ROWS",
            "dominant_band": None,
            "band_c_share": 0.0,
        }
    if len(counter_rows) != len(tuple(sampled_states)):
        return {
            "band_counter_dominance_ok": False,
            "fail_closed_reason": "BAND_COUNTER_ROW_COUNT_MISMATCH",
            "dominant_band": None,
            "band_c_share": 0.0,
        }

    totals = {"a": 0, "c": 0, "e": 0}
    per_state: list[dict[str, Any]] = []
    for row in counter_rows:
        proxies = dict(row.get("s1d7_band_counters") or {}).get("byte_proxies") or {}
        a_bytes = int(proxies.get("band_a_bytes") or 0)
        c_bytes = int(proxies.get("band_c_bytes") or 0)
        e_bytes = int(proxies.get("band_e_bytes") or 0)
        state_total = a_bytes + c_bytes + e_bytes
        per_state.append(
            {
                "state_index": row.get("state_index"),
                "band_a_bytes": a_bytes,
                "band_c_bytes": c_bytes,
                "band_e_bytes": e_bytes,
                "state_total_bytes": state_total,
            }
        )
        if state_total <= 0:
            return {
                "band_counter_dominance_ok": False,
                "fail_closed_reason": "BAND_COUNTER_ALL_ZERO_ACTIVITY",
                "dominant_band": None,
                "band_c_share": 0.0,
                "per_state": per_state,
            }
        totals["a"] += a_bytes
        totals["c"] += c_bytes
        totals["e"] += e_bytes
        state_dominant = max(("a", a_bytes), ("c", c_bytes), ("e", e_bytes), key=lambda item: item[1])[0]
        if state_dominant != "c":
            return {
                "band_counter_dominance_ok": False,
                "fail_closed_reason": "BAND_COUNTER_C_NOT_TOP_IN_STATE",
                "dominant_band": state_dominant,
                "band_c_share": 0.0,
                "per_state": per_state,
                "state_index": row.get("state_index"),
            }

    grand_total = totals["a"] + totals["c"] + totals["e"]
    if grand_total <= 0:
        return {
            "band_counter_dominance_ok": False,
            "fail_closed_reason": "BAND_COUNTER_ALL_ZERO_ACTIVITY",
            "dominant_band": None,
            "band_c_share": 0.0,
            "per_state": per_state,
        }
    c_share = float(totals["c"]) / float(grand_total)
    next_largest = max(totals["a"], totals["e"])
    if c_share < DOMINANCE_C_SHARE_MIN:
        return {
            "band_counter_dominance_ok": False,
            "fail_closed_reason": "BAND_COUNTER_C_SHARE_FAIL",
            "dominant_band": "c",
            "band_c_share": c_share,
            "per_state": per_state,
            "aggregate_band_bytes": totals,
        }
    if totals["c"] < DOMINANCE_C_MULTIPLIER_MIN * next_largest:
        return {
            "band_counter_dominance_ok": False,
            "fail_closed_reason": "BAND_COUNTER_C_MULTIPLIER_FAIL",
            "dominant_band": "c",
            "band_c_share": c_share,
            "per_state": per_state,
            "aggregate_band_bytes": totals,
        }
    return {
        "band_counter_dominance_ok": True,
        "fail_closed_reason": None,
        "dominant_band": "c",
        "band_c_share": c_share,
        "per_state": per_state,
        "aggregate_band_bytes": totals,
    }


def attribute_s1d7_band_counter_call_site_from_marks(
    marks_b: Sequence[Mapping[str, Any]],
    *,
    sampled_states: Sequence[int],
    guards: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _ = guards
    unresolved = {
        "call_site_status": "UNRESOLVED",
        "call_site_origin_file_line": None,
        "tracemalloc_perturbed": False,
        "s1d7_call_site_in_bracket_ok": False,
        "s1d7_band_counter_mark_count": 0,
        "s1d7_tracemalloc_mark_pair_count": 0,
        "s1d7_tracemalloc_mark_schema": "band_counter_only",
    }
    rows: list[dict[str, Any]] = []
    for row in marks_b:
        if str(row.get("event") or "") != S1D7_BAND_COUNTER_EVENT:
            continue
        if row.get("state_index") is None:
            continue
        rows.append(dict(row))

    sampled_states_tuple = tuple(int(state_idx) for state_idx in sampled_states)
    sampled_state_set = set(sampled_states_tuple)
    state_indices = [int(row["state_index"]) for row in rows]

    if len(rows) != len(sampled_states_tuple):
        return {
            **unresolved,
            "fail_closed_reason": "BAND_COUNTER_ROW_COUNT_MISMATCH",
            "s1d7_band_counter_mark_count": len(rows),
        }
    if len(set(state_indices)) != len(state_indices):
        return {
            **unresolved,
            "fail_closed_reason": "BAND_COUNTER_DUPLICATE_ROW",
            "s1d7_band_counter_mark_count": len(rows),
        }
    unexpected_states = set(state_indices) - sampled_state_set
    if unexpected_states:
        return {
            **unresolved,
            "fail_closed_reason": "BAND_COUNTER_UNEXPECTED_STATE",
            "s1d7_band_counter_mark_count": len(rows),
            "unexpected_state_indices": sorted(unexpected_states),
        }
    for state_idx in sampled_states_tuple:
        if state_idx not in state_indices:
            return {
                **unresolved,
                "fail_closed_reason": "BAND_COUNTER_MISSING_ROW",
                "s1d7_band_counter_mark_count": len(rows),
            }

    rows_by_state = {int(row["state_index"]): row for row in rows}
    ordered_rows = [rows_by_state[state_idx] for state_idx in sampled_states_tuple]
    dominance = evaluate_band_dominance(ordered_rows, sampled_states=sampled_states)
    if not dominance.get("band_counter_dominance_ok"):
        return {
            **unresolved,
            "fail_closed_reason": dominance.get("fail_closed_reason"),
            "s1d7_band_counter_mark_count": len(ordered_rows),
            "s1d7_band_counter_dominance": dominance,
        }
    return {
        "call_site_status": "RESOLVED",
        "call_site_origin_file_line": "event_coded_acc_live_carrier.py:896",
        "tracemalloc_perturbed": False,
        "s1d7_call_site_in_bracket_ok": True,
        "s1d7_call_site_candidate": "c",
        "s1d7_call_site_branch_outcome": S1D7_BRANCH_CANDIDATE_C,
        "s1d7_band_counter_mark_count": len(ordered_rows),
        "s1d7_tracemalloc_mark_pair_count": 0,
        "s1d7_tracemalloc_mark_schema": "band_counter_only",
        "s1d7_band_counter_dominance": dominance,
        "fail_closed_reason": None,
    }


def classify_forced_calibration_case(
    counters: Mapping[str, Any],
) -> str | None:
    proxies = dict(counters.get("byte_proxies") or {})
    a_bytes = int(proxies.get("band_a_bytes") or 0)
    c_bytes = int(proxies.get("band_c_bytes") or 0)
    e_bytes = int(proxies.get("band_e_bytes") or 0)
    total = a_bytes + c_bytes + e_bytes
    if total <= 0:
        return None
    dominant = max(("a", a_bytes), ("c", c_bytes), ("e", e_bytes), key=lambda item: item[1])[0]
    return dominant


def synthetic_band_a_counters(*, n_lanes: int) -> dict[str, Any]:
    return collect_s1d7_band_counters(
        crossing_indices_len=int(n_lanes),
        applied_indices_len=int(n_lanes),
        append_event_count=0,
        event_encoded_bytes_delta=0,
        q_level_writes=0,
        remove_idx=np.empty(0, dtype=np.int32),
        upd_idx=np.empty(0, dtype=np.int32),
        upd_val=np.empty(0, dtype=np.int16),
    )


def synthetic_band_e_counters(*, n_lanes: int) -> dict[str, Any]:
    n = max(int(n_lanes), 0)
    remove_idx = np.arange(n, dtype=np.int32)
    upd_idx = np.arange(n, dtype=np.int32)
    upd_val = np.full(n, 1, dtype=np.int16)
    return collect_s1d7_band_counters(
        crossing_indices_len=0,
        applied_indices_len=0,
        append_event_count=0,
        event_encoded_bytes_delta=0,
        q_level_writes=0,
        remove_idx=remove_idx,
        upd_idx=upd_idx,
        upd_val=upd_val,
    )


def branch_outcome_for_band(band: str | None) -> str | None:
    if band == "a":
        return S1D7_BRANCH_CANDIDATE_A
    if band == "c":
        return S1D7_BRANCH_CANDIDATE_C
    if band == "e":
        return S1D7_BRANCH_CANDIDATE_E
    return None


def maybe_emit_s1d7_band_counter_post(
    emit: Any,
    *,
    crossing_indices_len: int,
    applied_indices_len: int,
    append_event_count: int,
    event_encoded_bytes_delta: int,
    q_level_writes: int,
    remove_idx: Any,
    upd_idx: Any,
    upd_val: Any,
    origin_line: int,
    optimizer_step_index: int,
    state_index: int,
) -> None:
    if emit is None:
        return
    counters = collect_s1d7_band_counters(
        crossing_indices_len=int(crossing_indices_len),
        applied_indices_len=int(applied_indices_len),
        append_event_count=int(append_event_count),
        event_encoded_bytes_delta=int(event_encoded_bytes_delta),
        q_level_writes=int(q_level_writes),
        remove_idx=remove_idx,
        upd_idx=upd_idx,
        upd_val=upd_val,
    )
    emit(
        origin_file="event_coded_acc_live_carrier.py",
        origin_line=int(origin_line),
        counters=counters,
        optimizer_step_index=int(optimizer_step_index),
        state_index=int(state_index),
    )
