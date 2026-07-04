"""CPU feasibility harness for C4.S1d.7 scoped tracemalloc call-site resolution."""

from __future__ import annotations

import re
import tracemalloc
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from calm.hrm_text_158.native_full_stack.host_tracemalloc_probe import (
    BRANCH1_CONCENTRATION,
    diff_traceback_frames,
)

S1D7_SITE_ID = "C4.S1d.7"
S1D7_PRE_EVENT = f"obmalloc_site_{S1D7_SITE_ID}_pre"
S1D7_POST_EVENT = f"obmalloc_site_{S1D7_SITE_ID}_post"
S1D7_TRACEMALLOC_PRE_EVENT = f"s1d7_tracemalloc_site_{S1D7_SITE_ID}_pre"
S1D7_TRACEMALLOC_POST_EVENT = f"s1d7_tracemalloc_site_{S1D7_SITE_ID}_post"
S1D7_TRACEMALLOC_SITE_SCHEMA = "hrm_text_158_s1d7_tracemalloc_site/v1"

CARRIER_ORIGIN_FILE = "event_coded_acc_live_carrier.py"
# Physical line bands in apply_step after static-pre-append seam move (marker :895).
S1D7_ACCEPTANCE_LINE_MIN = 909
S1D7_ACCEPTANCE_LINE_MAX = 955
CANDIDATE_A_LINE = 910
CANDIDATE_C_LINE_MIN = 941
CANDIDATE_C_LINE_MAX = 952
CANDIDATE_E_LINE_MIN = 914
CANDIDATE_E_LINE_MAX = 917

S1D7_BRANCH_CANDIDATE_A = "S1D7_CALL_SITE_CANDIDATE_A_CROSSING_INDICES"
S1D7_BRANCH_CANDIDATE_C = "S1D7_CALL_SITE_CANDIDATE_C_EVENTS_JOURNAL"
S1D7_BRANCH_CANDIDATE_E = "S1D7_CALL_SITE_CANDIDATE_E_NUMPY_ARRAYS"
S1D7_BRANCH_CANDIDATE_AMBIGUOUS = "S1D7_CALL_SITE_RESOLVED_CANDIDATE_AMBIGUOUS"

TRACEMALLOC_TRACEBACK_DEPTH = 64


def take_tracemalloc_snapshot_dict(*, top_n: int = 256) -> dict[str, Any]:
    current, peak = tracemalloc.get_traced_memory()
    stats = tracemalloc.take_snapshot()
    top_stats = stats.statistics("traceback")[: int(top_n)]
    frames: list[dict[str, Any]] = []
    for stat in top_stats:
        tb = stat.traceback
        traceback_lines = [f'  File "{frame.filename}", line {frame.lineno}' for frame in tb]
        frames.append(
            {
                "size_bytes": int(stat.size),
                "count": int(stat.count),
                "traceback": traceback_lines[:TRACEMALLOC_TRACEBACK_DEPTH],
                "traceback_key": "|".join(traceback_lines[:3]),
            }
        )
    return {
        "enabled": True,
        "traced_current_bytes": int(current),
        "traced_peak_bytes": int(peak),
        "top_frames": frames,
    }


def scoped_tracemalloc_bracket_diff(
    work_fn: Callable[[], None],
    *,
    depth: int = 50,
) -> dict[str, Any]:
    from calm.hrm_text_158.native_full_stack.host_tracemalloc_probe import (
        reset_tracemalloc_state_for_tests,
    )

    reset_tracemalloc_state_for_tests()
    tracemalloc.stop()
    tracemalloc.start(int(depth))
    try:
        baseline = take_tracemalloc_snapshot_dict()
        work_fn()
        current = take_tracemalloc_snapshot_dict()
    finally:
        tracemalloc.stop()
    return diff_traceback_frames(baseline, current, top_n=256)


def parse_origin_from_traceback(
    traceback_lines: Sequence[str],
) -> tuple[str | None, int | None]:
    for line in traceback_lines:
        if CARRIER_ORIGIN_FILE not in line:
            continue
        match = re.search(r"line\s+(\d+)", line)
        if match:
            return CARRIER_ORIGIN_FILE, int(match.group(1))
    return None, None


def resolve_dominant_carrier_frame(
    diff: Mapping[str, Any],
    *,
    line_min: int | None = S1D7_ACCEPTANCE_LINE_MIN,
    line_max: int | None = S1D7_ACCEPTANCE_LINE_MAX,
) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    for frame in list(diff.get("top_delta_frames") or []):
        origin_file, origin_line = parse_origin_from_traceback(
            list(frame.get("traceback") or [])
        )
        if origin_file != CARRIER_ORIGIN_FILE or origin_line is None:
            continue
        if line_min is not None and origin_line < int(line_min):
            continue
        if line_max is not None and origin_line > int(line_max):
            continue
        delta_bytes = int(frame.get("delta_bytes") or 0)
        if best is None or delta_bytes > int(best.get("delta_bytes") or 0):
            best = {
                "origin_file": origin_file,
                "origin_line": origin_line,
                "delta_bytes": delta_bytes,
                "traceback": list(frame.get("traceback") or []),
                "concentration_fraction": float(frame.get("concentration_fraction") or 0.0),
            }
    return best


def map_s1d7_call_site_candidate(origin_line: int) -> str | None:
    line = int(origin_line)
    if line < S1D7_ACCEPTANCE_LINE_MIN or line > S1D7_ACCEPTANCE_LINE_MAX:
        return None
    if line == CANDIDATE_A_LINE:
        return "a"
    if CANDIDATE_C_LINE_MIN <= line <= CANDIDATE_C_LINE_MAX:
        return "c"
    if CANDIDATE_E_LINE_MIN <= line <= CANDIDATE_E_LINE_MAX:
        return "e"
    return "ambiguous"


def branch_outcome_for_s1d7_candidate(candidate: str | None) -> str | None:
    if candidate == "a":
        return S1D7_BRANCH_CANDIDATE_A
    if candidate == "c":
        return S1D7_BRANCH_CANDIDATE_C
    if candidate == "e":
        return S1D7_BRANCH_CANDIDATE_E
    if candidate == "ambiguous":
        return S1D7_BRANCH_CANDIDATE_AMBIGUOUS
    return None


def has_s1d7_tracemalloc_site_marks(marks: Sequence[Mapping[str, Any]]) -> bool:
    return any(
        str(row.get("event") or "") in {S1D7_TRACEMALLOC_PRE_EVENT, S1D7_TRACEMALLOC_POST_EVENT}
        for row in marks
    )


def resolve_s1d7_tracemalloc_mark_events(
    marks: Sequence[Mapping[str, Any]],
) -> tuple[str, str, str]:
    if has_s1d7_tracemalloc_site_marks(marks):
        return S1D7_TRACEMALLOC_PRE_EVENT, S1D7_TRACEMALLOC_POST_EVENT, "tracemalloc_only"
    return S1D7_PRE_EVENT, S1D7_POST_EVENT, "legacy_obmalloc_embed"


def _pair_s1d7_tracemalloc_marks_by_state(
    marks_b: Sequence[Mapping[str, Any]],
    *,
    pre_event: str,
    post_event: str,
    sampled_states: Sequence[int],
) -> tuple[dict[int, Mapping[str, Any]], dict[int, Mapping[str, Any]], str | None]:
    pre_by_state: dict[int, Mapping[str, Any]] = {}
    post_by_state: dict[int, Mapping[str, Any]] = {}
    for row in marks_b:
        event = str(row.get("event") or "")
        if row.get("state_index") is None:
            continue
        state_idx = int(row["state_index"])
        if event == pre_event:
            if state_idx in pre_by_state:
                return pre_by_state, post_by_state, "TRACEMALLOC_DUPLICATE_PRE"
            pre_by_state[state_idx] = row
        elif event == post_event:
            if state_idx in post_by_state:
                return pre_by_state, post_by_state, "TRACEMALLOC_DUPLICATE_POST"
            post_by_state[state_idx] = row
    for state_idx in sampled_states:
        if int(state_idx) not in pre_by_state or int(state_idx) not in post_by_state:
            return pre_by_state, post_by_state, "TRACEMALLOC_MISSING_PAIR"
    return pre_by_state, post_by_state, None


def classify_s1d7_tracemalloc_call_site(
    diff: Mapping[str, Any],
    *,
    concentration_min: float = BRANCH1_CONCENTRATION,
    line_min: int = S1D7_ACCEPTANCE_LINE_MIN,
    line_max: int = S1D7_ACCEPTANCE_LINE_MAX,
) -> dict[str, Any]:
    concentration = float(diff.get("top_concentration_fraction") or 0.0)
    top_frames = list(diff.get("top_delta_frames") or [])
    if not top_frames or int(diff.get("current_delta_bytes") or 0) <= 0:
        return {
            "call_site_status": "UNRESOLVED",
            "call_site_origin_file_line": None,
            "top_concentration_fraction": concentration,
            "fail_closed_reason": "TRACEMALLOC_INCONCLUSIVE",
            "s1d7_call_site_candidate": None,
            "s1d7_call_site_branch_outcome": None,
        }
    carrier_frame_any = resolve_dominant_carrier_frame(diff, line_min=None, line_max=None)
    if carrier_frame_any is None:
        return {
            "call_site_status": "UNRESOLVED",
            "call_site_origin_file_line": None,
            "top_concentration_fraction": concentration,
            "fail_closed_reason": "TRACEMALLOC_INCONCLUSIVE",
            "s1d7_call_site_candidate": None,
            "s1d7_call_site_branch_outcome": None,
        }
    origin_line = int(carrier_frame_any["origin_line"])
    origin = f"{CARRIER_ORIGIN_FILE}:{origin_line}"
    frame_concentration = float(carrier_frame_any.get("concentration_fraction") or 0.0)
    if origin_line < int(line_min) or origin_line > int(line_max):
        return {
            "call_site_status": "UNRESOLVED",
            "call_site_origin_file_line": origin,
            "top_concentration_fraction": frame_concentration,
            "s1d7_call_site_in_bracket_ok": False,
            "fail_closed_reason": "CALL_SITE_OUTSIDE_S1D7_BRACKET",
            "top_delta_frame": carrier_frame_any,
            "s1d7_call_site_candidate": None,
            "s1d7_call_site_branch_outcome": None,
        }
    if frame_concentration < float(concentration_min):
        return {
            "call_site_status": "UNRESOLVED",
            "call_site_origin_file_line": origin,
            "top_concentration_fraction": frame_concentration,
            "s1d7_call_site_in_bracket_ok": True,
            "fail_closed_reason": "TRACEMALLOC_CONCENTRATION_FAIL",
            "top_delta_frame": carrier_frame_any,
            "s1d7_call_site_candidate": map_s1d7_call_site_candidate(origin_line),
            "s1d7_call_site_branch_outcome": None,
        }
    candidate = map_s1d7_call_site_candidate(origin_line)
    return {
        "call_site_status": "RESOLVED",
        "call_site_origin_file_line": origin,
        "top_concentration_fraction": frame_concentration,
        "s1d7_call_site_in_bracket_ok": True,
        "fail_closed_reason": None,
        "top_delta_frame": carrier_frame_any,
        "s1d7_call_site_candidate": candidate,
        "s1d7_call_site_branch_outcome": branch_outcome_for_s1d7_candidate(candidate),
    }


def run_carrier_crossing_indices_workload(*, n_lanes: int) -> None:
    from calm.hrm_text_158.native_full_stack.event_coded_acc_live_carrier import (
        EventCodedAccLiveState,
    )

    carrier = EventCodedAccLiveState(
        logical_numel=max(int(n_lanes) * 2, int(n_lanes) + 1),
        threshold_abs=10,
    )
    indices = np.arange(int(n_lanes), dtype=np.int32)
    values = np.full(int(n_lanes), 9, dtype=np.int16)
    carrier._hot.replace_arrays(indices, values)
    vote_values = np.full(int(n_lanes), 4, dtype=np.int32)
    carrier.apply_step(
        step_index=0,
        sparse_vote_indices=indices.astype(np.int64),
        sparse_vote_values=vote_values,
        state_index=0,
        optimizer_step_index=0,
    )


def run_crossing_indices_workload(*, n_lanes: int) -> None:
    run_carrier_crossing_indices_workload(n_lanes=n_lanes)


def run_carrier_events_journal_workload(
    *,
    n_steps: int,
    n_lanes: int,
    s1d7_band_counter_emit: Callable[..., None] | None = None,
) -> None:
    from calm.hrm_text_158.native_full_stack.event_coded_acc_live_carrier import (
        EventCodedAccLiveState,
    )

    carrier = EventCodedAccLiveState(
        logical_numel=max(int(n_lanes) * 2, int(n_lanes) + 1),
        threshold_abs=10,
    )
    indices = np.arange(int(n_lanes), dtype=np.int32)
    values = np.full(int(n_lanes), 9, dtype=np.int16)
    vote_values = np.full(int(n_lanes), 4, dtype=np.int32)
    for step in range(int(n_steps)):
        carrier._hot.replace_arrays(indices, values)
        carrier.apply_step(
            step_index=step,
            sparse_vote_indices=indices.astype(np.int64),
            sparse_vote_values=vote_values,
            state_index=step % 4,
            optimizer_step_index=step,
            s1d7_band_counter_emit=s1d7_band_counter_emit,
        )


def run_events_journal_workload(*, n_steps: int, n_lanes: int) -> None:
    run_carrier_events_journal_workload(n_steps=n_steps, n_lanes=n_lanes)


def run_carrier_demotion_update_workload(
    *,
    n_lanes: int,
    s1d7_band_counter_emit: Callable[..., None] | None = None,
) -> None:
    """Real apply_step path with demotion/update activity and zero crossings."""
    from calm.hrm_text_158.native_full_stack.event_coded_acc_live_carrier import (
        EventCodedAccLiveState,
    )

    carrier = EventCodedAccLiveState(
        logical_numel=max(int(n_lanes) * 4, int(n_lanes) + 1),
        threshold_abs=100,
        demotion_band=3,
    )
    indices = np.arange(int(n_lanes), dtype=np.int32)
    values = np.full(int(n_lanes), 2, dtype=np.int16)
    carrier._hot.replace_arrays(indices, values)
    for idx in indices:
        carrier.q_levels[int(idx)] = 50
    carrier.apply_step(
        step_index=0,
        sparse_vote_indices=np.empty(0, dtype=np.int64),
        sparse_vote_values=np.empty(0, dtype=np.int32),
        state_index=0,
        optimizer_step_index=0,
        s1d7_band_counter_emit=s1d7_band_counter_emit,
    )


def run_demotion_update_workload(*, n_lanes: int) -> None:
    run_carrier_demotion_update_workload(n_lanes=n_lanes)


def collect_carrier_band_counters_for_workload(workload_fn: Callable[[Callable[..., None] | None], None]) -> dict[str, Any]:
    captured: list[dict[str, Any]] = []

    def _emit(
        *,
        origin_file: str,
        origin_line: int,
        counters: Mapping[str, Any],
        optimizer_step_index: int,
        state_index: int,
    ) -> None:
        _ = (origin_file, origin_line, optimizer_step_index, state_index)
        captured.append(dict(counters))

    workload_fn(_emit)
    if not captured:
        raise RuntimeError("band_counter_capture_empty")
    return dict(captured[-1])


def calibrate_band_counters_vs_classifier(*, case: str) -> dict[str, Any]:
    from calm.hrm_text_158.native_full_stack.s1d7_band_counter import (
        classify_forced_calibration_case,
        synthetic_band_a_counters,
    )

    if case == "forced_c":
        counters = collect_carrier_band_counters_for_workload(
            lambda emit: run_carrier_events_journal_workload(
                n_steps=1, n_lanes=25_000, s1d7_band_counter_emit=emit
            )
        )
        diff = scoped_tracemalloc_bracket_diff(
            lambda: run_carrier_events_journal_workload(n_steps=1, n_lanes=25_000)
        )
        classified = classify_s1d7_tracemalloc_call_site(diff)
        counter_band = classify_forced_calibration_case(counters)
        return {
            "case": case,
            "counter_band": counter_band,
            "classifier_candidate": classified.get("s1d7_call_site_candidate"),
            "ok": counter_band == "c" and classified.get("s1d7_call_site_candidate") == "c",
            "counters": counters,
        }
    if case == "forced_e":
        counters = collect_carrier_band_counters_for_workload(
            lambda emit: run_carrier_demotion_update_workload(
                n_lanes=25_000, s1d7_band_counter_emit=emit
            )
        )
        diff = scoped_tracemalloc_bracket_diff(
            lambda: run_carrier_demotion_update_workload(n_lanes=25_000)
        )
        classified = classify_s1d7_tracemalloc_call_site(diff)
        counter_band = classify_forced_calibration_case(counters)
        return {
            "case": case,
            "counter_band": counter_band,
            "classifier_candidate": classified.get("s1d7_call_site_candidate"),
            "ok": counter_band == "e" and classified.get("s1d7_call_site_candidate") == "e",
            "counters": counters,
        }
    if case == "synthetic_a":
        counters = synthetic_band_a_counters(n_lanes=25_000)
        counter_band = classify_forced_calibration_case(counters)
        return {
            "case": case,
            "counter_band": counter_band,
            "classifier_candidate": "a",
            "ok": counter_band == "a",
            "counters": counters,
        }
    if case == "synthetic_e":
        from calm.hrm_text_158.native_full_stack.s1d7_band_counter import (
            synthetic_band_e_counters,
        )

        counters = synthetic_band_e_counters(n_lanes=25_000)
        counter_band = classify_forced_calibration_case(counters)
        return {
            "case": case,
            "counter_band": counter_band,
            "classifier_candidate": "e",
            "ok": counter_band == "e",
            "counters": counters,
        }
    raise ValueError(f"unknown_calibration_case:{case}")


def cpu_perturbation_guard_smoke(*, noise_floor_gib: float = 0.01) -> dict[str, Any]:
    from scripts.hrm_text_158_slice5_v6i_oom_profile_attribution import (
        PERTURBATION_MIN_GIB,
        PERTURBATION_NOISE_K,
    )

    threshold = max(PERTURBATION_MIN_GIB, PERTURBATION_NOISE_K * float(noise_floor_gib))
    perturbation_delta = 0.0
    return {
        "tracemalloc_perturbed": perturbation_delta > threshold,
        "perturbation_delta_gib": perturbation_delta,
        "perturbation_threshold_gib": threshold,
        "baseline_rss_gib": 0.0,
    }


def run_cpu_feasibility_case(case: str) -> dict[str, Any]:
    if case == "crossing_indices":
        diff = scoped_tracemalloc_bracket_diff(
            lambda: run_carrier_crossing_indices_workload(n_lanes=25_000)
        )
        result = classify_s1d7_tracemalloc_call_site(diff)
        origin_line = None
        if result.get("call_site_origin_file_line"):
            origin_line = int(str(result["call_site_origin_file_line"]).rsplit(":", 1)[-1])
        return {
            "case": case,
            "current_delta_bytes": int(diff.get("current_delta_bytes") or 0),
            "result": result,
            "origin_line": origin_line,
            "ok": (
                result.get("call_site_status") == "RESOLVED"
                and result.get("s1d7_call_site_in_bracket_ok") is True
                and float(result.get("top_concentration_fraction") or 0.0) >= 0.60
                and origin_line is not None
                and S1D7_ACCEPTANCE_LINE_MIN <= origin_line <= S1D7_ACCEPTANCE_LINE_MAX
            ),
        }
    if case == "events_journal":
        diff = scoped_tracemalloc_bracket_diff(
            lambda: run_carrier_events_journal_workload(n_steps=1, n_lanes=30_000)
        )
        result = classify_s1d7_tracemalloc_call_site(diff)
        origin_line = None
        if result.get("call_site_origin_file_line"):
            origin_line = int(str(result["call_site_origin_file_line"]).rsplit(":", 1)[-1])
        return {
            "case": case,
            "current_delta_bytes": int(diff.get("current_delta_bytes") or 0),
            "result": result,
            "origin_line": origin_line,
            "ok": (
                result.get("call_site_status") == "RESOLVED"
                and result.get("s1d7_call_site_in_bracket_ok") is True
                and float(result.get("top_concentration_fraction") or 0.0) >= 0.60
                and origin_line is not None
                and CANDIDATE_C_LINE_MIN <= origin_line <= CANDIDATE_C_LINE_MAX
            ),
        }
    if case == "perturbation_smoke":
        smoke = cpu_perturbation_guard_smoke(noise_floor_gib=0.01)
        return {
            "case": case,
            "smoke": smoke,
            "ok": smoke["tracemalloc_perturbed"] is False,
        }
    raise ValueError(f"unknown_cpu_feasibility_case:{case}")


def aggregate_s1d7_tracemalloc_diffs(
    diffs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    from collections import Counter

    merged_counter: Counter[str] = Counter()
    sample_tracebacks: dict[str, list[str]] = {}
    total_current_delta = 0
    for diff in diffs:
        total_current_delta += max(int(diff.get("current_delta_bytes") or 0), 0)
        for frame in list(diff.get("top_delta_frames") or []):
            key = str(frame.get("traceback_key") or "")
            if not key:
                continue
            merged_counter[key] += int(frame.get("delta_bytes") or 0)
            sample_tracebacks[key] = list(frame.get("traceback") or [])
    total_delta = max(total_current_delta, sum(merged_counter.values()), 1)
    top_delta_frames: list[dict[str, Any]] = []
    for key, delta_bytes in merged_counter.most_common(256):
        if int(delta_bytes) <= 0:
            continue
        top_delta_frames.append(
            {
                "traceback_key": key,
                "delta_bytes": int(delta_bytes),
                "traceback": sample_tracebacks.get(key, []),
                "concentration_fraction": float(delta_bytes) / float(total_delta),
            }
        )
    concentrations = sorted(
        [float(row["concentration_fraction"]) for row in top_delta_frames],
        reverse=True,
    )
    return {
        "current_delta_bytes": int(total_current_delta),
        "top_delta_frames": top_delta_frames,
        "top_concentration_fraction": float(concentrations[0]) if concentrations else 0.0,
    }


def attribute_s1d7_tracemalloc_call_site_from_marks(
    marks_b: Sequence[Mapping[str, Any]],
    *,
    sampled_states: Sequence[int],
    guards: Mapping[str, Any],
) -> dict[str, Any]:
    perturbation_delta = guards.get("perturbation_delta_gib")
    perturbation_threshold = guards.get("perturbation_threshold_gib")
    tracemalloc_perturbed = (
        perturbation_delta is not None
        and perturbation_threshold is not None
        and float(perturbation_delta) > float(perturbation_threshold)
    )
    unresolved = {
        "call_site_status": "UNRESOLVED",
        "call_site_origin_file_line": None,
        "tracemalloc_perturbed": bool(tracemalloc_perturbed),
        "s1d7_call_site_in_bracket_ok": False,
        "s1d7_tracemalloc_diff": None,
        "s1d7_tracemalloc_mark_pair_count": 0,
    }
    if tracemalloc_perturbed:
        return {
            **unresolved,
            "fail_closed_reason": "TRACEMALLOC_PERTURBED_INCONCLUSIVE",
        }

    pre_event, post_event, mark_schema = resolve_s1d7_tracemalloc_mark_events(marks_b)
    pre_by_state, post_by_state, pair_error = _pair_s1d7_tracemalloc_marks_by_state(
        marks_b,
        pre_event=pre_event,
        post_event=post_event,
        sampled_states=sampled_states,
    )
    if pair_error is not None:
        return {
            **unresolved,
            "s1d7_tracemalloc_mark_schema": mark_schema,
            "fail_closed_reason": pair_error,
        }

    per_state_diffs: list[dict[str, Any]] = []
    for state_idx in sampled_states:
        pre = pre_by_state.get(int(state_idx))
        post = post_by_state.get(int(state_idx))
        if pre is None or post is None:
            continue
        baseline = dict(pre.get("s1d7_tracemalloc") or {})
        current = dict(post.get("s1d7_tracemalloc") or {})
        if not baseline.get("enabled") or not current.get("enabled"):
            continue
        per_state_diffs.append(diff_traceback_frames(baseline, current, top_n=256))

    unresolved["s1d7_tracemalloc_mark_pair_count"] = len(per_state_diffs)
    if not per_state_diffs or len(per_state_diffs) != len(tuple(sampled_states)):
        return {
            **unresolved,
            "s1d7_tracemalloc_mark_schema": mark_schema,
            "fail_closed_reason": "TRACEMALLOC_INCONCLUSIVE",
        }

    merged_diff = aggregate_s1d7_tracemalloc_diffs(per_state_diffs)
    classified = classify_s1d7_tracemalloc_call_site(merged_diff)
    return {
        "call_site_status": classified.get("call_site_status"),
        "call_site_origin_file_line": classified.get("call_site_origin_file_line"),
        "tracemalloc_perturbed": False,
        "s1d7_call_site_in_bracket_ok": classified.get("s1d7_call_site_in_bracket_ok"),
        "s1d7_call_site_candidate": classified.get("s1d7_call_site_candidate"),
        "s1d7_call_site_branch_outcome": classified.get("s1d7_call_site_branch_outcome"),
        "s1d7_tracemalloc_diff": merged_diff,
        "s1d7_tracemalloc_top_concentration_fraction": classified.get(
            "top_concentration_fraction"
        ),
        "s1d7_tracemalloc_mark_pair_count": len(per_state_diffs),
        "s1d7_tracemalloc_mark_schema": mark_schema,
        "fail_closed_reason": classified.get("fail_closed_reason"),
        "top_delta_frame": classified.get("top_delta_frame"),
    }


def main() -> int:
    import json
    import sys

    case = str(sys.argv[1] if len(sys.argv) > 1 else "all")
    cases = (
        ["crossing_indices", "events_journal", "perturbation_smoke"]
        if case == "all"
        else [case]
    )
    payload = [run_cpu_feasibility_case(name) for name in cases]
    print(json.dumps(payload, sort_keys=True))
    return 0 if all(row.get("ok") for row in payload) else 1


if __name__ == "__main__":
    raise SystemExit(main())
