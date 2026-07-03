"""Compact tracemalloc snapshots for Python allocator triangulation (marked diagnostic)."""

from __future__ import annotations

import tracemalloc
from collections import Counter
from typing import Any, Mapping, Sequence

TRACEMALLOC_TOP_N = 16
BRANCH1_CONCENTRATION = 0.60
BRANCH1_RECONCILE_MIN = 0.5
BRANCH1_RECONCILE_MAX = 1.5
BRANCH2_PEAK_FRAC = 0.50
BRANCH2_CURRENT_VS_PEAK = 0.25
BRANCH3_CURRENT_FRAC = 0.25
NON_CONCENTRATED_FRAME_FRAC = 0.20


def profile_tracemalloc_enabled() -> bool:
    import os

    rss_on = os.environ.get("HRM_TEXT_158_PROFILE_HOST_RSS", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    tm_on = os.environ.get("HRM_TEXT_158_PROFILE_TRACEMALLOC", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    return rss_on and tm_on


_tracemalloc_started = False


def ensure_tracemalloc_started(*, depth: int = 25) -> bool:
    global _tracemalloc_started
    if not profile_tracemalloc_enabled():
        return False
    if not _tracemalloc_started:
        tracemalloc.start(int(depth))
        _tracemalloc_started = True
    return True


def profile_s1d7_tracemalloc_site_enabled() -> bool:
    import os

    if not profile_tracemalloc_enabled():
        return False
    debugmallocstats_on = os.environ.get(
        "HRM_TEXT_158_PROFILE_DEBUGMALLOCSTATS", ""
    ).strip().lower() in {"1", "true", "yes", "on"}
    return not debugmallocstats_on


PROFILE_S1D7_TRACEMALLOC_FULL_TRACE_ENV = "HRM_TEXT_158_PROFILE_S1D7_TRACEMALLOC_FULL_TRACE"


def profile_s1d7_tracemalloc_full_trace_enabled() -> bool:
    import os

    if not profile_s1d7_tracemalloc_site_enabled():
        return False
    return os.environ.get(PROFILE_S1D7_TRACEMALLOC_FULL_TRACE_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def profile_s1d7_band_counter_enabled() -> bool:
    """B-arm default: cheap band counters instead of full tracemalloc (fallback D opt-in)."""
    return profile_s1d7_tracemalloc_site_enabled() and not profile_s1d7_tracemalloc_full_trace_enabled()


def begin_s1d7_tracemalloc_bracket(*, depth: int = 50) -> bool:
    """Open one continuous S1d.7 tracing window (defensive stop-before-start)."""
    global _tracemalloc_started
    if tracemalloc.is_tracing():
        try:
            tracemalloc.stop()
        except Exception:
            pass
        _tracemalloc_started = False
    tracemalloc.start(int(depth))
    _tracemalloc_started = True
    return True


def end_s1d7_tracemalloc_bracket() -> None:
    """Close the S1d.7 tracing window and clear the module flag."""
    global _tracemalloc_started
    if tracemalloc.is_tracing():
        try:
            tracemalloc.stop()
        except Exception:
            pass
    _tracemalloc_started = False


def reset_tracemalloc_state_for_tests() -> None:
    global _tracemalloc_started
    if _tracemalloc_started:
        try:
            tracemalloc.stop()
        except Exception:
            pass
    _tracemalloc_started = False


def snapshot_tracemalloc(*, top_n: int = TRACEMALLOC_TOP_N) -> dict[str, Any]:
    if not profile_tracemalloc_enabled():
        return {"enabled": False}
    ensure_tracemalloc_started()
    current, peak = tracemalloc.get_traced_memory()
    stats = tracemalloc.take_snapshot()
    top_stats = stats.statistics("traceback")[: int(top_n)]
    frames: list[dict[str, Any]] = []
    for stat in top_stats:
        tb = stat.traceback
        frames.append(
            {
                "size_bytes": int(stat.size),
                "count": int(stat.count),
                "traceback": [str(line) for line in tb.format()[:6]],
                "traceback_key": "|".join(str(line) for line in tb.format()[:3]),
            }
        )
    return {
        "enabled": True,
        "traced_current_bytes": int(current),
        "traced_peak_bytes": int(peak),
        "top_frames": frames,
    }


def _frame_sizes(frames: Sequence[Mapping[str, Any]]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for row in frames:
        key = str(row.get("traceback_key") or "")
        if not key:
            continue
        counter[key] += int(row.get("size_bytes") or 0)
    return counter


def diff_traceback_frames(
    baseline: Mapping[str, Any],
    current: Mapping[str, Any],
    *,
    top_n: int = TRACEMALLOC_TOP_N,
) -> dict[str, Any]:
    base_frames = list(baseline.get("top_frames") or [])
    curr_frames = list(current.get("top_frames") or [])
    base_sizes = _frame_sizes(base_frames)
    curr_sizes = _frame_sizes(curr_frames)
    keys = set(base_sizes) | set(curr_sizes)
    deltas: list[dict[str, Any]] = []
    for key in keys:
        delta = int(curr_sizes.get(key, 0)) - int(base_sizes.get(key, 0))
        if delta <= 0:
            continue
        sample = next((row for row in curr_frames if row.get("traceback_key") == key), None)
        deltas.append(
            {
                "traceback_key": key,
                "delta_bytes": int(delta),
                "traceback": list((sample or {}).get("traceback") or []),
            }
        )
    deltas.sort(key=lambda row: int(row["delta_bytes"]), reverse=True)
    deltas = deltas[: int(top_n)]
    current_delta = int(current.get("traced_current_bytes") or 0) - int(
        baseline.get("traced_current_bytes") or 0
    )
    peak_delta = int(current.get("traced_peak_bytes") or 0) - int(
        baseline.get("traced_current_bytes") or 0
    )
    total_delta = max(int(current_delta), 0)
    for row in deltas:
        row["concentration_fraction"] = (
            float(row["delta_bytes"]) / float(total_delta) if total_delta > 0 else 0.0
        )
    concentrations = sorted(
        [float(row["concentration_fraction"]) for row in deltas],
        reverse=True,
    )
    quantiles: dict[str, float | None] = {
        "p50": concentrations[len(concentrations) // 2] if concentrations else None,
        "p90": concentrations[int(len(concentrations) * 0.9)] if concentrations else None,
        "p99": concentrations[-1] if concentrations else None,
    }
    top_concentration = concentrations[0] if concentrations else 0.0
    return {
        "current_delta_bytes": int(current_delta),
        "peak_delta_bytes": int(peak_delta),
        "top_delta_frames": deltas,
        "quantile_concentration": quantiles,
        "top_concentration_fraction": float(top_concentration),
    }


def classify_branch1_concentration(diff: Mapping[str, Any]) -> bool:
    return float(diff.get("top_concentration_fraction") or 0.0) >= BRANCH1_CONCENTRATION
