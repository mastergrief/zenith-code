"""PURE phase topology reducer — enforcer-compatible state machine (IMPLEMENT_v5).

Matches scripts/sparse_live_carrier_gpu_phase_budget_enforcer.py acceptance rules:
adjacent START→matching-END, no nested/open phases, known types/phases only,
node_id non-empty str, numeric finite monotonic ts_monotonic, duration_s >= 0 on END.
Zero IO / GPU / launch imports.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from calm.hrm_text_158.native_full_stack.lands_ab_eval_schema import PHASE_ORDER

ALLOWED_TYPES = frozenset({"PHASE_START", "PHASE_END", "START", "END"})


def _norm_type(kind: str) -> str | None:
    if kind in ("PHASE_START", "START"):
        return "PHASE_START"
    if kind in ("PHASE_END", "END"):
        return "PHASE_END"
    return None


def classify_phase_topology(
    events: Sequence[Mapping[str, Any]],
    *,
    expected_node_id: str | None = None,
    require_enforcer_fields: bool = False,
) -> dict[str, Any]:
    """Classify topology. require_enforcer_fields=True for formal CUDA rows."""
    starts: list[str] = []
    ends: list[str] = []
    errors: list[str] = []
    open_phase: str | None = None
    last_ts: float | None = None
    seen_start_set: set[str] = set()
    seen_end_set: set[str] = set()
    observed_nodes: list[str] = []

    for i, e in enumerate(events):
        kind_raw = str(e.get("type") or e.get("kind") or "")
        etype = _norm_type(kind_raw)
        if etype is None:
            errors.append(f"unknown or missing type {kind_raw!r}")
            continue
        phase = str(e.get("phase") or "")
        if phase not in PHASE_ORDER:
            errors.append(f"unknown phase {phase!r}")
            continue

        if require_enforcer_fields or expected_node_id is not None or "node_id" in e or "ts_monotonic" in e:
            nid = e.get("node_id")
            if not isinstance(nid, str) or not nid:
                errors.append("node_id must be non-empty str")
            else:
                observed_nodes.append(nid)
                if expected_node_id is not None and nid != expected_node_id:
                    errors.append(f"node_id mismatch {nid} != {expected_node_id}")
            ts = e.get("ts_monotonic")
            if not isinstance(ts, (int, float)) or isinstance(ts, bool) or not math.isfinite(float(ts)):
                errors.append("ts_monotonic must be numeric finite")
            else:
                tsf = float(ts)
                if last_ts is not None and tsf < last_ts:
                    errors.append("non-monotonic ts_monotonic")
                last_ts = tsf
            if etype == "PHASE_END":
                dur = e.get("duration_s")
                if not isinstance(dur, (int, float)) or isinstance(dur, bool) or not math.isfinite(float(dur)) or float(dur) < 0:
                    errors.append("duration_s must be numeric finite >= 0")

        if etype == "PHASE_START":
            if open_phase is not None:
                errors.append(f"nested/unpaired START while open={open_phase}")
            if phase in seen_start_set:
                errors.append(f"duplicate START {phase}")
            seen_start_set.add(phase)
            starts.append(phase)
            open_phase = phase
        else:  # END
            if open_phase != phase:
                errors.append(f"END without matching START phase={phase} open={open_phase}")
            if phase in seen_end_set:
                errors.append(f"duplicate END {phase}")
            seen_end_set.add(phase)
            ends.append(phase)
            open_phase = None

    if open_phase is not None:
        errors.append(f"unclosed phase open={open_phase}")
    if starts != list(PHASE_ORDER):
        errors.append(f"missing phase coverage starts={starts}")
    if ends != list(PHASE_ORDER):
        errors.append(f"missing phase coverage ends={ends}")
    # extra events beyond the 8-event cycle
    if len(events) > 8:
        errors.append(f"extra events count={len(events)}")

    if errors:
        cls = "PHASE_TELEMETRY"
        if any("duplicate START" in x for x in errors):
            detail = "duplicate_start"
        elif any("nested" in x for x in errors):
            detail = "nested_start"
        elif any("missing phase coverage" in x for x in errors):
            detail = "missing_coverage"
        elif any("unknown or missing type" in x for x in errors):
            detail = "unknown_type"
        elif any("non-monotonic" in x for x in errors):
            detail = "nonmonotonic_ts"
        elif any("duration_s" in x for x in errors):
            detail = "bad_duration"
        elif any("node_id" in x for x in errors):
            detail = "bad_node_id"
        else:
            detail = "phase_telemetry"
    else:
        cls = "OK"
        detail = "good_topology"
    return {
        "terminal_class": cls,
        "detail": detail,
        "errors": errors,
        "starts": starts,
        "ends": ends,
        "good_topology": cls == "OK",
        "observed_node_ids": sorted(set(observed_nodes)),
    }


def synthesize_good_topology_events(
    *, node_id: str = "node", base_ts: float = 1000.0
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    t = base_ts
    for phase in PHASE_ORDER:
        out.append(
            {
                "type": "PHASE_START",
                "phase": phase,
                "node_id": node_id,
                "ts_monotonic": t,
            }
        )
        t += 0.01
        out.append(
            {
                "type": "PHASE_END",
                "phase": phase,
                "node_id": node_id,
                "ts_monotonic": t,
                "duration_s": 0.01,
            }
        )
        t += 0.01
    return out


def synthesize_duplicate_start_events(
    *, node_id: str = "node", base_ts: float = 1000.0
) -> list[dict[str, Any]]:
    out = synthesize_good_topology_events(node_id=node_id, base_ts=base_ts)
    # insert duplicate START for forward_backward after first pair (enforcer self-test shape)
    out.insert(
        2,
        {
            "type": "PHASE_START",
            "phase": "forward_backward",
            "node_id": node_id,
            "ts_monotonic": base_ts + 0.005,
        },
    )
    return out


def synthesize_missing_coverage_events(
    *, node_id: str = "node", base_ts: float = 1000.0
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    t = base_ts
    for phase in ("forward_backward", "update", "emission"):
        out.append({"type": "PHASE_START", "phase": phase, "node_id": node_id, "ts_monotonic": t})
        t += 0.01
        out.append(
            {
                "type": "PHASE_END",
                "phase": phase,
                "node_id": node_id,
                "ts_monotonic": t,
                "duration_s": 0.01,
            }
        )
        t += 0.01
    return out


def synthesize_nested_start_events(
    *, node_id: str = "node", base_ts: float = 1000.0
) -> list[dict[str, Any]]:
    """P3: START forward_backward then START update without END — nested open."""
    return [
        {
            "type": "PHASE_START",
            "phase": "forward_backward",
            "node_id": node_id,
            "ts_monotonic": base_ts,
        },
        {
            "type": "PHASE_START",
            "phase": "update",
            "node_id": node_id,
            "ts_monotonic": base_ts + 0.01,
        },
    ]


def topology_is_complete(events: Sequence[Mapping[str, Any]], **kwargs: Any) -> bool:
    return classify_phase_topology(events, **kwargs)["good_topology"] is True
