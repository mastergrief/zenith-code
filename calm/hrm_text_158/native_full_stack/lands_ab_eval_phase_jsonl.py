"""Enforcer JSONL live-relay + work-enclosing phase brackets (IMPLEMENT_v6/v7).

When SPARSE_LIVE_CARRIER_PHASE_EVENTS_JSONL is set, install tsa._PHASE_EMITTER
writing enforcer-schema lines. phase_start/phase_end bracket REAL work so
duration_s spans the interval (not microsecond empty pairs).
"""
from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from typing import Any, Callable, Iterator

import calm.hrm_text_158.native_full_stack.trainer_sub2_authority as tsa
from calm.hrm_text_158.native_full_stack.lands_ab_eval_schema import PHASE_ORDER

ENV_JSONL = "SPARSE_LIVE_CARRIER_PHASE_EVENTS_JSONL"


def make_enforcer_jsonl_emitter(node_id: str) -> Callable[[str, str], None] | None:
    path = os.environ.get(ENV_JSONL)
    if not path:
        return None
    if not isinstance(node_id, str) or not node_id:
        raise ValueError("node_id must be non-empty str for enforcer JSONL emitter")
    open_starts: dict[str, float] = {}

    def emit(kind: str, phase: str) -> None:
        now = time.monotonic()
        ev: dict[str, Any] = {
            "type": kind,
            "phase": phase,
            "node_id": node_id,
            "ts_monotonic": now,
        }
        if kind in ("PHASE_START", "START"):
            open_starts[phase] = now
        elif kind in ("PHASE_END", "END"):
            t0 = open_starts.pop(phase, now)
            ev["duration_s"] = max(0.0, now - t0)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(ev) + "\n")

    return emit


@contextmanager
def install_enforcer_jsonl_emitter(node_id: str) -> Iterator[Callable[[str, str], None] | None]:
    emit = make_enforcer_jsonl_emitter(node_id)
    if emit is None:
        yield None
        return
    prev = tsa._PHASE_EMITTER
    tsa._PHASE_EMITTER = emit
    try:
        yield emit
    finally:
        tsa._PHASE_EMITTER = prev


def phase_start(
    events: list[dict[str, Any]],
    *,
    phase: str,
    node_id: str,
    open_starts: dict[str, float],
) -> None:
    """START only — call BEFORE work. Records open timestamp for duration."""
    t0 = time.monotonic()
    open_starts[phase] = t0
    events.append(
        {
            "type": "PHASE_START",
            "phase": phase,
            "node_id": node_id,
            "ts_monotonic": t0,
        }
    )
    tsa._emit_phase("PHASE_START", phase)


def phase_end(
    events: list[dict[str, Any]],
    *,
    phase: str,
    node_id: str,
    open_starts: dict[str, float],
) -> None:
    """END only — call AFTER work completes (incl. CUDA sync). duration_s spans work."""
    t1 = time.monotonic()
    t0 = float(open_starts.pop(phase, t1))
    events.append(
        {
            "type": "PHASE_END",
            "phase": phase,
            "node_id": node_id,
            "ts_monotonic": t1,
            "duration_s": max(0.0, t1 - t0),
        }
    )
    tsa._emit_phase("PHASE_END", phase)


def emit_enforcer_phase_pair(
    events: list[dict[str, Any]],
    *,
    phase: str,
    node_id: str,
) -> None:
    """Legacy empty pair (tests only). Prefer phase_start/phase_end around work."""
    open_starts: dict[str, float] = {}
    phase_start(events, phase=phase, node_id=node_id, open_starts=open_starts)
    phase_end(events, phase=phase, node_id=node_id, open_starts=open_starts)


def emit_one_enforcer_cycle_to_memory_and_jsonl(node_id: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with install_enforcer_jsonl_emitter(node_id):
        for phase in PHASE_ORDER:
            emit_enforcer_phase_pair(events, phase=phase, node_id=node_id)
    return events


def emit_work_enclosing_cycle_with_sleep(
    node_id: str, *, work_s: float = 0.05
) -> list[dict[str, Any]]:
    """Characterization: sleep BETWEEN start/end so duration_s includes work."""
    events: list[dict[str, Any]] = []
    open_starts: dict[str, float] = {}
    with install_enforcer_jsonl_emitter(node_id):
        for phase in PHASE_ORDER:
            phase_start(events, phase=phase, node_id=node_id, open_starts=open_starts)
            time.sleep(float(work_s))
            phase_end(events, phase=phase, node_id=node_id, open_starts=open_starts)
    return events


def load_jsonl_events(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    from pathlib import Path

    out: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


@contextmanager
def install_capturing_phase_emitter(
    node_id: str,
    events: list[dict[str, Any]],
    open_starts: dict[str, float],
) -> Iterator[Callable[[str, str], None]]:
    """Install tsa._PHASE_EMITTER that records enforcer-schema events in-memory.

    Used to relay named builders' native _emit_phase stream into one topology
    cycle (IMPLEMENT_v12). Also appends to ENV JSONL when set.
    """
    if not isinstance(node_id, str) or not node_id:
        raise ValueError("node_id must be non-empty str")
    jsonl_path = os.environ.get(ENV_JSONL)

    def emit(kind: str, phase: str) -> None:
        now = time.monotonic()
        k = str(kind)
        ph = str(phase)
        if k in ("PHASE_START", "START"):
            open_starts[ph] = now
            ev: dict[str, Any] = {
                "type": "PHASE_START",
                "phase": ph,
                "node_id": node_id,
                "ts_monotonic": now,
            }
            events.append(ev)
        elif k in ("PHASE_END", "END"):
            t0 = float(open_starts.pop(ph, now))
            ev = {
                "type": "PHASE_END",
                "phase": ph,
                "node_id": node_id,
                "ts_monotonic": now,
                "duration_s": max(0.0, now - t0),
            }
            events.append(ev)
        else:
            return
        if jsonl_path:
            with open(jsonl_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(ev) + "\n")

    prev = tsa._PHASE_EMITTER
    tsa._PHASE_EMITTER = emit
    try:
        yield emit
    finally:
        tsa._PHASE_EMITTER = prev


BUILDER_PHASES: tuple[str, ...] = ("forward_backward", "update", "emission")


def _append_marked_pair(
    events: list[dict[str, Any]],
    *,
    phase: str,
    node_id: str,
    open_starts: dict[str, float],
    synthesized: bool,
    measurement_owned: bool = False,
) -> None:
    """Append START/END marked pairs WITHOUT tsa._emit_phase / JSONL relay.

    Fabricated pairs are always marked; never byte-identical to native captures.
    """
    t0 = time.monotonic()
    open_starts[phase] = t0
    start_ev: dict[str, Any] = {
        "type": "PHASE_START",
        "phase": phase,
        "node_id": node_id,
        "ts_monotonic": t0,
        "synthesized": bool(synthesized),
        "measurement_owned": bool(measurement_owned),
    }
    events.append(start_ev)
    t1 = time.monotonic()
    open_starts.pop(phase, None)
    end_ev: dict[str, Any] = {
        "type": "PHASE_END",
        "phase": phase,
        "node_id": node_id,
        "ts_monotonic": t1,
        "duration_s": max(0.0, t1 - t0),
        "synthesized": bool(synthesized),
        "measurement_owned": bool(measurement_owned),
    }
    events.append(end_ev)



def emit_measurement_owned_flush(
    events: list[dict[str, Any]],
    *,
    node_id: str,
    open_starts: dict[str, float],
    work_fn: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Emit flush START/END to memory AND enforcer JSONL (if env-armed), with work inside.

    measurement_owned=true, synthesized=false. Does NOT use tsa._PHASE_EMITTER so it
    does not depend on an active capturing install; dual-writes ENV JSONL directly.
    """
    if not isinstance(node_id, str) or not node_id:
        raise ValueError("node_id must be non-empty str")
    jsonl_path = os.environ.get(ENV_JSONL)
    phase = "flush"
    t0 = time.monotonic()
    open_starts[phase] = t0
    start_ev: dict[str, Any] = {
        "type": "PHASE_START",
        "phase": phase,
        "node_id": node_id,
        "ts_monotonic": t0,
        "synthesized": False,
        "measurement_owned": True,
    }
    events.append(start_ev)
    if jsonl_path:
        with open(jsonl_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(start_ev) + "\n")
    if work_fn is not None:
        work_fn()
    t1 = time.monotonic()
    open_starts.pop(phase, None)
    end_ev: dict[str, Any] = {
        "type": "PHASE_END",
        "phase": phase,
        "node_id": node_id,
        "ts_monotonic": t1,
        "duration_s": max(0.0, t1 - t0),
        "synthesized": False,
        "measurement_owned": True,
    }
    events.append(end_ev)
    if jsonl_path:
        with open(jsonl_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(end_ev) + "\n")
    return {"start": start_ev, "end": end_ev, "duration_s": float(end_ev["duration_s"])}


def phase_events_have_synthesized(events: list[dict[str, Any]] | None) -> bool:
    return any(bool(e.get("synthesized")) for e in (events or []))


def fold_native_builder_phases_plus_flush(
    events: list[dict[str, Any]],
    *,
    node_id: str,
    open_starts: dict[str, float],
) -> dict[str, Any]:
    """Fold native builder phase stream for topology + science gating (IMPLEMENT_v13).

    - Fully-empty (no builder START): synthesize MARKED (synthesized=true) pairs for
      full PHASE_ORDER transport ONLY. Never relayed via tsa emitter / enforcer JSONL.
    - Partial builder stream (some but not all of FB/update/emission): NO silent
      completion → phase_stream_anomaly=True; science fixture-fail.
    - Complete native builder phases: append measurement-owned flush if missing
      (measurement_owned=true, synthesized=false).
    """
    if not isinstance(node_id, str) or not node_id:
        raise ValueError("node_id must be non-empty str")
    present_starts = {
        str(e.get("phase"))
        for e in events
        if e.get("type") in ("PHASE_START", "START")
    }
    builder_present = present_starts & set(BUILDER_PHASES)
    fully_empty = len(builder_present) == 0
    complete_builder = set(BUILDER_PHASES).issubset(present_starts)
    partial = (not fully_empty) and (not complete_builder)

    synthesized_phases: list[str] = []
    phase_stream_anomaly = bool(partial)
    needs_measurement_flush = False
    if fully_empty:
        # empty/exception path only — marked synthesis for transport (no JSONL)
        for phase in PHASE_ORDER:
            if phase not in present_starts:
                _append_marked_pair(
                    events,
                    phase=phase,
                    node_id=node_id,
                    open_starts=open_starts,
                    synthesized=True,
                    measurement_owned=False,
                )
                synthesized_phases.append(phase)
        phase_stream_class = "empty_synthesized_transport"
    elif partial:
        # do NOT complete missing builder phases
        phase_stream_class = "partial_native_anomaly"
    else:
        # complete native builder stream — flush emitted separately with real work
        phase_stream_class = "native_complete"
        needs_measurement_flush = "flush" not in present_starts

    return {
        "events": events,
        "phase_stream_class": phase_stream_class,
        "phase_stream_anomaly": phase_stream_anomaly,
        "phase_events_synthesized": bool(synthesized_phases)
        or phase_events_have_synthesized(events),
        "synthesized_phases": synthesized_phases,
        "builder_phases_present": sorted(builder_present),
        "needs_measurement_flush": bool(needs_measurement_flush),
    }


def apply_phase_stream_science_gate(obs: dict[str, Any], *, gating_row: str) -> dict[str, Any]:
    """Fail-closed science polarity when phase events are synthesized or partial.

    Pure observation fold used by evidence consumer after topology classify.
    Does not invent events; only re-derives surfaces under fixture_fail.
    """
    from calm.hrm_text_158.native_full_stack.lands_ab_eval_metric_reducer import (
        recompute_surface_cells_from_primitives,
    )
    from calm.hrm_text_158.native_full_stack.lands_ab_eval_schema import APPLICABILITY_MAP

    events = list(obs.get("phase_events") or [])
    synthesized = any(bool(e.get("synthesized")) for e in events)
    builder_starts = {
        str(e.get("phase"))
        for e in events
        if e.get("type") in ("PHASE_START", "START")
        and str(e.get("phase")) in BUILDER_PHASES
    }
    partial = bool(builder_starts) and builder_starts != set(BUILDER_PHASES)
    anomaly = bool(obs.get("phase_stream_anomaly")) or partial
    out = dict(obs)
    out["phase_events_synthesized"] = bool(
        out.get("phase_events_synthesized") or synthesized
    )
    out["phase_stream_anomaly"] = anomaly
    topo = out.get("phase_topology") or {}
    if (
        topo.get("good_topology") is not True
        or synthesized
        or anomaly
    ):
        out["fixture_contract_raw_fail"] = True
        ku = [str(k) for k in (out.get("key_universe") or [])]
        try:
            out["measured_surfaces"] = recompute_surface_cells_from_primitives(
                gating_row=gating_row,
                metrics=dict(out.get("metrics") or {}),
                key_universe=ku,
                fixture_contract_raw_fail=True,
            )
        except Exception:
            out["measured_surfaces"] = {
                s: False for s in APPLICABILITY_MAP.get(gating_row, ())
            }
    return out
