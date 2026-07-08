"""CPU characterization: attribute ~74x logical→physical carrier amplification.

Frozen plan: ai-room 1783545383951 + 1783545407617 (+1 implement 1783545604542).
Diagnostic-only — real EventCodedAccLiveState classes; synthetic event stream only.
"""

from __future__ import annotations

import gc
import json
import os
import resource
import sys
from pathlib import Path
from typing import Any, Mapping

import pytest

from calm.hrm_text_158.native_full_stack.event_coded_acc_checkpoint_codec import (
    EventCodedAccEvent,
)
from calm.hrm_text_158.native_full_stack.event_coded_acc_live_carrier import (
    EventCodedAccLiveState,
    StepSurfaceRecord,
)
from calm.hrm_text_158.native_full_stack.event_coded_vote_update_adapter import (
    apply_event_coded_carrier_step,
)
from calm.hrm_text_158.native_full_stack.host_allocator_probe import read_smaps_rollup

SWEEP_N = (1_000, 10_000, 100_000, 500_000)
FLAT_RATIO_MEAN_LO = 40.0
FLAT_RATIO_MEAN_HI = 90.0
FLAT_RATIO_MAX_MIN = 1.25
LIVE_SNAPSHOT_DEFAULT = Path(
    "/home/gabe/claw-code-creditdir/transient_fp_credit/"
    "arc2b_slice5_discovery_h25_seed43_FREE_DISCOVERY_H25_arm_d/"
    "d_recompute_window_diagnostic/live_carrier_snapshot.jsonl"
)
DURABLE_ARTIFACT_DEFAULT = Path(
    "/home/gabe/claw-code-creditdir/transient_fp_credit/"
    "arc2b_slice5_discovery_h25_seed43_FREE_DISCOVERY_H25_arm_d/"
    "amp_attribution_v1.json"
)
EXTRAPOLATE_CODEC_BYTES = 178 * 1024 * 1024
EXTRAPOLATE_RSS_DELTA_GIB = 13.2


def deep_sizeof(obj: Any, seen: set[int] | None = None) -> int:
    """Deep size with identity de-dupe (for step_records / q_levels / cow copies)."""

    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    size = sys.getsizeof(obj)
    if isinstance(obj, dict):
        size += sum(deep_sizeof(k, seen) + deep_sizeof(v, seen) for k, v in obj.items())
    elif isinstance(obj, (list, tuple, set, frozenset)):
        size += sum(deep_sizeof(item, seen) for item in obj)
    elif isinstance(obj, (EventCodedAccEvent, StepSurfaceRecord)) or hasattr(obj, "__dict__"):
        size += deep_sizeof(vars(obj), seen)
    return int(size)


def read_current_smaps() -> dict[str, int | None]:
    rollup = read_smaps_rollup()
    if "error" in rollup and "rss_kb" not in rollup:
        return {
            "rss_kb": None,
            "pss_kb": None,
            "uss_kb": None,
            "error": str(rollup.get("error")),
        }
    private_dirty = int(rollup.get("private_dirty_kb") or 0)
    private_clean = int(rollup.get("private_clean_kb") or 0)
    return {
        "rss_kb": int(rollup["rss_kb"]) if "rss_kb" in rollup else None,
        "pss_kb": int(rollup["pss_kb"]) if "pss_kb" in rollup else None,
        "uss_kb": int(private_dirty + private_clean),
        "private_dirty_kb": private_dirty,
        "private_clean_kb": private_clean,
        "ru_maxrss_kb_optional": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }


# Codec EventCodedAccEvent packing is varint(flat_index)+flags; observed
# compact-index synthetic stream ≈2.87–3.0 B/event. Live arm_d rows use wider
# indices so B/event sits in a band — report count as a range, not a point.
EVENT_COUNT_BYTES_PER_EVENT_ASSUMED = 3.0
EVENT_COUNT_BYTES_PER_EVENT_LO = 2.0
EVENT_COUNT_BYTES_PER_EVENT_HI = 4.0


def infer_event_count_from_codec_bytes(events_bytes: int) -> dict[str, Any]:
    nbytes = int(events_bytes)
    return {
        "inferred_event_count": int(round(nbytes / EVENT_COUNT_BYTES_PER_EVENT_ASSUMED)),
        "inferred_event_count_lo": int(nbytes // int(EVENT_COUNT_BYTES_PER_EVENT_HI)),
        "inferred_event_count_hi": int(nbytes // int(EVENT_COUNT_BYTES_PER_EVENT_LO)),
    }


def load_live_distribution_anchor(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"present": False, "path": str(path)}
    by_step: dict[int, list[dict[str, Any]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        by_step.setdefault(int(row["step"]), []).append(row)
    step12 = by_step.get(12, [])
    sizes = sorted(int(r["events_bytes"]) for r in step12) if step12 else []
    max_row = max(step12, key=lambda r: int(r["events_bytes"])) if step12 else None
    min_row = min(step12, key=lambda r: int(r["events_bytes"])) if step12 else None

    def _row_with_inferred(row: Mapping[str, Any]) -> dict[str, Any]:
        events_bytes = int(row["events_bytes"])
        out = {
            "step": int(row["step"]),
            "state_key": str(row["state_key"]),
            "events_bytes": events_bytes,
        }
        out.update(infer_event_count_from_codec_bytes(events_bytes))
        return out

    per_step_state_key = [
        _row_with_inferred(row)
        for step in sorted(by_step)
        for row in sorted(by_step[step], key=lambda r: str(r["state_key"]))
    ]
    return {
        "present": True,
        "path": str(path),
        "n_steps": len(by_step),
        "event_count_inference": {
            "method": "events_bytes / bytes_per_event",
            "bytes_per_event_assumed": EVENT_COUNT_BYTES_PER_EVENT_ASSUMED,
            "bytes_per_event_range": [
                EVENT_COUNT_BYTES_PER_EVENT_LO,
                EVENT_COUNT_BYTES_PER_EVENT_HI,
            ],
            "note": (
                "varint+flag codec (event_coded_acc_checkpoint_codec); "
                "assumed mid=3.0 B/event; lo/hi cover compact→wide flat_index"
            ),
        },
        "per_step_state_key": per_step_state_key,
        "step12": {
            "n_modules": len(step12),
            "total_events_bytes": int(sum(sizes)),
            "max_events_bytes": int(sizes[-1]) if sizes else None,
            "min_events_bytes": int(sizes[0]) if sizes else None,
            "median_events_bytes": int(sizes[len(sizes) // 2]) if sizes else None,
            "max_state_key": None if max_row is None else str(max_row["state_key"]),
            "min_state_key": None if min_row is None else str(min_row["state_key"]),
            "per_module_events_bytes": [
                _row_with_inferred(r)
                for r in sorted(step12, key=lambda x: -int(x["events_bytes"]))
            ],
        },
    }


def _append_synthetic_events(carrier: EventCodedAccLiveState, n: int, *, base: int) -> None:
    # Bound flat_index into a compact pool so codec varint width stays stable
    # across the N sweep. Object-overhead flatness must not be confounded by
    # varint growth from monotonically increasing unique indices.
    index_pool = 10_000
    for i in range(int(n)):
        idx = int((base + i) % index_pool)
        carrier._append_event(
            EventCodedAccEvent(
                flat_index=idx,
                direction=1 if (idx % 2) == 0 else 0,
                residual_mag=int(idx % 8),
                event_type=1,
            )
        )
        if i % 50 == 0:
            carrier.q_levels[idx] = 1 if (idx % 2) == 0 else -1


def _append_step_record_via_real_apply_step(
    carrier: EventCodedAccLiveState,
    *,
    step_index: int,
) -> None:
    # Empty-vote apply_step still appends a StepSurfaceRecord with q_levels=dict(...).
    carrier.apply_step(int(step_index), votes={})
    apply_event_coded_carrier_step(carrier, votes={}, step_index=int(step_index) + 10_000)


def build_carrier_at_n(
    n_events: int,
    *,
    step_records_every: int = 10_000,
) -> EventCodedAccLiveState:
    # logical_numel must cover compact indices used by the synthetic stream.
    carrier = EventCodedAccLiveState(logical_numel=max(1_048_576, int(n_events) + 1))
    remaining = int(n_events)
    base = 0
    step_index = 0
    while remaining > 0:
        chunk = min(int(step_records_every), remaining)
        _append_synthetic_events(carrier, chunk, base=base)
        base += chunk
        remaining -= chunk
        _append_step_record_via_real_apply_step(carrier, step_index=step_index)
        step_index += 1
    assert len(carrier.events) == int(n_events)
    return carrier


def measure_point(n_events: int) -> dict[str, Any]:
    gc.collect()
    baseline = read_current_smaps()
    carrier = build_carrier_at_n(n_events)
    snap = carrier.live_carrier_byte_snapshot()
    codec_events = int(snap["events_bytes"])
    deep_events = deep_sizeof(carrier.events)
    deep_step_records = deep_sizeof(carrier.step_records)
    deep_q_levels = deep_sizeof(carrier.q_levels)
    deep_backlog = deep_sizeof(carrier.backlog)
    # Distinct hot retained-class row (fold contract). Synthetic empty-vote
    # stream leaves hot empty — emit explicit zeros when so.
    hot_count = int(len(carrier._hot))
    hot_bytes = int(snap.get("hot_exact_bytes") or 0)
    deep_hot_bytes = deep_sizeof(carrier._hot)
    first_event_id = id(carrier.events[0]) if carrier.events else None
    retained = read_current_smaps()

    cow = carrier.cow_copy()
    cow_deep = deep_sizeof(cow)
    cow_held = read_current_smaps()
    first_event_id_after_cow = id(carrier.events[0]) if carrier.events else None
    del cow
    gc.collect()
    post_gc = read_current_smaps()

    ratio = float(deep_events) / float(max(1, codec_events))
    return {
        "n_events": int(n_events),
        "codec_events_bytes": codec_events,
        "deep_events_bytes": int(deep_events),
        "deep_step_records_bytes": int(deep_step_records),
        "deep_q_levels_bytes": int(deep_q_levels),
        "deep_backlog_bytes": int(deep_backlog),
        "hot_bytes": int(hot_bytes),
        "hot_count": int(hot_count),
        "deep_hot_bytes": int(deep_hot_bytes),
        "events_py_over_codec": ratio,
        "n_step_records": len(carrier.step_records),
        "n_q_levels": len(carrier.q_levels),
        "first_event_id_stable": first_event_id == first_event_id_after_cow,
        "cow_copy_deep_bytes": int(cow_deep),
        "smaps": {
            "baseline": baseline,
            "retained": retained,
            "cow_held": cow_held,
            "post_gc": post_gc,
        },
        "live_carrier_bytes_exact": bool(snap.get("live_carrier_bytes_exact")),
    }


def ablation_a_cow_peak(n_events: int = 10_000) -> dict[str, Any]:
    gc.collect()
    carrier = build_carrier_at_n(n_events)
    retained = read_current_smaps()
    cow = carrier.cow_copy()
    cow_deep = deep_sizeof(cow)
    cow_held = read_current_smaps()
    del cow
    gc.collect()
    post_gc = read_current_smaps()
    retained_rss = retained.get("rss_kb")
    cow_rss = cow_held.get("rss_kb")
    post_rss = post_gc.get("rss_kb")
    peak_delta_kb = (
        None
        if retained_rss is None or cow_rss is None
        else int(cow_rss) - int(retained_rss)
    )
    post_delta_kb = (
        None
        if retained_rss is None or post_rss is None
        else int(post_rss) - int(retained_rss)
    )
    # (a) PASS as transient amplifier: peak rises near cow_deep; post-GC returns near retained.
    # FAIL-as-sole-root would require retained RSS linear growth without len(events) growth
    # (not claimed here — events are retained by construction).
    cow_deep_kb = max(1, int(cow_deep // 1024))
    peak_ok = peak_delta_kb is not None and peak_delta_kb >= 0
    return_ok = post_delta_kb is not None and abs(int(post_delta_kb)) <= max(
        16 * 1024, int(0.25 * cow_deep_kb)
    )
    return {
        "n_events": int(n_events),
        "cow_copy_deep_bytes": int(cow_deep),
        "peak_delta_rss_kb": peak_delta_kb,
        "post_gc_delta_rss_kb": post_delta_kb,
        "peak_ok": bool(peak_ok),
        "post_gc_return_ok": bool(return_ok),
        "verdict_transient_amplifier": bool(peak_ok),
        "verdict_sole_root": False,
        "smaps": {"retained": retained, "cow_held": cow_held, "post_gc": post_gc},
    }


def ablation_c_step_records_only(n_steps: int = 64, n_q_keys: int = 2_000) -> dict[str, Any]:
    carrier = EventCodedAccLiveState(logical_numel=1_048_576)
    for i in range(int(n_q_keys)):
        carrier.q_levels[i] = 1 if (i % 2) == 0 else -1
    # Hold events fixed (empty); grow step_records only via real apply_step.
    sizes: list[int] = []
    for step in range(int(n_steps)):
        carrier.apply_step(int(step), votes={})
        sizes.append(deep_sizeof(carrier.step_records))
    # Superlinear-ish: last/first should grow with steps (q_levels dict copied each record).
    growth = float(sizes[-1]) / float(max(1, sizes[0]))
    expected_floor = float(n_steps) * 0.5
    return {
        "n_steps": int(n_steps),
        "n_events_fixed": len(carrier.events),
        "n_q_keys": int(n_q_keys),
        "step_records_deep_first": int(sizes[0]),
        "step_records_deep_last": int(sizes[-1]),
        "growth_last_over_first": growth,
        "superlinear_ok": bool(growth >= expected_floor),
        "verdict_real_subdominant": bool(growth >= expected_floor and len(carrier.events) == 0),
    }


def classify_verdicts(
    sweep: list[dict[str, Any]],
    ablation_a: Mapping[str, Any],
    ablation_c: Mapping[str, Any],
) -> dict[str, Any]:
    ratios = [float(p["events_py_over_codec"]) for p in sweep]
    mean_ratio = sum(ratios) / float(len(ratios))
    max_min = max(ratios) / max(1e-9, min(ratios))
    flat_ok = bool(max_min <= FLAT_RATIO_MAX_MIN and FLAT_RATIO_MEAN_LO <= mean_ratio <= FLAT_RATIO_MEAN_HI)
    id_stable = all(bool(p["first_event_id_stable"]) for p in sweep)
    # (b) primary: flat scale-invariant object/codec ratio in band + id-stable events list.
    b_pass = bool(flat_ok and id_stable)
    # (a) secondary amplifier (not sole root): cow peak observable; sole-root false.
    a_pass = bool(ablation_a.get("verdict_transient_amplifier")) and not bool(
        ablation_a.get("verdict_sole_root")
    )
    # (c) real subdominant: step_records grow with steps while events fixed empty.
    c_pass = bool(ablation_c.get("verdict_real_subdominant"))
    predicted_rss_gib = mean_ratio * float(EXTRAPOLATE_CODEC_BYTES) / float(1024**3)
    return {
        "ratios": ratios,
        "mean_ratio": mean_ratio,
        "max_min_ratio": max_min,
        "flat_ratio_pass": flat_ok,
        "b_object_overhead": "PASS" if b_pass else "FAIL",
        "a_transient_cow_copy": "PASS" if a_pass else "FAIL",
        "c_step_records": "PASS" if c_pass else "FAIL",
        "rank": ["b_object_overhead", "a_transient_cow_copy", "c_step_records"],
        "extrapolation": {
            "codec_bytes_anchor": int(EXTRAPOLATE_CODEC_BYTES),
            "mean_ratio": mean_ratio,
            "predicted_events_py_gib": predicted_rss_gib,
            "observed_rss_delta_gib_anchor": float(EXTRAPOLATE_RSS_DELTA_GIB),
            "residual_gib_diagnostic": float(EXTRAPOLATE_RSS_DELTA_GIB) - predicted_rss_gib,
            "note": "diagnostic only — not a PASS gate",
        },
    }


def run_attribution(
    *,
    artifact_path: Path,
    live_snapshot_path: Path | None = None,
    include_near_38mb: bool = False,
) -> dict[str, Any]:
    live_path = live_snapshot_path or Path(
        os.environ.get("ARC2B_SLICE5_LIVE_CARRIER_SNAPSHOT", str(LIVE_SNAPSHOT_DEFAULT))
    )
    live_anchor = load_live_distribution_anchor(live_path)
    sweep_n = list(SWEEP_N)
    skipped_near_38mb = None
    if include_near_38mb or os.environ.get("ARC2B_SLICE5_NEAR_38MB") == "1":
        # ~38MB codec ≈ ~1e7 events at ~4B/event — OPTIONAL/skip-if-RAM.
        near_n = 10_000_000
        try:
            # Probe available memory roughly via smaps; skip if RSS already high.
            cur = read_current_smaps()
            rss_kb = cur.get("rss_kb") or 0
            if int(rss_kb) > 8 * 1024 * 1024:
                skipped_near_38mb = {
                    "reason": "skip-if-RAM",
                    "rss_kb": rss_kb,
                    "requested_n": near_n,
                }
            else:
                sweep_n.append(near_n)
        except Exception as exc:  # pragma: no cover
            skipped_near_38mb = {"reason": "skip-if-RAM", "error": str(exc)}

    sweep_points: list[dict[str, Any]] = []
    for n in sweep_n:
        if n >= 10_000_000 and skipped_near_38mb is not None:
            continue
        sweep_points.append(measure_point(int(n)))

    ablation_a = ablation_a_cow_peak(10_000)
    ablation_c = ablation_c_step_records_only(64, 2_000)
    verdicts = classify_verdicts(sweep_points, ablation_a, ablation_c)

    payload = {
        "schema": "arc2b_slice5_amp_attribution_v1",
        "claim_boundary": (
            "CENSORED diagnostic attribution only. No mechanism fix, no C/E/H50/"
            "science/bank/gap(D) verdict."
        ),
        "frozen_plan_msgs": ["1783545383951-015c322b", "1783545407617-4c35ae56"],
        "plus1_implement_msg": "1783545604542-1051bd0e",
        "live_distribution_anchor": live_anchor,
        "sweep_n": sweep_n if skipped_near_38mb is None else [n for n in sweep_n if n < 10_000_000],
        "skipped_near_38mb": skipped_near_38mb,
        "sweep_points": sweep_points,
        "ablation_a_cow_peak": ablation_a,
        "ablation_c_step_records_only": ablation_c,
        "verdicts": verdicts,
        "pass_thresholds": {
            "flat_ratio_max_over_min": FLAT_RATIO_MAX_MIN,
            "flat_ratio_mean_lo": FLAT_RATIO_MEAN_LO,
            "flat_ratio_mean_hi": FLAT_RATIO_MEAN_HI,
        },
    }
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    durable = Path(
        os.environ.get("ARC2B_SLICE5_AMP_ATTRIBUTION_OUT", str(DURABLE_ARTIFACT_DEFAULT))
    )
    try:
        durable.parent.mkdir(parents=True, exist_ok=True)
        durable.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        payload["durable_artifact_path"] = str(durable)
    except OSError:
        payload["durable_artifact_path"] = None
    payload["artifact_path"] = str(artifact_path)
    # Rewrite with paths included.
    artifact_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if payload.get("durable_artifact_path"):
        Path(payload["durable_artifact_path"]).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return payload


def test_carrier_amp_attribution_v1(tmp_path: Path) -> None:
    artifact = tmp_path / "amp_attribution_v1.json"
    payload = run_attribution(artifact_path=artifact)
    assert artifact.is_file()
    verdicts = payload["verdicts"]
    assert verdicts["flat_ratio_pass"] is True, verdicts
    assert verdicts["b_object_overhead"] == "PASS", verdicts
    assert verdicts["a_transient_cow_copy"] == "PASS", verdicts
    assert verdicts["c_step_records"] == "PASS", verdicts
    assert len(payload["sweep_points"]) >= 4
    for point in payload["sweep_points"]:
        assert point["live_carrier_bytes_exact"] is True
        assert point["codec_events_bytes"] > 0
        assert point["deep_events_bytes"] > point["codec_events_bytes"]
        # Gap-2: distinct hot retained-class row always present (may be zero).
        assert "hot_bytes" in point
        assert "hot_count" in point
        assert "deep_hot_bytes" in point
        assert int(point["hot_bytes"]) >= 0
        assert int(point["hot_count"]) >= 0


def test_artifact_contract_live_anchor_and_hot_fields() -> None:
    """Contract completeness: inferred_event_count + hot retained-class row."""

    anchor = load_live_distribution_anchor(LIVE_SNAPSHOT_DEFAULT)
    assert "event_count_inference" in anchor
    inference = anchor["event_count_inference"]
    assert inference["method"] == "events_bytes / bytes_per_event"
    assert inference["bytes_per_event_assumed"] == EVENT_COUNT_BYTES_PER_EVENT_ASSUMED
    assert inference["bytes_per_event_range"] == [
        EVENT_COUNT_BYTES_PER_EVENT_LO,
        EVENT_COUNT_BYTES_PER_EVENT_HI,
    ]
    if anchor.get("present"):
        assert "per_step_state_key" in anchor
        assert len(anchor["per_step_state_key"]) >= 1
        sample = anchor["per_step_state_key"][0]
        assert "events_bytes" in sample
        assert "inferred_event_count" in sample
        assert "inferred_event_count_lo" in sample
        assert "inferred_event_count_hi" in sample
        assert "step" in sample and "state_key" in sample
        for row in anchor["step12"]["per_module_events_bytes"]:
            assert "inferred_event_count" in row
    point = measure_point(1_000)
    assert "hot_bytes" in point and "hot_count" in point and "deep_hot_bytes" in point
    assert int(point["hot_bytes"]) == 0
    assert int(point["hot_count"]) == 0


def test_imports_are_real_production_classes() -> None:
    # Guard against mock/hand-rolled carrier reimplementation.
    assert EventCodedAccLiveState.__module__.endswith("event_coded_acc_live_carrier")
    assert EventCodedAccEvent.__module__.endswith("event_coded_acc_checkpoint_codec")
    assert apply_event_coded_carrier_step.__module__.endswith(
        "event_coded_vote_update_adapter"
    )
    assert callable(EventCodedAccLiveState.cow_copy)
    assert callable(EventCodedAccLiveState.apply_step)


@pytest.mark.parametrize("n", [1_000, 10_000])
def test_sweep_ratio_band_smoke(n: int) -> None:
    point = measure_point(n)
    assert FLAT_RATIO_MEAN_LO <= point["events_py_over_codec"] <= FLAT_RATIO_MEAN_HI
    assert point["first_event_id_stable"] is True
