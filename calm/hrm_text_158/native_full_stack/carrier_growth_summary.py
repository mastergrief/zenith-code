"""Carrier growth summary sidecar (Phase 0 stub + Phase 1 production wiring)."""
from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from calm.hrm_text_158.native_full_stack.bounded_delta_learner import BoundedDeltaTensorState
from calm.hrm_text_158.native_full_stack.event_coded_acc_checkpoint_codec import (
    EventCodedAccEvent,
)
from calm.hrm_text_158.native_full_stack.event_coded_acc_live_carrier import (
    EventCodedAccLiveState,
    StepSurfaceRecord,
    promotion_carry_threshold,
)
from calm.hrm_text_158.native_full_stack.event_coded_vote_update_adapter import (
    EventCodedVoteUpdateState,
    _active_lane_indices,
)
from calm.hrm_text_158.native_full_stack.sparse_vote_inputs_svp1 import (
    write_sidecar_atomically,
)

CARRIER_GROWTH_SUMMARY_SCHEMA_VERSION = "hrm_text_158_carrier_growth_summary/v0"
EST_BYTES_PER_EVENT = 4
EST_BYTES_PER_HOT_ROW = 5
HOT_CARRY_BUCKET_KEYS: tuple[str, ...] = (
    "0",
    "1",
    "2_3",
    "4_6",
    "7_9",
    "10_plus",
)
COMPACT_ROLLUP_KEYS: frozenset[str] = frozenset(
    {
        "module_count",
        "event_count_after",
        "hot_exact_row_count_after",
        "backlog_count_after",
        "new_crossing_events",
        "events_on_q_locked_not_hot",
        "est_events_payload_bytes",
        "est_hot_exact_payload_bytes",
        "est_saved_bytes_v5_clear",
        "est_saved_bytes_v2_coalesce",
        "sidecar_bytes",
    }
)
DEFAULT_R4V_METADATA_BYTES = 768


def estimate_events_payload_bytes(*, event_count: int) -> int:
    """Rolling O(1) estimate — NOT full encode_event_coded_acc_events."""

    return int(max(0, int(event_count))) * EST_BYTES_PER_EVENT


def estimate_hot_exact_payload_bytes(*, hot_row_count: int) -> int:
    """Rolling O(1) estimate — NOT carrier.hot_packed_bytes()."""

    return int(max(0, int(hot_row_count))) * EST_BYTES_PER_HOT_ROW


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def sidecar_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _event_dup_summary(events: Sequence[EventCodedAccEvent]) -> dict[str, int]:
    if not events:
        return {
            "event_dup_lanes_gt1": 0,
            "event_dup_max_per_lane": 0,
            "event_dup_p95_per_lane": 0,
        }
    counts = Counter(int(event.flat_index) for event in events)
    per_lane = sorted(counts.values())
    gt1 = sum(1 for value in per_lane if value > 1)
    p95_index = max(0, int(round(0.95 * (len(per_lane) - 1))))
    return {
        "event_dup_lanes_gt1": int(gt1),
        "event_dup_max_per_lane": int(max(per_lane)),
        "event_dup_p95_per_lane": int(per_lane[p95_index]),
    }


def _events_on_q_locked_not_hot_count(carrier: EventCodedAccLiveState) -> int:
    hot = {int(i) for i in carrier._hot.indices_array().tolist()}
    q_locked = set(int(i) for i in carrier.q_levels)
    total = 0
    for event in carrier.events:
        idx = int(event.flat_index)
        if idx in q_locked and idx not in hot:
            total += 1
    return int(total)


def _estimate_v2_coalesce_saved_bytes(*, event_count: int, unique_lane_count: int) -> int:
    removed = max(0, int(event_count) - int(unique_lane_count))
    return int(removed) * EST_BYTES_PER_EVENT


def _hot_carry_bucket_counts(carrier: EventCodedAccLiveState) -> dict[str, int]:
    buckets = {key: 0 for key in HOT_CARRY_BUCKET_KEYS}
    values = carrier._hot.values_array()
    for raw in values.tolist():
        magnitude = abs(int(raw))
        if magnitude == 0:
            buckets["0"] += 1
        elif magnitude == 1:
            buckets["1"] += 1
        elif magnitude <= 3:
            buckets["2_3"] += 1
        elif magnitude <= 6:
            buckets["4_6"] += 1
        elif magnitude <= 9:
            buckets["7_9"] += 1
        else:
            buckets["10_plus"] += 1
    return buckets


def _hot_rows_vote_touched(carrier: EventCodedAccLiveState, votes: torch.Tensor) -> int:
    if votes.numel() == 0:
        return 0
    hot = {int(i) for i in carrier._hot.indices_array().tolist()}
    vote_flat = votes.detach().cpu().flatten()
    touched = {int(i) for i in torch.nonzero(vote_flat != 0, as_tuple=False).flatten().tolist()}
    return int(len(hot & touched))


def _hot_rows_in_proxy(
    carrier: EventCodedAccLiveState,
    votes: torch.Tensor,
) -> int:
    del votes  # hot-table carry is the hot-exact proxy authority at emit time.
    hot_values = carrier._hot.values_array()
    if hot_values.size == 0:
        return 0
    promote_at = promotion_carry_threshold(threshold_abs=int(carrier.threshold_abs))
    return int(np.sum(np.abs(hot_values.astype(np.int32)) >= int(promote_at)))


def _hot_rows_in_backlog(carrier: EventCodedAccLiveState) -> int:
    if not carrier.backlog:
        return 0
    hot = {int(i) for i in carrier._hot.indices_array().tolist()}
    return int(len(hot & {int(i) for i in carrier.backlog}))


def build_carrier_growth_module_row(
    *,
    state_key: str,
    carrier: EventCodedAccLiveState,
    step_record: StepSurfaceRecord,
    votes: torch.Tensor,
    cap_accepted_rows: int = 0,
    q_changed_rows: int = 0,
) -> dict[str, Any]:
    event_count = len(carrier.events)
    hot_count = len(carrier._hot)
    backlog_count = len(carrier.backlog)
    events_q_not_hot = _events_on_q_locked_not_hot_count(carrier)
    dup = _event_dup_summary(carrier.events)
    unique_lanes = int(len({int(event.flat_index) for event in carrier.events}))
    est_events = estimate_events_payload_bytes(event_count=event_count)
    est_hot = estimate_hot_exact_payload_bytes(hot_row_count=hot_count)
    est_v5_saved = int(events_q_not_hot) * EST_BYTES_PER_EVENT
    est_v2_saved = _estimate_v2_coalesce_saved_bytes(
        event_count=event_count,
        unique_lane_count=unique_lanes,
    )
    return {
        "state_key": str(state_key),
        "logical_numel": int(carrier.logical_numel),
        "event_count_after": int(event_count),
        "hot_exact_row_count_after": int(hot_count),
        "backlog_count_after": int(backlog_count),
        "new_crossing_events": int(len(step_record.crossing_indices)),
        "demotion_on_crossing_count": int(step_record.demotion_on_crossing_count),
        "demotion_on_decay_count": int(step_record.demotion_on_decay_count),
        "promotion_count": int(step_record.promotion_count),
        "cap_accepted_rows": int(cap_accepted_rows),
        "q_changed_rows": int(q_changed_rows),
        "events_on_q_locked_not_hot": int(events_q_not_hot),
        **dup,
        "hot_carry_abs_bucket_counts": _hot_carry_bucket_counts(carrier),
        "hot_rows_vote_touched": _hot_rows_vote_touched(carrier, votes),
        "hot_rows_in_proxy": _hot_rows_in_proxy(carrier, votes),
        "hot_rows_in_backlog": _hot_rows_in_backlog(carrier),
        "est_events_payload_bytes": int(est_events),
        "est_hot_exact_payload_bytes": int(est_hot),
        "est_saved_bytes_v5_clear": int(est_v5_saved),
        "est_saved_bytes_v2_coalesce": int(est_v2_saved),
        "active_lane_count": int(len(_active_lane_indices(carrier, votes))),
    }


def build_carrier_growth_step_record(
    *,
    optimizer_step_index: int,
    module_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = [dict(row) for row in module_rows]
    rollup: dict[str, Any] = {
        "module_count": len(rows),
        "event_count_after": sum(int(row["event_count_after"]) for row in rows),
        "hot_exact_row_count_after": sum(int(row["hot_exact_row_count_after"]) for row in rows),
        "backlog_count_after": sum(int(row["backlog_count_after"]) for row in rows),
        "new_crossing_events": sum(int(row["new_crossing_events"]) for row in rows),
        "events_on_q_locked_not_hot": sum(int(row["events_on_q_locked_not_hot"]) for row in rows),
        "est_events_payload_bytes": sum(int(row["est_events_payload_bytes"]) for row in rows),
        "est_hot_exact_payload_bytes": sum(int(row["est_hot_exact_payload_bytes"]) for row in rows),
        "est_saved_bytes_v5_clear": sum(int(row["est_saved_bytes_v5_clear"]) for row in rows),
        "est_saved_bytes_v2_coalesce": sum(int(row["est_saved_bytes_v2_coalesce"]) for row in rows),
    }
    payload = {
        "schema_version": CARRIER_GROWTH_SUMMARY_SCHEMA_VERSION,
        "optimizer_step_index": int(optimizer_step_index),
        "modules": rows,
        "rollup": rollup,
    }
    payload["rollup"]["sidecar_bytes"] = len(_canonical_json(payload).encode("utf-8"))
    return payload


def _rollup_from_module_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "module_count": len(rows),
        "event_count_after": sum(int(row["event_count_after"]) for row in rows),
        "hot_exact_row_count_after": sum(int(row["hot_exact_row_count_after"]) for row in rows),
        "backlog_count_after": sum(int(row["backlog_count_after"]) for row in rows),
        "new_crossing_events": sum(int(row["new_crossing_events"]) for row in rows),
        "events_on_q_locked_not_hot": sum(int(row["events_on_q_locked_not_hot"]) for row in rows),
        "est_events_payload_bytes": sum(int(row["est_events_payload_bytes"]) for row in rows),
        "est_hot_exact_payload_bytes": sum(int(row["est_hot_exact_payload_bytes"]) for row in rows),
        "est_saved_bytes_v5_clear": sum(int(row["est_saved_bytes_v5_clear"]) for row in rows),
        "est_saved_bytes_v2_coalesce": sum(int(row["est_saved_bytes_v2_coalesce"]) for row in rows),
    }


def build_carrier_growth_step_record_compact(
    *,
    optimizer_step_index: int,
    module_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = [dict(row) for row in module_rows]
    rows.sort(key=lambda row: str(row["state_key"]))
    module_digest_sha256 = sidecar_sha256({"modules": rows})
    rollup = _rollup_from_module_rows(rows)
    payload = {
        "schema_version": CARRIER_GROWTH_SUMMARY_SCHEMA_VERSION,
        "optimizer_step_index": int(optimizer_step_index),
        "compact": True,
        "module_digest_sha256": module_digest_sha256,
        "rollup": rollup,
    }
    payload["rollup"]["sidecar_bytes"] = len(_canonical_json(payload).encode("utf-8"))
    return payload


def phase2_oracle_required_sidecar_keys() -> frozenset[str]:
    return frozenset(
        {
            "est_events_payload_bytes",
            "est_hot_exact_payload_bytes",
            "est_saved_bytes_v5_clear",
            "est_saved_bytes_v2_coalesce",
            "events_on_q_locked_not_hot",
        }
    )


def project_best_combined_oracle_bpw(
    rollup: Mapping[str, Any],
    *,
    eligible_weight_count: int,
    metadata_bytes: int = DEFAULT_R4V_METADATA_BYTES,
    v1_max_hot_reduction_fraction: float = 0.0,
) -> float:
    """Delegate to the Phase 2 envelope projector facade."""

    from calm.hrm_text_158.native_full_stack.carrier_envelope_projector import (
        project_best_combined_oracle_bpw as _project_best_combined_oracle_bpw,
    )

    return _project_best_combined_oracle_bpw(
        rollup,
        eligible_weight_count=int(eligible_weight_count),
        metadata_bytes=int(metadata_bytes),
        v1_max_hot_reduction_fraction=float(v1_max_hot_reduction_fraction),
    )


class CarrierGrowthCollector:
    """Write per-step carrier growth sidecars under {root}/votes_emit/v1/carrier_growth."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.emit_root = self.root / "votes_emit" / "v1" / "carrier_growth"
        self.per_step_dir = self.emit_root / "per_step"
        self.per_step_dir.mkdir(parents=True, exist_ok=True)
        self._step_hashes: dict[str, str] = {}
        self._emit_timings_ms: list[float] = []

    def emit_step(
        self,
        payload: Mapping[str, Any],
        *,
        optimizer_step_index: int,
    ) -> dict[str, Any]:
        start = time.perf_counter()
        step_name = f"{int(optimizer_step_index):05d}"
        step_path = self.per_step_dir / f"{step_name}.json"
        canonical = _canonical_json(payload)
        write_sidecar_atomically(step_path, canonical.encode("utf-8"))
        step_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        self._step_hashes[step_name] = step_hash
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self._emit_timings_ms.append(float(elapsed_ms))
        manifest = self.write_manifest()
        return {
            "step_path": str(step_path),
            "step_hash": step_hash,
            "manifest_path": str(self.emit_root / "manifest.json"),
            "manifest_hash": manifest["manifest_sha256"],
            "emit_elapsed_ms": float(elapsed_ms),
            "sidecar_bytes": int(payload.get("rollup", {}).get("sidecar_bytes", 0)),
        }

    def write_manifest(self) -> dict[str, Any]:
        emit_timings_ms = [float(value) for value in self._emit_timings_ms]
        stable_manifest = {
            "schema_version": CARRIER_GROWTH_SUMMARY_SCHEMA_VERSION,
            "per_step_hashes": dict(sorted(self._step_hashes.items())),
            "step_count": int(len(self._step_hashes)),
        }
        manifest_sha256 = hashlib.sha256(
            _canonical_json(stable_manifest).encode("utf-8")
        ).hexdigest()
        manifest = {
            **stable_manifest,
            "emit_timings_ms": emit_timings_ms,
            "emit_sample_count": int(len(emit_timings_ms)),
            "manifest_sha256": manifest_sha256,
        }
        manifest_path = self.emit_root / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return manifest


def build_carrier_growth_step_record_from_states(
    *,
    optimizer_step_index: int,
    tensor_states: Mapping[str, BoundedDeltaTensorState],
    votes_by_key: Mapping[str, torch.Tensor],
    tensor_stats_by_key: Mapping[str, Mapping[str, Any]],
    compact: bool = True,
) -> dict[str, Any]:
    module_rows: list[dict[str, Any]] = []
    for state_key in sorted(tensor_states):
        state = tensor_states[state_key]
        vote_state = state.vote_update_state()
        if not isinstance(vote_state, EventCodedVoteUpdateState):
            continue
        carrier = vote_state.carrier
        if not carrier.step_records:
            raise ValueError(
                f"carrier growth emit requires step_records for {state_key!r}"
            )
        stats = dict(tensor_stats_by_key.get(state_key, {}))
        cap_accepted_rows = int(stats.get("global_rate_cap_accepted_count", 0))
        q_changed_rows = int(stats.get("q_changed_count", 0))
        module_rows.append(
            build_carrier_growth_module_row(
                state_key=str(state_key),
                carrier=carrier,
                step_record=carrier.step_records[-1],
                votes=votes_by_key[state_key],
                cap_accepted_rows=cap_accepted_rows,
                q_changed_rows=q_changed_rows,
            )
        )
    if compact:
        return build_carrier_growth_step_record_compact(
            optimizer_step_index=int(optimizer_step_index),
            module_rows=module_rows,
        )
    return build_carrier_growth_step_record(
        optimizer_step_index=int(optimizer_step_index),
        module_rows=module_rows,
    )


def maybe_emit_carrier_growth_step_record(
    *,
    enabled: bool,
    collector: CarrierGrowthCollector | None,
    optimizer_step_index: int,
    tensor_states: Mapping[str, BoundedDeltaTensorState],
    votes_by_key: Mapping[str, torch.Tensor],
    tensor_stats_by_key: Mapping[str, Mapping[str, Any]],
    compact: bool = True,
) -> dict[str, Any] | None:
    if not bool(enabled) or collector is None:
        return None
    payload = build_carrier_growth_step_record_from_states(
        optimizer_step_index=int(optimizer_step_index),
        tensor_states=tensor_states,
        votes_by_key=votes_by_key,
        tensor_stats_by_key=tensor_stats_by_key,
        compact=bool(compact),
    )
    return collector.emit_step(payload, optimizer_step_index=int(optimizer_step_index))
