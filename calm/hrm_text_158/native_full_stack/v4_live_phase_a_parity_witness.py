"""Deterministic Phase-A GPU-vs-CPU parity witness for V4-LIVE dynamics proof."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from calm.hrm_text_158.native_full_stack.event_coded_acc_live_carrier import (
    EventCodedAccLiveState,
    StepSurfaceRecord,
    decisive_surface_drift_details,
)
from calm.hrm_text_158.native_full_stack.event_coded_vote_update_adapter import (
    C8_DENSE_ACCUMULATOR_MATERIALIZED_NUMEL_KEY,
    C8_PERSISTENT_AUTHORITY_SCOPE_KEY,
    C8_TRANSIENT_DENSE_COMPUTE_NUMEL_KEY,
    carrier_content_sha256,
)
from calm.hrm_text_158.native_full_stack.votes_emit_collector import (
    VOTES_EMIT_SECTION6_CONTRACT_FIELDS,
)

V4_LIVE_PHASE_A_PARITY_WITNESS_SCHEMA = "hrm_text_158_v4_live_phase_a_parity_witness/v0"


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def surfaces_dict_to_record(step_index: int, payload: Mapping[str, Any]) -> StepSurfaceRecord:
    q_snapshot = {
        int(key): int(value)
        for key, value in dict(payload.get("decisive_q_snapshot", {})).items()
    }
    crossing = tuple(int(value) for value in payload.get("crossing_flat_indices", ()))
    applied = tuple(int(value) for value in payload.get("applied_flat_indices", ()))
    return StepSurfaceRecord(
        step_index=int(step_index),
        crossing_indices=crossing,
        applied_indices=applied,
        backlog_indices=(),
        q_levels=q_snapshot,
        hot_exact_row_count=0,
        promotion_count=0,
        demotion_on_decay_count=0,
        demotion_on_crossing_count=0,
    )


def sparse_votes_from_emit_record(record: Mapping[str, Any]) -> dict[str, dict[int, int]]:
    raw = dict(record.get("sparse_vote_inputs_by_state_key", {}))
    votes_by_key: dict[str, dict[int, int]] = {}
    for state_key in sorted(raw):
        lane_map = {
            int(flat_index): int(vote)
            for flat_index, vote in dict(raw[state_key]).items()
            if int(vote) != 0
        }
        votes_by_key[str(state_key)] = lane_map
    return votes_by_key


SPARSE_VOTE_INPUTS_FIELD = "sparse_vote_inputs_by_state_key"


def section6_complete(record: Mapping[str, Any]) -> bool:
    if not all(str(field) in record for field in VOTES_EMIT_SECTION6_CONTRACT_FIELDS):
        return False
    sparse = record.get(SPARSE_VOTE_INPUTS_FIELD)
    return isinstance(sparse, Mapping) and bool(sparse)


def _c8_scope_wording_valid(scope: Any) -> bool:
    if not isinstance(scope, str) or not scope.strip():
        return False
    lowered = scope.lower()
    return (
        "persistent" in lowered
        and "transient" in lowered
        and ("no dense" in lowered or "no_dense" in lowered)
    )


@dataclass(frozen=True)
class StepParityMismatch:
    step_index: int
    state_key: str
    reason: str
    detail: dict[str, Any]


def _cpu_replay_step(
    carrier: EventCodedAccLiveState,
    *,
    step_index: int,
    votes: Mapping[int, int],
) -> StepSurfaceRecord:
    carrier.apply_step(int(step_index), votes=dict(votes))
    if not carrier.step_records:
        raise ValueError("carrier replay produced no step record")
    return carrier.step_records[-1]


def _compare_surfaces(
    *,
    step_index: int,
    state_key: str,
    cpu_record: StepSurfaceRecord,
    gpu_surfaces: Mapping[str, Any],
) -> StepParityMismatch | None:
    gpu_record = surfaces_dict_to_record(step_index, gpu_surfaces)
    details = decisive_surface_drift_details((cpu_record,), (gpu_record,))
    if not details:
        return None
    return StepParityMismatch(
        step_index=int(step_index),
        state_key=str(state_key),
        reason="decisive_surface_mismatch",
        detail=dict(details[0]),
    )


def run_phase_a_parity_witness(
    phase_root: Path,
    *,
    demotion_band: int = 1,
    max_steps: int | None = None,
) -> dict[str, Any]:
    phase_root = Path(phase_root)
    receipt_path = phase_root / "receipt.json"
    votes_emit_root = phase_root / "votes_emit" / "v1"
    if not receipt_path.is_file():
        raise ValueError(f"missing receipt.json under phase root: {receipt_path}")
    if not votes_emit_root.is_dir():
        raise ValueError(f"missing votes_emit/v1 under phase root: {votes_emit_root}")

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    step_reports = dict(receipt.get("step_reports", {}))
    if not step_reports:
        raise ValueError("receipt.step_reports is empty")

    ordered_steps = sorted(int(step) for step in step_reports)
    if max_steps is not None:
        ordered_steps = [step for step in ordered_steps if step < int(max_steps)]

    carriers: dict[str, EventCodedAccLiveState] = {}
    per_step_mismatches: list[dict[str, Any]] = []
    section6_failures: list[int] = []
    sparse_vote_failures: list[int] = []
    zero_vote_step_failures: list[int] = []
    c8_failures: list[dict[str, Any]] = []
    carrier_sha_mismatches: list[dict[str, Any]] = []
    decisive_surface_diff_count = 0
    states_compared = 0

    for step_index in ordered_steps:
        emit_path = votes_emit_root / "per_step" / f"{int(step_index):05d}.json"
        if not emit_path.is_file():
            per_step_mismatches.append(
                {
                    "step_index": int(step_index),
                    "state_key": "*",
                    "reason": "missing_votes_emit_step",
                    "detail": {"path": str(emit_path)},
                }
            )
            continue
        emit_record = json.loads(emit_path.read_text(encoding="utf-8"))
        if int(emit_record.get("optimizer_step_index", -1)) != int(step_index):
            per_step_mismatches.append(
                {
                    "step_index": int(step_index),
                    "state_key": "*",
                    "reason": "optimizer_step_index_mismatch",
                    "detail": {
                        "expected": int(step_index),
                        "actual": emit_record.get("optimizer_step_index"),
                    },
                }
            )
        if not section6_complete(emit_record):
            section6_failures.append(int(step_index))
        raw_sparse = emit_record.get(SPARSE_VOTE_INPUTS_FIELD)
        if not isinstance(raw_sparse, Mapping):
            sparse_vote_failures.append(int(step_index))

        votes_by_key = sparse_votes_from_emit_record(emit_record)
        if not votes_by_key or all(not lane_votes for lane_votes in votes_by_key.values()):
            zero_vote_step_failures.append(int(step_index))
            continue
        step_result = dict(step_reports[str(step_index)].get("step_result", {}))
        tensor_stats = dict(step_result.get("tensor_stats", {}))

        for state_key in sorted(votes_by_key):
            stats = dict(tensor_stats.get(state_key, {}))
            if int(stats.get(C8_DENSE_ACCUMULATOR_MATERIALIZED_NUMEL_KEY, -1)) != 0:
                c8_failures.append(
                    {
                        "step_index": int(step_index),
                        "state_key": str(state_key),
                        "reason": "dense_persistent_authority_materialized",
                        "dense_accumulator_materialized_numel": stats.get(
                            C8_DENSE_ACCUMULATOR_MATERIALIZED_NUMEL_KEY
                        ),
                    }
                )
            scope = stats.get(C8_PERSISTENT_AUTHORITY_SCOPE_KEY)
            if not _c8_scope_wording_valid(scope):
                c8_failures.append(
                    {
                        "step_index": int(step_index),
                        "state_key": str(state_key),
                        "reason": "missing_or_invalid_c8_persistent_authority_scope",
                        "c8_persistent_authority_scope": scope,
                    }
                )
            transient = stats.get(C8_TRANSIENT_DENSE_COMPUTE_NUMEL_KEY)
            if transient is None or not isinstance(transient, (int, float)):
                c8_failures.append(
                    {
                        "step_index": int(step_index),
                        "state_key": str(state_key),
                        "reason": "missing_or_non_numeric_transient_dense_compute_numel",
                        "transient_dense_compute_numel": transient,
                    }
                )
            states_compared += 1
            if state_key not in carriers:
                logical_numel = int(stats.get("logical_numel", 0))
                if logical_numel <= 0:
                    # Fall back to max touched index + 1 when logical_numel not recorded.
                    touched = votes_by_key[state_key]
                    logical_numel = max(touched.keys(), default=-1) + 1
                if logical_numel <= 0:
                    per_step_mismatches.append(
                        {
                            "step_index": int(step_index),
                            "state_key": str(state_key),
                            "reason": "missing_logical_numel",
                            "detail": {},
                        }
                    )
                    continue
                carriers[state_key] = EventCodedAccLiveState(
                    logical_numel=int(logical_numel),
                    demotion_band=int(demotion_band),
                )

            cpu_record = _cpu_replay_step(
                carriers[state_key],
                step_index=int(step_index),
                votes=votes_by_key[state_key],
            )
            gpu_surfaces = dict(stats.get("v4_live_observed_surfaces", {}))
            if not gpu_surfaces:
                per_step_mismatches.append(
                    {
                        "step_index": int(step_index),
                        "state_key": str(state_key),
                        "reason": "missing_gpu_observed_surfaces",
                        "detail": {},
                    }
                )
                continue
            mismatch = _compare_surfaces(
                step_index=int(step_index),
                state_key=str(state_key),
                cpu_record=cpu_record,
                gpu_surfaces=gpu_surfaces,
            )
            if mismatch is not None:
                per_step_mismatches.append(
                    {
                        "step_index": mismatch.step_index,
                        "state_key": mismatch.state_key,
                        "reason": mismatch.reason,
                        "detail": mismatch.detail,
                    }
                )
                decisive_surface_diff_count += 1

            gpu_sha = stats.get("event_coded_live_carrier_content_sha256_after")
            cpu_sha = carrier_content_sha256(carriers[state_key])
            if gpu_sha is not None and str(gpu_sha) != str(cpu_sha):
                carrier_sha_mismatches.append(
                    {
                        "step_index": int(step_index),
                        "state_key": str(state_key),
                        "gpu_sha": str(gpu_sha),
                        "cpu_sha": str(cpu_sha),
                    }
                )

    r4v_ledger = dict(receipt.get("r4v_persistent_ledger", {}))
    ledger_pass = r4v_ledger.get("ledger_pass")
    if ledger_pass is None:
        ledger_pass = r4v_ledger.get("r4v_ledger_pass")
    if ledger_pass is None:
        # Fall back to event-coded acc inclusive field if present.
        ledger_pass = r4v_ledger.get("r4v_acc_inclusive_physical_bits_per_weight_pass")

    phase_a_parity_pass = (
        bool(ordered_steps)
        and states_compared > 0
        and not per_step_mismatches
        and not section6_failures
        and not sparse_vote_failures
        and not zero_vote_step_failures
        and not c8_failures
        and not carrier_sha_mismatches
        and decisive_surface_diff_count == 0
        and ledger_pass is True
    )

    return {
        "schema_version": V4_LIVE_PHASE_A_PARITY_WITNESS_SCHEMA,
        "phase_root": str(phase_root),
        "demotion_band": int(demotion_band),
        "steps_compared": [int(step) for step in ordered_steps],
        "states_compared_count": int(states_compared),
        "phase_a_parity_pass": bool(phase_a_parity_pass),
        "decisive_surface_diff_count": int(decisive_surface_diff_count),
        "per_step_mismatches": per_step_mismatches,
        "section6_missing_steps": section6_failures,
        "sparse_vote_inputs_missing_steps": sparse_vote_failures,
        "zero_vote_inputs_steps": zero_vote_step_failures,
        "c8_persistent_failures": c8_failures,
        "carrier_sha_mismatches": carrier_sha_mismatches,
        "measure_r4v_ledger_pass": ledger_pass,
        "c8_witnessed_from_run_stats": True,
        "transient_dense_disclosure_required": True,
    }
