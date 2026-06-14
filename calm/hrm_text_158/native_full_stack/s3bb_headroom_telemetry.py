"""S3bb observe-only headroom telemetry and dual-arm postrun classifier (CPU-first)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import torch

from calm.hrm_text_158.native_full_stack.narrow_accumulator_codec import (
    NarrowCarrierHeadroomBreach,
    W6_SIGNED_MAX,
    W6_SIGNED_MIN,
)
from calm.hrm_text_158.native_full_stack.two_tier_carry_reducers import (
    DEFAULT_CROSSING_THRESHOLD_ABS,
    crossing_bool_w6,
)

S3BB_W6_HEADROOM_DIAGNOSTIC_PHASE = "s3bb-w6-headroom-diagnostic"
HEADROOM_TELEMETRY_SCHEMA_VERSION = "hrm_text_158_s3bb_headroom_telemetry/v0"
W6_HEADROOM_K_DEFAULT = 2

CLASSIFIER_HARNESS_OR_LIVENESS_FAIL = "HARNESS_OR_LIVENESS_FAIL"
CLASSIFIER_HEADROOM_BREACH = "HEADROOM_BREACH"
CLASSIFIER_W6_DYNAMICS_DIVERGES = "W6_DYNAMICS_DIVERGES"
CLASSIFIER_W6_HEADROOM_SUFFICIENT_PARITY_OK = "W6_HEADROOM_SUFFICIENT_PARITY_OK"

CLASSIFIER_PRECEDENCE: tuple[str, ...] = (
    CLASSIFIER_HARNESS_OR_LIVENESS_FAIL,
    CLASSIFIER_HEADROOM_BREACH,
    CLASSIFIER_W6_DYNAMICS_DIVERGES,
    CLASSIFIER_W6_HEADROOM_SUFFICIENT_PARITY_OK,
)

CLASSIFIER_S3BC_HARNESS_FAIL = "S3BC_HARNESS_FAIL"
CLASSIFIER_S3BC_TELEMETRY_OR_CLASSIFIER_INCOMPLETE = "S3BC_TELEMETRY_OR_CLASSIFIER_INCOMPLETE"
CLASSIFIER_S3BC_HEADROOM_TELEMETRY_POSTRUN_OK = "S3BC_HEADROOM_TELEMETRY_POSTRUN_OK"

CLASSIFIER_S3BC_PRECEDENCE: tuple[str, ...] = (
    CLASSIFIER_S3BC_HARNESS_FAIL,
    CLASSIFIER_S3BC_TELEMETRY_OR_CLASSIFIER_INCOMPLETE,
    CLASSIFIER_S3BC_HEADROOM_TELEMETRY_POSTRUN_OK,
)

REQUIRED_HEADROOM_TELEMETRY_FIELDS: tuple[str, ...] = (
    "schema_version",
    "global_max_abs_accumulator",
    "margin_to_w6_boundary_min",
    "lanes_within_K_of_boundary_fraction",
    "out_of_domain_lane_count",
    "would_strict_raise_step",
    "strict_raise_count",
    "boundary_value_error_caught",
    "eligible_module_count",
    "total_lane_count",
)

MEASURED_STEPS_REQUIRED = 10
WARMUP_STEPS = 2

RECEIPT_EMIT_PROFILE_FULL = "full"
RECEIPT_EMIT_PROFILE_SLIM = "s3bb_headroom_diagnostic_slim"
RECEIPT_EMIT_PROFILE_CHOICES: tuple[str, ...] = (
    RECEIPT_EMIT_PROFILE_FULL,
    RECEIPT_EMIT_PROFILE_SLIM,
)

SNAPSHOT_MODE_FULL = "full"
SNAPSHOT_MODE_AGGREGATE_ONLY = "aggregate_only"
SNAPSHOT_MODE_CHOICES: tuple[str, ...] = (
    SNAPSHOT_MODE_FULL,
    SNAPSHOT_MODE_AGGREGATE_ONLY,
)

HEADROOM_WIRING_SIDECAR_SCHEMA_VERSION = (
    "hrm_text_158_s3bb_headroom_wiring_sidecar_chunk/v1"
)
HEADROOM_WIRING_SIDECAR_FILENAME = "headroom_wiring_sidecar.jsonl"


@dataclass(frozen=True)
class S3bbMaterializationOutcome:
    value: Any | None = None
    terminated: bool = False
    stop_reason: str | None = None


def accumulator_out_of_domain_mask(acc: torch.Tensor) -> torch.Tensor:
    """Predicate-only mask matching pack_w6_tensor reject without invoking pack."""

    if acc.dtype != torch.int16:
        raise ValueError(f"accumulator_out_of_domain_mask requires torch.int16, got {acc.dtype}")
    values = acc.to(dtype=torch.int32)
    return (values < W6_SIGNED_MIN) | (values > W6_SIGNED_MAX)


def compute_headroom_telemetry_from_accumulators(
    acc: torch.Tensor,
    *,
    k: int = W6_HEADROOM_K_DEFAULT,
) -> dict[str, Any]:
    """Observe-only headroom stats from raw int16 accumulators (never raises)."""

    if acc.dtype != torch.int16:
        raise ValueError(
            f"compute_headroom_telemetry_from_accumulators requires torch.int16, got {acc.dtype}"
        )
    if int(acc.numel()) == 0:
        return {
            "schema_version": HEADROOM_TELEMETRY_SCHEMA_VERSION,
            "global_max_abs_accumulator": 0,
            "margin_to_w6_boundary_min": int(W6_SIGNED_MAX),
            "lanes_within_K_of_boundary_fraction": 0.0,
            "out_of_domain_lane_count": 0,
            "would_strict_raise_step": False,
            "strict_raise_count": 0,
            "boundary_value_error_caught": False,
            "eligible_module_count": 0,
            "total_lane_count": 0,
        }

    values = acc.to(dtype=torch.int32)
    abs_values = values.abs()
    global_max_abs = int(abs_values.max().item())
    out_of_domain = accumulator_out_of_domain_mask(acc)
    out_of_domain_lane_count = int(out_of_domain.sum().item())
    boundary_threshold = int(W6_SIGNED_MAX) - int(k)
    near_boundary = abs_values >= boundary_threshold
    lanes_within_k_fraction = float(near_boundary.sum().item()) / float(acc.numel())
    would_strict_raise_step = out_of_domain_lane_count > 0
    strict_raise_count = 1 if would_strict_raise_step else 0

    return {
        "schema_version": HEADROOM_TELEMETRY_SCHEMA_VERSION,
        "global_max_abs_accumulator": global_max_abs,
        "margin_to_w6_boundary_min": int(W6_SIGNED_MAX) - global_max_abs,
        "lanes_within_K_of_boundary_fraction": lanes_within_k_fraction,
        "out_of_domain_lane_count": out_of_domain_lane_count,
        "would_strict_raise_step": bool(would_strict_raise_step),
        "strict_raise_count": int(strict_raise_count),
        "boundary_value_error_caught": False,
        "eligible_module_count": 1,
        "total_lane_count": int(acc.numel()),
    }


def _shadow_tensors_from_states(
    states: Mapping[str, Any],
) -> tuple[list[torch.Tensor], list[str]]:
    tensors: list[torch.Tensor] = []
    keys: list[str] = []
    for state_key, state in sorted(states.items()):
        shadow = getattr(state, "exact_accumulator_shadow", None)
        if shadow is None:
            continue
        if shadow.dtype != torch.int16:
            raise ValueError(
                f"{state_key} exact_accumulator_shadow must be torch.int16, got {shadow.dtype}"
            )
        tensors.append(shadow)
        keys.append(str(state_key))
    return tensors, keys


def aggregate_headroom_telemetry_for_tensor_states(
    states: Mapping[str, Any],
    *,
    k: int = W6_HEADROOM_K_DEFAULT,
    snapshot_mode: str = SNAPSHOT_MODE_FULL,
    headroom_wiring_sidecar_path: Path | None = None,
    step: int | None = None,
) -> dict[str, Any]:
    """Aggregate observe-only telemetry across eligible tensor states."""

    if str(snapshot_mode) not in SNAPSHOT_MODE_CHOICES:
        raise ValueError(
            f"snapshot_mode must be one of {SNAPSHOT_MODE_CHOICES}, got {snapshot_mode!r}"
        )

    tensors, keys = _shadow_tensors_from_states(states)
    if not tensors:
        empty = compute_headroom_telemetry_from_accumulators(
            torch.zeros((0,), dtype=torch.int16),
            k=k,
        )
        empty["eligible_module_count"] = 0
        if snapshot_mode == SNAPSHOT_MODE_FULL:
            empty["accumulator_snapshots_by_state_key"] = {}
            empty["q_snapshots_by_state_key"] = {}
        return empty

    flat_pieces = [tensor.reshape(-1) for tensor in tensors]
    concatenated = torch.cat(flat_pieces)
    telemetry = compute_headroom_telemetry_from_accumulators(concatenated, k=k)
    telemetry["eligible_module_count"] = len(tensors)
    if snapshot_mode == SNAPSHOT_MODE_FULL:
        telemetry["accumulator_snapshots_by_state_key"] = {
            key: tensor.detach().reshape(-1).tolist()
            for key, tensor in zip(keys, tensors, strict=True)
        }
        telemetry["q_snapshots_by_state_key"] = {
            key: getattr(states[key], "q_levels").detach().reshape(-1).tolist()
            for key in keys
        }
        return telemetry

    if (
        headroom_wiring_sidecar_path is not None
        and step is not None
        and int(step) > WARMUP_STEPS
    ):
        for key, tensor in zip(keys, tensors, strict=True):
            append_headroom_wiring_sidecar_chunk(
                headroom_wiring_sidecar_path,
                step=int(step),
                state_key=str(key),
                accumulator_lanes=tensor.detach().reshape(-1).tolist(),
                q_lanes=getattr(states[key], "q_levels").detach().reshape(-1).tolist(),
            )
    return telemetry


def append_headroom_wiring_sidecar_chunk(
    sidecar_path: Path | str,
    *,
    step: int,
    state_key: str,
    accumulator_lanes: Sequence[int],
    q_lanes: Sequence[int],
) -> None:
    """Append one chunked wiring record; caller must not retain lane lists afterward."""

    path = Path(sidecar_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": HEADROOM_WIRING_SIDECAR_SCHEMA_VERSION,
        "step": int(step),
        "state_key": str(state_key),
        "accumulator_lanes": [int(value) for value in accumulator_lanes],
        "q_lanes": [int(value) for value in q_lanes],
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, separators=(",", ":"), sort_keys=True))
        handle.write("\n")


def attach_s3bb_headroom_telemetry_to_step_report(
    step_report: dict[str, Any],
    *,
    phase: str,
    post_update_states: Mapping[str, Any],
    snapshot_mode: str = SNAPSHOT_MODE_FULL,
    headroom_wiring_sidecar_path: Path | None = None,
    step: int | None = None,
) -> dict[str, Any]:
    """Phase-gated hook: append headroom_telemetry only for S3bb diagnostic phase."""

    if str(phase) != S3BB_W6_HEADROOM_DIAGNOSTIC_PHASE:
        return step_report
    step_report["headroom_telemetry"] = aggregate_headroom_telemetry_for_tensor_states(
        post_update_states,
        snapshot_mode=str(snapshot_mode),
        headroom_wiring_sidecar_path=headroom_wiring_sidecar_path,
        step=step,
    )
    return step_report


def record_boundary_value_error_caught(step_report: dict[str, Any]) -> dict[str, Any]:
    """Mark an actual trainer-boundary NarrowCarrierHeadroomBreach on the step report."""

    telemetry = dict(step_report.get("headroom_telemetry") or {})
    telemetry.setdefault("schema_version", HEADROOM_TELEMETRY_SCHEMA_VERSION)
    telemetry["boundary_value_error_caught"] = True
    telemetry["would_strict_raise_step"] = True
    telemetry["strict_raise_count"] = 1
    step_report["headroom_telemetry"] = telemetry
    return step_report


def run_vote_materialization_with_s3bb_boundary_catch(
    *,
    phase: str,
    step_report: dict[str, Any],
    materialize: Callable[[], Any],
) -> S3bbMaterializationOutcome:
    """S3bb-only boundary catch; non-S3bb phases propagate ValueError normally."""

    if str(phase) != S3BB_W6_HEADROOM_DIAGNOSTIC_PHASE:
        return S3bbMaterializationOutcome(value=materialize())
    try:
        return S3bbMaterializationOutcome(value=materialize())
    except NarrowCarrierHeadroomBreach:
        record_boundary_value_error_caught(step_report)
        return S3bbMaterializationOutcome(
            terminated=True,
            stop_reason="headroom_breach",
        )


def _measured_step_ids(step_reports: Mapping[str, Any]) -> list[str]:
    measured: list[str] = []
    for step_id in sorted(step_reports, key=lambda value: int(value)):
        if int(step_id) <= WARMUP_STEPS:
            continue
        measured.append(str(step_id))
    return measured


def _treatment_headroom_breach(treatment_receipt: Mapping[str, Any]) -> bool:
    if str(treatment_receipt.get("stop_reason") or "") == "headroom_breach":
        return True
    step_reports = treatment_receipt.get("step_reports") or {}
    for report in step_reports.values():
        telemetry = report.get("headroom_telemetry") or {}
        if int(telemetry.get("strict_raise_count") or 0) > 0:
            return True
        if int(telemetry.get("global_max_abs_accumulator") or 0) > int(W6_SIGNED_MAX):
            return True
        if bool(telemetry.get("boundary_value_error_caught")):
            return True
    return False


def _receipt_has_inline_wiring_snapshots(receipt: Mapping[str, Any]) -> bool:
    step_reports = receipt.get("step_reports") or {}
    for report in step_reports.values():
        telemetry = report.get("headroom_telemetry") or {}
        snapshots = telemetry.get("accumulator_snapshots_by_state_key") or {}
        if snapshots:
            return True
    return False


def _shared_measured_step_ids(
    oracle_receipt: Mapping[str, Any],
    treatment_receipt: Mapping[str, Any],
) -> list[str]:
    oracle_steps = oracle_receipt.get("step_reports") or {}
    treatment_steps = treatment_receipt.get("step_reports") or {}
    shared_steps = sorted(
        set(oracle_steps).intersection(treatment_steps),
        key=lambda value: int(value),
    )
    return [step_id for step_id in shared_steps if int(step_id) > WARMUP_STEPS]


def _wiring_guard_skip_steps(
    oracle_receipt: Mapping[str, Any],
    treatment_receipt: Mapping[str, Any],
) -> set[int]:
    oracle_steps = oracle_receipt.get("step_reports") or {}
    treatment_steps = treatment_receipt.get("step_reports") or {}
    skip_steps: set[int] = set()
    for step_id in _shared_measured_step_ids(oracle_receipt, treatment_receipt):
        oracle_telemetry = oracle_steps[step_id].get("headroom_telemetry") or {}
        treatment_telemetry = treatment_steps[step_id].get("headroom_telemetry") or {}
        if bool(oracle_telemetry.get("would_strict_raise_step")) or bool(
            treatment_telemetry.get("would_strict_raise_step")
        ):
            skip_steps.add(int(step_id))
    return skip_steps


def _compare_lane_chunks(
    o_vals: Sequence[int],
    t_vals: Sequence[int],
    o_q: Sequence[int],
    t_q: Sequence[int],
    *,
    threshold_abs: int,
) -> tuple[float, int, int, int]:
    l1_max = 0.0
    crossing_disagreements = 0
    total_lanes = 0
    equal_lanes = 0
    if len(o_vals) != len(t_vals):
        return 0.0, 1, 0, 0
    for lane_index, (o_val, t_val) in enumerate(zip(o_vals, t_vals, strict=True)):
        total_lanes += 1
        delta = abs(int(o_val) - int(t_val))
        l1_max = max(l1_max, float(delta))
        if delta == 0:
            equal_lanes += 1
        q_level = int(o_q[lane_index]) if lane_index < len(o_q) else 0
        o_cross = crossing_bool_w6(int(o_val), q_level, threshold_abs=int(threshold_abs))
        t_cross = crossing_bool_w6(int(t_val), q_level, threshold_abs=int(threshold_abs))
        if o_cross != t_cross:
            crossing_disagreements += 1
    return l1_max, crossing_disagreements, equal_lanes, total_lanes


def _finalize_wiring_guard_stats(
    *,
    l1_max: float,
    crossing_disagreements: int,
    equal_lanes: int,
    total_lanes: int,
    measured_step_count: int,
) -> dict[str, Any]:
    equality_rate = 1.0 if total_lanes == 0 else float(equal_lanes) / float(total_lanes)
    return {
        "per_step_accumulator_l1_max_abs_delta": float(l1_max),
        "per_step_crossing_bool_disagreement_count": int(crossing_disagreements),
        "vote_update_state_accumulator_equality_rate": float(equality_rate),
        "measured_step_count": int(measured_step_count),
        "total_lane_count": int(total_lanes),
    }


def compare_arm_wiring_guards_inline(
    oracle_receipt: Mapping[str, Any],
    treatment_receipt: Mapping[str, Any],
    *,
    threshold_abs: int = DEFAULT_CROSSING_THRESHOLD_ABS,
) -> dict[str, Any]:
    """Wiring-bug guards from inline receipt snapshots."""

    oracle_steps = oracle_receipt.get("step_reports") or {}
    treatment_steps = treatment_receipt.get("step_reports") or {}
    measured = _shared_measured_step_ids(oracle_receipt, treatment_receipt)
    skip_steps = _wiring_guard_skip_steps(oracle_receipt, treatment_receipt)

    l1_max = 0.0
    crossing_disagreements = 0
    total_lanes = 0
    equal_lanes = 0

    for step_id in measured:
        if int(step_id) in skip_steps:
            continue
        oracle_telemetry = oracle_steps[step_id].get("headroom_telemetry") or {}
        treatment_telemetry = treatment_steps[step_id].get("headroom_telemetry") or {}
        oracle_acc = oracle_telemetry.get("accumulator_snapshots_by_state_key") or {}
        treatment_acc = treatment_telemetry.get("accumulator_snapshots_by_state_key") or {}
        oracle_q = oracle_telemetry.get("q_snapshots_by_state_key") or {}
        treatment_q = treatment_telemetry.get("q_snapshots_by_state_key") or {}

        for state_key in sorted(set(oracle_acc).intersection(treatment_acc)):
            o_vals = [int(v) for v in oracle_acc[state_key]]
            t_vals = [int(v) for v in treatment_acc[state_key]]
            o_q = [int(v) for v in oracle_q.get(state_key, [0] * len(o_vals))]
            t_q = [int(v) for v in treatment_q.get(state_key, [0] * len(t_vals))]
            chunk_l1, chunk_cross, chunk_equal, chunk_total = _compare_lane_chunks(
                o_vals,
                t_vals,
                o_q,
                t_q,
                threshold_abs=int(threshold_abs),
            )
            l1_max = max(l1_max, chunk_l1)
            crossing_disagreements += chunk_cross
            equal_lanes += chunk_equal
            total_lanes += chunk_total

    return _finalize_wiring_guard_stats(
        l1_max=l1_max,
        crossing_disagreements=crossing_disagreements,
        equal_lanes=equal_lanes,
        total_lanes=total_lanes,
        measured_step_count=len(measured),
    )


def _iter_sidecar_records(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            if not isinstance(record, dict):
                raise ValueError(f"sidecar record must be object at {path}:{line_number}")
            yield record


def _sidecar_record_key(record: Mapping[str, Any]) -> tuple[int, str]:
    return int(record["step"]), str(record["state_key"])


def compare_arm_wiring_guards_streaming(
    oracle_receipt: Mapping[str, Any],
    treatment_receipt: Mapping[str, Any],
    *,
    oracle_sidecar_path: Path | str,
    treatment_sidecar_path: Path | str,
    threshold_abs: int = DEFAULT_CROSSING_THRESHOLD_ABS,
) -> dict[str, Any]:
    """Lockstep streaming wiring guards from chunked sidecars (no receipt hydration)."""

    oracle_path = Path(oracle_sidecar_path)
    treatment_path = Path(treatment_sidecar_path)
    if not oracle_path.is_file():
        raise FileNotFoundError(f"missing oracle wiring sidecar: {oracle_path}")
    if not treatment_path.is_file():
        raise FileNotFoundError(f"missing treatment wiring sidecar: {treatment_path}")

    measured = _shared_measured_step_ids(oracle_receipt, treatment_receipt)
    skip_steps = _wiring_guard_skip_steps(oracle_receipt, treatment_receipt)

    l1_max = 0.0
    crossing_disagreements = 0
    total_lanes = 0
    equal_lanes = 0

    oracle_iter = _iter_sidecar_records(oracle_path)
    treatment_iter = _iter_sidecar_records(treatment_path)
    oracle_record = next(oracle_iter, None)
    treatment_record = next(treatment_iter, None)

    while oracle_record is not None or treatment_record is not None:
        if oracle_record is None or treatment_record is None:
            raise ValueError("oracle and treatment sidecar record counts differ")
        oracle_key = _sidecar_record_key(oracle_record)
        treatment_key = _sidecar_record_key(treatment_record)
        if oracle_key != treatment_key:
            raise ValueError(
                f"sidecar key mismatch at lockstep compare: {oracle_key} vs {treatment_key}"
            )
        step_id, _state_key = oracle_key
        if int(step_id) not in skip_steps and int(step_id) > WARMUP_STEPS:
            o_vals = [int(v) for v in oracle_record["accumulator_lanes"]]
            t_vals = [int(v) for v in treatment_record["accumulator_lanes"]]
            o_q = [int(v) for v in oracle_record["q_lanes"]]
            t_q = [int(v) for v in treatment_record["q_lanes"]]
            chunk_l1, chunk_cross, chunk_equal, chunk_total = _compare_lane_chunks(
                o_vals,
                t_vals,
                o_q,
                t_q,
                threshold_abs=int(threshold_abs),
            )
            l1_max = max(l1_max, chunk_l1)
            crossing_disagreements += chunk_cross
            equal_lanes += chunk_equal
            total_lanes += chunk_total
        oracle_record = next(oracle_iter, None)
        treatment_record = next(treatment_iter, None)

    return _finalize_wiring_guard_stats(
        l1_max=l1_max,
        crossing_disagreements=crossing_disagreements,
        equal_lanes=equal_lanes,
        total_lanes=total_lanes,
        measured_step_count=len(measured),
    )


def resolve_headroom_wiring_sidecar_path(
    receipt: Mapping[str, Any],
    *,
    arm_dir: Path | str | None = None,
) -> Path | None:
    explicit = receipt.get("headroom_wiring_sidecar_path")
    if explicit:
        return Path(str(explicit))
    if arm_dir is not None:
        return Path(arm_dir) / HEADROOM_WIRING_SIDECAR_FILENAME
    return None


def compare_arm_wiring_guards(
    oracle_receipt: Mapping[str, Any],
    treatment_receipt: Mapping[str, Any],
    *,
    threshold_abs: int = DEFAULT_CROSSING_THRESHOLD_ABS,
    oracle_sidecar_path: Path | str | None = None,
    treatment_sidecar_path: Path | str | None = None,
) -> dict[str, Any]:
    """Dispatch inline full-mode or streaming sidecar wiring guards."""

    if _receipt_has_inline_wiring_snapshots(oracle_receipt) and _receipt_has_inline_wiring_snapshots(
        treatment_receipt
    ):
        return compare_arm_wiring_guards_inline(
            oracle_receipt,
            treatment_receipt,
            threshold_abs=int(threshold_abs),
        )

    oracle_path = (
        Path(oracle_sidecar_path)
        if oracle_sidecar_path is not None
        else resolve_headroom_wiring_sidecar_path(oracle_receipt)
    )
    treatment_path = (
        Path(treatment_sidecar_path)
        if treatment_sidecar_path is not None
        else resolve_headroom_wiring_sidecar_path(treatment_receipt)
    )
    if oracle_path is None or treatment_path is None:
        raise ValueError(
            "wiring guards require inline snapshots or both headroom_wiring_sidecar_path values"
        )
    return compare_arm_wiring_guards_streaming(
        oracle_receipt,
        treatment_receipt,
        oracle_sidecar_path=oracle_path,
        treatment_sidecar_path=treatment_path,
        threshold_abs=int(threshold_abs),
    )


def classify_s3bb_run(
    oracle_receipt: Mapping[str, Any],
    treatment_receipt: Mapping[str, Any],
    *,
    harness_failures: Sequence[str] | None = None,
) -> str:
    """Emit primary classifier per S3bb v2 classifier_resolution rules 1-5."""

    failures = list(dict.fromkeys(harness_failures or ()))
    if failures:
        return CLASSIFIER_HARNESS_OR_LIVENESS_FAIL

    if _treatment_headroom_breach(treatment_receipt):
        return CLASSIFIER_HEADROOM_BREACH

    oracle_steps = int(oracle_receipt.get("steps_completed") or 0)
    treatment_steps = int(treatment_receipt.get("steps_completed") or 0)
    if oracle_steps < MEASURED_STEPS_REQUIRED or treatment_steps < MEASURED_STEPS_REQUIRED:
        return CLASSIFIER_HARNESS_OR_LIVENESS_FAIL

    guards = compare_arm_wiring_guards(oracle_receipt, treatment_receipt)
    if (
        float(guards["per_step_accumulator_l1_max_abs_delta"]) > 0.0
        or int(guards["per_step_crossing_bool_disagreement_count"]) > 0
        or float(guards["vote_update_state_accumulator_equality_rate"]) < 1.0
    ):
        return CLASSIFIER_W6_DYNAMICS_DIVERGES

    return CLASSIFIER_W6_HEADROOM_SUFFICIENT_PARITY_OK


def emit_s3bb_classifier_receipt(
    oracle_receipt: Mapping[str, Any],
    treatment_receipt: Mapping[str, Any],
    *,
    harness_failures: Sequence[str] | None = None,
) -> dict[str, Any]:
    primary = classify_s3bb_run(
        oracle_receipt,
        treatment_receipt,
        harness_failures=harness_failures,
    )
    guards = compare_arm_wiring_guards(oracle_receipt, treatment_receipt)
    return {
        "slice_id": "w6_gpu_dynamics_parity_run_s3bb_v0",
        "primary_classifier": primary,
        "classifier_precedence": list(CLASSIFIER_PRECEDENCE),
        "classifier_resolution_rules": "w6_narrow_carrier_gpu_dynamics_s3bb_launch_packet_v2.json",
        "harness_failures": list(dict.fromkeys(harness_failures or ())),
        "wiring_guards": guards,
        "oracle_steps_completed": int(oracle_receipt.get("steps_completed") or 0),
        "treatment_steps_completed": int(treatment_receipt.get("steps_completed") or 0),
        "treatment_stop_reason": str(treatment_receipt.get("stop_reason") or ""),
    }


def emit_s3bc_classifier_receipt(
    *,
    harness_failures: Sequence[str] | None = None,
    telemetry_tests_pass: bool = True,
    classifier_tests_pass: bool = True,
) -> dict[str, Any]:
    failures = list(dict.fromkeys(harness_failures or ()))
    if failures or not telemetry_tests_pass:
        primary = CLASSIFIER_S3BC_HARNESS_FAIL
    elif not classifier_tests_pass:
        primary = CLASSIFIER_S3BC_TELEMETRY_OR_CLASSIFIER_INCOMPLETE
    else:
        primary = CLASSIFIER_S3BC_HEADROOM_TELEMETRY_POSTRUN_OK
    return {
        "slice_id": "s3bc_headroom_telemetry_postrun_v0",
        "primary_classifier": primary,
        "classifier_precedence": list(CLASSIFIER_S3BC_PRECEDENCE),
        "harness_failures": failures,
        "satisfies_launch_packet_prereq": "s3bb_headroom_telemetry_postrun_prereq_v0",
    }


def validate_headroom_telemetry_block(telemetry: Mapping[str, Any]) -> None:
    missing = [field for field in REQUIRED_HEADROOM_TELEMETRY_FIELDS if field not in telemetry]
    if missing:
        raise ValueError(f"headroom_telemetry missing required fields: {missing}")
