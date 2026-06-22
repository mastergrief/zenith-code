#!/usr/bin/env python3
"""CPU scale smoke for compact R3 ledger emit vs legacy tolist+json witness."""
from __future__ import annotations

import argparse
import json
import resource
import threading
import time
from pathlib import Path
from typing import Any

import torch

from calm.hrm_text_158.native_full_stack.narrow_accumulator_codec import (
    pack_w6_lanes_to_bytes,
)
from scripts.hrm_text_158_bounded_delta_acquisition_probe import (
    build_r3_persistent_ledger_receipt,
)

METRICS_SCHEMA_VERSION = "hrm_text_158_r3_ledger_emit_scale_smoke/v1.2"
# gen-c banked tensor-wide: total_lane_count=234_881_024 across 32 modules.
DEFAULT_MODULES = 32
DEFAULT_LANES_PER_MODULE = 7_340_032
DEFAULT_TOTAL_LANES = DEFAULT_MODULES * DEFAULT_LANES_PER_MODULE
GEN_C_BANKED_TOTAL_LANE_COUNT = 234_881_024
LEGACY_UNSAFE_RATIO_TO_COMPACT = 4.0
MEASURED_PEAK_TOLERANCE_MIB = 64.0

GATES = {
    "compact_emit_transient_upper_bound_mib_lte": 512.0,
    "compact_receipt_file_size_mib_lte": 5.0,
    "compact_emit_wall_seconds_lte": 600.0,
    "legacy_to_compact_projected_rss_ratio_gte": LEGACY_UNSAFE_RATIO_TO_COMPACT,
    "compact_emit_measured_peak_within_analytic_tolerance": True,
}


def _read_rss_mib() -> float:
    status_path = Path("/proc/self/status")
    if status_path.is_file():
        for line in status_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return float(usage.ru_maxrss) / 1024.0


class _PhaseRssTracker:
    def __init__(self) -> None:
        self._baseline_mib = _read_rss_mib()
        self.peak_delta_mib = 0.0

    def sample(self) -> None:
        current_mib = _read_rss_mib()
        self.peak_delta_mib = max(self.peak_delta_mib, current_mib - self._baseline_mib)


class _SyntheticR3TensorState:
    """Minimal state surface for compact R3 receipt emit scale-smoke."""

    def __init__(
        self,
        *,
        q_levels: torch.Tensor,
        frozen_scale: torch.Tensor,
        accumulators_i16: torch.Tensor,
    ) -> None:
        self.q_levels = q_levels
        self.frozen_scale = frozen_scale
        self._accumulators_i16 = accumulators_i16

    def decoded_accumulators(self, *, rebuild_if_stale: bool = True) -> torch.Tensor:
        del rebuild_if_stale
        return self._accumulators_i16


def _lanes_to_2d_shape(lanes_per_module: int) -> tuple[int, int]:
    lanes = int(lanes_per_module)
    if lanes <= 0:
        raise ValueError("lanes_per_module must be positive")
    side = int(lanes**0.5)
    for out_features in range(side, 0, -1):
        if lanes % out_features == 0:
            return (out_features, lanes // out_features)
    return (1, lanes)


def _synthetic_flat_accumulators_i16(
    *,
    module_index: int,
    lanes_per_module: int,
) -> torch.Tensor:
    shape = _lanes_to_2d_shape(lanes_per_module)
    lanes = torch.arange(lanes_per_module, dtype=torch.int64)
    cold_default = ((module_index * 7) % 31) - 15
    flat = ((lanes + int(cold_default)) % 31 - 15).to(torch.int16)
    return flat.reshape(shape)


def _synthetic_tensor_state(
    *,
    module_index: int,
    lanes_per_module: int,
) -> _SyntheticR3TensorState:
    accumulators_i16 = _synthetic_flat_accumulators_i16(
        module_index=module_index,
        lanes_per_module=lanes_per_module,
    )
    q_levels = torch.zeros(accumulators_i16.shape, dtype=torch.int8)
    return _SyntheticR3TensorState(
        q_levels=q_levels,
        frozen_scale=torch.tensor(1.0, dtype=torch.float32),
        accumulators_i16=accumulators_i16,
    )


def _payload_bytes_for_lanes(lanes_per_module: int) -> int:
    return int((int(lanes_per_module) * 6 + 7) // 8)


def _analytic_compact_emit_transient_components(
    *,
    per_module_payload_bytes: list[int],
    total_lanes: int,
) -> dict[str, int]:
    """In-build simultaneous holdings during build_r3_persistent_ledger_receipt."""

    if not per_module_payload_bytes:
        return {
            "packed_payloads_simultaneous_bytes": 0,
            "max_hash_copy_bytes": 0,
            "q_levels_simultaneous_bytes": 0,
            "transient_upper_bound_bytes": 0,
        }
    simultaneous_packed_bytes = sum(int(value) for value in per_module_payload_bytes)
    max_hash_copy_bytes = max(int(value) for value in per_module_payload_bytes)
    # qscale_states retains detached int8 q_levels for every module until measure_r3.
    q_levels_simultaneous_bytes = int(total_lanes)
    transient_upper_bound_bytes = (
        simultaneous_packed_bytes + max_hash_copy_bytes + q_levels_simultaneous_bytes
    )
    return {
        "packed_payloads_simultaneous_bytes": simultaneous_packed_bytes,
        "max_hash_copy_bytes": max_hash_copy_bytes,
        "q_levels_simultaneous_bytes": q_levels_simultaneous_bytes,
        "transient_upper_bound_bytes": transient_upper_bound_bytes,
    }


def _analytic_compact_emit_transient_upper_bound_mib(
    *,
    per_module_payload_bytes: list[int],
    total_lanes: int,
) -> float:
    """Upper bound: packed payloads + one hash-copy + simultaneous q_levels copies."""

    components = _analytic_compact_emit_transient_components(
        per_module_payload_bytes=per_module_payload_bytes,
        total_lanes=total_lanes,
    )
    return float(components["transient_upper_bound_bytes"]) / (1024.0 * 1024.0)


def _build_receipt_with_measured_peak_rss(
    tensor_states: dict[str, _SyntheticR3TensorState],
) -> tuple[dict[str, Any], float, float]:
    """Run compact emit under background VmRSS polling for true in-build high-water."""

    baseline_mib = _read_rss_mib()
    peak_mib = baseline_mib
    stop_event = threading.Event()

    def _poll_rss() -> None:
        nonlocal peak_mib
        while not stop_event.is_set():
            peak_mib = max(peak_mib, _read_rss_mib())
            time.sleep(0.01)

    poll_thread = threading.Thread(target=_poll_rss, daemon=True)
    poll_thread.start()
    try:
        receipt = build_r3_persistent_ledger_receipt(
            tensor_states,
            byte_packed_enabled=True,
        )
    finally:
        stop_event.set()
        poll_thread.join(timeout=2.0)
    measured_peak_delta_mib = float(peak_mib - baseline_mib)
    return receipt, measured_peak_delta_mib, baseline_mib


def _legacy_json_witness_one_module(
    *,
    lanes_per_module: int,
) -> dict[str, float | int | str]:
    """Measure real legacy tolist+json peak-RSS on one representative module."""

    shape = _lanes_to_2d_shape(lanes_per_module)
    lanes = torch.arange(lanes_per_module, dtype=torch.int64)
    acc = ((lanes + 3) % 31 - 15).to(torch.int16).reshape(shape)
    payload = pack_w6_lanes_to_bytes(acc)
    tracker = _PhaseRssTracker()
    wall_start = time.perf_counter()
    packed_list = payload.packed.detach().cpu().tolist()
    tracker.sample()
    artifact_blob = {
        "schema": "r3_w6_byte_packed_checkpoint_artifact_probe/v0",
        "tensor_payloads": [
            {
                "logical_shape": list(payload.logical_shape),
                "logical_numel": int(payload.logical_numel),
                "packed_bytes": packed_list,
            }
        ],
    }
    encoded = json.dumps(artifact_blob, separators=(",", ":"), sort_keys=True)
    wall_seconds = time.perf_counter() - wall_start
    tracker.sample()
    measured_peak_mib = float(tracker.peak_delta_mib)
    # CPython interns small uint8 ints; RSS delta can read as 0 on a single sample.
    # Conservative fallback: interned-pointer list (~8 B/elem) + json string bytes.
    conservative_peak_mib = (
        float(payload.packed.numel()) * 8.0 + float(len(encoded))
    ) / (1024.0 * 1024.0)
    effective_peak_mib = max(measured_peak_mib, conservative_peak_mib)
    return {
        "lanes_per_module": int(lanes_per_module),
        "legacy_json_bytes_one_module": len(encoded.encode("utf-8")),
        "legacy_peak_rss_delta_mib_one_module": measured_peak_mib,
        "legacy_peak_rss_delta_mib_one_module_conservative": conservative_peak_mib,
        "legacy_peak_rss_delta_mib_one_module_effective": effective_peak_mib,
        "legacy_wall_seconds_one_module": float(wall_seconds),
        "cpython_int_interning_caveat": (
            "uint8 .tolist() yields interned small-int pointers (~8 B/elem), "
            "not full 28 B/non-interned int objects; projection uses measured "
            "micro-sample with conservative fallback."
        ),
    }


def run_scale_smoke(
    *,
    output_dir: Path,
    modules: int,
    lanes_per_module: int,
    gate_overrides: dict[str, float] | None = None,
) -> dict[str, Any]:
    effective_gates = dict(GATES)
    if gate_overrides:
        effective_gates.update(gate_overrides)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_lanes = int(modules) * int(lanes_per_module)
    legacy_micro = _legacy_json_witness_one_module(lanes_per_module=lanes_per_module)
    projected_legacy_rss_mib = (
        float(legacy_micro["legacy_peak_rss_delta_mib_one_module_effective"]) * float(modules)
    )
    projected_legacy_json_mib = (
        float(legacy_micro["legacy_json_bytes_one_module"]) * float(modules) / (1024.0 * 1024.0)
    )

    per_module_payload_bytes_theoretical = [
        _payload_bytes_for_lanes(lanes_per_module) for _ in range(modules)
    ]
    analytic_components_theoretical = _analytic_compact_emit_transient_components(
        per_module_payload_bytes=per_module_payload_bytes_theoretical,
        total_lanes=total_lanes,
    )
    analytic_transient_upper_bound_mib = float(
        analytic_components_theoretical["transient_upper_bound_bytes"]
    ) / (1024.0 * 1024.0)

    tensor_states = {
        f"model.module_{module_index}": _synthetic_tensor_state(
            module_index=module_index,
            lanes_per_module=lanes_per_module,
        )
        for module_index in range(modules)
    }
    compact_wall_start = time.perf_counter()
    receipt, compact_emit_measured_peak_rss_mib, compact_emit_baseline_rss_mib = (
        _build_receipt_with_measured_peak_rss(tensor_states)
    )
    receipt_path = output_dir / "r3_compact_receipt.json"
    receipt_encoded = json.dumps(receipt, separators=(",", ":"), sort_keys=True)
    receipt_path.write_text(receipt_encoded, encoding="utf-8")
    compact_wall_seconds = time.perf_counter() - compact_wall_start

    post_emit_tracker = _PhaseRssTracker()
    post_emit_tracker.sample()
    compact_post_emit_rss_delta_mib = float(post_emit_tracker.peak_delta_mib)
    compact_receipt_mib = receipt_path.stat().st_size / (1024.0 * 1024.0)

    per_module_rows = receipt.get("r3_per_module_payload_rows") or []
    per_module_payload_bytes_from_receipt = [
        int(row["payload_bytes"]) for row in per_module_rows
    ]
    analytic_components_from_receipt = _analytic_compact_emit_transient_components(
        per_module_payload_bytes=per_module_payload_bytes_from_receipt,
        total_lanes=total_lanes,
    )
    analytic_from_receipt_mib = float(
        analytic_components_from_receipt["transient_upper_bound_bytes"]
    ) / (1024.0 * 1024.0)
    compact_emit_transient_upper_bound_mib = max(
        analytic_transient_upper_bound_mib,
        analytic_from_receipt_mib,
    )
    measured_within_analytic_tolerance = (
        compact_emit_measured_peak_rss_mib
        <= compact_emit_transient_upper_bound_mib + MEASURED_PEAK_TOLERANCE_MIB
    )

    legacy_to_compact_ratio = (
        projected_legacy_rss_mib / compact_emit_transient_upper_bound_mib
        if compact_emit_transient_upper_bound_mib > 0.0
        else float("inf")
    )
    gate_results = {
        "compact_emit_transient_upper_bound_mib_lte": compact_emit_transient_upper_bound_mib
        <= float(effective_gates["compact_emit_transient_upper_bound_mib_lte"]),
        "compact_receipt_file_size_mib_lte": compact_receipt_mib
        <= float(effective_gates["compact_receipt_file_size_mib_lte"]),
        "compact_emit_wall_seconds_lte": compact_wall_seconds
        <= float(effective_gates["compact_emit_wall_seconds_lte"]),
        "legacy_to_compact_projected_rss_ratio_gte": legacy_to_compact_ratio
        >= float(effective_gates["legacy_to_compact_projected_rss_ratio_gte"]),
        "compact_emit_measured_peak_within_analytic_tolerance": measured_within_analytic_tolerance,
    }
    metrics = {
        "schema_version": METRICS_SCHEMA_VERSION,
        "modules": int(modules),
        "lanes_per_module": int(lanes_per_module),
        "total_lanes": total_lanes,
        "gen_c_banked_total_lane_count": GEN_C_BANKED_TOTAL_LANE_COUNT,
        "scale_matches_gen_c_banked_total_lanes": total_lanes == GEN_C_BANKED_TOTAL_LANE_COUNT,
        "gates": effective_gates,
        "gate_results": gate_results,
        "all_gates_pass": all(gate_results.values()),
        "legacy_micro_sample": legacy_micro,
        "projected_legacy_peak_rss_mib_tensor_wide": projected_legacy_rss_mib,
        "projected_legacy_json_mib_tensor_wide": projected_legacy_json_mib,
        "compact_emit_transient_upper_bound_mib": compact_emit_transient_upper_bound_mib,
        "compact_emit_transient_upper_bound_mib_theoretical": analytic_transient_upper_bound_mib,
        "compact_emit_transient_upper_bound_mib_from_receipt": analytic_from_receipt_mib,
        "compact_emit_transient_components_theoretical": analytic_components_theoretical,
        "compact_emit_transient_components_from_receipt": analytic_components_from_receipt,
        "compact_emit_measured_peak_rss_mib": compact_emit_measured_peak_rss_mib,
        "compact_emit_measured_peak_rss_baseline_mib": compact_emit_baseline_rss_mib,
        "compact_emit_measured_peak_rss_semantics": (
            "Background VmRSS poll every 10ms during build_r3_persistent_ledger_receipt; "
            "baseline taken immediately before the in-build call (after tensor_states exist)."
        ),
        "compact_emit_measured_peak_analytic_tolerance_mib": MEASURED_PEAK_TOLERANCE_MIB,
        "compact_post_emit_rss_delta_mib": compact_post_emit_rss_delta_mib,
        "compact_post_emit_rss_delta_semantics": (
            "Secondary observation only: current-VmRSS delta after build+receipt write; "
            "NOT the primary gate (transient packed payloads may already be freed)."
        ),
        "compact_receipt_file_size_mib": compact_receipt_mib,
        "compact_emit_wall_seconds": compact_wall_seconds,
        "legacy_to_compact_projected_rss_ratio": legacy_to_compact_ratio,
        "receipt_path": str(receipt_path),
        "r3_packed_payload_content_sha256": receipt.get("r3_packed_payload_content_sha256"),
        "r3_actual_acc_payload_bytes": receipt.get("r3_actual_acc_payload_bytes"),
        "r3_per_module_payload_row_count": len(per_module_rows),
    }
    metrics_path = output_dir / "r3_ledger_emit_scale_smoke_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--modules", type=int, default=DEFAULT_MODULES)
    parser.add_argument("--lanes-per-module", type=int, default=DEFAULT_LANES_PER_MODULE)
    args = parser.parse_args()
    metrics = run_scale_smoke(
        output_dir=args.output_dir,
        modules=int(args.modules),
        lanes_per_module=int(args.lanes_per_module),
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    if not bool(metrics["all_gates_pass"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
