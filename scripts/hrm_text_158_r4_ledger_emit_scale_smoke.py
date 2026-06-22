#!/usr/bin/env python3
"""CPU scale smoke for compact R4 combined q+acc ledger emit vs legacy tolist+json."""
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
from calm.hrm_text_158.native_full_stack.persistent_state_budget import (
    pack_ternary_q_2bit_reference,
)
from scripts.hrm_text_158_bounded_delta_acquisition_probe import (
    build_r4_persistent_ledger_receipt,
)

METRICS_SCHEMA_VERSION = "hrm_text_158_r4_ledger_emit_scale_smoke/v1"
DEFAULT_MODULES = 32
DEFAULT_LANES_PER_MODULE = 7_340_032
DEFAULT_TOTAL_LANES = DEFAULT_MODULES * DEFAULT_LANES_PER_MODULE
GEN_C_BANKED_TOTAL_LANE_COUNT = 234_881_024
LEGACY_UNSAFE_RATIO_TO_COMPACT = 4.0
MEASURED_PEAK_TOLERANCE_MIB = 64.0
R4_INCLUSIVE_BPW_NEAR = 8.0003
R4_INCLUSIVE_BPW_CEILING = 8.5
R4_Q_BPW_TARGET = 2.0
R4_ACC_BPW_TARGET = 6.0
R4_Q_PAYLOAD_MIB_TARGET = 56.0
R4_ACC_PAYLOAD_MIB_TARGET = 168.0
R4_PAYLOAD_MIB_TOLERANCE = 2.0

GATES = {
    "emit_only_transient_upper_bound_mib_lte": 512.0,
    "compact_receipt_file_size_mib_lte": 10.0,
    "compact_emit_wall_seconds_lte": 600.0,
    "legacy_to_compact_projected_rss_ratio_gte": LEGACY_UNSAFE_RATIO_TO_COMPACT,
    "compact_emit_measured_peak_within_analytic_tolerance": True,
    "r4_ledger_pass": True,
    "inclusive_bpw_lte_ceiling": True,
    "zero_raw_byte_lists_in_receipt": True,
}


def _read_rss_mib() -> float:
    status_path = Path("/proc/self/status")
    if status_path.is_file():
        for line in status_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return float(usage.ru_maxrss) / 1024.0


class _SyntheticR4TensorState:
    """Minimal state surface for compact R4 receipt emit scale-smoke."""

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


def _synthetic_flat_q_levels(*, module_index: int, lanes_per_module: int) -> torch.Tensor:
    shape = _lanes_to_2d_shape(lanes_per_module)
    lanes = torch.arange(lanes_per_module, dtype=torch.int64)
    pattern = ((lanes + int(module_index) * 3) % 3 - 1).to(torch.int8)
    return pattern.reshape(shape)


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
) -> _SyntheticR4TensorState:
    q_levels = _synthetic_flat_q_levels(
        module_index=module_index,
        lanes_per_module=lanes_per_module,
    )
    accumulators_i16 = _synthetic_flat_accumulators_i16(
        module_index=module_index,
        lanes_per_module=lanes_per_module,
    )
    return _SyntheticR4TensorState(
        q_levels=q_levels,
        frozen_scale=torch.tensor(1.0 + module_index * 0.01, dtype=torch.float32),
        accumulators_i16=accumulators_i16,
    )


def _q_payload_bytes_for_lanes(lanes_per_module: int) -> int:
    return int((int(lanes_per_module) + 3) // 4)


def _acc_payload_bytes_for_lanes(lanes_per_module: int) -> int:
    return int((int(lanes_per_module) * 6 + 7) // 8)


def _analytic_q_packer_intermediate_bytes(lanes_per_module: int) -> dict[str, int]:
    """Peak simultaneous holdings inside pack_ternary_q_2bit_reference for one module."""

    lanes = int(lanes_per_module)
    padded_codes = ((lanes + 3) // 4) * 4
    codes_i16_bytes = padded_codes * 2
    packed_i16_bytes = ((lanes + 3) // 4) * 2
    packed_uint8_bytes = _q_payload_bytes_for_lanes(lanes)
    simultaneous_bytes = codes_i16_bytes + packed_i16_bytes + packed_uint8_bytes
    return {
        "codes_i16_bytes": codes_i16_bytes,
        "packed_i16_bytes": packed_i16_bytes,
        "packed_uint8_bytes": packed_uint8_bytes,
        "simultaneous_bytes": simultaneous_bytes,
    }


def _analytic_w6_packer_intermediate_bytes(lanes_per_module: int) -> dict[str, int]:
    """Peak simultaneous holdings inside pack_w6_lanes_to_bytes for one module."""

    lanes = int(lanes_per_module)
    acc_payload_bytes = _acc_payload_bytes_for_lanes(lanes)
    int32_values_bytes = lanes * 4
    bool_mask_bytes = lanes * 1
    int16_lane_tensor_bytes = lanes * 2
    pylist_int64_pointer_bytes = lanes * 8
    bytearray_bytes = acc_payload_bytes
    list_out_bytes = acc_payload_bytes
    packed_uint8_bytes = acc_payload_bytes
    simultaneous_bytes = (
        int32_values_bytes
        + bool_mask_bytes
        + int16_lane_tensor_bytes
        + pylist_int64_pointer_bytes
        + bytearray_bytes
        + list_out_bytes
        + packed_uint8_bytes
    )
    return {
        "int32_values_bytes": int32_values_bytes,
        "bool_mask_bytes": bool_mask_bytes,
        "int16_lane_tensor_bytes": int16_lane_tensor_bytes,
        "pylist_int64_pointer_bytes": pylist_int64_pointer_bytes,
        "bytearray_bytes": bytearray_bytes,
        "list_out_bytes": list_out_bytes,
        "packed_uint8_bytes": packed_uint8_bytes,
        "simultaneous_bytes": simultaneous_bytes,
    }


def _analytic_emit_component_model(
    *,
    modules: int,
    lanes_per_module: int,
) -> dict[str, Any]:
    total_lanes = int(modules) * int(lanes_per_module)
    per_module_q_payload = _q_payload_bytes_for_lanes(lanes_per_module)
    per_module_acc_payload = _acc_payload_bytes_for_lanes(lanes_per_module)
    retained_payload_only_bytes = int(modules) * (per_module_q_payload + per_module_acc_payload)

    q_packer = _analytic_q_packer_intermediate_bytes(lanes_per_module)
    w6_packer = _analytic_w6_packer_intermediate_bytes(lanes_per_module)
    packer_intermediate_max_bytes = max(
        int(q_packer["simultaneous_bytes"]),
        int(w6_packer["simultaneous_bytes"]),
    )
    max_hash_copy_bytes = max(per_module_q_payload, per_module_acc_payload)
    qscale_q_levels_retained_bytes = total_lanes

    q_levels_bytes = total_lanes
    acc_i16_bytes = total_lanes * 2
    frozen_scale_bytes = int(modules) * 4
    synthetic_input_allocation_bytes = q_levels_bytes + acc_i16_bytes + frozen_scale_bytes

    emit_only_transient_upper_bound_bytes = (
        retained_payload_only_bytes
        + packer_intermediate_max_bytes
        + max_hash_copy_bytes
        + qscale_q_levels_retained_bytes
    )
    conservative_total_bound_bytes = (
        synthetic_input_allocation_bytes + emit_only_transient_upper_bound_bytes
    )

    def _to_mib(value: int) -> float:
        return float(value) / (1024.0 * 1024.0)

    return {
        "modules": int(modules),
        "lanes_per_module": int(lanes_per_module),
        "total_lanes": total_lanes,
        "per_module_q_payload_bytes": per_module_q_payload,
        "per_module_acc_payload_bytes": per_module_acc_payload,
        "retained_payload_only_bytes": retained_payload_only_bytes,
        "retained_payload_only_mib": _to_mib(retained_payload_only_bytes),
        "q_packer_intermediate_components": q_packer,
        "w6_packer_intermediate_components": w6_packer,
        "packer_intermediate_max_bytes": packer_intermediate_max_bytes,
        "packer_intermediate_max_mib": _to_mib(packer_intermediate_max_bytes),
        "max_hash_copy_bytes": max_hash_copy_bytes,
        "max_hash_copy_mib": _to_mib(max_hash_copy_bytes),
        "qscale_q_levels_retained_bytes": qscale_q_levels_retained_bytes,
        "qscale_q_levels_retained_mib": _to_mib(qscale_q_levels_retained_bytes),
        "synthetic_input_allocation_bytes": synthetic_input_allocation_bytes,
        "synthetic_input_allocation_mib": _to_mib(synthetic_input_allocation_bytes),
        "synthetic_input_components": {
            "q_levels_int8_bytes": q_levels_bytes,
            "accumulators_i16_bytes": acc_i16_bytes,
            "frozen_scale_fp32_bytes": frozen_scale_bytes,
        },
        "emit_only_transient_upper_bound_bytes": emit_only_transient_upper_bound_bytes,
        "emit_only_transient_upper_bound_mib": _to_mib(emit_only_transient_upper_bound_bytes),
        "conservative_total_bound_bytes": conservative_total_bound_bytes,
        "conservative_total_bound_mib": _to_mib(conservative_total_bound_bytes),
        "component_model_note": (
            "retained_payload_only_mib is final packed q+acc bytes only; "
            "emit_only_transient_upper_bound_mib also includes packer_intermediate_max, "
            "max hash copy, and qscale_state detached q_levels retained during emit; "
            "conservative_total_bound_mib adds pre-emit synthetic tensor allocations."
        ),
    }


def _build_receipt_with_measured_peak_rss(
    tensor_states: dict[str, _SyntheticR4TensorState],
) -> tuple[dict[str, Any], float, float, float]:
    """Run compact R4 emit under background VmRSS polling (emit call only)."""

    baseline_mib = _read_rss_mib()
    peak_mib = baseline_mib
    stop_event = threading.Event()

    def _poll_rss() -> None:
        nonlocal peak_mib
        while not stop_event.is_set():
            peak_mib = max(peak_mib, _read_rss_mib())
            time.sleep(0.01)

    poll_thread = threading.Thread(target=_poll_rss, daemon=True)
    emit_wall_start = time.perf_counter()
    poll_thread.start()
    try:
        receipt = build_r4_persistent_ledger_receipt(
            tensor_states,
            q_packed_enabled=True,
            acc_byte_packed_enabled=True,
        )
    finally:
        stop_event.set()
        poll_thread.join(timeout=2.0)
    emit_wall_seconds = time.perf_counter() - emit_wall_start
    measured_peak_delta_mib = float(peak_mib - baseline_mib)
    return receipt, measured_peak_delta_mib, baseline_mib, emit_wall_seconds


def _looks_like_raw_byte_list(value: Any) -> bool:
    if not isinstance(value, list):
        return False
    if len(value) < 64:
        return False
    if not all(isinstance(item, int) for item in value):
        return False
    return all(0 <= int(item) <= 255 for item in value)


def _find_raw_byte_lists(obj: Any, *, path: str = "") -> list[str]:
    hits: list[str] = []
    if _looks_like_raw_byte_list(obj):
        hits.append(path or "<root>")
        return hits
    if isinstance(obj, dict):
        for key, value in obj.items():
            child_path = f"{path}.{key}" if path else str(key)
            hits.extend(_find_raw_byte_lists(value, path=child_path))
    elif isinstance(obj, list) and not _looks_like_raw_byte_list(obj):
        for index, value in enumerate(obj):
            child_path = f"{path}[{index}]"
            hits.extend(_find_raw_byte_lists(value, path=child_path))
    return hits


def _legacy_json_witness_one_module(
    *,
    lanes_per_module: int,
) -> dict[str, float | int | str]:
    """Measure real legacy tolist+json peak-RSS on one representative module (q+acc)."""

    shape = _lanes_to_2d_shape(lanes_per_module)
    lanes = torch.arange(lanes_per_module, dtype=torch.int64)
    q = ((lanes + 1) % 3 - 1).to(torch.int8).reshape(shape)
    acc = ((lanes + 3) % 31 - 15).to(torch.int16).reshape(shape)
    baseline_mib = _read_rss_mib()
    peak_mib = baseline_mib
    stop_event = threading.Event()

    def _poll_rss() -> None:
        nonlocal peak_mib
        while not stop_event.is_set():
            peak_mib = max(peak_mib, _read_rss_mib())
            time.sleep(0.01)

    poll_thread = threading.Thread(target=_poll_rss, daemon=True)
    wall_start = time.perf_counter()
    poll_thread.start()
    try:
        packed_q = pack_ternary_q_2bit_reference(q)
        q_list = packed_q.packed.detach().cpu().tolist()
        packed_acc = pack_w6_lanes_to_bytes(acc)
        acc_list = packed_acc.packed.detach().cpu().tolist()
        artifact_blob = {
            "schema": "r4_legacy_byte_packed_checkpoint_artifact_probe/v0",
            "tensor_payloads": [
                {
                    "logical_shape": list(packed_q.logical_shape),
                    "logical_numel": int(packed_q.logical_numel),
                    "packed_q_bytes": q_list,
                },
                {
                    "logical_shape": list(packed_acc.logical_shape),
                    "logical_numel": int(packed_acc.logical_numel),
                    "packed_acc_bytes": acc_list,
                },
            ],
        }
        encoded = json.dumps(artifact_blob, separators=(",", ":"), sort_keys=True)
    finally:
        stop_event.set()
        poll_thread.join(timeout=2.0)
    wall_seconds = time.perf_counter() - wall_start
    measured_peak_mib = float(peak_mib - baseline_mib)
    conservative_peak_mib = (
        float(packed_q.packed.numel() + packed_acc.packed.numel()) * 8.0
        + float(len(encoded))
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


def _one_module_w6_pack_wall_seconds(lanes_per_module: int) -> float:
    shape = _lanes_to_2d_shape(lanes_per_module)
    lanes = torch.arange(lanes_per_module, dtype=torch.int64)
    acc = ((lanes + 3) % 31 - 15).to(torch.int16).reshape(shape)
    wall_start = time.perf_counter()
    pack_w6_lanes_to_bytes(acc)
    return float(time.perf_counter() - wall_start)


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
    component_model = _analytic_emit_component_model(
        modules=modules,
        lanes_per_module=lanes_per_module,
    )
    legacy_micro = _legacy_json_witness_one_module(lanes_per_module=lanes_per_module)
    projected_legacy_rss_mib = (
        float(legacy_micro["legacy_peak_rss_delta_mib_one_module_effective"]) * float(modules)
    )
    projected_legacy_json_mib = (
        float(legacy_micro["legacy_json_bytes_one_module"]) * float(modules) / (1024.0 * 1024.0)
    )

    synthetic_wall_start = time.perf_counter()
    tensor_states = {
        f"model.module_{module_index}": _synthetic_tensor_state(
            module_index=module_index,
            lanes_per_module=lanes_per_module,
        )
        for module_index in range(modules)
    }
    synthetic_construction_wall_seconds = time.perf_counter() - synthetic_wall_start

    receipt, compact_emit_measured_peak_rss_mib, compact_emit_baseline_rss_mib, emit_wall_seconds = (
        _build_receipt_with_measured_peak_rss(tensor_states)
    )
    full_scale_wall_seconds = synthetic_construction_wall_seconds + emit_wall_seconds

    receipt_path = output_dir / "r4_compact_receipt.json"
    receipt_encoded = json.dumps(receipt, separators=(",", ":"), sort_keys=True)
    receipt_path.write_text(receipt_encoded, encoding="utf-8")
    compact_receipt_mib = receipt_path.stat().st_size / (1024.0 * 1024.0)

    raw_byte_list_hits = _find_raw_byte_lists(receipt)
    q_rows = receipt.get("r4_per_module_q_rows") or []
    acc_rows = receipt.get("r4_per_module_acc_rows") or []

    emit_only_transient_upper_bound_mib = float(component_model["emit_only_transient_upper_bound_mib"])
    measured_within_analytic_tolerance = (
        compact_emit_measured_peak_rss_mib
        <= emit_only_transient_upper_bound_mib + MEASURED_PEAK_TOLERANCE_MIB
    )
    legacy_to_compact_ratio = (
        projected_legacy_rss_mib / emit_only_transient_upper_bound_mib
        if emit_only_transient_upper_bound_mib > 0.0
        else float("inf")
    )

    q_bpw = float(receipt.get("r4_q_physical_bits_per_weight") or -1.0)
    acc_bpw = float(receipt.get("r4_acc_physical_bits_per_weight") or -1.0)
    inclusive_bpw = float(receipt.get("r4_checkpoint_inclusive_physical_bits_per_weight") or 999.0)
    r4_ledger_pass = bool(receipt.get("r4_ledger_pass"))
    q_payload_mib = float(receipt.get("r4_actual_q_payload_bytes") or 0) / (1024.0 * 1024.0)
    acc_payload_mib = float(receipt.get("r4_actual_acc_payload_bytes") or 0) / (1024.0 * 1024.0)

    wall_projection: dict[str, Any] | None = None
    wall_gate_value = full_scale_wall_seconds
    wall_gate_pass = wall_gate_value <= float(effective_gates["compact_emit_wall_seconds_lte"])
    r4_1_checkpoint_serialization_cleared = True
    if not wall_gate_pass:
        one_module_w6_wall = _one_module_w6_pack_wall_seconds(lanes_per_module)
        projected_full_wall = synthetic_construction_wall_seconds + (one_module_w6_wall * float(modules))
        wall_projection = {
            "mode": "analytic_projection_only",
            "one_module_w6_pack_wall_seconds": one_module_w6_wall,
            "projected_full_scale_wall_seconds": projected_full_wall,
            "r4_1_checkpoint_serialization_cleared": False,
            "note": (
                "Full-scale wall exceeded 600s; one-module W6 packer projection recorded. "
                "R4.1 NOT fully cleared on wall — memory gates may still pass."
            ),
        }
        r4_1_checkpoint_serialization_cleared = False

    primary_bound_pass = emit_only_transient_upper_bound_mib <= float(
        effective_gates["emit_only_transient_upper_bound_mib_lte"]
    )
    primary_measured_pass = compact_emit_measured_peak_rss_mib <= float(
        effective_gates["emit_only_transient_upper_bound_mib_lte"]
    )
    primary_gate_pass = primary_bound_pass or primary_measured_pass
    measured_peak_hard_fail = compact_emit_measured_peak_rss_mib > float(
        effective_gates["emit_only_transient_upper_bound_mib_lte"]
    )

    gate_results = {
        "emit_only_transient_upper_bound_mib_lte": primary_bound_pass,
        "emit_measured_peak_mib_lte": primary_measured_pass,
        "primary_memory_gate_pass": primary_gate_pass,
        "measured_peak_hard_fail_over_512": measured_peak_hard_fail,
        "compact_receipt_file_size_mib_lte": compact_receipt_mib
        <= float(effective_gates["compact_receipt_file_size_mib_lte"]),
        "compact_emit_wall_seconds_lte": wall_gate_pass,
        "legacy_to_compact_projected_rss_ratio_gte": legacy_to_compact_ratio
        >= float(effective_gates["legacy_to_compact_projected_rss_ratio_gte"]),
        "compact_emit_measured_peak_within_analytic_tolerance": measured_within_analytic_tolerance,
        "r4_ledger_pass": r4_ledger_pass,
        "inclusive_bpw_lte_ceiling": inclusive_bpw <= R4_INCLUSIVE_BPW_CEILING,
        "zero_raw_byte_lists_in_receipt": len(raw_byte_list_hits) == 0,
    }
    memory_gates_pass = (
        primary_gate_pass
        and not measured_peak_hard_fail
        and gate_results["legacy_to_compact_projected_rss_ratio_gte"]
        and gate_results["compact_emit_measured_peak_within_analytic_tolerance"]
        and gate_results["r4_ledger_pass"]
        and gate_results["inclusive_bpw_lte_ceiling"]
        and gate_results["zero_raw_byte_lists_in_receipt"]
        and gate_results["compact_receipt_file_size_mib_lte"]
    )
    all_gates_pass = memory_gates_pass and wall_gate_pass

    metrics = {
        "schema_version": METRICS_SCHEMA_VERSION,
        "modules": int(modules),
        "lanes_per_module": int(lanes_per_module),
        "total_lanes": total_lanes,
        "gen_c_banked_total_lane_count": GEN_C_BANKED_TOTAL_LANE_COUNT,
        "scale_matches_gen_c_banked_total_lanes": total_lanes == GEN_C_BANKED_TOTAL_LANE_COUNT,
        "gates": effective_gates,
        "gate_results": gate_results,
        "memory_gates_pass": memory_gates_pass,
        "all_gates_pass": all_gates_pass,
        "r4_1_checkpoint_serialization_cleared": r4_1_checkpoint_serialization_cleared,
        "emit_component_model": component_model,
        "legacy_micro_sample": legacy_micro,
        "projected_legacy_peak_rss_mib_tensor_wide": projected_legacy_rss_mib,
        "projected_legacy_json_mib_tensor_wide": projected_legacy_json_mib,
        "legacy_to_compact_projected_rss_ratio": legacy_to_compact_ratio,
        "synthetic_construction_wall_seconds": synthetic_construction_wall_seconds,
        "emit_wall_seconds": emit_wall_seconds,
        "full_scale_wall_seconds": full_scale_wall_seconds,
        "wall_projection": wall_projection,
        "compact_emit_measured_peak_rss_mib": compact_emit_measured_peak_rss_mib,
        "compact_emit_measured_peak_rss_baseline_mib": compact_emit_baseline_rss_mib,
        "compact_emit_measured_peak_rss_semantics": (
            "Background VmRSS poll every 10ms during build_r4_persistent_ledger_receipt only; "
            "synthetic tensor_states construction excluded and separately timed."
        ),
        "compact_emit_measured_peak_analytic_tolerance_mib": MEASURED_PEAK_TOLERANCE_MIB,
        "compact_receipt_file_size_mib": compact_receipt_mib,
        "raw_byte_list_hits": raw_byte_list_hits,
        "r4_per_module_q_row_count": len(q_rows),
        "r4_per_module_acc_row_count": len(acc_rows),
        "r4_q_physical_bits_per_weight": q_bpw,
        "r4_acc_physical_bits_per_weight": acc_bpw,
        "r4_checkpoint_inclusive_physical_bits_per_weight": inclusive_bpw,
        "r4_actual_q_payload_mib": q_payload_mib,
        "r4_actual_acc_payload_mib": acc_payload_mib,
        "r4_ledger_pass": r4_ledger_pass,
        "r4_q_packed_content_sha256": receipt.get("r4_q_packed_content_sha256"),
        "r4_acc_packed_content_sha256": receipt.get("r4_acc_packed_content_sha256"),
        "receipt_path": str(receipt_path),
        "explicit_non_claims": [
            "cpu_emit_memory_safety_smoke_only",
            "not_science_parity_sub2_gpu_hot_path",
            "not_checkpoint_seam_roundtrip",
            "not_real_pt_weights",
        ],
    }
    metrics_path = output_dir / "r4_ledger_emit_scale_smoke_metrics.json"
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
