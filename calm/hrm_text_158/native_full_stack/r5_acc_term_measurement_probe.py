"""Read-only R5 acc-term measurement probe over banked accumulator state."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from calm.hrm_text_158.native_full_stack.acc_width_recorded_row_sweep import (
    CANONICAL_VOTE_UPDATE_THRESHOLD_ABS,
    DEFAULT_HEADROOM_FACTOR,
    headroom_passes,
)
from calm.hrm_text_158.native_full_stack.bounded_delta_accumulator import (
    project_bounded_delta_accumulator_bpw,
)
from calm.hrm_text_158.native_full_stack.narrow_accumulator_codec import (
    pack_w6_lanes_to_bytes,
)
from calm.hrm_text_158.native_full_stack.persistent_state_budget import (
    build_r3_per_module_payload_rows,
    canonical_r3_packed_payload_content_sha256,
)
from calm.hrm_text_158.native_full_stack.s3bb_headroom_telemetry import (
    _iter_sidecar_records,
)
from calm.hrm_text_158.native_full_stack.two_tier_carry_reducers import signed_w_max

R5_PROBE_SCHEMA_VERSION = "hrm_text_158_r5_acc_term_measurement_probe/v1"
SPARSE_BEATS_DENSE_MARGIN_BPW = 0.25
THRESHOLD_ABS = int(CANONICAL_VOTE_UPDATE_THRESHOLD_ABS)
WIDTH_GRID: tuple[int, ...] = (6, 5, 4, 3, 2)
DENSE_WIDTH_TO_LANE_BITS: dict[int, int] = {6: 6, 5: 5, 4: 4, 3: 3, 2: 2}

BRANCH_HARNESS_FAIL = "HARNESS_FAIL"
BRANCH_READ_PATH_FAIL = "READ_PATH_FAIL"
BRANCH_B_SPARSE_LOSSLESS_WINS = "R5_ACC_B_SPARSE_LOSSLESS_WINS"
BRANCH_A1_DENSE_LOSSLESS = "R5_ACC_A1_DENSE_W{n}_LOSSLESS"
BRANCH_A0_DENSE_W5_ONLY = "R5_ACC_A0_DENSE_W5_ONLY"
BRANCH_C_LOSSY_DECISION_PARITY = "R5_ACC_C_LOSSY_DECISION_PARITY_CANDIDATE"
BRANCH_D_REPRESENTATION_LIMIT = "R5_ACC_D_REPRESENTATION_LIMIT"

PARITY_LOSSLESS = "lossless_parity"
PARITY_DECISION = "decision_parity"

EXPLICIT_NON_CLAIMS: tuple[str, ...] = (
    "read_only_shape_measurement_only",
    "no_packer_implementation",
    "no_dynamics_parity",
    "no_bank",
    "no_sub2_claim",
    "no_readiness_claim",
    "no_hot_path_claim",
    "no_pt_mutation_or_commit",
    "no_raw_per_lane_arrays",
    "no_decision_surface_claim_from_static_probe",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exact_packed_bpw(width: int, lanes: int) -> float:
    lane_bits = DENSE_WIDTH_TO_LANE_BITS[int(width)]
    payload_bytes = (int(lanes) * lane_bits + 7) // 8
    if lanes <= 0:
        raise ValueError("lanes must be positive")
    return 8.0 * float(payload_bytes) / float(lanes)


def next_gate_parity_type(width: int) -> str:
    return (
        PARITY_LOSSLESS
        if signed_w_max(int(width)) >= THRESHOLD_ABS
        else PARITY_DECISION
    )


def fit_boolean_for_tensor(acc: torch.Tensor, width: int) -> bool:
    w_max = signed_w_max(int(width))
    flat = acc.reshape(-1).to(torch.int64)
    return bool(torch.all(flat >= -w_max) and torch.all(flat <= w_max))


def min_lossless_width_for_tensor(acc: torch.Tensor) -> int | None:
    fitting = [width for width in WIDTH_GRID if fit_boolean_for_tensor(acc, width)]
    return min(fitting) if fitting else None


def _abs_quantiles(flat: torch.Tensor) -> dict[str, float]:
    values = flat.abs().to(torch.float64)
    if values.numel() == 0:
        return {"p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0}
    sorted_vals = torch.sort(values).values
    n = int(sorted_vals.numel())

    def _pct(p: float) -> float:
        idx = max(0, min(n - 1, int(math.ceil(p * n)) - 1))
        return float(sorted_vals[idx].item())

    return {"p50": _pct(0.50), "p90": _pct(0.90), "p95": _pct(0.95), "p99": _pct(0.99)}


def _threshold_counts(flat: torch.Tensor) -> dict[str, int | float]:
    abs_vals = flat.abs().to(torch.int64)
    total = int(abs_vals.numel())
    counts = {
        "count_abs_gte_7": int(torch.sum(abs_vals >= 7).item()),
        "count_abs_gte_8": int(torch.sum(abs_vals >= 8).item()),
        "count_abs_gte_9": int(torch.sum(abs_vals >= 9).item()),
        "count_abs_gte_10": int(torch.sum(abs_vals >= 10).item()),
    }
    max_abs = int(abs_vals.max().item()) if total else 0
    counts["margin_to_threshold"] = float(THRESHOLD_ABS - max_abs)
    counts["fraction_abs_gte_10"] = float(counts["count_abs_gte_10"]) / float(total) if total else 0.0
    counts["max_abs"] = max_abs
    return counts


def _cold_default_value(flat: torch.Tensor) -> int:
    values = flat.reshape(-1).to(torch.int64)
    if values.numel() == 0:
        return 0
    unique_vals, counts = torch.unique(values, return_counts=True)
    max_count = int(counts.max().item())
    mode_mask = counts == max_count
    candidates = unique_vals[mode_mask]
    if int(candidates.numel()) == 1:
        return int(candidates[0].item())
    candidate_list = [int(value.item()) for value in candidates]
    if 0 in candidate_list:
        return 0
    return int(min(candidate_list))


def _sparse_projection(acc: torch.Tensor) -> dict[str, Any]:
    flat = acc.reshape(-1).to(torch.int64)
    lanes = int(flat.numel())
    cold_default = _cold_default_value(flat)
    non_default_mask = flat != int(cold_default)
    non_default_count = int(torch.sum(non_default_mask).item())
    non_default = flat[non_default_mask]
    if non_default.numel() == 0:
        survival: dict[str, int] = {}
    else:
        unique_abs, abs_counts = torch.unique(non_default.abs(), return_counts=True)
        survival = {
            str(int(abs_value.item())): int(count.item())
            for abs_value, count in zip(unique_abs, abs_counts, strict=True)
        }
    projection = project_bounded_delta_accumulator_bpw(
        eligible_weight_count=lanes,
        hot_exact_row_count=non_default_count,
        cold_exception_row_count=0,
        event_delta_count=0,
        backlog_entry_count=0,
        dense_cold_bits_per_weight=0.0,
    )
    lossless = True
    return {
        "cold_default_value": int(cold_default),
        "non_default_count": non_default_count,
        "survival_counts_by_abs_magnitude": survival,
        "bounded_delta_acc_bits_per_weight": float(projection.bounded_delta_acc_bits_per_weight),
        "lossless_or_lossy": "lossless" if lossless else "lossy",
        "storage_projection": projection.to_dict(),
    }


def _dense_frontier_module(acc: torch.Tensor) -> dict[str, Any]:
    flat = acc.reshape(-1).to(torch.int64)
    lanes = int(flat.numel())
    min_v = int(flat.min().item()) if lanes else 0
    max_v = int(flat.max().item()) if lanes else 0
    max_abs = int(flat.abs().max().item()) if lanes else 0
    widths: dict[str, Any] = {}
    for width in WIDTH_GRID:
        widths[str(width)] = {
            "fit_boolean": fit_boolean_for_tensor(acc, width),
            "exact_packed_bpw": exact_packed_bpw(width, lanes),
            "headroom_passes": headroom_passes(
                width,
                max_abs_acc_applied=max_abs,
                headroom_factor=DEFAULT_HEADROOM_FACTOR,
            ),
        }
    return {
        "lanes": lanes,
        "min": min_v,
        "max": max_v,
        "max_abs": max_abs,
        "abs_quantiles": _abs_quantiles(flat),
        "widths": widths,
        "threshold": _threshold_counts(flat),
        "min_lossless_width": min_lossless_width_for_tensor(acc),
    }


def _clip_to_width(acc: torch.Tensor, width: int) -> torch.Tensor:
    w_max = signed_w_max(int(width))
    return acc.clamp(min=-w_max, max=w_max).to(torch.int16)


def _c_deferred_clip_details(acc: torch.Tensor) -> dict[str, Any]:
    """Informational clip audit only; static probe never emits C from these fields."""
    details: dict[str, Any] = {}
    for width in (5, 4, 3, 2):
        clipped = _clip_to_width(acc, width)
        flat = clipped.reshape(-1).to(torch.int64)
        max_abs = int(flat.abs().max().item()) if flat.numel() else 0
        count_gte_10 = int(torch.sum(flat.abs() >= THRESHOLD_ABS).item())
        details[str(width)] = {
            "max_abs_after_clip": max_abs,
            "count_abs_gte_10": count_gte_10,
        }
    return details


def extract_last_sidecar_records(sidecar_path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    last: dict[str, tuple[int, dict[str, Any]]] = {}
    for record in _iter_sidecar_records(sidecar_path):
        state_key = str(record["state_key"])
        step = int(record["step"])
        prior = last.get(state_key)
        if prior is None or step >= prior[0]:
            last[state_key] = (step, record)
    records = {key: value[1] for key, value in last.items()}
    observed = {key: int(value[0]) for key, value in last.items()}
    return records, observed


def cross_check_sidecar_against_receipt(
    *,
    modules: Mapping[str, torch.Tensor],
    receipt_rows: Sequence[Mapping[str, Any]],
    expected_content_sha256: str,
) -> dict[str, Any]:
    row_by_key = {str(row["state_key"]): row for row in receipt_rows}
    state_keys = sorted(modules.keys())
    if set(state_keys) != set(row_by_key.keys()):
        return {
            "cross_check_pass": False,
            "reason": "state_key_mismatch",
            "module_keys": state_keys,
            "receipt_keys": sorted(row_by_key.keys()),
        }
    payloads = [pack_w6_lanes_to_bytes(modules[key]) for key in state_keys]
    built_rows = build_r3_per_module_payload_rows(state_keys, payloads)
    content_sha = canonical_r3_packed_payload_content_sha256(built_rows)
    per_module_ok = True
    mismatches: list[dict[str, Any]] = []
    for built, expected in zip(built_rows, [row_by_key[k] for k in state_keys], strict=True):
        if str(built["payload_sha256"]) != str(expected["payload_sha256"]):
            per_module_ok = False
            mismatches.append(
                {
                    "state_key": built["state_key"],
                    "built": built["payload_sha256"],
                    "expected": expected["payload_sha256"],
                }
            )
    cross_check_pass = (
        per_module_ok and str(content_sha) == str(expected_content_sha256)
    )
    return {
        "cross_check_pass": cross_check_pass,
        "built_content_sha256": content_sha,
        "expected_content_sha256": str(expected_content_sha256),
        "per_module_payload_sha256_match": per_module_ok,
        "mismatches": mismatches,
    }


def select_branch(
    *,
    harness_fail: bool,
    cross_check_pass: bool,
    aggregate_acc: torch.Tensor,
    aggregate_sparse: Mapping[str, Any],
) -> dict[str, Any]:
    if harness_fail:
        return {"branch": BRANCH_HARNESS_FAIL}
    if not cross_check_pass:
        return {"branch": BRANCH_READ_PATH_FAIL}

    min_lossless = min_lossless_width_for_tensor(aggregate_acc)
    if min_lossless is None:
        return {"branch": BRANCH_D_REPRESENTATION_LIMIT, "min_lossless_width": None}

    best_dense_lossless_bpw = exact_packed_bpw(min_lossless, int(aggregate_acc.numel()))
    sparse_lossless = str(aggregate_sparse.get("lossless_or_lossy")) == "lossless"
    sparse_bpw = float(aggregate_sparse.get("bounded_delta_acc_bits_per_weight", math.inf))

    if sparse_lossless and sparse_bpw <= best_dense_lossless_bpw - SPARSE_BEATS_DENSE_MARGIN_BPW:
        return {
            "branch": BRANCH_B_SPARSE_LOSSLESS_WINS,
            "min_lossless_width": int(min_lossless),
            "best_dense_lossless_bpw": best_dense_lossless_bpw,
            "sparse_bpw": sparse_bpw,
            "sparse_beats_dense_margin_bpw": SPARSE_BEATS_DENSE_MARGIN_BPW,
        }

    threshold = _threshold_counts(aggregate_acc.reshape(-1).to(torch.int64))
    a1_threshold_safe = (
        threshold["count_abs_gte_10"] == 0 and float(threshold["margin_to_threshold"]) >= 0.0
    )

    if int(min_lossless) in {4, 3, 2}:
        return {
            "branch": BRANCH_A1_DENSE_LOSSLESS.format(n=int(min_lossless)),
            "min_lossless_width": int(min_lossless),
            "a1_threshold_safe": bool(a1_threshold_safe),
            "a1_next_gate_parity_type": next_gate_parity_type(int(min_lossless)),
            "best_dense_lossless_bpw": best_dense_lossless_bpw,
            "sparse_bpw": sparse_bpw,
        }

    if int(min_lossless) == 5:
        return {
            "branch": BRANCH_A0_DENSE_W5_ONLY,
            "min_lossless_width": 5,
            "a0_next_gate_parity_type": next_gate_parity_type(5),
            "best_dense_lossless_bpw": best_dense_lossless_bpw,
            "sparse_bpw": sparse_bpw,
        }

    clip_details = _c_deferred_clip_details(aggregate_acc)
    return {
        "branch": BRANCH_D_REPRESENTATION_LIMIT,
        "min_lossless_width": int(min_lossless),
        "best_dense_lossless_bpw": best_dense_lossless_bpw,
        "sparse_bpw": sparse_bpw,
        "c_deferred_clip_details": clip_details,
        "c_deferred_note": (
            "C deferred — decision-surface plausibility is NOT measurable from a "
            "static acc snapshot; requires dynamics/trajectory (q_lanes), next rung."
        ),
    }


def build_measurement_from_modules(
    *,
    modules: Mapping[str, torch.Tensor],
    logical_shapes: Mapping[str, Sequence[int]],
    cross_check: Mapping[str, Any],
    harness_fail: bool = False,
    run_root: str | None = None,
    head_sha256: str | None = None,
    input_artifact_hashes: Mapping[str, str] | None = None,
    r4_baseline: Mapping[str, Any] | None = None,
    observed_max_step_per_module: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    per_module: dict[str, Any] = {}
    all_pieces: list[torch.Tensor] = []
    sparse_rows: list[dict[str, Any]] = []
    for state_key in sorted(modules.keys()):
        acc = modules[state_key].to(torch.int16)
        per_module[state_key] = {
            "logical_shape": [int(dim) for dim in logical_shapes[state_key]],
            "dense_frontier": _dense_frontier_module(acc),
            "sparse_projection": _sparse_projection(acc),
        }
        all_pieces.append(acc.reshape(-1))
        sparse_rows.append(per_module[state_key]["sparse_projection"])

    aggregate_acc = torch.cat(all_pieces) if all_pieces else torch.zeros(0, dtype=torch.int16)
    aggregate_dense = _dense_frontier_module(aggregate_acc)
    aggregate_sparse = _sparse_projection(aggregate_acc)
    branch = select_branch(
        harness_fail=harness_fail,
        cross_check_pass=bool(cross_check.get("cross_check_pass")),
        aggregate_acc=aggregate_acc,
        aggregate_sparse=aggregate_sparse,
    )
    observed_steps = dict(observed_max_step_per_module or {})
    receipt: dict[str, Any] = {
        "schema_version": R5_PROBE_SCHEMA_VERSION,
        "raw_arrays_included": False,
        "run_root": run_root,
        "head_sha256": head_sha256,
        "input_artifact_hashes": dict(input_artifact_hashes or {}),
        "r4_baseline": dict(r4_baseline or {}),
        "cross_check": dict(cross_check),
        "observed_max_step_per_module": observed_steps,
        "observed_max_step_aggregate": max(observed_steps.values()) if observed_steps else None,
        "per_module": per_module,
        "aggregate": {
            "dense_frontier": aggregate_dense,
            "sparse_projection": aggregate_sparse,
            "threshold_abs": THRESHOLD_ABS,
            "sparse_beats_dense_margin_bpw": SPARSE_BEATS_DENSE_MARGIN_BPW,
        },
        "branch_selection": branch,
        "explicit_non_claims": list(EXPLICIT_NON_CLAIMS),
    }
    return receipt


def modules_from_sidecar_records(
    records: Mapping[str, Mapping[str, Any]],
    logical_shapes: Mapping[str, Sequence[int]],
) -> dict[str, torch.Tensor]:
    modules: dict[str, torch.Tensor] = {}
    for state_key, record in records.items():
        lanes = torch.tensor(record["accumulator_lanes"], dtype=torch.int16)
        shape = tuple(int(dim) for dim in logical_shapes[state_key])
        modules[state_key] = lanes.view(shape).contiguous()
    return modules


def build_measurement_probe_receipt(
    *,
    run_root: Path,
    arm_dir: str = "w6_on_q_on_treatment",
    head_sha256: str,
    expected_receipt_sha256: str | None = None,
    expected_sidecar_sha256: str | None = None,
) -> dict[str, Any]:
    arm_path = run_root / arm_dir
    receipt_path = arm_path / "receipt.json"
    sidecar_path = arm_path / "headroom_wiring_sidecar.jsonl"
    if not receipt_path.is_file():
        raise FileNotFoundError(f"missing receipt: {receipt_path}")
    if not sidecar_path.is_file():
        raise FileNotFoundError(f"missing sidecar: {sidecar_path}")

    pre_receipt_sha = file_sha256(receipt_path)
    pre_sidecar_sha = file_sha256(sidecar_path)
    harness_fail = False
    harness_failures: list[str] = []
    if expected_receipt_sha256 and pre_receipt_sha != expected_receipt_sha256:
        harness_fail = True
        harness_failures.append("receipt_sha256_mismatch")
    if expected_sidecar_sha256 and pre_sidecar_sha != expected_sidecar_sha256:
        harness_fail = True
        harness_failures.append("sidecar_sha256_mismatch")

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    ledger = receipt.get("r4_persistent_ledger") or {}
    receipt_rows = ledger.get("r4_per_module_acc_rows") or []
    expected_content_sha = str(ledger.get("r4_acc_packed_content_sha256", ""))
    logical_shapes = {
        str(row["state_key"]): row["logical_shape"] for row in receipt_rows
    }

    sidecar_records, observed_steps = extract_last_sidecar_records(sidecar_path)
    modules = modules_from_sidecar_records(sidecar_records, logical_shapes)
    cross_check = cross_check_sidecar_against_receipt(
        modules=modules,
        receipt_rows=receipt_rows,
        expected_content_sha256=expected_content_sha,
    )

    post_receipt_sha = file_sha256(receipt_path)
    post_sidecar_sha = file_sha256(sidecar_path)
    if post_receipt_sha != pre_receipt_sha or post_sidecar_sha != pre_sidecar_sha:
        harness_fail = True
        harness_failures.append("input_artifact_mutated_during_read")

    r4_baseline = {
        "r4_q_physical_bits_per_weight": ledger.get("r4_q_physical_bits_per_weight"),
        "r4_acc_physical_bits_per_weight": ledger.get("r4_acc_physical_bits_per_weight"),
        "r4_checkpoint_inclusive_physical_bits_per_weight": ledger.get(
            "r4_checkpoint_inclusive_physical_bits_per_weight"
        ),
        "r4_acc_packed_content_sha256": expected_content_sha,
    }

    result = build_measurement_from_modules(
        modules=modules,
        logical_shapes=logical_shapes,
        cross_check=cross_check,
        harness_fail=harness_fail,
        run_root=str(run_root),
        head_sha256=head_sha256,
        input_artifact_hashes={
            "receipt_sha256_pre": pre_receipt_sha,
            "receipt_sha256_post": post_receipt_sha,
            "sidecar_sha256_pre": pre_sidecar_sha,
            "sidecar_sha256_post": post_sidecar_sha,
        },
        r4_baseline=r4_baseline,
        observed_max_step_per_module=observed_steps,
    )
    if harness_failures:
        result["harness_failures"] = harness_failures
    return result
