"""Pin carry_w6_second_capture_confirmed verdict surfaces (B8, measurement-only).

Loads recorded evidence read-only from the M2b dual-capture chain roots. Pins what
was measured — does not remeasure captures or assert stability/training/sub-2 claims.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from calm.hrm_text_158.native_full_stack.accumulator_policy_shadow_screen import (
    _file_sha256,
    _load_b2b_sequential_trace_steps,
    _stable_hash16,
)

CARRY_W6_SECOND_CAPTURE_REGRESSION_SCHEMA = (
    "hrm_text_158_carry_w6_second_capture_regression/v0"
)

VERDICT_CLASS = "carry_w6_second_capture_confirmed"
JOINT_VERDICT_MSG_ID = "1781169197825"

B2C_PRIMARY_LABEL_ACCUMULATOR_NO_TRACKING_NULL = "accumulator_no_tracking_null"
AUDIT_WINNING_FAMILY_CARRIED_PERSISTENT_BUCKET = "carried_persistent_bucket"
F5_TAXONOMY_ACC_SHRINK_TWO_TIER = "acc_shrink_two_tier"

CREDITDIR_ROOT = Path("/home/gabe/claw-code-creditdir/transient_fp_credit")

FORBIDDEN_SCIENCE_CLAIM_SURFACES = frozenset(
    {
        "sub_2_bit_physical_persistent",
        "sub_2_runtime_ready",
        "training_stable",
        "stability_claim",
        "main_science_launch_authorized",
    }
)


@dataclass(frozen=True)
class ChainCaptureSpec:
    capture_id: str
    chain_root: Path
    trace_relpath: str
    trace_hash: str
    seed: int
    parent_sha256: str | None
    b2c_relpath: str
    audit_relpath: str
    acc_width_relpath: str
    w6_value_drift_mismatch: int
    below_w6_crossing_mismatch_by_width: Mapping[int, int]
    w_min: int = 6
    provenance_extraction_date: str = "2026-06-10"


TRACE1_CAPTURE = ChainCaptureSpec(
    capture_id="trace1",
    chain_root=CREDITDIR_ROOT / "b2b_recapture_20260610T145044Z",
    trace_relpath="b2b_seed43/b2b_sequential_trace.ndjson",
    trace_hash="cb373de78030c5a9",
    seed=43,
    parent_sha256=None,
    b2c_relpath="b2c_replay/b2c_final_temporal_verdict_receipt.json",
    audit_relpath="audit_v0/transient_selection_information_audit_v0_receipt.json",
    acc_width_relpath=(
        "baseline_b0/acc_width_sweep/acc_width_recorded_row_sweep_v0_receipt.json"
    ),
    w6_value_drift_mismatch=838,
    below_w6_crossing_mismatch_by_width={4: 1314, 3: 1314, 2: 1314},
)

CAPTURE2_CAPTURE = ChainCaptureSpec(
    capture_id="capture2",
    chain_root=CREDITDIR_ROOT / "b2b_recapture_20260610T204129Z",
    trace_relpath="b2b_seed44/b2b_sequential_trace.ndjson",
    trace_hash="34310c423c2ed05c",
    seed=44,
    parent_sha256=(
        "9b4e311a22787e7d4808bde7bc2953568d767a2ee8ac648942a3f5dbb7b4d5ec"
    ),
    b2c_relpath="b2c_replay/b2c_final_temporal_verdict_receipt.json",
    audit_relpath="audit_v0/transient_selection_information_audit_v0_receipt.json",
    acc_width_relpath=(
        "acc_width_sweep_capture2/acc_width_recorded_row_sweep_v0_receipt.json"
    ),
    w6_value_drift_mismatch=786,
    below_w6_crossing_mismatch_by_width={4: 1281, 3: 1281, 2: 1281},
)

DUAL_CAPTURE_SPECS: tuple[ChainCaptureSpec, ...] = (TRACE1_CAPTURE, CAPTURE2_CAPTURE)

VALUE_DRIFT_NON_GATING_MISMATCHES: tuple[tuple[str, int], ...] = (
    ("trace1", TRACE1_CAPTURE.w6_value_drift_mismatch),
    ("capture2", CAPTURE2_CAPTURE.w6_value_drift_mismatch),
)


def chain_paths_available(spec: ChainCaptureSpec) -> bool:
    root = spec.chain_root
    required = (
        root / spec.trace_relpath,
        root / spec.b2c_relpath,
        root / spec.audit_relpath,
        root / spec.acc_width_relpath,
    )
    return all(path.is_file() for path in required)


def _canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def load_json_receipt_readonly(path: Path) -> tuple[dict[str, Any], str]:
    """Load a JSON receipt read-only; return payload and sha256 (stable across read)."""
    sha_before = _file_sha256(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"receipt must be a JSON object: path={path}")
    sha_after = _file_sha256(path)
    if sha_before != sha_after:
        raise ValueError(
            f"receipt mutated during read-only load: path={path}, "
            f"sha_before={sha_before}, sha_after={sha_after}"
        )
    return payload, sha_before


def compute_trace_hash_from_path(trace_path: Path) -> str:
    steps, load_failures = _load_b2b_sequential_trace_steps(trace_path)
    if load_failures:
        raise ValueError(
            f"trace load failures for {trace_path}: {load_failures}"
        )
    step_hashes = [str(step["source_table_hash"]) for step in steps]
    return _stable_hash16(step_hashes)


def reject_science_overclaim_surface(*, claim_surface: str) -> None:
    normalized = str(claim_surface).strip()
    if normalized in FORBIDDEN_SCIENCE_CLAIM_SURFACES:
        raise ValueError(
            "science overclaim surface rejected (B8 pins verdict class only): "
            f"claim_surface={claim_surface!r}"
        )


def assert_verdict_class_only(*, verdict_class: str) -> None:
    if str(verdict_class) != VERDICT_CLASS:
        raise ValueError(
            "B8 regression asserts carry_w6_second_capture_confirmed only: "
            f"verdict_class={verdict_class!r}"
        )
    reject_science_overclaim_surface(claim_surface=verdict_class)


def _width_result_by_width(
    acc_width_receipt: Mapping[str, Any],
    *,
    width: int,
) -> dict[str, Any]:
    width_results = acc_width_receipt.get("width_results")
    if not isinstance(width_results, list):
        raise ValueError("acc_width receipt missing width_results list")
    for entry in width_results:
        if isinstance(entry, Mapping) and int(entry.get("width", -1)) == int(width):
            return dict(entry)
    raise ValueError(f"width_results missing width={width}")


def validate_determinism_trace_hash(
    *,
    spec: ChainCaptureSpec,
    trace_path: Path | None = None,
) -> dict[str, Any]:
    path = trace_path or (spec.chain_root / spec.trace_relpath)
    observed = compute_trace_hash_from_path(path)
    if observed != spec.trace_hash:
        raise ValueError(
            "trace_hash mismatch vs recorded pin: "
            f"capture={spec.capture_id!r}, observed={observed!r}, "
            f"expected={spec.trace_hash!r}"
        )
    return {
        "surface": "determinism_trace_hash",
        "capture_id": spec.capture_id,
        "trace_path": str(path),
        "trace_hash": observed,
    }


def validate_b2c_accumulator_no_tracking_null(
    *,
    spec: ChainCaptureSpec,
    b2c_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    primary = str(b2c_receipt.get("primary_label", ""))
    if primary != B2C_PRIMARY_LABEL_ACCUMULATOR_NO_TRACKING_NULL:
        raise ValueError(
            "b2c primary_label surface mismatch: "
            f"capture={spec.capture_id!r}, primary_label={primary!r}"
        )
    arms = b2c_receipt.get("arms")
    if not isinstance(arms, Mapping):
        raise ValueError("b2c receipt missing arms mapping")
    accumulator_only = arms.get("accumulator_only")
    if not isinstance(accumulator_only, Mapping):
        raise ValueError("b2c receipt missing arms.accumulator_only")
    metrics = accumulator_only.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("b2c receipt missing accumulator_only.metrics")
    jaccard = float(metrics.get("jaccard_vs_int16", -1.0))
    if jaccard != 0.0:
        raise ValueError(
            "b2c accumulator_only jaccard_vs_int16 must be 0.0: "
            f"capture={spec.capture_id!r}, jaccard={jaccard}"
        )
    return {
        "surface": "b2c_accumulator_no_tracking_null",
        "capture_id": spec.capture_id,
        "primary_label": primary,
        "jaccard_vs_int16": jaccard,
    }


def validate_transient_compute_control_winning_family(
    *,
    spec: ChainCaptureSpec,
    audit_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    winning = audit_receipt.get("winning_family")
    if not isinstance(winning, Mapping):
        raise ValueError("audit receipt missing winning_family")
    family_id = str(winning.get("family_id", ""))
    if family_id != AUDIT_WINNING_FAMILY_CARRIED_PERSISTENT_BUCKET:
        raise ValueError(
            "audit winning_family surface mismatch: "
            f"capture={spec.capture_id!r}, family_id={family_id!r}"
        )
    return {
        "surface": "transient_compute_control_winning_family",
        "capture_id": spec.capture_id,
        "family_id": family_id,
    }


def validate_f5_acc_shrink_two_tier_w_min(
    *,
    spec: ChainCaptureSpec,
    acc_width_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    w_min = int(acc_width_receipt.get("w_min", -1))
    w_min_headroom_safe = int(acc_width_receipt.get("w_min_headroom_safe", -1))
    w_min_invariant = int(acc_width_receipt.get("w_min_invariant", -1))
    if w_min != spec.w_min or w_min_headroom_safe != spec.w_min or w_min_invariant != spec.w_min:
        raise ValueError(
            "F5 w_min surfaces must equal 6: "
            f"capture={spec.capture_id!r}, w_min={w_min}, "
            f"w_min_headroom_safe={w_min_headroom_safe}, "
            f"w_min_invariant={w_min_invariant}"
        )
    taxonomy = acc_width_receipt.get("taxonomy_labels")
    if not isinstance(taxonomy, list) or F5_TAXONOMY_ACC_SHRINK_TWO_TIER not in taxonomy:
        raise ValueError(
            "F5 taxonomy must include acc_shrink_two_tier: "
            f"capture={spec.capture_id!r}, taxonomy={taxonomy!r}"
        )
    return {
        "surface": "f5_acc_shrink_two_tier_w_min",
        "capture_id": spec.capture_id,
        "w_min": w_min,
        "taxonomy_labels": list(taxonomy),
    }


def validate_w6_floor_and_hard_break_below_w6(
    *,
    spec: ChainCaptureSpec,
    acc_width_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    w6 = _width_result_by_width(acc_width_receipt, width=6)
    crossing_w6 = int(w6.get("crossing_mismatch_count_vs_w16", -1))
    if crossing_w6 != 0:
        raise ValueError(
            "W6 crossing_mismatch_count_vs_w16 must be 0: "
            f"capture={spec.capture_id!r}, crossing={crossing_w6}"
        )
    mismatch_w6 = int(w6.get("mismatch_count_vs_w16_reference", -1))
    if mismatch_w6 != spec.w6_value_drift_mismatch:
        raise ValueError(
            "W6 value-drift mismatch pin mismatch: "
            f"capture={spec.capture_id!r}, observed={mismatch_w6}, "
            f"expected={spec.w6_value_drift_mismatch}"
        )
    below_break: dict[str, int] = {}
    for width, expected_crossing in spec.below_w6_crossing_mismatch_by_width.items():
        row = _width_result_by_width(acc_width_receipt, width=int(width))
        observed_crossing = int(row.get("crossing_mismatch_count_vs_w16", -1))
        if observed_crossing != int(expected_crossing):
            raise ValueError(
                "hard break below W6 crossing pin mismatch: "
                f"capture={spec.capture_id!r}, width={width}, "
                f"observed={observed_crossing}, expected={expected_crossing}"
            )
        below_break[str(width)] = observed_crossing
    return {
        "surface": "w6_floor_and_hard_break_below_w6",
        "capture_id": spec.capture_id,
        "w6_crossing_mismatch_count_vs_w16": crossing_w6,
        "w6_mismatch_count_vs_w16_reference": mismatch_w6,
        "below_w6_crossing_mismatch_by_width": below_break,
    }


def validate_value_drift_non_gating_across_captures(
    *,
    observed_by_capture: Mapping[str, int],
) -> dict[str, Any]:
    trace1 = int(observed_by_capture["trace1"])
    capture2 = int(observed_by_capture["capture2"])
    if trace1 == capture2:
        raise ValueError(
            "value-drift non-gating requires distinct W6 mismatch counts: "
            f"trace1={trace1}, capture2={capture2}"
        )
    if trace1 != TRACE1_CAPTURE.w6_value_drift_mismatch:
        raise ValueError(f"trace1 W6 mismatch pin: expected 838, got {trace1}")
    if capture2 != CAPTURE2_CAPTURE.w6_value_drift_mismatch:
        raise ValueError(f"capture2 W6 mismatch pin: expected 786, got {capture2}")
    return {
        "surface": "value_drift_786_vs_838_non_gating",
        "trace1_w6_mismatch": trace1,
        "capture2_w6_mismatch": capture2,
        "gating_surface": "crossing_mismatch_count_vs_w16_at_w6",
        "non_gating_surface": "mismatch_count_vs_w16_reference_at_w6",
    }


def evaluate_capture_surfaces(
    spec: ChainCaptureSpec,
) -> dict[str, Any]:
    if not chain_paths_available(spec):
        raise FileNotFoundError(
            f"chain capture paths unavailable for {spec.capture_id}: {spec.chain_root}"
        )
    trace_path = spec.chain_root / spec.trace_relpath
    b2c_path = spec.chain_root / spec.b2c_relpath
    audit_path = spec.chain_root / spec.audit_relpath
    acc_width_path = spec.chain_root / spec.acc_width_relpath

    b2c_receipt, b2c_sha = load_json_receipt_readonly(b2c_path)
    audit_receipt, audit_sha = load_json_receipt_readonly(audit_path)
    acc_width_receipt, acc_width_sha = load_json_receipt_readonly(acc_width_path)

    surfaces = {
        "determinism": validate_determinism_trace_hash(spec=spec, trace_path=trace_path),
        "b2c": validate_b2c_accumulator_no_tracking_null(
            spec=spec,
            b2c_receipt=b2c_receipt,
        ),
        "audit": validate_transient_compute_control_winning_family(
            spec=spec,
            audit_receipt=audit_receipt,
        ),
        "f5_w_min": validate_f5_acc_shrink_two_tier_w_min(
            spec=spec,
            acc_width_receipt=acc_width_receipt,
        ),
        "w6_floor_break": validate_w6_floor_and_hard_break_below_w6(
            spec=spec,
            acc_width_receipt=acc_width_receipt,
        ),
    }
    return {
        "capture_id": spec.capture_id,
        "chain_root": str(spec.chain_root),
        "seed": spec.seed,
        "parent_sha256": spec.parent_sha256,
        "receipt_shas": {
            "b2c": b2c_sha,
            "audit": audit_sha,
            "acc_width": acc_width_sha,
        },
        "surfaces": surfaces,
        "fixture_provenance": {
            "chain_root": str(spec.chain_root),
            "trace_hash": spec.trace_hash,
            "extraction_date": spec.provenance_extraction_date,
            "joint_verdict_msg_id": JOINT_VERDICT_MSG_ID,
            "load_mode": "read_only_disk_with_rehash",
        },
    }


def evaluate_carry_w6_second_capture_confirmed() -> dict[str, Any]:
    """Evaluate all pinned surfaces for both captures; assert verdict class only."""
    assert_verdict_class_only(verdict_class=VERDICT_CLASS)
    capture_blocks: list[dict[str, Any]] = []
    w6_mismatch_by_capture: dict[str, int] = {}
    for spec in DUAL_CAPTURE_SPECS:
        block = evaluate_capture_surfaces(spec)
        capture_blocks.append(block)
        w6_mismatch_by_capture[spec.capture_id] = int(
            block["surfaces"]["w6_floor_break"]["w6_mismatch_count_vs_w16_reference"]
        )
    value_drift = validate_value_drift_non_gating_across_captures(
        observed_by_capture=w6_mismatch_by_capture,
    )
    block = {
        "schema": CARRY_W6_SECOND_CAPTURE_REGRESSION_SCHEMA,
        "verdict_class": VERDICT_CLASS,
        "joint_verdict_msg_id": JOINT_VERDICT_MSG_ID,
        "claim_boundary": {
            "measurement_only": True,
            "no_stability_claim": True,
            "no_training_claim": True,
            "no_sub_2_claim": True,
            "pins_recorded_evidence_only": True,
        },
        "captures": capture_blocks,
        "cross_capture": value_drift,
    }
    return block


def sha256_canonical_regression_block(block: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(dict(block)).encode("utf-8")).hexdigest()
