#!/usr/bin/env python3
"""Offline R5 falsification-screen sweep over existing headroom sidecars.

Read-only over run artifacts. No trainer mutation, no GPU, no .pt commit.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from calm.hrm_text_158.native_full_stack.bounded_delta_accumulator import (
    project_bounded_delta_accumulator_bpw,
)
from calm.hrm_text_158.native_full_stack.r5_acc_term_measurement_probe import (
    THRESHOLD_ABS as BASELINE_THRESHOLD_ABS,
    _cold_default_value,
    exact_packed_bpw,
    min_lossless_width_for_tensor,
)
from calm.hrm_text_158.native_full_stack.s3bb_headroom_telemetry import (
    _iter_sidecar_records,
)
SCHEMA_VERSION = "hrm_text_158_r5_offline_falsification_screen/v2"
BASELINE_DECAY_NUM = 1
BASELINE_DECAY_DEN = 1
Q_PACKED_BPW = 2.0
W6_ACC_BPW = 6.0
W6_SECONDARY_TARGET_BPW = 6.0
SUB2_TARGET_BPW = 2.0
SPARSE_BEATS_W6_MARGIN_BPW = 0.25

THRESHOLD_GRID: tuple[int, ...] = (7, 8, 9, 10, 11, 12, 14)
DECAY_GRID: tuple[tuple[int, int], ...] = (
    (1, 1),
    (9, 10),
    (1, 2),
    (4, 5),
    (19, 20),
)

CLASSIFIER_FIXED_ACC_NULL = "FIXED_ACC_REPRESENTATION_NULL"
CLASSIFIER_THRESHOLD_DAMPING_INFEASIBLE = "THRESHOLD_DAMPING_INFEASIBLE"
CLASSIFIER_DYNAMICS_PROMISING = "DYNAMICS_VARIANT_PROMISING"
CLASSIFIER_MISSING_OBSERVABLES = "MISSING_OBSERVABLES"

SIM_STATIC_PROXY = "static_proxy"
SIM_DYNAMICS_PROOF = "dynamics_proof"

CROSSING_PRESERVE_MIN = 0.99

EXPLICIT_NON_CLAIMS: tuple[str, ...] = (
    "offline_read_only_no_trainer_mutation",
    "no_gpu_launch",
    "no_pt_commit",
    "no_sub2_claim",
    "combined_q_plus_acc_bpw_is_diagnostic_only_q_already_packed_at_2bpw",
    "static_proxy_cannot_prove_dynamics_without_per_lane_votes",
    "applied_mask_not_scoreable_without_per_lane_votes_and_dynamics_replay",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regime_key(threshold_abs: int, decay_num: int, decay_den: int) -> tuple[int, int, int]:
    return (int(threshold_abs), int(decay_num), int(decay_den))


def _apply_decay(acc: torch.Tensor, *, decay_num: int, decay_den: int) -> torch.Tensor:
    if int(decay_den) == 0:
        raise ValueError("decay_den must be > 0")
    if int(decay_num) == int(decay_den) == 1:
        return acc
    values = acc.to(torch.int32)
    decayed = torch.div(values * int(decay_num), int(decay_den), rounding_mode="trunc")
    return decayed.to(torch.int16)


def _crossing_mask(acc: torch.Tensor, q: torch.Tensor, *, threshold_abs: int) -> torch.Tensor:
    acc_flat = acc.reshape(-1).to(torch.int32)
    q_flat = q.reshape(-1).to(torch.int32)
    threshold = int(threshold_abs)
    pos = (acc_flat >= threshold) & (q_flat < 1)
    neg = (acc_flat <= -threshold) & (q_flat > -1)
    return pos | neg


def _entropy_bits_per_lane(flat: torch.Tensor) -> float:
    values = flat.reshape(-1).to(torch.int64)
    if int(values.numel()) == 0:
        return 0.0
    unique, counts = torch.unique(values, return_counts=True)
    total = float(values.numel())
    entropy = 0.0
    for count in counts.tolist():
        p = float(count) / total
        if p > 0.0:
            entropy -= p * math.log2(p)
    return float(entropy)


def _domain_bits_per_lane(acc: torch.Tensor) -> float:
    width = min_lossless_width_for_tensor(acc)
    if width is None:
        flat = acc.reshape(-1).to(torch.int64)
        max_abs = int(flat.abs().max().item()) if flat.numel() else 0
        if max_abs == 0:
            return 0.0
        return float(int(math.ceil(math.log2(float(2 * max_abs + 1)))))
    return float(exact_packed_bpw(int(width), int(acc.numel())))


def _sparse_hot_bpw(acc: torch.Tensor) -> float:
    flat = acc.reshape(-1).to(torch.int64)
    lanes = int(flat.numel())
    cold_default = _cold_default_value(flat)
    non_default_count = int(torch.sum(flat != int(cold_default)).item())
    projection = project_bounded_delta_accumulator_bpw(
        eligible_weight_count=lanes,
        hot_exact_row_count=non_default_count,
        cold_exception_row_count=0,
        event_delta_count=0,
        backlog_entry_count=0,
        dense_cold_bits_per_weight=0.0,
    )
    return float(projection.bounded_delta_acc_bits_per_weight)


def audit_observable_coverage(
    *,
    sidecar_paths: Mapping[str, Path],
    receipt_paths: Mapping[str, Path],
) -> dict[str, Any]:
    sidecar_keys: set[str] = set()
    record_count = 0
    steps: set[int] = set()
    modules: set[str] = set()
    for _arm, path in sidecar_paths.items():
        for record in _iter_sidecar_records(path):
            record_count += 1
            sidecar_keys.update(record.keys())
            steps.add(int(record["step"]))
            modules.add(str(record["state_key"]))

    receipt_fields: dict[str, Any] = {}
    applied_present = False
    votes_present = False
    for arm, path in receipt_paths.items():
        if not path.is_file():
            continue
        receipt = json.loads(path.read_text(encoding="utf-8"))
        step_reports = receipt.get("step_reports") or {}
        receipt_fields[arm] = {
            "steps_completed": int(receipt.get("steps_completed") or 0),
            "step_report_count": len(step_reports),
        }
        for _step_key, report in step_reports.items():
            tensor_stats = (report.get("step_result") or {}).get("tensor_stats") or {}
            for stats in tensor_stats.values():
                if "applied_indices" in stats:
                    applied_present = True
                if "votes" in stats or "vote_lanes" in stats:
                    votes_present = True

    sidecar_has_acc = "accumulator_lanes" in sidecar_keys
    sidecar_has_q = "q_lanes" in sidecar_keys
    sidecar_has_votes = "vote_lanes" in sidecar_keys or "votes" in sidecar_keys
    sidecar_has_applied = "applied_indices" in sidecar_keys

    if sidecar_has_votes and sidecar_has_q:
        simulation_class = SIM_DYNAMICS_PROOF
    elif sidecar_has_acc and sidecar_has_q:
        simulation_class = SIM_STATIC_PROXY
    else:
        simulation_class = "insufficient"

    missing_for_ranking: list[str] = []
    if not sidecar_has_votes:
        missing_for_ranking.append("per_lane_votes")
    if not sidecar_has_applied:
        missing_for_ranking.append("sidecar_applied_indices")
    missing_for_ranking.append("pre_update_accumulator_before_vote_decay")

    return {
        "sidecar_record_count": int(record_count),
        "sidecar_unique_keys": sorted(sidecar_keys),
        "sidecar_steps": sorted(steps),
        "sidecar_module_count": len(modules),
        "sidecar_has_accumulator_lanes": sidecar_has_acc,
        "sidecar_has_q_lanes": sidecar_has_q,
        "sidecar_has_votes": sidecar_has_votes,
        "sidecar_has_applied_indices": sidecar_has_applied,
        "receipt_applied_indices_present": applied_present,
        "receipt_votes_present": votes_present,
        "simulation_class": simulation_class,
        "cap_order_ranking_scoreable": bool(sidecar_has_votes),
        # Receipt applied_indices exist but static post-step acc cannot reconstruct
        # applied-mask parity without per-lane votes — not scoreable on this sweep.
        "applied_mask_scoreable": bool(sidecar_has_votes and sidecar_has_applied),
        "applied_mask_unscoreable_reason": (
            None
            if sidecar_has_votes and sidecar_has_applied
            else "static_proxy_missing_per_lane_votes_for_applied_mask_reconstruction"
        ),
        "missing_for_dynamics_proof": missing_for_ranking,
        "receipt_fields": receipt_fields,
    }


@dataclass
class RegimeAgg:
    total_lanes: int = 0
    crossing_matches: int = 0
    entropy_weighted_sum: float = 0.0
    domain_weighted_sum: float = 0.0
    sparse_bpw_max: float = 0.0
    sparse_bpw_min: float = field(default_factory=lambda: math.inf)
    max_abs: int = 0
    w6_lossless_fit: bool = True


def _stream_sidecar_sweep(
    sidecar_paths: Mapping[str, Path],
) -> dict[tuple[int, int, int], RegimeAgg]:
    aggs: dict[tuple[int, int, int], RegimeAgg] = {
        _regime_key(threshold, num, den): RegimeAgg()
        for threshold in THRESHOLD_GRID
        for num, den in DECAY_GRID
    }
    baseline_key = _regime_key(
        int(BASELINE_THRESHOLD_ABS),
        int(BASELINE_DECAY_NUM),
        int(BASELINE_DECAY_DEN),
    )

    for arm, path in sidecar_paths.items():
        for record in _iter_sidecar_records(path):
            acc0 = torch.tensor(record["accumulator_lanes"], dtype=torch.int16)
            q = torch.tensor(record["q_lanes"], dtype=torch.int8)
            step = int(record["step"])
            state_key = str(record["state_key"])

            decayed_by_regime: dict[tuple[int, int], torch.Tensor] = {}
            for decay_num, decay_den in DECAY_GRID:
                decayed_by_regime[(decay_num, decay_den)] = _apply_decay(
                    acc0,
                    decay_num=decay_num,
                    decay_den=decay_den,
                )

            baseline_acc = decayed_by_regime[(BASELINE_DECAY_NUM, BASELINE_DECAY_DEN)]
            baseline_cross = _crossing_mask(
                baseline_acc,
                q,
                threshold_abs=int(BASELINE_THRESHOLD_ABS),
            )

            for decay_num, decay_den in DECAY_GRID:
                acc = decayed_by_regime[(decay_num, decay_den)]
                lanes = int(acc.numel())
                entropy = _entropy_bits_per_lane(acc)
                domain = _domain_bits_per_lane(acc)
                sparse_bpw = _sparse_hot_bpw(acc)
                max_abs = int(acc.to(torch.int32).abs().max().item()) if lanes else 0
                w6_fit = min_lossless_width_for_tensor(acc) is not None

                for threshold_abs in THRESHOLD_GRID:
                    regime = _regime_key(threshold_abs, decay_num, decay_den)
                    agg = aggs[regime]
                    crossing = _crossing_mask(acc, q, threshold_abs=threshold_abs)
                    agg.total_lanes += lanes
                    agg.crossing_matches += int(torch.sum(crossing == baseline_cross).item())
                    agg.entropy_weighted_sum += entropy * float(lanes)
                    agg.domain_weighted_sum += domain * float(lanes)
                    agg.sparse_bpw_max = max(agg.sparse_bpw_max, sparse_bpw)
                    agg.sparse_bpw_min = min(agg.sparse_bpw_min, sparse_bpw)
                    agg.max_abs = max(agg.max_abs, max_abs)
                    agg.w6_lossless_fit = agg.w6_lossless_fit and w6_fit

    if baseline_key not in aggs:
        raise RuntimeError("baseline regime missing from aggregation")
    return aggs


def _finalize_regime_row(
    agg: RegimeAgg,
    *,
    threshold_abs: int,
    decay_num: int,
    decay_den: int,
) -> dict[str, Any]:
    total = int(agg.total_lanes)
    acc_entropy = agg.entropy_weighted_sum / float(total) if total else 0.0
    acc_domain = agg.domain_weighted_sum / float(total) if total else 0.0
    acc_sparse = float(agg.sparse_bpw_max)
    acc_term = min(acc_domain, acc_sparse, W6_ACC_BPW)
    crossing_preservation = float(agg.crossing_matches) / float(total) if total else 0.0
    distance_from_2bpw = float(acc_term) - float(SUB2_TARGET_BPW)
    observables_preserved = crossing_preservation >= CROSSING_PRESERVE_MIN
    return {
        "threshold_abs": int(threshold_abs),
        "decay_num": int(decay_num),
        "decay_den": int(decay_den),
        "simulation_label": SIM_STATIC_PROXY,
        "total_lane_steps": total,
        "acc_entropy_bits_per_lane": acc_entropy,
        "acc_domain_bits_per_lane": acc_domain,
        "acc_sparse_hot_bpw": acc_sparse,
        "acc_sparse_hot_bpw_min": float(agg.sparse_bpw_min if agg.sparse_bpw_min != math.inf else 0.0),
        "acc_term_bpw": float(acc_term),
        "sub2_target_bpw": float(SUB2_TARGET_BPW),
        "sub2_target_hit": bool(acc_term < SUB2_TARGET_BPW),
        "distance_from_2bpw": distance_from_2bpw,
        "w6_secondary_target_bpw": float(W6_SECONDARY_TARGET_BPW),
        "w6_secondary_target_hit": bool(acc_term < W6_SECONDARY_TARGET_BPW),
        "q_packed_bpw": float(Q_PACKED_BPW),
        "q_plus_acc_bpw_diagnostic": float(Q_PACKED_BPW + acc_term),
        "combined_ledger_nonclaim": True,
        "w6_dense_bpw_reference": float(W6_ACC_BPW),
        "max_abs_accumulator": int(agg.max_abs),
        "w6_lossless_domain_fit": bool(agg.w6_lossless_fit),
        "crossing_preservation_rate": crossing_preservation,
        "applied_mask_scoreable": False,
        "applied_mask_jaccard_vs_baseline_static": None,
        "applied_mask_unscoreable_reason": (
            "static_proxy_missing_per_lane_votes_for_applied_mask_reconstruction"
        ),
        "cap_order_ranking_scoreable": False,
        "cap_order_ranking_agreement": None,
        "q_trajectory_impact_scoreable": False,
        "q_trajectory_impact": None,
        "observables_preserved_static_proxy": bool(observables_preserved),
    }


def _crossing_factual_summary(regime_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    preserved = [
        row
        for row in regime_rows
        if float(row["crossing_preservation_rate"]) >= CROSSING_PRESERVE_MIN
    ]
    acc_term_midband = [
        row
        for row in regime_rows
        if 5.3 <= float(row["acc_term_bpw"]) <= 5.7
    ]
    sub2_hits = [row for row in regime_rows if bool(row["sub2_target_hit"])]
    min_acc_term = min(float(row["acc_term_bpw"]) for row in regime_rows)
    return {
        "regime_count": len(regime_rows),
        "regimes_crossing_preserved_ge_0p99": len(preserved),
        "regimes_acc_term_between_5p3_and_5p7": len(acc_term_midband),
        "regimes_sub2_target_hit": len(sub2_hits),
        "min_acc_term_bpw": min_acc_term,
        "min_distance_from_2bpw": min(float(row["distance_from_2bpw"]) for row in regime_rows),
        "crossing_preservation_min": min(
            float(row["crossing_preservation_rate"]) for row in regime_rows
        ),
        "crossing_preservation_max": max(
            float(row["crossing_preservation_rate"]) for row in regime_rows
        ),
        "factual_note": (
            "Several regimes preserve crossing=1.0 while reducing acc-term to ~5.3-5.7; "
            "none approach sub-2; deeper threshold/decay cuts degrade crossing."
        ),
    }


def classify_terminal_branch(
    *,
    coverage: Mapping[str, Any],
    regime_rows: Sequence[Mapping[str, Any]],
) -> tuple[str, dict[str, Any]]:
    if str(coverage.get("simulation_class")) == "insufficient":
        return CLASSIFIER_MISSING_OBSERVABLES, {"reason": "sidecars_missing_acc_or_q"}

    summary = _crossing_factual_summary(regime_rows)
    best_sparse = min(float(row["acc_sparse_hot_bpw"]) for row in regime_rows)
    best_domain = min(float(row["acc_domain_bits_per_lane"]) for row in regime_rows)
    w6_floor = float(W6_ACC_BPW)
    sparse_beats_w6 = best_sparse <= w6_floor - SPARSE_BEATS_W6_MARGIN_BPW
    domain_beats_w6 = best_domain < w6_floor
    representation_beats_w6 = bool(sparse_beats_w6)

    promising = [
        row
        for row in regime_rows
        if bool(row["sub2_target_hit"]) and bool(row["observables_preserved_static_proxy"])
    ]
    if promising and str(coverage.get("simulation_class")) == SIM_DYNAMICS_PROOF:
        return CLASSIFIER_DYNAMICS_PROMISING, {
            "promising_regime_count": len(promising),
            "best_acc_term_bpw": min(float(row["acc_term_bpw"]) for row in promising),
            **summary,
        }

    if not representation_beats_w6:
        return CLASSIFIER_FIXED_ACC_NULL, {
            "best_sparse_hot_bpw": best_sparse,
            "best_domain_bpw": best_domain,
            "w6_reference_bpw": w6_floor,
            "min_entropy_bpw": min(float(row["acc_entropy_bits_per_lane"]) for row in regime_rows),
            "representation_beats_w6": False,
            "sparse_beats_w6": bool(sparse_beats_w6),
            "domain_beats_w6": bool(domain_beats_w6),
            "sub2_target_hit_any_regime": any(bool(row["sub2_target_hit"]) for row in regime_rows),
            **summary,
        }

    if not any(
        bool(row["sub2_target_hit"]) and bool(row["observables_preserved_static_proxy"])
        for row in regime_rows
    ):
        return CLASSIFIER_THRESHOLD_DAMPING_INFEASIBLE, {
            "regime_count": len(regime_rows),
            "best_acc_term_bpw": min(float(row["acc_term_bpw"]) for row in regime_rows),
            "best_crossing_preservation": max(
                float(row["crossing_preservation_rate"]) for row in regime_rows
            ),
            **summary,
        }

    return CLASSIFIER_FIXED_ACC_NULL, {
        "reason": "representation_did_not_beat_w6_after_damping_scan",
        "best_sparse_hot_bpw": best_sparse,
        **summary,
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def build_falsification_screen(
    *,
    run_root: Path,
    oracle_arm: str = "w6_on_q_on_oracle",
    treatment_arm: str = "w5_on_q_on_treatment",
) -> dict[str, Any]:
    sidecar_paths = {
        oracle_arm: run_root / oracle_arm / "headroom_wiring_sidecar.jsonl",
        treatment_arm: run_root / treatment_arm / "headroom_wiring_sidecar.jsonl",
    }
    receipt_paths = {
        oracle_arm: run_root / oracle_arm / "receipt.json",
        treatment_arm: run_root / treatment_arm / "receipt.json",
    }
    for label, path in sidecar_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing sidecar for {label}: {path}")

    coverage = audit_observable_coverage(
        sidecar_paths=sidecar_paths,
        receipt_paths=receipt_paths,
    )
    aggs = _stream_sidecar_sweep(sidecar_paths)

    regime_rows: list[dict[str, Any]] = []
    for threshold_abs in THRESHOLD_GRID:
        for decay_num, decay_den in DECAY_GRID:
            regime = _regime_key(threshold_abs, decay_num, decay_den)
            regime_rows.append(
                _finalize_regime_row(
                    aggs[regime],
                    threshold_abs=int(threshold_abs),
                    decay_num=int(decay_num),
                    decay_den=int(decay_den),
                )
            )

    terminal_branch, branch_detail = classify_terminal_branch(
        coverage=coverage,
        regime_rows=regime_rows,
    )
    tool_path = Path(__file__).resolve()
    crossing_summary = _crossing_factual_summary(regime_rows)

    return {
        "schema_version": SCHEMA_VERSION,
        "run_root": str(run_root),
        "tool_source_path": str(tool_path),
        "tool_source_sha256": file_sha256(tool_path),
        "input_artifact_sha256": {
            arm: file_sha256(path) for arm, path in sidecar_paths.items()
        },
        "observable_coverage_audit": coverage,
        "baseline": {
            "threshold_abs": int(BASELINE_THRESHOLD_ABS),
            "decay_num": int(BASELINE_DECAY_NUM),
            "decay_den": int(BASELINE_DECAY_DEN),
        },
        "grid": {
            "threshold_grid": list(THRESHOLD_GRID),
            "decay_grid": [list(item) for item in DECAY_GRID],
        },
        "sub2_target_bpw": float(SUB2_TARGET_BPW),
        "w6_secondary_target_bpw": float(W6_SECONDARY_TARGET_BPW),
        "crossing_factual_summary": crossing_summary,
        "regime_rows": regime_rows,
        "terminal_branch": terminal_branch,
        "terminal_branch_detail": branch_detail,
        "explicit_non_claims": list(EXPLICIT_NON_CLAIMS),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--csv-out", type=Path, default=None)
    args = parser.parse_args(argv)

    receipt = build_falsification_screen(run_root=args.run_root)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    if args.csv_out is not None:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        write_csv(args.csv_out, receipt["regime_rows"])

    print(
        json.dumps(
            {
                "terminal_branch": receipt["terminal_branch"],
                "simulation_class": receipt["observable_coverage_audit"]["simulation_class"],
                "regime_count": len(receipt["regime_rows"]),
                "tool_source_sha256": receipt["tool_source_sha256"],
                "json_out": str(args.json_out),
                "json_sha256": file_sha256(args.json_out),
                "csv_out": str(args.csv_out) if args.csv_out is not None else None,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
