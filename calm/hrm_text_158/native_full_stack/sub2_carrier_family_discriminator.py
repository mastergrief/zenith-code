"""Read-only sub-2 carrier-family discriminator over banked W8 in-vivo sidecars.

Fail-closed CPU analyzer: inventories observables, records A/B/D as insufficient on
acc+q-only sidecars, computes a C-axis correlation annex, and emits dual booleans
separately (beats W8 dense acc-term vs sub-2 total under a declared q basis).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from calm.hrm_text_158.native_full_stack.persistent_state_budget import (
    BASE3_Q_PACK_GROUP_SIZE,
    PACKED_BASE3_TERNARY_Q_FORMAT,
    R4B_Q_PHYSICAL_BITS_PER_WEIGHT_BASE3,
    TARGET_PHYSICAL_BITS_PER_WEIGHT,
)
from calm.hrm_text_158.native_full_stack.r6_pressure_source_classifier_probe import (
    validate_index,
)
from calm.hrm_text_158.native_full_stack.s3bb_headroom_telemetry import (
    _iter_sidecar_records,
)
from calm.hrm_text_158.native_full_stack.two_tier_carry_reducers import (
    DEFAULT_CROSSING_THRESHOLD_ABS,
)

DISCRIMINATOR_SCHEMA_VERSION = "hrm_text_158_sub2_carrier_family_discriminator/v1"

REQUIRED_SIDECAR_FIELDS: tuple[str, ...] = (
    "accumulator_lanes",
    "q_lanes",
    "schema_version",
    "state_key",
    "step",
)

CLASSIFIER_MISSING_OBSERVABLES_OR_INVALID_WINDOW = "MISSING_OBSERVABLES_OR_INVALID_WINDOW"
CLASSIFIER_A_EVENT_SPARSE_REOPENED = "A_EVENT_SPARSE_REOPENED"
CLASSIFIER_A_EVENT_SPARSE_NEGATIVE_POST_W8 = "A_EVENT_SPARSE_NEGATIVE_POST_W8"
CLASSIFIER_B_APPROX_DENSE_LEAD = "B_APPROX_DENSE_LEAD"
CLASSIFIER_B_VARIANCE_RISK_FAIL = "B_VARIANCE_RISK_FAIL"
CLASSIFIER_C_GROUPED_ACC_LEAD = "C_GROUPED_ACC_LEAD"
CLASSIFIER_C_DECORRELATED_FAIL = "C_DECORRELATED_FAIL"
CLASSIFIER_D_RECOMPUTE_WINDOW_LEAD = "D_RECOMPUTE_WINDOW_LEAD"
CLASSIFIER_D_RECOMPUTE_UNBOUNDED_OR_UNOBSERVABLE = "D_RECOMPUTE_UNBOUNDED_OR_UNOBSERVABLE"
CLASSIFIER_NO_CARRIER_FAMILY_VIABLE = "NO_CARRIER_FAMILY_VIABLE_ON_EXISTING_ARTIFACTS"

CLASSIFIER_PRECEDENCE: tuple[str, ...] = (
    CLASSIFIER_MISSING_OBSERVABLES_OR_INVALID_WINDOW,
    CLASSIFIER_A_EVENT_SPARSE_REOPENED,
    CLASSIFIER_A_EVENT_SPARSE_NEGATIVE_POST_W8,
    CLASSIFIER_B_APPROX_DENSE_LEAD,
    CLASSIFIER_B_VARIANCE_RISK_FAIL,
    CLASSIFIER_C_GROUPED_ACC_LEAD,
    CLASSIFIER_C_DECORRELATED_FAIL,
    CLASSIFIER_D_RECOMPUTE_WINDOW_LEAD,
    CLASSIFIER_D_RECOMPUTE_UNBOUNDED_OR_UNOBSERVABLE,
    CLASSIFIER_NO_CARRIER_FAMILY_VIABLE,
)

DEFAULT_Q_BASIS = PACKED_BASE3_TERNARY_Q_FORMAT
W8_DENSE_ACC_TERM_BPW = 8.0
SUB2_INCLUSIVE_TARGET_BPW = float(TARGET_PHYSICAL_BITS_PER_WEIGHT)
DECLARED_Q_BPW_BASE3 = float(R4B_Q_PHYSICAL_BITS_PER_WEIGHT_BASE3)
ACC_BUDGET_BPW_UNDER_BASE3_Q = SUB2_INCLUSIVE_TARGET_BPW - DECLARED_Q_BPW_BASE3

A_REQUIRED_OBSERVABLES: tuple[str, ...] = (
    "event_bytes",
    "hot_exact_bytes",
    "frontier_bytes",
    "backlog_bytes",
    "backlog_metadata",
    "churn_rate",
    "p95_hot_row_count",
    "p99_hot_row_count",
)
B_REQUIRED_OBSERVABLES: tuple[str, ...] = (
    "per_lane_vote_values",
    "applied_mask_authority",
    "near_threshold_margin_distribution",
)
D_REQUIRED_OBSERVABLES: tuple[str, ...] = (
    "backlog_horizon_log",
    "cap_order_log",
    "reconstruct_from_log_hash",
)

EXPLICIT_NON_CLAIMS: tuple[str, ...] = (
    "no_family_winner_claim",
    "no_science_bank_claim",
    "no_sub2_total_claim",
    "no_held_rules_unlock",
    "w8_faithfulness_only_not_universal_transparency",
    "w7_negative_stands",
    "no_carrier_family_viable_is_shippable_null_not_exhaustive_proof",
    "b_static_proxy_annex_non_authoritative",
)

INSTRUMENTATION_FORK: dict[str, str] = {
    "A_event_sparse": (
        "minimal event/backlog byte ledger on slim diagnostic profile "
        "(events+backlog+hot_exact+metadata bytes per persistent_state_budget R4v seam)"
    ),
    "B_approx_dense": (
        "bounded per-lane vote + applied_mask authority on a small module subset "
        "for companding/Morris error-to-decision sim"
    ),
    "C_grouped_acc": (
        "validated grouping + declared byte model before beats_w8 or sub2 booleans "
        "may be asserted"
    ),
    "D_recompute_window": (
        "serialized backlog/horizon/cap-order log with reconstruct-from-log hash parity"
    ),
}


@dataclass(frozen=True)
class DualBooleanRecord:
    beats_w8_dense_acc_term: bool
    sub2_total_candidate_under_named_q_basis: bool
    q_basis_declared: str
    w8_dense_acc_term_bpw_baseline: float
    sub2_inclusive_target_bpw: float
    acc_budget_bpw_under_declared_q_basis: float
    byte_model_declared: bool
    notes: str


def dual_boolean_record(
    *,
    acc_term_bpw: float | None = None,
    acc_metadata_bpw: float = 0.0,
    byte_model_declared: bool = False,
    notes: str,
) -> DualBooleanRecord:
    beats_w8 = False
    sub2_candidate = False
    if byte_model_declared and acc_term_bpw is not None:
        beats_w8 = float(acc_term_bpw) < W8_DENSE_ACC_TERM_BPW
        inclusive_acc_bpw = float(acc_term_bpw) + float(acc_metadata_bpw)
        sub2_candidate = inclusive_acc_bpw < ACC_BUDGET_BPW_UNDER_BASE3_Q
    return DualBooleanRecord(
        beats_w8_dense_acc_term=beats_w8,
        sub2_total_candidate_under_named_q_basis=sub2_candidate,
        q_basis_declared=DEFAULT_Q_BASIS,
        w8_dense_acc_term_bpw_baseline=W8_DENSE_ACC_TERM_BPW,
        sub2_inclusive_target_bpw=SUB2_INCLUSIVE_TARGET_BPW,
        acc_budget_bpw_under_declared_q_basis=ACC_BUDGET_BPW_UNDER_BASE3_Q,
        byte_model_declared=byte_model_declared,
        notes=notes,
    )


def inventory_observables_from_receipt(receipt: Mapping[str, Any] | None) -> dict[str, Any]:
    present: list[str] = []
    absent: list[str] = list(A_REQUIRED_OBSERVABLES + B_REQUIRED_OBSERVABLES + D_REQUIRED_OBSERVABLES)
    sidecar_fields = list(REQUIRED_SIDECAR_FIELDS)
    if receipt is None:
        return {
            "sidecar_fields_observed": sidecar_fields,
            "present": present,
            "absent": absent,
            "checkpoint_payload_omitted": None,
            "raw_per_proposal_arrays_included": None,
        }

    checkpoint = receipt.get("checkpoint_payload")
    checkpoint_omitted = (
        isinstance(checkpoint, Mapping) and bool(checkpoint.get("checkpoint_payload_omitted"))
    )
    raw_arrays = None
    step_reports = receipt.get("step_reports")
    if isinstance(step_reports, Mapping) and step_reports:
        first_step = sorted(step_reports.keys(), key=lambda value: int(value))[0]
        first_report = step_reports[first_step]
        if isinstance(first_report, Mapping):
            vote_pressure = first_report.get("vote_pressure")
            if isinstance(vote_pressure, Mapping) and vote_pressure:
                first_module = next(iter(vote_pressure.values()))
                if isinstance(first_module, Mapping):
                    shape = first_module.get("pressure_shape_summary")
                    if isinstance(shape, Mapping):
                        raw_arrays = shape.get("raw_per_proposal_arrays_included")

    present.extend(["accumulator_lanes", "q_lanes", "aggregated_vote_pressure_shape"])
    for field in ("accumulator_lanes", "q_lanes", "aggregated_vote_pressure_shape"):
        if field in absent:
            absent.remove(field)

    return {
        "sidecar_fields_observed": sidecar_fields,
        "present": present,
        "absent": absent,
        "checkpoint_payload_omitted": checkpoint_omitted,
        "raw_per_proposal_arrays_included": raw_arrays,
        "event_coded_live_demotion_band": receipt.get("event_coded_live_demotion_band"),
        "persistent_accumulator_event_coded_live": receipt.get(
            "persistent_accumulator_event_coded_live"
        ),
    }


def _layer_type_cohort(state_key: str) -> str:
    if ".attn.gqkv_proj" in state_key:
        return "attn_gqkv_proj"
    if ".attn.o_proj" in state_key:
        return "attn_o_proj"
    if ".mlp.down_proj" in state_key:
        return "mlp_down_proj"
    if ".mlp.gate_up_proj" in state_key:
        return "mlp_gate_up_proj"
    return "other"


def _sign_correlation(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0:
        return 0.0
    a_nonzero = a != 0
    b_nonzero = b != 0
    both = a_nonzero & b_nonzero
    count = int(torch.sum(both).item())
    if count == 0:
        return 0.0
    same = (a[both].sign() == b[both].sign()).float().mean().item()
    return float(same)


def _active_nonzero_mask_jaccard(prev: torch.Tensor, curr: torch.Tensor) -> float:
    """Jaccard overlap of active (nonzero) lane sets at two time points."""

    active_prev = prev != 0
    active_curr = curr != 0
    union = active_prev | active_curr
    union_count = int(torch.sum(union).item())
    if union_count == 0:
        return 1.0
    intersection = active_prev & active_curr
    return float(torch.sum(intersection).item()) / float(union_count)


def _changed_transition_mask(acc_prev: torch.Tensor, acc_curr: torch.Tensor) -> torch.Tensor:
    return acc_prev != acc_curr


def _mask_jaccard(mask_a: torch.Tensor, mask_b: torch.Tensor) -> float:
    union = mask_a | mask_b
    union_count = int(torch.sum(union).item())
    if union_count == 0:
        return 1.0
    intersection = mask_a & mask_b
    return float(torch.sum(intersection).item()) / float(union_count)


def _changed_transition_mask_jaccard(
    acc_before: torch.Tensor,
    acc_mid: torch.Tensor,
    acc_after: torch.Tensor,
) -> float:
    """Jaccard of changed-lane sets for two adjacent transitions (before→mid, mid→after)."""

    transition_prev = _changed_transition_mask(acc_before, acc_mid)
    transition_curr = _changed_transition_mask(acc_mid, acc_after)
    return _mask_jaccard(transition_prev, transition_curr)


def _cross_module_changed_transition_comovement(
    index: Mapping[str, Mapping[int, Mapping[str, Any]]],
    *,
    step_from: int,
    step_to: int,
) -> dict[str, Any]:
    """Pairwise co-movement of changed-transition masks across modules at one step pair."""

    masks_by_module: dict[str, torch.Tensor] = {}
    for state_key in sorted(index.keys()):
        by_step = index[state_key]
        if step_from not in by_step or step_to not in by_step:
            continue
        acc_prev = torch.tensor(by_step[step_from]["accumulator_lanes"], dtype=torch.int16)
        acc_curr = torch.tensor(by_step[step_to]["accumulator_lanes"], dtype=torch.int16)
        if acc_prev.numel() != acc_curr.numel():
            continue
        masks_by_module[state_key] = _changed_transition_mask(acc_prev, acc_curr)

    module_keys = sorted(masks_by_module.keys())
    if len(module_keys) < 2:
        return {
            "step_from": step_from,
            "step_to": step_to,
            "modules_compared": len(module_keys),
            "mean_pairwise_changed_transition_jaccard": None,
            "pairwise_samples": [],
        }

    pairwise: list[dict[str, Any]] = []
    jaccards: list[float] = []
    for left_idx, left_key in enumerate(module_keys):
        for right_key in module_keys[left_idx + 1 :]:
            score = _mask_jaccard(masks_by_module[left_key], masks_by_module[right_key])
            jaccards.append(score)
            pairwise.append(
                {
                    "left_state_key": left_key,
                    "right_state_key": right_key,
                    "changed_transition_mask_jaccard": score,
                }
            )

    return {
        "step_from": step_from,
        "step_to": step_to,
        "modules_compared": len(module_keys),
        "mean_pairwise_changed_transition_jaccard": (
            float(sum(jaccards) / len(jaccards)) if jaccards else None
        ),
        "pairwise_samples": pairwise[:8],
    }


def compute_c_axis_annex(
    index: Mapping[str, Mapping[int, Mapping[str, Any]]],
    *,
    block_sizes: Sequence[int] = (256, 1024),
    max_blocks_per_row: int = 32,
) -> dict[str, Any]:
    module_adjacent: list[dict[str, Any]] = []
    block_summaries: list[dict[str, Any]] = []
    cohort_sign_corr: dict[str, list[float]] = {}
    transition_jaccards: list[float] = []
    cross_module_rows: list[dict[str, Any]] = []

    for state_key in sorted(index.keys()):
        steps = sorted(int(step) for step in index[state_key].keys())
        if len(steps) < 2:
            continue
        cohort = _layer_type_cohort(state_key)
        cohort_sign_corr.setdefault(cohort, [])

        for prev_step, curr_step in zip(steps[:-1], steps[1:], strict=False):
            prev = index[state_key][prev_step]
            curr = index[state_key][curr_step]
            acc_prev = torch.tensor(prev["accumulator_lanes"], dtype=torch.int16)
            acc_curr = torch.tensor(curr["accumulator_lanes"], dtype=torch.int16)
            changed_fraction = float(torch.sum(acc_prev != acc_curr).item()) / float(acc_prev.numel())
            nz_fraction = float(torch.sum(acc_curr != 0).item()) / float(acc_curr.numel())
            sign_corr = _sign_correlation(acc_prev, acc_curr)
            active_nonzero_jaccard = _active_nonzero_mask_jaccard(acc_prev, acc_curr)
            module_adjacent.append(
                {
                    "state_key": state_key,
                    "step_from": prev_step,
                    "step_to": curr_step,
                    "changed_lane_fraction": changed_fraction,
                    "nonzero_lane_fraction": nz_fraction,
                    "sign_correlation_nonzero": sign_corr,
                    "active_nonzero_mask_jaccard": active_nonzero_jaccard,
                }
            )
            cohort_sign_corr[cohort].append(sign_corr)

            lane_count = int(acc_prev.numel())
            for block_size in block_sizes:
                if block_size > lane_count:
                    continue
                max_blocks = min(max_blocks_per_row, lane_count // block_size)
                for block_idx in range(max_blocks):
                    lo = block_idx * block_size
                    hi = lo + block_size
                    block_prev = acc_prev[lo:hi]
                    block_curr = acc_curr[lo:hi]
                    block_summaries.append(
                        {
                            "state_key": state_key,
                            "step_from": prev_step,
                            "step_to": curr_step,
                            "block_size": block_size,
                            "block_index": block_idx,
                            "changed_lane_fraction": float(
                                torch.sum(block_prev != block_curr).item()
                            )
                            / float(block_size),
                            "sign_correlation_nonzero": _sign_correlation(block_prev, block_curr),
                            "active_nonzero_mask_jaccard": _active_nonzero_mask_jaccard(
                                block_prev, block_curr
                            ),
                        }
                    )

        for left, mid, right in zip(steps[:-2], steps[1:-1], steps[2:], strict=False):
            acc_left = torch.tensor(
                index[state_key][left]["accumulator_lanes"], dtype=torch.int16
            )
            acc_mid = torch.tensor(index[state_key][mid]["accumulator_lanes"], dtype=torch.int16)
            acc_right = torch.tensor(
                index[state_key][right]["accumulator_lanes"], dtype=torch.int16
            )
            transition_jaccards.append(
                _changed_transition_mask_jaccard(acc_left, acc_mid, acc_right)
            )

    global_steps = sorted(
        {
            int(step)
            for by_step in index.values()
            for step in by_step.keys()
        }
    )
    for prev_step, curr_step in zip(global_steps[:-1], global_steps[1:], strict=False):
        cross_module_rows.append(
            _cross_module_changed_transition_comovement(
                index,
                step_from=prev_step,
                step_to=curr_step,
            )
        )

    if not module_adjacent:
        return {
            "adjacent_pairs_observed": 0,
            "mean_changed_lane_fraction": None,
            "mean_nonzero_lane_fraction": None,
            "mean_sign_correlation_nonzero": None,
            "mean_active_nonzero_mask_jaccard": None,
            "mean_changed_transition_mask_jaccard": None,
            "cohort_sign_correlation_mean": {},
            "cross_module_changed_transition_comovement": cross_module_rows,
            "mean_cross_module_changed_transition_jaccard": None,
            "informational_c_branch_hint": None,
            "block_summaries_sampled": 0,
        }

    mean_changed = sum(row["changed_lane_fraction"] for row in module_adjacent) / len(
        module_adjacent
    )
    mean_nonzero = sum(row["nonzero_lane_fraction"] for row in module_adjacent) / len(
        module_adjacent
    )
    mean_sign = sum(row["sign_correlation_nonzero"] for row in module_adjacent) / len(
        module_adjacent
    )
    mean_active_nonzero_jaccard = sum(
        row["active_nonzero_mask_jaccard"] for row in module_adjacent
    ) / len(module_adjacent)
    mean_transition_jaccard = (
        float(sum(transition_jaccards) / len(transition_jaccards))
        if transition_jaccards
        else None
    )
    cross_module_scores = [
        float(row["mean_pairwise_changed_transition_jaccard"])
        for row in cross_module_rows
        if row.get("mean_pairwise_changed_transition_jaccard") is not None
    ]
    mean_cross_module_jaccard = (
        float(sum(cross_module_scores) / len(cross_module_scores))
        if cross_module_scores
        else None
    )
    cohort_means = {
        cohort: float(sum(values) / len(values))
        for cohort, values in cohort_sign_corr.items()
        if values
    }

    informational_hint = None
    if mean_nonzero is not None and mean_nonzero >= 0.85 and mean_changed >= 0.50:
        informational_hint = CLASSIFIER_C_DECORRELATED_FAIL

    return {
        "adjacent_pairs_observed": len(module_adjacent),
        "mean_changed_lane_fraction": mean_changed,
        "mean_nonzero_lane_fraction": mean_nonzero,
        "mean_sign_correlation_nonzero": mean_sign,
        "mean_active_nonzero_mask_jaccard": mean_active_nonzero_jaccard,
        "mean_changed_transition_mask_jaccard": mean_transition_jaccard,
        "changed_transition_pairs_observed": len(transition_jaccards),
        "cohort_sign_correlation_mean": cohort_means,
        "cross_module_changed_transition_comovement": cross_module_rows,
        "mean_cross_module_changed_transition_jaccard": mean_cross_module_jaccard,
        "informational_c_branch_hint": informational_hint,
        "block_summaries_sampled": len(block_summaries),
        "module_adjacent_sample": module_adjacent[:4],
        "block_summaries_head": block_summaries[:4],
    }


def _crosses_threshold_tensor(
    acc: torch.Tensor,
    q: torch.Tensor,
    *,
    threshold_abs: int,
) -> torch.Tensor:
    acc_i = acc.to(torch.int64)
    q_i = q.to(torch.int64)
    threshold = int(threshold_abs)
    return ((acc_i >= threshold) & (q_i < 1)) | ((acc_i <= -threshold) & (q_i > -1))


def compute_b_static_proxy_annex(
    index: Mapping[str, Mapping[int, Mapping[str, Any]]],
    *,
    threshold_abs: int = DEFAULT_CROSSING_THRESHOLD_ABS,
    companding_shift: int = 1,
) -> dict[str, Any]:
    """Non-authoritative crossing/companding read; MUST NOT emit B_APPROX_DENSE_LEAD."""

    compared = 0
    reference_flips = 0
    companded_flips = 0
    disagreements = 0

    for state_key in sorted(index.keys()):
        for step in sorted(int(value) for value in index[state_key].keys()):
            record = index[state_key][step]
            acc = torch.tensor(record["accumulator_lanes"], dtype=torch.int32)
            q = torch.tensor(record["q_lanes"], dtype=torch.int32)
            sample = acc[: min(acc.numel(), 4096)]
            q_sample = q[: sample.numel()]
            ref_cross = _crosses_threshold_tensor(sample, q_sample, threshold_abs=threshold_abs)
            comp_acc = torch.clamp(sample >> companding_shift, min=-127, max=127)
            comp_cross = _crosses_threshold_tensor(comp_acc, q_sample, threshold_abs=threshold_abs)
            compared += int(sample.numel())
            reference_flips += int(torch.sum(ref_cross).item())
            companded_flips += int(torch.sum(comp_cross).item())
            disagreements += int(torch.sum(ref_cross != comp_cross).item())

    flip_rate_reference = float(reference_flips) / float(compared) if compared else 0.0
    flip_rate_companded = float(companded_flips) / float(compared) if compared else 0.0
    return {
        "authoritative": False,
        "forbidden_primary_labels": [CLASSIFIER_B_APPROX_DENSE_LEAD],
        "compared_lanes_sampled": compared,
        "reference_crossing_flip_rate": flip_rate_reference,
        "companded_crossing_flip_rate": flip_rate_companded,
        "crossing_flip_disagreement_rate": (
            float(disagreements) / float(compared) if compared else 0.0
        ),
        "notes": (
            "static_proxy_from_acc_q_only; no per-lane vote or applied authority; "
            "cannot support B_APPROX_DENSE_LEAD"
        ),
    }


def _branch_record(
    family: str,
    *,
    observable_sufficiency: str,
    family_verdict: str | None,
    dual: DualBooleanRecord,
    missing_observables: Sequence[str],
) -> dict[str, Any]:
    return {
        "family": family,
        "observable_sufficiency": observable_sufficiency,
        "family_verdict": family_verdict,
        "missing_observables": list(missing_observables),
        "dual_booleans": asdict(dual),
    }


def classify_carrier_families(
    *,
    sidecar_index: Mapping[str, Mapping[int, Mapping[str, Any]]],
    receipt: Mapping[str, Any] | None = None,
    include_b_static_proxy: bool = True,
) -> dict[str, Any]:
    validation_failures = validate_index(sidecar_index)
    inventory = inventory_observables_from_receipt(receipt)

    a_missing = list(A_REQUIRED_OBSERVABLES)
    b_missing = list(B_REQUIRED_OBSERVABLES)
    d_missing = list(D_REQUIRED_OBSERVABLES)

    a_dual = dual_boolean_record(
        byte_model_declared=False,
        notes="A observables absent; no event/backlog byte ledger",
    )
    b_dual = dual_boolean_record(
        byte_model_declared=False,
        notes="B observables absent; no per-lane vote/applied authority",
    )
    c_dual = dual_boolean_record(
        byte_model_declared=False,
        notes="C annex metrics only; no declared grouping byte model",
    )
    d_dual = dual_boolean_record(
        byte_model_declared=False,
        notes="D observables absent; no backlog/horizon/cap-order log",
    )

    branch_records = [
        _branch_record(
            "A",
            observable_sufficiency="INSUFFICIENT",
            family_verdict=None,
            dual=a_dual,
            missing_observables=a_missing,
        ),
        _branch_record(
            "B",
            observable_sufficiency="INSUFFICIENT",
            family_verdict=CLASSIFIER_B_VARIANCE_RISK_FAIL,
            dual=b_dual,
            missing_observables=b_missing,
        ),
        _branch_record(
            "D",
            observable_sufficiency="INSUFFICIENT",
            family_verdict=CLASSIFIER_D_RECOMPUTE_UNBOUNDED_OR_UNOBSERVABLE,
            dual=d_dual,
            missing_observables=d_missing,
        ),
    ]

    c_annex = compute_c_axis_annex(sidecar_index)
    c_branch_verdict = c_annex.get("informational_c_branch_hint")
    branch_records.append(
        _branch_record(
            "C",
            observable_sufficiency="PARTIAL_METRICS_ONLY",
            family_verdict=c_branch_verdict,
            dual=c_dual,
            missing_observables=["declared_grouping_byte_model"],
        )
    )

    b_proxy_annex = compute_b_static_proxy_annex(sidecar_index) if include_b_static_proxy else None

    schema_invalid = bool(validation_failures)
    mandatory_missing = bool(a_missing or b_missing or d_missing)
    primary_classifier = CLASSIFIER_MISSING_OBSERVABLES_OR_INVALID_WINDOW
    if schema_invalid:
        primary_reason = "sidecar_schema_validation_failed"
    elif mandatory_missing:
        primary_reason = "mandatory_A_B_D_observables_absent_on_acc_q_only_sidecars"
    else:
        primary_reason = "invalid_window_or_incomplete_inventory"

    shippable_null = CLASSIFIER_NO_CARRIER_FAMILY_VIABLE

    return {
        "schema_version": DISCRIMINATOR_SCHEMA_VERSION,
        "primary_classifier": primary_classifier,
        "primary_reason": primary_reason,
        "shippable_null_conclusion": shippable_null,
        "classifier_precedence": list(CLASSIFIER_PRECEDENCE),
        "validation_failures": validation_failures,
        "observable_inventory": inventory,
        "branch_records": branch_records,
        "c_axis_annex": c_annex,
        "b_static_proxy_annex": b_proxy_annex,
        "instrumentation_fork": INSTRUMENTATION_FORK,
        "explicit_non_claims": list(EXPLICIT_NON_CLAIMS),
        "q_basis_reference": {
            "declared_q_basis": DEFAULT_Q_BASIS,
            "base3_group_size": BASE3_Q_PACK_GROUP_SIZE,
            "declared_q_bpw": DECLARED_Q_BPW_BASE3,
            "w8_dense_acc_term_bpw": W8_DENSE_ACC_TERM_BPW,
            "sub2_inclusive_target_bpw": SUB2_INCLUSIVE_TARGET_BPW,
            "acc_budget_bpw_under_declared_q_basis": ACC_BUDGET_BPW_UNDER_BASE3_Q,
        },
    }


def index_sidecar_records_filtered(
    path: Path,
    *,
    state_keys: set[str] | None = None,
) -> dict[str, dict[int, dict[str, Any]]]:
    index: dict[str, dict[int, dict[str, Any]]] = {}
    for record in _iter_sidecar_records(path):
        state_key = str(record["state_key"])
        if state_keys is not None and state_key not in state_keys:
            continue
        step = int(record["step"])
        by_step = index.setdefault(state_key, {})
        if step in by_step:
            raise ValueError(f"duplicate sidecar record for {state_key} step {step}")
        by_step[step] = record
    return index


def analyze_w8_in_vivo_run(
    run_root: Path,
    *,
    oracle_arm_dir: str = "int16_oracle_flag_off",
    c_annex_state_keys: Sequence[str] | None = None,
    include_b_static_proxy: bool = True,
) -> dict[str, Any]:
    run_root = Path(run_root)
    oracle_dir = run_root / oracle_arm_dir
    sidecar_path = oracle_dir / "headroom_wiring_sidecar.jsonl"
    receipt_path = oracle_dir / "receipt.json"
    classifier_path = run_root / "classifier_receipt.json"

    receipt = json.loads(receipt_path.read_text(encoding="utf-8")) if receipt_path.is_file() else None
    classifier_receipt = (
        json.loads(classifier_path.read_text(encoding="utf-8"))
        if classifier_path.is_file()
        else None
    )

    if c_annex_state_keys is None:
        c_annex_state_keys = ("model.H_level.core.layers.0.attn.gqkv_proj",)

    sidecar_index = index_sidecar_records_filtered(
        sidecar_path,
        state_keys=set(c_annex_state_keys),
    )
    verdict = classify_carrier_families(
        sidecar_index=sidecar_index,
        receipt=receipt,
        include_b_static_proxy=include_b_static_proxy,
    )
    verdict["run_root"] = str(run_root)
    verdict["oracle_arm_dir"] = oracle_arm_dir
    verdict["sidecar_path"] = str(sidecar_path)
    verdict["c_annex_state_keys"] = list(c_annex_state_keys)
    verdict["w8_classifier_receipt_primary"] = (
        classifier_receipt.get("primary_classifier") if isinstance(classifier_receipt, Mapping) else None
    )
    return verdict


def receipt_to_json(receipt: Mapping[str, Any]) -> str:
    return json.dumps(receipt, indent=2, sort_keys=True)
