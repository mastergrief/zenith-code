"""Cross-seed selector support invariance analysis with branch-5 shadows."""
from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

from calm.hrm_text_158.native_full_stack.pressure_shape_agreement import (
    branch4_pressure_agreement_established,
    build_pressure_shape_agreement,
    load_receipt,
    verify_pressure_shape_summary_preflight,
)
from calm.hrm_text_158.native_full_stack.selector_value_analysis import (
    DEFAULT_STATE_KEY,
    ExpectedSeedPair,
    PRIMARY_STEP_MAX,
    PRIMARY_STEP_MIN,
    extract_cap_window_steps,
    jaccard,
    run_full_analysis,
    run_outcome_analysis,
)

SCHEMA_VERSION = "hrm_text_158_selector_support_invariance_analysis/v0"
SHADOW_SCHEMA_VERSION = "hrm_text_158_shadow_arm_branch5/v1"
BRANCH_PRECEDENCE_SCHEMA = "hrm_text_158_branch_precedence_receipt/v0"

BRANCH_PRECEDENCE: tuple[str, ...] = (
    "screen_harness_or_gate_fail",
    "support_invariant_selector_cost_bound",
    "support_aliasing_drives_verdict_flip",
    "stable_pressure_cap_churn",
    "ranking_or_update_law_problem",
    "insufficient_selector_separation",
    "selection_identity_disjoint_but_outcome_robust",
    "measurement_ambiguous_no_branch",
)

SHADOW_ORDER_MATCHED = "order_matched_shadow"
SHADOW_INVERTED = "inverted_shadow"
SHADOW_RANDOM_NULL = "random_null_shadow"

# Preregistered identity thresholds (1781276293478 / branch_precedence lock 1781276314134).
HELD_MEDIAN_TOPK_JACCARD_SUPPORT_INVARIANT_MIN = 0.50
HELD_MEDIAN_TOPK_JACCARD_EFFECTIVELY_DISJOINT_MAX = 0.10
DISJOINT_FRACTION_EFFECTIVELY_DISJOINT_MIN = 0.90
# Proposed for dual sign-off: branch-4 "low overlap" = below support-invariant band.
BRANCH4_LOW_OVERLAP_HELD_MEDIAN_TOPK_JACCARD_MAX = (
    HELD_MEDIAN_TOPK_JACCARD_SUPPORT_INVARIANT_MIN
)


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _entropy(fractions: Sequence[float]) -> float:
    total = sum(float(value) for value in fractions)
    if total <= 0.0:
        return 0.0
    entropy = 0.0
    for value in fractions:
        prob = float(value) / total
        if prob > 0.0:
            entropy -= prob * math.log(prob)
    return entropy


def _uniform_null_distance(fractions: Sequence[float]) -> float:
    if not fractions:
        return 1.0
    uniform = 1.0 / float(len(fractions))
    return float(sum(abs(float(value) - uniform) for value in fractions) / 2.0)


def _reverse_bin_mass(fractions: Sequence[float]) -> list[float]:
    return list(reversed([float(value) for value in fractions]))


# Branch-5 shadow prereg constants (design v4 artifact ea81106c…).
EPSILON_UNIFORM = 0.02
DELTA_ENTROPY = 0.05
NULL_STRUCTURED_MIN = 0.05
ORDER_MATCH_HIGH = 0.80
INV_MATCH_HIGH = 0.80
BEAT_MARGIN = 0.10
DIRECTION_ASYMMETRY_MIN = 0.05
MIN_NON_DEGENERATE_OBS = 4
MIN_FRACTION_NON_DEGENERATE = 0.5
MIN_DIRECTION_ASYMMETRIC_OBS = 2


def _flatten_signed_2n(pos_fractions: Sequence[float], neg_fractions: Sequence[float]) -> list[float]:
    vector: list[float] = []
    for pos, neg in zip(pos_fractions, neg_fractions, strict=True):
        vector.extend([float(pos), float(neg)])
    return vector


def _swap_pos_neg_2n(pos_fractions: Sequence[float], neg_fractions: Sequence[float]) -> list[float]:
    return _flatten_signed_2n(list(neg_fractions), list(pos_fractions))


def _shape_near_uniform(fractions: Sequence[float]) -> bool:
    if not fractions:
        return True
    n = len(fractions)
    uniform = 1.0 / float(n)
    if max(abs(float(value) - uniform) for value in fractions) < EPSILON_UNIFORM:
        return True
    max_entropy = math.log(float(n)) if n > 1 else 0.0
    return _entropy(fractions) >= (1.0 - DELTA_ENTROPY) * max_entropy


def _max_direction_asymmetry(pos_fractions: Sequence[float], neg_fractions: Sequence[float]) -> float:
    if not pos_fractions:
        return 0.0
    return max(
        abs(float(pos) - float(neg)) / (float(pos) + float(neg) + 1e-12)
        for pos, neg in zip(pos_fractions, neg_fractions, strict=True)
    )


def _signed_agreement_score(primary_2n: Sequence[float], counterfactual_2n: Sequence[float]) -> float:
    if len(primary_2n) != len(counterfactual_2n) or not primary_2n:
        return 0.0
    return 1.0 - _uniform_null_distance(
        [abs(float(a) - float(b)) for a, b in zip(primary_2n, counterfactual_2n, strict=True)],
    )


def _read_signed_mass(summary: Mapping[str, Any]) -> tuple[list[float], list[float]] | None:
    signed = summary.get("signed_rank_bin_mass")
    if not isinstance(signed, Mapping):
        return None
    pos = signed.get("pos_bin_fraction")
    neg = signed.get("neg_bin_fraction")
    if not isinstance(pos, list) or not isinstance(neg, list) or not pos:
        return None
    return [float(value) for value in pos], [float(value) for value in neg]


def _read_a1_signed_mass(summary: Mapping[str, Any]) -> tuple[list[float], list[float], str] | None:
    counterfactual = summary.get("counterfactual_signed_rank_bin_mass")
    if not isinstance(counterfactual, Mapping):
        return None
    a1 = counterfactual.get("a1_order_matched")
    if not isinstance(a1, Mapping):
        return None
    pos = a1.get("pos_bin_fraction")
    neg = a1.get("neg_bin_fraction")
    if not isinstance(pos, list) or not isinstance(neg, list) or not pos:
        return None
    basis = str(counterfactual.get("order_matched_basis") or "a1_emitted")
    return [float(value) for value in pos], [float(value) for value in neg], basis


def _unsigned_reversal_fallback_2n(fractions: Sequence[float]) -> list[float]:
    reversed_bins = _reverse_bin_mass([float(value) for value in fractions])
    return _flatten_signed_2n(reversed_bins, [0.0] * len(reversed_bins))


def compute_within_run_shadow_arms(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """CPU-only branch-5 shadows from compact signed rank-bin mass fields."""

    order_matched_scores: list[float] = []
    inverted_scores: list[float] = []
    random_null_scores: list[float] = []
    steps_considered = 0
    non_degenerate_obs = 0
    direction_asymmetric_obs = 0
    order_matched_bases: set[str] = set()
    for step_key, step_entry in receipt.get("step_reports", {}).items():
        step = int(step_key)
        if step < PRIMARY_STEP_MIN or step > PRIMARY_STEP_MAX:
            continue
        vote_pressure = step_entry.get("vote_pressure")
        if not isinstance(vote_pressure, Mapping) or not vote_pressure:
            continue
        steps_considered += 1
        for pressure_entry in vote_pressure.values():
            if not isinstance(pressure_entry, Mapping):
                continue
            summary = pressure_entry.get("pressure_shape_summary")
            if not isinstance(summary, Mapping):
                continue
            fractions = summary.get("bin_mass_fraction") or []
            if fractions:
                random_null_scores.append(_uniform_null_distance([float(value) for value in fractions]))
            signed_primary = _read_signed_mass(summary)
            if signed_primary is None:
                continue
            pos_primary, neg_primary = signed_primary
            if _shape_near_uniform([float(value) for value in fractions]):
                continue
            non_degenerate_obs += 1
            primary_2n = _flatten_signed_2n(pos_primary, neg_primary)
            a1_payload = _read_a1_signed_mass(summary)
            if a1_payload is not None:
                pos_a1, neg_a1, basis = a1_payload
                order_matched_bases.add(basis)
                order_cf_2n = _flatten_signed_2n(pos_a1, neg_a1)
            elif fractions:
                order_matched_bases.add("reversal_fallback")
                order_cf_2n = _unsigned_reversal_fallback_2n(fractions)
            else:
                continue
            order_matched_scores.append(_signed_agreement_score(primary_2n, order_cf_2n))
            if _max_direction_asymmetry(pos_primary, neg_primary) < DIRECTION_ASYMMETRY_MIN:
                continue
            direction_asymmetric_obs += 1
            inverted_cf_2n = _swap_pos_neg_2n(pos_primary, neg_primary)
            inverted_scores.append(_signed_agreement_score(primary_2n, inverted_cf_2n))
    total_shape_obs = len(random_null_scores)
    fraction_non_degenerate = (
        float(non_degenerate_obs) / float(total_shape_obs) if total_shape_obs else 0.0
    )
    branch5_sufficient = (
        non_degenerate_obs >= MIN_NON_DEGENERATE_OBS
        and fraction_non_degenerate >= MIN_FRACTION_NON_DEGENERATE
    )
    inverted_contributing = direction_asymmetric_obs >= MIN_DIRECTION_ASYMMETRIC_OBS
    return {
        "schema": SHADOW_SCHEMA_VERSION,
        "step_window": {"min": PRIMARY_STEP_MIN, "max": PRIMARY_STEP_MAX},
        "steps_considered": steps_considered,
        "n_non_degenerate_observations": non_degenerate_obs,
        "fraction_non_degenerate": fraction_non_degenerate,
        "n_direction_asymmetric_observations": direction_asymmetric_obs,
        "order_matched_basis_observed": sorted(order_matched_bases),
        SHADOW_ORDER_MATCHED: {
            "mean_agreement_with_order_matched_proxy": _mean(order_matched_scores),
            "n_module_step_observations": len(order_matched_scores),
        },
        SHADOW_INVERTED: {
            "mean_inverted_signed_agreement": _mean(inverted_scores) if inverted_contributing else None,
            "n_module_step_observations": len(inverted_scores),
            "inverted_direction_degenerate": not inverted_contributing,
        },
        SHADOW_RANDOM_NULL: {
            "mean_uniform_null_distance": _mean(random_null_scores),
            "n_module_step_observations": len(random_null_scores),
        },
        "branch5_shadow_evidence_sufficient": branch5_sufficient,
    }


OUTCOME_VERDICT_FAVORS_OFF = "outcome_trajectory_favors_OFF"
OUTCOME_VERDICT_FAVORS_ON = "outcome_trajectory_favors_ON"
OUTCOME_VERDICT_INDIFFERENT = "outcome_indistinguishable"


def _outcome_direction_label(verdict: str | None) -> str | None:
    if verdict == OUTCOME_VERDICT_FAVORS_OFF:
        return "favors_off"
    if verdict == OUTCOME_VERDICT_FAVORS_ON:
        return "favors_on"
    if verdict == OUTCOME_VERDICT_INDIFFERENT:
        return "indistinguishable"
    return None


def _paired_outcome_verdict(
    on: Mapping[str, Any],
    off: Mapping[str, Any],
    expected_seeds: ExpectedSeedPair,
) -> dict[str, Any]:
    """Paired ON/OFF outcome verdict for one support-configuration arm."""

    outcome = run_outcome_analysis(on, off, expected_seeds)
    verdict = outcome.get("verdict")
    direction = _outcome_direction_label(str(verdict) if verdict is not None else None)
    measurable = direction is not None
    return {
        "verdict": verdict,
        "direction": direction,
        "measurable": measurable,
        "guards_ok": bool((outcome.get("guards") or {}).get("ok")),
    }


def _support_order_outcome_metrics(
    primary_on: Mapping[str, Any],
    primary_off: Mapping[str, Any],
    isolation_on: Mapping[str, Any],
    isolation_off: Mapping[str, Any],
    *,
    primary_seeds: ExpectedSeedPair,
    isolation_seeds: ExpectedSeedPair,
) -> dict[str, Any]:
    """Compare paired ON/OFF outcome verdicts across support-order seeds (D6/D8)."""

    primary = _paired_outcome_verdict(primary_on, primary_off, primary_seeds)
    isolation = _paired_outcome_verdict(isolation_on, isolation_off, isolation_seeds)
    measurable = bool(primary["measurable"] and isolation["measurable"])
    primary_direction = primary.get("direction")
    isolation_direction = isolation.get("direction")
    agrees = measurable and primary_direction == isolation_direction
    flips = (
        measurable
        and primary_direction in {"favors_off", "favors_on"}
        and isolation_direction in {"favors_off", "favors_on"}
        and primary_direction != isolation_direction
    )
    return {
        "primary": primary,
        "isolation": isolation,
        "measurable": measurable,
        "agrees": agrees,
        "flips": flips,
        "support_order_flip_primary_evidence": flips,
    }


def verify_pressure_shape_preflight_bundle(
    receipts: Mapping[str, tuple[Mapping[str, Any], Path | None]],
) -> dict[str, Any]:
    """Fail-closed preflight over every receipt feeding pressure comparisons (D9)."""

    per_receipt: dict[str, Any] = {}
    issues: list[str] = []
    for label, (receipt, path) in receipts.items():
        payload = verify_pressure_shape_summary_preflight(receipt, receipt_path=path)
        per_receipt[label] = payload
        if not bool(payload.get("pass")):
            for issue in payload.get("issues") or []:
                issues.append(f"{label}:{issue}")
    ok = len(issues) == 0
    return {
        "schema": "hrm_text_158_pressure_shape_summary_preflight_bundle/v0",
        "pass": ok,
        "failure_branch": None if ok else "missing_pressure_shape_summary",
        "per_receipt": per_receipt,
        "issues": issues,
    }


def _identity_eligible_state_keys(receipt: Mapping[str, Any]) -> list[str]:
    keys: set[str] = set()
    for step_key, step_entry in receipt.get("step_reports", {}).items():
        step = int(step_key)
        if step < PRIMARY_STEP_MIN or step > PRIMARY_STEP_MAX:
            continue
        vote_pressure = step_entry.get("vote_pressure")
        if isinstance(vote_pressure, Mapping) and vote_pressure:
            keys.update(str(state_key) for state_key in vote_pressure)
            continue
        step_result = step_entry.get("step_result")
        tensor_stats = (
            step_result.get("tensor_stats")
            if isinstance(step_result, Mapping)
            else None
        )
        if isinstance(tensor_stats, Mapping):
            keys.update(str(state_key) for state_key in tensor_stats)
    return sorted(keys)


def _single_module_identity_metrics(
    left_on: Mapping[str, Any],
    right_on: Mapping[str, Any],
    *,
    state_key: str,
) -> dict[str, Any]:
    left_steps = extract_cap_window_steps(left_on, state_key=state_key)
    right_steps = extract_cap_window_steps(right_on, state_key=state_key)
    jaccards: list[float] = []
    for step in range(PRIMARY_STEP_MIN, PRIMARY_STEP_MAX + 1):
        left_row = left_steps.get(step, {})
        right_row = right_steps.get(step, {})
        left_applied = list(left_row.get("applied_indices") or [])
        right_applied = list(right_row.get("applied_indices") or [])
        if not left_applied or not right_applied:
            continue
        jaccards.append(float(jaccard(left_applied, right_applied)))
    if not jaccards:
        return {
            "held_median_topk_jaccard": None,
            "disjoint_fraction": None,
            "step_count": 0,
        }
    return {
        "held_median_topk_jaccard": float(statistics.median(jaccards)),
        "disjoint_fraction": float(sum(1.0 - value for value in jaccards) / len(jaccards)),
        "step_count": len(jaccards),
    }


def _cross_seed_identity_metrics(
    left_on: Mapping[str, Any],
    right_on: Mapping[str, Any],
    *,
    state_key: str = DEFAULT_STATE_KEY,
) -> dict[str, Any]:
    eligible_keys = sorted(
        set(_identity_eligible_state_keys(left_on))
        & set(_identity_eligible_state_keys(right_on)),
    )
    if not eligible_keys:
        eligible_keys = [state_key]
    per_module: dict[str, dict[str, Any]] = {}
    module_medians: list[float] = []
    module_disjoints: list[float] = []
    for key in eligible_keys:
        module_metrics = _single_module_identity_metrics(left_on, right_on, state_key=key)
        per_module[key] = module_metrics
        median_value = module_metrics.get("held_median_topk_jaccard")
        disjoint_value = module_metrics.get("disjoint_fraction")
        if median_value is not None and disjoint_value is not None:
            module_medians.append(float(median_value))
            module_disjoints.append(float(disjoint_value))
    if not module_medians:
        return {
            "held_median_topk_jaccard": None,
            "disjoint_fraction": None,
            "step_count": 0,
            "n_identity_modules": 0,
            "per_module_identity": per_module,
            "default_state_key_metrics": per_module.get(state_key),
            "identity_aggregate": "median_per_module_median_jaccard_and_median_disjoint",
        }
    aggregate = {
        "held_median_topk_jaccard": float(statistics.median(module_medians)),
        "disjoint_fraction": float(statistics.median(module_disjoints)),
        "step_count": max(
            int(module_metrics.get("step_count") or 0) for module_metrics in per_module.values()
        ),
        "n_identity_modules": len(module_medians),
        "per_module_identity": per_module,
        "default_state_key_metrics": per_module.get(state_key),
        "identity_aggregate": "median_per_module_median_jaccard_and_median_disjoint",
    }
    if len(eligible_keys) == 1:
        single = per_module[eligible_keys[0]]
        aggregate["held_median_topk_jaccard"] = single.get("held_median_topk_jaccard")
        aggregate["disjoint_fraction"] = single.get("disjoint_fraction")
        aggregate["step_count"] = int(single.get("step_count") or 0)
    return aggregate


def identity_effectively_disjoint(
    held_median_topk_jaccard: float | None,
    disjoint_fraction: float | None,
) -> bool:
    if held_median_topk_jaccard is not None and float(held_median_topk_jaccard) <= (
        HELD_MEDIAN_TOPK_JACCARD_EFFECTIVELY_DISJOINT_MAX
    ):
        return True
    if disjoint_fraction is not None and float(disjoint_fraction) >= (
        DISJOINT_FRACTION_EFFECTIVELY_DISJOINT_MIN
    ):
        return True
    return False


def support_invariant_identity(held_median_topk_jaccard: float | None) -> bool:
    return (
        held_median_topk_jaccard is not None
        and float(held_median_topk_jaccard) >= HELD_MEDIAN_TOPK_JACCARD_SUPPORT_INVARIANT_MIN
    )


def branch4_low_identity_overlap(held_median_topk_jaccard: float | None) -> bool:
    return (
        held_median_topk_jaccard is not None
        and float(held_median_topk_jaccard) < BRANCH4_LOW_OVERLAP_HELD_MEDIAN_TOPK_JACCARD_MAX
    )


def insufficient_selector_separation(
    *,
    held_median_topk_jaccard: float | None,
    pressure_established: bool,
    outcome_direction_flips: bool,
    ranking_problem: bool,
) -> bool:
    if held_median_topk_jaccard is None:
        return False
    held = float(held_median_topk_jaccard)
    middle_band = (
        HELD_MEDIAN_TOPK_JACCARD_EFFECTIVELY_DISJOINT_MAX
        < held
        < HELD_MEDIAN_TOPK_JACCARD_SUPPORT_INVARIANT_MIN
    )
    return (
        middle_band
        and not pressure_established
        and not outcome_direction_flips
        and not ranking_problem
    )


def shadow_ranking_problem(shadows: Mapping[str, Any]) -> bool:
    if not bool(shadows.get("branch5_shadow_evidence_sufficient")):
        return False
    order_matched = (shadows.get(SHADOW_ORDER_MATCHED) or {}).get(
        "mean_agreement_with_order_matched_proxy",
    )
    inverted_block = shadows.get(SHADOW_INVERTED) or {}
    if bool(inverted_block.get("inverted_direction_degenerate")):
        return False
    inverted = inverted_block.get("mean_inverted_signed_agreement")
    random_null = (shadows.get(SHADOW_RANDOM_NULL) or {}).get("mean_uniform_null_distance")
    if order_matched is None or inverted is None or random_null is None:
        return False
    order = float(order_matched)
    inv = float(inverted)
    null_distance = float(random_null)
    if null_distance < NULL_STRUCTURED_MIN:
        return False
    dual_match = order >= ORDER_MATCH_HIGH and inv >= INV_MATCH_HIGH
    inverted_wins = inv >= order + BEAT_MARGIN
    return dual_match or inverted_wins


def classify_branch_precedence(inputs: Mapping[str, Any]) -> dict[str, Any]:
    preflight_ok = bool(inputs.get("pressure_shape_preflight_pass"))
    screen_harness_or_gate_fail = bool(inputs.get("screen_harness_or_gate_fail"))
    pressure = inputs.get("pressure_shape_agreement") or {}
    pressure_established = branch4_pressure_agreement_established(pressure)
    held_median_topk_jaccard = inputs.get("held_median_topk_jaccard")
    disjoint_fraction = inputs.get("disjoint_fraction")
    outcome_agrees = bool(inputs.get("outcome_direction_agrees"))
    outcome_flips = bool(inputs.get("outcome_direction_flips"))
    outcome_measurable = bool(inputs.get("outcome_direction_measurable"))
    shadows = inputs.get("shadow_arms") or {}
    ranking_problem = shadow_ranking_problem(shadows)
    effectively_disjoint = identity_effectively_disjoint(
        held_median_topk_jaccard,
        disjoint_fraction,
    )
    invariant_identity = support_invariant_identity(held_median_topk_jaccard)
    low_overlap = branch4_low_identity_overlap(held_median_topk_jaccard)

    branch = BRANCH_PRECEDENCE[-1]
    reason = "metrics_between_preregistered_thresholds"
    if not preflight_ok or screen_harness_or_gate_fail:
        branch = BRANCH_PRECEDENCE[0]
        reason = "missing_pressure_shape_summary_or_harness_gate_fail"
    elif invariant_identity and outcome_agrees and outcome_measurable:
        branch = BRANCH_PRECEDENCE[1]
        reason = (
            "held_median_topk_jaccard>="
            f"{HELD_MEDIAN_TOPK_JACCARD_SUPPORT_INVARIANT_MIN}_and_outcome_agrees"
        )
    elif effectively_disjoint and outcome_flips and outcome_measurable:
        branch = BRANCH_PRECEDENCE[2]
        reason = "identity_effectively_disjoint_and_outcome_flips"
    elif low_overlap and pressure_established and not (outcome_measurable and outcome_flips):
        branch = BRANCH_PRECEDENCE[3]
        reason = "low_overlap_pressure_agreement_without_outcome_flip"
    elif ranking_problem:
        branch = BRANCH_PRECEDENCE[4]
        reason = "shadow_arms_match_or_beat_primary_ranking_evidence"
    elif insufficient_selector_separation(
        held_median_topk_jaccard=held_median_topk_jaccard,
        pressure_established=pressure_established,
        outcome_direction_flips=outcome_flips,
        ranking_problem=ranking_problem,
    ):
        branch = BRANCH_PRECEDENCE[5]
        reason = "middle_identity_band_without_mechanism_separation"
    elif (
        effectively_disjoint
        and outcome_agrees
        and outcome_measurable
        and not pressure_established
    ):
        branch = BRANCH_PRECEDENCE[6]
        reason = "disjoint_identity_robust_outcome_low_pressure_agreement"
    else:
        branch = BRANCH_PRECEDENCE[7]
        reason = "metrics_between_preregistered_thresholds"

    return {
        "schema": BRANCH_PRECEDENCE_SCHEMA,
        "branch": branch,
        "branch_index": BRANCH_PRECEDENCE.index(branch),
        "reason": reason,
        "thresholds": {
            "held_median_topk_jaccard_support_invariant_min": (
                HELD_MEDIAN_TOPK_JACCARD_SUPPORT_INVARIANT_MIN
            ),
            "held_median_topk_jaccard_effectively_disjoint_max": (
                HELD_MEDIAN_TOPK_JACCARD_EFFECTIVELY_DISJOINT_MAX
            ),
            "disjoint_fraction_effectively_disjoint_min": (
                DISJOINT_FRACTION_EFFECTIVELY_DISJOINT_MIN
            ),
            "branch4_low_overlap_held_median_topk_jaccard_max_proposed": (
                BRANCH4_LOW_OVERLAP_HELD_MEDIAN_TOPK_JACCARD_MAX
            ),
        },
        "inputs": {
            "pressure_shape_preflight_pass": preflight_ok,
            "screen_harness_or_gate_fail": screen_harness_or_gate_fail,
            "pressure_shape_agreement": {
                "median_module_cosine": pressure.get("median_module_cosine"),
                "p10_module_cosine": pressure.get("p10_module_cosine"),
                "n_comparable_modules": pressure.get("n_comparable_modules"),
                "branch4_pressure_agreement_established": pressure_established,
            },
            "held_median_topk_jaccard": held_median_topk_jaccard,
            "disjoint_fraction": disjoint_fraction,
            "outcome_direction_agrees": outcome_agrees,
            "outcome_direction_flips": outcome_flips,
            "outcome_direction_measurable": outcome_measurable,
            "shadow_arms": {
                SHADOW_ORDER_MATCHED: (shadows.get(SHADOW_ORDER_MATCHED) or {}).get(
                    "mean_agreement_with_order_matched_proxy",
                ),
                SHADOW_INVERTED: (shadows.get(SHADOW_INVERTED) or {}).get(
                    "mean_inverted_signed_agreement",
                ),
                SHADOW_RANDOM_NULL: (shadows.get(SHADOW_RANDOM_NULL) or {}).get(
                    "mean_uniform_null_distance",
                ),
            },
        },
        "branch_precedence": list(BRANCH_PRECEDENCE),
        "no_carry_w6_reopen": True,
    }


def run_selector_support_invariance_analysis(
    run_root: Path,
    *,
    primary_label: str = "S44",
    isolation_label: str = "S44_iso43",
    corroboration_label: str = "S43",
) -> dict[str, Any]:
    primary_on = load_receipt(run_root / primary_label / "on" / "receipt.json")
    primary_off = load_receipt(run_root / primary_label / "off" / "receipt.json")
    isolation_on = load_receipt(run_root / isolation_label / "on" / "receipt.json")
    isolation_off = load_receipt(run_root / isolation_label / "off" / "receipt.json")
    corroboration_on = load_receipt(run_root / corroboration_label / "on" / "receipt.json")

    preflight_bundle = verify_pressure_shape_preflight_bundle(
        {
            f"{primary_label}_on": (
                primary_on,
                run_root / primary_label / "on" / "receipt.json",
            ),
            f"{isolation_label}_on": (
                isolation_on,
                run_root / isolation_label / "on" / "receipt.json",
            ),
        },
    )
    pressure_primary_vs_isolation = build_pressure_shape_agreement(
        left_receipt=primary_on,
        right_receipt=isolation_on,
        left_label=primary_label,
        right_label=isolation_label,
    )
    pressure_primary_vs_corroboration = build_pressure_shape_agreement(
        left_receipt=primary_on,
        right_receipt=corroboration_on,
        left_label=primary_label,
        right_label=corroboration_label,
    )
    shadows = compute_within_run_shadow_arms(primary_on)
    identity_metrics = _cross_seed_identity_metrics(primary_on, isolation_on)
    outcome_metrics = _support_order_outcome_metrics(
        primary_on,
        primary_off,
        isolation_on,
        isolation_off,
        primary_seeds=ExpectedSeedPair(44, 44),
        isolation_seeds=ExpectedSeedPair(44, 43),
    )
    branch = classify_branch_precedence(
        {
            "pressure_shape_preflight_pass": bool(preflight_bundle.get("pass")),
            "screen_harness_or_gate_fail": False,
            "pressure_shape_agreement": pressure_primary_vs_isolation,
            "held_median_topk_jaccard": identity_metrics.get("held_median_topk_jaccard"),
            "disjoint_fraction": identity_metrics.get("disjoint_fraction"),
            "outcome_direction_agrees": outcome_metrics.get("agrees", False),
            "outcome_direction_flips": outcome_metrics.get("flips", False),
            "outcome_direction_measurable": outcome_metrics.get("measurable", False),
            "support_order_flip_primary_evidence": outcome_metrics.get(
                "support_order_flip_primary_evidence",
                False,
            ),
            "shadow_arms": shadows,
        },
    )
    return {
        "schema": SCHEMA_VERSION,
        "run_root": str(run_root),
        "primary_comparison": f"{primary_label}_vs_{isolation_label}",
        "pressure_shape_preflight": preflight_bundle,
        "pressure_shape_agreement_primary_vs_isolation": pressure_primary_vs_isolation,
        "pressure_shape_agreement_primary_vs_corroboration": pressure_primary_vs_corroboration,
        "shadow_arm_branch5": shadows,
        "paired_analysis": {
            primary_label: run_full_analysis(
                primary_on,
                primary_off,
                ExpectedSeedPair(44, 44),
            ),
            isolation_label: run_full_analysis(
                isolation_on,
                isolation_off,
                ExpectedSeedPair(44, 43),
            ),
        },
        "primary_vs_isolation": {
            "identity_metrics": identity_metrics,
            "outcome_direction_metrics": outcome_metrics,
            "pressure_shape_agreement": pressure_primary_vs_isolation,
        },
        "branch_precedence_receipt": branch,
    }


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
