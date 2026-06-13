"""CPU tests for pressure_shape_summary agreement and preflight tooling."""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import pytest
import torch

from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
    RankVoteBin,
    RankVoteSpec,
    build_pressure_shape_summary_v1,
    compact_pressure_shape_summary,
    compact_signed_rank_bin_mass_summary,
    credit_from_weighted_grad,
    project_s1_gradient_to_moves,
    rank_bucketed_int16_votes,
)
from calm.hrm_text_158.native_full_stack.pressure_shape_agreement import (
    branch4_pressure_agreement_established,
    build_pressure_shape_agreement,
    compare_module_vectors,
    extract_module_pressure_vectors,
    receipt_has_pressure_shape_summary,
    verify_pressure_shape_summary_preflight,
)
from calm.hrm_text_158.native_full_stack.selector_support_invariance_analysis import (
    BEAT_MARGIN,
    BRANCH_PRECEDENCE,
    INV_MATCH_HIGH,
    NULL_STRUCTURED_MIN,
    ORDER_MATCH_HIGH,
    SHADOW_INVERTED,
    SHADOW_ORDER_MATCHED,
    SHADOW_RANDOM_NULL,
    _cross_seed_identity_metrics,
    _single_module_identity_metrics,
    _support_order_outcome_metrics,
    classify_branch_precedence,
    compute_within_run_shadow_arms,
    identity_effectively_disjoint,
    run_selector_support_invariance_analysis,
    shadow_ranking_problem,
    verify_pressure_shape_preflight_bundle,
)
from calm.hrm_text_158.native_full_stack.selector_value_analysis import (
    DEFAULT_STATE_KEY,
    ExpectedSeedPair,
)


def _rank_spec() -> RankVoteSpec:
    return RankVoteSpec(
        rank_bins=(
            RankVoteBin(0.0, 0.5, 1),
            RankVoteBin(0.5, 1.0, 4, include_hi=True),
        ),
    )


def _shape_summary(fractions: list[float]) -> dict:
    return {
        "schema": "hrm_text_158_pressure_shape_summary/v0",
        "rank_method": "grouped_bisect_right",
        "rank_bins": [],
        "bin_occupancy_count": [int(round(100 * value)) for value in fractions],
        "bin_mass_fraction": fractions,
        "candidate_count": 100,
        "raw_per_proposal_arrays_included": False,
    }


def _signed_shape_summary(
    *,
    fractions: list[float],
    pos: list[float],
    neg: list[float],
    a1_pos: list[float] | None = None,
    a1_neg: list[float] | None = None,
) -> dict:
    a1_pos = list(a1_pos if a1_pos is not None else pos)
    a1_neg = list(a1_neg if a1_neg is not None else neg)
    total_abs = sum(pos) + sum(neg)
    net = [
        (float(p) - float(n)) / total_abs if total_abs > 0 else 0.0
        for p, n in zip(pos, neg, strict=True)
    ]
    a1_total = sum(a1_pos) + sum(a1_neg)
    a1_net = [
        (float(p) - float(n)) / a1_total if a1_total > 0 else 0.0
        for p, n in zip(a1_pos, a1_neg, strict=True)
    ]
    return {
        "schema": "hrm_text_158_pressure_shape_summary/v1",
        "rank_method": "grouped_bisect_right",
        "rank_bins": [],
        "bin_occupancy_count": [int(round(100 * value)) for value in fractions],
        "bin_mass_fraction": fractions,
        "candidate_count": 100,
        "raw_per_proposal_arrays_included": False,
        "signed_rank_bin_mass": {
            "schema": "hrm_text_158_signed_rank_bin_mass/v0",
            "pos_bin_fraction": pos,
            "neg_bin_fraction": neg,
            "signed_bin_net_fraction": net,
            "total_abs_vote_mass": total_abs,
            "telemetry_only_net_fraction": True,
        },
        "counterfactual_signed_rank_bin_mass": {
            "a1_order_matched": {
                "schema": "hrm_text_158_signed_rank_bin_mass/v0",
                "pos_bin_fraction": a1_pos,
                "neg_bin_fraction": a1_neg,
                "signed_bin_net_fraction": a1_net,
                "total_abs_vote_mass": a1_total,
                "telemetry_only_net_fraction": True,
            },
            "order_matched_basis": "a1_emitted",
        },
    }


def _receipt_with_shapes(
    *,
    module_shapes: dict[str, list[float]],
    steps: range,
    signed_shapes: dict[str, tuple[list[float], list[float]]] | None = None,
) -> dict:
    step_reports = {}
    for step in steps:
        vote_pressure = {}
        for state_key, fractions in module_shapes.items():
            entry = {
                "state_key": state_key,
                "vote_positive_count": 1,
                "vote_negative_count": 1,
            }
            if signed_shapes and state_key in signed_shapes:
                pos, neg = signed_shapes[state_key]
                entry["pressure_shape_summary"] = _signed_shape_summary(
                    fractions=fractions,
                    pos=pos,
                    neg=neg,
                )
            else:
                entry["pressure_shape_summary"] = _shape_summary(fractions)
            vote_pressure[state_key] = entry
        step_reports[str(step)] = {"vote_pressure": vote_pressure, "loss": float(step)}
    return {"step_reports": step_reports}


def test_compact_pressure_shape_summary_matches_rank_bucket_votes() -> None:
    q = torch.tensor([[-1, 0, 0, 1, -1, 1]], dtype=torch.int8)
    grad = torch.tensor([[-1.0, -2.0, 3.0, 4.0, 5.0, -6.0]])
    spec = _rank_spec()
    moves = project_s1_gradient_to_moves(grad, q)
    credit = credit_from_weighted_grad(grad)
    votes = rank_bucketed_int16_votes(credit, moves, spec)
    summary = compact_pressure_shape_summary(credit, moves, spec)

    assert votes.tolist() == [[1, 4, -4, -4, 0, 0]]
    assert summary["schema"] == "hrm_text_158_pressure_shape_summary/v0"
    assert summary["candidate_count"] == 4
    assert summary["bin_mass_fraction"] == [0.25, 0.75]
    assert summary["raw_per_proposal_arrays_included"] is False


def test_compare_module_vectors_reports_median_and_p10() -> None:
    left = {
        "mod.a": [0.8, 0.2],
        "mod.b": [0.5, 0.5],
    }
    right = {
        "mod.a": [0.8, 0.2],
        "mod.b": [0.4, 0.6],
    }
    result = compare_module_vectors(left, right)
    assert result["n_comparable_modules"] == 2
    assert result["median_module_cosine"] == pytest.approx(0.99, abs=0.02)
    assert result["p10_module_cosine"] is not None
    assert result["per_module_cosine"]["mod.a"] == pytest.approx(1.0)


def test_branch4_threshold_gate() -> None:
    passing = {
        "computable": True,
        "median_module_cosine": 0.85,
        "p10_module_cosine": 0.65,
        "n_comparable_modules": 8,
    }
    failing = {
        "computable": True,
        "median_module_cosine": 0.85,
        "p10_module_cosine": 0.55,
        "n_comparable_modules": 8,
    }
    assert branch4_pressure_agreement_established(passing)
    assert not branch4_pressure_agreement_established(failing)


def test_preflight_passes_v1_pressure_shape_summary(tmp_path: Path) -> None:
    receipt = _receipt_with_shapes(
        module_shapes={"mod.a": [0.5, 0.5]},
        steps=range(3, 11),
        signed_shapes={"mod.a": ([0.25, 0.25], [0.25, 0.25])},
    )
    ok, issues = receipt_has_pressure_shape_summary(receipt)
    assert ok, issues
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    payload = verify_pressure_shape_summary_preflight(receipt, receipt_path=path)
    assert payload["pass"] is True
    vectors = extract_module_pressure_vectors(receipt)
    assert vectors["mod.a"] == pytest.approx([0.5, 0.5])


def test_build_pressure_shape_agreement_v1_matches_v0_unsigned_vectors() -> None:
    fractions = [0.7, 0.3]
    v0_receipt = _receipt_with_shapes(
        module_shapes={"mod.a": fractions},
        steps=range(3, 11),
    )
    v1_receipt = _receipt_with_shapes(
        module_shapes={"mod.a": fractions},
        steps=range(3, 11),
        signed_shapes={"mod.a": ([0.5, 0.1], [0.1, 0.1])},
    )
    assert extract_module_pressure_vectors(v0_receipt) == extract_module_pressure_vectors(v1_receipt)


def test_preflight_fails_without_pressure_shape_summary(tmp_path: Path) -> None:
    receipt = _receipt_with_shapes(
        module_shapes={"mod.a": [0.5, 0.5]},
        steps=range(3, 11),
    )
    del receipt["step_reports"]["3"]["vote_pressure"]["mod.a"]["pressure_shape_summary"]
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    ok, issues = receipt_has_pressure_shape_summary(receipt)
    assert not ok
    assert any("missing_pressure_shape_summary" in issue for issue in issues)
    payload = verify_pressure_shape_summary_preflight(receipt, receipt_path=path)
    assert payload["pass"] is False
    assert payload["failure_branch"] == "missing_pressure_shape_summary"


def test_build_pressure_shape_agreement_across_receipts() -> None:
    left = _receipt_with_shapes(
        module_shapes={f"mod.{index}": [0.7, 0.3] for index in range(10)},
        steps=range(3, 11),
    )
    right = _receipt_with_shapes(
        module_shapes={f"mod.{index}": [0.69, 0.31] for index in range(10)},
        steps=range(3, 11),
    )
    artifact = build_pressure_shape_agreement(
        left_receipt=left,
        right_receipt=right,
        left_label="S44",
        right_label="S44_iso43",
    )
    vectors = extract_module_pressure_vectors(left)
    assert len(vectors) == 10
    assert artifact["n_comparable_modules"] == 10
    assert artifact["branch4_pressure_agreement_established"] is True


def test_shadow_arms_compute_from_receipt() -> None:
    receipt = _receipt_with_shapes(
        module_shapes={"mod.a": [0.9, 0.1], "mod.b": [0.2, 0.8]},
        steps=range(3, 11),
        signed_shapes={
            "mod.a": ([0.7, 0.05], [0.05, 0.05]),
            "mod.b": ([0.1, 0.6], [0.05, 0.05]),
        },
    )
    shadows = compute_within_run_shadow_arms(receipt)
    assert shadows["branch5_shadow_evidence_sufficient"] is True
    assert shadows["order_matched_shadow"]["n_module_step_observations"] > 0
    assert shadows[SHADOW_INVERTED]["inverted_direction_degenerate"] is False


def _shadows_ranking_problem() -> dict:
    return {
        "branch5_shadow_evidence_sufficient": True,
        SHADOW_ORDER_MATCHED: {"mean_agreement_with_order_matched_proxy": 0.5},
        SHADOW_INVERTED: {
            "mean_inverted_signed_agreement": 0.9,
            "inverted_direction_degenerate": False,
        },
        SHADOW_RANDOM_NULL: {"mean_uniform_null_distance": 0.2},
    }


def test_shadow_ranking_problem_detects_inverted_beats_order() -> None:
    assert shadow_ranking_problem(_shadows_ranking_problem()) is True


def test_shadow_ranking_problem_rejects_uniform_shape_triviality() -> None:
    shadows = {
        "branch5_shadow_evidence_sufficient": True,
        SHADOW_ORDER_MATCHED: {"mean_agreement_with_order_matched_proxy": 0.5},
        SHADOW_INVERTED: {
            "mean_inverted_signed_agreement": 0.99,
            "inverted_direction_degenerate": False,
        },
        SHADOW_RANDOM_NULL: {"mean_uniform_null_distance": 1e-6},
    }
    assert shadow_ranking_problem(shadows) is False


def test_signed_flip_changes_inverted_summary_direction_asymmetric() -> None:
    q = torch.tensor([[-1, 0, 0, 1, -1, 1]], dtype=torch.int8)
    grad = torch.tensor([[-1.0, -2.0, 3.0, 4.0, 5.0, -6.0]])
    spec = _rank_spec()
    moves = project_s1_gradient_to_moves(grad, q)
    credit = credit_from_weighted_grad(grad)
    primary = compact_signed_rank_bin_mass_summary(credit, moves, spec)
    flipped = compact_signed_rank_bin_mass_summary(credit, -moves, spec)
    assert primary["pos_bin_fraction"] != flipped["pos_bin_fraction"]
    assert primary["neg_bin_fraction"] != flipped["neg_bin_fraction"]


def test_unsigned_bin_mass_unchanged_under_move_flip() -> None:
    q = torch.tensor([[-1, 0, 0, 1, -1, 1]], dtype=torch.int8)
    grad = torch.tensor([[-1.0, -2.0, 3.0, 4.0, 5.0, -6.0]])
    spec = _rank_spec()
    moves = project_s1_gradient_to_moves(grad, q)
    credit = credit_from_weighted_grad(grad)
    primary = compact_pressure_shape_summary(credit, moves, spec)
    flipped = compact_pressure_shape_summary(credit, -moves, spec)
    assert primary["bin_mass_fraction"] == flipped["bin_mass_fraction"]


def test_direction_symmetric_marks_inverted_degenerate_not_one() -> None:
    receipt = _receipt_with_shapes(
        module_shapes={f"mod.{index}": [0.5, 0.5] for index in range(6)},
        steps=range(3, 11),
        signed_shapes={
            f"mod.{index}": ([0.25, 0.25], [0.25, 0.25])
            for index in range(6)
        },
    )
    shadows = compute_within_run_shadow_arms(receipt)
    assert shadows["branch5_shadow_evidence_sufficient"] is False
    assert shadows[SHADOW_INVERTED]["inverted_direction_degenerate"] is True
    assert shadows[SHADOW_INVERTED]["mean_inverted_signed_agreement"] is None


def test_build_pressure_shape_summary_v1_emits_signed_fields() -> None:
    q = torch.tensor([[-1, 0, 0, 1]], dtype=torch.int8)
    grad = torch.tensor([[-1.0, -2.0, 3.0, 4.0]])
    spec = _rank_spec()
    moves = project_s1_gradient_to_moves(grad, q)
    credit = credit_from_weighted_grad(grad)
    summary = build_pressure_shape_summary_v1(credit, moves, spec)
    assert summary["schema"] == "hrm_text_158_pressure_shape_summary/v1"
    assert "signed_rank_bin_mass" in summary
    assert summary["counterfactual_signed_rank_bin_mass"]["order_matched_basis"] == "a1_emitted"


def test_signed_emit_cost_bounded() -> None:
    q = torch.tensor([[-1, 0, 0, 1, -1, 1]], dtype=torch.int8)
    grad = torch.tensor([[-1.0, -2.0, 3.0, 4.0, 5.0, -6.0]])
    spec = _rank_spec()
    moves = project_s1_gradient_to_moves(grad, q)
    credit = credit_from_weighted_grad(grad)
    start = time.perf_counter()
    for _ in range(200):
        build_pressure_shape_summary_v1(credit, moves, spec)
    elapsed_us = (time.perf_counter() - start) * 1e6 / 200.0
    assert elapsed_us < 5000.0


def _branch4_pressure() -> dict:
    return {
        "computable": True,
        "median_module_cosine": 0.85,
        "p10_module_cosine": 0.65,
        "n_comparable_modules": 8,
    }


def test_classify_branch1_screen_harness_or_gate_fail() -> None:
    branch = classify_branch_precedence(
        {
            "pressure_shape_preflight_pass": False,
            "screen_harness_or_gate_fail": True,
        },
    )
    assert branch["branch"] == BRANCH_PRECEDENCE[0]


def test_classify_branch2_support_invariant_win_not_branch6() -> None:
    """D3: high jaccard + agreeing outcome must be branch 2, not insufficient_separation."""
    branch = classify_branch_precedence(
        {
            "pressure_shape_preflight_pass": True,
            "screen_harness_or_gate_fail": False,
            "held_median_topk_jaccard": 0.60,
            "disjoint_fraction": 0.40,
            "outcome_direction_agrees": True,
            "outcome_direction_flips": False,
            "outcome_direction_measurable": True,
            "pressure_shape_agreement": {},
            "shadow_arms": {},
        },
    )
    assert branch["branch"] == BRANCH_PRECEDENCE[1]
    assert branch["branch"] != BRANCH_PRECEDENCE[5]


def test_classify_branch3_requires_disjoint_and_flip() -> None:
    branch_disjoint_flip = classify_branch_precedence(
        {
            "pressure_shape_preflight_pass": True,
            "screen_harness_or_gate_fail": False,
            "held_median_topk_jaccard": 0.05,
            "disjoint_fraction": 0.95,
            "outcome_direction_agrees": False,
            "outcome_direction_flips": True,
            "outcome_direction_measurable": True,
            "pressure_shape_agreement": {},
            "shadow_arms": {},
        },
    )
    assert branch_disjoint_flip["branch"] == BRANCH_PRECEDENCE[2]

    branch_flip_without_disjoint = classify_branch_precedence(
        {
            "pressure_shape_preflight_pass": True,
            "screen_harness_or_gate_fail": False,
            "held_median_topk_jaccard": 0.30,
            "disjoint_fraction": 0.70,
            "outcome_direction_agrees": False,
            "outcome_direction_flips": True,
            "outcome_direction_measurable": True,
            "pressure_shape_agreement": {},
            "shadow_arms": {},
        },
    )
    assert branch_flip_without_disjoint["branch"] != BRANCH_PRECEDENCE[2]


def test_classify_branch4_stable_pressure_cap_churn() -> None:
    branch = classify_branch_precedence(
        {
            "pressure_shape_preflight_pass": True,
            "screen_harness_or_gate_fail": False,
            "held_median_topk_jaccard": 0.30,
            "disjoint_fraction": 0.70,
            "outcome_direction_agrees": True,
            "outcome_direction_flips": False,
            "outcome_direction_measurable": True,
            "pressure_shape_agreement": _branch4_pressure(),
            "shadow_arms": {},
        },
    )
    assert branch["branch"] == BRANCH_PRECEDENCE[3]


def test_classify_branch5_ranking_problem() -> None:
    branch = classify_branch_precedence(
        {
            "pressure_shape_preflight_pass": True,
            "screen_harness_or_gate_fail": False,
            "held_median_topk_jaccard": 0.30,
            "disjoint_fraction": 0.70,
            "outcome_direction_agrees": True,
            "outcome_direction_flips": False,
            "outcome_direction_measurable": True,
            "pressure_shape_agreement": {},
            "shadow_arms": _shadows_ranking_problem(),
        },
    )
    assert branch["branch"] == BRANCH_PRECEDENCE[4]


def test_classify_branch6_insufficient_selector_separation() -> None:
    branch = classify_branch_precedence(
        {
            "pressure_shape_preflight_pass": True,
            "screen_harness_or_gate_fail": False,
            "held_median_topk_jaccard": 0.30,
            "disjoint_fraction": 0.70,
            "outcome_direction_agrees": True,
            "outcome_direction_flips": False,
            "outcome_direction_measurable": True,
            "pressure_shape_agreement": {},
            "shadow_arms": {},
        },
    )
    assert branch["branch"] == BRANCH_PRECEDENCE[5]


def test_classify_branch7_selection_identity_disjoint_but_outcome_robust() -> None:
    branch = classify_branch_precedence(
        {
            "pressure_shape_preflight_pass": True,
            "screen_harness_or_gate_fail": False,
            "held_median_topk_jaccard": 0.05,
            "disjoint_fraction": 0.95,
            "outcome_direction_agrees": True,
            "outcome_direction_flips": False,
            "outcome_direction_measurable": True,
            "pressure_shape_agreement": {},
            "shadow_arms": {},
        },
    )
    assert branch["branch"] == BRANCH_PRECEDENCE[6]


def _minimal_paired_probe_receipt(
    *,
    arm: str,
    delta_sign: int,
    seeds: ExpectedSeedPair,
) -> dict:
    support_batch = {"batch_content_hash16": "deadbeef", "row_ids": [0, 1]}
    step_reports: dict[str, dict] = {}
    for step in range(1, 11):
        off_loss = 1.0
        on_loss = off_loss + float(delta_sign) * 0.5
        step_reports[str(step)] = {
            "loss": on_loss if arm == "on" else off_loss,
            "loss_finite": True,
            "support_batch": support_batch,
            "metrics": {"exact_accuracy": [0.5, 0.5]},
        }
    return {
        "steps_completed": 10,
        "batch": {
            "seed": seeds.curriculum_seed,
            "support_order_seed": seeds.support_order_seed,
        },
        "step_reports": step_reports,
    }


def _receipt_with_applied_indices(
    *,
    applied_by_step: dict[int, list[int]],
    steps: range,
    state_key: str = DEFAULT_STATE_KEY,
) -> dict:
    step_reports = {}
    for step in steps:
        step_reports[str(step)] = {
            "vote_pressure": {state_key: {"state_key": state_key}},
            "step_result": {
                "tensor_stats": {
                    state_key: {
                        "applied_indices": applied_by_step.get(step, []),
                    },
                },
            },
        }
    return {"step_reports": step_reports}


def _receipt_with_multi_module_applied_indices(
    *,
    applied_by_module_step: dict[str, dict[int, list[int]]],
    left_right_offset: dict[str, tuple[list[int], list[int]]] | None = None,
    steps: range,
    as_left: bool = True,
) -> dict:
    step_reports = {}
    for step in steps:
        tensor_stats: dict[str, dict] = {}
        vote_pressure: dict[str, dict] = {}
        for state_key, applied_by_step in applied_by_module_step.items():
            if left_right_offset and state_key in left_right_offset:
                left_indices, right_indices = left_right_offset[state_key]
                applied = left_indices if as_left else right_indices
            else:
                applied = applied_by_step.get(step, [])
            tensor_stats[state_key] = {"applied_indices": applied}
            vote_pressure[state_key] = {"state_key": state_key}
        step_reports[str(step)] = {
            "vote_pressure": vote_pressure,
            "step_result": {"tensor_stats": tensor_stats},
        }
    return {"step_reports": step_reports}


def test_cross_seed_identity_n1_reduces_to_default_state_key() -> None:
    applied = {step: [1, 2, 3] for step in range(3, 11)}
    left = _receipt_with_applied_indices(applied_by_step=applied, steps=range(3, 11))
    right = _receipt_with_applied_indices(
        applied_by_step={step: [4, 5, 6] for step in range(3, 11)},
        steps=range(3, 11),
    )
    aggregate = _cross_seed_identity_metrics(left, right)
    single = _single_module_identity_metrics(left, right, state_key=DEFAULT_STATE_KEY)
    assert aggregate["held_median_topk_jaccard"] == single["held_median_topk_jaccard"]
    assert aggregate["disjoint_fraction"] == single["disjoint_fraction"]
    assert aggregate["n_identity_modules"] == 1


def test_cross_seed_identity_multi_module_median_aggregate() -> None:
    overlap_key = "mod.overlap"
    outlier_key = "mod.outlier"
    steps = range(3, 11)
    left = _receipt_with_multi_module_applied_indices(
        applied_by_module_step={
            overlap_key: {step: [1, 2, 3] for step in steps},
            outlier_key: {step: [1, 2, 3] for step in steps},
        },
        left_right_offset={
            overlap_key: ([1, 2], [1, 3, 4, 5, 6, 7, 8]),
            outlier_key: ([1, 2, 3], [4, 5, 6]),
        },
        steps=steps,
        as_left=True,
    )
    right = _receipt_with_multi_module_applied_indices(
        applied_by_module_step={
            overlap_key: {step: [1, 2, 3] for step in steps},
            outlier_key: {step: [1, 2, 3] for step in steps},
        },
        left_right_offset={
            overlap_key: ([1, 2], [1, 3, 4, 5, 6, 7, 8]),
            outlier_key: ([1, 2, 3], [4, 5, 6]),
        },
        steps=steps,
        as_left=False,
    )
    metrics = _cross_seed_identity_metrics(left, right)
    assert metrics["n_identity_modules"] == 2
    assert metrics["per_module_identity"][overlap_key]["held_median_topk_jaccard"] == pytest.approx(
        1.0 / 8.0,
        rel=1e-6,
    )
    assert metrics["per_module_identity"][outlier_key]["held_median_topk_jaccard"] == pytest.approx(0.0)
    assert metrics["held_median_topk_jaccard"] == pytest.approx(0.0625)
    assert metrics["disjoint_fraction"] == pytest.approx(0.9375)


def test_decision_a_median_disjoint_blocks_outlier_branch3_fire() -> None:
    steps = range(3, 11)
    module_keys = [f"mod.{index}" for index in range(10)]
    overlap_keys = module_keys[:7]
    outlier_keys = module_keys[7:]
    left_right_offset: dict[str, tuple[list[int], list[int]]] = {}
    for key in overlap_keys:
        left_right_offset[key] = ([1, 2], [1, 3, 4, 5, 6, 7, 8])
    for key in outlier_keys:
        left_right_offset[key] = ([1, 2, 3], [4, 5, 6])
    applied_by_module_step = {key: {step: [1, 2, 3] for step in steps} for key in module_keys}
    left = _receipt_with_multi_module_applied_indices(
        applied_by_module_step=applied_by_module_step,
        left_right_offset=left_right_offset,
        steps=steps,
        as_left=True,
    )
    right = _receipt_with_multi_module_applied_indices(
        applied_by_module_step=applied_by_module_step,
        left_right_offset=left_right_offset,
        steps=steps,
        as_left=False,
    )
    metrics = _cross_seed_identity_metrics(left, right)
    module_disjoints = [
        float(metrics["per_module_identity"][key]["disjoint_fraction"])
        for key in module_keys
    ]
    assert statistics.median(module_disjoints) == pytest.approx(0.875, rel=1e-6)
    assert sum(module_disjoints) / len(module_disjoints) == pytest.approx(0.9125, rel=1e-6)
    assert metrics["disjoint_fraction"] == pytest.approx(0.875, rel=1e-6)
    assert metrics["held_median_topk_jaccard"] == pytest.approx(0.125, rel=1e-6)
    assert sum(module_disjoints) / len(module_disjoints) >= 0.90
    assert not identity_effectively_disjoint(
        metrics["held_median_topk_jaccard"],
        metrics["disjoint_fraction"],
    )
    branch = classify_branch_precedence(
        {
            "pressure_shape_preflight_pass": True,
            "screen_harness_or_gate_fail": False,
            "pressure_shape_agreement": {},
            "held_median_topk_jaccard": metrics["held_median_topk_jaccard"],
            "disjoint_fraction": metrics["disjoint_fraction"],
            "outcome_direction_agrees": False,
            "outcome_direction_flips": True,
            "outcome_direction_measurable": True,
            "shadow_arms": {},
        },
    )
    assert branch["branch"] != BRANCH_PRECEDENCE[2]


def test_d6_support_order_flip_computed_from_paired_verdicts() -> None:
    seeds44 = ExpectedSeedPair(44, 44)
    seeds43 = ExpectedSeedPair(44, 43)
    flip_metrics = _support_order_outcome_metrics(
        _minimal_paired_probe_receipt(arm="on", delta_sign=1, seeds=seeds44),
        _minimal_paired_probe_receipt(arm="off", delta_sign=1, seeds=seeds44),
        _minimal_paired_probe_receipt(arm="on", delta_sign=-1, seeds=seeds43),
        _minimal_paired_probe_receipt(arm="off", delta_sign=-1, seeds=seeds43),
        primary_seeds=seeds44,
        isolation_seeds=seeds43,
    )
    assert flip_metrics["support_order_flip_primary_evidence"] is True
    assert flip_metrics["flips"] is True

    agree_metrics = _support_order_outcome_metrics(
        _minimal_paired_probe_receipt(arm="on", delta_sign=1, seeds=seeds44),
        _minimal_paired_probe_receipt(arm="off", delta_sign=1, seeds=seeds44),
        _minimal_paired_probe_receipt(arm="on", delta_sign=1, seeds=seeds43),
        _minimal_paired_probe_receipt(arm="off", delta_sign=1, seeds=seeds43),
        primary_seeds=seeds44,
        isolation_seeds=seeds43,
    )
    assert agree_metrics["support_order_flip_primary_evidence"] is False
    assert agree_metrics["agrees"] is True


def test_d7_identity_disjointness_uses_support_order_pair() -> None:
    """Cross-seed identity must compare S44 ON vs S44_iso43 ON, not within-seed ON/OFF."""
    primary_on = _receipt_with_applied_indices(
        applied_by_step={step: [1, 2, 3] for step in range(3, 11)},
        steps=range(3, 11),
    )
    isolation_on = _receipt_with_applied_indices(
        applied_by_step={step: [4, 5, 6] for step in range(3, 11)},
        steps=range(3, 11),
    )
    primary_off = _receipt_with_applied_indices(
        applied_by_step={step: [1, 2, 3] for step in range(3, 11)},
        steps=range(3, 11),
    )
    cross_seed = _cross_seed_identity_metrics(primary_on, isolation_on)
    within_seed = _cross_seed_identity_metrics(primary_on, primary_off)
    assert cross_seed["held_median_topk_jaccard"] == pytest.approx(0.0)
    assert within_seed["held_median_topk_jaccard"] == pytest.approx(1.0)


def _receipt_with_loss_trajectory(
    *,
    arm: str,
    on_loss_by_step: dict[int, float],
    off_loss_by_step: dict[int, float],
    seeds: ExpectedSeedPair,
) -> dict:
    support_batch = {"batch_content_hash16": "deadbeef", "row_ids": [0, 1]}
    step_reports: dict[str, dict] = {}
    for step in range(1, 11):
        on_loss = on_loss_by_step.get(step, 1.0)
        off_loss = off_loss_by_step.get(step, 1.0)
        step_reports[str(step)] = {
            "loss": on_loss if arm == "on" else off_loss,
            "loss_finite": True,
            "support_batch": support_batch,
            "metrics": {"exact_accuracy": [0.5, 0.5]},
        }
    return {
        "steps_completed": 10,
        "batch": {
            "seed": seeds.curriculum_seed,
            "support_order_seed": seeds.support_order_seed,
        },
        "step_reports": step_reports,
    }


def _on_only_loss_slope_agrees(on_left: dict, on_right: dict) -> bool:
    def slope(receipt: dict) -> float | None:
        losses = [
            float(receipt["step_reports"][str(step)]["loss"])
            for step in range(3, 11)
            if str(step) in receipt.get("step_reports", {})
        ]
        if len(losses) < 2:
            return None
        return losses[-1] - losses[0]

    left = slope(on_left)
    right = slope(on_right)
    if left is None or right is None:
        return False
    if left == 0.0 and right == 0.0:
        return True
    return (left > 0.0) == (right > 0.0)


def test_d8_on_only_slopes_agree_but_paired_verdicts_flip() -> None:
    """ON-only slope agreement must not mask a support-order paired-verdict flip."""
    seeds44 = ExpectedSeedPair(44, 44)
    seeds43 = ExpectedSeedPair(44, 43)
    rising_on = {step: 1.5 + 0.1 * (step - 2) for step in range(2, 11)}
    flat_off = {step: 1.0 for step in range(1, 11)}
    rising_off = {step: 2.0 + 0.15 * (step - 2) for step in range(2, 11)}
    primary_on = _receipt_with_loss_trajectory(
        arm="on",
        on_loss_by_step=rising_on,
        off_loss_by_step=flat_off,
        seeds=seeds44,
    )
    primary_off = _receipt_with_loss_trajectory(
        arm="off",
        on_loss_by_step=rising_on,
        off_loss_by_step=flat_off,
        seeds=seeds44,
    )
    isolation_on = _receipt_with_loss_trajectory(
        arm="on",
        on_loss_by_step=rising_on,
        off_loss_by_step=rising_off,
        seeds=seeds43,
    )
    isolation_off = _receipt_with_loss_trajectory(
        arm="off",
        on_loss_by_step=rising_on,
        off_loss_by_step=rising_off,
        seeds=seeds43,
    )
    assert _on_only_loss_slope_agrees(primary_on, isolation_on) is True
    metrics = _support_order_outcome_metrics(
        primary_on,
        primary_off,
        isolation_on,
        isolation_off,
        primary_seeds=seeds44,
        isolation_seeds=seeds43,
    )
    assert metrics["primary"]["direction"] == "favors_off"
    assert metrics["isolation"]["direction"] == "favors_on"
    assert metrics["flips"] is True
    branch = classify_branch_precedence(
        {
            "pressure_shape_preflight_pass": True,
            "screen_harness_or_gate_fail": False,
            "held_median_topk_jaccard": 0.05,
            "disjoint_fraction": 0.95,
            "outcome_direction_agrees": metrics["agrees"],
            "outcome_direction_flips": metrics["flips"],
            "outcome_direction_measurable": metrics["measurable"],
            "pressure_shape_agreement": {},
            "shadow_arms": {},
        },
    )
    assert branch["branch"] == BRANCH_PRECEDENCE[2]


def test_d8_outcome_agreement_uses_paired_on_off_verdicts() -> None:
    seeds44 = ExpectedSeedPair(44, 44)
    seeds43 = ExpectedSeedPair(44, 43)
    metrics = _support_order_outcome_metrics(
        _minimal_paired_probe_receipt(arm="on", delta_sign=1, seeds=seeds44),
        _minimal_paired_probe_receipt(arm="off", delta_sign=1, seeds=seeds44),
        _minimal_paired_probe_receipt(arm="on", delta_sign=1, seeds=seeds43),
        _minimal_paired_probe_receipt(arm="off", delta_sign=1, seeds=seeds43),
        primary_seeds=seeds44,
        isolation_seeds=seeds43,
    )
    assert metrics["primary"]["verdict"] == "outcome_trajectory_favors_OFF"
    assert metrics["isolation"]["verdict"] == "outcome_trajectory_favors_OFF"
    assert metrics["agrees"] is True
    assert metrics["flips"] is False


def test_d9_preflight_bundle_fails_if_any_pressure_receipt_missing_summary(
    tmp_path: Path,
) -> None:
    primary = _receipt_with_shapes(
        module_shapes={"mod.a": [0.5, 0.5]},
        steps=range(3, 11),
    )
    isolation = _receipt_with_shapes(
        module_shapes={"mod.a": [0.5, 0.5]},
        steps=range(3, 11),
    )
    del isolation["step_reports"]["3"]["vote_pressure"]["mod.a"]["pressure_shape_summary"]
    primary_path = tmp_path / "primary.json"
    isolation_path = tmp_path / "isolation.json"
    primary_path.write_text(json.dumps(primary), encoding="utf-8")
    isolation_path.write_text(json.dumps(isolation), encoding="utf-8")
    bundle = verify_pressure_shape_preflight_bundle(
        {
            "S44_on": (primary, primary_path),
            "S44_iso43_on": (isolation, isolation_path),
        },
    )
    assert bundle["pass"] is False
    assert bundle["failure_branch"] == "missing_pressure_shape_summary"
    assert any("S44_iso43_on:" in issue for issue in bundle["issues"])


def _write_minimal_run_root(tmp_path: Path, *, isolation_missing_shape: bool) -> Path:
    seeds44 = ExpectedSeedPair(44, 44)
    seeds43 = ExpectedSeedPair(44, 43)
    seeds43_corro = ExpectedSeedPair(43, 43)
    shape_receipt = _receipt_with_shapes(
        module_shapes={"mod.a": [0.5, 0.5]},
        steps=range(3, 11),
    )
    bare_receipt = _minimal_paired_probe_receipt(arm="on", delta_sign=1, seeds=seeds44)
    for label, seeds, missing_shape in (
        ("S44", seeds44, False),
        ("S44_iso43", seeds43, isolation_missing_shape),
        ("S43", seeds43_corro, False),
    ):
        for arm in ("on", "off"):
            if arm == "on" and missing_shape:
                receipt = _receipt_with_shapes(
                    module_shapes={"mod.a": [0.5, 0.5]},
                    steps=range(3, 11),
                )
                del receipt["step_reports"]["3"]["vote_pressure"]["mod.a"]["pressure_shape_summary"]
                receipt.update(
                    {
                        "steps_completed": 10,
                        "batch": {
                            "seed": seeds.curriculum_seed,
                            "support_order_seed": seeds.support_order_seed,
                        },
                    },
                )
            elif arm == "on":
                receipt = dict(shape_receipt)
                receipt.update(
                    {
                        "steps_completed": 10,
                        "batch": {
                            "seed": seeds.curriculum_seed,
                            "support_order_seed": seeds.support_order_seed,
                        },
                    },
                )
            else:
                receipt = _minimal_paired_probe_receipt(arm="off", delta_sign=1, seeds=seeds)
            arm_dir = tmp_path / label / arm
            arm_dir.mkdir(parents=True, exist_ok=True)
            (arm_dir / "receipt.json").write_text(
                json.dumps(receipt),
                encoding="utf-8",
            )
    return tmp_path


def test_d9_run_level_missing_isolation_pressure_routes_branch1(
    tmp_path: Path,
) -> None:
    run_root = _write_minimal_run_root(tmp_path, isolation_missing_shape=True)
    summary = run_selector_support_invariance_analysis(run_root)
    assert summary["pressure_shape_preflight"]["pass"] is False
    assert (
        summary["branch_precedence_receipt"]["branch"]
        == BRANCH_PRECEDENCE[0]
    )
    assert (
        summary["branch_precedence_receipt"]["reason"]
        == "missing_pressure_shape_summary_or_harness_gate_fail"
    )


def test_classify_branch8_measurement_ambiguous() -> None:
    branch = classify_branch_precedence(
        {
            "pressure_shape_preflight_pass": True,
            "screen_harness_or_gate_fail": False,
            "held_median_topk_jaccard": None,
            "disjoint_fraction": None,
            "outcome_direction_agrees": False,
            "outcome_direction_flips": False,
            "outcome_direction_measurable": False,
            "pressure_shape_agreement": {},
            "shadow_arms": {},
        },
    )
    assert branch["branch"] == BRANCH_PRECEDENCE[7]
