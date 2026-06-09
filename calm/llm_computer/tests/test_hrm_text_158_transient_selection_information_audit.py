from __future__ import annotations

import json
from pathlib import Path

import pytest

from calm.hrm_text_158.native_full_stack.transient_selection_information_audit import (
    BRANCH_COMPRESSIBLE,
    BRANCH_COMPUTE_CONTROL,
    BRANCH_HARNESS_FAIL,
    BRANCH_UPDATE_LAW,
    FIELD_ROLE_FORBIDDEN,
    FIELD_ROLE_OUTCOME,
    FIELD_ROLE_PERSISTENT,
    FIELD_ROLE_TRANSIENT_REF,
    HELD_THRESHOLD_JACCARD,
    HELD_THRESHOLD_ORACLE_TOP1,
    HELD_THRESHOLD_REGRET,
    PERSISTENT_FIELD_SPECS,
    PROVENANCE_CARRIED,
    PROVENANCE_IDENTITY,
    PROVENANCE_STEP_LOCAL,
    SCORE_DEPENDENCY_ENFORCEMENT,
    SELECTION_CARRIED_ONLY,
    SELECTION_CARRIED_PLUS_IDENTITY,
    SELECTION_CARRIED_PLUS_STEP_LOCAL,
    SELECTION_IDENTITY_ONLY,
    SELECTION_INVALID,
    SELECTION_STEP_LOCAL_ONLY,
    SummaryFamilySpec,
    SUMMARY_FAMILY_SPECS,
    _select_with_family,
    _selector_field_view,
    build_field_inventory,
    build_transient_selection_information_audit,
    classify_branch,
    classify_field_role,
    compute_budget_ledger,
    compute_step_metrics,
    evaluate_summary_families,
    partition_steps,
    reconstruct_transient_target,
    provenance_declaration_matches_fields,
    score_field_provenance_tags,
    score_fields_within_selector,
    selection_semantics_for_family,
    threshold_triple_passes,
    uses_outcome_or_forbidden_selector_field,
    verify_input_integrity,
)


def _candidate_row(
    candidate_id: str,
    *,
    local_loss_delta: float,
    rank: int = 1,
    pre_acc: int = 5,
    vote: float = 1.0,
    flat_index: int = 0,
) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "state_key": "layer.weight",
        "flat_index": flat_index,
        "flat_index_quartile": 0,
        "current_rank_position": rank,
        "current_rank_quartile_within_state": 0,
        "vote_value": vote,
        "abs_vote_value": abs(vote),
        "current_margin_abs": 10 - rank,
        "current_q_level": 0,
        "pre_accumulator_i16": pre_acc,
        "new_acc_i32_signed": 20 - rank,
        "proposal_direction": 1,
        "threshold_residual_signed": rank - 2,
        "proximity_to_threshold": rank,
        "tie_band_id": "target",
        "transition_class": "toward_threshold",
        "in_target_tie_band": True,
        "state_candidate_count": 3,
        "candidate_loss": 1.0 + local_loss_delta,
        "local_loss_delta": local_loss_delta,
        "regret_vs_target_tie_band_oracle_top1_local_loss_delta": local_loss_delta,
    }


def _step(step_index: int, rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "optimizer_step_index": step_index,
        "source_kind": "within_tie_band_discriminator",
        "source_table_hash": f"hash-{step_index}",
        "sampled_candidate_table": rows,
    }


def _fixture_stream(*, oracle_prefix: str = "oracle") -> list[dict[str, object]]:
    steps: list[dict[str, object]] = []
    for step_index in range(1, 51):
        oracle_id = f"{oracle_prefix}-{step_index}"
        rows = [
            _candidate_row(
                oracle_id,
                local_loss_delta=-0.95,
                rank=0,
                pre_acc=18,
                vote=3.0,
                flat_index=1,
            ),
            _candidate_row(
                f"decoy-{step_index}",
                local_loss_delta=-0.10,
                rank=2,
                pre_acc=4,
                vote=1.0,
                flat_index=9,
            ),
        ]
        steps.append(_step(step_index, rows))
    return steps


def test_field_inventory_gate_classifies_actual_b2b_fields_once() -> None:
    steps = _fixture_stream()
    inventory = build_field_inventory(steps)
    for field_name in inventory["actual_fields"]:
        assert field_name in inventory["classification"]
    assert inventory["classification"]["local_loss_delta"] == FIELD_ROLE_TRANSIENT_REF
    assert inventory["classification"]["candidate_loss"] == FIELD_ROLE_OUTCOME
    assert inventory["classification"]["pre_accumulator_i16"] == FIELD_ROLE_PERSISTENT
    assert inventory["provenance_tags"]["pre_accumulator_i16"] == PROVENANCE_CARRIED
    assert inventory["provenance_tags"]["vote_value"] == PROVENANCE_STEP_LOCAL
    assert inventory["provenance_tags"]["flat_index"] == PROVENANCE_IDENTITY


def test_field_inventory_gate_forbids_local_loss_as_persistent_selector() -> None:
    role, tag = classify_field_role("local_loss_delta")
    assert role == FIELD_ROLE_TRANSIENT_REF
    assert tag is None
    role, tag = classify_field_role("candidate_loss")
    assert role == FIELD_ROLE_OUTCOME
    assert tag is None


def test_transient_target_reconstruction_hash_matches_b2c_fixture() -> None:
    steps = _fixture_stream()
    selected, observed_hash = reconstruct_transient_target(steps)
    assert all(step[0].startswith("oracle-") for step in selected if step)
    selected_again, observed_hash_again = reconstruct_transient_target(steps)
    assert observed_hash == observed_hash_again

    mismatch_steps = _fixture_stream(oracle_prefix="wrong")
    _, mismatch_hash = reconstruct_transient_target(mismatch_steps)
    assert mismatch_hash != observed_hash


def test_held_split_uses_steps_1_25_fit_26_50_eval() -> None:
    steps = _fixture_stream()
    split = partition_steps(steps)
    assert split["fit_step_count"] == 25
    assert split["held_step_count"] == 25
    assert split["fit_step_range"] == [1, 25]
    assert split["held_step_range"] == [26, 50]


def test_summary_overfit_single_trace_fail_closed() -> None:
    family_results = [
        {
            "family_id": "demo",
            "held_threshold_pass": False,
            "full_trace_threshold_pass": True,
            "summary_overfit_single_trace": True,
            "selection_semantics": SELECTION_CARRIED_ONLY,
            "budget": {"diagnostic_budget_pass": True, "full_physical_budget_pass": True},
        }
    ]
    branch = classify_branch(
        harness_failures=[],
        field_inventory={"field_role_ambiguous": False},
        transient_target_reconstructed=True,
        family_results=family_results,
    )
    assert branch["primary_label"] == BRANCH_HARNESS_FAIL
    assert "summary_overfit_single_trace" in branch["sub_reasons"]


def test_threshold_reducer_requires_090_triple_and_reports_denominators() -> None:
    metrics = {
        "jaccard_vs_transient": HELD_THRESHOLD_JACCARD,
        "regret_capture_vs_oracle": HELD_THRESHOLD_REGRET,
        "oracle_top1_in_selected_rate": HELD_THRESHOLD_ORACLE_TOP1,
        "step_denominator": 25,
    }
    assert threshold_triple_passes(metrics)
    metrics["jaccard_vs_transient"] = HELD_THRESHOLD_JACCARD - 0.01
    assert not threshold_triple_passes(metrics)


def test_budget_reducer_separates_diagnostic_and_full_physical_pass() -> None:
    family = next(
        family for family in SUMMARY_FAMILY_SPECS if family.family_id == "carried_persistent_flip"
    )
    budget = compute_budget_ledger(family)
    assert budget["diagnostic_budget_pass"] is True
    assert budget["full_physical_budget_pass"] is False
    assert budget["q_scale_budget_not_created"] is True
    assert budget["algorithmic_proxy_not_physical_sub2"] is True


def test_branch_precedence_harness_fail_wins() -> None:
    branch = classify_branch(
        harness_failures=["input_integrity_mismatch"],
        field_inventory={"field_role_ambiguous": False},
        transient_target_reconstructed=True,
        family_results=[
            {
                "family_id": "would_win",
                "held_threshold_pass": True,
                "full_trace_threshold_pass": True,
                "summary_overfit_single_trace": False,
                "selection_semantics": SELECTION_CARRIED_ONLY,
                "budget": {
                    "diagnostic_budget_pass": True,
                    "full_physical_budget_pass": True,
                },
            }
        ],
    )
    assert branch["primary_label"] == BRANCH_HARNESS_FAIL


def test_update_law_predictive_disallowed_for_identity_only_signal() -> None:
    family_results = [
        {
            "family_id": "identity_locality_flat",
            "held_threshold_pass": True,
            "full_trace_threshold_pass": True,
            "summary_overfit_single_trace": False,
            "selection_semantics": SELECTION_IDENTITY_ONLY,
            "budget": {"diagnostic_budget_pass": True, "full_physical_budget_pass": False},
        }
    ]
    branch = classify_branch(
        harness_failures=[],
        field_inventory={"field_role_ambiguous": False},
        transient_target_reconstructed=True,
        family_results=family_results,
    )
    assert branch["primary_label"] == BRANCH_COMPUTE_CONTROL


def test_step_local_routing_to_update_law_predictive_candidate() -> None:
    family_results = [
        {
            "family_id": "step_local_vote_proximity",
            "held_threshold_pass": True,
            "full_trace_threshold_pass": True,
            "summary_overfit_single_trace": False,
            "selection_semantics": SELECTION_STEP_LOCAL_ONLY,
            "budget": {"diagnostic_budget_pass": True, "full_physical_budget_pass": False},
        }
    ]
    branch = classify_branch(
        harness_failures=[],
        field_inventory={"field_role_ambiguous": False},
        transient_target_reconstructed=True,
        family_results=family_results,
    )
    assert branch["primary_label"] == BRANCH_UPDATE_LAW
    assert "step_local_selection_dependency" in branch["sub_reasons"]


def test_provenance_tagging_and_selection_semantics() -> None:
    carried_family = next(
        family for family in SUMMARY_FAMILY_SPECS if family.family_id == "carried_persistent_flip"
    )
    step_local_family = next(
        family
        for family in SUMMARY_FAMILY_SPECS
        if family.family_id == "step_local_vote_proximity"
    )
    assert (
        selection_semantics_for_family(carried_family)
        == SELECTION_CARRIED_PLUS_STEP_LOCAL
    )
    assert selection_semantics_for_family(step_local_family) == SELECTION_STEP_LOCAL_ONLY


def test_mismatched_provenance_declaration_fail_closed_never_carried_only() -> None:
    mismatched = SummaryFamilySpec(
        family_id="declared_carried_with_step_local_field",
        selector_fields=("pre_accumulator_i16", "proximity_to_threshold"),
        provenance_tags=frozenset({PROVENANCE_CARRIED}),
        score_fields=("pre_accumulator_i16", "proximity_to_threshold"),
        score_fn=lambda row: float(row.get("pre_accumulator_i16", 0)),
    )
    assert provenance_declaration_matches_fields(mismatched) is False
    assert selection_semantics_for_family(mismatched) == SELECTION_INVALID
    family_results = [
        {
            "family_id": mismatched.family_id,
            "held_threshold_pass": True,
            "full_trace_threshold_pass": True,
            "summary_overfit_single_trace": False,
            "selection_semantics": SELECTION_INVALID,
            "provenance_declaration_mismatch": True,
            "budget": {"diagnostic_budget_pass": True, "full_physical_budget_pass": True},
        }
    ]
    branch = classify_branch(
        harness_failures=[],
        field_inventory={"field_role_ambiguous": False},
        transient_target_reconstructed=True,
        family_results=family_results,
    )
    assert branch["primary_label"] == BRANCH_HARNESS_FAIL
    assert "summary_family_provenance_declaration_mismatch" in branch["sub_reasons"]
    assert branch["primary_label"] != BRANCH_COMPRESSIBLE


def test_carried_persistent_flip_routes_step_local_not_compressible() -> None:
    carried_family = next(
        family for family in SUMMARY_FAMILY_SPECS if family.family_id == "carried_persistent_flip"
    )
    assert provenance_declaration_matches_fields(carried_family) is True
    assert (
        selection_semantics_for_family(carried_family)
        == SELECTION_CARRIED_PLUS_STEP_LOCAL
    )
    family_results = [
        {
            "family_id": "carried_persistent_flip",
            "held_threshold_pass": True,
            "full_trace_threshold_pass": True,
            "summary_overfit_single_trace": False,
            "selection_semantics": SELECTION_CARRIED_PLUS_STEP_LOCAL,
            "provenance_declaration_mismatch": False,
            "budget": {"diagnostic_budget_pass": True, "full_physical_budget_pass": True},
        }
    ]
    branch = classify_branch(
        harness_failures=[],
        field_inventory={"field_role_ambiguous": False},
        transient_target_reconstructed=True,
        family_results=family_results,
    )
    assert branch["primary_label"] == BRANCH_UPDATE_LAW
    assert branch["primary_label"] != BRANCH_COMPRESSIBLE


def test_evaluate_summary_families_reports_provenance_mix() -> None:
    steps = _fixture_stream()
    split = partition_steps(steps)
    transient_by_step, _ = reconstruct_transient_target(steps)
    results = evaluate_summary_families(
        steps,
        fit_steps=split["fit_steps"],
        held_steps=split["held_steps"],
        transient_by_step=transient_by_step,
    )
    assert results
    for result in results:
        assert "feature_provenance_mix" in result
        assert "selection_semantics" in result
        assert result["uses_outcome_or_forbidden_selector_field"] is False


def test_compute_step_metrics_reports_denominators() -> None:
    steps = _fixture_stream()[:2]
    transient_by_step, _ = reconstruct_transient_target(steps)
    selected = transient_by_step
    metrics = compute_step_metrics(
        steps=steps,
        selected_by_step=selected,
        transient_by_step=transient_by_step,
        rate_cap=1,
    )
    assert metrics["step_denominator"] == 2
    assert metrics["jaccard_vs_transient"] == 1.0


def test_verify_input_integrity_detects_sha_mismatch(tmp_path: Path) -> None:
    stable = tmp_path / "stable.ndjson"
    original = tmp_path / "original.ndjson"
    capture = tmp_path / "capture.json"
    b2c = tmp_path / "b2c.json"
    stable.write_text('{"schema":"x"}\n', encoding="utf-8")
    original.write_text('{"schema":"y"}\n', encoding="utf-8")
    capture.write_text("{}", encoding="utf-8")
    b2c.write_text(
        json.dumps(
            {
                "arms": {
                    "transient_resolver_only": {
                        "selected_candidate_ids_hash16": "abcd1234abcd1234"
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    integrity = verify_input_integrity(
        stable_trace_path=stable,
        original_trace_path=original,
        capture_receipt_path=capture,
        b2c_receipt_path=b2c,
    )
    assert integrity["passed"] is False
    assert "stable_copy_trace_sha_mismatch" in integrity["failure_reasons"]


def _write_b2b_trace(path: Path, steps: list[dict[str, object]]) -> Path:
    from calm.hrm_text_158.native_full_stack.accumulator_policy_shadow_screen import (
        B2B_SEQUENTIAL_TRACE_SCHEMA,
        SOURCE_KIND_WITHIN_TIE_BAND_DISCRIMINATOR,
        _stable_hash16,
    )

    lines = [json.dumps({"schema": B2B_SEQUENTIAL_TRACE_SCHEMA}, sort_keys=True)]
    for step in steps:
        rows = list(step["sampled_candidate_table"])
        canonical = sorted(
            [
                {
                    "candidate_id": str(row["candidate_id"]),
                    "current_rank_position": int(row["current_rank_position"]),
                    "local_loss_delta": float(row["local_loss_delta"]),
                    "pre_accumulator_i16": int(row["pre_accumulator_i16"]),
                    "new_acc_i32_signed": int(row["new_acc_i32_signed"]),
                    "proximity_to_threshold": int(row["proximity_to_threshold"]),
                }
                for row in rows
            ],
            key=lambda row: str(row["candidate_id"]),
        )
        lines.append(
            json.dumps(
                {
                    "source_kind": SOURCE_KIND_WITHIN_TIE_BAND_DISCRIMINATOR,
                    "optimizer_step_index": int(step["optimizer_step_index"]),
                    "pre_update_state_hash": f"pre{int(step['optimizer_step_index']):04d}",
                    "source_table_hash": _stable_hash16(canonical),
                    "sampled_candidate_table": canonical,
                    "post_update_telemetry": {"q_changed_count": 1},
                },
                sort_keys=True,
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_build_audit_emits_schema_and_claim_boundary(tmp_path: Path) -> None:
    steps = _fixture_stream()
    trace_path = _write_b2b_trace(tmp_path / "trace.ndjson", steps)
    capture = tmp_path / "capture.json"
    capture.write_text("{}", encoding="utf-8")
    b2c = tmp_path / "b2c.json"
    transient_by_step, transient_hash = reconstruct_transient_target(steps)
    b2c.write_text(
        json.dumps(
            {
                "arms": {
                    "transient_resolver_only": {
                        "selected_candidate_ids_hash16": transient_hash
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    receipt = build_transient_selection_information_audit(
        stable_trace_path=trace_path,
        original_trace_path=trace_path,
        capture_receipt_path=capture,
        b2c_receipt_path=b2c,
    )
    assert receipt["schema_version"] == "hrm_text_158_transient_selection_information_audit/v0"
    assert receipt["claim_boundary"]["measurement_only"] is True
    assert receipt["field_inventory_gate"]["transient_target_reconstructed"] is True
    assert receipt["summary_families"]
    assert receipt["seam_debt"]["screen_module_touched"] is False


def test_score_field_dependency_mismatch_fail_closed() -> None:
    bad = SummaryFamilySpec(
        family_id="score_reads_undeclared_selector_field",
        selector_fields=("pre_accumulator_i16",),
        provenance_tags=frozenset({PROVENANCE_CARRIED}),
        score_fields=("pre_accumulator_i16", "proximity_to_threshold"),
        score_fn=lambda row: float(row.get("pre_accumulator_i16", 0)),
    )
    assert score_fields_within_selector(bad) is False
    family_results = [
        {
            "family_id": bad.family_id,
            "held_threshold_pass": True,
            "full_trace_threshold_pass": True,
            "summary_overfit_single_trace": False,
            "selection_semantics": SELECTION_CARRIED_ONLY,
            "provenance_declaration_mismatch": False,
            "score_field_dependency_mismatch": True,
            "budget": {"diagnostic_budget_pass": True, "full_physical_budget_pass": True},
        }
    ]
    branch = classify_branch(
        harness_failures=[],
        field_inventory={"field_role_ambiguous": False},
        transient_target_reconstructed=True,
        family_results=family_results,
    )
    assert branch["primary_label"] == BRANCH_HARNESS_FAIL
    assert "summary_family_score_field_dependency_mismatch" in branch["sub_reasons"]
    assert branch["primary_label"] != BRANCH_COMPRESSIBLE


def test_carried_only_family_exists_and_evaluates() -> None:
    carried_only = [
        family
        for family in SUMMARY_FAMILY_SPECS
        if selection_semantics_for_family(family) == SELECTION_CARRIED_ONLY
    ]
    assert len(carried_only) >= 1
    family = carried_only[0]
    assert family.family_id == "carried_persistent_bucket"
    assert provenance_declaration_matches_fields(family) is True
    assert score_fields_within_selector(family) is True
    assert PROVENANCE_STEP_LOCAL not in score_field_provenance_tags(family)


def test_carried_plus_identity_does_not_use_step_local_score_fields() -> None:
    family = next(
        family for family in SUMMARY_FAMILY_SPECS if family.family_id == "carried_plus_identity"
    )
    assert score_fields_within_selector(family) is True
    assert selection_semantics_for_family(family) == SELECTION_CARRIED_PLUS_IDENTITY
    for field_name in family.score_fields:
        assert PERSISTENT_FIELD_SPECS.get(field_name) != PROVENANCE_STEP_LOCAL
    assert PROVENANCE_STEP_LOCAL not in score_field_provenance_tags(family)


def test_compressible_branch_requires_carried_only_and_full_budget() -> None:
    family_results = [
        {
            "family_id": "carried_persistent_flip",
            "held_threshold_pass": True,
            "full_trace_threshold_pass": True,
            "summary_overfit_single_trace": False,
            "selection_semantics": SELECTION_CARRIED_ONLY,
            "budget": {"diagnostic_budget_pass": True, "full_physical_budget_pass": True},
        }
    ]
    branch = classify_branch(
        harness_failures=[],
        field_inventory={"field_role_ambiguous": False},
        transient_target_reconstructed=True,
        family_results=family_results,
    )
    assert branch["primary_label"] == BRANCH_COMPRESSIBLE


def test_selector_field_view_drops_undeclared_fields() -> None:
    row = _candidate_row("a", local_loss_delta=-1.0, rank=5, pre_acc=10)
    view = _selector_field_view(row, ("pre_accumulator_i16",))
    assert set(view.keys()) == {"pre_accumulator_i16"}
    assert "proximity_to_threshold" not in view
    assert view["pre_accumulator_i16"] == 10


def test_field_filtered_score_view_prevents_undeclared_score_reads() -> None:
    rows = [
        _candidate_row("high-hidden", local_loss_delta=-0.5, rank=1, pre_acc=1),
        _candidate_row("low-hidden", local_loss_delta=-0.9, rank=1, pre_acc=99),
    ]
    rows[0]["proximity_to_threshold"] = 999
    rows[1]["proximity_to_threshold"] = 1
    family = SummaryFamilySpec(
        family_id="carried_only_pre_acc",
        selector_fields=("pre_accumulator_i16",),
        provenance_tags=frozenset({PROVENANCE_CARRIED}),
        score_fields=("pre_accumulator_i16",),
        score_fn=lambda row: float(row.get("pre_accumulator_i16", 0)),
    )
    selected = _select_with_family(rows, family, rate_cap=1)
    assert selected == ("low-hidden",)


def test_outcome_or_forbidden_selector_field_fail_closed() -> None:
    bad = SummaryFamilySpec(
        family_id="forbidden_selector",
        selector_fields=("candidate_loss",),
        provenance_tags=frozenset({PROVENANCE_CARRIED}),
        score_fields=("candidate_loss",),
        score_fn=lambda row: float(row.get("candidate_loss", 0)),
    )
    assert uses_outcome_or_forbidden_selector_field(bad) is True
    family_results = [
        {
            "family_id": bad.family_id,
            "held_threshold_pass": True,
            "full_trace_threshold_pass": True,
            "summary_overfit_single_trace": False,
            "selection_semantics": SELECTION_INVALID,
            "uses_outcome_or_forbidden_selector_field": True,
            "budget": {"diagnostic_budget_pass": True, "full_physical_budget_pass": True},
        }
    ]
    branch = classify_branch(
        harness_failures=[],
        field_inventory={"field_role_ambiguous": False},
        transient_target_reconstructed=True,
        family_results=family_results,
    )
    assert branch["primary_label"] == BRANCH_HARNESS_FAIL
    assert "summary_family_outcome_or_forbidden_selector_field" in branch["sub_reasons"]


def test_receipt_exposes_score_dependency_enforcement_and_complete_seam_debt(
    tmp_path: Path,
) -> None:
    steps = _fixture_stream()
    trace_path = _write_b2b_trace(tmp_path / "trace.ndjson", steps)
    capture = tmp_path / "capture.json"
    capture.write_text("{}", encoding="utf-8")
    b2c = tmp_path / "b2c.json"
    _, transient_hash = reconstruct_transient_target(steps)
    b2c.write_text(
        json.dumps(
            {
                "arms": {
                    "transient_resolver_only": {
                        "selected_candidate_ids_hash16": transient_hash
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    receipt = build_transient_selection_information_audit(
        stable_trace_path=trace_path,
        original_trace_path=trace_path,
        capture_receipt_path=capture,
        b2c_receipt_path=b2c,
    )
    assert receipt["score_dependency_enforcement"] == SCORE_DEPENDENCY_ENFORCEMENT
    assert receipt["seam_debt"]["private_helper_imports"] == [
        "_stable_hash16",
        "_file_sha256",
        "_load_b2b_sequential_trace_steps",
        "_current_repo_readiness_summary",
    ]
    for result in receipt["summary_families"]:
        assert result["uses_outcome_or_forbidden_selector_field"] is False
