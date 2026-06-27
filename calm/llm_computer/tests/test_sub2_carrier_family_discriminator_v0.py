"""CPU fixtures for the sub-2 carrier-family discriminator (read-only)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from calm.hrm_text_158.native_full_stack.r6_pressure_source_classifier_probe import (
    validate_record,
)
from calm.hrm_text_158.native_full_stack.sub2_carrier_family_discriminator import (
    ACC_BUDGET_BPW_UNDER_BASE3_Q,
    CLASSIFIER_B_APPROX_DENSE_LEAD,
    CLASSIFIER_C_DECORRELATED_FAIL,
    CLASSIFIER_C_GROUPED_ACC_LEAD,
    CLASSIFIER_D_RECOMPUTE_UNBOUNDED_OR_UNOBSERVABLE,
    CLASSIFIER_MISSING_OBSERVABLES_OR_INVALID_WINDOW,
    CLASSIFIER_NO_CARRIER_FAMILY_VIABLE,
    REQUIRED_SIDECAR_FIELDS,
    W8_DENSE_ACC_TERM_BPW,
    _active_nonzero_mask_jaccard,
    _changed_transition_mask_jaccard,
    analyze_w8_in_vivo_run,
    classify_carrier_families,
    compute_b_static_proxy_annex,
    compute_c_axis_annex,
    dual_boolean_record,
    inventory_observables_from_receipt,
)

W8_RUN_ROOT = Path(
    "/home/gabe/claw-code-creditdir/transient_fp_credit/"
    "w8_dense_acc_in_vivo_seed43_43_2189e72011"
)


def _record(
    step: int,
    acc: list[int],
    q: list[int],
    *,
    state_key: str = "model.H_level.core.layers.0.attn.gqkv_proj",
) -> dict[str, object]:
    return {
        "schema_version": "hrm_text_158_s3bb_headroom_wiring_sidecar_chunk/v1",
        "step": step,
        "state_key": state_key,
        "accumulator_lanes": [int(v) for v in acc],
        "q_lanes": [int(v) for v in q],
    }


def _index_from_records(records: list[dict[str, object]]) -> dict[str, dict[int, dict[str, object]]]:
    index: dict[str, dict[int, dict[str, object]]] = {}
    for record in records:
        state_key = str(record["state_key"])
        step = int(record["step"])
        index.setdefault(state_key, {})[step] = record
    return index


def test_acc_q_only_schema_fields_present() -> None:
    record = _record(3, [10, 20, 0], [1, 2, 3])
    assert validate_record(record) == []
    assert set(record.keys()) == set(REQUIRED_SIDECAR_FIELDS)


def test_inventory_marks_a_b_d_observables_absent() -> None:
    inventory = inventory_observables_from_receipt(
        {
            "checkpoint_payload": {"checkpoint_payload_omitted": True},
            "step_reports": {
                "3": {
                    "vote_pressure": {
                        "mod0": {
                            "pressure_shape_summary": {
                                "raw_per_proposal_arrays_included": False
                            }
                        }
                    }
                }
            },
        }
    )
    assert "event_bytes" in inventory["absent"]
    assert "per_lane_vote_values" in inventory["absent"]
    assert "backlog_horizon_log" in inventory["absent"]
    assert inventory["raw_per_proposal_arrays_included"] is False


def test_fail_closed_primary_missing_observables() -> None:
    records = [
        _record(3, [10, 12, 14], [1, 1, 1]),
        _record(4, [11, 13, 15], [1, 2, 1]),
    ]
    receipt = classify_carrier_families(
        sidecar_index=_index_from_records(records),
        receipt={"checkpoint_payload": {"checkpoint_payload_omitted": True}},
    )
    assert receipt["primary_classifier"] == CLASSIFIER_MISSING_OBSERVABLES_OR_INVALID_WINDOW
    assert receipt["shippable_null_conclusion"] == CLASSIFIER_NO_CARRIER_FAMILY_VIABLE
    a_branch = next(row for row in receipt["branch_records"] if row["family"] == "A")
    b_branch = next(row for row in receipt["branch_records"] if row["family"] == "B")
    d_branch = next(row for row in receipt["branch_records"] if row["family"] == "D")
    assert a_branch["family_verdict"] is None
    assert b_branch["family_verdict"] != CLASSIFIER_B_APPROX_DENSE_LEAD
    assert d_branch["family_verdict"] == CLASSIFIER_D_RECOMPUTE_UNBOUNDED_OR_UNOBSERVABLE


def test_b_static_proxy_annex_is_non_authoritative() -> None:
    records = [
        _record(3, [9, 10, 11, 12], [0, 0, 0, 0]),
        _record(4, [10, 11, 12, 13], [0, 0, 0, 0]),
    ]
    annex = compute_b_static_proxy_annex(_index_from_records(records))
    assert annex["authoritative"] is False
    assert CLASSIFIER_B_APPROX_DENSE_LEAD in annex["forbidden_primary_labels"]


def test_dual_booleans_separate_and_require_byte_model() -> None:
    without_model = dual_boolean_record(byte_model_declared=False, notes="no model")
    assert without_model.beats_w8_dense_acc_term is False
    assert without_model.sub2_total_candidate_under_named_q_basis is False

    beats_w8 = dual_boolean_record(
        acc_term_bpw=6.0,
        acc_metadata_bpw=0.0,
        byte_model_declared=True,
        notes="beats w8 only",
    )
    assert beats_w8.beats_w8_dense_acc_term is True
    assert beats_w8.sub2_total_candidate_under_named_q_basis is False

    sub2_candidate = dual_boolean_record(
        acc_term_bpw=0.3,
        acc_metadata_bpw=0.05,
        byte_model_declared=True,
        notes="sub2 candidate under base3 q basis",
    )
    assert sub2_candidate.beats_w8_dense_acc_term is True
    assert sub2_candidate.sub2_total_candidate_under_named_q_basis is True
    assert sub2_candidate.acc_budget_bpw_under_declared_q_basis == ACC_BUDGET_BPW_UNDER_BASE3_Q
    assert W8_DENSE_ACC_TERM_BPW == 8.0


def _legacy_wrong_nonzero_overlap_as_changed(prev: torch.Tensor, curr: torch.Tensor) -> float:
    """Pre-repair bug: treated active/nonzero overlap as changed-transition overlap."""

    active_prev = prev != 0
    active_curr = curr != 0
    union = active_prev | active_curr
    union_count = int(torch.sum(union).item())
    if union_count == 0:
        return 1.0
    return float(torch.sum(active_prev & active_curr).item()) / float(union_count)


def test_changed_transition_jaccard_discriminates_from_active_nonzero() -> None:
    records = [
        _record(3, [10, 20, 30, 40], [1, 1, 1, 1]),
        _record(4, [11, 20, 30, 40], [1, 1, 1, 1]),
        _record(5, [11, 21, 30, 40], [1, 1, 1, 1]),
    ]
    index = _index_from_records(records)
    annex = compute_c_axis_annex(index)

    acc3 = torch.tensor(records[0]["accumulator_lanes"], dtype=torch.int16)
    acc4 = torch.tensor(records[1]["accumulator_lanes"], dtype=torch.int16)
    acc5 = torch.tensor(records[2]["accumulator_lanes"], dtype=torch.int16)

    assert _legacy_wrong_nonzero_overlap_as_changed(acc3, acc4) == 1.0
    assert _legacy_wrong_nonzero_overlap_as_changed(acc4, acc5) == 1.0
    assert _active_nonzero_mask_jaccard(acc3, acc4) == 1.0
    assert _changed_transition_mask_jaccard(acc3, acc4, acc5) == 0.0

    assert annex["mean_active_nonzero_mask_jaccard"] == 1.0
    assert annex["mean_changed_transition_mask_jaccard"] == 0.0


def test_cross_module_changed_transition_comovement() -> None:
    mod0 = "model.H_level.core.layers.0.attn.gqkv_proj"
    mod1 = "model.H_level.core.layers.1.attn.gqkv_proj"

    comoving = [
        _record(3, [10, 20, 30], [1, 1, 1], state_key=mod0),
        _record(4, [11, 20, 30], [1, 1, 1], state_key=mod0),
        _record(3, [10, 20, 30], [1, 1, 1], state_key=mod1),
        _record(4, [11, 20, 30], [1, 1, 1], state_key=mod1),
    ]
    comoving_annex = compute_c_axis_annex(_index_from_records(comoving))
    assert comoving_annex["mean_cross_module_changed_transition_jaccard"] == 1.0

    decorrelated = [
        _record(3, [10, 20, 30], [1, 1, 1], state_key=mod0),
        _record(4, [11, 20, 30], [1, 1, 1], state_key=mod0),
        _record(3, [10, 20, 30], [1, 1, 1], state_key=mod1),
        _record(4, [10, 21, 30], [1, 1, 1], state_key=mod1),
    ]
    decorrelated_annex = compute_c_axis_annex(_index_from_records(decorrelated))
    assert decorrelated_annex["mean_cross_module_changed_transition_jaccard"] == 0.0


def test_c_axis_annex_reports_decorrelation_on_dense_churn_fixture() -> None:
    records = [
        _record(3, [10, 20, 30, 40], [1, 1, 1, 1]),
        _record(4, [11, 21, 31, 41], [1, 2, 1, 2]),
        _record(5, [12, 22, 32, 42], [2, 2, 2, 2]),
    ]
    annex = compute_c_axis_annex(_index_from_records(records), block_sizes=(2,))
    assert annex["adjacent_pairs_observed"] == 2
    assert annex["mean_nonzero_lane_fraction"] == 1.0
    assert annex["mean_changed_lane_fraction"] is not None
    assert annex["informational_c_branch_hint"] in (
        CLASSIFIER_C_DECORRELATED_FAIL,
        CLASSIFIER_C_GROUPED_ACC_LEAD,
        None,
    )


def test_classify_emits_dual_booleans_on_every_branch() -> None:
    records = [_record(3, [1, 2], [1, 1]), _record(4, [2, 3], [1, 2])]
    receipt = classify_carrier_families(sidecar_index=_index_from_records(records))
    for branch in receipt["branch_records"]:
        dual = branch["dual_booleans"]
        assert "beats_w8_dense_acc_term" in dual
        assert "sub2_total_candidate_under_named_q_basis" in dual
        assert dual["beats_w8_dense_acc_term"] is False
        assert dual["sub2_total_candidate_under_named_q_basis"] is False


@pytest.mark.slow
def test_analyze_w8_in_vivo_run_dry_run_primary_missing_observables() -> None:
    if not W8_RUN_ROOT.is_dir():
        pytest.skip("2189e72011 artifacts not present on this host")
    receipt = analyze_w8_in_vivo_run(
        W8_RUN_ROOT,
        c_annex_state_keys=("model.H_level.core.layers.0.attn.gqkv_proj",),
    )
    assert receipt["primary_classifier"] == CLASSIFIER_MISSING_OBSERVABLES_OR_INVALID_WINDOW
    assert receipt["shippable_null_conclusion"] == CLASSIFIER_NO_CARRIER_FAMILY_VIABLE
    assert receipt["w8_classifier_receipt_primary"] == "W8_IN_VIVO_CONFIRMED"
    assert receipt["c_axis_annex"]["adjacent_pairs_observed"] >= 1
    c_annex = receipt["c_axis_annex"]
    assert "mean_active_nonzero_mask_jaccard" in c_annex
    assert "mean_changed_transition_mask_jaccard" in c_annex
    assert "mean_cross_module_changed_transition_jaccard" in c_annex
    assert c_annex["informational_c_branch_hint"] == CLASSIFIER_C_DECORRELATED_FAIL
    b_annex = receipt["b_static_proxy_annex"]
    assert b_annex is not None
    assert b_annex["authoritative"] is False
    json.dumps(receipt)
