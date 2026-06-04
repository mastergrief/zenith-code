"""Front-C identity adapter/classifier tests."""
from __future__ import annotations

from copy import deepcopy
import json

import pytest

from calm.hrm_text_158.native_full_stack.front_c_fixtures import (
    front_c_count_only_timeline_artifact,
    front_c_prior_large_q_ledger,
    front_c_timeline_churn_fixture,
    front_c_zero_drift_decision_paths,
)
from calm.hrm_text_158.native_full_stack.front_c_identity_emitter import (
    FRONT_C_AMBIGUOUS_SPLIT_CONTRACT,
    FRONT_C_CANONICAL_STATE_KEY_SEMANTICS,
    FRONT_C_COUNT_ONLY,
    FRONT_C_DENSE_DECISION_SOURCE,
    FRONT_C_IDENTITY_EXTRACTABLE,
    FRONT_C_LOCAL_FLAT_INDEX_SEMANTICS,
    FRONT_C_PATH_B_CARRY_FORWARD_FOLDS,
    FRONT_C_RUN_DERIVED_ARTIFACT,
    FRONT_C_SPARSE_DECISION_SOURCE,
    FRONT_C_STATE_LAYOUT_HASH_SEMANTICS,
    FRONT_C_SYNTHETIC_FIXTURE_ARTIFACT,
    classify_front_c_conflict_overlap,
    classify_front_c_saved_audit_root,
    front_c_report_from_identity_artifact,
    require_front_c_identity_extractable_saved_audit_root,
    validate_front_c_identity_artifact,
)
from calm.hrm_text_158.native_full_stack.front_c_projection import (
    COUNT_ONLY_ARTIFACT_REJECTION,
    validate_front_c_projection_report,
)


def _valid_identity_payload(*, run_derived: bool = False) -> dict:
    row = front_c_prior_large_q_ledger()
    dense, sparse = front_c_zero_drift_decision_paths()
    state_layout_hash = "fixture-state-layout-sha256"
    metadata_hash = "fixture-state-layout-metadata-sha256"
    timeline = [
        step.to_dict()
        for step in front_c_timeline_churn_fixture(
            eligible_weight_count=row.eligible_weight_count,
        )
    ]
    derivation = (
        {
            "artifact_class": FRONT_C_RUN_DERIVED_ARTIFACT,
            "dense_source": FRONT_C_DENSE_DECISION_SOURCE,
            "sparse_source": FRONT_C_SPARSE_DECISION_SOURCE,
            "independent_sparse_derivation": True,
            "source_artifact_id": "run-derived-fixture",
            "state_layout_metadata_sha256": metadata_hash,
        }
        if run_derived
        else {
            "artifact_class": FRONT_C_SYNTHETIC_FIXTURE_ARTIFACT,
            "synthetic_fixture_non_claim": True,
            "independent_sparse_derivation": False,
            "source_artifact_id": "synthetic-fixture",
            "state_layout_metadata_sha256": metadata_hash,
        }
    )
    return {
        "timeline": timeline,
        "dense_decision_path": dense.to_dict(),
        "sparse_decision_path": sparse.to_dict(),
        "state_metadata": {
            "state_key_semantics": FRONT_C_CANONICAL_STATE_KEY_SEMANTICS,
            "flat_index_semantics": FRONT_C_LOCAL_FLAT_INDEX_SEMANTICS,
            "state_hash_semantics": FRONT_C_STATE_LAYOUT_HASH_SEMANTICS,
            "state_layout_metadata_sha256": metadata_hash,
            "states": [
                {
                    "state_key": "fixture",
                    "logical_shape": [row.eligible_weight_count],
                    "eligible_weight_count": row.eligible_weight_count,
                    "state_layout_sha256": state_layout_hash,
                },
            ],
            "step_state_layout_sha256": {
                str(item["step"]): {"fixture": state_layout_hash}
                for item in timeline
            },
        },
        "decision_path_derivation": derivation,
        "value_bits_per_row": 16,
        "flag_bits_per_row": 2,
    }


def test_valid_synthetic_identity_artifact_feeds_scaffold_without_live_viability_claim():
    row = front_c_prior_large_q_ledger()
    payload = _valid_identity_payload()

    validation = validate_front_c_identity_artifact(payload, q_ledger_row=row)
    report = front_c_report_from_identity_artifact(payload, q_ledger_row=row)

    assert validation.status == FRONT_C_IDENTITY_EXTRACTABLE
    assert validation.synthetic_fixture is True
    assert validation.claimed_front_c_viable is False
    assert validation.eligible_weight_count == row.eligible_weight_count
    assert report.decision_equivalence.zero_drift is True
    assert "no Front-C viability claim" in " ".join(report.non_claims)
    validate_front_c_projection_report(report, claimed_front_c_viable=False)


def test_saved_b2_like_count_hash_root_is_classified_not_identity_extractable(tmp_path):
    summary_dir = tmp_path / "audits" / "step_0000"
    summary_dir.mkdir(parents=True)
    (summary_dir / "summary.json").write_text(
        json.dumps(
            {
                "dry_run": True,
                "checkpoint_written": False,
                "checkpoint_payload_summary": {
                    "tensor_summary_count": 0,
                    "authoritative_state_sha256": "abc",
                    "updater_config_sha256": "def",
                },
                "target_audit": {
                    "batch_reports": [
                        {
                            "metadata": {
                                "row_ids": ["0:deadbeef"],
                                "row_count": 1,
                            },
                            "failure_examples": [{"row_index": 0}],
                        },
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    report = classify_front_c_saved_audit_root(tmp_path)

    assert report.status == FRONT_C_COUNT_ONLY
    assert report.identity_extractable is False
    assert "timeline" in report.missing_required_keys
    assert "row_ids" in report.observed_count_or_hash_keys
    with pytest.raises(ValueError, match="not identity-extractable"):
        require_front_c_identity_extractable_saved_audit_root(tmp_path)


def test_split_required_keys_across_files_is_ambiguous_not_identity_extractable(tmp_path):
    first = tmp_path / "audits" / "step_0000"
    second = tmp_path / "audits" / "step_0020"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    (first / "summary.json").write_text(
        json.dumps({"timeline": [], "dense_decision_path": {}}),
        encoding="utf-8",
    )
    (second / "summary.json").write_text(
        json.dumps(
            {
                "sparse_decision_path": {},
                "state_metadata": {},
                "decision_path_derivation": {},
            }
        ),
        encoding="utf-8",
    )

    report = classify_front_c_saved_audit_root(tmp_path)

    assert report.status == FRONT_C_AMBIGUOUS_SPLIT_CONTRACT
    assert report.identity_extractable is False
    assert report.matched_artifact_path == ""
    assert set(report.aggregate_only_required_keys) == {
        "timeline",
        "dense_decision_path",
        "sparse_decision_path",
        "state_metadata",
        "decision_path_derivation",
    }
    assert report.missing_required_keys == ()
    with pytest.raises(ValueError, match="observed only in aggregate"):
        require_front_c_identity_extractable_saved_audit_root(tmp_path)


def test_self_contained_artifact_reports_match_path_and_external_q_ledger_need(tmp_path):
    summary_dir = tmp_path / "audits" / "step_0000"
    summary_dir.mkdir(parents=True)
    payload = _valid_identity_payload()
    artifact_path = summary_dir / "summary.json"
    artifact_path.write_text(json.dumps(payload), encoding="utf-8")

    report = classify_front_c_saved_audit_root(tmp_path)

    assert report.status == FRONT_C_IDENTITY_EXTRACTABLE
    assert report.identity_extractable is True
    assert report.matched_artifact_path == str(artifact_path)
    assert report.external_q_ledger_required is True
    assert report.aggregate_only_required_keys == ()


def test_adapter_rejects_mixed_count_identity_artifacts_before_scaffold_claims_density():
    row = front_c_prior_large_q_ledger()
    payload = _valid_identity_payload()
    payload["timeline"] = [
        {
            "step": 0,
            "eligible_weight_count": row.eligible_weight_count,
            "active_next_step_keys": [{"state_key": "fixture", "flat_index": 1}],
            "decision_relevant_exact_count": 1,
        },
    ]

    with pytest.raises(ValueError, match=COUNT_ONLY_ARTIFACT_REJECTION):
        front_c_report_from_identity_artifact(payload, q_ledger_row=row)


def test_no_false_zero_drift_requires_derivation_for_live_artifacts():
    row = front_c_prior_large_q_ledger()
    payload = _valid_identity_payload(run_derived=True)
    payload["decision_path_derivation"]["independent_sparse_derivation"] = False

    with pytest.raises(ValueError, match="independent_sparse_derivation"):
        front_c_report_from_identity_artifact(
            payload,
            q_ledger_row=row,
            claimed_front_c_viable=True,
        )

    payload["decision_path_derivation"]["independent_sparse_derivation"] = True
    report = front_c_report_from_identity_artifact(
        payload,
        q_ledger_row=row,
        claimed_front_c_viable=True,
    )

    assert report.final_gate_passed is True


def test_denominator_q_ledger_and_state_metadata_are_hard_validations():
    row = front_c_prior_large_q_ledger()
    payload = _valid_identity_payload()
    unstable = deepcopy(payload)
    unstable["timeline"][1]["eligible_weight_count"] = row.eligible_weight_count + 1
    with pytest.raises(ValueError, match="stable across rows"):
        validate_front_c_identity_artifact(unstable, q_ledger_row=row)

    q_mismatch = deepcopy(payload)
    q_mismatch["q_ledger"] = {
        "logical_shapes": [[row.eligible_weight_count + 1]],
        "scale_count": 1,
    }
    with pytest.raises(ValueError, match="q ledger eligible weight count"):
        validate_front_c_identity_artifact(q_mismatch)

    state_missing = deepcopy(payload)
    state_missing["state_metadata"]["states"][0]["state_key"] = "renamed_fixture"
    state_missing["state_metadata"]["step_state_layout_sha256"] = {
        key: {"renamed_fixture": "fixture-state-layout-sha256"}
        for key in state_missing["state_metadata"]["step_state_layout_sha256"]
    }
    with pytest.raises(ValueError, match="missing from state_metadata"):
        validate_front_c_identity_artifact(state_missing, q_ledger_row=row)

    state_drift = deepcopy(payload)
    state_drift["state_metadata"]["step_state_layout_sha256"]["1"]["fixture"] = "changed"
    with pytest.raises(ValueError, match="step_state_layout_sha256 drift"):
        validate_front_c_identity_artifact(state_drift, q_ledger_row=row)


def test_mutable_value_hash_diagnostics_do_not_drive_layout_stability():
    row = front_c_prior_large_q_ledger()
    payload = _valid_identity_payload()
    payload["diagnostics"] = {
        "step_q_value_sha256": {
            "0": {"fixture": "q-value-before"},
            "1": {"fixture": "q-value-after"},
        },
        "note": "mutable q/acc value hashes are diagnostics, not layout guards",
    }

    validation = validate_front_c_identity_artifact(payload, q_ledger_row=row)

    assert validation.status == FRONT_C_IDENTITY_EXTRACTABLE

    layout_drift = deepcopy(payload)
    layout_drift["state_metadata"]["step_state_layout_sha256"]["1"]["fixture"] = "layout-changed"
    with pytest.raises(ValueError, match="layout hash"):
        validate_front_c_identity_artifact(layout_drift, q_ledger_row=row)


def test_conflict_overlap_classifier_is_diagnostic_only():
    conflict = classify_front_c_conflict_overlap(
        target_helping_q_directions=[
            {"state_key": "fixture", "flat_index": 5, "direction": 1},
            {"state_key": "fixture", "flat_index": 7, "direction": -1},
        ],
        prior_serving_q_directions=[
            {"state_key": "fixture", "flat_index": 5, "direction": -1},
            {"state_key": "fixture", "flat_index": 8, "direction": 1},
        ],
        accepted_prior_harming_keys=[{"state_key": "fixture", "flat_index": 8}],
    )
    assert conflict.diagnostic_only is True
    assert conflict.classification == "representational_conflict_or_isolation"
    assert conflict.same_identity_opposite_direction_overlap_count == 1

    selection_pressure = classify_front_c_conflict_overlap(
        target_helping_q_directions=[
            {"state_key": "fixture", "flat_index": 5, "direction": 1},
        ],
        prior_serving_q_directions=[
            {"state_key": "fixture", "flat_index": 8, "direction": 1},
        ],
        accepted_prior_harming_keys=[{"state_key": "fixture", "flat_index": 8}],
    )
    assert selection_pressure.classification == "selection_or_cap_pressure"
    assert selection_pressure.same_identity_opposite_direction_overlap_count == 0


def test_path_b_carry_forward_folds_are_documented_not_executed_in_cpu_slice():
    joined = " ".join(FRONT_C_PATH_B_CARRY_FORWARD_FOLDS)
    assert "metadata bits" in joined
    assert "independently derive sparse_decision_path" in joined
    assert "multiple ordered timeline rows" in joined
    assert "step-0 prior audit" in joined


def test_count_only_timeline_fixture_remains_rejected_through_adapter():
    row = front_c_prior_large_q_ledger()
    payload = _valid_identity_payload()
    payload["timeline"] = [front_c_count_only_timeline_artifact()]

    with pytest.raises(ValueError, match=COUNT_ONLY_ARTIFACT_REJECTION):
        validate_front_c_identity_artifact(payload, q_ledger_row=row)
