"""Seam helpers: schema, snapshot immutability, expected identity, authority flags."""
from __future__ import annotations

import hashlib
import json

from calm.hrm_text_158.native_full_stack.forgotten_accum_ordered_apply_event import (
    APPLY_OUTCOME_SUCCESS,
    ATTACHMENT_KEY,
    ExpectedIdentity,
    FULL_PAYLOAD_FIELDS,
    IDENTITY_PROJECTION_FIELDS,
    PRODUCER_LITERAL,
    TELEMETRY_EXCLUDED_FROM_EQUALITY,
    VALIDATION_SCHEMA_ID,
    build_expected_identity_projection,
    full_payload_sha256,
    identity_projection_sha256,
    make_success_apply_event,
    snapshot_ordered_apply_event_log,
    validate_ordered_apply_event_sequence,
)


def test_frozen_schema_constants():
    assert VALIDATION_SCHEMA_ID == "forgotten_accum_ordered_apply_event_validation_v1"
    assert ATTACHMENT_KEY == "ordered_apply_event_validation_summary"
    assert PRODUCER_LITERAL == "run_bounded_delta_steps_post_states_rebind"
    assert APPLY_OUTCOME_SUCCESS == "SUCCESS"
    assert "q_changed_count" in TELEMETRY_EXCLUDED_FROM_EQUALITY
    assert "tensor_state_key_count" in TELEMETRY_EXCLUDED_FROM_EQUALITY
    for field in TELEMETRY_EXCLUDED_FROM_EQUALITY:
        assert field not in IDENTITY_PROJECTION_FIELDS
    assert set(IDENTITY_PROJECTION_FIELDS).issubset(set(FULL_PAYLOAD_FIELDS))


def test_expected_identity_independent_of_observed_events():
    expected = ExpectedIdentity(arm_id="U", start_step=10, steps=3)
    proj = build_expected_identity_projection(expected)
    assert proj == [
        {
            "seq": 0,
            "arm_id": "U",
            "optimizer_step_id": 10,
            "apply_outcome": "SUCCESS",
            "producer": PRODUCER_LITERAL,
        },
        {
            "seq": 1,
            "arm_id": "U",
            "optimizer_step_id": 11,
            "apply_outcome": "SUCCESS",
            "producer": PRODUCER_LITERAL,
        },
        {
            "seq": 2,
            "arm_id": "U",
            "optimizer_step_id": 12,
            "apply_outcome": "SUCCESS",
            "producer": PRODUCER_LITERAL,
        },
    ]
    # Contaminating "observed" must not feed expected builder.
    bogus = [make_success_apply_event(
        seq=99, arm_id="X", optimizer_step_id=1,
        q_changed_count=0, tensor_state_key_count=0,
    )]
    assert build_expected_identity_projection(expected) == proj
    assert bogus[0]["optimizer_step_id"] not in {
        row["optimizer_step_id"] for row in proj
    }


def test_snapshot_immutable_against_live_mutation():
    live = [
        make_success_apply_event(
            seq=0, arm_id="E", optimizer_step_id=1,
            q_changed_count=3, tensor_state_key_count=7,
        )
    ]
    snap = snapshot_ordered_apply_event_log(live)
    live[0]["q_changed_count"] = 999
    live[0]["optimizer_step_id"] = 42
    live.append(
        make_success_apply_event(
            seq=1, arm_id="E", optimizer_step_id=2,
            q_changed_count=1, tensor_state_key_count=1,
        )
    )
    assert len(snap) == 1
    assert snap[0]["q_changed_count"] == 3
    assert snap[0]["optimizer_step_id"] == 1
    try:
        snap[0]["q_changed_count"] = 0  # type: ignore[index]
        raised = False
    except TypeError:
        raised = True
    assert raised


def test_identity_hash_excludes_telemetry_full_payload_does_not():
    a = make_success_apply_event(
        seq=0, arm_id="U", optimizer_step_id=1,
        q_changed_count=1, tensor_state_key_count=1,
    )
    b = make_success_apply_event(
        seq=0, arm_id="U", optimizer_step_id=1,
        q_changed_count=99, tensor_state_key_count=88,
    )
    snap_a = snapshot_ordered_apply_event_log([a])
    snap_b = snapshot_ordered_apply_event_log([b])
    id_a = identity_projection_sha256(
        [{k: snap_a[0][k] for k in IDENTITY_PROJECTION_FIELDS}]
    )
    id_b = identity_projection_sha256(
        [{k: snap_b[0][k] for k in IDENTITY_PROJECTION_FIELDS}]
    )
    assert id_a == id_b
    assert full_payload_sha256(snap_a) != full_payload_sha256(snap_b)


def test_canonical_hash_matches_frozen_serialization_rule():
    proj = build_expected_identity_projection(
        ExpectedIdentity(arm_id="U", start_step=1, steps=2)
    )
    blob = (
        json.dumps(proj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    )
    assert identity_projection_sha256(proj) == hashlib.sha256(
        blob.encode("utf-8")
    ).hexdigest()


def test_exact_match_summary_authority_flags_always_false():
    expected = ExpectedIdentity(arm_id="U", start_step=1, steps=2)
    events = [
        make_success_apply_event(
            seq=i, arm_id="U", optimizer_step_id=1 + i,
            q_changed_count=i, tensor_state_key_count=i + 1,
        )
        for i in range(2)
    ]
    summary = validate_ordered_apply_event_sequence(
        snapshot_ordered_apply_event_log(events), expected
    )
    assert summary["schema_id"] == VALIDATION_SCHEMA_ID
    assert summary["sequence_exact_ok"] is True
    assert summary["claimable"] is False
    assert summary["bankable"] is False
    assert summary["forensic_only"] is True
    assert summary["runtime_proven"] is False
    assert summary["expected_identity_projection_sha256"] == summary[
        "observed_identity_projection_sha256"
    ]
    assert isinstance(summary["full_payload_sha256"], str)
    assert len(summary["full_payload_sha256"]) == 64
