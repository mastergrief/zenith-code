"""Sequence validator: reorder/wrong-arm/extra/missing+dup + forensic hash role."""
from __future__ import annotations

from calm.hrm_text_158.native_full_stack.forgotten_accum_ordered_apply_event import (
    ExpectedIdentity,
    IDENTITY_PROJECTION_FIELDS,
    full_payload_sha256,
    identity_projection_sha256,
    make_success_apply_event,
    snapshot_ordered_apply_event_log,
    validate_ordered_apply_event_sequence,
)


def _exact_events(arm: str, start: int, steps: int, *, q: int = 0):
    return [
        make_success_apply_event(
            seq=i,
            arm_id=arm,
            optimizer_step_id=start + i,
            q_changed_count=q + i,
            tensor_state_key_count=1,
        )
        for i in range(steps)
    ]


def test_reorder_same_multiset_fails():
    expected = ExpectedIdentity(arm_id="U", start_step=1, steps=3)
    events = _exact_events("U", 1, 3)
    events[0], events[2] = events[2], events[0]
    # Fix seq labels to look locally consistent while step order is wrong.
    for i, event in enumerate(events):
        event["seq"] = i
    summary = validate_ordered_apply_event_sequence(
        snapshot_ordered_apply_event_log(events), expected
    )
    assert summary["observed_count"] == summary["expected_count"]
    assert summary["reorder_detected"] is True
    assert summary["sequence_exact_ok"] is False
    assert (
        summary["expected_identity_projection_sha256"]
        != summary["observed_identity_projection_sha256"]
    )


def test_wrong_arm_and_extra_fail():
    expected = ExpectedIdentity(arm_id="U", start_step=1, steps=2)
    events = [
        make_success_apply_event(
            seq=0, arm_id="E", optimizer_step_id=1,
            q_changed_count=0, tensor_state_key_count=1,
        ),
        make_success_apply_event(
            seq=1, arm_id="U", optimizer_step_id=2,
            q_changed_count=0, tensor_state_key_count=1,
        ),
        make_success_apply_event(
            seq=2, arm_id="U", optimizer_step_id=99,
            q_changed_count=0, tensor_state_key_count=1,
        ),
    ]
    summary = validate_ordered_apply_event_sequence(
        snapshot_ordered_apply_event_log(events), expected
    )
    assert summary["wrong_arm_count"] >= 1
    assert summary["extra_count"] >= 1
    assert summary["sequence_exact_ok"] is False


def test_full_payload_hash_forensic_not_equality_gate():
    expected = ExpectedIdentity(arm_id="U", start_step=1, steps=1)
    e1 = make_success_apply_event(
        seq=0, arm_id="U", optimizer_step_id=1,
        q_changed_count=1, tensor_state_key_count=1,
    )
    e2 = make_success_apply_event(
        seq=0, arm_id="U", optimizer_step_id=1,
        q_changed_count=50, tensor_state_key_count=50,
    )
    s1 = validate_ordered_apply_event_sequence(
        snapshot_ordered_apply_event_log([e1]), expected
    )
    s2 = validate_ordered_apply_event_sequence(
        snapshot_ordered_apply_event_log([e2]), expected
    )
    # Identity gate passes for both; forensic hashes differ with telemetry.
    assert s1["sequence_exact_ok"] is True
    assert s2["sequence_exact_ok"] is True
    assert s1["expected_identity_projection_sha256"] == s2[
        "expected_identity_projection_sha256"
    ]
    assert s1["observed_identity_projection_sha256"] == s2[
        "observed_identity_projection_sha256"
    ]
    assert s1["full_payload_sha256"] != s2["full_payload_sha256"]
    # Explicit: equality uses identity hash, not full payload.
    id_hash = identity_projection_sha256(
        [{k: e1[k] for k in IDENTITY_PROJECTION_FIELDS}]
    )
    assert s1["observed_identity_projection_sha256"] == id_hash
    assert s1["full_payload_sha256"] == full_payload_sha256(
        snapshot_ordered_apply_event_log([e1])
    )


def test_empty_steps_exact():
    summary = validate_ordered_apply_event_sequence(
        snapshot_ordered_apply_event_log([]),
        ExpectedIdentity(arm_id="U", start_step=1, steps=0),
    )
    assert summary["sequence_exact_ok"] is True
    assert summary["expected_count"] == 0
    assert summary["observed_count"] == 0
    assert summary["claimable"] is False
    assert summary["runtime_proven"] is False


def test_summary_key_set_frozen():
    summary = validate_ordered_apply_event_sequence(
        snapshot_ordered_apply_event_log(_exact_events("U", 1, 1)),
        ExpectedIdentity(arm_id="U", start_step=1, steps=1),
    )
    required = {
        "schema_id",
        "arm_id",
        "start_step",
        "steps",
        "expected_count",
        "observed_count",
        "sequence_exact_ok",
        "missing_count",
        "duplicate_count",
        "extra_count",
        "wrong_arm_count",
        "reorder_detected",
        "expected_identity_projection_sha256",
        "observed_identity_projection_sha256",
        "full_payload_sha256",
        "claimable",
        "bankable",
        "forensic_only",
        "runtime_proven",
    }
    assert set(summary) == required
