"""Ownership / attack surface — pure-level fail-closed proofs (no production wiring)."""
from __future__ import annotations

import pytest

from calm.hrm_text_158.native_full_stack.forgotten_accum_ordered_apply_event import (
    ExpectedIdentity,
    OrderedApplyEventLogRefuse,
    make_success_apply_event,
    require_empty_ordered_apply_event_log,
    snapshot_ordered_apply_event_log,
    validate_ordered_apply_event_sequence,
)


def test_prefill_non_empty_refused_none_disabled_ok():
    require_empty_ordered_apply_event_log(None)
    require_empty_ordered_apply_event_log([])
    with pytest.raises(OrderedApplyEventLogRefuse, match="must be empty"):
        require_empty_ordered_apply_event_log(
            [make_success_apply_event(
                seq=0, arm_id="U", optimizer_step_id=1,
                q_changed_count=0, tensor_state_key_count=0,
            )]
        )


def test_non_list_empty_containers_refused_before_work():
    """Private-list type contract: only None or built-in list; empty tuple fails."""

    class _EmptySeq:
        def __len__(self) -> int:
            return 0

    with pytest.raises(OrderedApplyEventLogRefuse, match="built-in list"):
        require_empty_ordered_apply_event_log(())  # type: ignore[arg-type]
    with pytest.raises(OrderedApplyEventLogRefuse, match="built-in list"):
        require_empty_ordered_apply_event_log(_EmptySeq())  # type: ignore[arg-type]
    with pytest.raises(OrderedApplyEventLogRefuse, match="built-in list"):
        require_empty_ordered_apply_event_log({})  # type: ignore[arg-type]


def test_missing_plus_duplicate_equal_count_fails_sequence_exact():
    expected = ExpectedIdentity(arm_id="U", start_step=1, steps=3)
    # Expected ids 1,2,3 — observed 1,1,2: count equal, missing 3 + duplicate 1.
    events = [
        make_success_apply_event(
            seq=0, arm_id="U", optimizer_step_id=1,
            q_changed_count=0, tensor_state_key_count=1,
        ),
        make_success_apply_event(
            seq=1, arm_id="U", optimizer_step_id=1,
            q_changed_count=0, tensor_state_key_count=1,
        ),
        make_success_apply_event(
            seq=2, arm_id="U", optimizer_step_id=2,
            q_changed_count=0, tensor_state_key_count=1,
        ),
    ]
    summary = validate_ordered_apply_event_sequence(
        snapshot_ordered_apply_event_log(events), expected
    )
    assert summary["observed_count"] == summary["expected_count"] == 3
    assert summary["missing_count"] >= 1
    assert summary["duplicate_count"] >= 1
    assert summary["sequence_exact_ok"] is False
    assert summary["full_payload_sha256"]
    assert summary["claimable"] is False
    assert summary["runtime_proven"] is False


def test_post_snapshot_mutation_cannot_change_summary():
    expected = ExpectedIdentity(arm_id="E", start_step=5, steps=2)
    live = [
        make_success_apply_event(
            seq=i, arm_id="E", optimizer_step_id=5 + i,
            q_changed_count=1, tensor_state_key_count=2,
        )
        for i in range(2)
    ]
    snap = snapshot_ordered_apply_event_log(live)
    summary_before = validate_ordered_apply_event_sequence(snap, expected)
    live.clear()
    live.append(
        make_success_apply_event(
            seq=99, arm_id="X", optimizer_step_id=999,
            q_changed_count=0, tensor_state_key_count=0,
        )
    )
    summary_after = validate_ordered_apply_event_sequence(snap, expected)
    assert summary_after == summary_before
    assert summary_before["sequence_exact_ok"] is True


def test_cross_arm_expected_identity_isolates_arms():
    shared_live = [
        make_success_apply_event(
            seq=0, arm_id="U", optimizer_step_id=1,
            q_changed_count=0, tensor_state_key_count=1,
        )
    ]
    snap = snapshot_ordered_apply_event_log(shared_live)
    u = validate_ordered_apply_event_sequence(
        snap, ExpectedIdentity(arm_id="U", start_step=1, steps=1)
    )
    e = validate_ordered_apply_event_sequence(
        snap, ExpectedIdentity(arm_id="E", start_step=1, steps=1)
    )
    assert u["sequence_exact_ok"] is True
    assert e["sequence_exact_ok"] is False
    assert e["wrong_arm_count"] == 1
    assert e["claimable"] is False
    assert e["runtime_proven"] is False


def test_same_step_ids_across_arms_do_not_merge_evidence():
    u_events = [
        make_success_apply_event(
            seq=0, arm_id="U", optimizer_step_id=1,
            q_changed_count=0, tensor_state_key_count=1,
        )
    ]
    e_events = [
        make_success_apply_event(
            seq=0, arm_id="E", optimizer_step_id=1,
            q_changed_count=0, tensor_state_key_count=1,
        )
    ]
    u_sum = validate_ordered_apply_event_sequence(
        snapshot_ordered_apply_event_log(u_events),
        ExpectedIdentity(arm_id="U", start_step=1, steps=1),
    )
    e_sum = validate_ordered_apply_event_sequence(
        snapshot_ordered_apply_event_log(e_events),
        ExpectedIdentity(arm_id="E", start_step=1, steps=1),
    )
    assert u_sum["sequence_exact_ok"] and e_sum["sequence_exact_ok"]
    assert u_sum["arm_id"] != e_sum["arm_id"]
    assert u_sum["observed_identity_projection_sha256"] != e_sum[
        "observed_identity_projection_sha256"
    ]


def test_fake_stub_log_cannot_set_authority_true():
    fake = [
        make_success_apply_event(
            seq=0, arm_id="FAKE", optimizer_step_id=1,
            q_changed_count=0, tensor_state_key_count=0,
        )
    ]
    summary = validate_ordered_apply_event_sequence(
        snapshot_ordered_apply_event_log(fake),
        ExpectedIdentity(arm_id="FAKE", start_step=1, steps=1),
    )
    assert summary["sequence_exact_ok"] is True
    assert summary["claimable"] is False
    assert summary["bankable"] is False
    assert summary["runtime_proven"] is False
    assert summary["forensic_only"] is True
