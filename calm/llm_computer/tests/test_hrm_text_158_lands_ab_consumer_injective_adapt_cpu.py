"""PLAN_v6 Phase A: consumer injective hostiles (named IDs authority)."""
from __future__ import annotations

import copy

import pytest
import torch

from calm.hrm_text_158.native_full_stack.lands_ab_eval_production_post_state import (
    INJECTIVE_POST_ACC_BINDING_RO_AVAILABLE,
    TRANSITION_PROOF_FIELDS,
    crosscheck_production_q_vs_receipt_proof,
    evaluate_family_s2,
    recompute_s1_and_compare,
)
from calm.hrm_text_158.native_full_stack.named_receipt_binding import (
    build_sparse_event_binding_by_key,
    logical_shape_by_key_from_q_levels,
)
from calm.hrm_text_158.native_full_stack.sparse_vote_events import SparseVoteEvents


def _good_proof(q: str = "a" * 64) -> dict:
    return {
        "candidate_q_sha256_after": q,
        "q_changed_identities_sha256": "b" * 64,
        "applied_row_identities_sha256": "c" * 64,
        "ordered_applied_row_identities_sha256": "c" * 64,
        "applied_directions_sha256": "d" * 64,
        "applied_thresholds_sha256": "e" * 64,
        "residual_after_threshold_sha256": "f" * 64,
        "bounded_accumulator_summary_after": {"hot_exact_row_count": 1},
        "q_changed_count": 1,
        "applied_row_count": 1,
        "event_vote_count": 3,
        "candidate_count": 4,
    }


class _Q:
    def __init__(self, t):
        self.q_levels = t


def _prior(shape=(4,)):
    return {"lin": _Q(torch.zeros(shape, dtype=torch.int8))}


def _events(pairs: dict[int, int]) -> SparseVoteEvents:
    return SparseVoteEvents.from_dict(pairs)


def test_H_FLAG_TRUE_WITHOUT_MAPS_FAIL():
    assert INJECTIVE_POST_ACC_BINDING_RO_AVAILABLE is True
    proof = {"lin": _good_proof()}
    res = crosscheck_production_q_vs_receipt_proof(
        production_post_q_sha256_by_key={"lin": "a" * 64},
        receipt_proof_by_key=proof,
        builder_receipt_pass=True,
        reapply_proof_by_key=proof,
        s1_compare=None,
        s2_compare=None,
    )
    assert res["crosscheck_ok"] is False
    assert "injective_flag_true_without_maps" in res["reason"]


def test_H_VACUOUS_FLAG_ONLY():
    """Flag True + empty S1/S2 maps → fail-closed."""
    proof = {"lin": _good_proof()}
    res = crosscheck_production_q_vs_receipt_proof(
        production_post_q_sha256_by_key={"lin": "a" * 64},
        receipt_proof_by_key=proof,
        builder_receipt_pass=True,
        reapply_proof_by_key=proof,
        s1_compare={},
        s2_compare={},
    )
    assert res["crosscheck_ok"] is False
    assert "injective_flag_true_without_maps" in res["reason"]


def test_H_BINDING_MISMATCH_FAIL():
    prior = _prior()
    events = {"lin": _events({0: 1, 2: -3})}
    shapes = logical_shape_by_key_from_q_levels({k: prior[k].q_levels for k in prior})
    true_bind = recompute_s1_and_compare(
        sparse_events_by_key=events,
        prior_states=prior,
        named_binding_by_key=None,
        named_shape_by_key=None,
    )["recomputed_binding_by_key"]
    assert true_bind
    bad = {k: ("0" * 64) for k in true_bind}
    cmp = recompute_s1_and_compare(
        sparse_events_by_key=events,
        prior_states=prior,
        named_binding_by_key=bad,
        named_shape_by_key=shapes,
    )
    assert cmp["s1_ok"] is False
    assert cmp["s1_binding_ok"] is False or cmp["keys_equal"] is True
    proof = {"lin": _good_proof()}
    res = crosscheck_production_q_vs_receipt_proof(
        production_post_q_sha256_by_key={"lin": "a" * 64},
        receipt_proof_by_key=proof,
        builder_receipt_pass=True,
        reapply_proof_by_key=proof,
        s1_compare=cmp,
        s2_compare={"s2_ok": True, "family": "B1"},
    )
    assert res["crosscheck_ok"] is False
    assert "s1_recompute_mismatch" in res["reason"]


def test_H_DECODE_MISMATCH_FAIL():
    s2 = evaluate_family_s2(
        family="B1",
        named_s2_decode_by_key={"lin": "1" * 64},
        production_logical_acc_by_key={"lin": "2" * 64},
        named_post_payload_sha256=None,
        twin_or_canonical_post_payload_sha256=None,
    )
    assert s2["s2_ok"] is False
    proof = {"lin": _good_proof()}
    # provide a passing s1 so reason is s2
    prior = _prior()
    events = {"lin": _events({0: 1})}
    shapes = logical_shape_by_key_from_q_levels({k: prior[k].q_levels for k in prior})
    true = recompute_s1_and_compare(
        sparse_events_by_key=events,
        prior_states=prior,
        named_binding_by_key=None,
        named_shape_by_key=None,
    )
    s1 = recompute_s1_and_compare(
        sparse_events_by_key=events,
        prior_states=prior,
        named_binding_by_key=true["recomputed_binding_by_key"],
        named_shape_by_key=shapes,
    )
    res = crosscheck_production_q_vs_receipt_proof(
        production_post_q_sha256_by_key={"lin": "a" * 64},
        receipt_proof_by_key=proof,
        builder_receipt_pass=True,
        reapply_proof_by_key=proof,
        s1_compare=s1,
        s2_compare=s2,
    )
    assert res["crosscheck_ok"] is False
    assert "s2_family_mismatch" in res["reason"]


def test_H_EQUAL_PASS():
    prior = _prior()
    events = {"lin": _events({0: 7, 1: -2})}
    true = recompute_s1_and_compare(
        sparse_events_by_key=events,
        prior_states=prior,
        named_binding_by_key=None,
        named_shape_by_key=None,
    )
    named = dict(true["recomputed_binding_by_key"])
    shapes = logical_shape_by_key_from_q_levels(
        {k: prior[k].q_levels for k in prior}
    )
    s1 = recompute_s1_and_compare(
        sparse_events_by_key=events,
        prior_states=prior,
        named_binding_by_key=named,
        named_shape_by_key=shapes,
    )
    assert s1["s1_ok"] is True
    assert s1["shape_ok"] is True
    acc = "c" * 64
    s2 = evaluate_family_s2(
        family="B1",
        named_s2_decode_by_key={"lin": acc},
        production_logical_acc_by_key={"lin": acc},
        named_post_payload_sha256=None,
        twin_or_canonical_post_payload_sha256=None,
    )
    assert s2["s2_ok"] is True
    proof = {"lin": _good_proof()}
    res = crosscheck_production_q_vs_receipt_proof(
        production_post_q_sha256_by_key={"lin": "a" * 64},
        receipt_proof_by_key=proof,
        builder_receipt_pass=True,
        reapply_proof_by_key=proof,
        s1_compare=s1,
        s2_compare=s2,
    )
    assert res["crosscheck_ok"] is True
    assert res["reason"] == "transition_proof_and_injective_post_acc_equal"


def test_H_B3_NAMED_MAP_REQUIRED():
    """B3 family S2 requires both payload shas (named map/payload present)."""
    s2 = evaluate_family_s2(
        family="B3",
        named_s2_decode_by_key=None,
        production_logical_acc_by_key=None,
        named_post_payload_sha256=None,
        twin_or_canonical_post_payload_sha256="a" * 64,
    )
    assert s2["s2_ok"] is False
    s2b = evaluate_family_s2(
        family="B3",
        named_s2_decode_by_key=None,
        production_logical_acc_by_key=None,
        named_post_payload_sha256="a" * 64,
        twin_or_canonical_post_payload_sha256="a" * 64,
    )
    assert s2b["s2_ok"] is True
    # S1 with empty named map fails
    prior = _prior()
    events = {"lin": _events({0: 1})}
    s1 = recompute_s1_and_compare(
        sparse_events_by_key=events,
        prior_states=prior,
        named_binding_by_key={},
        named_shape_by_key=None,
    )
    assert s1["s1_ok"] is False


def test_H_SHAPE_MISMATCH_FAIL():
    """Same events + coordinated wrong named binding + wrong same-numel shape fails.

    Non-vacuity: if named shape were recompute input, a matching-wrong-pair could
    self-pass; independent geometry forces fail.
    """
    prior = _prior(shape=(4,))
    events = {"lin": _events({0: 1, 3: 2})}
    true = recompute_s1_and_compare(
        sparse_events_by_key=events,
        prior_states=prior,
        named_binding_by_key=None,
        named_shape_by_key=None,
    )
    # wrong shape same numel (2x2 vs 4) and coordinated wrong binding
    wrong_shape = {"lin": (2, 2)}
    # craft a binding under wrong geometry
    wrong_bind = build_sparse_event_binding_by_key(
        events, logical_shape_by_key=wrong_shape
    )
    assert wrong_bind != true["recomputed_binding_by_key"]
    cmp = recompute_s1_and_compare(
        sparse_events_by_key=events,
        prior_states=prior,
        named_binding_by_key=wrong_bind,
        named_shape_by_key=wrong_shape,
    )
    assert cmp["s1_ok"] is False
    # shape equality fails even if binding coincidentally matched independent
    cmp2 = recompute_s1_and_compare(
        sparse_events_by_key=events,
        prior_states=prior,
        named_binding_by_key=true["recomputed_binding_by_key"],
        named_shape_by_key=wrong_shape,
    )
    assert cmp2["shape_ok"] is False
    assert cmp2["s1_ok"] is False
    # control: independent shapes pass
    good_shapes = logical_shape_by_key_from_q_levels(
        {k: prior[k].q_levels for k in prior}
    )
    cmp3 = recompute_s1_and_compare(
        sparse_events_by_key=events,
        prior_states=prior,
        named_binding_by_key=true["recomputed_binding_by_key"],
        named_shape_by_key=good_shapes,
    )
    assert cmp3["s1_ok"] is True


def test_flag_true_invariant():
    assert INJECTIVE_POST_ACC_BINDING_RO_AVAILABLE is True
    assert set(TRANSITION_PROOF_FIELDS)


def test_H_MISSING_SHAPE_MAP_FAIL():
    """D5: absent named-shape map must FAIL (was soft-ok)."""
    prior = _prior()
    events = {"lin": _events({0: 1, 3: 2})}
    true = recompute_s1_and_compare(
        sparse_events_by_key=events,
        prior_states=prior,
        named_binding_by_key=None,
        named_shape_by_key=None,
    )
    cmp = recompute_s1_and_compare(
        sparse_events_by_key=events,
        prior_states=prior,
        named_binding_by_key=true["recomputed_binding_by_key"],
        named_shape_by_key=None,  # ABSENT
    )
    assert cmp["s1_ok"] is False
    assert cmp["shape_ok"] is False
    assert cmp["keys_equal"] is False


def test_H_DROPPED_EVENT_KEY_FAIL():
    """D5: event key missing vs production keys must FAIL (intersection would pass)."""
    prior = {"lin": _Q(torch.zeros((4,), dtype=torch.int8)), "lin2": _Q(torch.zeros((4,), dtype=torch.int8))}
    events = {"lin": _events({0: 1})}  # missing lin2
    shapes = logical_shape_by_key_from_q_levels({k: prior[k].q_levels for k in prior})
    # craft named bind only for lin
    true_lin = recompute_s1_and_compare(
        sparse_events_by_key={"lin": events["lin"]},
        prior_states={"lin": prior["lin"]},
        named_binding_by_key=None,
        named_shape_by_key=None,
    )["recomputed_binding_by_key"]
    # named maps claim both keys but events only has lin
    named_bind = dict(true_lin)
    named_bind["lin2"] = "a" * 64
    cmp = recompute_s1_and_compare(
        sparse_events_by_key=events,
        prior_states=prior,
        named_binding_by_key=named_bind,
        named_shape_by_key=shapes,
    )
    assert cmp["s1_ok"] is False
    assert cmp["keys_equal"] is False


def test_family_b2_crosscheck_no_transition_proof():
    """D1: B2 builder_pass with S1+S2 ok does not require receipt_proof_by_key."""
    prior = _prior()
    events = {"lin": _events({0: 7})}
    shapes = logical_shape_by_key_from_q_levels({k: prior[k].q_levels for k in prior})
    true = recompute_s1_and_compare(
        sparse_events_by_key=events,
        prior_states=prior,
        named_binding_by_key=None,
        named_shape_by_key=None,
    )
    s1 = recompute_s1_and_compare(
        sparse_events_by_key=events,
        prior_states=prior,
        named_binding_by_key=true["recomputed_binding_by_key"],
        named_shape_by_key=shapes,
    )
    assert s1["s1_ok"] is True
    payload = "d" * 64
    s2 = evaluate_family_s2(
        family="B2",
        named_s2_decode_by_key=None,
        production_logical_acc_by_key=None,
        named_post_payload_sha256=payload,
        twin_or_canonical_post_payload_sha256=payload,
    )
    assert s2["s2_ok"] is True
    res = crosscheck_production_q_vs_receipt_proof(
        production_post_q_sha256_by_key={"lin": "a" * 64},
        receipt_proof_by_key={},  # empty — B1 would fail
        builder_receipt_pass=True,
        reapply_proof_by_key={},
        s1_compare=s1,
        s2_compare=s2,
        family="B2",
    )
    assert res["crosscheck_ok"] is True
    assert res.get("family") == "B2"

